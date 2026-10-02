// P13-4 — multivariate Generator distributions, permuted, and choice(p=) (D-162).
// Adds multinomial, dirichlet, multivariateHypergeometric, permuted, and choice(p=)
// to np.random.Generator via declaration merging.

import { array } from "./creation.js";
import { NDArray } from "./ndarray.js";
import { take } from "./indexing.js";
import type { NativeNDArray } from "./addon.js";
import { Generator } from "./random.js";
import type { Size } from "./random.js";
import { ValueError, wrapNative } from "./errors.js";

// ---- helpers ----

function toShape(size: Size | null | undefined): number[] | null {
  if (size === null || size === undefined) return null;
  const s = typeof size === "number" ? [size] : [...size];
  for (const d of s) {
    if (!Number.isInteger(d)) throw new TypeError("size must contain integers");
  }
  return s;
}

export function wrapArr(raw: NativeNDArray): NDArray {
  return NDArray._wrap(raw);
}

type Population = number | bigint | NDArray | number[][];

function resolvePopulation(a: Population, ax: number): { pop: number; src: NDArray | null } {
  if (typeof a === "number" || typeof a === "bigint") {
    return { pop: Number(a), src: null };
  }
  const arr = a instanceof NDArray ? a : array(a as number[] | number[][]);
  const axis = ax < 0 ? arr.ndim + ax : ax;
  return { pop: arr.shape[axis] ?? 0, src: arr };
}

// ---- declaration merging ----

declare module "./random.js" {
  interface Generator {
    /**
     * `multinomial(n, pvals, size?)`
     * Draw samples from a multinomial distribution.
     * Returns an int64 NDArray of shape `(*size, d)` where `d = pvals.length`.
     */
    multinomial(n: number, pvals: readonly number[], size?: Size | null): NDArray;

    /**
     * `dirichlet(alpha, size?)`
     * Draw samples from a Dirichlet distribution.
     * Returns float64 NDArray of shape `(*size, d)` where `d = alpha.length`.
     */
    dirichlet(alpha: readonly number[], size?: Size | null): NDArray;

    /**
     * `multivariateHypergeometric(colors, nsample, size?, method?)`
     * Draw samples from a multivariate hypergeometric distribution.
     * Returns int64 NDArray of shape `(*size, len(colors))`.
     */
    multivariateHypergeometric(
      colors: readonly number[],
      nsample: number,
      size?: Size | null,
      method?: "count" | "marginals",
    ): NDArray;

    /**
     * `permuted(x, axis?)`
     * Return a copy of `x` with lanes along `axis` independently shuffled.
     */
    permuted(x: NDArray, axis?: number): NDArray;
  }
}

// ---- parameter validation ----

function checkPvals(pvals: readonly number[]): void {
  if (pvals.length === 0) throw new ValueError("pvals must be non-empty");
  let sum = 0;
  for (const p of pvals) {
    if (!Number.isFinite(p) || p < 0) throw new ValueError("pvals must be non-negative");
    sum += p;
  }
  if (sum > 1.0 + 1e-8) throw new ValueError("sum(pvals[:-1]) > 1.0");
}

function checkColors(colors: readonly number[]): void {
  if (colors.length === 0) throw new ValueError("colors must be non-empty");
  for (const c of colors) {
    if (!Number.isInteger(c) || c < 0) throw new ValueError("colors must be non-negative integers");
  }
}

// ---- prototype methods ----

Generator.prototype.multinomial = function (
  this: Generator,
  n: number,
  pvals: readonly number[],
  size?: Size | null,
): NDArray {
  if (!Number.isInteger(n) || n < 0) throw new ValueError("n must be a non-negative integer");
  checkPvals(pvals);
  const shape = toShape(size) ?? [];
  return wrapNative(() => wrapArr(this._bg.multinomial(n, Array.from(pvals), shape)));
};

Generator.prototype.dirichlet = function (
  this: Generator,
  alpha: readonly number[],
  size?: Size | null,
): NDArray {
  if (alpha.length === 0) throw new ValueError("alpha must be non-empty");
  for (const a of alpha) {
    if (!Number.isFinite(a) || a <= 0) throw new ValueError("alpha values must be > 0");
  }
  const shape = toShape(size) ?? [];
  return wrapNative(() => wrapArr(this._bg.dirichlet(Array.from(alpha), shape)));
};

Generator.prototype.multivariateHypergeometric = function (
  this: Generator,
  colors: readonly number[],
  nsample: number,
  size?: Size | null,
  method: "count" | "marginals" = "marginals",
): NDArray {
  checkColors(colors);
  const total = colors.reduce((a, b) => a + b, 0);
  if (!Number.isInteger(nsample) || nsample < 0) {
    throw new ValueError("nsample must be a non-negative integer");
  }
  if (nsample > total) throw new ValueError("nsample > sum(colors)");
  const shape = toShape(size) ?? [];
  const colorsArr = Array.from(colors);
  return wrapNative(() => {
    if (method === "count") {
      return wrapArr(this._bg.mvhgCount(colorsArr, nsample, shape));
    }
    return wrapArr(this._bg.mvhgMarginals(colorsArr, nsample, shape));
  });
};

Generator.prototype.permuted = function (
  this: Generator,
  x: NDArray,
  axis = 0,
): NDArray {
  if (!(x instanceof NDArray)) throw new TypeError("permuted requires an NDArray");
  if (x.ndim === 0) throw new ValueError("x must be at least 1-dimensional");
  const ax = axis < 0 ? x.ndim + axis : axis;
  if (ax < 0 || ax >= x.ndim) throw new ValueError("axis out of bounds");
  return wrapNative(() => wrapArr(this._bg.permuted(x._native, ax)));
};

// ---- choice(p=) augmentation ----
// Override Generator.prototype.choice to support weighted sampling (p= argument).

const _baseChoice = Generator.prototype.choice;
Generator.prototype.choice = function (
  this: Generator,
  a: Parameters<Generator["choice"]>[0],
  size?: Parameters<Generator["choice"]>[1],
  replace = true,
  p: unknown = null,
  axis = 0,
  shuffle = true,
): ReturnType<Generator["choice"]> {
  const isOpts =
    typeof size === "object" && size !== null && !Array.isArray(size) && !(size instanceof NDArray);
  const opts = isOpts
    ? (size as { size?: Size; replace?: boolean; p?: unknown; axis?: number; shuffle?: boolean })
    : { size: size as Size | undefined, replace, p, axis, shuffle };

  // No p= weights: delegate to base implementation.
  if (opts.p === null || opts.p === undefined) {
    return _baseChoice.call(this, a, size, replace, p, axis, shuffle);
  }

  // Weighted sampling.
  const pArr = opts.p instanceof NDArray ? opts.p : array(opts.p as number[]);
  if (pArr.ndim !== 1) throw new ValueError("p must be 1-dimensional");

  const ax = opts.axis ?? 0;
  const { pop, src } = resolvePopulation(a as Population, ax);

  if (pArr.shape[0] !== pop) throw new ValueError("a and p must have the same length");

  const pData = pArr.astype("float64").toTypedArray() as Float64Array;
  const cdf: number[] = new Array<number>(pop);
  let cum = 0;
  for (let i = 0; i < pop; i++) {
    if ((pData[i] as number) < 0) throw new ValueError("p values must be non-negative");
    cum += pData[i] as number;
    cdf[i] = cum;
  }
  if (Math.abs(cum - 1.0) > 1e-6) throw new ValueError("probabilities do not sum to 1");
  cdf[pop - 1] = 1.0;

  const shape = toShape(opts.size ?? undefined);
  const withReplacement = opts.replace ?? true;

  return wrapNative(() => {
    const idxArr = wrapArr(this._bg.choiceP(pop, shape ?? [], cdf));
    if (!withReplacement) {
      const flat = Array.from(idxArr.toTypedArray() as BigInt64Array).map(Number);
      const seen = new Set<number>();
      for (const i of flat) {
        if (seen.has(i)) {
          throw new ValueError(
            "Cannot take a larger sample than population when replace=False with p=",
          );
        }
        seen.add(i);
      }
    }
    if (src === null) {
      return shape === null ? (idxArr.item() as number) : idxArr;
    }
    const result = take(src, idxArr, ax);
    return shape === null ? result.item() : result;
  });
};
