// P13-3 — discrete Generator distributions (D-161).
// Adds binomial, negativeBinomial, poisson, zipf, geometric, hypergeometric,
// and logseries to np.random.Generator via declaration merging.

import { NDArray } from "./ndarray.js";
import type { NativeNDArray } from "./addon.js";
import { Generator } from "./random.js";
import type { Size } from "./random.js";
import { ValueError, wrapNative } from "./errors.js";

// ---- helpers (module-private) ----

function toShape(size: Size | null | undefined): number[] | null {
  if (size === null || size === undefined) return null;
  const s = typeof size === "number" ? [size] : [...size];
  for (const d of s) {
    if (!Number.isInteger(d)) throw new TypeError("size must contain integers");
  }
  return s;
}

function finish(raw: NativeNDArray, shape: number[] | null): NDArray | number {
  const arr = NDArray._wrap(raw);
  return shape === null ? (arr.item() as number) : arr;
}

// ---- declaration merging ----

declare module "./random.js" {
  interface Generator {
    /**
     * `binomial(n, p, size?)`
     * Draw samples from a binomial distribution.
     * Equivalent to `numpy.random.Generator.binomial`.
     */
    binomial(n: number, p: number, size?: Size | null): NDArray | number;
    /**
     * `negativeBinomial(n, p, size?)`
     * Draw samples from a negative binomial distribution.
     * Equivalent to `numpy.random.Generator.negative_binomial`.
     */
    negativeBinomial(n: number, p: number, size?: Size | null): NDArray | number;
    /**
     * `poisson(lam?, size?)`
     * Draw samples from a Poisson distribution.
     * Equivalent to `numpy.random.Generator.poisson`.
     */
    poisson(lam?: number, size?: Size | null): NDArray | number;
    /**
     * `zipf(a, size?)`
     * Draw samples from a Zipf distribution.
     * Equivalent to `numpy.random.Generator.zipf`.
     */
    zipf(a: number, size?: Size | null): NDArray | number;
    /**
     * `geometric(p, size?)`
     * Draw samples from a geometric distribution.
     * Equivalent to `numpy.random.Generator.geometric`.
     */
    geometric(p: number, size?: Size | null): NDArray | number;
    /**
     * `hypergeometric(ngood, nbad, nsample, size?)`
     * Draw samples from a hypergeometric distribution.
     * Equivalent to `numpy.random.Generator.hypergeometric`.
     */
    hypergeometric(ngood: number, nbad: number, nsample: number, size?: Size | null): NDArray | number;
    /**
     * `logseries(p, size?)`
     * Draw samples from a logarithmic series distribution.
     * Equivalent to `numpy.random.Generator.logseries`.
     */
    logseries(p: number, size?: Size | null): NDArray | number;
  }
}

// ---- parameter validation (mirrors NumPy errors) ----

function checkBinomial(n: number, p: number): void {
  if (!Number.isFinite(n) || n < 0 || Math.trunc(n) !== n) {
    throw new ValueError("n < 0");
  }
  if (!Number.isFinite(p) || p < 0 || p > 1) {
    throw new ValueError("p < 0, p > 1 or p is NaN");
  }
}

function checkNegativeBinomial(n: number, p: number): void {
  if (!Number.isFinite(n) || n <= 0) {
    throw new ValueError("n <= 0");
  }
  if (!Number.isFinite(p) || p <= 0 || p > 1) {
    throw new ValueError("p <= 0, p > 1 or p contains NaNs");
  }
}

function checkPoisson(lam: number): void {
  if (!Number.isFinite(lam) || lam < 0) {
    throw new ValueError("lam < 0 or lam is NaN");
  }
}

function checkZipf(a: number): void {
  if (!Number.isFinite(a) || a <= 1) {
    throw new ValueError("a <= 1 or a is NaN");
  }
}

function checkGeometric(p: number): void {
  if (!Number.isFinite(p) || p <= 0 || p > 1) {
    throw new ValueError("p <= 0, p > 1 or p contains NaNs");
  }
}

function checkHypergeometric(ngood: number, nbad: number, nsample: number): void {
  if (!Number.isFinite(ngood) || ngood < 0) {
    throw new ValueError("ngood < 0");
  }
  if (!Number.isFinite(nbad) || nbad < 0) {
    throw new ValueError("nbad < 0");
  }
  if (!Number.isFinite(nsample) || nsample < 0) {
    throw new ValueError("nsample < 0");
  }
  if (nsample > ngood + nbad) {
    throw new ValueError("ngood + nbad < nsample");
  }
}

function checkLogseries(p: number): void {
  if (!Number.isFinite(p) || p < 0 || p >= 1) {
    throw new ValueError("p < 0, p >= 1 or p is NaN");
  }
}

// ---- prototype methods ----

Generator.prototype.binomial = function (
  this: Generator,
  n: number,
  p: number,
  size?: Size | null,
): NDArray | number {
  checkBinomial(n, p);
  const shape = toShape(size);
  return wrapNative(() => {
    const raw = this._bg.binomial(Math.trunc(n), p, shape ?? []);
    return finish(raw, shape);
  });
};

Generator.prototype.negativeBinomial = function (
  this: Generator,
  n: number,
  p: number,
  size?: Size | null,
): NDArray | number {
  checkNegativeBinomial(n, p);
  const shape = toShape(size);
  return wrapNative(() => {
    const raw = this._bg.negativeBinomial(n, p, shape ?? []);
    return finish(raw, shape);
  });
};

Generator.prototype.poisson = function (
  this: Generator,
  lam = 1.0,
  size?: Size | null,
): NDArray | number {
  checkPoisson(lam);
  const shape = toShape(size);
  return wrapNative(() => {
    const raw = this._bg.poisson(lam, shape ?? []);
    return finish(raw, shape);
  });
};

Generator.prototype.zipf = function (
  this: Generator,
  a: number,
  size?: Size | null,
): NDArray | number {
  checkZipf(a);
  const shape = toShape(size);
  return wrapNative(() => {
    const raw = this._bg.zipf(a, shape ?? []);
    return finish(raw, shape);
  });
};

Generator.prototype.geometric = function (
  this: Generator,
  p: number,
  size?: Size | null,
): NDArray | number {
  checkGeometric(p);
  const shape = toShape(size);
  return wrapNative(() => {
    const raw = this._bg.geometric(p, shape ?? []);
    return finish(raw, shape);
  });
};

Generator.prototype.hypergeometric = function (
  this: Generator,
  ngood: number,
  nbad: number,
  nsample: number,
  size?: Size | null,
): NDArray | number {
  checkHypergeometric(ngood, nbad, nsample);
  const shape = toShape(size);
  return wrapNative(() => {
    const raw = this._bg.hypergeometric(
      Math.trunc(ngood),
      Math.trunc(nbad),
      Math.trunc(nsample),
      shape ?? [],
    );
    return finish(raw, shape);
  });
};

Generator.prototype.logseries = function (
  this: Generator,
  p: number,
  size?: Size | null,
): NDArray | number {
  checkLogseries(p);
  const shape = toShape(size);
  return wrapNative(() => {
    const raw = this._bg.logseries(p, shape ?? []);
    return finish(raw, shape);
  });
};
