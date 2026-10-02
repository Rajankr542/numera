// P13-5 — RandomState legacy distributions (D-163).
// Adds all legacy distribution methods to np.random.RandomState via declaration merging.
// Uses the legacy* native kernels which maintain the RandomState stream exactly.

import { NDArray } from "./ndarray.js";
import type { NativeNDArray } from "./addon.js";
import { RandomState } from "./random.js";
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

function finish(raw: NativeNDArray, shape: number[] | null): NDArray | number {
  const arr = NDArray._wrap(raw);
  return shape === null ? (arr.item() as number) : arr;
}

// ---- declaration merging ----

declare module "./random.js" {
  interface RandomState {
    // Continuous
    standardExponential(size?: Size | null): NDArray | number;
    exponential(scale?: number, size?: Size | null): NDArray | number;
    standardGamma(shape: number, size?: Size | null): NDArray | number;
    gamma(shape: number, scale?: number, size?: Size | null): NDArray | number;
    beta(a: number, b: number, size?: Size | null): NDArray | number;
    chisquare(df: number, size?: Size | null): NDArray | number;
    f(dfnum: number, dfden: number, size?: Size | null): NDArray | number;
    standardCauchy(size?: Size | null): NDArray | number;
    pareto(a: number, size?: Size | null): NDArray | number;
    weibull(a: number, size?: Size | null): NDArray | number;
    power(a: number, size?: Size | null): NDArray | number;
    laplace(loc?: number, scale?: number, size?: Size | null): NDArray | number;
    gumbel(loc?: number, scale?: number, size?: Size | null): NDArray | number;
    logistic(loc?: number, scale?: number, size?: Size | null): NDArray | number;
    lognormal(mean?: number, sigma?: number, size?: Size | null): NDArray | number;
    rayleigh(scale?: number, size?: Size | null): NDArray | number;
    standardT(df: number, size?: Size | null): NDArray | number;
    noncentralChisquare(df: number, nonc: number, size?: Size | null): NDArray | number;
    noncentralF(dfnum: number, dfden: number, nonc: number, size?: Size | null): NDArray | number;
    wald(mean: number, scale: number, size?: Size | null): NDArray | number;
    vonmises(mu: number, kappa: number, size?: Size | null): NDArray | number;
    // Discrete
    binomial(n: number, p: number, size?: Size | null): NDArray | number;
    negativeBinomial(n: number, p: number, size?: Size | null): NDArray | number;
    poisson(lam?: number, size?: Size | null): NDArray | number;
    zipf(a: number, size?: Size | null): NDArray | number;
    geometric(p: number, size?: Size | null): NDArray | number;
    hypergeometric(ngood: number, nbad: number, nsample: number, size?: Size | null): NDArray | number;
    logseries(p: number, size?: Size | null): NDArray | number;
    // State + bytes
    getState(): bigint[];
    setState(words: bigint[]): void;
    bytes(length: number): Uint8Array;
  }
}

// ---- parameter validation (mirrors NumPy errors) ----

function checkScale(v: number, name: string): void {
  if (!Number.isFinite(v) || v <= 0) throw new ValueError(`${name} <= 0`);
}
function checkScaleNonNeg(v: number, name: string): void {
  if (!Number.isFinite(v) || v < 0) throw new ValueError(`${name} < 0`);
}
function checkDf(df: number): void {
  if (!Number.isFinite(df) || df <= 0) throw new ValueError("df <= 0");
}
function checkNonc(nonc: number): void {
  if (!Number.isFinite(nonc) || nonc < 0) throw new ValueError("nonc < 0");
}
function checkAlpha(a: number, name = "a"): void {
  if (!Number.isFinite(a) || a <= 0) throw new ValueError(`${name} <= 0`);
}

// ---- prototype methods ----

RandomState.prototype.standardExponential = function (
  this: RandomState,
  size?: Size | null,
): NDArray | number {
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyStandardExponential(shape ?? []), shape));
};

RandomState.prototype.exponential = function (
  this: RandomState,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScale(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyExponential(scale, shape ?? []), shape));
};

RandomState.prototype.standardGamma = function (
  this: RandomState,
  shape: number,
  size?: Size | null,
): NDArray | number {
  checkAlpha(shape, "shape");
  const sz = toShape(size);
  return wrapNative(() => finish(this._bg.legacyStandardGamma(shape, sz ?? []), sz));
};

RandomState.prototype.gamma = function (
  this: RandomState,
  shape: number,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkAlpha(shape, "shape");
  checkScale(scale, "scale");
  const sz = toShape(size);
  return wrapNative(() => finish(this._bg.legacyGamma(shape, scale, sz ?? []), sz));
};

RandomState.prototype.beta = function (
  this: RandomState,
  a: number,
  b: number,
  size?: Size | null,
): NDArray | number {
  checkAlpha(a, "a");
  checkAlpha(b, "b");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyBeta(a, b, shape ?? []), shape));
};

RandomState.prototype.chisquare = function (
  this: RandomState,
  df: number,
  size?: Size | null,
): NDArray | number {
  checkDf(df);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyChisquare(df, shape ?? []), shape));
};

RandomState.prototype.f = function (
  this: RandomState,
  dfnum: number,
  dfden: number,
  size?: Size | null,
): NDArray | number {
  checkDf(dfnum);
  checkDf(dfden);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyF(dfnum, dfden, shape ?? []), shape));
};

RandomState.prototype.standardCauchy = function (
  this: RandomState,
  size?: Size | null,
): NDArray | number {
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyStandardCauchy(shape ?? []), shape));
};

RandomState.prototype.pareto = function (
  this: RandomState,
  a: number,
  size?: Size | null,
): NDArray | number {
  checkAlpha(a);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyPareto(a, shape ?? []), shape));
};

RandomState.prototype.weibull = function (
  this: RandomState,
  a: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(a) || a < 0) throw new ValueError("a < 0");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyWeibull(a, shape ?? []), shape));
};

RandomState.prototype.power = function (
  this: RandomState,
  a: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(a) || a <= 0 || a > 1) throw new ValueError("a must be in (0, 1]");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyPower(a, shape ?? []), shape));
};

RandomState.prototype.laplace = function (
  this: RandomState,
  loc = 0.0,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.laplace(loc, scale, shape ?? []), shape));
};

RandomState.prototype.gumbel = function (
  this: RandomState,
  loc = 0.0,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.gumbel(loc, scale, shape ?? []), shape));
};

RandomState.prototype.logistic = function (
  this: RandomState,
  loc = 0.0,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.logistic(loc, scale, shape ?? []), shape));
};

RandomState.prototype.lognormal = function (
  this: RandomState,
  mean = 0.0,
  sigma = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(sigma, "sigma");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyLognormal(mean, sigma, shape ?? []), shape));
};

RandomState.prototype.rayleigh = function (
  this: RandomState,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyRayleigh(scale, shape ?? []), shape));
};

RandomState.prototype.standardT = function (
  this: RandomState,
  df: number,
  size?: Size | null,
): NDArray | number {
  checkDf(df);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyStandardT(df, shape ?? []), shape));
};

RandomState.prototype.noncentralChisquare = function (
  this: RandomState,
  df: number,
  nonc: number,
  size?: Size | null,
): NDArray | number {
  checkDf(df);
  checkNonc(nonc);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyNoncentralChisquare(df, nonc, shape ?? []), shape));
};

RandomState.prototype.noncentralF = function (
  this: RandomState,
  dfnum: number,
  dfden: number,
  nonc: number,
  size?: Size | null,
): NDArray | number {
  checkDf(dfnum);
  checkDf(dfden);
  checkNonc(nonc);
  const shape = toShape(size);
  return wrapNative(() =>
    finish(this._bg.legacyNoncentralF(dfnum, dfden, nonc, shape ?? []), shape),
  );
};

RandomState.prototype.wald = function (
  this: RandomState,
  mean: number,
  scale: number,
  size?: Size | null,
): NDArray | number {
  checkScale(mean, "mean");
  checkScale(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyWald(mean, scale, shape ?? []), shape));
};

RandomState.prototype.vonmises = function (
  this: RandomState,
  mu: number,
  kappa: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(kappa) || kappa < 0) throw new ValueError("kappa < 0");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyVonmises(mu, kappa, shape ?? []), shape));
};

RandomState.prototype.binomial = function (
  this: RandomState,
  n: number,
  p: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(n) || n < 0 || Math.trunc(n) !== n) throw new ValueError("n < 0");
  if (!Number.isFinite(p) || p < 0 || p > 1) throw new ValueError("p < 0, p > 1 or p is NaN");
  const shape = toShape(size);
  return wrapNative(() =>
    finish(this._bg.legacyBinomial(Math.trunc(n), p, shape ?? []), shape),
  );
};

RandomState.prototype.negativeBinomial = function (
  this: RandomState,
  n: number,
  p: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(n) || n <= 0) throw new ValueError("n <= 0");
  if (!Number.isFinite(p) || p <= 0 || p > 1) throw new ValueError("p <= 0 or p > 1");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyNegativeBinomial(n, p, shape ?? []), shape));
};

RandomState.prototype.poisson = function (
  this: RandomState,
  lam = 1.0,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(lam) || lam < 0) throw new ValueError("lam < 0 or lam is NaN");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.poisson(lam, shape ?? []), shape));
};

RandomState.prototype.zipf = function (
  this: RandomState,
  a: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(a) || a <= 1) throw new ValueError("a <= 1 or a is NaN");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyZipf(a, shape ?? []), shape));
};

RandomState.prototype.geometric = function (
  this: RandomState,
  p: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(p) || p <= 0 || p > 1) throw new ValueError("p <= 0, p > 1 or p contains NaNs");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyGeometric(p, shape ?? []), shape));
};

RandomState.prototype.hypergeometric = function (
  this: RandomState,
  ngood: number,
  nbad: number,
  nsample: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(ngood) || ngood < 0) throw new ValueError("ngood < 0");
  if (!Number.isFinite(nbad) || nbad < 0) throw new ValueError("nbad < 0");
  if (!Number.isFinite(nsample) || nsample < 0) throw new ValueError("nsample < 0");
  if (nsample > ngood + nbad) throw new ValueError("ngood + nbad < nsample");
  const shape = toShape(size);
  return wrapNative(() =>
    finish(
      this._bg.legacyHypergeometric(
        Math.trunc(ngood),
        Math.trunc(nbad),
        Math.trunc(nsample),
        shape ?? [],
      ),
      shape,
    ),
  );
};

RandomState.prototype.logseries = function (
  this: RandomState,
  p: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(p) || p < 0 || p >= 1) throw new ValueError("p < 0, p >= 1 or p is NaN");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.legacyLogseries(p, shape ?? []), shape));
};

RandomState.prototype.getState = function (this: RandomState): bigint[] {
  return wrapNative(() => this._bg.getState());
};

RandomState.prototype.setState = function (this: RandomState, words: bigint[]): void {
  wrapNative(() => this._bg.setState(words));
};

RandomState.prototype.bytes = function (this: RandomState, length: number): Uint8Array {
  return wrapNative(() => this._bg.bytes(length));
};

// ---- triangular ----
declare module "./random.js" {
  interface RandomState {
    triangular(left: number, mode: number, right: number, size?: Size | null): NDArray | number;
    random_integers(low: number, high?: number | null, size?: Size | null): NDArray | number;
    multinomial(n: number, pvals: readonly number[], size?: Size | null): NDArray;
    dirichlet(alpha: readonly number[], size?: Size | null): NDArray;
  }
}

RandomState.prototype.triangular = function (
  this: RandomState,
  left: number,
  mode: number,
  right: number,
  size?: Size | null,
): NDArray | number {
  if (left > mode || mode > right) throw new ValueError("left <= mode <= right required");
  if (left === right) throw new ValueError("left == right");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.triangular(left, mode, right, shape ?? []), shape));
};

RandomState.prototype.random_integers = function (
  this: RandomState,
  low: number,
  high: number | null = null,
  size?: Size | null,
): NDArray | number {
  // np.random.random_integers(low, high=None) → integers in [low, high]
  const hi = high === null ? low : high;
  const lo = high === null ? 1 : low;
  const shape = toShape(size);
  return wrapNative(() =>
    finish(this._bg.integers(lo, hi, true, shape ?? [], "int64", true), shape),
  );
};

RandomState.prototype.multinomial = function (
  this: RandomState,
  n: number,
  pvals: readonly number[],
  size?: Size | null,
): NDArray {
  const shape = toShape(size);
  return wrapNative(() => NDArray._wrap(this._bg.multinomial(Math.trunc(n), Array.from(pvals), shape ?? [])));
};

RandomState.prototype.dirichlet = function (
  this: RandomState,
  alpha: readonly number[],
  size?: Size | null,
): NDArray {
  const shape = toShape(size);
  return wrapNative(() => NDArray._wrap(this._bg.dirichlet(Array.from(alpha), shape ?? [])));
};
