// P13-2 — continuous Generator distributions (D-161).
// Adds all continuous univariate distributions to np.random.Generator via
// declaration merging. The numerical kernels live in native/random/p13_distributions.hpp.

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
    triangular(left: number, mode: number, right: number, size?: Size | null): NDArray | number;
  }
}

// ---- parameter validation ----

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
function checkProb(p: number, name = "p"): void {
  if (!Number.isFinite(p) || p <= 0 || p > 1) throw new ValueError(`${name} out of (0, 1]`);
}

// ---- prototype methods ----

Generator.prototype.standardExponential = function (
  this: Generator,
  size?: Size | null,
): NDArray | number {
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.standardExponential(shape ?? []), shape));
};

Generator.prototype.exponential = function (
  this: Generator,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScale(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.exponential(scale, shape ?? []), shape));
};

Generator.prototype.standardGamma = function (
  this: Generator,
  shape: number,
  size?: Size | null,
): NDArray | number {
  checkAlpha(shape, "shape");
  const sz = toShape(size);
  return wrapNative(() => finish(this._bg.standardGamma(shape, sz ?? []), sz));
};

Generator.prototype.gamma = function (
  this: Generator,
  shape: number,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkAlpha(shape, "shape");
  checkScale(scale, "scale");
  const sz = toShape(size);
  return wrapNative(() => finish(this._bg.gamma(shape, scale, sz ?? []), sz));
};

Generator.prototype.beta = function (
  this: Generator,
  a: number,
  b: number,
  size?: Size | null,
): NDArray | number {
  checkAlpha(a, "a");
  checkAlpha(b, "b");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.beta(a, b, shape ?? []), shape));
};

Generator.prototype.chisquare = function (
  this: Generator,
  df: number,
  size?: Size | null,
): NDArray | number {
  checkDf(df);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.chisquare(df, shape ?? []), shape));
};

Generator.prototype.f = function (
  this: Generator,
  dfnum: number,
  dfden: number,
  size?: Size | null,
): NDArray | number {
  checkDf(dfnum);
  checkDf(dfden);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.f(dfnum, dfden, shape ?? []), shape));
};

Generator.prototype.standardCauchy = function (
  this: Generator,
  size?: Size | null,
): NDArray | number {
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.standardCauchy(shape ?? []), shape));
};

Generator.prototype.pareto = function (
  this: Generator,
  a: number,
  size?: Size | null,
): NDArray | number {
  checkAlpha(a);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.pareto(a, shape ?? []), shape));
};

Generator.prototype.weibull = function (
  this: Generator,
  a: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(a) || a < 0) throw new ValueError("a < 0");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.weibull(a, shape ?? []), shape));
};

Generator.prototype.power = function (
  this: Generator,
  a: number,
  size?: Size | null,
): NDArray | number {
  checkProb(a, "a");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.power(a, shape ?? []), shape));
};

Generator.prototype.laplace = function (
  this: Generator,
  loc = 0.0,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.laplace(loc, scale, shape ?? []), shape));
};

Generator.prototype.gumbel = function (
  this: Generator,
  loc = 0.0,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.gumbel(loc, scale, shape ?? []), shape));
};

Generator.prototype.logistic = function (
  this: Generator,
  loc = 0.0,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.logistic(loc, scale, shape ?? []), shape));
};

Generator.prototype.lognormal = function (
  this: Generator,
  mean = 0.0,
  sigma = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(sigma, "sigma");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.lognormal(mean, sigma, shape ?? []), shape));
};

Generator.prototype.rayleigh = function (
  this: Generator,
  scale = 1.0,
  size?: Size | null,
): NDArray | number {
  checkScaleNonNeg(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.rayleigh(scale, shape ?? []), shape));
};

Generator.prototype.standardT = function (
  this: Generator,
  df: number,
  size?: Size | null,
): NDArray | number {
  checkDf(df);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.standardT(df, shape ?? []), shape));
};

Generator.prototype.noncentralChisquare = function (
  this: Generator,
  df: number,
  nonc: number,
  size?: Size | null,
): NDArray | number {
  checkDf(df);
  checkNonc(nonc);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.noncentralChisquare(df, nonc, shape ?? []), shape));
};

Generator.prototype.noncentralF = function (
  this: Generator,
  dfnum: number,
  dfden: number,
  nonc: number,
  size?: Size | null,
): NDArray | number {
  checkDf(dfnum);
  checkDf(dfden);
  checkNonc(nonc);
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.noncentralF(dfnum, dfden, nonc, shape ?? []), shape));
};

Generator.prototype.wald = function (
  this: Generator,
  mean: number,
  scale: number,
  size?: Size | null,
): NDArray | number {
  checkScale(mean, "mean");
  checkScale(scale, "scale");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.wald(mean, scale, shape ?? []), shape));
};

Generator.prototype.vonmises = function (
  this: Generator,
  mu: number,
  kappa: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(kappa) || kappa < 0) throw new ValueError("kappa < 0");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.vonmises(mu, kappa, shape ?? []), shape));
};

Generator.prototype.triangular = function (
  this: Generator,
  left: number,
  mode: number,
  right: number,
  size?: Size | null,
): NDArray | number {
  if (!Number.isFinite(left) || !Number.isFinite(mode) || !Number.isFinite(right)) {
    throw new ValueError("left, mode, and right must be finite");
  }
  if (left > mode || mode > right) throw new ValueError("left <= mode <= right required");
  if (left === right) throw new ValueError("left == right");
  const shape = toShape(size);
  return wrapNative(() => finish(this._bg.triangular(left, mode, right, shape ?? []), shape));
};
