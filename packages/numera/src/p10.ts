// Parity milestone P10: statistics and NaN reductions (D-056, D-130..D-136).
// Native kernels: native/core/p10_*.{hpp,cpp}, exposed as `addon.p10`.
import { nativeModule, type NativeNDArray } from "./addon.js";
import { array } from "./creation.js";
import { dtype as toDType, type DTypeLike } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import type { ArgReduceOptions, ReduceOptions, VarOptions } from "./reduce.js";
import type { ArrayLike } from "./ufunc.js";

type N = NativeNDArray;
interface P10Native {
  quantile(
    a: N,
    q: N,
    opts: {
      axis: number[] | null;
      keepdims: boolean;
      method?: string;
      weights?: N | null;
      weakQ: boolean;
      percentile: boolean;
      ignoreNan: boolean;
    },
  ): N;
  median(a: N, axis: number[] | null, keepdims: boolean, ignoreNan: boolean): N;
  cumulative(
    prod: boolean,
    a: N,
    opts: { axis: number | null; dtype: string | null; includeInitial: boolean; skipNan: boolean; arrayApi: boolean },
    out: N | null,
  ): N;
  diff(a: N, n: number, axis: number, prepend: N | null, append: N | null): N;
  ptp(a: N, axis: number[] | null, keepdims: boolean): N;
  nanReduce(op: string, a: N, opts: NativeReduceOptions): N;
  nanArgReduce(isMax: boolean, a: N, axis: number | null, keepdims: boolean): N;
  average(a: N, axis: number[] | null, weights: N | null, keepdims: boolean): [N, N];
  cov(m: N, opts: NativeCovOptions): [N, boolean];
  gradient(f: N, spacing: (number | N)[], axis: number[] | null, edgeOrder: number): N[];
  trapezoid(y: N, x: N | null, dx: number, axis: number): N;
}

interface NativeCovOptions {
  y: N | null;
  rowvar: boolean;
  bias: boolean;
  ddof: number | null;
  fweights: N | null;
  aweights: N | null;
  dtype: string | null;
  corrcoef: boolean;
}

interface NativeReduceOptions {
  axis: number[] | null;
  keepdims: boolean;
  dtype: string | null;
  initial: number | null;
  ddof: number;
}

const native = (): P10Native => nativeModule<P10Native>("p10");
const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));
const wrap = (f: () => N): NDArray => wrapNative(() => NDArray._wrap(f()));

/** Axis option shared by the P10 reductions: one axis, several, or `null` (all). */
export type Axis = number | readonly number[] | null;
const axes = (axis: Axis | undefined): number[] | null =>
  axis == null ? null : typeof axis === "number" ? [axis] : [...axis];

// ---- quantile / percentile / median (D-131) ----

/** NumPy quantile `method=` names. */
export type QuantileMethod =
  | "inverted_cdf"
  | "averaged_inverted_cdf"
  | "closest_observation"
  | "interpolated_inverted_cdf"
  | "hazen"
  | "weibull"
  | "linear"
  | "median_unbiased"
  | "normal_unbiased"
  | "lower"
  | "higher"
  | "midpoint"
  | "nearest";

export interface QuantileOptions {
  axis?: Axis;
  keepdims?: boolean;
  /** Default `"linear"`. */
  method?: QuantileMethod;
  /** Non-negative weights (only with `method: "inverted_cdf"`). */
  weights?: ArrayLike | null;
}

export interface MedianOptions {
  axis?: Axis;
  keepdims?: boolean;
}

function quantileImpl(
  a: ArrayLike,
  q: ArrayLike,
  opts: QuantileOptions,
  percentile: boolean,
  ignoreNan: boolean,
): NDArray {
  const x = toArray(a);
  const qa = toArray(q);
  const w = opts.weights == null ? null : toArray(opts.weights);
  return wrap(() =>
    native().quantile(x._native, qa._native, {
      axis: axes(opts.axis),
      keepdims: opts.keepdims ?? false,
      method: opts.method,
      weights: w?._native ?? null,
      weakQ: typeof q === "number",
      percentile,
      ignoreNan,
    }),
  );
}

/** NumPy quantile: `q` in [0, 1]; result shape is `q.shape` + the reduced shape. */
export const quantile = (a: ArrayLike, q: ArrayLike, opts: QuantileOptions = {}): NDArray =>
  quantileImpl(a, q, opts, false, false);
/** NumPy percentile: `q` in [0, 100]. */
export const percentile = (a: ArrayLike, q: ArrayLike, opts: QuantileOptions = {}): NDArray =>
  quantileImpl(a, q, opts, true, false);
/** NumPy nanquantile: NaNs are ignored (all-NaN slices give NaN). */
export const nanquantile = (a: ArrayLike, q: ArrayLike, opts: QuantileOptions = {}): NDArray =>
  quantileImpl(a, q, opts, false, true);
/** NumPy nanpercentile. */
export const nanpercentile = (a: ArrayLike, q: ArrayLike, opts: QuantileOptions = {}): NDArray =>
  quantileImpl(a, q, opts, true, true);

function medianImpl(a: ArrayLike, opts: MedianOptions, ignoreNan: boolean): NDArray {
  const x = toArray(a);
  return wrap(() => native().median(x._native, axes(opts.axis), opts.keepdims ?? false, ignoreNan));
}

/** NumPy median (mean of the middle values; int input gives float64). */
export const median = (a: ArrayLike, opts: MedianOptions = {}): NDArray => medianImpl(a, opts, false);
/** NumPy nanmedian. */
export const nanmedian = (a: ArrayLike, opts: MedianOptions = {}): NDArray => medianImpl(a, opts, true);

// ---- cumulative sums / products, diff, ptp (D-132) ----

export interface CumsumOptions {
  /** Axis to accumulate along; default the flattened array. */
  axis?: number | null;
  /** Accumulator/result dtype (default as `sum`: integers widen to int64/uint64). */
  dtype?: DTypeLike | null;
  /** Result array of the exact result shape (values cast unsafely). */
  out?: NDArray | null;
}

export interface CumulativeOptions extends CumsumOptions {
  /** Prepend the identity (0 or 1) along `axis`. */
  includeInitial?: boolean;
}

function cumImpl(prod: boolean, a: ArrayLike, opts: CumulativeOptions, skipNan: boolean, arrayApi: boolean): NDArray {
  const x = toArray(a);
  return wrap(() =>
    native().cumulative(
      prod,
      x._native,
      {
        axis: opts.axis ?? null,
        dtype: opts.dtype == null ? null : toDType(opts.dtype).name,
        includeInitial: opts.includeInitial ?? false,
        skipNan,
        arrayApi,
      },
      opts.out?._native ?? null,
    ),
  );
}

/** NumPy cumsum (flattened unless `axis` is given). */
export const cumsum = (a: ArrayLike, opts: CumsumOptions = {}): NDArray => cumImpl(false, a, opts, false, false);
/** NumPy cumprod. */
export const cumprod = (a: ArrayLike, opts: CumsumOptions = {}): NDArray => cumImpl(true, a, opts, false, false);
/** NumPy cumulative_sum (array API): `axis` is required for ndim > 1. */
export const cumulativeSum = (x: ArrayLike, opts: CumulativeOptions = {}): NDArray =>
  cumImpl(false, x, opts, false, true);
/** NumPy cumulative_prod (array API). */
export const cumulativeProd = (x: ArrayLike, opts: CumulativeOptions = {}): NDArray =>
  cumImpl(true, x, opts, false, true);
/** NumPy nancumsum: NaNs count as 0. */
export const nancumsum = (a: ArrayLike, opts: CumsumOptions = {}): NDArray => cumImpl(false, a, opts, true, false);
/** NumPy nancumprod: NaNs count as 1. */
export const nancumprod = (a: ArrayLike, opts: CumsumOptions = {}): NDArray => cumImpl(true, a, opts, true, false);

export interface DiffOptions {
  /** Number of times to difference (default 1). */
  n?: number;
  /** Axis (default -1). */
  axis?: number;
  /** Values joined before / after `a` along `axis` (scalars are broadcast). */
  prepend?: ArrayLike | null;
  append?: ArrayLike | null;
}

/** NumPy diff: `n`-th discrete difference along `axis` (bool input uses `!=`). */
export function diff(a: ArrayLike, n: number | DiffOptions = 1, axis = -1): NDArray {
  const o: DiffOptions = typeof n === "number" ? { n, axis } : n;
  const x = toArray(a);
  const pre = o.prepend == null ? null : toArray(o.prepend);
  const app = o.append == null ? null : toArray(o.append);
  return wrap(() => native().diff(x._native, o.n ?? 1, o.axis ?? -1, pre?._native ?? null, app?._native ?? null));
}

export interface PtpOptions {
  axis?: Axis;
  keepdims?: boolean;
}

/** NumPy ptp: `max - min` in the input dtype (integers wrap). */
export function ptp(a: ArrayLike, opts: PtpOptions = {}): NDArray {
  const x = toArray(a);
  return wrap(() => native().ptp(x._native, axes(opts.axis), opts.keepdims ?? false));
}

// ---- NaN reductions (D-133) ----

function nanImpl(op: string, a: ArrayLike, opts: ReduceOptions & { ddof?: number }): NDArray {
  const x = toArray(a);
  return wrap(() =>
    native().nanReduce(op, x._native, {
      axis: axes(opts.axis),
      keepdims: opts.keepdims ?? false,
      dtype: opts.dtype == null ? null : toDType(opts.dtype).name,
      initial: opts.initial ?? null,
      ddof: opts.ddof ?? 0,
    }),
  );
}

/** NumPy nansum: NaNs count as 0. */
export const nansum = (a: ArrayLike, opts: ReduceOptions = {}): NDArray => nanImpl("sum", a, opts);
/** NumPy nanprod: NaNs count as 1. */
export const nanprod = (a: ArrayLike, opts: ReduceOptions = {}): NDArray => nanImpl("prod", a, opts);
/** NumPy nanmean: mean of the non-NaN values (all-NaN slices give NaN). */
export const nanmean = (a: ArrayLike, opts: Omit<ReduceOptions, "initial"> = {}): NDArray => nanImpl("mean", a, opts);
/** NumPy nanvar. */
export const nanvar = (a: ArrayLike, opts: VarOptions = {}): NDArray => nanImpl("var", a, opts);
/** NumPy nanstd. */
export const nanstd = (a: ArrayLike, opts: VarOptions = {}): NDArray => nanImpl("std", a, opts);
/** NumPy nanmin (all-NaN slices give NaN). */
export const nanmin = (a: ArrayLike, opts: Omit<ReduceOptions, "dtype"> = {}): NDArray => nanImpl("min", a, opts);
/** NumPy nanmax. */
export const nanmax = (a: ArrayLike, opts: Omit<ReduceOptions, "dtype"> = {}): NDArray => nanImpl("max", a, opts);

function nanArg(isMax: boolean, a: ArrayLike, opts: ArgReduceOptions): NDArray {
  const x = toArray(a);
  return wrap(() => native().nanArgReduce(isMax, x._native, opts.axis ?? null, opts.keepdims ?? false));
}
/** NumPy nanargmin: raises `ValueError` for an all-NaN slice. */
export const nanargmin = (a: ArrayLike, opts: ArgReduceOptions = {}): NDArray => nanArg(false, a, opts);
/** NumPy nanargmax. */
export const nanargmax = (a: ArrayLike, opts: ArgReduceOptions = {}): NDArray => nanArg(true, a, opts);

// ---- average / cov / corrcoef / gradient / trapezoid (D-135, D-137) ----

export interface AverageOptions {
  axis?: Axis;
  /** Weights with `a`'s shape, or the shape of `a` along `axis`. */
  weights?: ArrayLike | null;
  /** Also return the sum of the weights (`[avg, sumOfWeights]`). */
  returned?: boolean;
  keepdims?: boolean;
}

/** NumPy average: weighted mean along `axis` (integers average in float64). */
export function average(a: ArrayLike, opts: AverageOptions & { returned: true }): [NDArray, NDArray];
export function average(a: ArrayLike, opts?: AverageOptions): NDArray;
export function average(a: ArrayLike, opts: AverageOptions = {}): NDArray | [NDArray, NDArray] {
  const x = toArray(a);
  const w = opts.weights == null ? null : toArray(opts.weights);
  const [avg, scl] = wrapNative(() =>
    native().average(x._native, axes(opts.axis), w?._native ?? null, opts.keepdims ?? false),
  );
  const r = NDArray._wrap(avg);
  return opts.returned ? [r, NDArray._wrap(scl)] : r;
}

export interface CovOptions {
  /** Extra variables, stacked after `m`. */
  y?: ArrayLike | null;
  /** Rows are variables (default true); false: columns are variables. */
  rowvar?: boolean;
  /** Normalise by N (ddof 0) instead of N - 1. */
  bias?: boolean;
  /** Overrides `bias`. */
  ddof?: number | null;
  /** Integer frequency weights per observation. */
  fweights?: ArrayLike | null;
  /** Observation weights. */
  aweights?: ArrayLike | null;
  dtype?: DTypeLike | null;
}

function covImpl(m: ArrayLike, opts: CovOptions, corrcoef: boolean): NDArray {
  if (opts.ddof != null && !Number.isInteger(opts.ddof)) throw new ValueError("ddof must be integer");
  const x = toArray(m);
  const opt = (v: ArrayLike | null | undefined) => (v == null ? null : toArray(v)._native);
  const [c, warn] = wrapNative(() =>
    native().cov(x._native, {
      y: opt(opts.y),
      rowvar: opts.rowvar ?? true,
      bias: opts.bias ?? false,
      ddof: opts.ddof ?? null,
      fweights: opt(opts.fweights),
      aweights: opt(opts.aweights),
      dtype: opts.dtype == null ? null : toDType(opts.dtype).name,
      corrcoef,
    }),
  );
  if (warn) process.emitWarning("Degrees of freedom <= 0 for slice", "RuntimeWarning");
  return NDArray._wrap(c);
}

/** NumPy cov: covariance matrix (squeezed; a single variable gives a 0-d array). */
export const cov = (m: ArrayLike, opts: CovOptions = {}): NDArray => covImpl(m, opts, false);

export type CorrcoefOptions = Pick<CovOptions, "y" | "rowvar" | "dtype">;
/** NumPy corrcoef: Pearson correlation coefficients, clipped to [-1, 1]. */
export const corrcoef = (x: ArrayLike, opts: CorrcoefOptions = {}): NDArray =>
  covImpl(x, { y: opts.y, rowvar: opts.rowvar, dtype: opts.dtype }, true);

export interface GradientOptions {
  axis?: Axis;
  /** 1 (default) or 2: accuracy of the one-sided edge differences. */
  edgeOrder?: number;
}

/** A sample spacing: a scalar distance or the 1-d coordinates along an axis. */
export type Spacing = number | ArrayLike;

/**
 * NumPy gradient(f, *varargs, axis, edge_order). Spacings follow `f`
 * (none, one for every axis, or one per axis); a trailing plain object is the
 * options. Returns an `NDArray` for one axis, otherwise an `NDArray[]`.
 */
export function gradient(f: ArrayLike, ...args: (Spacing | GradientOptions)[]): NDArray | NDArray[] {
  let opts: GradientOptions = {};
  const last = args[args.length - 1];
  if (last !== undefined && typeof last === "object" && !(last instanceof NDArray) && !Array.isArray(last)) {
    opts = last as GradientOptions;
    args = args.slice(0, -1);
  }
  const x = toArray(f);
  const sp = (args as Spacing[]).map((s) => (typeof s === "number" ? s : toArray(s)._native));
  const outs = wrapNative(() => native().gradient(x._native, sp, axes(opts.axis), opts.edgeOrder ?? 1));
  const r = outs.map((o) => NDArray._wrap(o));
  return r.length === 1 ? r[0]! : r;
}

export interface TrapezoidOptions {
  /** Sample coordinates (1-d along `axis`, or `y`'s shape). */
  x?: ArrayLike | null;
  /** Spacing when `x` is not given (default 1). */
  dx?: number;
  /** Default -1. */
  axis?: number;
}

/** NumPy trapezoid: integral of `y` by the composite trapezoidal rule. */
export function trapezoid(y: ArrayLike, opts: TrapezoidOptions = {}): NDArray {
  const ya = toArray(y);
  const xa = opts.x == null ? null : toArray(opts.x);
  return wrap(() => native().trapezoid(ya._native, xa?._native ?? null, opts.dx ?? 1, opts.axis ?? -1));
}

declare module "./ndarray.js" {
  interface NDArray {
    /** NumPy `a.cumsum({axis, dtype, out})`. */
    cumsum(opts?: CumsumOptions): NDArray;
    /** NumPy `a.cumprod({axis, dtype, out})`. */
    cumprod(opts?: CumsumOptions): NDArray;
    /** NumPy `np.ptp(a, {axis, keepdims})` as a method. */
    ptp(opts?: PtpOptions): NDArray;
  }
}
NDArray.prototype.cumsum = function (this: NDArray, opts: CumsumOptions = {}) {
  return cumsum(this, opts);
};
NDArray.prototype.cumprod = function (this: NDArray, opts: CumsumOptions = {}) {
  return cumprod(this, opts);
};
NDArray.prototype.ptp = function (this: NDArray, opts: PtpOptions = {}) {
  return ptp(this, opts);
};

export const p10 = {
  quantile,
  percentile,
  nanquantile,
  nanpercentile,
  median,
  nanmedian,
  cumsum,
  cumprod,
  cumulativeSum,
  cumulativeProd,
  nancumsum,
  nancumprod,
  diff,
  ptp,
  nansum,
  nanprod,
  nanmean,
  nanvar,
  nanstd,
  nanmin,
  nanmax,
  nanargmin,
  nanargmax,
  average,
  cov,
  corrcoef,
  gradient,
  trapezoid,
} as const;
