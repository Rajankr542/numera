// Parity milestone P10: statistics and NaN reductions (D-056, D-130..D-136).
// Native kernels: native/core/p10_*.{hpp,cpp}, exposed as `addon.p10`.
import { nativeModule, type NativeNDArray } from "./addon.js";
import { array } from "./creation.js";
import { dtype as toDType, type DTypeLike } from "./dtype.js";
import { wrapNative } from "./errors.js";
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
} as const;
