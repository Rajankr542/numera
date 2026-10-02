// Parity milestone P10: statistics and NaN reductions (D-056, D-130..D-136).
// Native kernels: native/core/p10_*.{hpp,cpp}, exposed as `addon.p10`.
import { nativeModule, type NativeNDArray } from "./addon.js";
import { array } from "./creation.js";
import { wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
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

export const p10 = {
  quantile,
  percentile,
  nanquantile,
  nanpercentile,
  median,
  nanmedian,
} as const;
