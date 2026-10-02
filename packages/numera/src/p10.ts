// Parity milestone P10: statistics and NaN reductions (D-056, D-130..D-139).
// Native kernels: native/core/p10_*.{hpp,cpp}, exposed as `addon.p10`.
import { nativeModule, type NativeNDArray } from "./addon.js";
import { array } from "./creation.js";
import { dtype as toDType, type DTypeLike } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import type { ArgReduceOptions, ReduceOptions, VarOptions } from "./reduce.js";
import type { ArrayLike } from "./ufunc.js";

type N = NativeNDArray;

interface NativeHistResult {
  hist: N;
  edges: N;
  warnings: string[];
}

interface NativeHistDdResult {
  hist: N;
  edges: N[];
}

type NativeBinsSpec = number | string | N | { count?: number; estimator?: string; edges?: N };

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
  correlate(a: N, v: N, mode: string): N;
  convolve(a: N, v: N, mode: string): N;
  histogram(
    a: N,
    bins: NativeBinsSpec,
    range: [number, number] | null,
    density: boolean,
    weights: N | null,
    edgesOnly: boolean,
  ): NativeHistResult;
  histogramdd(
    cols: N[],
    bins: NativeBinsSpec | NativeBinsSpec[],
    range: ([number, number] | null)[],
    density: boolean,
    weights: N | null,
  ): NativeHistDdResult;
  bincount(x: N, weights: N | null, minlength: number, fromList: boolean): N;
  digitize(x: N, bins: N, right: boolean): N;
  interp(
    x: N,
    xp: N,
    fp: N,
    left: { re: number; im: number } | null,
    right: { re: number; im: number } | null,
    period: number | null,
  ): N;
  reduceWhere(
    op: string,
    a: N,
    opts: {
      axis: number[] | null;
      keepdims: boolean;
      dtype: string | null;
      initial: number | null;
      ddof: number;
      where: N | null;
      out: N | null;
    },
  ): N;
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

// Used by reduce.ts to call reduceWhere when where= or out= are present (D-136).
export type { P10Native };
export const _p10native = (): P10Native => nativeModule<P10Native>("p10");

const native = (): P10Native => _p10native();
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

// ---- histogram / histogramBinEdges / histogram2d / histogramdd / bincount / digitize / interp (D-134, D-138) ----

/** NumPy `bins=` argument: integer count, estimator name, or explicit 1-d edges. */
export type BinsArg =
  | number
  | "auto" | "fd" | "sturges" | "scott" | "rice" | "doane" | "sqrt"
  | ArrayLike;

export interface HistogramResult {
  hist: NDArray;
  edges: NDArray;
}

export interface HistogramOptions {
  /** [min, max]; default: the data range (NaN ignored). */
  range?: [number, number] | null;
  /** Normalise so that the integral equals 1. */
  density?: boolean;
  /** Per-sample weights (same shape as `a`). */
  weights?: ArrayLike | null;
}

function binsToNative(bins: BinsArg): NativeBinsSpec {
  if (typeof bins === "number") return bins;
  if (typeof bins === "string") return bins;
  return toArray(bins)._native;
}

/**
 * NumPy `histogram` — count samples into evenly or explicitly spaced bins.
 *
 * Returns `{ hist, edges }` where `edges` has one more element than `hist`.
 * RuntimeWarning messages (e.g. identical range endpoints) are forwarded to
 * `process.emitWarning`.
 */
export function histogram(a: ArrayLike, bins: BinsArg = 10, opts: HistogramOptions = {}): HistogramResult {
  const xa = toArray(a);
  const w = opts.weights == null ? null : toArray(opts.weights);
  const { hist, edges, warnings } = wrapNative(() =>
    native().histogram(
      xa._native,
      binsToNative(bins),
      opts.range ?? null,
      opts.density ?? false,
      w?._native ?? null,
      false,
    ),
  );
  for (const msg of warnings) process.emitWarning(msg, "RuntimeWarning");
  return { hist: NDArray._wrap(hist), edges: NDArray._wrap(edges) };
}

/**
 * NumPy `histogram_bin_edges` — compute bin edges only.
 */
export function histogramBinEdges(a: ArrayLike, bins: BinsArg = 10, opts: HistogramOptions = {}): NDArray {
  const xa = toArray(a);
  const w = opts.weights == null ? null : toArray(opts.weights);
  const { edges, warnings } = wrapNative(() =>
    native().histogram(
      xa._native,
      binsToNative(bins),
      opts.range ?? null,
      false,
      w?._native ?? null,
      true,
    ),
  );
  for (const msg of warnings) process.emitWarning(msg, "RuntimeWarning");
  return NDArray._wrap(edges);
}

export interface Histogram2dResult {
  hist: NDArray;
  xedges: NDArray;
  yedges: NDArray;
}

export interface Histogram2dOptions extends Omit<HistogramOptions, "range"> {
  /** Per-axis bins: `[xbins, ybins]`. */
  bins?: [BinsArg, BinsArg];
  /** Per-axis ranges: `[[xmin, xmax], [ymin, ymax]]`. */
  range?: [[number, number], [number, number]] | null;
}

/**
 * NumPy `histogram2d` — 2-D histogram of two 1-D samples.
 *
 * Returns `{ hist, xedges, yedges }`.
 */
export function histogram2d(
  x: ArrayLike,
  y: ArrayLike,
  bins: BinsArg | [BinsArg, BinsArg] = 10,
  opts: Omit<Histogram2dOptions, "bins"> = {},
): Histogram2dResult {
  const xa = toArray(x);
  const ya = toArray(y);
  const w = opts.weights == null ? null : toArray(opts.weights);

  // Normalise bins to per-axis array.
  const binsSpec: NativeBinsSpec[] = Array.isArray(bins) && bins.length === 2 &&
    (typeof bins[0] !== "number" || typeof bins[1] !== "number" || bins.length === 2)
    ? (bins as [BinsArg, BinsArg]).map(binsToNative)
    : [binsToNative(bins as BinsArg), binsToNative(bins as BinsArg)];

  // Range.
  const rangeArr: ([number, number] | null)[] = opts.range
    ? [opts.range[0], opts.range[1]]
    : [null, null];

  const { hist, edges } = wrapNative(() =>
    native().histogramdd(
      [xa._native, ya._native],
      binsSpec,
      rangeArr,
      opts.density ?? false,
      w?._native ?? null,
    ),
  );
  return {
    hist: NDArray._wrap(hist),
    xedges: NDArray._wrap(edges[0]!),
    yedges: NDArray._wrap(edges[1]!),
  };
}

export interface HistogramddResult {
  hist: NDArray;
  edges: NDArray[];
}

export interface HistogramddOptions {
  density?: boolean;
  weights?: ArrayLike | null;
}

/**
 * NumPy `histogramdd` — multi-dimensional histogram.
 *
 * `sample` may be an `(N, D)` array or a 1-D array treated as a single column.
 * Returns `{ hist, edges }` where `edges` is an array of `D` edge arrays.
 */
export function histogramdd(
  sample: ArrayLike,
  bins: BinsArg | BinsArg[] = 10,
  opts: HistogramddOptions = {},
): HistogramddResult {
  const sa = toArray(sample);
  const w = opts.weights == null ? null : toArray(opts.weights);

  // Determine D from sample shape (or 1 if 1-D / scalar).
  const ndim = sa.ndim;
  const D = ndim >= 2 ? sa.shape[ndim - 1]! : 1;

  // Split (N, D) sample into D column arrays for the native call.
  // The native `histogramdd` binding accepts an array of column NDArrays.
  // We pass `fromList = (ndim === 1)` via the special column-split binding.
  // Actually, expose `sample_columns` via a wrapper: pass the whole array
  // and let the native side split it using the from_list flag (ndim <= 1).
  // Simplest: just pass the flattened sample and let native split it.
  // The binding's histogramdd receives cols: NDArray[] — pass the sample
  // as a single element and set fromList when 1-D.
  //
  // For 2-D (N, D) samples we split into D 1-D slices from JS:
  let cols: N[];
  if (ndim <= 1) {
    cols = [sa._native];
  } else {
    // (N, D) → D columns via row iteration using reshape/slice.
    // Simplest reliable approach: let native split via `sample_columns`.
    // We pass the sample as a 1-element array and signal fromList.
    // The C++ `histogramdd` signature expects actual columns, so we need
    // to either split here or use a different native entry point.
    // Use a flat pass: D columns extracted by JS.
    const n = sa.shape[0]!;
    cols = [];
    for (let d = 0; d < D; d++) {
      // Slice column d: reshape sa to (N, D) and extract column d.
      // Use NDArray indexing - slice with axis=1, index d.
      // Simplest: pass each column as a flat typed array via array().
      const col: number[] = [];
      const arr2d = sa.toArray() as number[][];
      for (let r = 0; r < n; r++) col.push((arr2d[r] as number[])[d]!);
      cols.push(toArray(col)._native);
    }
  }

  const binsSpec: NativeBinsSpec[] = Array.isArray(bins)
    ? (bins as BinsArg[]).map(binsToNative)
    : Array.from({ length: cols.length }, () => binsToNative(bins as BinsArg));

  const { hist, edges } = wrapNative(() =>
    native().histogramdd(cols, binsSpec, [], opts.density ?? false, w?._native ?? null),
  );
  return { hist: NDArray._wrap(hist), edges: edges.map((e) => NDArray._wrap(e)) };
}

export interface BincountOptions {
  /** Non-negative weights (same length as `x`). */
  weights?: ArrayLike | null;
  /** Minimum length of the output. */
  minlength?: number;
}

/**
 * NumPy `bincount` — count occurrences of each non-negative integer in `x`.
 *
 * Returns a 1-D integer (or float when `weights` is given) array of length
 * `max(x) + 1` or `minlength`, whichever is larger.
 */
export function bincount(x: ArrayLike, opts: BincountOptions = {}): NDArray {
  const fromList = !( x instanceof NDArray);
  const xa = toArray(x);
  const w = opts.weights == null ? null : toArray(opts.weights);
  return wrap(() =>
    native().bincount(xa._native, w?._native ?? null, opts.minlength ?? 0, fromList),
  );
}

/**
 * NumPy `digitize(x, bins, right)` — return the bin indices of the values in `x`.
 *
 * `bins` must be monotonically increasing or decreasing.
 * `right=false` (default): bins[i-1] <= x < bins[i].
 * `right=true`: bins[i-1] < x <= bins[i].
 */
export function digitize(x: ArrayLike, bins: ArrayLike, right = false): NDArray {
  return wrap(() => native().digitize(toArray(x)._native, toArray(bins)._native, right));
}

export interface InterpOptions {
  /** Value below `xp[0]`; default `fp[0]`. */
  left?: number | null;
  /** Value above `xp[-1]`; default `fp[-1]`. */
  right?: number | null;
  /** Wrap-around period (cyclic interpolation). */
  period?: number | null;
}

/**
 * NumPy `interp(x, xp, fp, left, right, period)` — 1-D piecewise linear interpolation.
 *
 * `xp` must be increasing (or decreasing when `period` is given). Complex `fp`
 * is supported; the result is complex128 when `fp` is complex.
 */
export function interp(x: ArrayLike, xp: ArrayLike, fp: ArrayLike, opts: InterpOptions = {}): NDArray {
  const toComplex = (v: number | null | undefined): { re: number; im: number } | null =>
    v == null ? null : { re: v, im: 0 };
  return wrap(() =>
    native().interp(
      toArray(x)._native,
      toArray(xp)._native,
      toArray(fp)._native,
      toComplex(opts.left),
      toComplex(opts.right),
      opts.period ?? null,
    ),
  );
}

// ---- where= / out= for np.sum/prod/min/max/mean/var/std (D-136, P10-7) ----
// These functions are used by reduce.ts when where= or out= is present.
// They live here to keep the where= kernel in p10.ts near the p10 binding.

export interface WhereReduceOptions extends ReduceOptions {
  /** Boolean mask: only elements where `where` is True contribute. */
  where?: ArrayLike | null;
  /** Pre-allocated output array. */
  out?: NDArray | null;
}

export interface WhereVarOptions extends VarOptions {
  where?: ArrayLike | null;
  out?: NDArray | null;
}

export function reduceWhere(
  op: string,
  a: ArrayLike,
  opts: WhereReduceOptions & { ddof?: number },
): NDArray {
  const x = toArray(a);
  const axis = opts.axis ?? null;
  const where = opts.where == null ? null : toArray(opts.where);
  const out = opts.out ?? null;
  return wrapNative(() =>
    NDArray._wrap(
      native().reduceWhere(op, x._native, {
        axis: axis === null ? null : typeof axis === "number" ? [axis] : [...axis],
        keepdims: opts.keepdims ?? false,
        dtype: opts.dtype == null ? null : toDType(opts.dtype).name,
        initial: opts.initial ?? null,
        ddof: opts.ddof ?? 0,
        where: where?._native ?? null,
        out: out?._native ?? null,
      }),
    ),
  );
}



/** Mode for {@link correlate} and {@link convolve}. */
export type ConvMode = "full" | "same" | "valid";

/**
 * NumPy `correlate(a, v, mode)` — 1-D cross-correlation.
 *
 * `c[k] = Σ_j  a[j+k] · conj(v[j])`
 *
 * Default mode is `"valid"` (matching NumPy).
 */
export function correlate(a: ArrayLike, v: ArrayLike, mode: ConvMode = "valid"): NDArray {
  return wrap(() => native().correlate(toArray(a)._native, toArray(v)._native, mode));
}

/**
 * NumPy `convolve(a, v, mode)` — discrete linear convolution.
 *
 * `c[k] = Σ_j  a[j] · v[k-j]`
 *
 * Default mode is `"full"` (matching NumPy).
 */
export function convolve(a: ArrayLike, v: ArrayLike, mode: ConvMode = "full"): NDArray {
  return wrap(() => native().convolve(toArray(a)._native, toArray(v)._native, mode));
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
  correlate,
  convolve,
  histogram,
  histogramBinEdges,
  histogram2d,
  histogramdd,
  bincount,
  digitize,
  interp,
} as const;
