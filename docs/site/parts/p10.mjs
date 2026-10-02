// P10 API reference entries (D-056, D-130..D-136). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.
const arr = (name = "a") => ({ name, type: "ArrayLike", desc: "An `NDArray`, nested JS array or scalar." });
const axisArg = { name: "[options.axis]", type: "number | number[] | null", desc: "Axis or axes to reduce; default all (flattened)." };
const keepdimsArg = { name: "[options.keepdims]", type: "boolean", desc: "Keep reduced axes with length 1." };

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "statistics",
    title: "Statistics",
    intro: "Order statistics, averages, correlations and histograms (NumPy `numpy` statistics routines). Results are `NDArray`s (0-d when everything is reduced); tuple results are JS arrays.",
    entries: [
      {
        name: "quantile",
        sig: "np.quantile(a, q, { axis?, keepdims?, method?, weights? }) · np.percentile(a, q, ...) · np.nanquantile · np.nanpercentile",
        desc: "The `q`-th quantile (`q` in [0, 1]; `percentile` takes [0, 100]). All 13 NumPy `method`s are supported (`\"linear\"` default, `\"lower\"`, `\"higher\"`, `\"nearest\"`, `\"midpoint\"`, `\"inverted_cdf\"`, `\"averaged_inverted_cdf\"`, `\"closest_observation\"`, `\"interpolated_inverted_cdf\"`, `\"hazen\"`, `\"weibull\"`, `\"median_unbiased\"`, `\"normal_unbiased\"`). The result shape is `q.shape` followed by the reduced shape. Discrete methods keep the input dtype; the others give float64 for integer input, and a JS-number `q` keeps float32 input as float32. A slice containing NaN gives NaN; the `nan*` variants ignore NaNs. `weights` (non-negative) need `method: \"inverted_cdf\"` and either `a`'s shape or the shape of the reduced axes.",
        args: [arr(), { name: "q", type: "ArrayLike", desc: "Quantile(s), scalar or up to 2-d." }, axisArg, keepdimsArg, { name: "[options.method]", type: "QuantileMethod", desc: "Estimation method (default `\"linear\"`)." }, { name: "[options.weights]", type: "ArrayLike", desc: "Sample weights (`inverted_cdf` only)." }],
        returns: "NDArray",
        example: `const a = np.array([[10, 7, 4], [3, 2, 1]]);
np.quantile(a, 0.5);                       // => 3.5
np.percentile(a, [25, 75], { axis: 1 });   // => [[5.5, 1.5], [8.5, 2.5]]
np.quantile(a, 0.5, { method: "lower" });  // => 3
np.nanquantile([1, NaN, 3], 0.5);          // => 2
np.quantile([1, 2, 3], 0.5, { weights: [1, 1, 4], method: "inverted_cdf" }); // => 3`,
      },
      {
        name: "median",
        sig: "np.median(a, { axis?, keepdims? }) · np.nanmedian(a, ...)",
        desc: "Median along the given axes: the mean of the middle values, so integer input gives float64 (float16/float32 are kept). Any NaN in a slice gives NaN; `nanmedian` ignores NaNs (an all-NaN slice gives NaN). An empty input gives NaN.",
        args: [arr(), axisArg, keepdimsArg],
        returns: "NDArray",
        example: `np.median([[10, 7, 4], [3, 2, 1]], { axis: 0 }); // => [6.5, 4.5, 2.5]
Number.isNaN(np.median([1, NaN, 3]).item()); // => true
np.nanmedian([1, NaN, 3]); // => 2`,
      },
      {
        name: "cumsum",
        sig: "np.cumsum(a, { axis?, dtype?, out? }) · np.cumprod(...) · a.cumsum(...) · a.cumprod(...) · np.nancumsum · np.nancumprod",
        desc: "Running sum / product along `axis` (default: the flattened array). Integer and bool inputs accumulate in int64 (uint64 for unsigned), like `sum`. `nancumsum`/`nancumprod` treat NaN as 0 / 1. `out` must have the result shape; values are cast to its dtype.",
        args: [arr(), { name: "[options.axis]", type: "number | null", desc: "Axis; default flattened." }, { name: "[options.dtype]", type: "DTypeLike", desc: "Accumulator/result dtype." }, { name: "[options.out]", type: "NDArray", desc: "Destination array." }],
        returns: "NDArray",
        example: `const a = np.array([[1, 2], [3, 4]]);
np.cumsum(a);                 // => [1, 3, 6, 10]
a.cumprod({ axis: 0 });       // => [[1, 2], [3, 8]]
np.nancumsum([1, NaN, 2]);    // => [1, 1, 3]`,
      },
      {
        name: "cumulativeSum",
        sig: "np.cumulativeSum(x, { axis?, dtype?, includeInitial?, out? }) · np.cumulativeProd(x, ...)",
        desc: "Array-API `cumulative_sum` / `cumulative_prod`: like `cumsum`, but `axis` is required when `x` has more than one dimension, and `includeInitial: true` prepends the identity (0 or 1), so the axis grows by one.",
        args: [arr("x"), { name: "[options.axis]", type: "number", desc: "Axis (required for ndim > 1)." }, { name: "[options.includeInitial]", type: "boolean", desc: "Prepend the identity." }],
        returns: "NDArray",
        example: `np.cumulativeSum([1, 2, 3], { includeInitial: true }); // => [0, 1, 3, 6]
np.cumulativeProd([[1, 2], [3, 4]], { axis: 1 });      // => [[1, 2], [3, 12]]`,
      },
      {
        name: "diff",
        sig: "np.diff(a, n = 1, axis = -1) · np.diff(a, { n?, axis?, prepend?, append? })",
        desc: "The `n`-th discrete difference `a[i+1] - a[i]` along `axis` (bool input uses `!=`). `prepend`/`append` are joined to `a` along `axis` first; scalars are broadcast to length 1 along it. Integer differences wrap in the input dtype.",
        args: [arr(), { name: "[n]", type: "number", desc: "Number of differences (default 1)." }, { name: "[axis]", type: "number", desc: "Axis (default -1)." }, { name: "[options.prepend]", type: "ArrayLike", desc: "Values before `a`." }, { name: "[options.append]", type: "ArrayLike", desc: "Values after `a`." }],
        returns: "NDArray",
        example: `np.diff([1, 4, 9, 16]);                 // => [3, 5, 7]
np.diff([1, 4, 9, 16], 2);              // => [2, 2]
np.diff([1, 2], { prepend: 0 });        // => [1, 1]
np.diff([[1, 2], [4, 8]], { axis: 0 }); // => [[3, 6]]`,
      },
      {
        name: "ptp",
        sig: "np.ptp(a, { axis?, keepdims? }) · a.ptp(...)",
        desc: "Range of values (`max - min`) along the axes, in the input dtype (integer results can wrap, as in NumPy). Bool input raises `DTypeError`; empty input raises `ValueError`.",
        args: [arr(), axisArg, keepdimsArg],
        returns: "NDArray",
        example: `np.ptp([[1, 5], [2, 9]]);            // => 8
np.ptp([[1, 5], [2, 9]], { axis: 0 }); // => [1, 4]`,
      },
    ],
  },
];
