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
    ],
  },
];
