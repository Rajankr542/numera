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
      {
        name: "nansum",
        sig: "np.nansum · np.nanprod · np.nanmean · np.nanvar · np.nanstd · np.nanmin · np.nanmax (a, { axis?, keepdims?, dtype?, initial?, ddof? })",
        desc: "Reductions that ignore NaN: NaN counts as 0 for `nansum`, 1 for `nanprod`, and is left out of the count for `nanmean`/`nanvar`/`nanstd`. All-NaN slices give 0 (`nansum`), 1 (`nanprod`) or NaN (the others; NumPy also warns). Integer input behaves like the plain reduction. Options are those of `sum`/`mean`/`var`/`min`.",
        args: [arr(), axisArg, keepdimsArg, { name: "[options.ddof]", type: "number", desc: "`nanvar`/`nanstd`: divisor is count - ddof." }],
        returns: "NDArray",
        example: `const x = np.array([[1, NaN, 3], [4, 5, NaN]]);
np.nansum(x);                 // => 13
np.nanmean(x, { axis: 1 });   // => [2, 4.5]
np.nanmax(x, { axis: 0 });    // => [4, 5, 3]
np.nanvar([1, NaN, 2], { ddof: 1 }); // => 0.5`,
      },
      {
        name: "nanargmin",
        sig: "np.nanargmin(a, { axis?, keepdims? }) · np.nanargmax(a, ...)",
        desc: "Index of the minimum / maximum ignoring NaNs. An all-NaN slice raises `ValueError(\"All-NaN slice encountered\")`.",
        args: [arr(), { name: "[options.axis]", type: "number | null", desc: "Axis; default the flattened array." }, keepdimsArg],
        returns: "NDArray (int64)",
        example: `np.nanargmax([NaN, 2, 5, NaN]);                  // => 2
np.nanargmin([[NaN, 1], [2, 3]], { axis: 0 });   // => [1, 0]`,
      },
      {
        name: "average",
        sig: "np.average(a, { axis?, weights?, returned?, keepdims? })",
        desc: "Weighted mean. `weights` has `a`'s shape or the shape of `a` along `axis`; integer input averages in float64. With `returned: true` the result is `[avg, sumOfWeights]`. Weights summing to zero raise `ValueError` (NumPy: `ZeroDivisionError`).",
        args: [arr(), axisArg, { name: "[options.weights]", type: "ArrayLike", desc: "Weights." }, { name: "[options.returned]", type: "boolean", desc: "Also return the sum of weights." }, keepdimsArg],
        returns: "NDArray | [NDArray, NDArray]",
        example: `np.average([1, 2, 3, 4]);                          // => 2.5
np.average([1, 2, 3], { weights: [3, 0, 1] });      // => 1.5
np.average([[1, 2], [3, 4]], { axis: 1, weights: [1, 3], returned: true }).map((r) => r.toArray()); // => [[1.75, 3.75], [4, 4]]`,
      },
      {
        name: "cov",
        sig: "np.cov(m, { y?, rowvar?, bias?, ddof?, fweights?, aweights?, dtype? }) · np.corrcoef(x, { y?, rowvar?, dtype? })",
        desc: "Covariance matrix (rows are variables unless `rowvar: false`), normalised by N - 1 (N with `bias`, N - ddof with `ddof`), with optional frequency and observation weights. `corrcoef` divides by the standard deviations and clips to [-1, 1]. Results are squeezed (one variable gives a 0-d array).",
        args: [arr("m"), { name: "[options.y]", type: "ArrayLike", desc: "Extra variables." }, { name: "[options.rowvar]", type: "boolean", desc: "Default true." }, { name: "[options.ddof]", type: "number", desc: "Overrides `bias`." }],
        returns: "NDArray",
        example: `np.cov([1, 2, 3]);                          // => 1
np.cov([1, 2, 3], { y: [1, 5, 2] });        // => [[1, 0.5], [0.5, 4.333333333333334]]
np.corrcoef([[1, 2, 3], [3, 2, 1]]);        // => [[1, -1], [-1, 1]]`,
      },
      {
        name: "gradient",
        sig: "np.gradient(f, ...spacing, { axis?, edgeOrder? })",
        desc: "Central differences in the interior and one-sided (order 1 or 2) differences at the edges. Spacing is a scalar, one scalar or coordinate array per axis, or nothing (unit spacing). Returns an `NDArray` for one axis, otherwise one array per axis. Integer input gives float64.",
        args: [arr("f"), { name: "...spacing", type: "number | ArrayLike", desc: "Scalar distances or 1-d coordinates." }, { name: "[options.edgeOrder]", type: "1 | 2", desc: "Edge accuracy (default 1)." }],
        returns: "NDArray | NDArray[]",
        example: `np.gradient([1, 2, 4, 7, 11]);                     // => [1, 1.5, 2.5, 3.5, 4]
np.gradient([1, 2, 4, 7, 11], 2);                  // => [0.5, 0.75, 1.25, 1.75, 2]
np.gradient([1, 2, 4, 7, 11], { edgeOrder: 2 });   // => [0.5, 1.5, 2.5, 3.5, 4.5]`,
      },
      {
        name: "trapezoid",
        sig: "np.trapezoid(y, { x?, dx?, axis? })",
        desc: "Composite trapezoidal integral of `y` along `axis` (default -1), using sample points `x` or uniform spacing `dx` (default 1).",
        args: [arr("y"), { name: "[options.x]", type: "ArrayLike", desc: "Sample points." }, { name: "[options.dx]", type: "number", desc: "Spacing when `x` is absent." }, { name: "[options.axis]", type: "number", desc: "Default -1." }],
        returns: "NDArray",
        example: `np.trapezoid([1, 2, 3]);                    // => 4
np.trapezoid([1, 2, 3], { x: [0, 1, 3] });  // => 6.5`,
      },
      {
        name: "correlate",
        sig: "np.correlate(a, v, mode?)",
        desc: "1-D cross-correlation: `c[k] = Σ_j a[j+k]·conj(v[j])`. Default mode is `'valid'`. When `mode='valid'` and `len(v) > len(a)`, returns the NumPy-compatible reversed result. Output dtype is `promote_types(a, v)`.",
        args: [arr("a"), arr("v"), { name: "[mode]", type: "'full' | 'same' | 'valid'", desc: "Default 'valid'." }],
        returns: "NDArray",
        example: `np.correlate([1, 2, 3], [0, 1, 0.5]);             // => [3.5]
np.correlate([1, 2, 3], [0, 1, 0.5], 'full');     // => [0.5, 2, 3.5, 3, 0]
np.correlate([1, 2, 3], [0, 1, 0.5], 'same');     // => [2, 3.5, 3]`,
      },
      {
        name: "convolve",
        sig: "np.convolve(a, v, mode?)",
        desc: "Discrete linear convolution: `c[k] = Σ_j a[j]·v[k-j]`. Equivalent to correlating `a` with the reversed (un-conjugated) `v`. Default mode is `'full'`. Output dtype is `promote_types(a, v)`.",
        args: [arr("a"), arr("v"), { name: "[mode]", type: "'full' | 'same' | 'valid'", desc: "Default 'full'." }],
        returns: "NDArray",
        example: `np.convolve([1, 2, 3], [0, 1, 0.5]);             // => [0, 1, 2.5, 4, 1.5]
np.convolve([1, 2, 3], [0, 1, 0.5], 'same');     // => [1, 2.5, 4]
np.convolve([1, 2, 3], [0, 1, 0.5], 'valid');    // => [2.5]`,
      },
      {
        name: "histogram",
        sig: "np.histogram(a, bins?, { range?, density?, weights? }) · np.histogramBinEdges(a, bins?, ...)",
        desc: "`histogram` counts sample values into `bins` (integer count, estimator name, or explicit edges) and returns `{ hist, edges }` where `edges.length === hist.length + 1`. The rightmost bin is closed on both sides (NumPy behaviour). `density: true` normalises so that the integral over the histogram equals 1. `histogramBinEdges` returns only the edges.",
        args: [arr(), { name: "[bins]", type: "number | BinsArg", desc: "Default 10. Estimators: 'auto', 'fd', 'sturges', 'scott', 'rice', 'doane', 'sqrt'." }, { name: "[options.range]", type: "[number, number]", desc: "Data range [min, max]." }, { name: "[options.density]", type: "boolean", desc: "Normalise to density." }, { name: "[options.weights]", type: "ArrayLike", desc: "Per-sample weights." }],
        returns: "{ hist: NDArray; edges: NDArray }",
        example: `const { hist, edges } = np.histogram([1, 2, 1, 3], 3);
hist.toArray();   // => [2, 1, 1]
edges.toArray();  // => [1, 1.6666666666666667, 2.333333333333333, 3]
np.histogramBinEdges([1, 2, 3, 4], 2).toArray(); // => [1, 2.5, 4]`,
      },
      {
        name: "histogram2d",
        sig: "np.histogram2d(x, y, bins?, { range?, density?, weights? })",
        desc: "2-D histogram of two 1-D samples. Returns `{ hist, xedges, yedges }` where `hist.shape === [xbins, ybins]`. `bins` may be a scalar (applied to both axes) or `[xbins, ybins]`.",
        args: [arr("x"), arr("y"), { name: "[bins]", type: "number | [BinsArg, BinsArg]", desc: "Default 10." }],
        returns: "{ hist: NDArray; xedges: NDArray; yedges: NDArray }",
        example: `const { hist, xedges, yedges } = np.histogram2d([0,1,2],[0,1,2], 3);
hist.shape;         // => [3, 3]
xedges.size;        // => 4`,
      },
      {
        name: "histogramdd",
        sig: "np.histogramdd(sample, bins?, { density?, weights? })",
        desc: "Multi-dimensional histogram. `sample` is an `(N, D)` array or a 1-D array (treated as 1 column). Returns `{ hist, edges }` where `edges` is an array of D edge arrays.",
        args: [arr("sample"), { name: "[bins]", type: "number | BinsArg[]", desc: "Per-axis bins (scalar broadcast to all axes)." }],
        returns: "{ hist: NDArray; edges: NDArray[] }",
        example: `const { hist, edges } = np.histogramdd([[0,0],[1,1],[2,2]], 2);
hist.shape;       // => [2, 2]
edges.length;     // => 2`,
      },
      {
        name: "bincount",
        sig: "np.bincount(x, { weights?, minlength? })",
        desc: "Count occurrences of each non-negative integer in `x`. Returns a 1-D array of length `max(x) + 1` or `minlength`, whichever is larger. With `weights`, sums weights instead of counting.",
        args: [arr(), { name: "[options.weights]", type: "ArrayLike", desc: "Per-element weights." }, { name: "[options.minlength]", type: "number", desc: "Minimum output length." }],
        returns: "NDArray",
        example: `np.bincount([1, 0, 2, 0, 1]).toArray();           // => [2, 2, 1]
np.bincount([0, 1], { minlength: 5 }).toArray();   // => [1, 1, 0, 0, 0]`,
      },
      {
        name: "digitize",
        sig: "np.digitize(x, bins, right?)",
        desc: "Return indices such that `bins[i-1] <= x < bins[i]` (`right=false`, default) or `bins[i-1] < x <= bins[i]` (`right=true`). `bins` must be monotonic. Output shape matches `x`.",
        args: [arr(), { name: "bins", type: "ArrayLike", desc: "Monotonic bin edges." }, { name: "[right]", type: "boolean", desc: "Default false." }],
        returns: "NDArray",
        example: `np.digitize([0.2, 6.4, 3.0, 1.6], [0, 1, 2.5, 4, 10]).toArray(); // => [1, 4, 3, 2]`,
      },
      {
        name: "interp",
        sig: "np.interp(x, xp, fp, { left?, right?, period? })",
        desc: "1-D piecewise-linear interpolation. `xp` must be increasing (or decreasing when `period` is given). Values outside the range clamp to `fp[0]` / `fp[-1]` unless `left` / `right` are given. Complex `fp` is supported.",
        args: [arr(), { name: "xp", type: "ArrayLike", desc: "Sorted x-coordinates of the data." }, { name: "fp", type: "ArrayLike", desc: "y-coordinates (may be complex)." }, { name: "[options.left]", type: "number", desc: "Fill below xp[0]." }, { name: "[options.right]", type: "number", desc: "Fill above xp[-1]." }, { name: "[options.period]", type: "number", desc: "Wrap-around period." }],
        returns: "NDArray",
        example: `np.interp([0, 1, 1.5, 2, 2.5, 3], [1, 2, 3], [3, 2, 0]).toArray(); // => [3, 3, 2.5, 2, 1, 0]
np.interp([-1, 5], [0, 1, 2], [0, 1, 2], { left: -99, right: 99 }).toArray(); // => [-99, 99]`,
      },
    ],
  },
  {
    id: "reduce-where",
    title: "Reductions with where= / out=",
    intro: "All seven scalar reductions (`sum`, `prod`, `min`, `max`, `mean`, `var`, `std`) accept `where` (a boolean mask broadcast to `a`) and `out` (a pre-allocated result array). Only elements where the mask is `true` contribute; masked-out positions use the reduction identity (0 for sum, 1 for prod; for `min`/`max` you must supply `initial`). `mean`/`var`/`std` divide by the count of `true` positions.",
    entries: [
      {
        name: "sum (where=)",
        sig: "np.sum(a, { where, out? })",
        desc: "`where` is a bool `ArrayLike` broadcast to `a`. Masked-out positions contribute 0.",
        args: [arr(), { name: "options.where", type: "ArrayLike", desc: "Boolean mask; `true` positions are summed." }, { name: "[options.out]", type: "NDArray", desc: "Pre-allocated output (exact result shape)." }],
        returns: "NDArray",
        example: `const a = np.array([1, 2, 3, 4, 5], { dtype: "float64" });
const mask = np.array([true, false, true, false, true]);
np.sum(a, { where: mask }).item();   // => 9`,
      },
      {
        name: "mean (where=)",
        sig: "np.mean(a, { where, out? })",
        desc: "Mean of the unmasked elements; divisor is the count of `true` positions.",
        args: [arr(), { name: "options.where", type: "ArrayLike", desc: "Boolean mask." }],
        returns: "NDArray",
        example: `const a = np.array([1, 2, 3, 4, 5], { dtype: "float64" });
const mask = np.array([true, false, true, false, true]);
np.mean(a, { where: mask }).item();  // => 3`,
      },
      {
        name: "var / std (where=)",
        sig: "np.var(a, { where, ddof? }) · np.std(a, { where, ddof? })",
        desc: "Variance / standard deviation over unmasked elements.",
        args: [arr(), { name: "options.where", type: "ArrayLike", desc: "Boolean mask." }, { name: "[options.ddof]", type: "number", desc: "Delta degrees of freedom (default 0)." }],
        returns: "NDArray",
        example: `const a = np.array([1, 2, 3, 4, 5], { dtype: "float64" });
const mask = np.array([true, false, true, false, true]);
np.var(a, { where: mask }).item();   // => 2.6666666666666665
np.std(a, { where: mask }).item();   // => 1.632993161855452`,
      },
      {
        name: "min / max (where= + initial=)",
        sig: "np.min(a, { where, initial }) · np.max(a, { where, initial })",
        desc: "`min`/`max` with a mask require `initial` (the identity / seed value), matching NumPy's requirement.",
        args: [arr(), { name: "options.where", type: "ArrayLike", desc: "Boolean mask." }, { name: "options.initial", type: "number", desc: "Required seed value." }],
        returns: "NDArray",
        example: `const a = np.array([1, 2, 3, 4], { dtype: "float64" });
const mask = np.array([false, true, false, true]);
np.min(a, { where: mask, initial: 1e10 }).item();  // => 2
np.max(a, { where: mask, initial: -1e10 }).item(); // => 4`,
      },
      {
        name: "out= parameter",
        sig: "np.sum/prod/min/max/mean/var/std(a, { out })",
        desc: "Write the result into a pre-allocated `NDArray`. `out.shape` must exactly match the result shape. The same array is returned.",
        args: [arr(), { name: "options.out", type: "NDArray", desc: "Pre-allocated output (exact result shape)." }],
        returns: "NDArray (same object as out)",
        example: `const a = np.array([1, 2, 3, 4, 5], { dtype: "float64" });
const out = np.zeros([]);
np.sum(a, { out });
out.item();  // => 15`,
      },
    ],
  },
];
