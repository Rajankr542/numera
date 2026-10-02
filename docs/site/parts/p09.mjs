// P9 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.

const axisSort = { name: "[options.axis]", type: "number | null", desc: "Axis to work along (default `-1`). `null` uses the flattened array." };
const kindSort = { name: "[options.kind]", type: "string", desc: "`\"quicksort\"`, `\"heapsort\"`, `\"mergesort\"` or `\"stable\"`. Cannot be combined with `stable`/`descending`." };
const stableOpt = { name: "[options.stable]", type: "boolean", desc: "Keep equal elements in their original order." };
const descOpt = { name: "[options.descending]", type: "boolean", desc: "Sort largest first. NaNs still go last." };

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "sorting",
    title: "Sorting, searching and sets",
    intro: "Sorts, partitions, binary search, unique values and set operations. NaNs sort after all other values. Complex numbers are ordered by real part, then imaginary part.",
    entries: [
      {
        name: "sort",
        sig: "np.sort(a, [options])",
        desc: "Returns a sorted copy. `a.sort(options)` sorts in place along an integer axis.",
        args: [{ name: "a", type: "ArrayLike", desc: "Input array." }, axisSort, kindSort, stableOpt, descOpt],
        returns: "NDArray",
        example: `np.sort([3, 1, 2]); // => [1, 2, 3]
np.sort([[3, 1], [0, 2]], { axis: 0 }); // => [[0, 1], [3, 2]]
np.sort([[3, 1], [0, 2]], { axis: null }); // => [0, 1, 2, 3]
np.sort([3, 1, 2], { descending: true }); // => [3, 2, 1]
const a = np.array([[3, 1, 2], [9, 7, 8]]);
a.sort();
a; // => [[1, 2, 3], [7, 8, 9]]`,
      },
      {
        name: "argsort",
        sig: "np.argsort(a, [options])",
        desc: "Indices that would sort the array, as int64. Also available as `a.argsort(options)`.",
        args: [{ name: "a", type: "ArrayLike", desc: "Input array." }, axisSort, kindSort, stableOpt, descOpt],
        returns: "NDArray (int64)",
        example: `np.argsort([1, 0, 1, 0], { stable: true }); // => [1, 3, 0, 2]
np.array([3, 1, 2]).argsort(); // => [1, 2, 0]`,
      },
      {
        name: "sortComplex",
        sig: "np.sortComplex(a)",
        desc: "Sorts along the last axis and returns a complex array (complex64 for 8- and 16-bit integer input, otherwise complex128).",
        returns: "NDArray (complex)",
        example: `np.sortComplex([5, 3, 6]).dtype.name; // => "complex128"
np.real(np.sortComplex([5, 3, 6])); // => [3, 5, 6]`,
      },
      {
        name: "partition",
        sig: "np.partition(a, kth, [options])",
        desc: "Copy in which element `kth` (an index or list of indices) is where it would be after sorting. Smaller elements come before it and larger ones after it, in no particular order. `a.partition(kth)` works in place.",
        args: [
          { name: "a", type: "ArrayLike", desc: "Input array." },
          { name: "kth", type: "number | number[] | NDArray", desc: "Index or indices to place. Negative values count from the end." },
          { name: "[options.axis]", type: "number | null", desc: "Axis (default `-1`). `null` uses the flattened array." },
          { name: "[options.kind]", type: "\"introselect\"", desc: "Selection algorithm." },
        ],
        returns: "NDArray",
        example: `np.partition([3, 4, 2, 1], 2).get(2).item(); // => 3
np.partition([3, 4, 2, 1], [1, 2]); // => [1, 2, 3, 4]`,
      },
      {
        name: "argpartition",
        sig: "np.argpartition(a, kth, [options])",
        desc: "Indices that would partition the array, as int64. Also available as `a.argpartition(kth)`.",
        returns: "NDArray (int64)",
        example: `np.argpartition([4, 3, 9], [0, 2]); // => [1, 0, 2]`,
      },
      {
        name: "lexsort",
        sig: "np.lexsort(keys, [options])",
        desc: "Stable indirect sort on several keys. The **last** key is the primary sort key. `keys` is a list of equal-shape arrays, or an array whose rows are the keys.",
        args: [
          { name: "keys", type: "ArrayLike[] | NDArray", desc: "Sort keys." },
          { name: "[options.axis]", type: "number", desc: "Axis to sort along (default `-1`)." },
        ],
        returns: "NDArray (int64)",
        example: `const first = [0, 2, 1, 1];
const last = [3, 1, 2, 1];
np.lexsort([first, last]); // => [3, 1, 2, 0]`,
      },
      {
        name: "searchsorted",
        sig: "np.searchsorted(a, v, [options])",
        desc: "Indices at which to insert `v` into the sorted 1-D array `a` so that it stays sorted. Comparisons use the common dtype of `a` and `v`. Also available as `a.searchsorted(v)`.",
        args: [
          { name: "a", type: "ArrayLike", desc: "Sorted 1-D array (or unsorted with `sorter`)." },
          { name: "v", type: "ArrayLike | number", desc: "Values to insert." },
          { name: "[options.side]", type: "\"left\" | \"right\"", desc: "`left` gives the first suitable index, `right` the last." },
          { name: "[options.sorter]", type: "ArrayLike", desc: "Indices that sort `a`, e.g. from `argsort`." },
        ],
        returns: "NDArray (int64)",
        example: `np.searchsorted([1, 2, 2, 3], [2, 0, 4]); // => [1, 0, 4]
np.searchsorted([1, 2, 2, 3], 2, { side: "right" }).item(); // => 3
np.searchsorted([3, 1, 2], [2.5], { sorter: [1, 2, 0] }); // => [2]`,
      },
      {
        name: "unique",
        sig: "np.unique(a, [options])",
        desc: "Sorted unique values. If any `return*` flag is set, it returns `{ values, indices?, inverse?, counts? }` instead. With `axis`, whole sub-arrays along that axis are compared. Results are always sorted, even with `sorted: false`.",
        args: [
          { name: "a", type: "ArrayLike", desc: "Input array (flattened unless `axis` is given)." },
          { name: "[options.returnIndex]", type: "boolean", desc: "Also return the index of the first occurrence of each value." },
          { name: "[options.returnInverse]", type: "boolean", desc: "Also return the indices that rebuild `a` from `values`." },
          { name: "[options.returnCounts]", type: "boolean", desc: "Also return how many times each value occurs." },
          { name: "[options.axis]", type: "number | null", desc: "Axis whose sub-arrays are compared." },
          { name: "[options.equalNan]", type: "boolean", desc: "Count all NaNs as one value (default `true`)." },
        ],
        returns: "NDArray | UniqueResult",
        example: `np.unique([1, 1, 2, 2, 3]); // => [1, 2, 3]
const r = np.unique([1, 3, 4, 3], { returnInverse: true, returnCounts: true });
r.inverse; // => [0, 1, 2, 1]
r.counts; // => [1, 2, 1]
np.unique([[1, 0], [0, 1], [1, 0]], { axis: 0 }); // => [[0, 1], [1, 0]]`,
      },
      {
        name: "uniqueAll",
        sig: "np.uniqueAll(x)",
        desc: "Array API helpers. `uniqueAll` returns `{ values, indices, inverseIndices, counts }`. `uniqueCounts` returns `{ values, counts }`, `uniqueInverse` returns `{ values, inverseIndices }`, and `uniqueValues` returns the values only. NaNs are not merged.",
        returns: "object | NDArray",
        example: `np.uniqueAll([2, 1, 2]).indices; // => [1, 0]
np.uniqueCounts([2, 1, 2]).counts; // => [1, 2]
np.uniqueInverse([2, 1, 2]).inverseIndices; // => [1, 0, 1]
np.uniqueValues([2, 1, 2]); // => [1, 2]`,
      },
      {
        name: "intersect1d",
        sig: "np.intersect1d(a, b, [options])",
        desc: "Sorted unique values present in both inputs. With `returnIndices: true`, it returns `{ values, indices1, indices2 }`, giving the first occurrences in each input.",
        args: [
          { name: "[options.assumeUnique]", type: "boolean", desc: "Skip removing duplicates first (inputs must already be unique)." },
          { name: "[options.returnIndices]", type: "boolean", desc: "Also return the indices into `a` and `b`." },
        ],
        returns: "NDArray | IntersectResult",
        example: `np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1]); // => [1, 3]
np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1], { returnIndices: true }).indices2; // => [1, 0]`,
      },
      {
        name: "union1d",
        sig: "np.union1d(a, b)",
        desc: "Sorted unique values found in either input.",
        returns: "NDArray",
        example: `np.union1d([-1, 0, 1], [-2, 0, 2]); // => [-2, -1, 0, 1, 2]`,
      },
      {
        name: "setdiff1d",
        sig: "np.setdiff1d(a, b, [options])",
        desc: "Sorted unique values of `a` that are not in `b`. With `assumeUnique`, `a` keeps its original order.",
        returns: "NDArray",
        example: `np.setdiff1d([1, 2, 3, 2, 4, 1], [3, 4, 5, 6]); // => [1, 2]`,
      },
      {
        name: "setxor1d",
        sig: "np.setxor1d(a, b, [options])",
        desc: "Sorted unique values found in exactly one of the inputs.",
        returns: "NDArray",
        example: `np.setxor1d([1, 2, 3, 2, 4], [2, 3, 5, 7, 5]); // => [1, 4, 5, 7]`,
      },
      {
        name: "isin",
        sig: "np.isin(element, testElements, [options])",
        desc: "Boolean array with the shape of `element`: whether each value appears in `testElements`. `kind` accepts `null`, `\"sort\"` or `\"table\"` (bool/integer inputs only). Every kind gives the same result.",
        args: [
          { name: "[options.invert]", type: "boolean", desc: "Return `true` for values that are **not** present." },
          { name: "[options.assumeUnique]", type: "boolean", desc: "Accepted for NumPy parity." },
          { name: "[options.kind]", type: "\"sort\" | \"table\" | null", desc: "Algorithm hint." },
        ],
        returns: "NDArray (bool)",
        example: `np.isin([[0, 2], [4, 6]], [1, 2, 4, 8]); // => [[false, true], [true, false]]
np.isin([1, 5], [1], { invert: true }); // => [false, true]`,
      },
      {
        name: "ediff1d",
        sig: "np.ediff1d(a, [options])",
        desc: "Differences between consecutive elements of the flattened array. Values from `toBegin`/`toEnd` are added to the start/end and must cast to the input dtype under `same_kind`.",
        args: [
          { name: "[options.toBegin]", type: "ArrayLike", desc: "Values to prepend." },
          { name: "[options.toEnd]", type: "ArrayLike", desc: "Values to append." },
        ],
        returns: "NDArray",
        example: `np.ediff1d([1, 2, 4, 7]); // => [1, 2, 3]
np.ediff1d([[1, 2], [4, 7]], { toBegin: [-9], toEnd: [100] }); // => [-9, 1, 2, 3, 100]`,
      },
    ],
  },
];
