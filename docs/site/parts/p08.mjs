// P8 API reference entries (D-056, D-110). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.
const arr = (name = "a") => ({ name, type: "ArrayLike", desc: "An `NDArray`, nested JS array or scalar." });
const modeArg = { name: "[options.mode]", type: "\"raise\" | \"wrap\" | \"clip\"", desc: "Out-of-range indices: raise (default), wrap around, or clip to the valid range." };

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "indexing",
    title: "Indexing",
    entries: [
      {
        name: "a.take",
        sig: "np.take(a, indices, [axis], { mode? }) · np.take(a, indices, { axis?, mode? }) · a.take(indices, ...)",
        desc: "`take` with NumPy's `mode=`: `\"raise\"` (default; negative indices count from the end), `\"wrap\"` (modulo the axis length) or `\"clip\"` (negative indices become 0). Also available as the method `a.take`.",
        args: [arr(), { name: "indices", type: "ArrayLike", desc: "Integer indices." }, { name: "[axis]", type: "number | null", desc: "Axis; default the flattened array." }, modeArg],
        returns: "NDArray",
        example: `const a = np.array([[1, 2], [3, 4]]);
np.take(a, [-1, 5], { mode: "wrap" }); // => [4, 2]
a.take([9], 1, { mode: "clip" });      // => [[2], [4]]`,
      },
      {
        name: "takeAlongAxis",
        sig: "np.takeAlongAxis(arr, indices, axis = -1) · np.putAlongAxis(arr, indices, values, axis)",
        desc: "Pick (or write) values using an index array that has the same number of dimensions as `arr`, matched along `axis` and broadcast along the other axes, as for the output of `argmax(..., keepdims)`. `axis: null` works on the flattened array and needs 1-d indices. `putAlongAxis` writes in place and returns `undefined`.",
        args: [arr("arr"), { name: "indices", type: "ArrayLike", desc: "Integer indices with `arr.ndim` dimensions." }, { name: "values", type: "ArrayLike", desc: "`putAlongAxis` only: values, broadcast to the selection." }, { name: "axis", type: "number | null", desc: "Axis to index along." }],
        returns: "NDArray (takeAlongAxis) · void (putAlongAxis)",
        example: `const a = np.array([[10, 30, 20], [60, 40, 50]]);
np.takeAlongAxis(a, [[1], [0]], 1); // => [[30], [60]]
np.putAlongAxis(a, [[1], [0]], 0, 1);
a; // => [[10, 0, 20], [0, 40, 50]]`,
      },
      {
        name: "put",
        sig: "np.put(a, ind, v, { mode? }) · a.put(ind, v, { mode? })",
        desc: "Sets `a.flat[ind] = v` in place, repeating `v` if it is shorter than `ind`. Values are cast to `a.dtype`. Non-contiguous views are written in flat C order.",
        args: [{ name: "a", type: "NDArray", desc: "Target array (must be writeable)." }, { name: "ind", type: "ArrayLike", desc: "Flat integer indices." }, { name: "v", type: "ArrayLike", desc: "Values." }, modeArg],
        returns: "void",
        example: `const x = np.zeros(4);
np.put(x, [0, 6], [7, 8], { mode: "wrap" });
x; // => [7, 0, 8, 0]`,
      },
      {
        name: "putmask",
        sig: "np.putmask(a, mask, values) · np.place(arr, mask, vals)",
        desc: "Write values where `mask` is true, in place. `putmask` uses `values[i % n]` for flat position `i`; `place` uses the masked elements in order, `vals[k % n]` for the k-th one. An `NDArray` of values must cast safely to the target dtype; JS values are converted to it.",
        args: [{ name: "a", type: "NDArray", desc: "Target array." }, { name: "mask", type: "ArrayLike", desc: "Boolean mask with `a.size` elements." }, { name: "values", type: "ArrayLike", desc: "Values to write." }],
        returns: "void",
        example: `const x = np.array([0, 1, 2, 3, 4]);
np.putmask(x, [false, false, true, true, true], [10, 20]);
x; // => [0, 1, 10, 20, 10]
const y = np.array([0, 1, 2, 3, 4]);
np.place(y, [false, false, true, true, true], [10, 20]);
y; // => [0, 1, 10, 20, 10]`,
      },
      {
        name: "choose",
        sig: "np.choose(a, choices, { mode? }) · a.choose(choices, { mode? })",
        desc: "Builds an array from an index array and a list of choices: `out[i] = choices[a[i]][i]`, with `a` and every choice broadcast together. The result dtype is the promoted dtype of the choices (JS numbers are weak scalars). With `\"raise\"`, an out-of-range index raises `ValueError`.",
        args: [{ name: "a", type: "ArrayLike", desc: "Integer indices into `choices`." }, { name: "choices", type: "ArrayLike[] | NDArray", desc: "Choice arrays (or an array whose first axis enumerates them)." }, modeArg],
        returns: "NDArray",
        example: `np.choose([0, 1, 2], [[1, 2, 3], [4, 5, 6], [7, 8, 9]]); // => [1, 5, 9]
np.choose([0, 5], [[1, 2], [3, 4]], { mode: "clip" });      // => [1, 4]`,
      },
      {
        name: "compress",
        sig: "np.compress(condition, a, [axis]) · a.compress(condition, [axis]) · np.extract(condition, arr)",
        desc: "`compress` keeps the slices along `axis` where the 1-d `condition` is true (the flattened array if `axis` is omitted). `extract` flattens both `condition` and `arr` and returns the elements where the condition is true.",
        args: [{ name: "condition", type: "ArrayLike", desc: "Boolean selector." }, arr(), { name: "[axis]", type: "number | null", desc: "`compress` only." }],
        returns: "NDArray",
        example: `const a = np.array([[1, 2], [3, 4], [5, 6]]);
np.compress([false, true, true], a, 0); // => [[3, 4], [5, 6]]
np.extract([[true, false], [false, true], [true, false]], a); // => [1, 4, 5]`,
      },
      {
        name: "select",
        sig: "np.select(condlist, choicelist, { default? })",
        desc: "For each element, takes the value from the first choice whose condition is true, otherwise `default` (0). All arrays broadcast together. Conditions must be boolean arrays.",
        args: [{ name: "condlist", type: "ArrayLike[]", desc: "Boolean conditions." }, { name: "choicelist", type: "ArrayLike[]", desc: "One choice per condition." }, { name: "[options.default]", type: "number | ArrayLike", desc: "Value where no condition holds." }],
        returns: "NDArray",
        example: `const x = np.array([0, 1, 2, 3]);
np.select([[true, true, false, false], [false, true, true, false]], [x, np.multiply(x, 10)], { default: -1 }); // => [0, 1, 20, -1]`,
      },
      {
        name: "piecewise",
        sig: "np.piecewise(x, condlist, funclist)",
        desc: "Evaluates a function defined piece by piece. Each entry of `funclist` is a constant or a JS callback that receives the selected elements `x[cond]` (only when there are any). An extra last entry applies where no condition is true. Elements covered by no piece are 0.",
        args: [arr("x"), { name: "condlist", type: "ArrayLike | ArrayLike[]", desc: "One boolean condition or a list of them." }, { name: "funclist", type: "(number | ((x: NDArray) => ArrayLike))[]", desc: "Pieces, one per condition (plus an optional otherwise piece)." }],
        returns: "NDArray",
        example: `const x = np.array([-2, -1, 0, 1, 2]);
np.piecewise(x, [[true, true, false, false, false]], [(v) => np.negative(v), 100]); // => [2, 1, 100, 100, 100]`,
      },
      {
        name: "argwhere",
        sig: "np.argwhere(a) · np.flatnonzero(a)",
        desc: "`argwhere` returns the indices of the non-zero elements as an `int64` array of shape `(N, a.ndim)`. `flatnonzero` returns the indices in the flattened array.",
        args: [arr()],
        returns: "NDArray",
        example: `np.argwhere([[0, 3], [4, 0]]);   // => [[0, 1], [1, 0]]
np.flatnonzero([[0, 3], [4, 0]]); // => [1, 2]`,
      },
      {
        name: "countNonzero",
        sig: "np.countNonzero(a, { axis?, keepdims? })",
        desc: "Counts the non-zero elements (NaN counts as non-zero). Returns `int64`; without `axis` the result is a 0-d array, so call `.item()` for a number.",
        args: [arr(), { name: "[options.axis]", type: "number | number[] | null", desc: "Axes to count over; default all." }, { name: "[options.keepdims]", type: "boolean", desc: "Keep reduced axes with length 1." }],
        returns: "NDArray",
        example: `const a = np.array([[0, 1, 2], [3, 0, 0]]);
np.countNonzero(a).item();          // => 3
np.countNonzero(a, { axis: 0 });    // => [1, 1, 1]`,
      },
      {
        name: "ravelMultiIndex",
        sig: "np.ravelMultiIndex(multiIndex, dims, { mode?, order? }) · np.unravelIndex(indices, shape, { order? })",
        desc: "Convert between per-dimension indices and flat indices of an array with shape `dims`. `mode` (one value or one per dimension) handles out-of-range coordinates; `order` is `\"C\"` (default) or `\"F\"`. `unravelIndex` returns one `int64` array per dimension.",
        args: [{ name: "multiIndex", type: "ArrayLike[]", desc: "One integer array per dimension (broadcast together)." }, { name: "dims", type: "number[]", desc: "Array shape." }, modeArg, { name: "[options.order]", type: "\"C\" | \"F\"", desc: "Index order." }],
        returns: "NDArray · NDArray[]",
        example: `np.ravelMultiIndex([[1, 2], [3, 1]], [3, 4]); // => [7, 9]
np.unravelIndex([7, 9], [3, 4]).map((ix) => ix.toArray()); // => [[1, 2], [3, 1]]
np.ravelMultiIndex([1, 2], [3, 4], { order: "F" }).item(); // => 7`,
      },
      {
        name: "diagonal",
        sig: "np.diagonal(a, { offset?, axis1?, axis2? }) · a.diagonal(...)",
        desc: "The diagonal of the 2-d sub-arrays over `axis1` and `axis2` (default 0 and 1), as a **read-only view** like NumPy. The diagonal becomes the last axis. `offset` > 0 is above the main diagonal. A number argument is the offset.",
        args: [arr(), { name: "[options.offset]", type: "number", desc: "Diagonal offset. Default 0." }, { name: "[options.axis1], [options.axis2]", type: "number", desc: "The two axes. Default 0 and 1." }],
        returns: "NDArray",
        example: `const m = np.arange(9).reshape(3, 3);
np.diagonal(m);          // => [0, 4, 8]
m.diagonal(1);           // => [1, 5]
m.diagonal().flags.writeable; // => false`,
      },
      {
        name: "trace",
        sig: "np.trace(a, { offset?, axis1?, axis2?, dtype? }) · a.trace(...)",
        desc: "Sum along the diagonal (see `diagonal`). Uses the `sum` dtype rules: bool and signed integers give `int64`, unsigned integers `uint64`, unless `dtype` is given. Returns a 0-d array for 2-d input.",
        args: [arr(), { name: "[options.offset]", type: "number", desc: "Diagonal offset." }, { name: "[options.axis1], [options.axis2]", type: "number", desc: "The two axes." }, { name: "[options.dtype]", type: "DTypeLike", desc: "Accumulator/result dtype." }],
        returns: "NDArray",
        example: `np.trace([[1, 2], [3, 4]]).item(); // => 5
np.trace(np.arange(8).reshape(2, 2, 2), { axis1: 1, axis2: 2 }); // => [3, 11]`,
      },
      {
        name: "a.nonzero",
        sig: "a.nonzero()",
        desc: "Method form of `np.nonzero`: one `int64` index array per dimension.",
        args: [],
        returns: "NDArray[]",
        example: `np.array([0, 2, 0, 3]).nonzero()[0]; // => [1, 3]`,
      },
    ],
  },
];
