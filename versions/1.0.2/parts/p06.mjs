// P6 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.
/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "shape",
    entries: [
      {
        name: "concatenate",
        sig: "np.concatenate(arrays, [axis | {axis, dtype, casting, out}])",
        desc: "Joins arrays along an existing axis (default 0). `axis: null` flattens the inputs first. `concat` is the same function (array API name). `casting` defaults to `\"same_kind\"`; `out` and `dtype` cannot be combined.",
        args: [
          { name: "arrays", type: "ArrayLike[]", desc: "Arrays with the same shape except along `axis`." },
          { name: "[axis]", type: "number | null | object", desc: "Join axis, or an options object." },
        ],
        returns: "NDArray (or `out`)",
        example: `np.concatenate([[1, 2], [3]]);                       // => [1, 2, 3]
np.concatenate([[[1], [2]], [[3], [4]]], 1);          // => [[1, 3], [2, 4]]
np.concat([[[1, 2]], [3]], { axis: null });           // => [1, 2, 3]
np.concatenate([[1.7], [2]], { dtype: "int8", casting: "unsafe" }); // => [1, 2]`,
      },
      {
        name: "stack",
        sig: "np.stack(arrays, [axis | {axis, dtype, casting, out}])",
        desc: "Joins same-shaped arrays along a new axis.",
        args: [{ name: "arrays", type: "ArrayLike[]", desc: "Arrays of identical shape." }],
        returns: "NDArray",
        example: `np.stack([[1, 2], [3, 4]], 1); // => [[1, 3], [2, 4]]`,
      },
      {
        name: "vstack",
        sig: "np.vstack(arrays, [{dtype, casting}])",
        desc: "Stacks inputs (made at least 2-d) along axis 0.",
        returns: "NDArray",
        example: `np.vstack([[1, 2], [3, 4]]); // => [[1, 2], [3, 4]]`,
      },
      {
        name: "hstack",
        sig: "np.hstack(arrays, [{dtype, casting}])",
        desc: "Joins along axis 1, or axis 0 for 1-d inputs.",
        returns: "NDArray",
        example: `np.hstack([[1, 2], [3]]); // => [1, 2, 3]`,
      },
      {
        name: "dstack",
        sig: "np.dstack(arrays)",
        desc: "Joins along axis 2 after making each input at least 3-d.",
        returns: "NDArray",
        example: `np.dstack([[1, 2], [3, 4]]); // => [[[1, 3], [2, 4]]]`,
      },
      {
        name: "columnStack",
        sig: "np.columnStack(arrays)",
        desc: "Stacks 1-d arrays as columns of a 2-d array (2-d inputs are joined as they are).",
        returns: "NDArray",
        example: `np.columnStack([[1, 2], [3, 4]]); // => [[1, 3], [2, 4]]`,
      },
      {
        name: "block",
        sig: "np.block(arrays)",
        desc: "Assembles an array from nested lists of blocks: the innermost lists join along the last axis, the next level along the one before, and so on.",
        args: [{ name: "arrays", type: "nested (NDArray | number)[]", desc: "Nested lists of arrays or scalars." }],
        returns: "NDArray",
        example: `np.block([[np.ones([1, 2]), np.zeros([1, 1])], [np.zeros([1, 3])]]); // => [[1, 1, 0], [0, 0, 0]]`,
      },
      {
        name: "unstack",
        sig: "np.unstack(x, [axis])",
        desc: "Splits an array into a list of views along `axis` (default 0), removing that axis.",
        returns: "NDArray[]",
        example: `np.unstack(np.array([[1, 2], [3, 4]]), 1).map((v) => v.toArray()); // => [[1, 3], [2, 4]]`,
      },
      {
        name: "split",
        sig: "np.split(a, sectionsOrIndices, [axis])",
        desc: "Splits into equal sections (must divide the axis) or at the given indices. Returns views. `arraySplit` allows unequal sections; `hsplit`, `vsplit` and `dsplit` split along axis 1 (0 for 1-d), 0 and 2.",
        args: [
          { name: "a", type: "ArrayLike", desc: "Array to split." },
          { name: "sectionsOrIndices", type: "number | number[]", desc: "Number of sections, or split points." },
        ],
        returns: "NDArray[]",
        example: `np.split(np.arange(6), 3).map((v) => v.toArray());       // => [[0, 1], [2, 3], [4, 5]]
np.split(np.arange(5), [2]).map((v) => v.toArray());      // => [[0, 1], [2, 3, 4]]
np.arraySplit(np.arange(5), 2).map((v) => v.toArray());   // => [[0, 1, 2], [3, 4]]
np.hsplit(np.arange(4).reshape(2, 2), 2)[1].toArray();    // => [[1], [3]]
np.vsplit(np.arange(4).reshape(2, 2), 2)[1].toArray();    // => [[2, 3]]
np.dsplit(np.zeros([1, 1, 4]), 2)[0].shape;               // => [1, 1, 2]`,
      },
      {
        name: "atleast1d",
        sig: "np.atleast1d(...arys)",
        desc: "Views with at least 1 (`atleast1d`), 2 (`atleast2d`, `(N)` → `(1, N)`) or 3 (`atleast3d`, `(N)` → `(1, N, 1)`, `(M, N)` → `(M, N, 1)`) dimensions. One argument returns one array, several return a list.",
        returns: "NDArray | NDArray[]",
        example: `np.atleast1d(5).shape;          // => [1]
np.atleast2d([1, 2]).shape;     // => [1, 2]
np.atleast3d([1, 2]).shape;     // => [1, 2, 1]
np.atleast1d(1, [2, 3]).length; // => 2`,
      },
      {
        name: "tile",
        sig: "np.tile(a, reps)",
        desc: "Repeats the whole array `reps` times along each axis; `reps` shorter than `a.ndim` is padded with 1s on the left, longer prepends new axes. Always returns a copy.",
        args: [{ name: "reps", type: "number | number[]", desc: "Repetitions per axis." }],
        returns: "NDArray",
        example: `np.tile([1, 2], 2);         // => [1, 2, 1, 2]
np.tile([1, 2], [2, 1]);    // => [[1, 2], [1, 2]]`,
      },
      {
        name: "repeat",
        sig: "np.repeat(a, repeats, [axis]) / a.repeat(repeats, [axis])",
        desc: "Repeats each element `repeats` times (an integer, or one count per element along `axis`). Without `axis` the input is flattened first.",
        args: [
          { name: "repeats", type: "number | number[]", desc: "Non-negative repetition counts." },
          { name: "[axis]", type: "number | null", desc: "Axis to repeat along." },
        ],
        returns: "NDArray",
        example: `np.repeat([[1, 2], [3, 4]], 2);          // => [1, 1, 2, 2, 3, 3, 4, 4]
np.repeat([[1, 2], [3, 4]], [1, 2], 0);  // => [[1, 2], [3, 4], [3, 4]]
np.array([1, 2]).repeat(2);              // => [1, 1, 2, 2]`,
      },
      {
        name: "pad",
        sig: "np.pad(a, padWidth, [mode], [{constantValues, endValues, statLength, reflectType}])",
        desc: "Pads an array. Modes: `constant` (default), `edge`, `linear_ramp`, `maximum`, `mean`, `median`, `minimum`, `reflect`, `symmetric`, `wrap`, `empty`, or a JS function `(vector, [before, after], axis, options)` that fills each 1-d lane in place. `padWidth` is `n`, `[before, after]`, one pair per axis, or `{axis: width}`. Statistics on integer arrays are rounded to the nearest integer.",
        args: [
          { name: "padWidth", type: "number | number[] | number[][] | object", desc: "Number of values padded before/after each axis." },
          { name: "[mode]", type: "string | function", desc: "Padding mode." },
        ],
        returns: "NDArray",
        example: `np.pad([1, 2, 3], [1, 2]);                                   // => [0, 1, 2, 3, 0, 0]
np.pad([1, 2, 3], 1, "constant", { constantValues: [7, 8] });  // => [7, 1, 2, 3, 8]
np.pad([1, 2, 3], 2, "edge");                                // => [1, 1, 1, 2, 3, 3, 3]
np.pad([1, 2, 3], 2, "reflect");                             // => [3, 2, 1, 2, 3, 2, 1]
np.pad([1, 2, 3], 2, "symmetric");                           // => [2, 1, 1, 2, 3, 3, 2]
np.pad([1, 2, 3], 2, "wrap");                                // => [2, 3, 1, 2, 3, 1, 2]
np.pad([1, 2, 3], 1, "mean");                                // => [2, 1, 2, 3, 2]
np.pad([0, 4], [2, 0], "linear_ramp", { endValues: 4 });     // => [4, 2, 0, 4]`,
      },
      {
        name: "append",
        sig: "np.append(arr, values, [axis])",
        desc: "Appends `values` to `arr` (both flattened when `axis` is omitted); a new array.",
        returns: "NDArray",
        example: `np.append([1, 2], [[3, 4]]);       // => [1, 2, 3, 4]
np.append([[1, 2]], [[3, 4]], 0);   // => [[1, 2], [3, 4]]`,
      },
      {
        name: "insert",
        sig: "np.insert(arr, obj, values, [axis])",
        desc: "Inserts `values` before index/indices `obj` (an integer, list, boolean mask or slice object `{start, stop, step}`). Without `axis` the array is flattened. Values are cast to `arr.dtype`.",
        returns: "NDArray",
        example: `np.insert([1, 2, 3], 1, 9);                       // => [1, 9, 2, 3]
np.insert([[1, 1], [2, 2]], 1, 5, 1);              // => [[1, 5, 1], [2, 5, 2]]
np.insert(np.arange(4), { start: 1, stop: 3 }, 0); // => [0, 0, 1, 0, 2, 3]`,
      },
      {
        name: "delete",
        sig: "np.delete(arr, obj, [axis])",
        desc: "Removes the entries at `obj` (integer, list, boolean mask or slice object) along `axis`, flattening first when `axis` is omitted. Named export `delete` (implemented as `del`).",
        returns: "NDArray",
        example: `np.delete([1, 2, 3, 4], [0, -1]);                 // => [2, 3]
np.delete([[1, 2], [3, 4]], 0, 1);                 // => [[2], [4]]
np.delete(np.arange(6), { start: 0, step: 2 });    // => [1, 3, 5]`,
      },
      {
        name: "resize",
        sig: "np.resize(a, newShape) / a.resize(newShape, [{refcheck}])",
        desc: "`np.resize` returns a new array filled by repeating `a`'s data. The method `a.resize` changes `a` itself (in place, returns `undefined`): data is truncated or zero-filled in memory order; `a` must own contiguous data, and while other arrays (views) still share it a `ValueError` is raised unless `refcheck: false`.",
        returns: "NDArray / undefined",
        example: `np.resize([1, 2, 3], [2, 4]);                     // => [[1, 2, 3, 1], [2, 3, 1, 2]]
const x = np.array([1, 2, 3]);
x.resize([5]);
x.toArray();                                       // => [1, 2, 3, 0, 0]`,
      },
      {
        name: "trimZeros",
        sig: "np.trimZeros(filt, [trim], [axis])",
        desc: "Trims leading (`\"f\"`) and/or trailing (`\"b\"`) zeros (default `\"fb\"`). N-d input is trimmed to the bounding box of the nonzero values on the selected axes. Returns a view.",
        returns: "NDArray",
        example: `np.trimZeros([0, 0, 1, 0, 2, 0]);        // => [1, 0, 2]
np.trimZeros([0, 0, 1, 0, 2, 0], "b");   // => [0, 0, 1, 0, 2]
np.trimZeros([[0, 0], [0, 3]]);          // => [[3]]`,
      },
      {
        name: "flip",
        sig: "np.flip(m, [axis])",
        desc: "Reverses element order along `axis` (an integer or list; all axes when omitted). `fliplr` reverses axis 1 and `flipud` axis 0. Views with negative strides.",
        returns: "NDArray",
        example: `np.flip([[1, 2], [3, 4]]);       // => [[4, 3], [2, 1]]
np.flip([[1, 2], [3, 4]], 1);    // => [[2, 1], [4, 3]]
np.fliplr([[1, 2], [3, 4]]);     // => [[2, 1], [4, 3]]
np.flipud([[1, 2], [3, 4]]);     // => [[3, 4], [1, 2]]`,
      },
      {
        name: "roll",
        sig: "np.roll(a, shift, [axis])",
        desc: "Shifts elements cyclically. Without `axis` the flattened array is rolled and the shape restored; `shift` and `axis` may be lists (broadcast against each other, shifts on the same axis add up). Returns a copy.",
        returns: "NDArray",
        example: `np.roll([1, 2, 3, 4], 1);                    // => [4, 1, 2, 3]
np.roll([[1, 2], [3, 4]], 1);                // => [[4, 1], [2, 3]]
np.roll([[1, 2], [3, 4]], [1, 1], [0, 1]);   // => [[4, 3], [2, 1]]`,
      },
      {
        name: "rot90",
        sig: "np.rot90(m, [k], [axes])",
        desc: "Rotates by 90° `k` times (default 1) in the plane of `axes` (default `[0, 1]`), from the first axis towards the second. A view.",
        returns: "NDArray",
        example: `np.rot90([[1, 2], [3, 4]]);       // => [[2, 4], [1, 3]]
np.rot90([[1, 2], [3, 4]], 2);    // => [[4, 3], [2, 1]]`,
      },
      {
        name: "rollaxis",
        sig: "np.rollaxis(a, axis, [start])",
        desc: "Moves `axis` so that it lies before position `start` (default 0). Prefer `moveAxis`. `permuteDims(a, axes)` permutes axes (array API transpose) and `matrixTranspose(x)` swaps the last two axes. All return views.",
        returns: "NDArray",
        example: `np.rollaxis(np.zeros([3, 4, 5]), 2).shape;          // => [5, 3, 4]
np.permuteDims(np.zeros([2, 3, 4]), [2, 0, 1]).shape; // => [4, 2, 3]
np.matrixTranspose(np.zeros([2, 3, 4])).shape;       // => [2, 4, 3]`,
      },
      {
        name: "copyto",
        sig: "np.copyto(dst, src, [{casting, where}])",
        desc: "Copies `src` (broadcast to `dst`'s shape) into `dst` in place. `casting` defaults to `\"same_kind\"`; JS scalars follow NEP 50 (an integer must fit an integer `dst`). `where` is a boolean mask; only elements where it is true are written.",
        returns: "undefined",
        example: `const d = np.zeros([2, 3]);
np.copyto(d, [1, 2, 3], { where: [true, false, true] });
d.toArray();                                  // => [[1, 0, 3], [1, 0, 3]]`,
      },
      {
        name: "broadcastArrays",
        sig: "np.broadcastArrays(...arrays)",
        desc: "Broadcasts the inputs against each other and returns views of the common shape (inputs already of that shape are returned as is).",
        returns: "NDArray[]",
        example: `np.broadcastArrays([1, 2, 3], [[1], [2]]).map((x) => x.shape); // => [[2, 3], [2, 3]]`,
      },
      {
        name: "asanyarray",
        sig: "np.asanyarray(a, [{dtype}]) / np.asarrayChkfinite(a, [{dtype}])",
        desc: "`asanyarray` is `asarray` (numera has no array subclasses). `asarrayChkfinite` also raises `ValueError` if the result contains NaN or an infinity.",
        returns: "NDArray",
        example: `np.asanyarray([1, 2]).dtype.name;   // => "int64"
np.asarrayChkfinite([1, 2]).size;    // => 2`,
      },
      {
        name: "require",
        sig: "np.require(a, [dtype], [requirements])",
        desc: "Returns `a` as an array of `dtype` that satisfies the requirement flags, copying only if needed: `C`/`C_CONTIGUOUS`/`CONTIGUOUS`, `F`/`F_CONTIGUOUS`/`FORTRAN`, `A`/`ALIGNED`, `W`/`WRITEABLE`, `O`/`OWNDATA`, `E`/`ENSUREARRAY`, as a string of letters or a list.",
        returns: "NDArray",
        example: `np.require([[1, 2], [3, 4]], "float32", ["F", "W"]).flags.fContiguous; // => true`,
      },
      {
        name: "shape",
        sig: "np.shape(a) / np.size(a, [axis]) / np.ndim(a) / np.isfortran(a)",
        desc: "Metadata of any array-like: shape, number of elements (or the length of `axis`), number of dimensions. `isfortran` is true for arrays that are F- but not C-contiguous.",
        returns: "number[] / number / boolean",
        example: `np.shape([[1, 2, 3]]);                          // => [1, 3]
np.size([[1, 2, 3]], 1);                        // => 3
np.ndim(5);                                     // => 0
np.isfortran(np.zeros([2, 3], { order: "F" })); // => true`,
      },
      {
        name: "applyAlongAxis",
        sig: "np.applyAlongAxis(func1d, axis, arr, ...args)",
        desc: "Calls the JS function `func1d(lane, ...args)` on every 1-d lane of `arr` along `axis`. Results must all have the first result's shape; they replace that axis (scalars remove it).",
        returns: "NDArray",
        example: `np.applyAlongAxis((v) => v.sum(), 1, [[1, 2], [3, 4]]);        // => [3, 7]
np.applyAlongAxis((v) => np.flip(v), 1, [[1, 2], [3, 4]]);      // => [[2, 1], [4, 3]]`,
      },
      {
        name: "applyOverAxes",
        sig: "np.applyOverAxes(func, a, axes)",
        desc: "Applies `func(val, axis)` for each axis in turn; a result with one dimension fewer gets the axis back as length 1 (like `keepdims`).",
        returns: "NDArray",
        example: `np.applyOverAxes((x, ax) => x.sum({ axis: ax }), np.arange(6).reshape(2, 3), [0, 1]); // => [[15]]`,
      },
    ],
  },
];
