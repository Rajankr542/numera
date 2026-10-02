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
    ],
  },
];
