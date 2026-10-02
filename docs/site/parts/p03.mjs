// P3 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.
/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "ndarray",
    entries: [
      {
        name: "a.fill",
        sig: "a.fill(value)",
        desc: "Sets every element to `value`, cast to the array's dtype like `np.array(value, {dtype})` (floats truncate for integer dtypes; out-of-range integers and NaN raise `ValueError`).",
        args: [{ name: "value", type: "number | boolean | bigint | Complex | NDArray", desc: "A scalar or a size-1 array." }],
        returns: "void",
        example: `const a = np.zeros([2], { dtype: "int32" });
a.fill(7.9);
a.toArray(); // => [7, 7]`,
      },
      {
        name: "a.tolist",
        sig: "a.tolist()",
        desc: "Nested JS arrays of the elements, the same as `toArray()` (NumPy name).",
        returns: "NestedArray",
        example: `np.array([[1, 2], [3, 4]]).tolist(); // => [[1, 2], [3, 4]]`,
      },
      {
        name: "a.tobytes",
        sig: "a.tobytes({ order })",
        desc: "A new `Uint8Array` with the raw bytes of the elements, in `\"C\"` (default and `\"K\"`), `\"F\"` or `\"A\"` order.",
        returns: "Uint8Array",
        example: `const a = np.array([[0, 1], [2, 3]], { dtype: "uint8" });
Array.from(a.tobytes());                 // => [0, 1, 2, 3]
Array.from(a.tobytes({ order: "F" }));   // => [0, 2, 1, 3]`,
      },
      {
        name: "a.view",
        sig: "a.view([dtype])",
        desc: "A new view of the same memory, optionally reinterpreted as another dtype. With a different item size the last axis is rescaled; it must be contiguous and its byte size divisible by the new item size (otherwise `ValueError`, as NumPy).",
        returns: "NDArray",
        example: `const a = np.array([1, 2], { dtype: "int32" });
a.view("uint8").toArray(); // => [1, 0, 0, 0, 2, 0, 0, 0]
a.view().base === a;       // => true`,
      },
      {
        name: "a.byteswap",
        sig: "a.byteswap({ inplace = false })",
        desc: "Reverses the bytes of every element (each component for complex). Returns a copy, or swaps in place and returns `a`.",
        returns: "NDArray",
        example: `np.array([1, 2], { dtype: "int16" }).byteswap(); // => [256, 512]`,
      },
      {
        name: "a.setflags",
        sig: "a.setflags({ write })",
        desc: "Sets the writeable flag. `write: true` raises `ValueError` when the array that owns the memory is read-only. `align`/`uic` are not supported.",
        returns: "void",
        example: `const a = np.array([1, 2]);
a.setflags({ write: false });
a.flags.writeable; // => false`,
      },
      {
        name: "a.base",
        sig: "a.base",
        desc: "The array that owns the memory this array views, or `null` if it owns its data (NumPy `base`, same object identity).",
        returns: "NDArray | null",
        example: `const a = np.array([1, 2, 3]);
a.base;               // => null
a.get([1, 3]).base === a; // => true`,
      },
      {
        name: "a.mT",
        sig: "a.mT",
        desc: "View with the last two axes swapped (matrix transpose of a stack). Arrays with fewer than 2 dimensions raise `ValueError`.",
        returns: "NDArray",
        example: `np.zeros([5, 2, 3]).mT.shape; // => [5, 3, 2]`,
      },
      {
        name: "a.flat",
        sig: "a.flat",
        desc: "A `FlatIter`: a 1-D, C-order view of the elements. `flat.get(i)` takes an integer (returns a JS scalar), a slice tuple, an integer index array or a boolean mask (returns a new array); `flat.set(i, value)` repeats `value` cyclically over the selection; `a.flat = v` assigns to every element. It is also iterable, and has `base`, `index`, `coords`, `length` and `copy()`.",
        returns: "FlatIter",
        example: `const a = np.array([[1, 2], [3, 4]]);
a.T.flat.get(1);      // => 3
a.flat.get([1, 4]);   // => [2, 3, 4]
a.flat.set(np.array([0, 3]), [9, 8]);
a.toArray();          // => [[9, 2], [3, 8]]
[...a.T.flat];        // => [9, 3, 2, 8]`,
      },
      {
        name: "a.astype options",
        sig: "a.astype(dtype, { order, copy, casting })",
        desc: "Converted copy. `casting` (default `\"unsafe\"`) is checked with `canCast` first and a disallowed cast raises `DTypeError`; `copy: false` returns `a` itself when nothing would change; `order` defaults to `\"K\"`.",
        returns: "NDArray",
        example: `np.array([1.7, -2.2]).astype("int32"); // => [1, -2]
const f = np.array([1.5]);
f.astype("float64", { copy: false }) === f; // => true`,
      },
    ],
  },
  {
    id: "utilities",
    entries: [
      {
        name: "ndindex",
        sig: "np.ndindex(...shape)",
        desc: "Generator of every index of `shape` (dimensions as arguments or one array) in C order. `ndindex()` yields `[]` once.",
        returns: "Generator<number[]>",
        example: `[...np.ndindex(2, 2)]; // => [[0, 0], [0, 1], [1, 0], [1, 1]]`,
      },
      {
        name: "ndenumerate",
        sig: "np.ndenumerate(a)",
        desc: "Generator of `[index, value]` pairs in C order; values are JS scalars.",
        returns: "Generator<[number[], number | boolean | Complex]>",
        example: `[...np.ndenumerate(np.array([[1, 2], [3, 4]]))]; // => [[[0, 0], 1], [[0, 1], 2], [[1, 0], 3], [[1, 1], 4]]`,
      },
      {
        name: "nditer",
        sig: "np.nditer(op | ops, { flags, order = \"K\", opFlags })",
        desc: "Read-only multi-dimensional iterator (`NDIter`). Each step yields a 0-d read-only view (an array of views for several operands, which broadcast together). `\"K\"` walks memory order. Flags: `multi_index`, `c_index`, `f_index`, `zerosize_ok`; members `multiIndex`, `index`, `iterindex`, `itersize`, `shape`, `ndim`, `nop`, `operands`, `value`, `finished`, `iternext()`, `reset()`. Buffering, `external_loop` and writable operands raise `NotImplementedError`.",
        returns: "NDIter",
        example: `const out = [];
for (const x of np.nditer(np.array([[1, 2], [3, 4]]).T)) out.push(x.item());
out; // => [1, 2, 3, 4]
const it = np.nditer([np.array([[1], [2]]), np.array([10, 20])]);
[...it].map(([x, y]) => x.item() + y.item()); // => [11, 21, 12, 22]`,
      },
    ],
  },
];
