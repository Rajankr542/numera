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
  {
    id: "dtype",
    entries: [
      {
        name: "finfo",
        sig: "np.finfo(dtype | array)",
        desc: "Machine limits of a float dtype (complex: of its component). Fields as NumPy in camelCase: `bits, eps, epsneg, max, min, tiny, smallestNormal, smallestSubnormal, resolution, precision, iexp, nexp, nmant, machep, negep, minexp, maxexp, dtype`. Values are JS numbers holding the dtype's value. Non-float dtypes raise `ValueError`.",
        returns: "FInfo",
        example: `np.finfo("float16").max;  // => 65504
np.finfo("float32").nmant; // => 23
np.finfo("float64").eps;   // => 2.220446049250313e-16`,
      },
      {
        name: "iinfo",
        sig: "np.iinfo(dtype | array)",
        desc: "Limits of an integer dtype: `bits, min, max, dtype, kind`; `min`/`max` are numbers (lossy beyond 2^53) and `minExact`/`maxExact` exact bigints. Non-integer dtypes raise `ValueError`.",
        returns: "IInfo",
        example: `np.iinfo("int8").min;   // => -128
np.iinfo("uint16").max; // => 65535
np.iinfo("uint64").maxExact === 18446744073709551615n; // => true`,
      },
      {
        name: "resultType",
        sig: "np.resultType(...arraysAndDtypes)",
        desc: "The dtype NumPy's type promotion gives. DTypes, dtype names and arrays are promoted with `promoteTypes`; JS scalars are weak (NEP 50): they can raise the kind (int → float → complex) but not the size.",
        returns: "DType",
        example: `np.resultType("int8", 300).name;          // => "int8"
np.resultType("int8", 1.5).name;          // => "float64"
np.resultType("float32", np.complex(0, 1)).name; // => "complex64"
np.resultType("int8", "uint8").name;      // => "int16"`,
      },
      {
        name: "minScalarType",
        sig: "np.minScalarType(x)",
        desc: "The smallest dtype that holds the value of a JS scalar or a 0-d array (unsigned integers preferred for non-negative values; floats sized by NumPy's thresholds). Arrays with `ndim > 0` return their dtype. JS numbers count as integers only when they are safe integers; use a bigint beyond 2^53.",
        returns: "DType",
        example: `np.minScalarType(10).name;   // => "uint8"
np.minScalarType(-129).name; // => "int16"
np.minScalarType(3.1).name;  // => "float16"
np.minScalarType(1e50).name; // => "float64"`,
      },
      {
        name: "issubdtype",
        sig: "np.issubdtype(a, b)",
        desc: "Whether `a` is `b` (concrete dtype) or descends from the abstract dtype `b` (`np.generic`, `np.number`, `np.integer`, `np.signedinteger`, `np.unsignedinteger`, `np.inexact`, `np.floating`, `np.complexfloating`).",
        returns: "boolean",
        example: `np.issubdtype("float32", np.floating); // => true
np.issubdtype("bool", np.integer);     // => false
np.issubdtype("int8", "int16");        // => false`,
      },
      {
        name: "isdtype",
        sig: "np.isdtype(dtype, kind)",
        desc: "Array API dtype test. `kind` is a DType, one of `\"bool\"`, `\"signed integer\"`, `\"unsigned integer\"`, `\"integral\"`, `\"real floating\"`, `\"complex floating\"`, `\"numeric\"`, or an array of these. `dtype` must be a DType object.",
        returns: "boolean",
        example: `np.isdtype(np.float64, "real floating");          // => true
np.isdtype(np.int8, ["bool", "unsigned integer"]); // => false
np.isdtype(np.bool, "numeric");                    // => false`,
      },
      {
        name: "commonType",
        sig: "np.commonType(...arrays)",
        desc: "The float or complex dtype all inputs convert to (integers count as float64; bool raises `DTypeError`). NumPy returns a scalar type; this returns a DType.",
        returns: "DType",
        example: `np.commonType(np.arange(3)).name; // => "float64"
np.commonType(np.zeros([1], { dtype: "float32" }), np.zeros([1], { dtype: "complex64" })).name; // => "complex64"`,
      },
      {
        name: "mintypecode",
        sig: "np.mintypecode(typechars, typeset = \"GDFgdf\", default = \"d\")",
        desc: "NumPy's smallest-size type character from `typeset` among `typechars` (a string of type characters or a list of characters, DTypes or arrays). `F` with `d` gives `D`.",
        returns: "string",
        example: `np.mintypecode(["d", "f"]); // => "d"
np.mintypecode("dF");       // => "D"
np.mintypecode("i");        // => "d"`,
      },
      {
        name: "generic",
        sig: "np.generic",
        desc: "Abstract dtype (NumPy's scalar type hierarchy, parent: `null`), for `issubdtype`. Not usable as an array dtype.",
        returns: "AbstractDType",
        example: `np.issubdtype("float32", np.generic); // => true`,
      },
      {
        name: "number",
        sig: "np.number",
        desc: "Abstract dtype (NumPy's scalar type hierarchy, parent: `generic`), for `issubdtype`. Not usable as an array dtype.",
        returns: "AbstractDType",
        example: `np.issubdtype("float32", np.number); // => true`,
      },
      {
        name: "integer",
        sig: "np.integer",
        desc: "Abstract dtype (NumPy's scalar type hierarchy, parent: `number`), for `issubdtype`. Not usable as an array dtype.",
        returns: "AbstractDType",
        example: `np.issubdtype("float32", np.integer); // => false`,
      },
      {
        name: "signedinteger",
        sig: "np.signedinteger",
        desc: "Abstract dtype (NumPy's scalar type hierarchy, parent: `integer`), for `issubdtype`. Not usable as an array dtype.",
        returns: "AbstractDType",
        example: `np.issubdtype("float32", np.signedinteger); // => false`,
      },
      {
        name: "unsignedinteger",
        sig: "np.unsignedinteger",
        desc: "Abstract dtype (NumPy's scalar type hierarchy, parent: `integer`), for `issubdtype`. Not usable as an array dtype.",
        returns: "AbstractDType",
        example: `np.issubdtype("float32", np.unsignedinteger); // => false`,
      },
      {
        name: "inexact",
        sig: "np.inexact",
        desc: "Abstract dtype (NumPy's scalar type hierarchy, parent: `number`), for `issubdtype`. Not usable as an array dtype.",
        returns: "AbstractDType",
        example: `np.issubdtype("float32", np.inexact); // => true`,
      },
      {
        name: "floating",
        sig: "np.floating",
        desc: "Abstract dtype (NumPy's scalar type hierarchy, parent: `inexact`), for `issubdtype`. Not usable as an array dtype.",
        returns: "AbstractDType",
        example: `np.issubdtype("float32", np.floating); // => true`,
      },
      {
        name: "complexfloating",
        sig: "np.complexfloating",
        desc: "Abstract dtype (NumPy's scalar type hierarchy, parent: `inexact`), for `issubdtype`. Not usable as an array dtype.",
        returns: "AbstractDType",
        example: `np.issubdtype("float32", np.complexfloating); // => false`,
      },
    ],
  },
];
