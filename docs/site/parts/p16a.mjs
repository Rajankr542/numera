// P16A API reference entries (D-191).
// All examples are executed by packages/numera/test/docs_site.test.ts;
// every `// => <json>` line must evaluate exactly to that value.

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "utilities",
    title: "Utilities",
    entries: [
      {
        name: "pi",
        sig: "np.pi",
        desc: "Mathematical constant π ≈ 3.14159…",
        returns: "number",
        example: `np.pi; // => 3.141592653589793`,
      },
      {
        name: "e",
        sig: "np.e",
        desc: "Euler's number, the base of natural logarithms, ≈ 2.71828…",
        returns: "number",
        example: `np.e; // => 2.718281828459045`,
      },
      {
        name: "inf",
        sig: "np.inf",
        desc: "IEEE 754 positive infinity. Aliases: `np.PINF`, `np.Inf`, `np.Infinity`.",
        returns: "number",
        example: `np.inf > 1e308; // => true`,
      },
      {
        name: "nan",
        sig: "np.nan",
        desc: "IEEE 754 not-a-number. Alias: `np.NaN`.",
        returns: "number",
        example: `np.nan !== np.nan; // => true`,
      },
      {
        name: "euler_gamma",
        sig: "np.euler_gamma",
        desc: "Euler–Mascheroni constant γ ≈ 0.5772156649…",
        returns: "number",
        example: `Math.abs(np.euler_gamma - 0.5772156649015329) < 1e-15; // => true`,
      },
      {
        name: "True_",
        sig: "np.True_",
        desc: "Python `True` (JS `true`). Provided for NumPy API parity.",
        returns: "boolean",
        example: `np.True_ === true; // => true`,
      },
      {
        name: "False_",
        sig: "np.False_",
        desc: "Python `False` (JS `false`). Provided for NumPy API parity.",
        returns: "boolean",
        example: `np.False_ === false; // => true`,
      },
      {
        name: "PINF",
        sig: "np.PINF",
        desc: "Positive infinity (alias of `np.inf`).",
        returns: "number",
        example: `np.PINF === Infinity; // => true`,
      },
      {
        name: "NINF",
        sig: "np.NINF",
        desc: "Negative infinity.",
        returns: "number",
        example: `np.NINF === -Infinity; // => true`,
      },
      {
        name: "PZERO",
        sig: "np.PZERO",
        desc: "Positive zero.",
        returns: "number",
        example: `1 / np.PZERO === Infinity; // => true`,
      },
      {
        name: "NZERO",
        sig: "np.NZERO",
        desc: "Negative zero.",
        returns: "number",
        example: `1 / np.NZERO === -Infinity; // => true`,
      },
      {
        name: "vectorize",
        sig: "np.vectorize(fn, [options])",
        desc: "Wraps a JS callback so it is applied element-wise over its inputs, which are broadcast to a common shape. Returns a callable with a `pyfunc` property. `otypes[0]` forces the output dtype; otherwise it is inferred from the first call. `signature` must be `null` (scalar-in / scalar-out only).",
        args: [
          { name: "fn", type: "(...args: unknown[]) => unknown", desc: "The scalar function to apply element-wise." },
          { name: "[options.otypes]", type: "DTypeLike[]", desc: "Output dtype(s); only the first element is used." },
          { name: "[options.signature]", type: "null", desc: "Must be `null` or omitted (generalised ufunc signatures are not supported)." },
        ],
        returns: "VectorizedFn",
        example: `const vf = np.vectorize((x, y) => x + y);
vf([1, 2], [10, 20]).toArray(); // => [11, 22]
vf.pyfunc(3, 4); // => 7`,
      },
      {
        name: "shares_memory",
        sig: "np.shares_memory(a, b, [options])",
        desc: "Returns `true` if arrays `a` and `b` share any underlying memory. Delegates to native buffer identity and byte-range overlap. `maxWork` is accepted for NumPy API compatibility but ignored.",
        args: [
          { name: "a", type: "NDArray", desc: "First array." },
          { name: "b", type: "NDArray", desc: "Second array." },
          { name: "[options.maxWork]", type: "number", desc: "Accepted but not used." },
        ],
        returns: "boolean",
        example: `const a = np.array([1, 2, 3]);
const b = a.slice([[0, 2]]);
np.shares_memory(a, b); // => true
np.shares_memory(a, np.array([1, 2, 3])); // => false`,
      },
      {
        name: "cumsum",
        sig: "np.cumsum(a, [axis], [options])",
        desc: "Cumulative sum of array elements along the given axis. If `axis` is omitted or `null`, the array is flattened first.",
        args: [
          { name: "a", type: "ArrayLike", desc: "Input array." },
          { name: "[axis]", type: "number | null", desc: "Axis to accumulate along; `null` flattens first." },
          { name: "[options.dtype]", type: "DTypeLike", desc: "Accumulator dtype." },
        ],
        returns: "NDArray",
        example: `np.cumsum([1, 2, 3]).toArray(); // => [1, 3, 6]
np.cumsum([[1, 2], [3, 4]], 0).toArray(); // => [[1, 2], [4, 6]]`,
      },
      {
        name: "cumprod",
        sig: "np.cumprod(a, [axis], [options])",
        desc: "Cumulative product of array elements along the given axis. If `axis` is omitted or `null`, the array is flattened first.",
        args: [
          { name: "a", type: "ArrayLike", desc: "Input array." },
          { name: "[axis]", type: "number | null", desc: "Axis to accumulate along; `null` flattens first." },
          { name: "[options.dtype]", type: "DTypeLike", desc: "Accumulator dtype." },
        ],
        returns: "NDArray",
        example: `np.cumprod([1, 2, 3, 4]).toArray(); // => [1, 2, 6, 24]
np.cumprod([[1, 2], [3, 4]], 1).toArray(); // => [[1, 2], [3, 12]]`,
      },
    ],
  },
  {
    id: "dtype",
    title: "Data types",
    entries: [
      {
        name: "int_",
        sig: "np.int_",
        desc: "Default Python integer dtype: `int64` on 64-bit platforms.",
        returns: "DType",
        example: `np.int_.name; // => "int64"`,
      },
      {
        name: "intc",
        sig: "np.intc",
        desc: "C `int` dtype: `int32`.",
        returns: "DType",
        example: `np.intc.name; // => "int32"`,
      },
      {
        name: "intp",
        sig: "np.intp",
        desc: "Pointer-sized signed integer dtype: `int64` on 64-bit platforms.",
        returns: "DType",
        example: `np.intp.name; // => "int64"`,
      },
      {
        name: "short",
        sig: "np.short",
        desc: "C `short` dtype: `int16`.",
        returns: "DType",
        example: `np.short.name; // => "int16"`,
      },
      {
        name: "byte",
        sig: "np.byte",
        desc: "C `signed char` dtype: `int8`.",
        returns: "DType",
        example: `np.byte.name; // => "int8"`,
      },
      {
        name: "ubyte",
        sig: "np.ubyte",
        desc: "C `unsigned char` dtype: `uint8`.",
        returns: "DType",
        example: `np.ubyte.name; // => "uint8"`,
      },
      {
        name: "ushort",
        sig: "np.ushort",
        desc: "C `unsigned short` dtype: `uint16`.",
        returns: "DType",
        example: `np.ushort.name; // => "uint16"`,
      },
      {
        name: "uint",
        sig: "np.uint",
        desc: "Platform unsigned integer dtype: `uint64` on 64-bit platforms.",
        returns: "DType",
        example: `np.uint.name; // => "uint64"`,
      },
      {
        name: "uintc",
        sig: "np.uintc",
        desc: "C `unsigned int` dtype: `uint32`.",
        returns: "DType",
        example: `np.uintc.name; // => "uint32"`,
      },
      {
        name: "uintp",
        sig: "np.uintp",
        desc: "Pointer-sized unsigned integer dtype: `uint64` on 64-bit platforms.",
        returns: "DType",
        example: `np.uintp.name; // => "uint64"`,
      },
      {
        name: "ulong",
        sig: "np.ulong",
        desc: "C `unsigned long` dtype: `uint64` on 64-bit platforms.",
        returns: "DType",
        example: `np.ulong.name; // => "uint64"`,
      },
      {
        name: "double",
        sig: "np.double",
        desc: "Double-precision float dtype: `float64`.",
        returns: "DType",
        example: `np.double.name; // => "float64"`,
      },
      {
        name: "single",
        sig: "np.single",
        desc: "Single-precision float dtype: `float32`.",
        returns: "DType",
        example: `np.single.name; // => "float32"`,
      },
      {
        name: "half",
        sig: "np.half",
        desc: "Half-precision float dtype: `float16`.",
        returns: "DType",
        example: `np.half.name; // => "float16"`,
      },
      {
        name: "cdouble",
        sig: "np.cdouble",
        desc: "Double-precision complex dtype: `complex128`.",
        returns: "DType",
        example: `np.cdouble.name; // => "complex128"`,
      },
      {
        name: "csingle",
        sig: "np.csingle",
        desc: "Single-precision complex dtype: `complex64`.",
        returns: "DType",
        example: `np.csingle.name; // => "complex64"`,
      },
      {
        name: "longdouble",
        sig: "np.longdouble",
        desc: "Long double dtype: aliased to `float64` (no 80-bit float in JS).",
        returns: "DType",
        example: `np.longdouble.name; // => "float64"`,
      },
      {
        name: "clongdouble",
        sig: "np.clongdouble",
        desc: "Long double complex dtype: aliased to `complex128` (no 80-bit complex in JS).",
        returns: "DType",
        example: `np.clongdouble.name; // => "complex128"`,
      },
    ],
  },
];
