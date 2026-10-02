// P14 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.

const fileArg = { name: "file", type: "string | URL | null", desc: "Path to write (the extension is appended when missing), or `null` to return the bytes as a `Buffer`." };

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "io",
    title: "Input and output",
    intro: "NumPy `.npy`/`.npz` files and text files. Paths are read and written with Node's `fs`. Pass `null` instead of a path to get the bytes back as a `Buffer`, and pass bytes to `load` instead of a path.",
    entries: [
      {
        name: "save",
        sig: "np.save(file, arr)",
        desc: "Writes one array in NumPy's `.npy` format, byte for byte as NumPy writes it. Supported dtypes: bool, integers, floats and complex.",
        args: [fileArg, { name: "arr", type: "ArrayLike", desc: "Array to save." }],
        returns: "Buffer when `file` is null, otherwise undefined",
        example: `np.save(null, [1, 2, 3]).length; // => 152
np.save(null, [1, 2, 3]).subarray(1, 6).toString(); // => "NUMPY"`,
      },
      {
        name: "load",
        sig: "np.load(file, [options])",
        desc: "Reads a `.npy` file (returns an NDArray) or a `.npz` archive (returns an `NpzFile`). Reads big-endian data and Fortran-ordered arrays. Pickled object arrays and `mmapMode` are not supported.",
        args: [
          { name: "file", type: "string | URL | Buffer | Uint8Array | ArrayBuffer", desc: "Path, or the file contents." },
          { name: "[options.mmapMode]", type: "null", desc: "Only `null` (no memory mapping)." },
        ],
        returns: "NDArray | NpzFile",
        example: `np.load(np.save(null, [[1, 2], [3, 4]])); // => [[1, 2], [3, 4]]
np.load(np.save(null, np.ones(2, { dtype: "float32" }))).dtype.name; // => "float32"`,
      },
      {
        name: "savez",
        sig: "np.savez(file, ...arrays, [named])",
        desc: "Writes several arrays to an uncompressed `.npz` zip archive. Positional arrays are named `arr_0`, `arr_1`, ...; a trailing plain object maps names to arrays (NumPy keyword arguments). The output is byte-identical to NumPy's.",
        args: [fileArg, { name: "arrays", type: "ArrayLike[]", desc: "Arrays to store." }, { name: "[named]", type: "Record<string, ArrayLike>", desc: "Arrays stored under their keys." }],
        returns: "Buffer when `file` is null, otherwise undefined",
        example: `np.load(np.savez(null, [1, 2], { w: [3] })).files; // => ["w", "arr_0"]`,
      },
      {
        name: "savezCompressed",
        sig: "np.savezCompressed(file, ...arrays, [named])",
        desc: "Like `savez`, but the members are deflate-compressed (with Node's zlib).",
        args: [fileArg, { name: "arrays", type: "ArrayLike[]", desc: "Arrays to store." }, { name: "[named]", type: "Record<string, ArrayLike>", desc: "Arrays stored under their keys." }],
        returns: "Buffer when `file` is null, otherwise undefined",
        example: `np.load(np.savezCompressed(null, { x: np.zeros(3) })).get("x"); // => [0, 0, 0]`,
      },
      {
        name: "NpzFile",
        sig: "np.load(npz) → NpzFile",
        desc: "A loaded `.npz` archive. `files` lists the member names without `.npy`; `get(name)` decodes one member (NumPy `npz[name]`); also `has`, `keys`, `entries`, iteration over names and `close()`.",
        returns: "NpzFile",
        example: `const z = np.load(np.savez(null, { a: [1, 2], b: [true] }));
z instanceof np.NpzFile; // => true
z.get("a.npy"); // => [1, 2]
[...z]; // => ["a", "b"]
z.has("c"); // => false`,
      },
      {
        name: "loadtxt",
        sig: "np.loadtxt(fname, [options])",
        desc: "Reads a numeric table from a text file. Blank lines and comments are skipped and every row must have the same number of columns. The result is squeezed (one row or one column gives a 1-D array), then `ndmin` and `unpack` are applied. Only numeric and bool dtypes are supported (bool cells are integers).",
        args: [
          { name: "fname", type: "string | URL | Uint8Array | string[]", desc: "Path (`.gz` is decompressed), the contents as bytes, or a list of lines." },
          { name: "[options.dtype]", type: "DTypeLike", desc: "Default `float64`." },
          { name: "[options.delimiter]", type: "string | null", desc: "One character; `null` (default) splits on whitespace." },
          { name: "[options.comments]", type: "string | string[] | null", desc: "Comment prefixes (default `\"#\"`)." },
          { name: "[options.skiprows]", type: "number", desc: "Leading lines to skip." },
          { name: "[options.usecols]", type: "number | number[]", desc: "Columns to read; negative indices count from the end." },
          { name: "[options.maxRows]", type: "number", desc: "Read at most this many rows." },
          { name: "[options.unpack]", type: "boolean", desc: "Transpose the result." },
          { name: "[options.ndmin]", type: "0 | 1 | 2", desc: "Minimum number of dimensions." },
          { name: "[options.quotechar]", type: "string", desc: "Quote character for fields that contain the delimiter." },
        ],
        returns: "NDArray",
        example: `np.loadtxt(["# x y", "1 2", "3 4"]); // => [[1, 2], [3, 4]]
np.loadtxt(Buffer.from("1,2\\n3,4\\n"), { delimiter: ",", usecols: 1, dtype: "int32" }); // => [2, 4]
np.loadtxt(["1 2 3"], { ndmin: 2 }).shape; // => [1, 3]`,
      },
      {
        name: "savetxt",
        sig: "np.savetxt(fname, X, [options])",
        desc: "Writes a 1-D or 2-D array as text with Python `%`-formatting (`d i u o x X e E f F g G s`). A 1-D array is written as one column. With one format, complex values are written as `(re+imj)`. A path ending in `.gz` is gzip-compressed.",
        args: [
          { name: "fname", type: "string | URL | null", desc: "Path to write, or `null` to return the text." },
          { name: "X", type: "ArrayLike", desc: "1-D or 2-D data." },
          { name: "[options.fmt]", type: "string | string[]", desc: "One format, one per column, or a whole-row format. Default `\"%.18e\"`." },
          { name: "[options.delimiter]", type: "string", desc: "Column separator (default `\" \"`)." },
          { name: "[options.newline]", type: "string", desc: "Line terminator (default `\"\\n\"`)." },
          { name: "[options.header]", type: "string", desc: "Text written before the data, each line prefixed by `comments`." },
          { name: "[options.footer]", type: "string", desc: "Text written after the data." },
          { name: "[options.comments]", type: "string", desc: "Prefix for header and footer lines (default `\"# \"`)." },
        ],
        returns: "string when `fname` is null, otherwise undefined",
        example: `np.savetxt(null, [[1, 2], [3, 4]], { fmt: "%d", delimiter: "," }); // => "1,2\\n3,4\\n"
np.savetxt(null, [0.5, 1e16], { fmt: "%s", header: "v" }); // => "# v\\n0.5\\n1e+16\\n"`,
      },
      {
        name: "genfromtxt",
        sig: "np.genfromtxt(fname, [options])",
        desc: "Reads a table and fills missing or invalid cells. Cells that do not convert become filling values: NaN for floats, -1 for integers, false for bool, unless `loose` is false (then only `missingValues` are filled). Bool cells are `true`/`false` in any case. A numeric dtype is required (no type inference, names, converters or masks).",
        args: [
          { name: "fname", type: "string | URL | Uint8Array | string[]", desc: "Path, contents as bytes, or a list of lines." },
          { name: "[options.dtype]", type: "DTypeLike", desc: "Default `float64`." },
          { name: "[options.delimiter]", type: "string | number | number[]", desc: "Separator, a field width, or field widths. Default: whitespace." },
          { name: "[options.comments]", type: "string | null", desc: "Comment marker (default `\"#\"`)." },
          { name: "[options.skipHeader]", type: "number", desc: "Lines to skip at the start." },
          { name: "[options.skipFooter]", type: "number", desc: "Lines to drop at the end." },
          { name: "[options.missingValues]", type: "string | string[] | {col: value} | Map", desc: "Strings that mean missing (a string is split at `,`)." },
          { name: "[options.fillingValues]", type: "value | value[] | {col: value} | Map", desc: "Replacement for missing or invalid cells." },
          { name: "[options.usecols]", type: "number | number[]", desc: "Columns to read." },
          { name: "[options.invalidRaise]", type: "boolean", desc: "Raise on rows with a wrong column count (default true), or drop them with a warning." },
          { name: "[options.loose]", type: "boolean", desc: "Fill unconvertible cells (default true)." },
          { name: "[options.autostrip]", type: "boolean", desc: "Strip spaces around fields." },
          { name: "[options.maxRows]", type: "number", desc: "Read at most this many rows." },
          { name: "[options.unpack]", type: "boolean", desc: "Transpose the result." },
          { name: "[options.ndmin]", type: "0 | 1 | 2", desc: "Minimum number of dimensions." },
        ],
        returns: "NDArray",
        example: `np.isnan(np.genfromtxt(["1,2", "3,"], { delimiter: "," })); // => [[false, false], [false, true]]
np.genfromtxt(["1 x", "3 4"], { dtype: "int32", fillingValues: 0 }); // => [[1, 0], [3, 4]]
np.genfromtxt(["12345"], { delimiter: [2, 3], dtype: "int32" }); // => [12, 345]`,
      },
      {
        name: "fromregex",
        sig: "np.fromregex(file, regexp, dtype)",
        desc: "Every match of `regexp` in the text is a record, and its capture groups fill the fields of `dtype`. Structured arrays are not available, so the result is an object with one 1-D array per field.",
        args: [
          { name: "file", type: "string | URL | Uint8Array | string[]", desc: "Path, contents as bytes, or a list of lines." },
          { name: "regexp", type: "RegExp | string", desc: "Pattern; the `g` flag is added." },
          { name: "dtype", type: "[name, DTypeLike][]", desc: "Field names and dtypes, one per capture group." },
        ],
        returns: "Record<string, NDArray>",
        example: `const r = np.fromregex(["a=1 b=22"], /(\\w)=(\\d+)/, [["key", "bool"], ["n", "int32"]]);
r.n; // => [1, 22]`,
      },
      {
        name: "fromfile",
        sig: "np.fromfile(file, [options])",
        desc: "Reads raw binary data, as written by `a.tofile()`, or a text file of numbers separated by `sep`. Binary data has no header, so you must give the dtype. A trailing partial item is ignored.",
        args: [
          { name: "file", type: "string | URL | Uint8Array", desc: "Path, or the file contents." },
          { name: "[options.dtype]", type: "DTypeLike", desc: "Default `float64`." },
          { name: "[options.count]", type: "number", desc: "Items to read; -1 (default) reads all." },
          { name: "[options.sep]", type: "string", desc: "Item separator; empty (default) means binary." },
          { name: "[options.offset]", type: "number", desc: "Bytes to skip (binary mode only)." },
        ],
        returns: "NDArray",
        example: `np.fromfile(np.array([1, 2, 3]).astype("int16").tofile(null), { dtype: "int16" }); // => [1, 2, 3]
np.fromfile(Buffer.from("1, 2, 3"), { sep: ",", count: 2 }); // => [1, 2]`,
      },
      {
        name: "tofile",
        sig: "a.tofile(file, [options])",
        desc: "Method of NDArray. Writes the items in C order as raw bytes in native byte order, or with a `sep` as text where each item is formatted like Python `str` (or with `format`, a `%`-format). No shape or dtype is stored; use `save` to keep them.",
        args: [
          { name: "file", type: "string | URL | null", desc: "Path to write, or `null` to return the bytes." },
          { name: "[options.sep]", type: "string", desc: "Item separator; empty (default) writes binary." },
          { name: "[options.format]", type: "string", desc: "Format applied to each item in text mode, e.g. `\"%.2f\"`." },
        ],
        returns: "Buffer when `file` is null, otherwise undefined",
        example: `np.array([[1.5, 2], [3, 4]]).tofile(null, { sep: "," }).toString(); // => "1.5,2.0,3.0,4.0"
np.array([1, 2]).tofile(null, { sep: " ", format: "%03d" }).toString(); // => "001 002"
np.arange(3).astype("uint8").tofile(null).length; // => 3`,
      },
    ],
  },
  {
    id: "utilities",
    title: "Utilities",
    entries: [
      {
        name: "baseRepr",
        sig: "np.baseRepr(number, [base], [padding])",
        desc: "String of an integer in `base` (2 to 36, default 2), with `padding` zeros added on the left. Negative numbers get a minus sign. Accepts `number`, `bigint` or a 0-d integer array.",
        args: [
          { name: "number", type: "number | bigint | NDArray", desc: "Integer to convert." },
          { name: "[base]", type: "number", desc: "Base, 2 to 36 (default 2)." },
          { name: "[padding]", type: "number", desc: "Zeros to prepend (default 0)." },
        ],
        returns: "string",
        example: `np.baseRepr(255, 16); // => "FF"
np.baseRepr(-7, 2, 3); // => "-000111"`,
      },
      {
        name: "binaryRepr",
        sig: "np.binaryRepr(num, [options])",
        desc: "Binary string of an integer. Without `width`, negative numbers get a minus sign; with `width`, they are written in two's complement. A `width` that is too small raises `ValueError`.",
        args: [
          { name: "num", type: "number | bigint | NDArray", desc: "Integer to convert." },
          { name: "[options.width]", type: "number", desc: "Output length (zero-padded, or two's complement for negatives)." },
        ],
        returns: "string",
        example: `np.binaryRepr(5); // => "101"
np.binaryRepr(-5); // => "-101"
np.binaryRepr(-5, { width: 8 }); // => "11111011"`,
      },
    ],
  },
  {
    id: "windows",
    title: "Window functions",
    intro: "Tapering windows used in signal processing, as in NumPy. They are computed natively in float64 with NumPy's formulas.",
    entries: [
      {
        name: "bartlett",
        sig: "np.bartlett(M)",
        desc: "Triangular (Bartlett) window. Returns `M` float64 values; an empty array when `M < 1` and `[1]` when `M = 1`.",
        args: [{ name: "M", type: "number", desc: "Number of points." }],
        returns: "NDArray",
        example: `np.bartlett(5); // => [0, 0.5, 1, 0.5, 0]`,
      },
      {
        name: "blackman",
        sig: "np.blackman(M)",
        desc: "Blackman window: `0.42 + 0.5 cos(πn/(M-1)) + 0.08 cos(2πn/(M-1))`. Returns `M` float64 values; an empty array when `M < 1` and `[1]` when `M = 1`.",
        args: [{ name: "M", type: "number", desc: "Number of points." }],
        returns: "NDArray",
        example: `np.blackman(3); // => [-1.3877787807814457e-17, 1, -1.3877787807814457e-17]`,
      },
      {
        name: "hamming",
        sig: "np.hamming(M)",
        desc: "Hamming window: `0.54 + 0.46 cos(πn/(M-1))`. Returns `M` float64 values; an empty array when `M < 1` and `[1]` when `M = 1`.",
        args: [{ name: "M", type: "number", desc: "Number of points." }],
        returns: "NDArray",
        example: `np.hamming(3); // => [0.08000000000000002, 1, 0.08000000000000002]`,
      },
      {
        name: "hanning",
        sig: "np.hanning(M)",
        desc: "Hann window: `0.5 + 0.5 cos(πn/(M-1))`. Returns `M` float64 values; an empty array when `M < 1` and `[1]` when `M = 1`.",
        args: [{ name: "M", type: "number", desc: "Number of points." }],
        returns: "NDArray",
        example: `np.hanning(5); // => [0, 0.5, 1, 0.5, 0]`,
      },
      {
        name: "kaiser",
        sig: "np.kaiser(M, beta)",
        desc: "Kaiser window: `i0(beta·sqrt(1 - ((n - α)/α)²)) / i0(beta)` with `α = (M-1)/2`, using NumPy's Chebyshev approximation of the Bessel function `i0`. `beta = 0` gives a rectangular window.",
        args: [
          { name: "M", type: "number", desc: "Number of points." },
          { name: "beta", type: "number", desc: "Shape parameter." },
        ],
        returns: "NDArray",
        example: `np.kaiser(3, 0); // => [1, 1, 1]
np.kaiser(4, 5); // => [0.036710892271286676, 0.7753221044454067, 0.7753221044454067, 0.036710892271286676]`,
      },
    ],
  },
  {
    id: "polynomials",
    title: "Polynomials",
    intro: "NumPy's legacy polynomial API (`np.poly1d` and the `np.poly*` functions). Coefficients are listed highest power first.",
    entries: [
      {
        name: "poly",
        sig: "np.poly(seqOfZeros)",
        desc: "Coefficients of the monic polynomial with the given roots, or the characteristic polynomial of a square matrix. Returns `1` when there are no roots. Conjugate-pair roots give real coefficients.",
        args: [{ name: "seqOfZeros", type: "ArrayLike", desc: "Roots (1-D) or a square matrix." }],
        returns: "NDArray | number",
        example: `np.poly([1, 2]); // => [1, -3, 2]
np.poly([]); // => 1`,
      },
      {
        name: "roots",
        sig: "np.roots(p)",
        desc: "Roots of a polynomial, computed as the eigenvalues of its companion matrix. Real when every root is real, otherwise complex.",
        args: [{ name: "p", type: "ArrayLike | poly1d", desc: "Coefficients, highest power first." }],
        returns: "NDArray",
        example: `np.roots([1, -3, 2]); // => [2, 1]
np.roots([1, 0, 0]); // => [0, 0]`,
      },
      {
        name: "polyval",
        sig: "np.polyval(p, x)",
        desc: "Evaluates a polynomial at `x` with Horner's scheme. With a `poly1d` as `x`, the result is the composed polynomial.",
        args: [{ name: "p", type: "ArrayLike | poly1d", desc: "Coefficients, highest power first." }, { name: "x", type: "ArrayLike | poly1d", desc: "Points to evaluate at." }],
        returns: "NDArray | poly1d",
        example: `np.polyval([1, 2, 3], [0, 1, 2]); // => [3, 6, 11]`,
      },
      {
        name: "polyadd",
        sig: "np.polyadd(a1, a2)",
        desc: "Sum of two polynomials. The result is a `poly1d` if either input is one.",
        args: [{ name: "a1", type: "ArrayLike | poly1d", desc: "First polynomial." }, { name: "a2", type: "ArrayLike | poly1d", desc: "Second polynomial." }],
        returns: "NDArray | poly1d",
        example: `np.polyadd([1, 2], [3, 4, 5]); // => [3, 5, 7]`,
      },
      {
        name: "polysub",
        sig: "np.polysub(a1, a2)",
        desc: "Difference `a1 - a2` of two polynomials.",
        args: [{ name: "a1", type: "ArrayLike | poly1d", desc: "Minuend." }, { name: "a2", type: "ArrayLike | poly1d", desc: "Subtrahend." }],
        returns: "NDArray | poly1d",
        example: `np.polysub([2, 2], [1]); // => [2, 1]`,
      },
      {
        name: "polymul",
        sig: "np.polymul(a1, a2)",
        desc: "Product of two polynomials (a full convolution, computed natively).",
        args: [{ name: "a1", type: "ArrayLike | poly1d", desc: "First factor." }, { name: "a2", type: "ArrayLike | poly1d", desc: "Second factor." }],
        returns: "NDArray | poly1d",
        example: `np.polymul([1, 2], [3, 4]); // => [3, 10, 8]`,
      },
      {
        name: "polydiv",
        sig: "np.polydiv(u, v)",
        desc: "Polynomial long division. Returns `[quotient, remainder]` in floating point; leading near-zero remainder terms are dropped.",
        args: [{ name: "u", type: "ArrayLike | poly1d", desc: "Dividend." }, { name: "v", type: "ArrayLike | poly1d", desc: "Divisor." }],
        returns: "[NDArray, NDArray] | [poly1d, poly1d]",
        example: `const [q, r] = np.polydiv([1, -3, 2], [1, -1]);
q; // => [1, -2]
r; // => [0]`,
      },
      {
        name: "polyder",
        sig: "np.polyder(p, [m])",
        desc: "The `m`-th derivative (default 1).",
        args: [{ name: "p", type: "ArrayLike | poly1d", desc: "Coefficients, highest power first." }, { name: "[m]", type: "number", desc: "Order of the derivative." }],
        returns: "NDArray | poly1d",
        example: `np.polyder([1, 2, 3]); // => [2, 2]`,
      },
      {
        name: "polyint",
        sig: "np.polyint(p, [m], [k])",
        desc: "The `m`-th antiderivative (default 1). `k` gives the integration constants: one value for all, or one per integration.",
        args: [{ name: "p", type: "ArrayLike | poly1d", desc: "Coefficients, highest power first." }, { name: "[m]", type: "number", desc: "Order of the integral." }, { name: "[k]", type: "number | number[]", desc: "Integration constants (default 0)." }],
        returns: "NDArray | poly1d",
        example: `np.polyint([3, 2], 1, 5); // => [1.5, 2, 5]`,
      },
      {
        name: "polyfit",
        sig: "np.polyfit(x, y, deg, [options])",
        desc: "Least-squares polynomial fit of degree `deg`, coefficients highest power first. `y` may be 2-D (one fit per column). With `full`, returns `[c, residuals, rank, singularValues, rcond]`; with `cov`, returns `[c, covariance]`. A rank-deficient fit emits a `RankWarning`.",
        args: [{ name: "x", type: "ArrayLike", desc: "Sample points (1-D)." }, { name: "y", type: "ArrayLike", desc: "Values, 1-D or 2-D." }, { name: "deg", type: "number", desc: "Degree of the fit." }, { name: "[options.rcond]", type: "number", desc: "Singular value cutoff (default `len(x) * eps`)." }, { name: "[options.full]", type: "boolean", desc: "Return diagnostic values too." }, { name: "[options.w]", type: "ArrayLike", desc: "Weights." }, { name: "[options.cov]", type: "boolean | \"unscaled\"", desc: "Return the covariance matrix too." }],
        returns: "NDArray | Array",
        example: `np.allclose(np.polyfit([0, 1, 2], [1, 3, 5], 1), [2, 1]); // => true`,
      },
      {
        name: "poly1d",
        sig: "new np.poly1d(c, [options])",
        desc: "A polynomial object. Leading zeros are trimmed. Python operators are methods: `call(x)`, `add`, `sub`, `mul`, `div` (a scalar, or a polynomial giving `[q, r]`), `pow`, `neg`, `equals`. `get(k)` and `set(k, v)` read and write the coefficient of x**k. Also available: `coeffs`, `order`, `length`, `roots`, `variable`, `integ(m, k)`, `deriv(m)`, iteration over the coefficients, and `toString()`, which matches NumPy's `str(p)`.",
        args: [{ name: "c", type: "ArrayLike | poly1d", desc: "Coefficients, highest power first (or roots with `r: true`)." }, { name: "[options.r]", type: "boolean", desc: "Treat `c` as roots." }, { name: "[options.variable]", type: "string", desc: "Variable name for printing (default `\"x\"`)." }],
        returns: "poly1d",
        example: `const p = new np.poly1d([1, -2, 3]);
p.call(2); // => 3
p.mul(p).coeffs; // => [1, -4, 10, -12, 9]
p.deriv().coeffs; // => [2, -2]
String(p); // => "   2\\n1 x - 2 x + 3"`,
      },
    ],
  },
];
