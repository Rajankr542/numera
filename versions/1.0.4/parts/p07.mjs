// P7 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.

const arr = (name = "a") => ({ name, type: "ArrayLike", desc: "An `NDArray` or nested JS array." });
const kArg = { name: "[k]", type: "number", desc: "Diagonal offset: `0` main, `> 0` above, `< 0` below. Default `0`." };
const dtypeOpt = (def) => ({ name: "[options.dtype]", type: "DTypeLike", desc: `Element type. Default ${def}.` });

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "creation",
    title: "Array creation",
    entries: [
      {
        name: "logspace",
        sig: "np.logspace(start, stop, [num], [options])",
        desc: "`num` numbers spaced evenly on a log scale: `base ** linspace(start, stop, num)`. `start`/`stop` are scalars (real or complex); array bounds and `axis` are not supported.",
        args: [
          { name: "start", type: "number | Complex", desc: "Exponent of the first value." },
          { name: "stop", type: "number | Complex", desc: "Exponent of the last value." },
          { name: "[num]", type: "number", desc: "Number of samples. Default `50`." },
          { name: "[options.endpoint]", type: "boolean", desc: "Include `base ** stop`. Default `true`." },
          { name: "[options.base]", type: "number", desc: "Base of the log space. Default `10`." },
          dtypeOpt("`float64` (complex inputs give `complex128`)"),
        ],
        returns: "NDArray",
        example: `np.logspace(0, 2, 3);                    // => [1, 10, 100]
np.logspace(0, 3, 4, { base: 2 });        // => [1, 2, 4, 8]
np.logspace(0, 2.5, 3, { dtype: "int64" }); // => [1, 17, 316]`,
      },
      {
        name: "geomspace",
        sig: "np.geomspace(start, stop, [num], [options])",
        desc: "`num` numbers forming a geometric progression from `start` to `stop` (endpoints exact). Zero bounds raise `ValueError`; a sign change gives `NaN` interior values like NumPy.",
        args: [
          { name: "start", type: "number | Complex", desc: "First value." },
          { name: "stop", type: "number | Complex", desc: "Last value." },
          { name: "[num]", type: "number", desc: "Number of samples. Default `50`." },
          { name: "[options.endpoint]", type: "boolean", desc: "Include `stop`. Default `true`." },
          dtypeOpt("`float64`"),
        ],
        returns: "NDArray",
        example: `np.geomspace(1, 1000, 4);  // => [1, 10, 100, 1000]
np.geomspace(-1000, -1, 4); // => [-1000, -100, -10, -1]`,
      },
      {
        name: "tri",
        sig: "np.tri(n, [m], [options])",
        desc: "An `n × m` array with ones at and below the `k`-th diagonal and zeros elsewhere.",
        args: [
          { name: "n", type: "number", desc: "Rows." },
          { name: "[m]", type: "number | null", desc: "Columns. Default `n`." },
          { name: "[options.k]", type: "number", desc: "Diagonal offset. Default `0`." },
          dtypeOpt("`float64`"),
        ],
        returns: "NDArray",
        example: `np.tri(3, 4, { k: 1 }); // => [[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]`,
      },
      {
        name: "tril",
        sig: "np.tril(m, [k])",
        desc: "Copy of `m` with the elements above the `k`-th diagonal zeroed. Applies to the last two axes of stacked input.",
        args: [arr("m"), kArg],
        returns: "NDArray",
        example: `np.tril([[1, 2, 3], [4, 5, 6], [7, 8, 9]], -1); // => [[0, 0, 0], [4, 0, 0], [7, 8, 0]]`,
      },
      {
        name: "triu",
        sig: "np.triu(m, [k])",
        desc: "Copy of `m` with the elements below the `k`-th diagonal zeroed.",
        args: [arr("m"), kArg],
        returns: "NDArray",
        example: `np.triu([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 1); // => [[0, 2, 3], [0, 0, 6], [0, 0, 0]]`,
      },
      {
        name: "diag",
        sig: "np.diag(v, [k])",
        desc: "For a 2-D input, returns the `k`-th diagonal as a read-only view. For a 1-D input, builds a 2-D array with `v` on the `k`-th diagonal.",
        args: [arr("v"), kArg],
        returns: "NDArray",
        example: `np.diag([[1, 2, 3], [4, 5, 6], [7, 8, 9]]); // => [1, 5, 9]
np.diag([1, 2], 1); // => [[0, 1, 0], [0, 0, 2], [0, 0, 0]]`,
      },
      {
        name: "diagflat",
        sig: "np.diagflat(v, [k])",
        desc: "Flattens `v` and builds a 2-D array with it on the `k`-th diagonal.",
        args: [arr("v"), kArg],
        returns: "NDArray",
        example: `np.diagflat([[1, 2], [3, 4]]).shape; // => [4, 4]
np.diagflat([1, 2], -1); // => [[0, 0, 0], [1, 0, 0], [0, 2, 0]]`,
      },
      {
        name: "vander",
        sig: "np.vander(x, [n], [options])",
        desc: "Vandermonde matrix: column `j` is `x ** (n - 1 - j)` (or `x ** j` when `increasing`).",
        args: [
          arr("x"),
          { name: "[n]", type: "number | null", desc: "Number of columns. Default `x.length`." },
          { name: "[options.increasing]", type: "boolean", desc: "Increasing powers left to right. Default `false`." },
        ],
        returns: "NDArray",
        example: `np.vander([1, 2, 3], 3); // => [[1, 1, 1], [4, 2, 1], [9, 3, 1]]
np.vander([1, 2, 3], null, { increasing: true }); // => [[1, 1, 1], [1, 2, 4], [1, 3, 9]]`,
      },
      {
        name: "fromfunction",
        sig: "np.fromfunction(fn, shape, [options])",
        desc: "Calls `fn` once with one coordinate array per axis (like `np.indices`) and returns its result.",
        args: [
          { name: "fn", type: "(...coords: NDArray[]) => R", desc: "Function of the coordinate arrays." },
          { name: "shape", type: "number[]", desc: "Grid shape." },
          dtypeOpt("`float64` (dtype of the coordinate arrays)"),
        ],
        returns: "R",
        example: `np.fromfunction((i, j) => np.add(i, j), [2, 3]); // => [[0, 1, 2], [1, 2, 3]]`,
      },
      {
        name: "fromiter",
        sig: "np.fromiter(iterable, dtype, [count])",
        desc: "Builds a 1-D array from any JS iterable. `count` reads at most that many items and raises `ValueError` if the iterator is shorter.",
        args: [
          { name: "iterable", type: "Iterable<number | boolean | bigint | Complex>", desc: "Source values." },
          { name: "dtype", type: "DTypeLike", desc: "Element type (required)." },
          { name: "[count]", type: "number", desc: "Items to read; `-1` (default) reads all." },
        ],
        returns: "NDArray",
        example: `np.fromiter(new Set([1, 2, 3]), "float32"); // => [1, 2, 3]
np.fromiter([5, 6, 7, 8], "int64", 2); // => [5, 6]`,
      },
      {
        name: "frombuffer",
        sig: "np.frombuffer(buffer, [options])",
        desc: "Interprets raw bytes (`ArrayBuffer`, typed array or `DataView`) as a 1-D array. The data is copied; byte order is native.",
        args: [
          { name: "buffer", type: "ArrayBufferLike | ArrayBufferView", desc: "Source bytes." },
          dtypeOpt("`float64`"),
          { name: "[options.count]", type: "number", desc: "Items to read; `-1` (default) reads all." },
          { name: "[options.offset]", type: "number", desc: "Start offset in bytes. Default `0`." },
        ],
        returns: "NDArray",
        example: `np.frombuffer(new Int16Array([1, 2, 3]), { dtype: "int16", offset: 2 }); // => [2, 3]
np.frombuffer(new Uint8Array([1, 2, 3]).buffer, { dtype: "uint8", count: 2 }); // => [1, 2]`,
      },
      {
        name: "fromstring",
        sig: "np.fromstring(text, [options])",
        desc: "Parses separated numbers from text (NumPy's text mode). An empty `sep` (binary mode) raises `ValueError`; use `frombuffer`. Whitespace in `sep` matches any run of whitespace.",
        args: [
          { name: "text", type: "string", desc: "Text to parse." },
          dtypeOpt("`float64`"),
          { name: "[options.count]", type: "number", desc: "Items to read; `-1` (default) reads all." },
          { name: "[options.sep]", type: "string", desc: "Separator (required, non-empty)." },
        ],
        returns: "NDArray",
        example: `np.fromstring("1 2 3", { sep: " ", dtype: "int64" }); // => [1, 2, 3]
np.fromstring("1.5, 2, 3", { sep: "," }); // => [1.5, 2, 3]`,
      },
      {
        name: "astype",
        sig: "np.astype(x, dtype, [options])",
        desc: "Function form of `a.astype` (array API). `x` must be an `NDArray`.",
        args: [
          { name: "x", type: "NDArray", desc: "Input array." },
          { name: "dtype", type: "DTypeLike", desc: "Target dtype." },
          { name: "[options.copy]", type: "boolean", desc: "Default `true`; `false` returns `x` when no cast is needed." },
        ],
        returns: "NDArray",
        example: `np.astype(np.array([1.7, -2.2]), "int32"); // => [1, -2]`,
      },
    ],
  },
  {
    id: "grids",
    title: "Grids and index helpers",
    intro: "JS has no `obj[...]` overloading, so NumPy's index-trick objects (`mgrid`, `ogrid`, `r_`, `c_`, `s_`, `index_exp`) are functions here. Slices are `[start, stop, step]` tuples or strings like `\"1:4\"`.",
    entries: [
      {
        name: "indices",
        sig: "np.indices(dimensions, [options])",
        desc: "An `(N, ...dimensions)` array whose `i`-th sub-array holds the index along axis `i`. With `sparse: true`, returns one broadcastable array per axis.",
        args: [
          { name: "dimensions", type: "number[]", desc: "Grid shape." },
          dtypeOpt("`int64`"),
          { name: "[options.sparse]", type: "boolean", desc: "Return a list of sparse arrays. Default `false`." },
        ],
        returns: "NDArray | NDArray[]",
        example: `np.indices([2, 3]); // => [[[0, 0, 0], [1, 1, 1]], [[0, 1, 2], [0, 1, 2]]]
np.indices([2, 3], { sparse: true }).map((a) => a.shape); // => [[2, 1], [1, 3]]`,
      },
      {
        name: "meshgrid",
        sig: "np.meshgrid(...xi, [options])",
        desc: "Coordinate matrices from coordinate vectors. `\"xy\"` indexing swaps the first two output axes (Cartesian); `\"ij\"` keeps matrix order. `copy: false` returns read-only broadcast views.",
        args: [
          { name: "...xi", type: "ArrayLike[]", desc: "1-D coordinate vectors (flattened otherwise)." },
          { name: "[options.indexing]", type: "\"xy\" | \"ij\"", desc: "Default `\"xy\"`." },
          { name: "[options.sparse]", type: "boolean", desc: "Return broadcastable arrays instead of full grids. Default `false`." },
          { name: "[options.copy]", type: "boolean", desc: "Default `true`." },
        ],
        returns: "NDArray[]",
        example: `const [X, Y] = np.meshgrid([1, 2, 3], [4, 5]);
X; // => [[1, 2, 3], [1, 2, 3]]
Y; // => [[4, 4, 4], [5, 5, 5]]
np.meshgrid([1, 2, 3], [4, 5], { indexing: "ij", sparse: true }).map((a) => a.shape); // => [[3, 1], [1, 2]]`,
      },
      {
        name: "mgrid",
        sig: "np.mgrid(...slices)",
        desc: "Dense multi-dimensional grid; `np.mgrid([0, 2], [0, 3])` is NumPy's `np.mgrid[0:2, 0:3]`. A complex step `np.complex(0, n)` means `n` points with the stop included. One slice returns a 1-D array.",
        args: [{ name: "...slices", type: "[start, stop, step?][]", desc: "One slice per axis; `start` may be `null`." }],
        returns: "NDArray",
        example: `np.mgrid([0, 2], [0, 3]); // => [[[0, 0, 0], [1, 1, 1]], [[0, 1, 2], [0, 1, 2]]]
np.mgrid([-1, 1, np.complex(0, 5)]); // => [-1, -0.5, 0, 0.5, 1]`,
      },
      {
        name: "ogrid",
        sig: "np.ogrid(...slices)",
        desc: "Open (sparse) version of `mgrid`: one broadcastable array per slice.",
        args: [{ name: "...slices", type: "[start, stop, step?][]", desc: "One slice per axis." }],
        returns: "NDArray[] (NDArray for a single slice)",
        example: `const [i, j] = np.ogrid([0, 2], [0, 3]);
i; // => [[0], [1]]
j; // => [[0, 1, 2]]`,
      },
      {
        name: "ix_",
        sig: "np.ix_(...seqs)",
        desc: "Open mesh from 1-D sequences for cross-product indexing. Boolean sequences become their nonzero indices.",
        args: [{ name: "...seqs", type: "ArrayLike[]", desc: "1-D integer or boolean sequences." }],
        returns: "NDArray[]",
        example: `const [r, c] = np.ix_([0, 1], [2, 4]);
r; // => [[0], [1]]
c; // => [[2, 4]]
np.ix_([true, false, true])[0]; // => [0, 2]`,
      },
      {
        name: "r_",
        sig: "np.r_(...items)",
        desc: "Concatenates along the first axis. Items are arrays, scalars or slice strings (`\"1:4\"`, `\"0:1:5j\"`). A leading directive string such as `\"0,2\"` (axis, ndmin) or `\"1,2,0\"` (axis, ndmin, trans1d) works as in NumPy; the matrix directives `\"r\"`/`\"c\"` are not supported.",
        args: [{ name: "...items", type: "(ArrayLike | string)[]", desc: "Pieces to join." }],
        returns: "NDArray",
        example: `np.r_("1:4", 0, 4, [7, 8]); // => [1, 2, 3, 0, 4, 7, 8]
np.r_("0:1:5j"); // => [0, 0.25, 0.5, 0.75, 1]
np.r_("0,2", [1, 2], [3, 4]); // => [[1, 2], [3, 4]]`,
      },
      {
        name: "c_",
        sig: "np.c_(...items)",
        desc: "Like `r_`, but stacks 1-D inputs as columns (directive `\"-1,2,0\"`).",
        args: [{ name: "...items", type: "(ArrayLike | string)[]", desc: "Pieces to join." }],
        returns: "NDArray",
        example: `np.c_([1, 2, 3], [4, 5, 6]); // => [[1, 4], [2, 5], [3, 6]]`,
      },
      {
        name: "s_",
        sig: "np.s_(...specs)",
        desc: "Builds index specs for `a.get(...)`: slice strings become `[start, stop, step]` tuples and `\"...\"` stays the ellipsis. One spec returns the spec itself.",
        args: [{ name: "...specs", type: "(IndexSpec | string)[]", desc: "Indices or slice strings." }],
        returns: "IndexSpec | IndexSpec[]",
        example: `np.s_("1:3"); // => [1, 3, null]
np.s_(0, "::2"); // => [0, [null, null, 2]]
np.arange(6).get(np.s_("1:5:2")); // => [1, 3]`,
      },
      {
        name: "indexExp",
        sig: "np.indexExp(...specs)",
        desc: "NumPy `index_exp`: like `s_`, but always returns an array of specs.",
        args: [{ name: "...specs", type: "(IndexSpec | string)[]", desc: "Indices or slice strings." }],
        returns: "IndexSpec[]",
        example: `np.indexExp(1); // => [1]
np.indexExp("2:"); // => [[2, null, null]]`,
      },
    ],
  },
  {
    id: "tri-indices",
    title: "Triangle and diagonal indices",
    entries: [
      {
        name: "trilIndices",
        sig: "np.trilIndices(n, [k], [m])",
        desc: "Row and column indices of the lower triangle of an `n × m` array.",
        args: [
          { name: "n", type: "number", desc: "Rows." },
          kArg,
          { name: "[m]", type: "number | null", desc: "Columns. Default `n`." },
        ],
        returns: "[NDArray, NDArray]",
        example: `np.trilIndices(3).map((a) => a.toArray()); // => [[0, 1, 1, 2, 2, 2], [0, 0, 1, 0, 1, 2]]`,
      },
      {
        name: "triuIndices",
        sig: "np.triuIndices(n, [k], [m])",
        desc: "Row and column indices of the upper triangle of an `n × m` array.",
        args: [
          { name: "n", type: "number", desc: "Rows." },
          kArg,
          { name: "[m]", type: "number | null", desc: "Columns. Default `n`." },
        ],
        returns: "[NDArray, NDArray]",
        example: `np.triuIndices(3, 1).map((a) => a.toArray()); // => [[0, 0, 1], [1, 2, 2]]`,
      },
      {
        name: "trilIndicesFrom",
        sig: "np.trilIndicesFrom(arr, [k])",
        desc: "`trilIndices` for the shape of a 2-D array.",
        args: [{ name: "arr", type: "NDArray", desc: "2-D array." }, kArg],
        returns: "[NDArray, NDArray]",
        example: `np.trilIndicesFrom(np.zeros([2, 3])).map((a) => a.toArray()); // => [[0, 1, 1], [0, 0, 1]]`,
      },
      {
        name: "triuIndicesFrom",
        sig: "np.triuIndicesFrom(arr, [k])",
        desc: "`triuIndices` for the shape of a 2-D array.",
        args: [{ name: "arr", type: "NDArray", desc: "2-D array." }, kArg],
        returns: "[NDArray, NDArray]",
        example: `np.triuIndicesFrom(np.zeros([2, 3])).map((a) => a.toArray()); // => [[0, 0, 0, 1, 1], [0, 1, 2, 1, 2]]`,
      },
      {
        name: "diagIndices",
        sig: "np.diagIndices(n, [ndim])",
        desc: "Indices of the main diagonal of an `ndim`-dimensional array with all sides `n`.",
        args: [
          { name: "n", type: "number", desc: "Side length." },
          { name: "[ndim]", type: "number", desc: "Dimensions. Default `2`." },
        ],
        returns: "NDArray[]",
        example: `np.diagIndices(3).map((a) => a.toArray()); // => [[0, 1, 2], [0, 1, 2]]`,
      },
      {
        name: "diagIndicesFrom",
        sig: "np.diagIndicesFrom(arr)",
        desc: "`diagIndices` for an array whose sides are all equal (at least 2-D).",
        args: [{ name: "arr", type: "NDArray", desc: "Input array." }],
        returns: "NDArray[]",
        example: `np.diagIndicesFrom(np.zeros([2, 2, 2])).map((a) => a.toArray()); // => [[0, 1], [0, 1], [0, 1]]`,
      },
      {
        name: "maskIndices",
        sig: "np.maskIndices(n, maskFunc, [k])",
        desc: "Indices where `maskFunc(np.ones([n, n]), k)` is nonzero, e.g. `np.triu`.",
        args: [
          { name: "n", type: "number", desc: "Side length." },
          { name: "maskFunc", type: "(m: NDArray, k: number) => NDArray", desc: "Mask function such as `np.tril`." },
          kArg,
        ],
        returns: "NDArray[]",
        example: `np.maskIndices(3, np.triu, 1).map((a) => a.toArray()); // => [[0, 0, 1], [1, 2, 2]]`,
      },
      {
        name: "fillDiagonal",
        sig: "np.fillDiagonal(a, val, [options])",
        desc: "Fills the main diagonal of `a` in place (values repeat cyclically). For tall 2-D arrays, `wrap: true` restarts the diagonal after every `ncols + 1` rows, like NumPy.",
        args: [
          { name: "a", type: "NDArray", desc: "Writeable array, at least 2-D (all sides equal when > 2-D)." },
          { name: "val", type: "ArrayLike", desc: "Scalar or values to write." },
          { name: "[options.wrap]", type: "boolean", desc: "Default `false`." },
        ],
        returns: "void",
        example: `const a = np.zeros([3, 3]);
np.fillDiagonal(a, [1, 2]);
a; // => [[1, 0, 0], [0, 2, 0], [0, 0, 1]]
const b = np.zeros([5, 2]);
np.fillDiagonal(b, 7, { wrap: true });
b; // => [[7, 0], [0, 7], [0, 0], [7, 0], [0, 7]]`,
      },
    ],
  },
];
