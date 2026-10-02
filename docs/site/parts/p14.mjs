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
    ],
  },
];
