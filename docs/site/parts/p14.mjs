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
    ],
  },
];
