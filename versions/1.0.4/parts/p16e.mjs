// P16E API reference entries (D-230).
/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "rec",
    title: "Record arrays (np.rec)",
    intro: "Lightweight structured record arrays as named NDArray columns. `np.recarray` is the class constructor; `np.rec` is the module namespace.",
    entries: [
      {
        name: "recarray",
        sig: "new np.rec.recarray(shape, {names, formats})",
        desc: "Create a zero-filled record array with named fields. `np.recarray` is also exported at the top level.",
        args: [
          { name: "shape", type: "number | number[]", desc: "Number of records, or shape array." },
          { name: "[options.names]", type: "string | string[]", desc: "Field names." },
          { name: "[options.formats]", type: "string | string[]", desc: "Format strings (e.g. `\"float64\"`, `\"i4\"`)." },
        ],
        returns: "recarray",
        example: `const r = new np.rec.recarray(3, { names: ["x", "y"], formats: ["float64", "int32"] });
r.field("x").shape; // => [3]
r.field("y").dtype.name; // => "int32"`,
      },
      {
        name: "fromarrays",
        sig: "np.rec.fromarrays(arrayList, {names, formats})",
        desc: "Build a record array from a list of column arrays.",
        args: [
          { name: "arrayList", type: "ArrayLike[]", desc: "One array per field." },
          { name: "[options.names]", type: "string | string[]", desc: "Field names." },
          { name: "[options.formats]", type: "string | string[]", desc: "Optional format strings." },
        ],
        returns: "recarray",
        example: `const r = np.rec.fromarrays([np.array([1, 2, 3]), np.array([4.0, 5.0, 6.0])], { names: ["x", "y"] });
r.field("x").toArray(); // => [1, 2, 3]
r.field("y").toArray(); // => [4, 5, 6]`,
      },
      {
        name: "fromrecords",
        sig: "np.rec.fromrecords(recList, {names, formats})",
        desc: "Build a record array from a list of row tuples.",
        args: [
          { name: "recList", type: "unknown[][]", desc: "Array of rows." },
          { name: "[options.names]", type: "string | string[]", desc: "Field names." },
        ],
        returns: "recarray",
        example: `const r = np.rec.fromrecords([[1, 2], [3, 4]], { names: ["a", "b"] });
r.field("a").toArray(); // => [1, 3]
r.field("b").toArray(); // => [2, 4]`,
      },
      {
        name: "rec.array",
        sig: "np.rec.array(obj, {names, formats})",
        desc: "Flexible record array constructor. Delegates to `fromarrays`, `fromrecords`, or clones an existing `recarray`.",
        args: [
          { name: "obj", type: "recarray | ArrayLike[] | unknown[][]", desc: "Source data." },
          { name: "[options]", type: "RecArrayOptions", desc: "Names, formats." },
        ],
        returns: "recarray",
        example: `const r = np.rec.array([[1, 2.0], [3, 4.0]], { names: "a,b" });
r.field("a").toArray(); // => [1, 3]`,
      },
      {
        name: "find_duplicate",
        sig: "np.rec.find_duplicate(list)",
        desc: "Return values that appear more than once in `list`.",
        args: [{ name: "list", type: "unknown[]", desc: "Input list." }],
        returns: "unknown[]",
        example: `np.rec.find_duplicate([1, 2, 1, 3, 2]); // => [1, 2]`,
      },
    ],
  },
  {
    id: "utilities",
    title: "Utilities",
    entries: [
      {
        name: "sharesMemory",
        sig: "np.sharesMemory(a, b, [options])",
        desc: "Return `true` if `a` and `b` share the same underlying memory buffer (NumPy `np.shares_memory`). Unlike `mayShareMemory`, this is always an exact check.",
        args: [
          { name: "a", type: "NDArray", desc: "First array." },
          { name: "b", type: "NDArray", desc: "Second array." },
          { name: "[options.maxWork]", type: "number", desc: "Ignored (always exact)." },
        ],
        returns: "boolean",
        example: `const a = np.arange(6);
const b = a.reshape([2, 3]);
np.sharesMemory(a, b); // => true
np.sharesMemory(a, np.array([1, 2])); // => false`,
      },
    ],
  },
];
