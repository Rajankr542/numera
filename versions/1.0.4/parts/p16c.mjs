// P16C API reference entries — np.strings and np.char (D-210–D-213).
/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "strings",
    title: "String operations (np.strings / np.char)",
    intro: "Element-wise string operations on `StringArray` objects. `np.strings` is the modern namespace; `np.char` is the legacy alias (NumPy deprecated). Both operate on `StringArray` — a thin JS wrapper over a flat `string[]` with shape metadata.",
    entries: [
      {
        name: "strings.array",
        sig: "np.char.array(data)",
        desc: "Construct a `StringArray` from a string, 1-D or 2-D nested JS string array. (Exposed under `np.char.array`.)",
        args: [
          { name: "data", type: "string | string[] | string[][]", desc: "String data." },
        ],
        returns: "StringArray",
        example: `const a = np.char.array(["hello", "world"]);
a.shape;          // => [2]
a.toArray();      // => ["hello", "world"]
a.dtype;          // => "str_"`,
      },
      {
        name: "strings.upper",
        sig: "np.strings.upper(a)",
        desc: "Convert each element to uppercase.",
        args: [{ name: "a", type: "StringArray | string[]", desc: "Input string array." }],
        returns: "StringArray",
        example: `np.strings.upper(["hello", "world"]).toArray(); // => ["HELLO", "WORLD"]`,
      },
      {
        name: "strings.lower",
        sig: "np.strings.lower(a)",
        desc: "Convert each element to lowercase.",
        args: [{ name: "a", type: "StringArray | string[]", desc: "Input string array." }],
        returns: "StringArray",
        example: `np.strings.lower(["HELLO", "WORLD"]).toArray(); // => ["hello", "world"]`,
      },
      {
        name: "strings.capitalize",
        sig: "np.strings.capitalize(a)",
        desc: "Capitalize the first character of each element; lowercase the rest.",
        args: [{ name: "a", type: "StringArray | string[]", desc: "Input string array." }],
        returns: "StringArray",
        example: `np.strings.capitalize(["hello WORLD"]).toArray(); // => ["Hello world"]`,
      },
      {
        name: "strings.add",
        sig: "np.strings.add(a, b)",
        desc: "Element-wise string concatenation.",
        args: [
          { name: "a", type: "StringArray | string[]", desc: "Left operand." },
          { name: "b", type: "StringArray | string[]", desc: "Right operand." },
        ],
        returns: "StringArray",
        example: `np.strings.add(["hello"], [" world"]).toArray(); // => ["hello world"]`,
      },
      {
        name: "strings.multiply",
        sig: "np.strings.multiply(a, i)",
        desc: "Repeat each element `i` times.",
        args: [
          { name: "a", type: "StringArray | string[]", desc: "Input string array." },
          { name: "i", type: "number", desc: "Repetition count." },
        ],
        returns: "StringArray",
        example: `np.strings.multiply(["ab"], 3).toArray(); // => ["ababab"]`,
      },
      {
        name: "strings.equal",
        sig: "np.strings.equal(a, b)",
        desc: "Element-wise equality comparison.",
        args: [
          { name: "a", type: "StringArray | string[]", desc: "Left operand." },
          { name: "b", type: "StringArray | string[]", desc: "Right operand." },
        ],
        returns: "NDArray (bool)",
        example: `np.strings.equal(["abc", "xyz"], ["abc", "abc"]).toArray(); // => [true, false]`,
      },
      {
        name: "strings.str_len",
        sig: "np.strings.str_len(a)",
        desc: "Return the length of each element.",
        args: [{ name: "a", type: "StringArray | string[]", desc: "Input string array." }],
        returns: "NDArray (int64)",
        example: `np.strings.str_len(["hello", "hi"]).toArray(); // => [5, 2]`,
      },
    ],
  },
];
