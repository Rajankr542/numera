// P5 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.

const arrayArg = (name = "a") => ({ name, type: "ArrayLike", desc: "An `NDArray`, nested JS array or scalar." });
const ufuncOpts = {
  name: "[options]",
  type: "UfuncOptions",
  desc: "`out`, `where`, `casting`, `order`; `dtype` accepts only `\"bool\"` (the output type).",
};

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "logic",
    title: "Comparison, logic and bitwise",
    intro:
      "Element-wise comparisons and logical operations return `bool` arrays. They are ufuncs with NumPy broadcasting and the `.reduce`/`.accumulate`/`.outer`/`.at` methods.",
    entries: [
      {
        name: "equal",
        sig: "np.equal(a, b, options?) · np.notEqual · np.less · np.lessEqual · np.greater · np.greaterEqual",
        desc: "Element-wise `==`, `!=`, `<`, `<=`, `>`, `>=` with a `bool` result. Complex values compare lexicographically (real part first), NaN compares false (`notEqual` gives true). JS integers outside an integer array's range compare by exact value.",
        args: [arrayArg("a"), arrayArg("b"), ufuncOpts],
        returns: "NDArray (bool)",
        example: `np.equal([1, 2, 3], 2);          // => [false, true, false]
np.notEqual([1, NaN], [1, NaN]);  // => [false, true]
np.less([[1], [3]], [2, 4]);      // => [[true, true], [false, true]]
np.lessEqual([1, 2], 1);          // => [true, false]
np.greater(np.array([1], { dtype: "int8" }), -1000); // => [true]
np.greaterEqual([1, 2], 2);       // => [false, true]`,
      },
    ],
  },
];
