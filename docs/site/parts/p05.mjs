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
      {
        name: "logicalAnd",
        sig: "np.logicalAnd(a, b, options?) · np.logicalOr · np.logicalXor · np.logicalNot(a, options?)",
        desc: "Element-wise truth-value AND, OR, XOR and NOT with a `bool` result (zero, `false` and `0+0j` are false; NaN is true). `.reduce` uses NumPy's identities (`logicalAnd` true, the others false).",
        args: [arrayArg("a"), arrayArg("b"), ufuncOpts],
        returns: "NDArray (bool)",
        example: `np.logicalAnd([1, 0, 2], [1, 1, 0]); // => [true, false, false]
np.logicalOr([0, 0, 2], [0, 1, 0]);  // => [false, true, true]
np.logicalXor([1, 1], [0, 1]);       // => [true, false]
np.logicalNot([0, 1.5, NaN]);        // => [true, false, false]
np.logicalXor.reduce([1, 1, 1]);     // => true`,
      },
      {
        name: "all",
        sig: "np.all(a, { axis?, keepdims?, where?, out? }) · np.any · a.all() · a.any()",
        desc: "Whether every (`all`) or some (`any`) element is truthy, over all axes by default. Empty input gives `true` for `all` and `false` for `any`. Also available as `NDArray` methods.",
        args: [
          arrayArg("a"),
          { name: "[options.axis]", type: "number | number[] | null", desc: "Axis or axes to reduce. Omitted or `null` reduces all axes." },
          { name: "[options.keepdims]", type: "boolean", desc: "Keep reduced axes with length 1." },
          { name: "[options.where]", type: "ArrayLike", desc: "Bool mask of the elements to include." },
          { name: "[options.out]", type: "NDArray", desc: "Output array (the result is cast to its dtype)." },
        ],
        returns: "NDArray (bool)",
        example: `np.all([[1, 0], [1, 1]]);              // => false
np.all([[1, 0], [1, 1]], { axis: 0 });  // => [true, false]
np.any([[0, 0], [0, 1]], { axis: 1 });  // => [false, true]
np.all([1, 0], { where: [true, false] }); // => true
np.array([0, 2]).any();                 // => true`,
      },
      {
        name: "isnan",
        sig: "np.isnan(a, options?) · np.isinf · np.isfinite · np.isnat",
        desc: "Element-wise NaN, infinity and finiteness tests with a `bool` result. Integers are never NaN or infinite. A complex value is NaN or infinite if either part is, and finite only if both parts are. `isnat` needs a datetime dtype, which numera does not have yet, so it always raises `DTypeError` (as NumPy does for non-datetime input).",
        args: [arrayArg("a"), ufuncOpts],
        returns: "NDArray (bool)",
        example: `np.isnan([1, NaN, Infinity]);    // => [false, true, false]
np.isinf([1, NaN, -Infinity]);   // => [false, false, true]
np.isfinite([1, NaN, Infinity]); // => [true, false, false]
np.isnan(np.array([np.complex(1, NaN)])); // => [true]`,
      },
      {
        name: "isposinf",
        sig: "np.isposinf(a, { out? }) · np.isneginf(a, { out? })",
        desc: "Element-wise test for `+Infinity` / `-Infinity`. Complex input raises `DTypeError`.",
        args: [arrayArg("a"), { name: "[options.out]", type: "NDArray", desc: "Output array." }],
        returns: "NDArray (bool)",
        example: `np.isposinf([-Infinity, Infinity, 1]); // => [false, true, false]
np.isneginf([-Infinity, Infinity, 1]); // => [true, false, false]`,
      },
      {
        name: "isscalar",
        sig: "np.isscalar(x)",
        desc: "True for JS numbers, booleans, bigints, strings and complex scalars. False for any `NDArray` (also 0-d) and for lists.",
        args: [{ name: "x", type: "unknown", desc: "Any value." }],
        returns: "boolean",
        example: `np.isscalar(3.5);          // => true
np.isscalar(np.array(3.5)); // => false
np.isscalar([1]);           // => false`,
      },
      {
        name: "bitwiseAnd",
        sig: "np.bitwiseAnd(a, b, options?) · np.bitwiseOr · np.bitwiseXor · np.invert(a, options?)",
        desc: "Element-wise `&`, `|`, `^` and `~` on integer and bool arrays (float input raises `DTypeError`). `invert` of bool is logical NOT. `np.bitwiseNot` and `np.bitwiseInvert` are the same ufunc as `np.invert`. `bitwiseAnd.reduce` of an empty array gives all ones.",
        args: [arrayArg("a"), arrayArg("b"), { name: "[options]", type: "UfuncOptions", desc: "`out`, `where`, `dtype`, `casting`, `order`." }],
        returns: "NDArray",
        example: `np.bitwiseAnd([12, 10], [10, 6]); // => [8, 2]
np.bitwiseOr([12, 10], 1);        // => [13, 11]
np.bitwiseXor([12, 10], [10, 6]); // => [6, 12]
np.invert(np.array([5], { dtype: "uint8" })); // => [250]
np.bitwiseNot([true, false]);     // => [false, true]
np.bitwiseInvert([0]);            // => [-1]`,
      },
      {
        name: "leftShift",
        sig: "np.leftShift(a, b, options?) · np.rightShift · np.bitwiseLeftShift · np.bitwiseRightShift",
        desc: "Element-wise `a << b` and arithmetic `a >> b` on integers. A shift count that is negative or at least the bit width gives 0 (`-1` when right-shifting a negative value). `bitwiseLeftShift`/`bitwiseRightShift` are the same ufuncs.",
        args: [arrayArg("a"), arrayArg("b"), { name: "[options]", type: "UfuncOptions", desc: "`out`, `where`, `dtype`, `casting`, `order`." }],
        returns: "NDArray",
        example: `np.leftShift(1, [1, 2, 3]);    // => [2, 4, 8]
np.rightShift([-8, 8], 1);      // => [-4, 4]
np.bitwiseLeftShift(np.array([1], { dtype: "int8" }), 9); // => [0]
np.bitwiseRightShift([16], 2);  // => [4]`,
      },
      {
        name: "bitwiseCount",
        sig: "np.bitwiseCount(a, options?)",
        desc: "Number of 1 bits in the absolute value of each integer (or bool), as `uint8`.",
        args: [arrayArg("a"), { name: "[options]", type: "UfuncOptions", desc: "`out`, `where`, `dtype` (`uint8` only), `casting`, `order`." }],
        returns: "NDArray (uint8)",
        example: `np.bitwiseCount([0, 7, -1, 255]); // => [0, 3, 1, 8]`,
      },
    ],
  },
];
