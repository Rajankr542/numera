// P15 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.

const arrayArg = (name = "x") => ({ name, type: "ArrayLike", desc: "An `NDArray`, nested JS array or scalar." });

const emathEntries = [
  {
    name: "emath",
    sig: "np.emath.{sqrt, log, log2, log10, logn, power, arccos, arcsin, arctanh}",
    desc: "The `numpy.emath` (`numpy.lib.scimath`) namespace: math functions that return complex results where the real function is undefined, instead of NaN.",
    returns: "namespace object",
    example: `Object.keys(np.emath).length; // => 9`,
  },
  {
    name: "emath.sqrt",
    sig: "np.emath.sqrt(x) · np.emath.log(x) · np.emath.log2(x) · np.emath.log10(x)",
    desc: "Square root and logarithms. When any real element is negative the whole input is converted to complex first (`complex64` for int8/uint8/int16/uint16/float32, otherwise `complex128`); otherwise the result is real with the ufunc's float dtype. Complex input stays complex.",
    args: [arrayArg()],
    returns: "NDArray (real or complex)",
    example: `np.emath.sqrt([4, 9]);                  // => [2, 3]
np.real(np.emath.sqrt([-4, 4]));        // => [0, 2]
np.imag(np.emath.sqrt([-4, 4]));        // => [2, 0]
np.emath.sqrt([-4]).dtype.name;         // => "complex128"
np.emath.log10([100]);                  // => [2]
np.imag(np.emath.log(-1)).item();       // => 3.141592653589793`,
  },
  {
    name: "emath.arcsin",
    sig: "np.emath.arcsin(x) · np.emath.arccos(x) · np.emath.arctanh(x)",
    desc: "Inverse sine, cosine and hyperbolic tangent; real input with any `|x| > 1` is converted to complex first (same dtype rule as `emath.sqrt`).",
    args: [arrayArg()],
    returns: "NDArray (real or complex)",
    example: `np.emath.arcsin([0]);                   // => [0]
np.emath.arccos([2]).dtype.name;        // => "complex128"
np.real(np.emath.arccos([2]));          // => [0]
np.imag(np.emath.arcsin([2])).item();   // => 1.3169578969248166`,
  },
  {
    name: "emath.logn",
    sig: "np.emath.logn(n, x) · np.emath.power(x, p)",
    desc: "`logn` is the base-`n` logarithm `log(x) / log(n)`. `power` is `x ** p`, complex when a real `x` is negative; a negative integer `p` is converted to float64 first, so `power([2], -1)` is `0.5`.",
    args: [arrayArg("n / x"), arrayArg("x / p")],
    returns: "NDArray",
    example: `np.emath.logn(2, [4, 8]);               // => [2, 3]
np.emath.power([2, 4], -1);             // => [0.5, 0.25]
np.imag(np.emath.power([-4], 0.5));     // => [2]`,
  },
];


const testingEntries = [
  {
    name: "testing",
    sig: "np.testing.{assertEqual, assertArrayEqual, assertAllclose, ...} · AssertionError",
    desc: "The `numpy.testing` assertion functions. A failing assertion throws `AssertionError` (also a named export) with NumPy's message layout: header, mismatch counts, the first mismatching indices, the largest absolute and relative differences, and the reprs of ACTUAL and DESIRED.",
    returns: "namespace object",
    example: `np.testing.assertArrayEqual([1, 2], [1, 2]);  // passes
let msg = "";
try { np.testing.assertArrayEqual([1, 2, 3], [1, 2, 4]); } catch (e) { msg = e.message; }
msg.split("\\n")[1];                             // => "Arrays are not equal"
msg.split("\\n")[3];                             // => "Mismatched elements: 1 / 3 (33.3%)"
msg.split("\\n")[5];                             // => " [2]: 3 (ACTUAL), 4 (DESIRED)"
msg.split("\\n").at(-1);                         // => " DESIRED: array([1, 2, 4])"`,
  },
  {
    name: "testing.assertArrayEqual",
    sig: "np.testing.assertArrayEqual(actual, desired, { errMsg?, verbose?, strict? }) · assertArrayLess(x, y, opts?) · assertArrayCompare(comparison, x, y, opts?)",
    desc: "`assertArrayEqual`: equal shapes (a 0-d side broadcasts unless `strict`, which also requires equal dtypes) and equal elements; NaNs and Infs must sit at the same positions. `assertArrayLess` checks `x < y`. `assertArrayCompare` is the shared driver: it takes any element-wise comparison returning a bool array, plus `header`, `precision`, `equalNan`, `equalInf`, `names`.",
    args: [arrayArg("actual"), arrayArg("desired")],
    returns: "void (throws AssertionError)",
    example: `np.testing.assertArrayEqual([1, NaN], [1, NaN]);   // passes
np.testing.assertArrayLess([1, 2], [2, 3]);         // passes
np.testing.assertArrayCompare((x, y) => np.lessEqual(x, y), [1, 2], [1, 3]); // passes
np.testing.assertRaises(np.testing.AssertionError, () => np.testing.assertArrayEqual([1, 2], 1, { strict: true })).message.split("\\n")[3]; // => "(shapes (2,), () mismatch)"`,
  },
  {
    name: "testing.assertAllclose",
    sig: "np.testing.assertAllclose(actual, desired, { rtol=1e-7, atol=0, equalNan=true, errMsg?, verbose?, strict? }) · assertArrayAlmostEqual(actual, desired, { decimal=6 })",
    desc: "`assertAllclose` checks `|actual - desired| <= atol + rtol * |desired|` (np.isclose). `assertArrayAlmostEqual` checks `|desired - actual| < 1.5 * 10**-decimal`.",
    args: [arrayArg("actual"), arrayArg("desired")],
    returns: "void (throws AssertionError)",
    example: `np.testing.assertAllclose([1, 2], [1, 2 + 1e-8]);  // passes
np.testing.assertArrayAlmostEqual([1, 2], [1, 2.0000001]); // passes
np.testing.assertRaises(np.testing.AssertionError, () => np.testing.assertAllclose([1, 2], [1, 2.1])).message.split("\\n")[1]; // => "Not equal to tolerance rtol=1e-07, atol=0"`,
  },
  {
    name: "testing.assertEqual",
    sig: "np.testing.assertEqual(actual, desired, opts?) · assertAlmostEqual(actual, desired, { decimal=7 }) · assertApproxEqual(actual, desired, { significant=7 })",
    desc: "`assertEqual` compares plain objects and JS arrays recursively (`key=`/`item=` lines in the message), NDArrays with `assertArrayEqual`, and scalars with NaN equal to NaN and `0` different from `-0`. `assertAlmostEqual` is the scalar/array decimal check; `assertApproxEqual` compares scalars to a number of significant digits.",
    args: [arrayArg("actual"), arrayArg("desired")],
    returns: "void (throws AssertionError)",
    example: `np.testing.assertEqual({ a: [1, NaN] }, { a: [1, NaN] }); // passes
np.testing.assertAlmostEqual(1, 1 + 1e-8);                  // passes
np.testing.assertApproxEqual(1234.5, 1234.6, { significant: 4 }); // passes
np.testing.assertRaises(np.testing.AssertionError, () => np.testing.assertEqual([1, 2], [1, 3])).message; // => "\\nItems are not equal:\\nitem=1\\n\\n ACTUAL: 2\\n DESIRED: 3"`,
  },
  {
    name: "testing.assertArrayMaxUlp",
    sig: "np.testing.assertArrayMaxUlp(a, b, { maxulp=1, dtype? }) · assertArrayAlmostEqualNulp(x, y, nulp=1)",
    desc: "Units-in-the-last-place checks for float arrays. `assertArrayMaxUlp` returns the ULP distances (shape `[1, ...shape]`, common float dtype) and fails when any exceeds `maxulp`. `assertArrayAlmostEqualNulp` checks `|x - y| <= nulp * spacing(max(|x|, |y|))`. Complex input to `assertArrayMaxUlp` throws `NotImplementedError`.",
    args: [arrayArg("a"), arrayArg("b")],
    returns: "NDArray / void",
    example: `np.testing.assertArrayMaxUlp([1], [1 + 2 ** -52]);  // => [[1]]
np.testing.assertArrayAlmostEqualNulp([1], [1 + 2 ** -52]); // passes
np.testing.assertRaises(np.testing.AssertionError, () => np.testing.assertArrayMaxUlp([1], [1 + 1e-15])).message; // => "Arrays are not almost equal up to 1 ULP (max difference is 5 ULP)"`,
  },
  {
    name: "testing.assertRaises",
    sig: "np.testing.assertRaises(ErrorClass, fn, ...args) · assertRaisesRegex(ErrorClass, pattern, fn, ...args) · assert_(val, msg?)",
    desc: "`assertRaises` calls `fn(...args)` and returns the error if it is an instance of `ErrorClass`; other errors propagate, and no error raises `AssertionError(\"<Class> not raised by <fn>\")`. `assertRaisesRegex` also searches the message for `pattern`. `assert_` throws `AssertionError(msg)` for a falsy value (`msg` may be a function). Async functions are not awaited.",
    args: [{ name: "ErrorClass", type: "class", desc: "Expected error class, e.g. `np.ValueError`." }, { name: "fn", type: "function", desc: "Called with `args`." }],
    returns: "Error",
    example: `np.testing.assertRaises(np.ShapeError, () => np.zeros(3).reshape([2])).name; // => "ShapeError"
np.testing.assertRaisesRegex(np.ShapeError, "reshape", () => np.zeros(3).reshape([2])).name; // => "ShapeError"
np.testing.assert_(true);   // passes`,
  },
  {
    name: "testing.assertWarns",
    sig: "np.testing.assertWarns(warningType | null, fn, ...args) · assertNoWarnings(fn, ...args)",
    desc: "Node has no Python warnings module; these watch `process.emitWarning` while `fn` runs (numera emits `RuntimeWarning`s there under `np.seterr` \"warn\"). Captured warnings are not printed. `assertWarns` requires at least one warning of `warningType` (`null`: any) and returns `fn`'s result; `assertNoWarnings` requires none.",
    args: [{ name: "warningType", type: "string | null", desc: "Warning type name, e.g. `\"RuntimeWarning\"`." }, { name: "fn", type: "function", desc: "Called with `args`." }],
    returns: "fn's result",
    example: `np.testing.assertWarns("RuntimeWarning", () => np.divide([1], [0])).dtype.name; // => "float64"
np.testing.assertNoWarnings(() => np.divide([1], [2]));                // => [0.5]`,
  },
  {
    name: "testing.assertStringEqual",
    sig: "np.testing.assertStringEqual(actual, desired) · buildErrMsg(arrays, errMsg?, { header, verbose, names, precision }) · printAssertEqual(testString, actual, desired)",
    desc: "`assertStringEqual` reports a line diff (`- ` actual, `+ ` desired). `buildErrMsg` builds the standard header + ACTUAL/DESIRED message. `printAssertEqual` deep-compares JS values.",
    args: [{ name: "actual", type: "string", desc: "Actual string." }, { name: "desired", type: "string", desc: "Desired string." }],
    returns: "void / string",
    example: `np.testing.buildErrMsg([np.array([1, 2]), 3], "msg"); // => "\\nItems are not equal: msg\\n ACTUAL: array([1, 2])\\n DESIRED: 3"
np.testing.assertStringEqual("abc", "abc");  // passes`,
  },
];

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "emath",
    title: "np.emath (complex-valued math)",
    intro: "Functions of `numpy.emath` that switch to complex results outside the real domain.",
    entries: emathEntries,
  },
  {
    id: "testing",
    title: "np.testing (assertions)",
    intro: "NumPy's array assertion helpers. They throw `AssertionError` (a `NativpyError` subclass) with NumPy-style messages.",
    entries: testingEntries,
  },
];
