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

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "emath",
    title: "np.emath (complex-valued math)",
    intro: "Functions of `numpy.emath` that switch to complex results outside the real domain.",
    entries: emathEntries,
  },
];
