// P4 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.
const arrayArg = (name = "a") => ({ name, type: "ArrayLike", desc: "An `NDArray`, nested JS array or scalar." });
const ufuncOpts = { name: "[opts]", type: "UfuncOptions", desc: "`out`, `where`, `dtype`, `casting`, `order` (NumPy ufunc keywords)." };

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "math",
    entries: [
      {
        name: "sin",
        sig: "np.sin(x, opts?) · np.cos · np.tan · np.arcsin · np.arccos · np.arctan (aliases np.asin · np.acos · np.atan)",
        desc: "Element-wise trigonometric functions and their inverses, in radians. Integer inputs give the smallest float that holds them (`int8` → `float16`, `int16` → `float32`, wider → `float64`). Complex inputs use complex loops.",
        args: [arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.sin([0, Math.PI / 2]);  // => [0, 1]
np.cos([0]);               // => [1]
np.arctan([1]);            // => [0.7853981633974483]
np.asin === np.arcsin;     // => true`,
      },
      {
        name: "sinh",
        sig: "np.sinh(x, opts?) · np.cosh · np.tanh · np.arcsinh · np.arccosh · np.arctanh (aliases np.asinh · np.acosh · np.atanh)",
        desc: "Element-wise hyperbolic functions and their inverses, with complex loops.",
        args: [arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.sinh([0]);         // => [0]
np.cosh([0]);         // => [1]
np.tanh([Infinity]);  // => [1]
np.arccosh([1]);      // => [0]
np.atanh === np.arctanh; // => true`,
      },
      {
        name: "arctan2",
        sig: "np.arctan2(y, x, opts?) (alias np.atan2) · np.hypot(a, b, opts?)",
        desc: "`arctan2` is the quadrant-aware angle of the point `(x, y)`; `hypot` is `sqrt(a² + b²)` without intermediate overflow. Real (float) loops only.",
        args: [arrayArg("y"), arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.arctan2([1, -1], [1, -1]); // => [0.7853981633974483, -2.356194490192345]
np.hypot([3, 5], [4, 12]);    // => [5, 13]
np.atan2 === np.arctan2;      // => true`,
      },
      {
        name: "deg2rad",
        sig: "np.deg2rad(x, opts?) · np.radians · np.rad2deg · np.degrees",
        desc: "Angle conversion. `radians`/`degrees` are the same operations as `deg2rad`/`rad2deg`.",
        args: [arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.deg2rad([180]);       // => [3.141592653589793]
np.radians([90]);        // => [1.5707963267948966]
np.rad2deg([Math.PI]);   // => [180]
np.degrees([Math.PI / 2]); // => [90]`,
      },
      {
        name: "exp2",
        sig: "np.exp2(x, opts?) · np.expm1 · np.log2 · np.log10 · np.log1p",
        desc: "`2**x`, `exp(x) - 1`, base-2 / base-10 logarithms and `log(1 + x)`. `expm1`/`log1p` stay accurate for tiny `x`. Complex loops are available.",
        args: [arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.exp2([3, -1]);   // => [8, 0.5]
np.log2([8]);       // => [3]
np.log10([1000]);   // => [3]
np.expm1([0]);      // => [0]
np.log1p([0]);      // => [0]`,
      },
      {
        name: "logaddexp",
        sig: "np.logaddexp(a, b, opts?) · np.logaddexp2(a, b, opts?)",
        desc: "`log(exp(a) + exp(b))` and `log2(2**a + 2**b)` without overflow. Identity `-Infinity`, so `.reduce` works on empty input.",
        args: [arrayArg("a"), arrayArg("b"), ufuncOpts],
        returns: "NDArray",
        example: `np.logaddexp(0, 0).toArray();         // => 0.6931471805599453
np.logaddexp2(1, 1).toArray();        // => 2
np.logaddexp2.reduce([1, 1, 2]).toArray(); // => 3`,
      },
      {
        name: "cbrt",
        sig: "np.cbrt(x, opts?) · np.square(x, opts?) · np.reciprocal(x, opts?)",
        desc: "Cube root (real only); `x*x` and `1/x`, which keep integer dtypes (integer `reciprocal` truncates, `bool` uses `int8`).",
        args: [arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.cbrt([-8, 27]);       // => [-2, 3]
np.square([2, -3]);      // => [4, 9]
np.reciprocal([4, 0.5]); // => [0.25, 2]
np.reciprocal([2, 1]);   // => [0, 1]`,
      },
      {
        name: "floor",
        sig: "np.floor(x, opts?) · np.ceil · np.trunc · np.fix · np.rint",
        desc: "Round down, up, toward zero (`fix` is the same as `trunc`) and to the nearest even integer. `floor`/`ceil`/`trunc` keep integer and bool dtypes; `rint` gives floats and also rounds complex parts.",
        args: [arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.floor([-1.5, 1.5]);  // => [-2, 1]
np.ceil([-1.5, 1.5]);   // => [-1, 2]
np.trunc([-1.7, 1.7]);  // => [-1, 1]
np.fix([-1.7, 1.7]);    // => [-1, 1]
np.rint([0.5, 1.5]);    // => [0, 2]`,
      },
      {
        name: "round",
        sig: "np.round(a, decimals = 0, { out? }) · np.around · a.round(decimals?)",
        desc: "Round half to even to `decimals` places; negative `decimals` round to tens, hundreds, …. Floats compute `rint(a * 10**d) / 10**d` like NumPy. Integer inputs keep their dtype.",
        args: [arrayArg(), { name: "[decimals]", type: "number", desc: "Number of decimal places (integer, default 0)." }],
        returns: "NDArray",
        example: `np.round([0.5, 1.5, 2.5]);   // => [0, 2, 2]
np.round([1.25, 2.567], 2);  // => [1.25, 2.57]
np.around([15, 25], -1);     // => [20, 20]
np.array([3.14159]).round(3); // => [3.142]`,
      },
      {
        name: "positive",
        sig: "np.positive(x, opts?)",
        desc: "Element-wise `+x` (a copy). No `bool` loop, as in NumPy.",
        args: [arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.positive([-3, 2]); // => [-3, 2]`,
      },
      {
        name: "fmod",
        sig: "np.fmod(a, b, opts?) · np.remainder (alias of np.mod) · np.trueDivide (alias of np.divide) · np.pow (alias of np.power) · np.absolute (alias of np.abs)",
        desc: "`fmod` is the C remainder: the result takes the sign of the dividend (unlike `mod`/`remainder`, which follow the divisor). Integer `x % 0` is 0. The aliases are the same objects as the original ufuncs.",
        args: [arrayArg("a"), arrayArg("b"), ufuncOpts],
        returns: "NDArray",
        example: `np.fmod([-7, 7], 3);       // => [-1, 1]
np.remainder([-7, 7], 3);  // => [2, 1]
np.pow === np.power;       // => true`,
      },
      {
        name: "floatPower",
        sig: "np.floatPower(a, b, opts?)",
        desc: "Element-wise `a ** b` computed in `float64` (or `complex128`), so integer inputs can take negative exponents.",
        args: [arrayArg("a"), arrayArg("b"), ufuncOpts],
        returns: "NDArray",
        example: `np.floatPower([2, 3], 2);  // => [4, 9]
np.floatPower([2], -1);    // => [0.5]`,
      },
      {
        name: "sign",
        sig: "np.sign(x, opts?) · np.heaviside(x, h0, opts?) · np.fabs(x, opts?)",
        desc: "`sign` gives -1, 0 or 1 (NaN stays NaN; complex `z/|z|`). `heaviside` is 0 for `x < 0`, `h0` at 0 and 1 for `x > 0`. `fabs` is the float-only absolute value.",
        args: [arrayArg("x"), ufuncOpts],
        returns: "NDArray",
        example: `np.sign([-2, 0, 3]);            // => [-1, 0, 1]
np.heaviside([-1, 0, 2], 0.5);  // => [0, 0.5, 1]
np.fabs([-1.5]);                // => [1.5]`,
      },
      {
        name: "maximum",
        sig: "np.maximum(a, b, opts?) · np.minimum · np.fmax · np.fmin",
        desc: "Element-wise maximum/minimum. `maximum`/`minimum` propagate NaN; `fmax`/`fmin` return the non-NaN operand. Use `.reduce` for an axis-wise max without an identity.",
        args: [arrayArg("a"), arrayArg("b"), ufuncOpts],
        returns: "NDArray",
        example: `np.maximum([1, 5], [3, 2]);      // => [3, 5]
np.minimum([NaN, 1], [0, 0]).toArray()[1]; // => 0
np.fmax([NaN, 1], [0, 0]);       // => [0, 1]
np.fmin([NaN, 1], [0, 0]);       // => [0, 0]
np.maximum.reduce([3, 9, 2]).toArray(); // => 9`,
      },
      {
        name: "clip",
        sig: "np.clip(a, min?, max?, { out? }) · a.clip(min?, max?)",
        desc: "Limit values to `[min, max]`; `null` skips a side. Computed as `minimum(maximum(a, min), max)` like NumPy, so NaN propagates and `min > max` gives `max`. Also adds `a.conjugate()` (complex conjugate).",
        args: [arrayArg(), arrayArg("min"), arrayArg("max")],
        returns: "NDArray",
        example: `np.clip([1, 5, 9], 2, 6);        // => [2, 5, 6]
np.clip([1, 5, 9], null, 4);     // => [1, 4, 4]
np.array([1, 5, 9]).clip(4);     // => [4, 5, 9]
np.array([1, 2]).conjugate();    // => [1, 2]`,
      },
      {
        name: "copysign",
        sig: "np.copysign(a, b, opts?) · np.nextafter(a, b, opts?) · np.spacing(x, opts?) · np.signbit(x, opts?)",
        desc: "Floating-point bit helpers: magnitude of `a` with the sign of `b`; the next representable value after `a` toward `b`; the gap to the next value away from zero; whether the sign bit is set (`bool` result, true for `-0`). Float loops only.",
        args: [arrayArg("a"), arrayArg("b"), ufuncOpts],
        returns: "NDArray",
        example: `np.copysign([3, 2], [-1, 1]);  // => [-3, 2]
np.nextafter([0], [-1]);       // => [-5e-324]
np.spacing([1]);               // => [2.220446049250313e-16]
np.signbit([-0.0, 1]);         // => [true, false]`,
      },
      {
        name: "ldexp",
        sig: "np.ldexp(x, n, opts?)",
        desc: "`x * 2**n` for an integer exponent array `n` (non-integer or `uint64` exponents raise `DTypeError`, as in NumPy).",
        args: [arrayArg("x"), arrayArg("n"), ufuncOpts],
        returns: "NDArray",
        example: `np.ldexp([1.5, 1.5], [3, -1]); // => [12, 0.75]`,
      },
      {
        name: "gcd",
        sig: "np.gcd(a, b, opts?) · np.lcm(a, b, opts?)",
        desc: "Greatest common divisor and least common multiple of integers (results are non-negative). `gcd` has identity 0.",
        args: [arrayArg("a"), arrayArg("b"), ufuncOpts],
        returns: "NDArray",
        example: `np.gcd([12, -12, 7], [18, 18, 0]); // => [6, 6, 7]
np.lcm([4, -3], [6, 7]);           // => [12, 21]
np.gcd.reduce([12, 18, 27]).toArray(); // => 3`,
      },
      {
        name: "divmod",
        sig: "np.divmod(a, b, { out?, dtype?, casting? }) · np.modf(x, opts?) · np.frexp(x, opts?)",
        desc: "Two-output ufuncs returning `[NDArray, NDArray]`. `divmod` gives `[floorDivide(a, b), mod(a, b)]`; `modf` gives `[fractional, integral]` parts; `frexp` gives `[mantissa, exponent]` with `x = m * 2**e` and an `int32` exponent. `out` is a pair (entries may be `null`).",
        args: [arrayArg("a"), arrayArg("b"), { name: "[opts]", type: "MultiUfuncOptions", desc: "`out: [o1, o2]`, `dtype`, `casting`." }],
        returns: "[NDArray, NDArray]",
        example: `np.divmod([7, -7], 2).map((x) => x.toArray()); // => [[3, -4], [1, 1]]
np.modf([-2.5]).map((x) => x.toArray());       // => [[-0.5], [-2]]
np.frexp([8]).map((x) => x.toArray());         // => [[0.5], [4]]`,
      },
    ],
  },
];
