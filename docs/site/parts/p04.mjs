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
    ],
  },
];
