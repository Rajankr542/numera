// P15 np.emath (numpy.lib.scimath), D-180.
import { nativeModule, type NativeNDArray } from "./addon.js";
import { asarray } from "./creation.js";
import { wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { any, less } from "./p05_compare.js";
import { divide, power as ufuncPower, type ArrayLike } from "./ufunc.js";

interface P15EmathNative {
  emath(name: string, x: NativeNDArray): NativeNDArray;
}
const native = nativeModule<P15EmathNative>("p15");

const SINGLE = new Set(["int8", "uint8", "int16", "uint16", "float32", "complex64"]);

const isNegativeAnywhere = (x: NDArray): boolean =>
  x.dtype.kind !== "c" && x.size > 0 && Boolean((any(less(x, 0)) as NDArray).item());

// NumPy `_fix_real_lt_zero`: real input with a negative element -> complex.
function fixRealLtZero(x: NDArray): NDArray {
  if (!isNegativeAnywhere(x)) return x;
  return x.astype(SINGLE.has(x.dtype.name) ? "complex64" : "complex128");
}

// NumPy `_fix_int_lt_zero`: real input with a negative element -> x * 1.0.
function fixIntLtZero(x: NDArray): NDArray {
  if (!isNegativeAnywhere(x) || x.dtype.kind === "f") return x;
  return x.astype("float64");
}

const unary = (name: string) => (x: ArrayLike): NDArray =>
  wrapNative(() => NDArray._wrap(native.emath(name, asarray(x)._native)));

/** numpy.emath.sqrt: complex results for negative real input. */
export const sqrt = unary("sqrt");
/** numpy.emath.log: natural log; complex results for negative real input. */
export const log = unary("log");
/** numpy.emath.log2: complex results for negative real input. */
export const log2 = unary("log2");
/** numpy.emath.log10: complex results for negative real input. */
export const log10 = unary("log10");
/** numpy.emath.arccos: complex results for real input with |x| > 1. */
export const arccos = unary("arccos");
/** numpy.emath.arcsin: complex results for real input with |x| > 1. */
export const arcsin = unary("arcsin");
/** numpy.emath.arctanh: complex results for real input with |x| > 1. */
export const arctanh = unary("arctanh");

/** numpy.emath.logn(n, x): log base `n` of `x`, i.e. `log(x) / log(n)`. */
export function logn(n: ArrayLike, x: ArrayLike): NDArray {
  return divide(log(fixRealLtZero(asarray(x))), log(fixRealLtZero(asarray(n))));
}

/**
 * numpy.emath.power(x, p): `x ** p`, complex for negative real `x`; a negative
 * integer `p` is converted to float64 first. JS scalars are 0-d arrays
 * (int64/float64), as `np.asarray` makes them in NumPy.
 */
export function power(x: ArrayLike, p: ArrayLike): NDArray {
  return ufuncPower(fixRealLtZero(asarray(x)), fixIntLtZero(asarray(p)));
}

export const emath = { sqrt, log, log2, log10, logn, power, arccos, arcsin, arctanh } as const;
