// P5 float classification and isscalar (D-081).
import { isComplexLike } from "./complex.js";
import { DTypeError } from "./errors.js";
import { dtype as toDType } from "./dtype.js";
import { NDArray } from "./ndarray.js";
import { unaryUfunc, type ArrayLike, type UfuncOptions, type UnaryUfunc } from "./ufunc.js";

// dtype= on a bool-output unary ufunc: only bool (checked natively too).
function boolUnaryUfunc(op: string): UnaryUfunc {
  const base = unaryUfunc(op);
  const f = (a: ArrayLike, opts?: UfuncOptions): NDArray => {
    if (opts?.dtype != null && toDType(opts.dtype).name !== "bool") {
      throw new DTypeError(`No loop matching the specified signature and casting was found for ufunc ${op}`);
    }
    return base(a, opts);
  };
  return Object.assign(f, { at: base.at });
}

/** NumPy isnan: element-wise NaN test (bool; false for integers). */
export const isnan: UnaryUfunc = boolUnaryUfunc("isnan");
/** NumPy isinf: element-wise ±inf test (bool; complex: either part). */
export const isinf: UnaryUfunc = boolUnaryUfunc("isinf");
/** NumPy isfinite: element-wise finite test (bool; complex: both parts). */
export const isfinite: UnaryUfunc = boolUnaryUfunc("isfinite");
/**
 * NumPy isnat: NaT test for datetime/timedelta arrays. numera has no datetime
 * dtypes yet, so every input raises DTypeError, like NumPy for other dtypes.
 */
export const isnat: UnaryUfunc = boolUnaryUfunc("isnat");

const posinf = unaryUfunc("isposinf");
const neginf = unaryUfunc("isneginf");

/** NumPy isposinf: element-wise +inf test (complex input raises DTypeError). */
export const isposinf = (x: ArrayLike, opts: { out?: NDArray | null } = {}): NDArray =>
  posinf(x, opts.out == null ? {} : { out: opts.out });
/** NumPy isneginf: element-wise -inf test (complex input raises DTypeError). */
export const isneginf = (x: ArrayLike, opts: { out?: NDArray | null } = {}): NDArray =>
  neginf(x, opts.out == null ? {} : { out: opts.out });

/**
 * NumPy isscalar: true for JS numbers, booleans, bigints, strings and complex
 * scalars; false for NDArrays (also 0-d), lists and other objects.
 */
export function isscalar(x: unknown): boolean {
  const t = typeof x;
  return t === "number" || t === "boolean" || t === "bigint" || t === "string" || isComplexLike(x);
}
