import { addon } from "./addon.js";
import { wrapNative } from "./errors.js";
import { array } from "./creation.js";
import { NDArray, type NestedArray, type Shape } from "./ndarray.js";
import { isComplexLike, type ComplexLike } from "./complex.js";

/**
 * Element-wise arithmetic (PLAN §15, M4) with NumPy broadcasting (M5).
 * Semantics and divergences: DECISIONS D-014, complex: D-033.
 */
export type ArrayLike = NDArray | NestedArray;
/** A ufunc operand: an array, nested list or scalar (including complex). */
export type Operand = ArrayLike | ComplexLike;
type Scalar = number | boolean | bigint | ComplexLike;

const isScalar = (x: Operand): x is Scalar =>
  typeof x === "number" || typeof x === "boolean" || typeof x === "bigint" || isComplexLike(x);

// Weak scalar promotion (NumPy 2, NEP 50) for JS values (D-014):
// - integer numbers and bigints act like Python int;
// - other numbers act like Python float;
// - booleans act like Python bool;
// - {re, im} acts like Python complex: complex64 with float16/float32/complex64
//   arrays, complex128 otherwise (D-033).
function scalarFor(value: Scalar, like: NDArray): NDArray {
  const kind = like.dtype.kind;
  if (isComplexLike(value)) {
    const small = ["float16", "float32", "complex64"].includes(like.dtype.name);
    return array(value, { dtype: small ? "complex64" : "complex128" });
  }
  if (typeof value === "boolean") return array(value, { dtype: like.dtype });
  const isInt = typeof value === "bigint" || Number.isInteger(value);
  if (isInt) {
    if (kind === "b") return array(value, { dtype: "int64" });
    return array(value, { dtype: like.dtype });
  }
  if (kind === "f" || kind === "c") return array(value, { dtype: like.dtype });
  return array(value, { dtype: "float64" });
}

function operands(a: Operand, b: Operand): [NDArray, NDArray] {
  const aArr = a instanceof NDArray ? a : isScalar(a) ? undefined : array(a);
  const bArr = b instanceof NDArray ? b : isScalar(b) ? undefined : array(b);
  if (aArr && bArr) return [aArr, bArr];
  if (aArr) return [aArr, scalarFor(b as Scalar, aArr)];
  if (bArr) return [scalarFor(a as Scalar, bArr), bArr];
  return [array(a), array(b)];
}

const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));

function binary(op: string, a: Operand, b: Operand): NDArray {
  const [x, y] = operands(a, b);
  return wrapNative(() => NDArray._wrap(addon.binary(op, x._native, y._native)));
}

function unary(op: string, a: ArrayLike): NDArray {
  const x = toArray(a);
  return wrapNative(() => NDArray._wrap(addon.unary(op, x._native)));
}

export const add = (a: Operand, b: Operand): NDArray => binary("add", a, b);
export const subtract = (a: Operand, b: Operand): NDArray => binary("subtract", a, b);
export const multiply = (a: Operand, b: Operand): NDArray => binary("multiply", a, b);
export const divide = (a: Operand, b: Operand): NDArray => binary("divide", a, b);
export const power = (a: Operand, b: Operand): NDArray => binary("power", a, b);
export const mod = (a: Operand, b: Operand): NDArray => binary("mod", a, b);
export const floorDivide = (a: Operand, b: Operand): NDArray => binary("floorDivide", a, b);

export const abs = (a: ArrayLike): NDArray => unary("abs", a);
export const negative = (a: ArrayLike): NDArray => unary("negative", a);
export const sqrt = (a: ArrayLike): NDArray => unary("sqrt", a);
export const exp = (a: ArrayLike): NDArray => unary("exp", a);
export const log = (a: ArrayLike): NDArray => unary("log", a);

// ---- Complex helpers (P1, D-033) ----

/** NumPy conjugate: complex conjugate; identity (a copy) for real input. */
export const conjugate = (a: ArrayLike): NDArray => unary("conjugate", a);
/** NumPy conj (alias of conjugate). */
export const conj = conjugate;

/** NumPy angle: the argument atan2(im, re), in radians or degrees. */
export function angle(z: ArrayLike, deg = false): NDArray {
  const r = unary("angle", z);
  return deg ? multiply(r, 180 / Math.PI) : r;
}

/** NumPy real: a view of the real part (the array itself for real input). */
export function real(a: ArrayLike): NDArray {
  const x = toArray(a);
  return wrapNative(() => NDArray._wrap(addon.complexPart(x._native, false)));
}

/** NumPy imag: a view of the imaginary part (read-only zeros for real input). */
export function imag(a: ArrayLike): NDArray {
  const x = toArray(a);
  return wrapNative(() => NDArray._wrap(addon.complexPart(x._native, true)));
}

/** NumPy iscomplex: element-wise imag != 0 (all false for real dtypes). */
export function iscomplex(a: ArrayLike): NDArray {
  const x = toArray(a);
  return wrapNative(() => NDArray._wrap(addon.isComplexElementwise(x._native, true)));
}

/** NumPy isreal: element-wise imag == 0 (all true for real dtypes). */
export function isreal(a: ArrayLike): NDArray {
  const x = toArray(a);
  return wrapNative(() => NDArray._wrap(addon.isComplexElementwise(x._native, false)));
}

/** NumPy iscomplexobj: true if the (inferred) dtype is complex. */
export const iscomplexobj = (a: ArrayLike): boolean => toArray(a).dtype.kind === "c";

/** NumPy isrealobj: true if the (inferred) dtype is not complex. */
export const isrealobj = (a: ArrayLike): boolean => !iscomplexobj(a);

/** NumPy broadcast_shapes. Throws BroadcastError on mismatch. */
export function broadcastShapes(...shapes: (Shape | number)[]): number[] {
  const list = shapes.map((s) => (typeof s === "number" ? [s] : [...s]));
  return wrapNative(() => addon.broadcastShapes(list));
}

/** NumPy broadcast_to: a zero-stride view (treat as read-only). */
export function broadcastTo(a: ArrayLike, shape: Shape | number): NDArray {
  const x = toArray(a);
  const s = typeof shape === "number" ? [shape] : [...shape];
  return wrapNative(() => NDArray._wrap(addon.broadcastTo(x._native, s)));
}
