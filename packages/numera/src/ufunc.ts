import { addon, type NativeUfuncParams } from "./addon.js";
import { DTypeError, wrapNative } from "./errors.js";
import { array } from "./creation.js";
import { dtype, type Casting, type DType, type DTypeLike } from "./dtype.js";
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
function scalarFor(value: Scalar, like: DType): NDArray {
  const kind = like.kind;
  if (isComplexLike(value)) {
    const small = ["float16", "float32", "complex64"].includes(like.name);
    return array(value, { dtype: small ? "complex64" : "complex128" });
  }
  if (typeof value === "boolean") return array(value, { dtype: like });
  const isInt = typeof value === "bigint" || Number.isInteger(value);
  if (isInt) {
    if (kind === "b") return array(value, { dtype: "int64" });
    return array(value, { dtype: like });
  }
  if (kind === "f" || kind === "c") return array(value, { dtype: like });
  return array(value, { dtype: "float64" });
}

// With `dtype=` (D-048), a JS scalar is weak relative to that loop dtype
// instead of the other operand, as in NumPy.
function operands(a: Operand, b: Operand, loop?: DType): [NDArray, NDArray] {
  const aArr = a instanceof NDArray ? a : isScalar(a) ? undefined : array(a);
  const bArr = b instanceof NDArray ? b : isScalar(b) ? undefined : array(b);
  const weak = (v: Operand, other?: NDArray): NDArray => {
    const like = loop ?? other?.dtype;
    return like ? scalarFor(v as Scalar, like) : array(v);
  };
  return [aArr ?? weak(a, bArr), bArr ?? weak(b, aArr)];
}

const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));

/** Ufunc keyword options (D-047, D-048). */
export interface UfuncOptions {
  /** Write the result into this array and return it (NumPy `out=`, D-046). */
  out?: NDArray | null;
  /** Loop dtype: inputs are cast to it and the result has it (NumPy `dtype=`). */
  dtype?: DTypeLike | null;
  /** Casting rule for inputs and `out` (NumPy `casting=`, default "same_kind"). */
  casting?: Casting;
  /**
   * Bool mask (NumPy `where=`, D-049): only true positions are computed. With
   * `out` the others keep their values; without `out` they are zero.
   */
  where?: ArrayLike | boolean | number;
}

function outArg(opts: UfuncOptions): NDArray | undefined {
  const out = opts.out;
  if (out === undefined || out === null) return undefined;
  if (!(out instanceof NDArray)) throw new DTypeError("return arrays must be of ArrayType");
  return out;
}

function loopDtype(opts: UfuncOptions): DType | undefined {
  return opts.dtype === undefined || opts.dtype === null ? undefined : dtype(opts.dtype);
}

// where= (D-049): `true`/undefined mean no mask; NDArray masks pass through
// (the native side requires bool); lists and scalars convert like NumPy's.
function whereArg(opts: UfuncOptions): NDArray | undefined {
  const w = opts.where;
  if (w === undefined || w === true) return undefined;
  if (w === null) throw new DTypeError("where= must be an array, nested list, boolean or number, not null");
  if (w instanceof NDArray) return w;
  return array(w, { dtype: "bool" });
}

function nativeParams(loop: DType | undefined, opts: UfuncOptions): NativeUfuncParams {
  const p: NativeUfuncParams = {};
  if (loop) p.dtype = loop.name;
  if (opts.casting !== undefined) p.casting = opts.casting;
  const where = whereArg(opts);
  if (where) p.where = where._native;
  return p;
}

function binary(op: string, a: Operand, b: Operand, opts: UfuncOptions = {}): NDArray {
  const out = outArg(opts);
  const loop = loopDtype(opts);
  const [x, y] = operands(a, b, loop);
  const params = nativeParams(loop, opts);
  if (out) {
    wrapNative(() => addon.binary(op, x._native, y._native, out._native, params));
    return out;
  }
  return wrapNative(() => NDArray._wrap(addon.binary(op, x._native, y._native, null, params)));
}

function unary(op: string, a: ArrayLike, opts: UfuncOptions = {}): NDArray {
  const out = outArg(opts);
  const params = nativeParams(loopDtype(opts), opts);
  const x = toArray(a);
  if (out) {
    wrapNative(() => addon.unary(op, x._native, out._native, params));
    return out;
  }
  return wrapNative(() => NDArray._wrap(addon.unary(op, x._native, null, params)));
}

type BinaryUfunc = (a: Operand, b: Operand, opts?: UfuncOptions) => NDArray;
type UnaryUfunc = (a: ArrayLike, opts?: UfuncOptions) => NDArray;

export const add: BinaryUfunc = (a, b, opts) => binary("add", a, b, opts);
export const subtract: BinaryUfunc = (a, b, opts) => binary("subtract", a, b, opts);
export const multiply: BinaryUfunc = (a, b, opts) => binary("multiply", a, b, opts);
export const divide: BinaryUfunc = (a, b, opts) => binary("divide", a, b, opts);
export const power: BinaryUfunc = (a, b, opts) => binary("power", a, b, opts);
export const mod: BinaryUfunc = (a, b, opts) => binary("mod", a, b, opts);
export const floorDivide: BinaryUfunc = (a, b, opts) => binary("floorDivide", a, b, opts);

export const abs: UnaryUfunc = (a, opts) => unary("abs", a, opts);
export const negative: UnaryUfunc = (a, opts) => unary("negative", a, opts);
export const sqrt: UnaryUfunc = (a, opts) => unary("sqrt", a, opts);
export const exp: UnaryUfunc = (a, opts) => unary("exp", a, opts);
export const log: UnaryUfunc = (a, opts) => unary("log", a, opts);

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
