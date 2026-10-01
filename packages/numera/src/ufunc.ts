import { addon, type NativeUfuncMethodOptions, type NativeUfuncParams } from "./addon.js";
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
  /**
   * Memory layout of a new result (NumPy `order=`, D-050): "K" (default) keeps
   * the inputs' layout, "C"/"F" force row/column-major, and "A" uses F only if
   * every input is F-contiguous. Ignored when `out` is given.
   */
  order?: UfuncOrder | null;
}

/** NumPy ufunc `order=` values (either case, D-050). */
export type UfuncOrder = "C" | "F" | "A" | "K" | "c" | "f" | "a" | "k";

function outArg(opts: { out?: NDArray | null }): NDArray | undefined {
  const out = opts.out;
  if (out === undefined || out === null) return undefined;
  if (!(out instanceof NDArray)) throw new DTypeError("return arrays must be of ArrayType");
  return out;
}

function loopDtype(opts: { dtype?: DTypeLike | null }): DType | undefined {
  return opts.dtype === undefined || opts.dtype === null ? undefined : dtype(opts.dtype);
}

// where= (D-049): `true`/undefined mean no mask; NDArray masks pass through
// (the native side requires bool); lists and scalars convert like NumPy's.
function whereArg(opts: { where?: ArrayLike | boolean | number }): NDArray | undefined {
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
  if (opts.order !== undefined && opts.order !== null) p.order = opts.order;
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

/** `ufunc.reduce` options (D-052). */
export interface UfuncReduceOptions {
  /** Axis or axes to reduce (default 0); `null` reduces all axes, `[]` none. */
  axis?: number | readonly number[] | null;
  /** Loop/result dtype. */
  dtype?: DTypeLike | null;
  /** Write the result into this array and return it. */
  out?: NDArray | null;
  /** Keep reduced axes as size-1 dimensions. */
  keepdims?: boolean;
  /** Starting value (required with `where` for ufuncs without an identity). */
  initial?: number | null;
  /** Bool mask: only true elements take part in the reduction. */
  where?: ArrayLike | boolean | number;
}

/** `ufunc.accumulate` options (D-052). */
export interface UfuncAccumulateOptions {
  /** Axis to accumulate along (default 0). */
  axis?: number;
  dtype?: DTypeLike | null;
  out?: NDArray | null;
}

type BinaryUfunc = ((a: Operand, b: Operand, opts?: UfuncOptions) => NDArray) & {
  /** NumPy ufunc.reduce (D-052). */
  reduce(a: ArrayLike, opts?: UfuncReduceOptions): NDArray;
  /** NumPy ufunc.accumulate (D-052). */
  accumulate(a: ArrayLike, opts?: UfuncAccumulateOptions): NDArray;
};
type UnaryUfunc = (a: ArrayLike, opts?: UfuncOptions) => NDArray;

const REDUCE_KEYS = new Set(["axis", "dtype", "out", "keepdims", "initial", "where"]);
const ACCUMULATE_KEYS = new Set(["axis", "dtype", "out"]);

function checkKeys(method: string, opts: object, allowed: Set<string>): void {
  for (const k of Object.keys(opts)) {
    if (!allowed.has(k)) throw new DTypeError(`${method}() got an unexpected keyword argument '${k}'`);
  }
}

function ufuncMethod(
  method: "reduce" | "accumulate",
  op: string,
  a: ArrayLike,
  opts: UfuncReduceOptions,
): NDArray {
  const out = outArg(opts);
  const x = toArray(a);
  const native: NativeUfuncMethodOptions = {};
  if (opts.axis !== undefined) {
    native.axis = opts.axis === null ? null : typeof opts.axis === "number" ? [opts.axis] : [...opts.axis];
  }
  const loop = loopDtype(opts);
  if (loop) native.dtype = loop.name;
  if (opts.keepdims) native.keepdims = true;
  if (opts.initial !== undefined && opts.initial !== null) native.initial = opts.initial;
  const where = whereArg(opts);
  if (where) native.where = where._native;
  if (out) {
    wrapNative(() => addon.ufuncMethod(method, op, x._native, out._native, native));
    return out;
  }
  return wrapNative(() => NDArray._wrap(addon.ufuncMethod(method, op, x._native, null, native)!));
}

function binaryUfunc(op: string): BinaryUfunc {
  const f = (a: Operand, b: Operand, opts?: UfuncOptions): NDArray => binary(op, a, b, opts);
  return Object.assign(f, {
    reduce: (a: ArrayLike, opts: UfuncReduceOptions = {}): NDArray => {
      checkKeys("reduce", opts, REDUCE_KEYS);
      return ufuncMethod("reduce", op, a, opts);
    },
    accumulate: (a: ArrayLike, opts: UfuncAccumulateOptions = {}): NDArray => {
      checkKeys("accumulate", opts, ACCUMULATE_KEYS);
      if (opts.axis !== undefined && typeof opts.axis !== "number") {
        throw new DTypeError("accumulate axis must be an integer");
      }
      return ufuncMethod("accumulate", op, a, opts);
    },
  });
}

export const add: BinaryUfunc = binaryUfunc("add");
export const subtract: BinaryUfunc = binaryUfunc("subtract");
export const multiply: BinaryUfunc = binaryUfunc("multiply");
export const divide: BinaryUfunc = binaryUfunc("divide");
export const power: BinaryUfunc = binaryUfunc("power");
export const mod: BinaryUfunc = binaryUfunc("mod");
export const floorDivide: BinaryUfunc = binaryUfunc("floorDivide");

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
