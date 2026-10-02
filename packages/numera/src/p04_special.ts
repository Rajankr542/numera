import { nativeModule, type NativeNDArray } from "./addon.js";
import { DTypeError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { toArray, type ArrayLike } from "./ufunc.js";

// P4-6: special functions (native/core/p04_special.cpp, D-074).

interface P04SpecialNative {
  i0(a: NativeNDArray): NativeNDArray;
  sinc(a: NativeNDArray): NativeNDArray;
  nanToNum(a: NativeNDArray, copy: boolean, nan: number, posinf: number | null, neginf: number | null): NativeNDArray;
  imagAllBelow(a: NativeNDArray, tol: number): boolean;
  unwrap(
    a: NativeNDArray,
    period: number,
    discont: number | null,
    axis: number,
    integer: boolean,
    out: string,
  ): NativeNDArray;
}
const native = nativeModule<P04SpecialNative>("p04");
const call = (f: () => NativeNDArray): NDArray => wrapNative(() => NDArray._wrap(f()));

/** NumPy i0: modified Bessel function of the first kind, order 0 (real input only). */
export function i0(x: ArrayLike | number): NDArray {
  const a = toArray(x as ArrayLike);
  return call(() => native.i0(a._native));
}

/** NumPy sinc: normalized `sin(pi x) / (pi x)`, 1 at 0. Complex input is supported. */
export function sinc(x: ArrayLike | number): NDArray {
  const a = toArray(x as ArrayLike);
  return call(() => native.sinc(a._native));
}

/** `np.nanToNum` options (scalar replacements only). */
export interface NanToNumOptions {
  /** Return a copy (default) or modify `x` in place (`x` must be an NDArray). */
  copy?: boolean;
  /** Replacement for NaN (default 0). */
  nan?: number;
  /** Replacement for +Infinity (default: largest finite value of the dtype). */
  posinf?: number | null;
  /** Replacement for -Infinity (default: most negative finite value). */
  neginf?: number | null;
}

/** NumPy nan_to_num: replace NaN and infinities (both parts of complex values). */
export function nanToNum(x: ArrayLike | number, opts: NanToNumOptions = {}): NDArray {
  const copy = opts.copy ?? true;
  if (!copy && !(x instanceof NDArray)) throw new DTypeError("nanToNum: copy=false requires an NDArray");
  const a = toArray(x as ArrayLike);
  return call(() => native.nanToNum(a._native, copy, opts.nan ?? 0, opts.posinf ?? null, opts.neginf ?? null));
}

const EPS: Record<string, number> = { complex64: 2 ** -23, complex128: 2 ** -52 };

/** NumPy real_if_close: the real part if every imaginary part is below `tol` (in machine epsilons when `tol > 1`). */
export function realIfClose(x: ArrayLike | number, tol = 100): NDArray {
  const a = toArray(x as ArrayLike);
  const eps = EPS[a.dtype.name];
  if (eps === undefined) return a;
  const t = tol > 1 ? eps * tol : tol;
  return wrapNative(() => native.imagAllBelow(a._native, t)) ? a.real : a;
}

/** `np.unwrap` options. */
export interface UnwrapOptions {
  /** Maximum jump left alone (default `period / 2`). */
  discont?: number | null;
  /** Axis to unwrap along (default -1). */
  axis?: number;
  /** Period of the signal (default `2 * pi`). */
  period?: number;
}

/**
 * NumPy unwrap: remove jumps larger than `discont` by adding multiples of
 * `period`. Integer input with an integer `period` keeps its dtype (NumPy's
 * integer path); otherwise the result is float (float16/float32 kept).
 */
export function unwrap(p: ArrayLike, opts: UnwrapOptions = {}): NDArray {
  const a = toArray(p);
  const period = opts.period ?? 2 * Math.PI;
  const kind = a.dtype.kind;
  if (kind === "c") throw new DTypeError("ufunc 'remainder' not supported for the input types");
  const integer = (kind === "i" || kind === "u" || kind === "b") && Number.isInteger(period);
  let out: string;
  if (integer) out = kind === "b" ? "int64" : a.dtype.name;
  else out = kind === "f" ? a.dtype.name : "float64";
  return call(() => native.unwrap(a._native, period, opts.discont ?? null, opts.axis ?? -1, integer, out));
}
