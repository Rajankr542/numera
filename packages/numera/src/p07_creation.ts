// P7 creation routines (D-100, D-101): logspace/geomspace, triangular and
// diagonal matrices, vander, tri/diag index helpers, fillDiagonal and the
// from* constructors. Kernels: native/core/p07_creation.cpp (`addon.p07`).
import { isComplexLike, type ComplexLike } from "./complex.js";
import { array, asarray, fromTypedArray } from "./creation.js";
import { dtype as toDType, type DTypeLike } from "./dtype.js";
import { DTypeError, ValueError, wrapNative } from "./errors.js";
import { nonzero } from "./indexing.js";
import { native, wrapP07 } from "./p07_native.js";
import { NDArray, type AstypeOptions, type NestedArray, type Shape } from "./ndarray.js";

type ArrayInput = NDArray | NestedArray;
type ScalarLike = number | ComplexLike;

const parts = (v: ScalarLike): [number, number] =>
  isComplexLike(v) ? [v.re, v.im ?? 0] : [v, 0];

export interface LogspaceOptions {
  endpoint?: boolean;
  /** Default 10. */
  base?: number;
  dtype?: DTypeLike;
}

/** np.logspace(start, stop, num = 50): `base ** linspace(start, stop, num)`. Scalar arguments. */
export function logspace(start: ScalarLike, stop: ScalarLike, num = 50, options: LogspaceOptions = {}): NDArray {
  const [sr, si] = parts(start);
  const [er, ei] = parts(stop);
  const cplx = isComplexLike(start) || isComplexLike(stop);
  const dt = options.dtype === undefined ? null : toDType(options.dtype).name;
  return wrapP07((n) =>
    n.logspace(sr, si, er, ei, cplx, num, options.endpoint ?? true, options.base ?? 10, dt),
  );
}

export interface GeomspaceOptions {
  endpoint?: boolean;
  dtype?: DTypeLike;
}

/** np.geomspace(start, stop, num = 50): a geometric progression (negative and complex endpoints allowed). */
export function geomspace(start: ScalarLike, stop: ScalarLike, num = 50, options: GeomspaceOptions = {}): NDArray {
  const [sr, si] = parts(start);
  const [er, ei] = parts(stop);
  const want = options.dtype === undefined ? undefined : toDType(options.dtype);
  const cplx = isComplexLike(start) || isComplexLike(stop) || want?.kind === "c";
  const dt = want ?? toDType(cplx ? "complex128" : "float64");
  return wrapP07((n) => n.geomspace(sr, si, er, ei, cplx, num, options.endpoint ?? true, dt.name));
}

export interface TriOptions {
  k?: number;
  dtype?: DTypeLike;
}

/** np.tri(N, M = N, {k = 0, dtype = float64}): ones at and below diagonal `k`. */
export function tri(n: number, m?: number | null, options: TriOptions = {}): NDArray {
  const dt = toDType(options.dtype ?? "float64").name;
  return wrapP07((x) => x.tri(n, m ?? n, options.k ?? 0, dt));
}

/** np.tril(m, k = 0): copy with elements above diagonal `k` zeroed (last two axes). */
export function tril(m: ArrayInput, k = 0): NDArray {
  const a = asarray(m);
  return wrapP07((n) => n.tril(a._native, k));
}

/** np.triu(m, k = 0): copy with elements below diagonal `k` zeroed (last two axes). */
export function triu(m: ArrayInput, k = 0): NDArray {
  const a = asarray(m);
  return wrapP07((n) => n.triu(a._native, k));
}

/** np.diag(v, k = 0): 1-D input builds a 2-D array; 2-D input returns a read-only view of diagonal `k`. */
export function diag(v: ArrayInput, k = 0): NDArray {
  const a = asarray(v);
  return wrapP07((n) => n.diag(a._native, k));
}

/** np.diagflat(v, k = 0): 2-D array with the flattened input on diagonal `k`. */
export function diagflat(v: ArrayInput, k = 0): NDArray {
  const a = asarray(v).ravel();
  return wrapP07((n) => n.diag(a._native, k));
}

/** np.vander(x, N = len(x), {increasing = false}): Vandermonde matrix. */
export function vander(x: ArrayInput, n?: number | null, options: { increasing?: boolean } = {}): NDArray {
  const a = asarray(x);
  return wrapP07((m) => m.vander(a._native, n ?? null, options.increasing ?? false));
}

const triIdx = (n: number, k: number, m: number, upper: boolean): NDArray[] =>
  wrapNative(() => native.triIndices(n, m, k, upper).map((h) => NDArray._wrap(h)));

/** np.tril_indices(n, k = 0, m = n): [rows, cols] of the lower triangle. */
export function trilIndices(n: number, k = 0, m?: number | null): NDArray[] {
  return triIdx(n, k, m ?? n, false);
}

/** np.triu_indices(n, k = 0, m = n): [rows, cols] of the upper triangle. */
export function triuIndices(n: number, k = 0, m?: number | null): NDArray[] {
  return triIdx(n, k, m ?? n, true);
}

const shape2d = (arr: NDArray): [number, number] => {
  if (arr.ndim !== 2) throw new ValueError("input array must be 2-d");
  return [arr.shape[0] as number, arr.shape[1] as number];
};

/** np.tril_indices_from(arr, k = 0). */
export function trilIndicesFrom(arr: NDArray, k = 0): NDArray[] {
  const [r, c] = shape2d(arr);
  return triIdx(r, k, c, false);
}

/** np.triu_indices_from(arr, k = 0). */
export function triuIndicesFrom(arr: NDArray, k = 0): NDArray[] {
  const [r, c] = shape2d(arr);
  return triIdx(r, k, c, true);
}

/** np.diag_indices(n, ndim = 2): `ndim` copies of `arange(n)` (the same array, as NumPy). */
export function diagIndices(n: number, ndim = 2): NDArray[] {
  const idx = wrapP07((x) => x.gridAxis(0, 1, n, "int64"));
  return Array.from({ length: ndim }, () => idx);
}

/** np.diag_indices_from(arr): main-diagonal indices of an n-d array with equal dimensions. */
export function diagIndicesFrom(arr: NDArray): NDArray[] {
  if (arr.ndim < 2) throw new ValueError("input array must be at least 2-d");
  const n = arr.shape[0] as number;
  if (!arr.shape.every((d) => d === n)) {
    throw new ValueError("All dimensions of input must be of equal length");
  }
  return diagIndices(n, arr.ndim);
}

/** np.mask_indices(n, maskFunc, k = 0): indices where `maskFunc(ones((n, n)), k)` is nonzero. */
export function maskIndices(n: number, maskFunc: (m: NDArray, k: number) => NDArray, k = 0): NDArray[] {
  const m = wrapP07((x) => x.tri(n, n, n, "int64"));
  return nonzero(maskFunc(m, k));
}

/**
 * np.fill_diagonal(a, val, {wrap = false}), in place. `val` is converted to
 * `a`'s dtype (D-009) and repeats cyclically along the diagonal.
 */
export function fillDiagonal(a: NDArray, val: ArrayInput, options: { wrap?: boolean } = {}): void {
  const v = val instanceof NDArray ? val.astype(a.dtype) : array(val, { dtype: a.dtype });
  wrapNative(() => native.fillDiagonal(a._native, v._native, options.wrap ?? false));
}

/**
 * np.fromfunction(fn, shape, {dtype = float64}): calls `fn` once with one
 * coordinate array per axis (like NumPy) and returns its result.
 */
export function fromfunction<R>(fn: (...coords: NDArray[]) => R, shape: Shape, options: { dtype?: DTypeLike } = {}): R {
  const dt = toDType(options.dtype ?? "float64").name;
  const grid = wrapP07((n) => n.indices([...shape], dt));
  return fn(...shape.map((_, k) => grid.get(k)));
}

/** np.fromiter(iterable, dtype, count = -1): a 1-D array from an iterable (D-101). */
export function fromiter(
  iterable: Iterable<number | boolean | bigint | ComplexLike>,
  dtype: DTypeLike,
  count = -1,
): NDArray {
  const dt = toDType(dtype);
  const values: (number | boolean | bigint | ComplexLike)[] = [];
  if (count !== 0) {
    for (const v of iterable) {
      values.push(v);
      if (count > 0 && values.length >= count) break;
    }
  }
  if (count > 0 && values.length < count) {
    throw new ValueError(
      `iterator too short: Expected ${count} but iterator had only ${values.length} items.`,
    );
  }
  return array(values as NestedArray, { dtype: dt });
}

export interface FrombufferOptions {
  /** Default float64. */
  dtype?: DTypeLike;
  /** Number of items; -1 (default) reads to the end. */
  count?: number;
  /** Start offset in bytes. */
  offset?: number;
}

/**
 * np.frombuffer(buffer, {dtype, count, offset}): interprets raw bytes of an
 * ArrayBuffer / TypedArray / DataView. The data is copied (D-101).
 */
export function frombuffer(buffer: ArrayBufferLike | ArrayBufferView, options: FrombufferOptions = {}): NDArray {
  const bytes = ArrayBuffer.isView(buffer)
    ? new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength)
    : new Uint8Array(buffer);
  const dt = toDType(options.dtype ?? "float64");
  const offset = options.offset ?? 0;
  const count = options.count ?? -1;
  const len = bytes.byteLength;
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(count)) {
    throw new ValueError("count and offset must be integers");
  }
  if (offset < 0 || offset > len) {
    throw new ValueError(`offset must be non-negative and no greater than buffer length (${len})`);
  }
  const avail = len - offset;
  let n = count;
  if (count < 0) {
    if (avail % dt.itemSize !== 0) throw new ValueError("buffer size must be a multiple of element size");
    n = avail / dt.itemSize;
  } else if (avail < count * dt.itemSize) {
    throw new ValueError("buffer is smaller than requested size");
  }
  return fromTypedArray(bytes.subarray(offset, offset + n * dt.itemSize), [n], { dtype: dt });
}

export interface FromstringOptions {
  dtype?: DTypeLike;
  count?: number;
  /** Item separator; required (text mode only, D-101). Whitespace matches any run of whitespace. */
  sep?: string;
}

/** np.fromstring(string, {dtype, count, sep}): parses numbers separated by `sep` (D-101). */
export function fromstring(text: string, options: FromstringOptions = {}): NDArray {
  const dt = toDType(options.dtype ?? "float64").name;
  return wrapP07((n) => n.fromstring(text, dt, options.count ?? -1, options.sep ?? ""));
}

/** np.astype(x, dtype, {copy = true}) (array API): `x.astype(dtype)`; `x` must be an NDArray. */
export function astype(x: NDArray, dtype: DTypeLike, options: Pick<AstypeOptions, "copy"> = {}): NDArray {
  if (!(x instanceof NDArray)) {
    const t = Array.isArray(x) ? "list" : typeof x;
    throw new DTypeError(`Input should be a NumPy array or scalar. It is a ${t} instead.`);
  }
  return x.astype(dtype, { copy: options.copy ?? true });
}
