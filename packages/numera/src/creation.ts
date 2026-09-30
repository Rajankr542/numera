import { addon } from "./addon.js";
import { dtype as toDType, type DType, type DTypeLike } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";
import { isComplexLike, type ComplexLike } from "./complex.js";
import { NDArray, type NestedArray, type Shape } from "./ndarray.js";

export interface ArrayOptions {
  dtype?: DTypeLike;
}

/** Default dtype inference mirroring NumPy (DECISIONS D-004). */
export function inferDType(data: unknown): DType {
  let sawBool = false;
  let sawInt = false;
  let sawFloat = false;
  let sawComplex = false;
  const visit = (v: unknown): void => {
    if (Array.isArray(v)) {
      for (const x of v) visit(x);
      return;
    }
    if (typeof v === "boolean") sawBool = true;
    else if (typeof v === "bigint") sawInt = true;
    else if (typeof v === "number") {
      if (Number.isSafeInteger(v) && !Object.is(v, -0)) sawInt = true;
      else sawFloat = true;
    } else if (isComplexLike(v)) {
      sawComplex = true;
    } else {
      throw new ValueError(`unsupported array element type: ${typeof v}`);
    }
  };
  visit(data);
  if (sawComplex) return toDType("complex128");
  if (sawFloat) return toDType("float64");
  if (sawInt) return toDType("int64");
  if (sawBool) return toDType("bool");
  return toDType("float64"); // empty input
}

/**
 * Fast path for rectangular, all-`number` nested arrays (PLAN §80): one JS
 * pass collects the shape, the D-004 inference flags and the values into a
 * Float64Array, so the native side needs a single bulk conversion instead of
 * one Node-API call per element. Returns undefined for anything else
 * (booleans, bigints, ragged input, scalars), which falls back to fromNested
 * and its error reporting.
 */
function collectNumbers(
  data: NestedArray,
): { shape: number[]; flat: Float64Array; allInt: boolean } | undefined {
  if (!Array.isArray(data)) return undefined;
  const shape: number[] = [];
  let probe: unknown = data;
  while (Array.isArray(probe)) {
    shape.push(probe.length);
    if (shape.length > 64 || probe.length === 0) break;
    probe = probe[0];
  }
  const size = shape.reduce((p, d) => p * d, 1);
  const flat = new Float64Array(size);
  let pos = 0;
  let allInt = true;
  const last = shape.length - 1;
  const walk = (v: unknown, dim: number): boolean => {
    if (!Array.isArray(v) || v.length !== shape[dim]) return false;
    if (dim === last) {
      for (let i = 0; i < v.length; i++) {
        const x: unknown = v[i];
        if (typeof x !== "number") return false;
        if (allInt && (!Number.isSafeInteger(x) || Object.is(x, -0))) allInt = false;
        flat[pos++] = x;
      }
      return true;
    }
    for (let i = 0; i < v.length; i++) if (!walk(v[i], dim + 1)) return false;
    return true;
  };
  if (!walk(data, 0)) return undefined;
  return { shape, flat, allInt };
}

/** np.array: always copies (PLAN §11). */
export function array(data: NestedArray | NDArray, options: ArrayOptions = {}): NDArray {
  if (data instanceof NDArray) {
    return options.dtype === undefined ? data.copy() : data.astype(options.dtype);
  }
  const explicit = options.dtype === undefined ? undefined : toDType(options.dtype);
  if (explicit === undefined || !explicit.name.startsWith("complex")) {
    const fast = collectNumbers(data);
    if (fast !== undefined) {
      const dt =
        explicit ?? toDType(fast.flat.length === 0 || !fast.allInt ? "float64" : "int64");
      return wrapNative(() => NDArray._wrap(addon.fromFloat64(fast.flat, fast.shape, dt.name)));
    }
  }
  const dt = explicit ?? inferDType(data);
  return wrapNative(() => NDArray._wrap(addon.fromNested(data, dt.name)));
}

function normalizeShape(shape: Shape | number): number[] {
  return typeof shape === "number" ? [shape] : [...shape];
}

/** np.empty: uninitialized memory. */
export function empty(shape: Shape | number, options: ArrayOptions = {}): NDArray {
  const dt = toDType(options.dtype ?? "float64");
  return wrapNative(() => NDArray._wrap(addon.empty(normalizeShape(shape), dt.name)));
}

/** np.zeros. */
export function zeros(shape: Shape | number, options: ArrayOptions = {}): NDArray {
  const dt = toDType(options.dtype ?? "float64");
  return wrapNative(() => NDArray._wrap(addon.zeros(normalizeShape(shape), dt.name)));
}

/** np.ones. */
export function ones(shape: Shape | number, options: ArrayOptions = {}): NDArray {
  const dt = toDType(options.dtype ?? "float64");
  return wrapNative(() => NDArray._wrap(addon.ones(normalizeShape(shape), dt.name)));
}

/**
 * np.full. Without `dtype` the dtype is inferred from the value like
 * `np.array(value)` (D-004). With `dtype`, the value follows the D-009
 * conversion rules (out-of-range integers and NaN→int raise).
 */
export function full(
  shape: Shape | number,
  fillValue: number | boolean | bigint | ComplexLike,
  options: ArrayOptions = {},
): NDArray {
  const value = array(fillValue, options);
  return wrapNative(() => NDArray._wrap(addon.full(normalizeShape(shape), value._native)));
}

/** np.zerosLike / onesLike / fullLike / emptyLike: same shape, dtype defaults to the input's. */
export function zerosLike(a: NDArray, options: ArrayOptions = {}): NDArray {
  return zeros(a.shape, { dtype: options.dtype ?? a.dtype });
}
export function onesLike(a: NDArray, options: ArrayOptions = {}): NDArray {
  return ones(a.shape, { dtype: options.dtype ?? a.dtype });
}
export function emptyLike(a: NDArray, options: ArrayOptions = {}): NDArray {
  return empty(a.shape, { dtype: options.dtype ?? a.dtype });
}
export function fullLike(
  a: NDArray,
  fillValue: number | boolean | bigint | ComplexLike,
  options: ArrayOptions = {},
): NDArray {
  return full(a.shape, fillValue, { dtype: options.dtype ?? a.dtype });
}

/**
 * np.asarray: returns the input unchanged when it is already an NDArray of the
 * requested dtype (no copy), otherwise converts like np.array.
 */
export function asarray(data: NestedArray | NDArray, options: ArrayOptions = {}): NDArray {
  if (data instanceof NDArray) {
    if (options.dtype === undefined || toDType(options.dtype) === data.dtype) return data;
    return data.astype(options.dtype);
  }
  return array(data, options);
}

const isInt = (v: number): boolean => Number.isSafeInteger(v) && !Object.is(v, -0);

/**
 * np.arange(stop) / np.arange(start, stop, step?). Default dtype is int64 when
 * all arguments are integers, else float64 (D-004 inference).
 */
export function arange(
  startOrStop: number,
  stop?: number,
  step?: number,
  options: ArrayOptions = {},
): NDArray {
  const [s, e] = stop === undefined ? [0, startOrStop] : [startOrStop, stop];
  const st = step ?? 1;
  const dt =
    options.dtype !== undefined
      ? toDType(options.dtype)
      : toDType(isInt(s) && isInt(e) && isInt(st) ? "int64" : "float64");
  return wrapNative(() => NDArray._wrap(addon.arange(s, e, st, dt.name)));
}

export interface LinspaceOptions extends ArrayOptions {
  endpoint?: boolean;
}

/** np.linspace(start, stop, num = 50). */
export function linspace(
  start: number,
  stop: number,
  num = 50,
  options: LinspaceOptions = {},
): NDArray {
  const dt = toDType(options.dtype ?? "float64");
  return wrapNative(() =>
    NDArray._wrap(addon.linspace(start, stop, num, options.endpoint ?? true, dt.name)),
  );
}

export interface EyeOptions extends ArrayOptions {
  k?: number;
}

/** np.eye(N, M = N, {k = 0, dtype = float64}). */
export function eye(n: number, m?: number, options: EyeOptions = {}): NDArray {
  const dt = toDType(options.dtype ?? "float64");
  return wrapNative(() => NDArray._wrap(addon.eye(n, m ?? n, options.k ?? 0, dt.name)));
}

/** np.identity(n). */
export function identity(n: number, options: ArrayOptions = {}): NDArray {
  return eye(n, n, options);
}

const typedArrayDType = new Map<string, string>([
  ["Int8Array", "int8"],
  ["Uint8Array", "uint8"],
  ["Uint8ClampedArray", "uint8"],
  ["Int16Array", "int16"],
  ["Uint16Array", "uint16"],
  ["Int32Array", "int32"],
  ["Uint32Array", "uint32"],
  ["BigInt64Array", "int64"],
  ["BigUint64Array", "uint64"],
  ["Float32Array", "float32"],
  ["Float64Array", "float64"],
]);

/**
 * Creates an array from a TypedArray. Data is **copied**; zero-copy import is
 * deferred until ownership/lifetime rules are designed (PLAN §31). The dtype
 * defaults to the TypedArray element type; an explicit dtype reinterprets the
 * raw bytes (e.g. Uint16Array bits as float16). Node Buffers map to uint8.
 */
export function fromTypedArray(
  data: ArrayBufferView,
  shape?: Shape | number,
  options: ArrayOptions = {},
): NDArray {
  const inferred = typedArrayDType.get(data.constructor.name);
  const dt = toDType(options.dtype ?? inferred ?? "uint8");
  const target =
    shape === undefined ? [Math.floor(data.byteLength / dt.itemSize)] : normalizeShape(shape);
  return wrapNative(() => NDArray._wrap(addon.fromTypedArray(data, target, dt.name)));
}

/** NumPy may_share_memory: same buffer and overlapping non-empty byte extents (D-011). */
export function mayShareMemory(a: NDArray, b: NDArray): boolean {
  return a._native.sharesMemory(b._native);
}

/**
 * Bounds-checked equivalent of numpy.lib.stride_tricks.as_strided.
 * `offset` is the absolute byte offset into the underlying buffer
 * (defaults to the array's own offset). Strides are in bytes.
 */
export function asStrided(a: NDArray, shape: Shape, strides: Shape, offset?: number): NDArray {
  return wrapNative(() =>
    NDArray._wrap(a._native.view([...shape], [...strides], offset ?? a._native.offset())),
  );
}
