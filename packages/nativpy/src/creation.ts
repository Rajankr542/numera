import { addon } from "./addon.js";
import { dtype as toDType, type DType, type DTypeLike } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray, type Shape } from "./ndarray.js";

export interface ArrayOptions {
  dtype?: DTypeLike;
}

/** Default dtype inference mirroring NumPy (DECISIONS D-004). */
export function inferDType(data: unknown): DType {
  let sawBool = false;
  let sawInt = false;
  let sawFloat = false;
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
    } else {
      throw new ValueError(`unsupported array element type: ${typeof v}`);
    }
  };
  visit(data);
  if (sawFloat) return toDType("float64");
  if (sawInt) return toDType("int64");
  if (sawBool) return toDType("bool");
  return toDType("float64"); // empty input
}

/** np.array: always copies (PLAN §11). */
export function array(data: NestedArray | NDArray, options: ArrayOptions = {}): NDArray {
  if (data instanceof NDArray) {
    return options.dtype === undefined ? data.copy() : data.astype(options.dtype);
  }
  const dt = options.dtype === undefined ? inferDType(data) : toDType(options.dtype);
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
