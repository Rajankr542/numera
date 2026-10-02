// P5 packbits / unpackbits (D-084).
import { nativeModule, type NativeNDArray } from "./addon.js";
import { asarray } from "./creation.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import type { ArrayLike } from "./ufunc.js";

interface P05BitsNative {
  packbits(a: NativeNDArray, axis: number | null, little: boolean): NativeNDArray;
  unpackbits(a: NativeNDArray, axis: number | null, count: number | null, little: boolean): NativeNDArray;
}
const native = nativeModule<P05BitsNative>("p05");

export type BitOrder = "big" | "little";

export interface PackbitsOptions {
  /** Axis to pack along; `null`/omitted packs the flattened array. */
  axis?: number | null;
  /** Bit order inside each byte (default `"big"`). */
  bitorder?: BitOrder;
}

export interface UnpackbitsOptions extends PackbitsOptions {
  /** Number of bits to keep along the axis (negative drops from the end). */
  count?: number | null;
}

function isLittle(order: string | undefined, message: string): boolean {
  if (order === undefined || order === "big") return false;
  if (order === "little") return true;
  throw new ValueError(message);
}

/** NumPy packbits: packs truthy elements into the bits of a uint8 array. */
export function packbits(a: ArrayLike, opts: PackbitsOptions = {}): NDArray {
  const little = isLittle(opts.bitorder, "'order' must be either 'little' or 'big'");
  const x = asarray(a);
  return wrapNative(() => NDArray._wrap(native.packbits(x._native, opts.axis ?? null, little)));
}

/** NumPy unpackbits: expands each uint8 element into 8 bits (0/1, uint8). */
export function unpackbits(a: ArrayLike, opts: UnpackbitsOptions = {}): NDArray {
  const little = isLittle(opts.bitorder, "'order' must begin with 'l' or 'b'");
  const x = asarray(a);
  return wrapNative(() =>
    NDArray._wrap(native.unpackbits(x._native, opts.axis ?? null, opts.count ?? null, little)),
  );
}
