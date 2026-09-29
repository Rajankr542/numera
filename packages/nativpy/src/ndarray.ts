import type { NativeNDArray } from "./addon.js";
import { dtype as toDType, type DType, type DTypeLike } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";

export type Shape = readonly number[];
export type NestedArray = number | boolean | bigint | readonly NestedArray[];

export interface ArrayFlags {
  readonly cContiguous: boolean;
  readonly fContiguous: boolean;
  readonly ownData: boolean;
  readonly writeable: boolean;
}

const internal = Symbol("nativpy.internal");

/**
 * N-dimensional array backed by native memory (PLAN §8, D-003).
 * Strides are in bytes, like `numpy.ndarray.strides`.
 */
export class NDArray {
  /** @internal */
  readonly _native: NativeNDArray;

  /** @internal Use np.array / np.zeros / np.empty to create arrays. */
  constructor(token: typeof internal, handle: NativeNDArray) {
    if (token !== internal) {
      throw new ValueError("NDArray cannot be constructed directly; use np.array()");
    }
    this._native = handle;
  }

  /** @internal */
  static _wrap(handle: NativeNDArray): NDArray {
    return new NDArray(internal, handle);
  }

  get shape(): number[] {
    return this._native.shape();
  }
  get strides(): number[] {
    return this._native.strides();
  }
  get ndim(): number {
    return this._native.ndim();
  }
  get size(): number {
    return this._native.size();
  }
  get dtype(): DType {
    return toDType(this._native.dtype());
  }
  get itemSize(): number {
    return this._native.itemsize();
  }
  get nbytes(): number {
    return this._native.nbytes();
  }
  get flags(): ArrayFlags {
    return this._native.flags();
  }

  /** Returns a view when the array is C-contiguous, otherwise a copy. Supports one -1. */
  reshape(shape: Shape | number, ...rest: number[]): NDArray {
    const target = typeof shape === "number" ? [shape, ...rest] : [...shape];
    return wrapNative(() => NDArray._wrap(this._native.reshape(target)));
  }

  /** C-contiguous deep copy. */
  copy(): NDArray {
    return wrapNative(() => NDArray._wrap(this._native.copy()));
  }

  /** Converted C-contiguous copy (NumPy `astype`, unsafe casting). */
  astype(dt: DTypeLike): NDArray {
    const target = toDType(dt);
    return wrapNative(() => NDArray._wrap(this._native.astype(target.name)));
  }

  /** Scalar at a full integer index (negative allowed). With no index, the array must have size 1. */
  item(...index: number[]): number | boolean {
    return wrapNative(() => {
      if (index.length === 0) {
        if (this.size !== 1) {
          throw new ValueError("can only convert an array of size 1 to a scalar");
        }
        return this._native.getItem(new Array<number>(this.ndim).fill(0));
      }
      return this._native.getItem(index);
    });
  }

  /** Nested JS arrays (a conversion, not the internal representation — PLAN §30). */
  toArray(): NestedArray {
    return wrapNative(() => this._native.toList() as NestedArray);
  }

  /** New C-order TypedArray copy of the data (D-005, D-010). */
  toTypedArray(): ArrayBufferView {
    return wrapNative(() => this._native.toTypedArray());
  }

  toString(): string {
    return `array(${JSON.stringify(this.toArray())}, dtype=${this.dtype.name})`;
  }
}
