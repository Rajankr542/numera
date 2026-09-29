import { addon, type NativeNDArray } from "./addon.js";
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

  /** Permuted-axes view (reverses axes by default). */
  transpose(...axes: number[] | [readonly number[]]): NDArray {
    const list = (axes.length === 1 && Array.isArray(axes[0]) ? axes[0] : axes) as number[];
    return wrapNative(() => NDArray._wrap(addon.transpose(this._native, [...list])));
  }

  /** Transposed view (`a.T`). */
  get T(): NDArray {
    return this.transpose();
  }

  /** Removes size-1 axes (all, or the given ones). Always a view. */
  squeeze(axis?: number | readonly number[]): NDArray {
    const axes = axis === undefined ? undefined : typeof axis === "number" ? [axis] : [...axis];
    return wrapNative(() => NDArray._wrap(addon.squeeze(this._native, axes)));
  }

  swapAxes(axis1: number, axis2: number): NDArray {
    return wrapNative(() => NDArray._wrap(addon.swapaxes(this._native, axis1, axis2)));
  }

  /** 1-D view when possible, otherwise a copy (NumPy ravel). */
  ravel(): NDArray {
    return wrapNative(() => NDArray._wrap(addon.ravel(this._native)));
  }

  /** 1-D copy, always (NumPy flatten). */
  flatten(): NDArray {
    return wrapNative(() => NDArray._wrap(addon.flatten(this._native)));
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
