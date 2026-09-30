import { addon, type NativeIndexItem, type NativeNDArray } from "./addon.js";
import { dtype as toDType, type DType, type DTypeLike } from "./dtype.js";
import { IndexError, ValueError, wrapNative } from "./errors.js";
import type { Complex, ComplexLike } from "./complex.js";

export type Shape = readonly number[];
export type NestedArray = number | boolean | bigint | ComplexLike | readonly NestedArray[];

/**
 * Per-axis index spec (PLAN §13, D-015), mirroring NumPy's `a[...]` elements:
 * - `number`                 integer index (drops the axis; negative allowed)
 * - `null`                   full slice `:`
 * - `[start, stop, step?]`   slice; entries may be `null` (Python `None`)
 * - `np.ellipsis` (`"..."`)  ellipsis
 * - `np.newaxis`             new axis of length 1
 * - `boolean`                0-d boolean index (NumPy `a[True]`)
 * - `NDArray`                integer or boolean array (advanced indexing → copy)
 */
export const newaxis = "newaxis" as const;
export const ellipsis = "..." as const;
export type SliceTuple =
  | readonly []
  | readonly [number | null]
  | readonly [number | null, number | null]
  | readonly [number | null, number | null, number | null];
export type IndexSpec =
  | number
  | boolean
  | null
  | SliceTuple
  | typeof newaxis
  | typeof ellipsis
  | NDArray;

function encodeIndex(specs: readonly IndexSpec[]): NativeIndexItem[] {
  return specs.map((spec): NativeIndexItem => {
    if (spec === null) return {};
    if (typeof spec === "number" || typeof spec === "boolean") return spec;
    if (spec === ellipsis) return "...";
    if (spec === newaxis) return null;
    if (spec instanceof NDArray) return spec._native;
    if (Array.isArray(spec)) {
      if (spec.length > 3) throw new IndexError("slice tuples take at most [start, stop, step]");
      const [start, stop, step] = spec as readonly (number | null | undefined)[];
      return { start: start ?? null, stop: stop ?? null, step: step ?? null };
    }
    throw new IndexError(
      "only integers, slices, newaxis, ellipsis, and integer or boolean arrays are valid indices",
    );
  });
}

function isSliceTuple(x: unknown): x is SliceTuple {
  return (
    Array.isArray(x) &&
    x.length >= 1 &&
    x.length <= 3 &&
    x.every((v) => v === null || typeof v === "number")
  );
}

export interface ArrayFlags {
  readonly cContiguous: boolean;
  readonly fContiguous: boolean;
  readonly ownData: boolean;
  readonly writeable: boolean;
}

/** Options accepted by the NDArray reduction methods (D-017). */
export interface MethodReduceOptions {
  axis?: number | readonly number[] | null;
  keepdims?: boolean;
  dtype?: DTypeLike | null;
  initial?: number | null;
  ddof?: number;
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

  /** Real part (NumPy `a.real`): a view; the array itself for real dtypes (D-033). */
  get real(): NDArray {
    return wrapNative(() => NDArray._wrap(addon.complexPart(this._native, false)));
  }

  /** Imaginary part (NumPy `a.imag`): a view; read-only zeros for real dtypes (D-033). */
  get imag(): NDArray {
    return wrapNative(() => NDArray._wrap(addon.complexPart(this._native, true)));
  }

  /** Complex conjugate (NumPy `a.conj()`); a copy for real dtypes. */
  conj(): NDArray {
    return wrapNative(() => NDArray._wrap(addon.unary("conjugate", this._native)));
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

  // ---- reductions (M7, D-017); see np.sum etc. in reduce.ts ----
  private reduce(op: string, opts: MethodReduceOptions): NDArray {
    const axis = opts.axis ?? null;
    return wrapNative(() =>
      NDArray._wrap(
        addon.reduce(op, this._native, {
          axis: axis === null ? null : typeof axis === "number" ? [axis] : [...axis],
          keepdims: opts.keepdims ?? false,
          dtype: opts.dtype == null ? null : toDType(opts.dtype).name,
          initial: opts.initial ?? null,
          ddof: opts.ddof ?? 0,
        }),
      ),
    );
  }
  sum(opts: MethodReduceOptions = {}): NDArray {
    return this.reduce("sum", opts);
  }
  prod(opts: MethodReduceOptions = {}): NDArray {
    return this.reduce("prod", opts);
  }
  min(opts: Omit<MethodReduceOptions, "dtype" | "ddof"> = {}): NDArray {
    return this.reduce("min", opts);
  }
  max(opts: Omit<MethodReduceOptions, "dtype" | "ddof"> = {}): NDArray {
    return this.reduce("max", opts);
  }
  mean(opts: Omit<MethodReduceOptions, "initial" | "ddof"> = {}): NDArray {
    return this.reduce("mean", opts);
  }
  var(opts: Omit<MethodReduceOptions, "initial"> = {}): NDArray {
    return this.reduce("var", opts);
  }
  std(opts: Omit<MethodReduceOptions, "initial"> = {}): NDArray {
    return this.reduce("std", opts);
  }
  argmin(opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray {
    return wrapNative(() =>
      NDArray._wrap(addon.argReduce(false, this._native, opts.axis ?? null, opts.keepdims ?? false)),
    );
  }
  argmax(opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray {
    return wrapNative(() =>
      NDArray._wrap(addon.argReduce(true, this._native, opts.axis ?? null, opts.keepdims ?? false)),
    );
  }

  /** Scalar at a full integer index (negative allowed). With no index, the array must have size 1. */
  item(...index: number[]): number | boolean | Complex {
    return wrapNative(() => {
      if (index.length === 0) {
        if (this.size !== 1) {
          throw new ValueError("can only convert an array of size 1 to a scalar");
        }
        return this._native.getItem(new Array<number>(this.ndim).fill(0)) as number | boolean | Complex;
      }
      return this._native.getItem(index) as number | boolean | Complex;
    });
  }

  /** Nested JS arrays (a conversion, not the internal representation — PLAN §30). */
  toArray(): NestedArray {
    return wrapNative(() => this._native.toList() as NestedArray);
  }

  /**
   * NumPy `a[i, j, ...]` with one spec per argument (D-015). Always returns an
   * NDArray (0-d for a full integer index; use `item()` for a JS scalar).
   * Basic indices return views; any array index returns a copy.
   */
  get(...index: IndexSpec[]): NDArray {
    return wrapNative(() => NDArray._wrap(addon.getIndex(this._native, encodeIndex(index))));
  }

  /**
   * NumPy `a[index]` (PLAN §13). `index` is a list of per-axis specs, e.g.
   * `a.slice([null, [1, 3]])` is `a[:, 1:3]`. A flat list of only numbers/nulls
   * (length 1–3) is a single slice tuple: `a.slice([0, 5])` is `a[0:5]` (D-015).
   */
  slice(index: readonly IndexSpec[] | SliceTuple): NDArray {
    return this.get(...(isSliceTuple(index) ? [index] : (index as IndexSpec[])));
  }

  /**
   * NumPy `a[index] = value`. `index` is a list of per-axis specs (like the
   * arguments of `get`) or a single non-tuple spec. `value` broadcasts to the
   * selection and is cast unsafely to this array's dtype. Writes through
   * views; for repeated advanced indices the last value wins.
   */
  set(index: readonly IndexSpec[] | Exclude<IndexSpec, SliceTuple>, value: NDArray | NestedArray): void {
    const specs = Array.isArray(index) ? (index as IndexSpec[]) : [index as IndexSpec];
    wrapNative(() => {
      const v = value instanceof NDArray ? value._native : addon.fromNested(value, this.dtype.name);
      addon.setIndex(this._native, encodeIndex(specs), v);
    });
  }

  /** New C-order TypedArray copy of the data (D-005, D-010). */
  toTypedArray(): ArrayBufferView {
    return wrapNative(() => this._native.toTypedArray());
  }

  toString(): string {
    return `array(${JSON.stringify(this.toArray())}, dtype=${this.dtype.name})`;
  }
}
