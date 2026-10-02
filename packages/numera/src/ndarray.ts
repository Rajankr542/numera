import { addon, type NativeIndexItem, type NativeNDArray } from "./addon.js";
import { dtype as toDType, type DType, type DTypeLike } from "./dtype.js";
import { DTypeError, IndexError, NotImplementedError, ValueError, wrapNative } from "./errors.js";
import type { Complex, ComplexLike } from "./complex.js";
import { p03native } from "./p03_native.js";

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

/** Memory order (NumPy `order=`, D-055); either case. */
export type MemoryOrder = "C" | "F" | "A" | "K" | "c" | "f" | "a" | "k";

/** Options for `reshape`/`ravel`/`flatten`. */
export interface OrderOptions {
  order?: MemoryOrder | null;
}

/** Options for `astype` (NumPy `order=`, `copy=`). */
export interface AstypeOptions {
  /** Result layout; default "K" (keep the input's). */
  order?: MemoryOrder | null;
  /** When false, returns this array itself if dtype and layout already match. */
  copy?: boolean;
  /** Casting rule checked before converting (NumPy `casting=`); default "unsafe" (D-060). */
  casting?: "no" | "equiv" | "safe" | "same_kind" | "unsafe";
}

/** A JS element value as read from an array (D-005, D-033). */
export type ScalarValue = number | boolean | Complex;

/**
 * Index for `a.flat.get` / `a.flat.set` (D-060): an integer, a slice tuple
 * `[start, stop, step?]` (as in D-015), an integer index array or a boolean mask.
 */
export type FlatIndex = number | SliceTuple | NDArray | readonly boolean[];

const orderArg = (o: { order?: MemoryOrder | null } | undefined): string | undefined =>
  o?.order === undefined || o.order === null ? undefined : o.order;

const isOrderOptions = (x: unknown): x is OrderOptions =>
  typeof x === "object" && x !== null && !Array.isArray(x);

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

// The TS wrapper of a native handle, so `base` returns the same object (D-060).
interface Owned {
  __ts?: NDArray;
}

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
    (handle as Owned).__ts = this;
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

  /**
   * Returns a view when possible, otherwise a copy. Supports one -1.
   * `reshape([2, 3], { order: "F" })` reads/writes in column-major index order.
   */
  reshape(shape: Shape | number, ...rest: (number | OrderOptions)[]): NDArray {
    const last = rest[rest.length - 1];
    const opts = isOrderOptions(last) ? last : undefined;
    const dims = (opts ? rest.slice(0, -1) : rest) as number[];
    const target = typeof shape === "number" ? [shape, ...dims] : [...shape];
    const order = orderArg(opts);
    if (order === undefined) return wrapNative(() => NDArray._wrap(this._native.reshape(target)));
    return wrapNative(() => NDArray._wrap(addon.reshapeOrder(this._native, target, order)));
  }

  /** Deep copy. `order`: "C" (default), "F", "A" or "K" (NumPy `a.copy(order)`). */
  copy(opts: OrderOptions = {}): NDArray {
    const order = orderArg(opts);
    if (order === undefined) return wrapNative(() => NDArray._wrap(this._native.copy()));
    return wrapNative(() => NDArray._wrap(addon.copyOrder(this._native, null, order)));
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

  /** 1-D view when possible, otherwise a copy (NumPy ravel; `order` C/F/A/K). */
  ravel(opts: OrderOptions = {}): NDArray {
    return wrapNative(() => NDArray._wrap(addon.ravel(this._native, orderArg(opts))));
  }

  /** 1-D copy, always (NumPy flatten; `order` C/F/A/K). */
  flatten(opts: OrderOptions = {}): NDArray {
    return wrapNative(() => NDArray._wrap(addon.flatten(this._native, orderArg(opts))));
  }

  /** Converted copy (NumPy `astype`, default casting "unsafe", default order "K"). */
  astype(dt: DTypeLike, opts: AstypeOptions = {}): NDArray {
    const target = toDType(dt);
    const casting = opts.casting ?? "unsafe";
    if (casting !== "unsafe" && !wrapNative(() => addon.canCast(this.dtype.name, target.name, casting))) {
      throw new DTypeError(
        `Cannot cast array data from dtype('${this.dtype.name}') to dtype('${target.name}') according to the rule '${casting}'`,
      );
    }
    const order = orderArg(opts) ?? "K";
    if (opts.copy === false && target === this.dtype) {
      const f = this.flags;
      const o = order.toUpperCase();
      const ok =
        o === "K" || (o === "C" && f.cContiguous) || (o === "F" && f.fContiguous) ||
        (o === "A" && (f.cContiguous || f.fContiguous));
      if (ok) return this;
    }
    return wrapNative(() => NDArray._wrap(addon.copyOrder(this._native, target.name, order)));
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

  /**
   * The array whose memory this one views (NumPy `a.base`), or `null` if this
   * array owns its data (D-060).
   */
  get base(): NDArray | null {
    const h = this._native.base();
    if (h === null) return null;
    return (h as Owned).__ts ?? NDArray._wrap(h);
  }

  /** Matrix transpose of the last two axes (NumPy `a.mT`), a view. */
  get mT(): NDArray {
    return wrapNative(() => NDArray._wrap(p03native.matrixTranspose(this._native)));
  }

  /** 1-D iterator/indexer over the elements in C order (NumPy `a.flat`, D-060). */
  get flat(): FlatIter {
    return new FlatIter(this);
  }
  /** `a.flat = v`: assigns `v` (repeated cyclically) to every element. */
  set flat(value: NDArray | NestedArray) {
    new FlatIter(this)._assign(null, value);
  }

  /** Sets every element to `value` (NumPy `a.fill`); cast to this dtype. */
  fill(value: number | boolean | bigint | ComplexLike | NDArray): void {
    wrapNative(() => {
      const v = value instanceof NDArray ? value._native : addon.fromNested(value, this.dtype.name);
      p03native.fill(this._native, v);
    });
  }

  /** Nested JS arrays, like `toArray()` (NumPy `a.tolist()`; D-005 applies). */
  tolist(): NestedArray {
    return this.toArray();
  }

  /** Raw bytes in C (default), F or A order as a new Uint8Array (NumPy `tobytes`). */
  tobytes(opts: OrderOptions = {}): Uint8Array {
    return wrapNative(() => p03native.tobytes(this._native, orderArg(opts)));
  }

  /**
   * New view of the same memory (NumPy `a.view(dtype)`). With a dtype of a
   * different item size the last axis is rescaled (D-060).
   */
  view(dt?: DTypeLike): NDArray {
    const target = dt === undefined ? this.dtype : toDType(dt);
    return wrapNative(() => NDArray._wrap(p03native.viewAs(this._native, target.name)));
  }

  /** Byte-swapped copy, or swaps in place and returns this array (NumPy `byteswap`). */
  byteswap(opts: { inplace?: boolean } = {}): NDArray {
    const inplace = opts.inplace ?? false;
    const r = wrapNative(() => p03native.byteswap(this._native, inplace));
    return inplace ? this : NDArray._wrap(r);
  }

  /** NumPy `setflags`: only `write` is supported (D-060). */
  setflags(opts: { write?: boolean | null; align?: boolean | null; uic?: boolean | null } = {}): void {
    if (opts.align != null || opts.uic != null) {
      throw new NotImplementedError("setflags supports only the write flag");
    }
    if (opts.write != null) {
      const w = opts.write;
      wrapNative(() => this._native.setWriteable(w));
    }
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

/**
 * NumPy `flatiter` (D-060): a 1-D, C-order view of an array's elements.
 * `get`/`set` take a number, a slice tuple, an integer index array or a boolean
 * mask of the array's size.
 */
export class FlatIter implements Iterable<ScalarValue> {
  /** The array being iterated (NumPy `flat.base`). */
  readonly base: NDArray;
  private pos = 0;

  /** @internal */
  constructor(base: NDArray) {
    this.base = base;
  }

  get length(): number {
    return this.base.size;
  }
  /** Flat position of the next element (NumPy `flat.index`). */
  get index(): number {
    return this.pos;
  }
  /** Multi-index of the next element (NumPy `flat.coords`). */
  get coords(): number[] {
    return unravel(this.pos, this.base.shape);
  }

  /** Flat positions (int64 array) selected by `index`; null for all elements. */
  private positions(index: FlatIndex): NDArray {
    const size = this.base.size;
    const flatIdx = (spec: IndexSpec): NDArray =>
      wrapNative(() => {
        const iota = NDArray._wrap(addon.arange(0, size, 1, "int64"));
        return NDArray._wrap(addon.getIndex(iota._native, encodeIndex([spec])));
      });
    if (index instanceof NDArray) {
      if (index.dtype.kind === "b") {
        if (index.ndim !== 1 || index.size !== size) {
          throw new IndexError(
            `boolean index did not match indexed flat iterator along axis 0; size of axis is ${size} but size of corresponding boolean axis is ${index.size}`,
          );
        }
        return flatIdx(index);
      }
      if (index.dtype.kind !== "i" && index.dtype.kind !== "u") {
        throw new IndexError("only integers, slices, and integer or boolean arrays are valid flat indices");
      }
      return flatIdx(index);
    }
    if (Array.isArray(index) && index.length > 0 && index.every((v) => typeof v === "boolean")) {
      return this.positions(NDArray._wrap(addon.fromNested(index, "bool")));
    }
    if (isSliceTuple(index) || (Array.isArray(index) && index.length === 0)) {
      return flatIdx(index as SliceTuple);
    }
    if (typeof index === "number") {
      if (!Number.isInteger(index)) {
        throw new IndexError("only integers, slices, and integer or boolean arrays are valid flat indices");
      }
      if (index < -size || index >= size) {
        throw new IndexError(`index ${index} is out of bounds for size ${size}`);
      }
      return NDArray._wrap(addon.fromNested(index < 0 ? index + size : index, "int64"));
    }
    throw new IndexError("only integers, slices, and integer or boolean arrays are valid flat indices");
  }

  /**
   * `a.flat[i]`: a JS scalar for a number, otherwise a new array (1-D for a
   * slice or mask, the index's shape for an integer array).
   */
  get(index: number): ScalarValue;
  get(index: Exclude<FlatIndex, number>): NDArray;
  get(index: FlatIndex): ScalarValue | NDArray {
    const pos = this.positions(index);
    const flat = this.base.ravel();
    if (typeof index === "number") return flat.item(pos.item() as number);
    return wrapNative(() => NDArray._wrap(addon.take(flat._native, pos._native, null)));
  }

  /** `a.flat[i] = value`; values repeat cyclically over the selection (NumPy). */
  set(index: FlatIndex, value: NDArray | NestedArray | ComplexLike): void {
    this._assign(this.positions(index), value);
  }

  /** @internal `pos` null selects every element. */
  _assign(pos: NDArray | null, value: NDArray | NestedArray | ComplexLike): void {
    wrapNative(() => {
      const v = value instanceof NDArray ? value._native : addon.fromNested(value, this.base.dtype.name);
      p03native.flatAssign(this.base._native, pos === null ? null : pos._native, v);
    });
  }

  /** 1-D copy of all elements (NumPy `flat.copy()`). */
  copy(): NDArray {
    return this.base.flatten();
  }

  toArray(): NestedArray {
    return this.base.flatten().toArray();
  }

  next(): IteratorResult<ScalarValue> {
    if (this.pos >= this.base.size) return { value: undefined, done: true };
    return { value: this.base.item(...unravel(this.pos++, this.base.shape)), done: false };
  }

  [Symbol.iterator](): Iterator<ScalarValue> {
    return this;
  }
}

function unravel(flat: number, shape: readonly number[]): number[] {
  const out = new Array<number>(shape.length).fill(0);
  for (let d = shape.length - 1; d >= 0; d--) {
    const n = shape[d]!;
    out[d] = n === 0 ? 0 : flat % n;
    flat = n === 0 ? 0 : Math.floor(flat / n);
  }
  return out;
}
