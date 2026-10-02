import { array } from "./creation.js";
import { IndexError, ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray } from "./ndarray.js";
import { concatenate } from "./p06_join.js";
import { native } from "./p06_native.js";
import { asArr, checkInt, normAxis, wrap, type ArrayLike } from "./p06_util.js";
import { moveAxis } from "./shape.js";

/** append / insert / delete / resize / trimZeros (P6, D-093). */

/** A Python slice `start:stop:step` (JS has no slice syntax). */
export interface SliceObject {
  start?: number | null;
  stop?: number | null;
  step?: number | null;
}

/** Index argument of insert/delete: integer, integer or boolean list/array, or a slice object. */
export type EditIndex = number | readonly (number | boolean)[] | NDArray | SliceObject;

const isSlice = (x: unknown): x is SliceObject =>
  typeof x === "object" && x !== null && !Array.isArray(x) && !(x instanceof NDArray);

// Python range(*slice.indices(n)) as a list.
function sliceRange(s: SliceObject, n: number): number[] {
  const step = s.step ?? 1;
  checkInt(step, "slice step");
  if (step === 0) throw new ValueError("slice step cannot be zero");
  const clamp = (v: number | null | undefined, def: number): number => {
    if (v === undefined || v === null) return def;
    checkInt(v, "slice index");
    if (v < 0) v += n;
    return step > 0 ? Math.min(Math.max(v, 0), n) : Math.min(Math.max(v, -1), n - 1);
  };
  const start = clamp(s.start, step > 0 ? 0 : n - 1);
  const stop = clamp(s.stop, step > 0 ? n : -1);
  const out: number[] = [];
  for (let i = start; step > 0 ? i < stop : i > stop; i += step) out.push(i);
  return out;
}

// Flattens (axis null) and normalises the axis like NumPy insert/delete.
function prepare(arr: ArrayLike, axis: number | null | undefined): [NDArray, number] {
  let a = asArr(arr);
  if (axis === undefined || axis === null) {
    if (a.ndim !== 1) a = a.ravel();
    return [a, a.ndim - 1];
  }
  return [a, normAxis(axis, a.ndim)];
}

function indexArray(obj: Exclude<EditIndex, SliceObject>): NDArray {
  const ix = obj instanceof NDArray ? obj : typeof obj === "number" ? array(obj) : array(obj as NestedArray);
  if (!(obj instanceof NDArray) && ix.size === 0) return ix.astype("int64");
  return ix;
}

function intValues(ix: NDArray): number[] {
  if (ix.dtype.kind !== "i" && ix.dtype.kind !== "u") {
    throw new IndexError("arrays used as indices must be of integer (or boolean) type");
  }
  return (ix.ravel().toArray() as number[]).map(Number);
}

const oob = (i: number, axis: number, n: number): IndexError =>
  new IndexError(`index ${i} is out of bounds for axis ${axis} with size ${n}`);

/** NumPy insert: inserts `values` before the given indices (flattened when `axis` is omitted). */
export function insert(arr: ArrayLike, obj: EditIndex, values: ArrayLike, axis?: number | null): NDArray {
  const [a, ax] = prepare(arr, axis);
  if (a.ndim === 0) throw new IndexError("axis 0 is out of bounds for array of dimension 0");
  const n = a.shape[ax]!;
  let indices: number[];
  let scalarObj = false;
  if (isSlice(obj)) {
    indices = sliceRange(obj, n);
  } else {
    const ix = indexArray(obj);
    if (ix.dtype.kind === "b") {
      if (ix.ndim !== 1) throw new ValueError("boolean array argument obj to insert must be one dimensional");
      indices = (ix.toArray() as boolean[]).flatMap((b, i) => (b ? [i] : []));
    } else {
      if (ix.ndim > 1) {
        throw new ValueError("index array argument obj to insert must be one dimensional or scalar");
      }
      indices = intValues(ix);
      scalarObj = ix.ndim === 0;
    }
  }
  return wrapNative(() => {
    let v = asArr(values);
    if (v.dtype.name !== a.dtype.name) v = v.astype(a.dtype);
    if (indices.length === 1 && !isSlice(obj)) {
      let index = indices[0]!;
      if (index < -n || index > n) throw oob(index, ax, n);
      if (index < 0) index += n;
      while (v.ndim < a.ndim) v = v.reshape([1, ...v.shape]);
      if (scalarObj) v = moveAxis(v, 0, ax);
      const numnew = v.ndim === 0 ? 1 : v.shape[ax]!;
      const pos = Array.from({ length: numnew }, (_, k) => index + k);
      return wrap(native.insertAlong(a._native, ax, pos, v._native));
    }
    const adj = indices.map((i) => {
      if (i < -n || i > n) throw oob(i, ax, n + indices.length);
      return i < 0 ? i + n : i;
    });
    const order = adj.map((_, k) => k).sort((p, q) => adj[p]! - adj[q]! || p - q);
    order.forEach((k, rank) => {
      adj[k] = adj[k]! + rank;
    });
    return wrap(native.insertAlong(a._native, ax, adj, v._native));
  });
}

/** NumPy delete (exported as `np.delete`): removes the given indices along `axis`. */
export function del(arr: ArrayLike, obj: EditIndex, axis?: number | null): NDArray {
  const [a, ax] = prepare(arr, axis);
  if (a.ndim === 0) throw new IndexError("axis 0 is out of bounds for array of dimension 0");
  const n = a.shape[ax]!;
  const keep = new Array<boolean>(n).fill(true);
  if (isSlice(obj)) {
    for (const i of sliceRange(obj, n)) keep[i] = false;
  } else {
    const ix = indexArray(obj);
    if (ix.dtype.kind === "b") {
      if (ix.ndim !== 1 || ix.shape[0] !== n) {
        throw new ValueError(
          `boolean array argument obj to delete must be one dimensional and match the axis length of ${n}`,
        );
      }
      (ix.toArray() as boolean[]).forEach((b, i) => {
        keep[i] = !b;
      });
    } else {
      for (const i of intValues(ix)) {
        if (i < -n || i >= n) throw oob(i, ax, n);
        keep[i < 0 ? i + n : i] = false;
      }
    }
  }
  return wrapNative(() => wrap(native.deleteAlong(a._native, ax, keep)));
}

export { del as delete };

/** NumPy append: concatenates `values` to `arr` (both flattened when `axis` is omitted). */
export function append(arr: ArrayLike, values: ArrayLike, axis?: number | null): NDArray {
  if (axis === undefined || axis === null) return concatenate([asArr(arr).ravel(), asArr(values).ravel()], 0);
  return concatenate([arr, values], axis);
}

function shapeArg(s: number | readonly number[]): number[] {
  const list = typeof s === "number" ? [s] : [...s];
  list.forEach((d) => checkInt(d, "new_shape"));
  return list;
}

/** NumPy resize (function): new array of `newShape` filled by cycling `a`'s data (zeros if empty). */
export function resize(a: ArrayLike, newShape: number | readonly number[]): NDArray {
  const s = shapeArg(newShape);
  return wrapNative(() => wrap(native.resize(asArr(a)._native, s)));
}

type TrimMode = "fb" | "bf" | "f" | "b" | "FB" | "BF" | "F" | "B";

/** NumPy trim_zeros: view trimmed to the nonzero bounding box on the selected axes. */
export function trimZeros(filt: ArrayLike, trim: TrimMode | string = "fb", axis?: number | readonly number[] | null): NDArray {
  const a = asArr(filt);
  const t = trim.toLowerCase();
  if (!["fb", "bf", "f", "b"].includes(t)) {
    throw new ValueError(`unexpected character(s) in \`trim\`: '${trim}'`);
  }
  const sel = new Array<boolean>(a.ndim).fill(axis === undefined || axis === null);
  if (axis !== undefined && axis !== null) {
    for (const ax of typeof axis === "number" ? [axis] : axis) {
      const d = normAxis(ax, a.ndim);
      if (sel[d]) throw new ValueError("repeated axis in `axis` argument");
      sel[d] = true;
    }
  }
  if (!sel.some(Boolean)) return a;
  return wrapNative(() => wrap(native.trimZeros(a._native, t.includes("f"), t.includes("b"), sel)));
}

export interface ResizeOptions {
  /** Refuse to resize while other arrays share the buffer (default true). */
  refcheck?: boolean;
}

declare module "./ndarray.js" {
  interface NDArray {
    /**
     * NumPy ndarray.resize: resizes this array in place (D-093). The data is
     * truncated or zero-extended in memory order; returns undefined.
     */
    resize(newShape: number | readonly number[], options?: ResizeOptions): void;
    resize(...dims: number[]): void;
  }
}

NDArray.prototype.resize = function (this: NDArray, ...args: unknown[]): void {
  let options: ResizeOptions = {};
  const last = args[args.length - 1];
  if (typeof last === "object" && last !== null && !Array.isArray(last)) {
    options = last as ResizeOptions;
    args = args.slice(0, -1);
  }
  const shape =
    args.length === 1 && Array.isArray(args[0]) ? shapeArg(args[0] as number[]) : shapeArg(args as number[]);
  if (shape.some((d) => d < 0)) throw new ValueError("negative dimensions not allowed");
  wrapNative(() => {
    const flags = this.flags;
    if (!flags.cContiguous && !flags.fContiguous) {
      throw new ValueError("resize only works on single-segment arrays");
    }
    if (!flags.ownData) throw new ValueError("cannot resize this array: it does not own its data");
    if ((options.refcheck ?? true) && native.bufferRefs(this._native) > 1) {
      throw new ValueError(
        "cannot resize an array that references or is referenced\n" +
          "by another array in this way.\nUse the np.resize function or refcheck=False",
      );
    }
    const h = native.resizeInplace(this._native, shape);
    (this as unknown as { _native: NDArray["_native"] })._native = h;
  });
};
