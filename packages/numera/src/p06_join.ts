import { array } from "./creation.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { native } from "./p06_native.js";
import { asArr, checkInt, isOpts, joinArgs, wrap, wrapList, type ArrayLike, type JoinOptions } from "./p06_util.js";

/** Join, split and dimension helpers (P6, D-090). */

export type { JoinOptions } from "./p06_util.js";

export interface ConcatenateOptions extends JoinOptions {
  /** Join axis (default 0); `null` flattens every input first. */
  axis?: number | null;
}

export interface StackOptions extends JoinOptions {
  /** Position of the new axis in the result (default 0). */
  axis?: number;
}

type Sequence = readonly ArrayLike[] | NDArray;

const toList = (arrays: Sequence): NDArray[] =>
  arrays instanceof NDArray ? unstack(arrays) : arrays.map(asArr);

function joined(out: NDArray | null | undefined, h: NDArray["_native"]): NDArray {
  return out ? out : wrap(h);
}

/**
 * NumPy concatenate: joins arrays along an existing axis.
 * `concatenate([a, b], 1)` or `concatenate([a, b], { axis: null, dtype, casting, out })`.
 */
export function concatenate(
  arrays: Sequence,
  axis?: number | null | ConcatenateOptions,
  options: JoinOptions = {},
): NDArray {
  const opts: ConcatenateOptions = isOpts(axis) ? axis : { ...options, axis: axis as number | null | undefined };
  const list = toList(arrays);
  const ax = opts.axis === undefined ? 0 : opts.axis;
  if (ax !== null) checkInt(ax, "axis");
  return wrapNative(() =>
    joined(opts.out, native.concatenate(list.map((a) => a._native), ax, ...joinArgs(opts))),
  );
}

/** NumPy concat (array API name of concatenate). */
export const concat = concatenate;

/** NumPy stack: joins same-shaped arrays along a new axis. */
export function stack(arrays: Sequence, axis?: number | StackOptions, options: JoinOptions = {}): NDArray {
  const opts: StackOptions = isOpts(axis) ? axis : { ...options, axis: axis as number | undefined };
  const list = toList(arrays);
  const ax = checkInt(opts.axis ?? 0, "axis");
  return wrapNative(() => joined(opts.out, native.stack(list.map((a) => a._native), ax, ...joinArgs(opts))));
}

/** Options for vstack/hstack (NumPy `dtype=`, `casting=`). */
export type VHStackOptions = Omit<JoinOptions, "out">;

/** NumPy vstack: stacks rows (inputs promoted to at least 2-d). */
export function vstack(arrays: Sequence, options: VHStackOptions = {}): NDArray {
  return concatenate(toList(arrays).map((a) => atleastNd(a, 2)), { ...options, axis: 0 });
}

/** NumPy hstack: joins along axis 1 (axis 0 for 1-d inputs). */
export function hstack(arrays: Sequence, options: VHStackOptions = {}): NDArray {
  const list = toList(arrays).map((a) => atleastNd(a, 1));
  const axis = list.length > 0 && list[0]!.ndim === 1 ? 0 : 1;
  return concatenate(list, { ...options, axis });
}

/** NumPy dstack: joins along axis 2 (inputs promoted to at least 3-d). */
export function dstack(arrays: Sequence): NDArray {
  return concatenate(toList(arrays).map(atleast3), 2);
}

/** NumPy column_stack: 1-d inputs become columns, then joins along axis 1. */
export function columnStack(arrays: Sequence): NDArray {
  return concatenate(
    toList(arrays).map((a) => (a.ndim < 2 ? atleastNd(a, 2).transpose() : a)),
    1,
  );
}

/** Nested block layout for `np.block`: leaves are arrays or scalars. */
export type BlockArg = NDArray | number | boolean | bigint | readonly BlockArg[];

function blockIndex(index: readonly (number | null)[]): string {
  return "arrays" + index.filter((i) => i !== null).map((i) => `[${i}]`).join("");
}

// NumPy _block_check_depths_match: [bottom index, max leaf ndim].
function blockDepths(arrays: BlockArg, parent: (number | null)[]): [(number | null)[], number] {
  if (Array.isArray(arrays)) {
    if (arrays.length === 0) return [[...parent, null], 0];
    let first: (number | null)[] | undefined;
    let maxNd = 0;
    (arrays as readonly BlockArg[]).forEach((x, i) => {
      const [index, nd] = blockDepths(x, [...parent, i]);
      maxNd = Math.max(maxNd, nd);
      if (first === undefined) {
        first = index;
        return;
      }
      if (index.length !== first.length) {
        throw new ValueError(
          `List depths are mismatched. First element was at depth ${first.length}, ` +
            `but there is an element at depth ${index.length} (${blockIndex(index)})`,
        );
      }
      if (index[index.length - 1] === null) first = index;
    });
    return [first!, maxNd];
  }
  return [parent, arrays instanceof NDArray ? arrays.ndim : 0];
}

function blockRec(arrays: BlockArg, maxDepth: number, ndim: number, depth: number): NDArray {
  if (depth < maxDepth) {
    const parts = (arrays as readonly BlockArg[]).map((x) => blockRec(x, maxDepth, ndim, depth + 1));
    return concatenate(parts, -(maxDepth - depth));
  }
  return atleastNd(arrays instanceof NDArray ? arrays : array(arrays as number), ndim);
}

/** NumPy block: assembles an array from nested lists of blocks (D-090). */
export function block(arrays: BlockArg): NDArray {
  const [bottom, arrNd] = blockDepths(arrays, []);
  if (bottom.length > 0 && bottom[bottom.length - 1] === null) {
    throw new ValueError(`List at ${blockIndex(bottom)} cannot be empty`);
  }
  const listNd = bottom.length;
  const r = blockRec(arrays, listNd, Math.max(arrNd, listNd), 0);
  return listNd === 0 ? r.copy() : r;
}

/** NumPy unstack: views along `axis` (default 0). */
export function unstack(x: ArrayLike, axis: number | { axis?: number } = 0): NDArray[] {
  const ax = checkInt(typeof axis === "number" ? axis : (axis.axis ?? 0), "axis");
  const a = asArr(x);
  return wrapNative(() => wrapList(native.unstack(a._native, ax)));
}

// ---- split ----

function splitImpl(a: NDArray, sections: number | readonly number[] | NDArray, axis: number, equal: boolean): NDArray[] {
  checkInt(axis, "axis");
  if (typeof sections === "number") {
    checkInt(sections, "sections");
    return wrapNative(() => wrapList(native.splitSections(a._native, sections, axis, equal)));
  }
  const idx = sections instanceof NDArray ? (sections.toArray() as number[]) : [...sections];
  idx.forEach((v) => checkInt(v, "indices"));
  return wrapNative(() => wrapList(native.splitAt(a._native, idx, axis)));
}

/**
 * NumPy split: `sections` equal parts (must divide the axis) or split points.
 * Returns views.
 */
export function split(a: ArrayLike, indicesOrSections: number | readonly number[] | NDArray, axis = 0): NDArray[] {
  return splitImpl(asArr(a), indicesOrSections, axis, true);
}

/** NumPy array_split: like split, but sections need not divide the axis. */
export function arraySplit(a: ArrayLike, indicesOrSections: number | readonly number[] | NDArray, axis = 0): NDArray[] {
  return splitImpl(asArr(a), indicesOrSections, axis, false);
}

/** NumPy hsplit: split along axis 1 (axis 0 for 1-d input). */
export function hsplit(a: ArrayLike, indicesOrSections: number | readonly number[] | NDArray): NDArray[] {
  const x = asArr(a);
  if (x.ndim === 0) throw new ValueError("hsplit only works on arrays of 1 or more dimensions");
  return split(x, indicesOrSections, x.ndim > 1 ? 1 : 0);
}

/** NumPy vsplit: split along axis 0 (input at least 2-d). */
export function vsplit(a: ArrayLike, indicesOrSections: number | readonly number[] | NDArray): NDArray[] {
  const x = asArr(a);
  if (x.ndim < 2) throw new ValueError("vsplit only works on arrays of 2 or more dimensions");
  return split(x, indicesOrSections, 0);
}

/** NumPy dsplit: split along axis 2 (input at least 3-d). */
export function dsplit(a: ArrayLike, indicesOrSections: number | readonly number[] | NDArray): NDArray[] {
  const x = asArr(a);
  if (x.ndim < 3) throw new ValueError("dsplit only works on arrays of 3 or more dimensions");
  return split(x, indicesOrSections, 2);
}

// ---- atleast_nd ----

/** View of `a` with leading length-1 axes up to `ndim` (NumPy ndmin=). */
export function atleastNd(a: NDArray, ndim: number): NDArray {
  return a.ndim >= ndim ? a : a.reshape([...new Array<number>(ndim - a.ndim).fill(1), ...a.shape]);
}

function atleast3(a: NDArray): NDArray {
  if (a.ndim === 0) return a.reshape([1, 1, 1]);
  if (a.ndim === 1) return a.reshape([1, a.shape[0]!, 1]);
  if (a.ndim === 2) return a.reshape([...a.shape, 1]);
  return a;
}

function many(arys: ArrayLike[], f: (a: NDArray) => NDArray): NDArray | NDArray[] {
  const r = arys.map((x) => f(asArr(x)));
  return r.length === 1 ? r[0]! : r;
}

/** NumPy atleast_1d: one input → one array (a view); several → a list. */
export function atleast1d(a: ArrayLike): NDArray;
export function atleast1d(...arys: ArrayLike[]): NDArray | NDArray[];
export function atleast1d(...arys: ArrayLike[]): NDArray | NDArray[] {
  return many(arys, (a) => atleastNd(a, 1));
}

/** NumPy atleast_2d: 1-d (N) becomes (1, N). */
export function atleast2d(a: ArrayLike): NDArray;
export function atleast2d(...arys: ArrayLike[]): NDArray | NDArray[];
export function atleast2d(...arys: ArrayLike[]): NDArray | NDArray[] {
  return many(arys, (a) => atleastNd(a, 2));
}

/** NumPy atleast_3d: (N) → (1, N, 1), (M, N) → (M, N, 1). */
export function atleast3d(a: ArrayLike): NDArray;
export function atleast3d(...arys: ArrayLike[]): NDArray | NDArray[];
export function atleast3d(...arys: ArrayLike[]): NDArray | NDArray[] {
  return many(arys, atleast3);
}
