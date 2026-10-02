// Parity milestone P9: sorting, searching and set functions (D-120–D-124).
// Native kernels: native/core/p09_sorting.cpp, p09_sets.cpp (`addon.p09`).

import { addon, nativeModule, type NativeNDArray } from "./addon.js";
import { array } from "./creation.js";
import { canCast, dtype as toDType, promoteTypes, type DType } from "./dtype.js";
import { DTypeError, IndexError, ValueError, wrapNative } from "./errors.js";
import { isComplexLike, type ComplexLike } from "./complex.js";
import { NDArray } from "./ndarray.js";
import { toArray, type ArrayLike } from "./ufunc.js";

type N = NativeNDArray;
interface UniqueNative {
  values: N;
  indices?: N;
  inverse?: N;
  counts?: N;
}
interface P09Native {
  sort(a: N, axis: number | null, kind: string, descending: boolean): N;
  sortInPlace(a: N, axis: number, kind: string, descending: boolean): undefined;
  argsort(a: N, axis: number | null, kind: string, descending: boolean): N;
  partition(a: N, kth: number[], axis: number | null): N;
  partitionInPlace(a: N, kth: number[], axis: number): undefined;
  argpartition(a: N, kth: number[], axis: number | null): N;
  lexsort(keys: N[], axis: number): N;
  searchsorted(a: N, v: N, right: boolean, sorter: N | null): N;
  unique1d(a: N, ri: boolean, rinv: boolean, rc: boolean, equalNan: boolean): UniqueNative;
  uniqueRows(a: N, ri: boolean, rinv: boolean, rc: boolean): UniqueNative;
  isin(a: N, b: N, invert: boolean): N;
  intersect1d(a: N, b: N, assumeUnique: boolean, returnIndices: boolean): {
    values: N;
    indices1?: N;
    indices2?: N;
  };
  union1d(a: N, b: N): N;
  setxor1d(a: N, b: N, assumeUnique: boolean): N;
  setdiff1d(a: N, b: N, assumeUnique: boolean): N;
  ediff1d(a: N, toBegin: N | null, toEnd: N | null): N;
}

const native = nativeModule<P09Native>("p09");
const W = (h: N): NDArray => NDArray._wrap(h);

// ---------------------------------------------------------------- sort (D-120)

export type SortKind = "quicksort" | "mergesort" | "heapsort" | "stable" | string;

export interface SortOptions {
  /** Axis to sort along (default -1); `null` sorts the flattened array. */
  axis?: number | null;
  /** NumPy `kind`: the first letter selects q(uick)/h(eap) or m(erge)/s(table). */
  kind?: SortKind | null;
  /** Guarantee a stable sort (NumPy `stable=`). Cannot be combined with `kind`. */
  stable?: boolean | null;
  /** Sort in descending order; NaNs still sort last (NumPy `descending=`). */
  descending?: boolean | null;
}

function sortKind(opts: SortOptions): { kind: string; descending: boolean } {
  const kind = opts.kind ?? null;
  const hasKw = (opts.stable ?? null) !== null || opts.descending !== undefined;
  if (kind !== null && hasKw) {
    throw new ValueError(
      "`kind` and keyword parameters can't be provided at the same time. Use only one of them.",
    );
  }
  const descending = opts.descending === true;
  if (kind === null) return { kind: opts.stable === true ? "stable" : "quick", descending };
  if (typeof kind !== "string") throw new DTypeError("sort kind must be str");
  const c = kind.charAt(0).toLowerCase();
  if (c === "q" || c === "h") return { kind: "quick", descending };
  if (c === "m" || c === "s") return { kind: "stable", descending };
  throw new ValueError(`sort kind must be one of 'quick', 'heap', or 'stable' (got '${kind}')`);
}

const axisOf = (axis: number | null | undefined): number | null => (axis === undefined ? -1 : axis);

/** Sorted copy of an array (NumPy `np.sort`). */
export function sort(a: ArrayLike, opts: SortOptions = {}): NDArray {
  const x = toArray(a);
  const { kind, descending } = sortKind(opts);
  return wrapNative(() => W(native.sort(x._native, axisOf(opts.axis), kind, descending)));
}

/** Indices that sort the array (NumPy `np.argsort`), int64. */
export function argsort(a: ArrayLike, opts: SortOptions = {}): NDArray {
  const x = toArray(a);
  const { kind, descending } = sortKind(opts);
  return wrapNative(() => W(native.argsort(x._native, axisOf(opts.axis), kind, descending)));
}

/** Sorts a complex array along the last axis, real part first (NumPy `np.sort_complex`). */
export function sortComplex(a: ArrayLike): NDArray {
  const x = toArray(a);
  const small = ["int8", "uint8", "int16", "uint16"].includes(x.dtype.name);
  const target = x.dtype.kind === "c" ? x.dtype : toDType(small ? "complex64" : "complex128");
  const y = x.dtype === target ? x : x.astype(target);
  return wrapNative(() => W(native.sort(y._native, -1, "quick", false)));
}

// ----------------------------------------------------------- partition (D-121)

export type Kth = number | readonly number[] | NDArray;

export interface PartitionOptions {
  /** Axis (default -1); `null` partitions the flattened array. */
  axis?: number | null;
  /** Only "introselect" is accepted (NumPy `kind=`). */
  kind?: string;
}

function kthList(kth: Kth): number[] {
  const check = (v: unknown): number => {
    if (typeof v === "boolean") throw new ValueError("Booleans unacceptable as partition index");
    if (typeof v === "bigint") return Number(v);
    if (typeof v !== "number" || !Number.isInteger(v)) throw new DTypeError("Partition index must be integer");
    return v;
  };
  if (kth instanceof NDArray) {
    if (kth.dtype.kind === "b") throw new ValueError("Booleans unacceptable as partition index");
    if (kth.dtype.kind !== "i" && kth.dtype.kind !== "u") throw new DTypeError("Partition index must be integer");
    if (kth.ndim > 1) throw new ValueError("kth array must have dimension <= 1");
    const list = kth.toArray();
    return (Array.isArray(list) ? list : [list]).map(check);
  }
  if (Array.isArray(kth)) {
    if (kth.length === 0) throw new DTypeError("Partition index must be integer");
    if (kth.some((k) => Array.isArray(k))) throw new ValueError("kth array must have dimension <= 1");
    if (kth.every((k) => typeof k === "boolean")) throw new ValueError("Booleans unacceptable as partition index");
    return kth.map(check);
  }
  return [check(kth)];
}

function checkSelectKind(kind: string | undefined): void {
  if (kind !== undefined && kind !== "introselect") {
    throw new ValueError(`select kind must be 'introselect' (got '${kind}')`);
  }
}

/** Partitioned copy: element `kth` is in its sorted position (NumPy `np.partition`). */
export function partition(a: ArrayLike, kth: Kth, opts: PartitionOptions = {}): NDArray {
  checkSelectKind(opts.kind);
  const x = toArray(a);
  const ks = kthList(kth);
  return wrapNative(() => W(native.partition(x._native, ks, axisOf(opts.axis))));
}

/** Indices that partition the array (NumPy `np.argpartition`), int64. */
export function argpartition(a: ArrayLike, kth: Kth, opts: PartitionOptions = {}): NDArray {
  checkSelectKind(opts.kind);
  const x = toArray(a);
  const ks = kthList(kth);
  return wrapNative(() => W(native.argpartition(x._native, ks, axisOf(opts.axis))));
}

// ------------------------------------------------- lexsort / searchsorted (D-122)

/**
 * Indirect stable sort on several keys; the last key is the primary one
 * (NumPy `np.lexsort`). `keys` is a list of arrays or an array whose first
 * axis indexes the keys.
 */
export function lexsort(keys: readonly ArrayLike[] | NDArray, opts: { axis?: number | null } = {}): NDArray {
  if (opts.axis === null) throw new DTypeError("'NoneType' object cannot be interpreted as an integer");
  let list: NDArray[];
  if (keys instanceof NDArray) {
    if (keys.ndim === 0) throw new DTypeError("need sequence of keys with len > 0 in lexsort");
    list = Array.from({ length: keys.shape[0] ?? 0 }, (_, i) => keys.get(i));
  } else {
    list = keys.map(toArray);
  }
  return wrapNative(() => W(native.lexsort(list.map((k) => k._native), opts.axis ?? -1)));
}

export interface SearchsortedOptions {
  /** "left" (default): first suitable index; "right": last. */
  side?: "left" | "right";
  /** Indices that sort `a` (e.g. from argsort). */
  sorter?: ArrayLike | null;
}

type Scalar = number | boolean | bigint | ComplexLike;
const isScalar = (v: unknown): v is Scalar =>
  typeof v === "number" || typeof v === "boolean" || typeof v === "bigint" || isComplexLike(v);

const intRange: Record<string, [bigint, bigint]> = {
  int8: [-128n, 127n],
  uint8: [0n, 255n],
  int16: [-32768n, 32767n],
  uint16: [0n, 65535n],
  int32: [-2147483648n, 2147483647n],
  uint32: [0n, 4294967295n],
  int64: [-(2n ** 63n), 2n ** 63n - 1n],
  uint64: [0n, 2n ** 64n - 1n],
};

// Common dtype of an array and a JS scalar (weak scalar, NEP 50 style; D-122).
function scalarCommon(like: DType, v: Scalar): DType {
  if (typeof v === "boolean") return like;
  if (isComplexLike(v)) {
    if (like.kind === "c") return like;
    return toDType(["float16", "float32"].includes(like.name) ? "complex64" : "complex128");
  }
  const isInt = typeof v === "bigint" || Number.isInteger(v);
  if (!isInt) return like.kind === "f" || like.kind === "c" ? like : toDType("float64");
  if (like.kind === "b") return toDType("int64");
  if (like.kind === "f" || like.kind === "c") return like;
  const r = intRange[like.name];
  const big = BigInt(v);
  if (r && big >= r[0] && big <= r[1]) return like;
  return promoteTypes(like, big < 0n ? "int64" : big > 2n ** 63n - 1n ? "uint64" : "int64");
}

/** Insertion indices that keep `a` sorted (NumPy `np.searchsorted`), int64. */
export function searchsorted(a: ArrayLike, v: ArrayLike | Scalar, opts: SearchsortedOptions = {}): NDArray {
  const side = opts.side ?? "left";
  if (side !== "left" && side !== "right") {
    const s = String(side);
    const c = s.charAt(0).toLowerCase();
    if (c === "l" || c === "r") throw new ValueError("search side must be one of 'left' or 'right'");
    throw new ValueError(`search side must be 'left' or 'right' (got '${s}')`);
  }
  const x = toArray(a);
  let common: DType;
  let vArr: NDArray | null = null;
  if (v instanceof NDArray || !isScalar(v)) {
    vArr = toArray(v as ArrayLike);
    common = promoteTypes(x.dtype, vArr.dtype);
  } else {
    common = scalarCommon(x.dtype, v);
  }
  const keys = vArr ?? array(v as Scalar as never, { dtype: common });
  const xs = x.dtype === common ? x : x.astype(common);
  const ks = keys.dtype === common ? keys : keys.astype(common);
  let sorter: NDArray | null = null;
  if (opts.sorter != null) {
    sorter = toArray(opts.sorter);
    if (sorter.dtype.kind !== "i" && sorter.dtype.kind !== "u") {
      throw new DTypeError("sorter must only contain integers");
    }
  }
  return wrapNative(() =>
    W(native.searchsorted(xs._native, ks._native, side === "right", sorter?._native ?? null)),
  );
}

// -------------------------------------------------------------- unique (D-123)

export interface UniqueOptions {
  returnIndex?: boolean;
  returnInverse?: boolean;
  returnCounts?: boolean;
  /** Axis whose sub-arrays are compared; `null`/omitted flattens. */
  axis?: number | null;
  /** Treat all NaNs as one value (default true). */
  equalNan?: boolean;
  /** Accepted for NumPy parity; the result is always sorted (D-123). */
  sorted?: boolean;
}

export interface UniqueResult {
  values: NDArray;
  indices?: NDArray;
  inverse?: NDArray;
  counts?: NDArray;
}

function normAxis(axis: number, ndim: number): number {
  if (!Number.isInteger(axis) || axis < -ndim || axis >= ndim) {
    throw new IndexError(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }
  return axis < 0 ? axis + ndim : axis;
}

function uniqueImpl(a: ArrayLike, o: UniqueOptions): UniqueResult {
  const x = toArray(a);
  const ri = o.returnIndex === true;
  const rinv = o.returnInverse === true;
  const rc = o.returnCounts === true;
  const axis = o.axis ?? null;
  return wrapNative(() => {
    if (axis === null || x.ndim === 1) {
      if (axis !== null) normAxis(axis, x.ndim);
      const r = native.unique1d(x._native, ri, rinv, rc, o.equalNan ?? true);
      const out: UniqueResult = { values: W(r.values) };
      if (r.indices) out.indices = W(r.indices);
      if (r.inverse) out.inverse = W(r.inverse).reshape(x.shape);
      if (r.counts) out.counts = W(r.counts);
      return out;
    }
    const ax = normAxis(axis, x.ndim);
    const moved = W(addon.moveaxis(x._native, [ax], [0]));
    const rest = moved.shape.slice(1);
    const m = rest.reduce((p, q) => p * q, 1);
    const n = moved.shape[0] ?? 0;
    const flat2 = W(addon.copyOrder(moved._native, null, "C")).reshape([n, m]);
    const r = native.uniqueRows(flat2._native, ri, rinv, rc);
    const vals = W(r.values);
    const k = vals.shape[0] ?? 0;
    const shaped = vals.reshape([k, ...rest]);
    const out: UniqueResult = { values: W(addon.moveaxis(shaped._native, [0], [ax])) };
    if (r.indices) out.indices = W(r.indices);
    if (r.inverse) out.inverse = W(r.inverse);
    if (r.counts) out.counts = W(r.counts);
    return out;
  });
}

/**
 * Sorted unique elements (NumPy `np.unique`). Returns the values array, or an
 * object `{values, indices?, inverse?, counts?}` when a `return*` flag is set.
 */
export function unique(a: ArrayLike, opts?: UniqueOptions & { returnIndex?: false; returnInverse?: false; returnCounts?: false }): NDArray;
export function unique(a: ArrayLike, opts: UniqueOptions): UniqueResult;
export function unique(a: ArrayLike, opts: UniqueOptions = {}): NDArray | UniqueResult {
  const r = uniqueImpl(a, opts);
  if (opts.returnIndex || opts.returnInverse || opts.returnCounts) return r;
  return r.values;
}

export interface UniqueAllResult {
  values: NDArray;
  indices: NDArray;
  inverseIndices: NDArray;
  counts: NDArray;
}
export interface UniqueCountsResult {
  values: NDArray;
  counts: NDArray;
}
export interface UniqueInverseResult {
  values: NDArray;
  inverseIndices: NDArray;
}

const all = (a: ArrayLike): Required<UniqueResult> =>
  uniqueImpl(a, { returnIndex: true, returnInverse: true, returnCounts: true, equalNan: false }) as Required<UniqueResult>;

/** Array API `unique_all`: values, first indices, inverse indices and counts. */
export function uniqueAll(x: ArrayLike): UniqueAllResult {
  const r = all(x);
  return { values: r.values, indices: r.indices, inverseIndices: r.inverse, counts: r.counts };
}
/** Array API `unique_counts`. */
export function uniqueCounts(x: ArrayLike): UniqueCountsResult {
  const r = uniqueImpl(x, { returnCounts: true, equalNan: false });
  return { values: r.values, counts: r.counts as NDArray };
}
/** Array API `unique_inverse`. */
export function uniqueInverse(x: ArrayLike): UniqueInverseResult {
  const r = uniqueImpl(x, { returnInverse: true, equalNan: false });
  return { values: r.values, inverseIndices: r.inverse as NDArray };
}
/** Array API `unique_values` (NaNs are not merged). */
export function uniqueValues(x: ArrayLike): NDArray {
  return uniqueImpl(x, { equalNan: false }).values;
}

// ------------------------------------------------------ set functions (D-124)

function common2(a: ArrayLike, b: ArrayLike): [NDArray, NDArray] {
  const x = toArray(a);
  const y = toArray(b);
  const dt = promoteTypes(x.dtype, y.dtype);
  return [x.dtype === dt ? x : x.astype(dt), y.dtype === dt ? y : y.astype(dt)];
}

export interface IntersectResult {
  values: NDArray;
  indices1: NDArray;
  indices2: NDArray;
}

/** Sorted unique values present in both inputs (NumPy `np.intersect1d`). */
export function intersect1d(a: ArrayLike, b: ArrayLike, opts?: { assumeUnique?: boolean; returnIndices?: false }): NDArray;
export function intersect1d(a: ArrayLike, b: ArrayLike, opts: { assumeUnique?: boolean; returnIndices: true }): IntersectResult;
export function intersect1d(
  a: ArrayLike,
  b: ArrayLike,
  opts: { assumeUnique?: boolean; returnIndices?: boolean } = {},
): NDArray | IntersectResult {
  const [x, y] = common2(a, b);
  const ri = opts.returnIndices === true;
  const r = wrapNative(() => native.intersect1d(x._native, y._native, opts.assumeUnique === true, ri));
  if (!ri) return W(r.values);
  return { values: W(r.values), indices1: W(r.indices1 as N), indices2: W(r.indices2 as N) };
}

/** Sorted unique values of either input (NumPy `np.union1d`). */
export function union1d(a: ArrayLike, b: ArrayLike): NDArray {
  const [x, y] = common2(a, b);
  return wrapNative(() => W(native.union1d(x._native, y._native)));
}

/** Sorted unique values in exactly one input (NumPy `np.setxor1d`). */
export function setxor1d(a: ArrayLike, b: ArrayLike, opts: { assumeUnique?: boolean } = {}): NDArray {
  const [x, y] = common2(a, b);
  return wrapNative(() => W(native.setxor1d(x._native, y._native, opts.assumeUnique === true)));
}

/** Unique values of `a` not in `b` (NumPy `np.setdiff1d`). */
export function setdiff1d(a: ArrayLike, b: ArrayLike, opts: { assumeUnique?: boolean } = {}): NDArray {
  const [x, y] = common2(a, b);
  return wrapNative(() => W(native.setdiff1d(x._native, y._native, opts.assumeUnique === true)));
}

export interface IsinOptions {
  assumeUnique?: boolean;
  invert?: boolean;
  /** null, "sort" or "table" (table only for bool/integer input); same result for all. */
  kind?: "sort" | "table" | null;
}

/** Bool array of `element`'s shape: is each value in `testElements`? (NumPy `np.isin`). */
export function isin(element: ArrayLike, testElements: ArrayLike, opts: IsinOptions = {}): NDArray {
  const kind = opts.kind ?? null;
  if (kind !== null && kind !== "sort" && kind !== "table") {
    throw new ValueError(`Invalid kind: '${String(kind)}'. Please use None, 'sort' or 'table'.`);
  }
  const [x, y] = common2(element, testElements);
  const ints = (d: DType): boolean => d.kind === "b" || d.kind === "i" || d.kind === "u";
  if (kind === "table" && !(ints(toArray(element).dtype) && ints(toArray(testElements).dtype))) {
    throw new ValueError(
      "The 'table' method is only supported for boolean or integer arrays. Please select 'sort' or None for kind.",
    );
  }
  return wrapNative(() => W(native.isin(x._native, y._native, opts.invert === true)));
}

export interface Ediff1dOptions {
  toEnd?: ArrayLike | null;
  toBegin?: ArrayLike | null;
}

/** Differences between consecutive elements of the flattened array (NumPy `np.ediff1d`). */
export function ediff1d(a: ArrayLike, opts: Ediff1dOptions = {}): NDArray {
  const x = toArray(a);
  const extra = (v: ArrayLike | null | undefined, name: string): NDArray | null => {
    if (v == null) return null;
    const e = toArray(v);
    if (!canCast(e.dtype, x.dtype, "same_kind")) {
      throw new DTypeError(`dtype of \`${name}\` must be compatible with input \`ary\` under the \`same_kind\` rule.`);
    }
    return e.dtype === x.dtype ? e : e.astype(x.dtype);
  };
  const b = extra(opts.toBegin, "to_begin");
  const e = extra(opts.toEnd, "to_end");
  return wrapNative(() => W(native.ediff1d(x._native, b?._native ?? null, e?._native ?? null)));
}

// ------------------------------------------------------------ NDArray methods

declare module "./ndarray.js" {
  interface NDArray {
    /** Sorts in place along `axis` (default -1; NumPy `a.sort`). */
    sort(opts?: Omit<SortOptions, "axis"> & { axis?: number }): void;
    /** NumPy `a.argsort`. */
    argsort(opts?: SortOptions): NDArray;
    /** Partitions in place (NumPy `a.partition`). */
    partition(kth: Kth, opts?: Omit<PartitionOptions, "axis"> & { axis?: number }): void;
    /** NumPy `a.argpartition`. */
    argpartition(kth: Kth, opts?: PartitionOptions): NDArray;
    /** NumPy `a.searchsorted` (this array must be 1-D and sorted). */
    searchsorted(v: ArrayLike | Scalar, opts?: SearchsortedOptions): NDArray;
  }
}

const noneAxis = (): never => {
  throw new DTypeError("'NoneType' object cannot be interpreted as an integer");
};

NDArray.prototype.sort = function (this: NDArray, opts: Omit<SortOptions, "axis"> & { axis?: number } = {}): void {
  if ((opts as SortOptions).axis === null) noneAxis();
  const { kind, descending } = sortKind(opts);
  wrapNative(() => native.sortInPlace(this._native, opts.axis ?? -1, kind, descending));
};
NDArray.prototype.argsort = function (this: NDArray, opts: SortOptions = {}): NDArray {
  return argsort(this, opts);
};
NDArray.prototype.partition = function (
  this: NDArray,
  kth: Kth,
  opts: Omit<PartitionOptions, "axis"> & { axis?: number } = {},
): void {
  if ((opts as PartitionOptions).axis === null) noneAxis();
  checkSelectKind(opts.kind);
  const ks = kthList(kth);
  wrapNative(() => native.partitionInPlace(this._native, ks, opts.axis ?? -1));
};
NDArray.prototype.argpartition = function (this: NDArray, kth: Kth, opts: PartitionOptions = {}): NDArray {
  return argpartition(this, kth, opts);
};
NDArray.prototype.searchsorted = function (
  this: NDArray,
  v: ArrayLike | Scalar,
  opts: SearchsortedOptions = {},
): NDArray {
  return searchsorted(this, v, opts);
};

export const p09 = {
  sort,
  argsort,
  sortComplex,
  partition,
  argpartition,
  lexsort,
  searchsorted,
  unique,
  uniqueAll,
  uniqueCounts,
  uniqueInverse,
  uniqueValues,
  intersect1d,
  union1d,
  setxor1d,
  setdiff1d,
  isin,
  ediff1d,
} as const;
