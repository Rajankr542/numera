// Parity milestone P8: indexing extras (D-056, D-110, D-111).
// Native kernels: native/core/p08_indexing.{hpp,cpp}, exposed as `addon.p08`.
import { nativeModule, type NativeNDArray } from "./addon.js";
import { array, zerosLike } from "./creation.js";
import { dtype as toDType, promoteTypes, type DType, type DTypeLike } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";
import { nonzero, take, type ClipMode, type TakeOptions } from "./indexing.js";
import { NDArray, type MemoryOrder, type NestedArray, type Shape } from "./ndarray.js";
import { isComplexLike } from "./complex.js";
import { operands, type ArrayLike, type Operand } from "./ufunc.js";

export type { ClipMode, TakeOptions } from "./indexing.js";

type N = NativeNDArray;
interface P08Native {
  takeAlongAxis(a: N, indices: N, axis: number | null): N;
  putAlongAxis(a: N, indices: N, values: N, axis: number | null): void;
  put(a: N, ind: N, v: N, mode?: string): void;
  putmask(a: N, mask: N, values: N, safe: boolean): void;
  place(a: N, mask: N, vals: N, safe: boolean): void;
  choose(a: N, choices: N[], mode?: string): N;
  compress(condition: N, a: N, axis: number | null): N;
  extract(condition: N, a: N): N;
  select(condlist: N[], choicelist: N[], dflt: N): N;
  noneOf(condlist: N[]): N;
  argwhere(a: N): N;
  flatnonzero(a: N): N;
  countNonzero(a: N, axis: number[] | null, keepdims: boolean): N;
  ravelMultiIndex(arrays: N[], dims: number[], mode: string | string[] | undefined, order?: string): N;
  unravelIndex(indices: N, shape: number[], order?: string): N[];
  diagonal(a: N, offset: number, axis1: number, axis2: number): N;
  trace(a: N, offset: number, axis1: number, axis2: number, dtype: string | null): N;
}

const native = (): P08Native => nativeModule<P08Native>("p08");
const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));
const wrap = (f: () => N): NDArray => wrapNative(() => NDArray._wrap(f()));

/** Index operand: dtype inferred (floats are rejected natively); an empty list is int64. */
function toIndex(a: ArrayLike): NDArray {
  if (a instanceof NDArray) return a;
  const x = array(a);
  return x.size === 0 && x.dtype.kind === "f" ? x.astype("int64") : x;
}

const isWeakScalar = (x: unknown): x is number | boolean | bigint =>
  typeof x === "number" || typeof x === "boolean" || typeof x === "bigint" || isComplexLike(x);

/** Converts operands, treating JS scalars as weak relative to the other operands' dtype (D-111). */
function weakAll(values: readonly Operand[], extra?: DType): NDArray[] {
  let like: DType | undefined = extra;
  for (const v of values) {
    if (isWeakScalar(v)) continue;
    const d = toArray(v as ArrayLike).dtype;
    like = like ? promoteTypes(like, d) : d;
  }
  return values.map((v) => {
    if (!isWeakScalar(v)) return toArray(v as ArrayLike);
    return like ? operands(v, 0, like)[0] : array(v as NestedArray);
  });
}

// ---- take_along_axis / put_along_axis ----

/** NumPy take_along_axis. `axis` defaults to -1; `null` uses the flattened array (1-d indices). */
export function takeAlongAxis(arr: ArrayLike, indices: ArrayLike, axis: number | null = -1): NDArray {
  const a = toArray(arr);
  const i = toIndex(indices);
  return wrap(() => native().takeAlongAxis(a._native, i._native, axis));
}

/** NumPy put_along_axis: writes `values` (broadcast, unsafe cast) into `arr` in place. */
export function putAlongAxis(
  arr: NDArray,
  indices: ArrayLike,
  values: ArrayLike,
  axis: number | null,
): void {
  const i = toIndex(indices);
  const v = toArray(values);
  wrapNative(() => native().putAlongAxis(arr._native, i._native, v._native, axis));
}

// ---- put / putmask / place ----

/** Options for `put`. */
export interface PutOptions {
  mode?: ClipMode;
}

/** NumPy put: `a.flat[ind] = v` in place (v repeats; unsafe cast). */
export function put(a: NDArray, ind: ArrayLike, v: ArrayLike, opts: PutOptions = {}): void {
  const i = toIndex(ind);
  const vals = toArray(v);
  wrapNative(() => native().put(a._native, i._native, vals._native, opts.mode));
}

function maskedValues(a: NDArray, values: ArrayLike): [NDArray, boolean] {
  if (values instanceof NDArray) return [values, true];
  return [array(values, { dtype: a.dtype }), false];
}

/** NumPy putmask: `a.flat[i] = values[i % n]` where `mask.flat[i]` (D-111 casting). */
export function putmask(a: NDArray, mask: ArrayLike, values: ArrayLike): void {
  const m = toArray(mask);
  const [v, safe] = maskedValues(a, values);
  wrapNative(() => native().putmask(a._native, m._native, v._native, safe));
}

/** NumPy place: the k-th masked element of `arr` gets `vals[k % n]` (D-111 casting). */
export function place(arr: NDArray, mask: ArrayLike, vals: ArrayLike): void {
  const m = toArray(mask);
  const [v, safe] = maskedValues(arr, vals);
  wrapNative(() => native().place(arr._native, m._native, v._native, safe));
}

// ---- choose / compress / extract / select ----

/** Options for `choose`. */
export interface ChooseOptions {
  mode?: ClipMode;
}

/** NumPy choose. `choices` is a list of arrays/scalars, or an array whose first axis enumerates them. */
export function choose(
  a: ArrayLike,
  choices: NDArray | readonly Operand[],
  opts: ChooseOptions = {},
): NDArray {
  const idx = toIndex(a);
  const list =
    choices instanceof NDArray
      ? Array.from({ length: choices.shape[0] ?? 0 }, (_, k) => choices.get(k))
      : weakAll(choices);
  if (choices instanceof NDArray && choices.ndim === 0) {
    throw new ValueError("choices must be a sequence of arrays");
  }
  return wrap(() => native().choose(idx._native, list.map((c) => c._native), opts.mode));
}

/** Options for `compress`. */
export interface AxisOptions {
  axis?: number | null;
}

/** NumPy compress: slices of `a` along `axis` (flattened if omitted) where `condition` is true. */
export function compress(
  condition: ArrayLike,
  a: ArrayLike,
  axis?: number | null | AxisOptions,
): NDArray {
  const ax = typeof axis === "object" && axis !== null ? axis.axis : axis;
  const c = toArray(condition);
  const x = toArray(a);
  return wrap(() => native().compress(c._native, x._native, ax ?? null));
}

/** NumPy extract: elements of `arr` (flattened) where `condition` (flattened) is true. */
export function extract(condition: ArrayLike, arr: ArrayLike): NDArray {
  const c = toArray(condition);
  const x = toArray(arr);
  return wrap(() => native().extract(c._native, x._native));
}

/** Options for `select`. */
export interface SelectOptions {
  /** Value where no condition holds; default 0 (a weak scalar). */
  default?: Operand;
}

/** NumPy select: element from the first `choicelist` entry whose condition is true. */
export function select(
  condlist: readonly ArrayLike[],
  choicelist: readonly ArrayLike[],
  opts: SelectOptions = {},
): NDArray {
  const conds = condlist.map(toArray);
  const choices = choicelist.map(toArray);
  const dv = opts.default ?? 0;
  const like = choices.reduce<DType | undefined>(
    (acc, c) => (acc ? promoteTypes(acc, c.dtype) : c.dtype),
    undefined,
  );
  const d = isWeakScalar(dv) && like ? operands(dv, 0, like)[0] : toArray(dv as ArrayLike);
  return wrap(() =>
    native().select(
      conds.map((c) => c._native),
      choices.map((c) => c._native),
      d._native,
    ),
  );
}

// ---- piecewise ----

/** A `piecewise` piece: a constant, or a callback receiving the selected elements. */
export type PiecewiseFunc = number | boolean | ((x: NDArray) => ArrayLike);

/**
 * NumPy piecewise. `condlist` is one condition or a list of them; one extra
 * function in `funclist` is the "otherwise" piece. Callbacks run in JS on
 * `x[cond]` and only when that selection is non-empty (D-110).
 */
export function piecewise(
  x: ArrayLike,
  condlist: ArrayLike | readonly ArrayLike[],
  funclist: readonly PiecewiseFunc[],
): NDArray {
  const xa = toArray(x);
  // NumPy rule: wrap a scalar, or a list whose first entry is not a list/array (x not 0-d).
  const single =
    typeof condlist === "boolean" ||
    (condlist instanceof NDArray && (condlist.ndim === 0 || (condlist.ndim === 1 && xa.ndim !== 0))) ||
    (Array.isArray(condlist) &&
      condlist.length > 0 &&
      !(condlist[0] instanceof NDArray) &&
      !Array.isArray(condlist[0]) &&
      xa.ndim !== 0);
  const raw: readonly ArrayLike[] = single
    ? [condlist as ArrayLike]
    : condlist instanceof NDArray
      ? Array.from({ length: condlist.shape[0] ?? 0 }, (_, k) => condlist.get(k))
      : (condlist as readonly ArrayLike[]);
  const conds = raw.map((c) => {
    const a = toArray(c);
    return a.dtype.name === "bool" ? a : a.astype("bool");
  });
  const n = conds.length;
  if (n === funclist.length - 1) {
    conds.push(wrap(() => native().noneOf(conds.map((c) => c._native))));
  } else if (n !== funclist.length) {
    throw new ValueError(
      `with ${n} condition(s), either ${n} or ${n + 1} functions are expected`,
    );
  }
  const y = zerosLike(xa);
  conds.forEach((cond, k) => {
    const f = funclist[k];
    if (typeof f !== "function") {
      y.set([cond], f as number | boolean);
      return;
    }
    const vals = xa.get(cond);
    if (vals.size > 0) y.set([cond], f(vals));
  });
  return y;
}

// ---- argwhere / flatnonzero / countNonzero ----

/** NumPy argwhere: `(N, a.ndim)` int64 indices of the non-zero elements. */
export function argwhere(a: ArrayLike): NDArray {
  const x = toArray(a);
  return wrap(() => native().argwhere(x._native));
}

/** NumPy flatnonzero: indices of the non-zero elements of the flattened array. */
export function flatnonzero(a: ArrayLike): NDArray {
  const x = toArray(a);
  return wrap(() => native().flatnonzero(x._native));
}

/** Options for `countNonzero`. */
export interface CountNonzeroOptions {
  axis?: number | readonly number[] | null;
  keepdims?: boolean;
}

/** NumPy count_nonzero: int64 counts (a 0-d array without `axis`, D-110). */
export function countNonzero(a: ArrayLike, opts: CountNonzeroOptions = {}): NDArray {
  const x = toArray(a);
  const ax = opts.axis ?? null;
  const axes = ax === null ? null : typeof ax === "number" ? [ax] : [...ax];
  return wrap(() => native().countNonzero(x._native, axes, opts.keepdims ?? false));
}

// ---- ravelMultiIndex / unravelIndex ----

/** Options for `ravelMultiIndex`. */
export interface RavelMultiIndexOptions {
  /** One mode, or one per dimension. Default "raise". */
  mode?: ClipMode | readonly ClipMode[];
  /** "C" (row-major, default) or "F". */
  order?: MemoryOrder;
}

const dimsOf = (s: Shape | number): number[] => (typeof s === "number" ? [s] : [...s]);

/** NumPy ravel_multi_index: flat int64 indices from per-dimension index arrays. */
export function ravelMultiIndex(
  multiIndex: readonly ArrayLike[],
  dims: Shape | number,
  opts: RavelMultiIndexOptions = {},
): NDArray {
  const arrays = multiIndex.map(toIndex);
  const mode = opts.mode === undefined ? undefined : typeof opts.mode === "string" ? opts.mode : [...opts.mode];
  return wrap(() =>
    native().ravelMultiIndex(
      arrays.map((x) => x._native),
      dimsOf(dims),
      mode,
      opts.order,
    ),
  );
}

/** NumPy unravel_index: one int64 array per dimension of `shape`. */
export function unravelIndex(
  indices: ArrayLike,
  shape: Shape | number,
  opts: { order?: MemoryOrder } = {},
): NDArray[] {
  const i = toIndex(indices);
  return wrapNative(() =>
    native().unravelIndex(i._native, dimsOf(shape), opts.order).map((h) => NDArray._wrap(h)),
  );
}

// ---- diagonal / trace ----

/** Options for `diagonal` / `trace` (NumPy `offset`, `axis1`, `axis2`). */
export interface DiagonalOptions {
  offset?: number;
  axis1?: number;
  axis2?: number;
}

/** Options for `trace`. */
export interface TraceOptions extends DiagonalOptions {
  dtype?: DTypeLike | null;
}

/** NumPy diagonal: a read-only view; the diagonal axis is last (D-110). */
export function diagonal(a: ArrayLike, opts: DiagonalOptions | number = {}): NDArray {
  const o = typeof opts === "number" ? { offset: opts } : opts;
  const x = toArray(a);
  return wrap(() => native().diagonal(x._native, o.offset ?? 0, o.axis1 ?? 0, o.axis2 ?? 1));
}

/** NumPy trace: sum along the diagonal (sum dtype rules, D-017). */
export function trace(a: ArrayLike, opts: TraceOptions | number = {}): NDArray {
  const o: TraceOptions = typeof opts === "number" ? { offset: opts } : opts;
  const x = toArray(a);
  const dt = o.dtype == null ? null : toDType(o.dtype).name;
  return wrap(() => native().trace(x._native, o.offset ?? 0, o.axis1 ?? 0, o.axis2 ?? 1, dt));
}

// ---- NDArray methods (declaration merging, D-056) ----

declare module "./ndarray.js" {
  interface NDArray {
    /** NumPy `a.take(indices, axis?, {mode})`. */
    take(indices: ArrayLike, axis?: number | null | TakeOptions, opts?: TakeOptions): NDArray;
    /** NumPy `a.put(ind, v, {mode})` (in place). */
    put(ind: ArrayLike, v: ArrayLike, opts?: PutOptions): void;
    /** NumPy `a.choose(choices, {mode})`. */
    choose(choices: NDArray | readonly Operand[], opts?: ChooseOptions): NDArray;
    /** NumPy `a.compress(condition, axis?)`. */
    compress(condition: ArrayLike, axis?: number | null | AxisOptions): NDArray;
    /** NumPy `a.diagonal({offset, axis1, axis2})` (read-only view). */
    diagonal(opts?: DiagonalOptions | number): NDArray;
    /** NumPy `a.trace({offset, axis1, axis2, dtype})`. */
    trace(opts?: TraceOptions | number): NDArray;
    /** NumPy `a.nonzero()`. */
    nonzero(): NDArray[];
  }
}

NDArray.prototype.take = function (this: NDArray, indices, axis, opts) {
  return take(this, indices, axis, opts);
};
NDArray.prototype.put = function (this: NDArray, ind, v, opts) {
  put(this, ind, v, opts);
};
NDArray.prototype.choose = function (this: NDArray, choices, opts) {
  return choose(this, choices, opts);
};
NDArray.prototype.compress = function (this: NDArray, condition, axis) {
  return compress(condition, this, axis);
};
NDArray.prototype.diagonal = function (this: NDArray, opts) {
  return diagonal(this, opts);
};
NDArray.prototype.trace = function (this: NDArray, opts) {
  return trace(this, opts);
};
NDArray.prototype.nonzero = function (this: NDArray) {
  return nonzero(this);
};

export const p08 = {
  takeAlongAxis,
  putAlongAxis,
  put,
  putmask,
  place,
  choose,
  compress,
  extract,
  select,
  piecewise,
  argwhere,
  flatnonzero,
  countNonzero,
  ravelMultiIndex,
  unravelIndex,
  diagonal,
  trace,
} as const;
