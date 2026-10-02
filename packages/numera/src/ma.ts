/**
 * np.ma — Masked Array module (P16B, D-200..D-209).
 *
 * A MaskedArray wraps two NDArrays: `_data` (any dtype) and `_mask` (bool
 * NDArray or the `nomask` sentinel `false`).  Computation happens on the
 * underlying plain NDArrays; the mask is propagated (D-205/D-206).
 *
 * Pure TypeScript — no new C++ needed.
 */

import { array as _array, zeros as _zeros, ones as _ones, empty as _empty } from "./creation.js";
import { NDArray } from "./ndarray.js";
import { dtype as toDType } from "./dtype.js";
import type { DTypeLike } from "./dtype.js";
import { ValueError } from "./errors.js";
import type { Shape } from "./ndarray.js";
import type { ArrayLike } from "./ufunc.js";
import {
  add as _add, subtract as _subtract, multiply as _multiply, divide as _divide,
  mod as _mod, power as _power, abs as _abs, negative as _negative, sqrt as _sqrt,
  exp as _exp, log as _log, floorDivide as _floorDivide, angle as _angle,
  broadcastTo as _broadcastTo,
} from "./ufunc.js";
import { dot as _dot, inner as _inner, outer as _outer } from "./linalg.js";
import {
  intersect1d as _intersect1d, union1d as _union1d, setdiff1d as _setdiff1d,
  setxor1d as _setxor1d, isin as _isin, unique as _unique, sort as _sort, argsort as _argsort,
} from "./p09.js";
import { diag as _diag, diagflat as _diagflat, vander as _vander } from "./p07_creation.js";
import { concatenate as _concatenate } from "./p06_join.js";
import { polyfit as _polyfit } from "./p14_poly.js";

// ---------------------------------------------------------------------------
// Lazily-imported ufuncs from p04/p05 (avoid circular import issues)
// ---------------------------------------------------------------------------

// We use dynamic method-like calls via the addon for ops not in ufunc.ts.
// For comparisons and trig we use the standalone ufuncs imported below.
import {
  sin as _sin, cos as _cos, tan as _tan,
  arcsin as _arcsin, arccos as _arccos, arctan as _arctan,
  arctan2 as _arctan2, sinh as _sinh, cosh as _cosh, tanh as _tanh,
  arcsinh as _arcsinh, arccosh as _arccosh, arctanh as _arctanh,
  hypot as _hypot,
} from "./p04_trig.js";
import {
  log2 as _log2, log10 as _log10,
} from "./p04_explog.js";
import { ceil as _ceil, floor as _floor } from "./p04_rounding.js";
import { clip as _clip, fmod as _fmod, maximum as _maximum, minimum as _minimum } from "./p04_arith.js";
import {
  equal as _equal, notEqual as _notEqual,
  less as _less, lessEqual as _lessEqual,
  greater as _greater, greaterEqual as _greaterEqual,
  logicalAnd as _logicalAnd, logicalOr as _logicalOr,
  logicalXor as _logicalXor, logicalNot as _logicalNot,
  all as _all, any as _any,
} from "./p05_compare.js";
import {
  bitwiseAnd as _bitwiseAnd, bitwiseOr as _bitwiseOr, bitwiseXor as _bitwiseXor,
  leftShift as _leftShift, rightShift as _rightShift,
} from "./p05_bitwise.js";

// ---------------------------------------------------------------------------
// Error classes (D-202)
// ---------------------------------------------------------------------------

export class MaskError extends ValueError {}
export const MAError = MaskError;
/** Type name for mask arrays (always bool). */
export const MaskType = "bool" as const;

// ---------------------------------------------------------------------------
// nomask constant (D-201): exactly `false`, like NumPy
// ---------------------------------------------------------------------------

export const nomask: false = false;

// ---------------------------------------------------------------------------
// Helper types
// ---------------------------------------------------------------------------

export type MaskLike = boolean | NDArray | readonly boolean[] | null | undefined;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** True if a bool NDArray has no true elements. */
function _allFalse(mask: NDArray): boolean {
  const flat = mask.ravel();
  const n = flat.size;
  for (let i = 0; i < n; i++) {
    if (flat.item(i) === true) return false;
  }
  return true;
}

/** Converts a flat linear index to N-dimensional coordinates. */
function _flatToCoords(flatIdx: number, shape: number[]): number[] {
  const coords: number[] = new Array<number>(shape.length);
  let rem = flatIdx;
  for (let i = shape.length - 1; i >= 0; i--) {
    coords[i] = rem % (shape[i]!);
    rem = Math.floor(rem / (shape[i]!));
  }
  return coords;
}

/** OR two bool NDArrays element-wise; returns false if result is all-false. */
function _orNDMasks(a: NDArray, b: NDArray, shape: number[]): NDArray | false {
  const ra = a.ravel();
  const rb = b.ravel();
  const n = ra.size;
  const out: boolean[] = new Array<boolean>(n);
  for (let i = 0; i < n; i++) {
    out[i] = (ra.item(i) as boolean) || (rb.item(i) as boolean);
  }
  const result = _array(out, { dtype: "bool" }).reshape(shape) as NDArray;
  return _allFalse(result) ? false : result;
}

/** OR two masks; either may be false (nomask). */
function _orMasks(a: NDArray | false, b: NDArray | false, shape: number[]): NDArray | false {
  if (a === false && b === false) return false;
  if (a === false) return _allFalse(b as NDArray) ? false : b;
  if (b === false) return _allFalse(a as NDArray) ? false : a;
  return _orNDMasks(a, b, shape);
}

/** Coerce a MaskLike to a bool NDArray or false. */
function _toMaskArr(m: MaskLike, shape: number[]): NDArray | false {
  if (m === null || m === false || m === undefined) return false;
  if (m === true) {
    const a = _zeros(shape, { dtype: "bool" }) as NDArray;
    a.fill(true);
    return a;
  }
  if (m instanceof NDArray) {
    return m.dtype.name !== "bool" ? (m.astype("bool") as NDArray) : m;
  }
  if (Array.isArray(m)) {
    return _array(m as boolean[], { dtype: "bool" }) as NDArray;
  }
  return false;
}

/** Broadcast mask to the given shape, or return false. */
function _broadcastMask(mask: NDArray | false, shape: number[]): NDArray | false {
  if (mask === false) return false;
  const ms = mask.shape;
  if (ms.length === shape.length && ms.every((v, i) => v === shape[i])) return mask;
  try {
    return _broadcastTo(mask, shape) as NDArray;
  } catch {
    return mask;
  }
}

// ---------------------------------------------------------------------------
// MaskedArray class (D-200)
// ---------------------------------------------------------------------------

export class MaskedArray {
  /** Underlying data NDArray. */
  readonly _data: NDArray;
  /** Mask: bool NDArray (same shape as data) or false (nomask). */
  _mask: NDArray | false;
  /** Fill value. */
  _fill_value: number | boolean;

  constructor(data: NDArray, mask: NDArray | false, fill_value?: number | boolean) {
    this._data = data;
    this._mask = mask;
    this._fill_value = fill_value ?? _defaultFillValueForDtype(data.dtype.name);
  }

  // ---- basic attributes ----

  get data(): NDArray { return this._data; }
  get mask(): NDArray | false { return this._mask; }
  get fill_value(): number | boolean { return this._fill_value; }
  set fill_value(v: number | boolean) { this._fill_value = v; }
  get shape(): number[] { return this._data.shape; }
  get ndim(): number { return this._data.ndim; }
  get size(): number { return this._data.size; }
  get dtype() { return this._data.dtype; }

  /** Returns data with masked positions filled by fill_value (D-203). */
  filled(fill_value?: number | boolean): NDArray {
    const fv = fill_value ?? this._fill_value;
    if (this._mask === false) return this._data;
    const result = this._data.copy();
    const flatMask = (this._mask as NDArray).ravel();
    const n = flatMask.size;
    for (let i = 0; i < n; i++) {
      if (flatMask.item(i) === true) {
        result.flat.set(i, fv as number);
      }
    }
    return result;
  }

  /** Returns a 1-D array of unmasked data. */
  compressed(): NDArray {
    if (this._mask === false) return this._data.ravel();
    const flat = this._data.ravel();
    const maskFlat = (this._mask as NDArray).ravel();
    const n = flat.size;
    const vals: number[] = [];
    for (let i = 0; i < n; i++) {
      if (!(maskFlat.item(i) as boolean)) vals.push(flat.item(i) as number);
    }
    if (vals.length === 0) return _empty([0], { dtype: this._data.dtype.name }) as NDArray;
    return _array(vals, { dtype: this._data.dtype.name }) as NDArray;
  }

  /** Count of unmasked elements. */
  count(axis?: number | null): number {
    if (this._mask === false) return this._data.size;
    const flat = (this._mask as NDArray).ravel();
    const n = flat.size;
    let cnt = 0;
    for (let i = 0; i < n; i++) {
      if (!(flat.item(i) as boolean)) cnt++;
    }
    void axis;
    return cnt;
  }

  getdata(): NDArray { return this._data; }
  getmask(): NDArray | false { return this._mask; }

  // ---- reductions ----

  sum(opts: { axis?: number | null; keepdims?: boolean; dtype?: DTypeLike } = {}): NDArray | MaskedArray {
    return _maReduce(this, "sum", opts);
  }
  prod(opts: { axis?: number | null; keepdims?: boolean; dtype?: DTypeLike } = {}): NDArray | MaskedArray {
    return _maReduce(this, "prod", opts);
  }
  min(opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray {
    return _maReduce(this, "min", opts);
  }
  max(opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray {
    return _maReduce(this, "max", opts);
  }
  mean(opts: { axis?: number | null; keepdims?: boolean; dtype?: DTypeLike } = {}): NDArray | MaskedArray {
    return _maReduce(this, "mean", opts);
  }
  var(opts: { axis?: number | null; keepdims?: boolean; ddof?: number } = {}): NDArray | MaskedArray {
    return _maReduce(this, "var", opts);
  }
  std(opts: { axis?: number | null; keepdims?: boolean; ddof?: number } = {}): NDArray | MaskedArray {
    return _maReduce(this, "std", opts);
  }
  all(opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray {
    return _maReduce(this, "all", opts);
  }
  any(opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray {
    return _maReduce(this, "any", opts);
  }
  argmin(opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray {
    return _maArgReduce(this, false, opts.axis ?? null, opts.keepdims ?? false);
  }
  argmax(opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray {
    return _maArgReduce(this, true, opts.axis ?? null, opts.keepdims ?? false);
  }
  cumsum(opts: { axis?: number | null; dtype?: DTypeLike } = {}): MaskedArray {
    return _maCumReduce(this, "cumsum", opts);
  }
  cumprod(opts: { axis?: number | null; dtype?: DTypeLike } = {}): MaskedArray {
    return _maCumReduce(this, "cumprod", opts);
  }

  // ---- shape/manipulation ----

  reshape(shape: Shape | number, ...rest: number[]): MaskedArray {
    const target = typeof shape === "number" ? [shape, ...rest] : [...shape];
    return new MaskedArray(
      this._data.reshape(target),
      this._mask === false ? false : (this._mask as NDArray).reshape(target),
      this._fill_value,
    );
  }

  ravel(): MaskedArray {
    return new MaskedArray(
      this._data.ravel(),
      this._mask === false ? false : (this._mask as NDArray).ravel(),
      this._fill_value,
    );
  }

  flatten(): MaskedArray {
    return new MaskedArray(
      this._data.flatten(),
      this._mask === false ? false : (this._mask as NDArray).flatten(),
      this._fill_value,
    );
  }

  transpose(...axes: number[]): MaskedArray {
    return new MaskedArray(
      this._data.transpose(...axes),
      this._mask === false ? false : (this._mask as NDArray).transpose(...axes),
      this._fill_value,
    );
  }

  get T(): MaskedArray { return this.transpose(); }

  squeeze(axis?: number | readonly number[]): MaskedArray {
    return new MaskedArray(
      this._data.squeeze(axis),
      this._mask === false ? false : (this._mask as NDArray).squeeze(axis),
      this._fill_value,
    );
  }

  swapaxes(axis1: number, axis2: number): MaskedArray {
    return new MaskedArray(
      this._data.swapAxes(axis1, axis2),
      this._mask === false ? false : (this._mask as NDArray).swapAxes(axis1, axis2),
      this._fill_value,
    );
  }

  repeat(repeats: number | number[], axis?: number | null): MaskedArray {
    return new MaskedArray(
      this._data.repeat(repeats, axis ?? null),
      this._mask === false ? false : (this._mask as NDArray).repeat(repeats, axis ?? null),
      this._fill_value,
    );
  }

  copy(): MaskedArray {
    return new MaskedArray(
      this._data.copy(),
      this._mask === false ? false : (this._mask as NDArray).copy(),
      this._fill_value,
    );
  }

  sort(opts: { axis?: number; kind?: string } = {}): MaskedArray {
    const filled = this.filled(maximum_fill_value(this) as number);
    const sorted = filled.copy();
    sorted.sort({ axis: opts.axis, kind: opts.kind });
    // Build new mask: masked values went to end, so last (count_masked) elements masked
    if (this._mask === false) return new MaskedArray(sorted, false, this._fill_value);
    const maskSorted = (this._mask as NDArray).copy();
    maskSorted.sort({ axis: opts.axis, kind: opts.kind });
    return new MaskedArray(sorted, _allFalse(maskSorted) ? false : maskSorted, this._fill_value);
  }

  argsort(opts: { axis?: number; kind?: string } = {}): NDArray {
    const filled = this.filled(maximum_fill_value(this) as number);
    return _argsort(filled, { axis: opts.axis ?? -1, kind: opts.kind });
  }

  take(indices: NDArray | number[], axis?: number | null): MaskedArray {
    const idx = indices instanceof NDArray ? indices : _array(indices, { dtype: "int64" }) as NDArray;
    return new MaskedArray(
      this._data.take(idx, axis),
      this._mask === false ? false : (this._mask as NDArray).take(idx, axis),
      this._fill_value,
    );
  }

  put(indices: NDArray | number[], values: MaskedArray | NDArray | number[]): void {
    const idx = indices instanceof NDArray ? indices : _array(indices, { dtype: "int64" }) as NDArray;
    const vals = _toMa(values);
    this._data.put(idx, vals._data);
    if (vals._mask !== false || this._mask !== false) {
      const currentMask = this._mask === false
        ? (_zeros(this.shape, { dtype: "bool" }) as NDArray)
        : (this._mask as NDArray).copy();
      const vm = vals._mask === false
        ? (_zeros(vals.shape, { dtype: "bool" }) as NDArray)
        : (vals._mask as NDArray);
      currentMask.put(idx, vm);
      this._mask = _allFalse(currentMask) ? false : currentMask;
    }
  }

  harden_mask(): void { /* no-op for API completeness */ }
  soften_mask(): void { /* no-op for API completeness */ }

  toString(): string {
    if (this._mask === false) return `masked_array(data=${JSON.stringify(this._data.tolist())}, mask=${nomask})`;
    return `masked_array(data=${_printData(this)}, mask=${_printMask(this._mask as NDArray)})`;
  }
}

// ---- print helpers ----

function _printData(ma: MaskedArray): string {
  const flat = ma._data.ravel();
  const maskFlat = ma._mask === false ? null : (ma._mask as NDArray).ravel();
  const n = flat.size;
  const parts: string[] = [];
  for (let i = 0; i < n; i++) {
    parts.push(maskFlat && (maskFlat.item(i) as boolean) ? masked_print_option.value : String(flat.item(i)));
  }
  return `[${parts.join(", ")}]`;
}

function _printMask(mask: NDArray): string {
  const flat = mask.ravel();
  const n = flat.size;
  return `[${Array.from({ length: n }, (_, i) => String(flat.item(i))).join(", ")}]`;
}

// ---------------------------------------------------------------------------
// masked singleton and print option (D-201)
// ---------------------------------------------------------------------------

export const masked: MaskedArray = (() => {
  const m = _zeros([1], { dtype: "bool" }) as NDArray;
  m.fill(true);
  return new MaskedArray(_array([0], { dtype: "float64" }) as NDArray, m, 1e20);
})();

export const masked_singleton: MaskedArray = masked;

export class MaskedPrintOption {
  value: string = "--";
  set(s: string): void { this.value = s; }
  toString(): string { return this.value; }
}

export const masked_print_option = new MaskedPrintOption();

// ---------------------------------------------------------------------------
// MaskedIterator (D-202)
// ---------------------------------------------------------------------------

export class MaskedIterator implements Iterator<MaskedArray | number | boolean> {
  private _ma: MaskedArray;
  private _index: number;

  constructor(ma: MaskedArray) {
    this._ma = ma;
    this._index = 0;
  }

  next(): IteratorResult<MaskedArray | number | boolean> {
    if (this._index >= this._ma.size) return { done: true, value: undefined as unknown as MaskedArray };
    const flat = this._ma._data.ravel();
    const maskFlat = this._ma._mask === false ? null : (this._ma._mask as NDArray).ravel();
    const isMasked = maskFlat ? (maskFlat.item(this._index) as boolean) : false;
    const value = isMasked ? masked : (flat.item(this._index) as number | boolean);
    this._index++;
    return { done: false, value };
  }

  [Symbol.iterator](): this { return this; }
}

// ---------------------------------------------------------------------------
// Default fill values (D-209)
// ---------------------------------------------------------------------------

function _defaultFillValueForDtype(dtname: string): number | boolean {
  if (dtname === "bool") return true;
  if (dtname.startsWith("int") || dtname.startsWith("uint")) return 999999;
  return 1e20;
}

function _defaultFillValueForValue(v: unknown): number | boolean {
  if (typeof v === "boolean") return true;
  if (typeof v === "number" && Number.isInteger(v)) return 999999;
  return 1e20;
}

export function default_fill_value(obj: MaskedArray | NDArray | number | boolean): number | boolean {
  if (obj instanceof MaskedArray) return _defaultFillValueForDtype(obj._data.dtype.name);
  if (obj instanceof NDArray) return _defaultFillValueForDtype(obj.dtype.name);
  return _defaultFillValueForValue(obj);
}

export function maximum_fill_value(obj: MaskedArray | NDArray): number | boolean {
  const dtname = obj instanceof MaskedArray ? obj._data.dtype.name : (obj as NDArray).dtype.name;
  if (dtname === "bool") return false;
  if (dtname === "int8") return 127;
  if (dtname === "uint8") return 255;
  if (dtname === "int16") return 32767;
  if (dtname === "uint16") return 65535;
  if (dtname === "int32") return 2147483647;
  if (dtname === "uint32") return 4294967295;
  if (dtname === "int64" || dtname === "uint64") return Number.MAX_SAFE_INTEGER;
  if (dtname === "float32") return 3.4028235e38;
  return 1.7976931348623157e308;
}

export function minimum_fill_value(obj: MaskedArray | NDArray): number | boolean {
  const dtname = obj instanceof MaskedArray ? obj._data.dtype.name : (obj as NDArray).dtype.name;
  if (dtname === "bool") return true;
  if (dtname === "int8") return -128;
  if (dtname === "uint8") return 0;
  if (dtname === "int16") return -32768;
  if (dtname === "uint16") return 0;
  if (dtname === "int32") return -2147483648;
  if (dtname === "uint32") return 0;
  if (dtname === "int64") return Number.MIN_SAFE_INTEGER;
  if (dtname === "uint64") return 0;
  if (dtname === "float32") return -3.4028235e38;
  return -1.7976931348623157e308;
}

export function set_fill_value(a: MaskedArray, fill_value: number | boolean): void {
  a.fill_value = fill_value;
}

export function common_fill_value(a: MaskedArray, b: MaskedArray): number | boolean | typeof masked {
  return a.fill_value === b.fill_value ? a.fill_value : masked;
}

// ---------------------------------------------------------------------------
// Coerce to MaskedArray
// ---------------------------------------------------------------------------

function _toMa(x: MaskedArray | NDArray | ArrayLike | number | boolean): MaskedArray {
  if (x instanceof MaskedArray) return x;
  if (x instanceof NDArray) return new MaskedArray(x, false);
  if (typeof x === "number" || typeof x === "boolean") {
    return new MaskedArray(_array([x]) as NDArray, false);
  }
  return new MaskedArray(_array(x as ArrayLike) as NDArray, false);
}

// ---------------------------------------------------------------------------
// Reduction helpers (D-206)
// ---------------------------------------------------------------------------

function _fillForOp(op: string, dt: string): number | boolean {
  switch (op) {
    case "sum": case "cumsum": return 0;
    case "prod": case "cumprod": return 1;
    case "min": return maximum_fill_value({ dtype: toDType(dt) } as NDArray);
    case "max": return minimum_fill_value({ dtype: toDType(dt) } as NDArray);
    case "all": return true;
    case "any": return false;
    default: return 0;
  }
}

function _computeOutMask(
  mask: NDArray, shape: number[], axis: number | readonly number[] | null, keepdims: boolean,
): NDArray | false {
  if (axis === null) {
    const flat = mask.ravel();
    const n = flat.size;
    for (let i = 0; i < n; i++) if (!(flat.item(i) as boolean)) return false;
    const r = _ones([1], { dtype: "bool" }) as NDArray;
    return keepdims ? r.reshape(shape.map(() => 1)) : r;
  }
  const axes = typeof axis === "number" ? [axis] : [...axis];
  const outShape = keepdims
    ? shape.map((s, i) => (axes.includes(i) ? 1 : s))
    : shape.filter((_, i) => !axes.includes(i));
  if (outShape.length === 0) outShape.push(1);
  const total = outShape.reduce((a, b) => a * b, 1);
  const outMaskArr: boolean[] = new Array<boolean>(total).fill(true);
  const axLens: number[] = axes.map((ax) => shape[ax]!);
  for (let flatIdx = 0; flatIdx < total; flatIdx++) {
    const outCoords = _flatToCoords(flatIdx, outShape);
    const numKCombos = axLens.reduce((a, b) => a * b, 1);
    outer: for (let k = 0; k < numKCombos; k++) {
      const kCoords: number[] = [];
      let rem = k;
      for (let ai = axLens.length - 1; ai >= 0; ai--) {
        kCoords.unshift(rem % axLens[ai]!);
        rem = Math.floor(rem / axLens[ai]!);
      }
      const inCoords: number[] = [];
      let outI = 0; let kI = 0;
      for (let di = 0; di < shape.length; di++) {
        if (axes.includes(di)) inCoords.push(kCoords[kI++]!);
        else inCoords.push(outCoords[outI++]!);
      }
      if (!(mask.item(...inCoords) as boolean)) { outMaskArr[flatIdx] = false; break outer; }
    }
  }
  const result = _array(outMaskArr, { dtype: "bool" }).reshape(outShape) as NDArray;
  return _allFalse(result) ? false : result;
}

function _maReduce(
  ma: MaskedArray, op: string,
  opts: { axis?: number | null; keepdims?: boolean; dtype?: DTypeLike; ddof?: number },
): NDArray | MaskedArray {
  const fv = _fillForOp(op, ma._data.dtype.name);
  const filled = ma.filled(fv as number);

  let result: NDArray;
  const axis = opts.axis ?? null;
  const keepdims = opts.keepdims ?? false;

  switch (op) {
    case "all":   result = _all(filled, { axis, keepdims }); break;
    case "any":   result = _any(filled, { axis, keepdims }); break;
    case "min":   result = filled.min({ axis, keepdims }); break;
    case "max":   result = filled.max({ axis, keepdims }); break;
    case "mean":  result = filled.mean({ axis, keepdims, dtype: opts.dtype }); break;
    case "var":   result = filled.var({ axis, keepdims, ddof: opts.ddof ?? 0 }); break;
    case "std":   result = filled.std({ axis, keepdims, ddof: opts.ddof ?? 0 }); break;
    case "sum":   result = filled.sum({ axis, keepdims, dtype: opts.dtype }); break;
    case "prod":  result = filled.prod({ axis, keepdims, dtype: opts.dtype }); break;
    default:      result = filled.sum({ axis, keepdims }); break;
  }

  if (ma._mask === false) return result;
  const outMask = _computeOutMask(ma._mask as NDArray, ma.shape, axis, keepdims);
  if (outMask === false) return result;
  return new MaskedArray(result, outMask, ma._fill_value);
}

function _maArgReduce(ma: MaskedArray, isMax: boolean, axis: number | null, keepdims: boolean): NDArray {
  const fv = (isMax ? minimum_fill_value(ma) : maximum_fill_value(ma)) as number;
  const filled = ma.filled(fv);
  return isMax ? filled.argmax({ axis: axis ?? undefined, keepdims }) : filled.argmin({ axis: axis ?? undefined, keepdims });
}

function _maCumReduce(
  ma: MaskedArray, op: "cumsum" | "cumprod", opts: { axis?: number | null; dtype?: DTypeLike },
): MaskedArray {
  const fv = op === "cumsum" ? 0 : 1;
  const filled = ma.filled(fv);
  // Implement cumsum/cumprod by ravel + prefix scan
  const flat = filled.ravel();
  const n = flat.size;
  const out: number[] = new Array<number>(n);
  out[0] = flat.item(0) as number;
  for (let i = 1; i < n; i++) {
    out[i] = op === "cumsum"
      ? (out[i - 1]! + (flat.item(i) as number))
      : (out[i - 1]! * (flat.item(i) as number));
  }
  const result = _array(out, { dtype: opts.dtype ?? filled.dtype.name }).reshape(filled.shape) as NDArray;
  if (ma._mask === false) return new MaskedArray(result, false, ma._fill_value);
  const maskFlat = (ma._mask as NDArray).ravel();
  const outMaskArr: boolean[] = [];
  for (let i = 0; i < n; i++) outMaskArr.push(maskFlat.item(i) as boolean);
  const outMask = _array(outMaskArr, { dtype: "bool" }).reshape(result.shape) as NDArray;
  return new MaskedArray(result, _allFalse(outMask) ? false : outMask, ma._fill_value);
}

// ---------------------------------------------------------------------------
// Constructors / factory functions (D-203)
// ---------------------------------------------------------------------------

export interface MaskedArrayOptions {
  mask?: MaskLike;
  fill_value?: number | boolean;
  dtype?: DTypeLike;
  copy?: boolean;
}

export function masked_array(
  data: MaskedArray | NDArray | ArrayLike,
  opts: MaskedArrayOptions = {},
): MaskedArray {
  let baseData: NDArray;
  let baseMask: NDArray | false;

  if (data instanceof MaskedArray) {
    baseData = opts.copy ? data._data.copy() : data._data;
    baseMask = data._mask === false ? false
      : (opts.copy ? (data._mask as NDArray).copy() : (data._mask as NDArray));
  } else if (data instanceof NDArray) {
    baseData = opts.copy ? data.copy() : data;
    baseMask = false;
  } else {
    baseData = _array(data as ArrayLike, { dtype: opts.dtype }) as NDArray;
    baseMask = false;
  }

  if (opts.dtype && !(data instanceof NDArray && !opts.copy)) {
    baseData = baseData.astype(opts.dtype);
  }

  const shape = baseData.shape;
  if (opts.mask !== undefined) {
    const rawMask = _toMaskArr(opts.mask, shape);
    baseMask = rawMask === false ? false : (_broadcastMask(rawMask, shape) ?? false);
  }

  return new MaskedArray(
    baseData, baseMask,
    opts.fill_value ?? _defaultFillValueForDtype(baseData.dtype.name),
  );
}

export const array = masked_array;
export function asarray(data: MaskedArray | NDArray | ArrayLike, dtype?: DTypeLike): MaskedArray {
  return masked_array(data, { copy: false, dtype });
}
export const asanyarray = asarray;

// Predicate constructors (D-203)
export function masked_where(
  condition: NDArray | ArrayLike | boolean[],
  a: MaskedArray | NDArray | ArrayLike,
  copy = true,
): MaskedArray {
  const cond = condition instanceof NDArray ? condition : _array(condition as ArrayLike, { dtype: "bool" }) as NDArray;
  const ma = _toMa(a);
  const baseData = copy ? ma._data.copy() : ma._data;
  const newMask = _orMasks(cond, ma._mask, baseData.shape);
  return new MaskedArray(baseData, newMask, ma._fill_value);
}

function _cmpWhere(
  a: MaskedArray | NDArray | ArrayLike, value: number,
  op: "eq" | "ne" | "gt" | "ge" | "lt" | "le",
): MaskedArray {
  const ma = _toMa(a);
  const flat = ma._data.ravel();
  const n = flat.size;
  const condArr: boolean[] = new Array<boolean>(n);
  for (let i = 0; i < n; i++) {
    const v = flat.item(i) as number;
    condArr[i] = op === "eq" ? v === value : op === "ne" ? v !== value
      : op === "gt" ? v > value : op === "ge" ? v >= value
      : op === "lt" ? v < value : v <= value;
  }
  const cond = _array(condArr, { dtype: "bool" }).reshape(ma.shape) as NDArray;
  return masked_where(cond, ma);
}

export const masked_equal = (a: MaskedArray | NDArray | ArrayLike, value: number): MaskedArray => _cmpWhere(a, value, "eq");
export const masked_not_equal = (a: MaskedArray | NDArray | ArrayLike, value: number): MaskedArray => _cmpWhere(a, value, "ne");
export const masked_greater = (a: MaskedArray | NDArray | ArrayLike, value: number): MaskedArray => _cmpWhere(a, value, "gt");
export const masked_greater_equal = (a: MaskedArray | NDArray | ArrayLike, value: number): MaskedArray => _cmpWhere(a, value, "ge");
export const masked_less = (a: MaskedArray | NDArray | ArrayLike, value: number): MaskedArray => _cmpWhere(a, value, "lt");
export const masked_less_equal = (a: MaskedArray | NDArray | ArrayLike, value: number): MaskedArray => _cmpWhere(a, value, "le");

export function masked_inside(a: MaskedArray | NDArray | ArrayLike, v1: number, v2: number): MaskedArray {
  const ma = _toMa(a);
  const lo = Math.min(v1, v2); const hi = Math.max(v1, v2);
  const flat = ma._data.ravel(); const n = flat.size;
  const condArr: boolean[] = [];
  for (let i = 0; i < n; i++) { const v = flat.item(i) as number; condArr.push(v >= lo && v <= hi); }
  return masked_where(_array(condArr, { dtype: "bool" }).reshape(ma.shape) as NDArray, ma);
}

export function masked_outside(a: MaskedArray | NDArray | ArrayLike, v1: number, v2: number): MaskedArray {
  const ma = _toMa(a);
  const lo = Math.min(v1, v2); const hi = Math.max(v1, v2);
  const flat = ma._data.ravel(); const n = flat.size;
  const condArr: boolean[] = [];
  for (let i = 0; i < n; i++) { const v = flat.item(i) as number; condArr.push(v < lo || v > hi); }
  return masked_where(_array(condArr, { dtype: "bool" }).reshape(ma.shape) as NDArray, ma);
}

export function masked_invalid(a: MaskedArray | NDArray | ArrayLike, copy = true): MaskedArray {
  const ma = _toMa(a);
  const flat = ma._data.ravel(); const n = flat.size;
  const condArr: boolean[] = [];
  for (let i = 0; i < n; i++) { const v = flat.item(i) as number; condArr.push(!isFinite(v) || isNaN(v)); }
  return masked_where(_array(condArr, { dtype: "bool" }).reshape(ma.shape) as NDArray, ma, copy);
}

export function masked_values(
  a: MaskedArray | NDArray | ArrayLike, value: number,
  opts: { rtol?: number; atol?: number; copy?: boolean } = {},
): MaskedArray {
  const rtol = opts.rtol ?? 1e-5; const atol = opts.atol ?? 1e-8;
  const ma = _toMa(a);
  const flat = ma._data.ravel(); const n = flat.size;
  const condArr: boolean[] = [];
  for (let i = 0; i < n; i++) {
    condArr.push(Math.abs((flat.item(i) as number) - value) <= atol + rtol * Math.abs(value));
  }
  const result = masked_where(_array(condArr, { dtype: "bool" }).reshape(ma.shape) as NDArray, ma, opts.copy !== false);
  result.fill_value = value;
  return result;
}

export function masked_object(a: MaskedArray | NDArray | ArrayLike, value: unknown, copy = true): MaskedArray {
  const ma = _toMa(a);
  const flat = ma._data.ravel(); const n = flat.size;
  const condArr: boolean[] = [];
  for (let i = 0; i < n; i++) condArr.push(flat.item(i) === value);
  return masked_where(_array(condArr, { dtype: "bool" }).reshape(ma.shape) as NDArray, ma, copy);
}

export function masked_all(shape: Shape | number, dtype: DTypeLike = "float64"): MaskedArray {
  const sh = typeof shape === "number" ? [shape] : [...shape];
  const data = _empty(sh, { dtype }) as NDArray;
  const mask = _ones(sh, { dtype: "bool" }) as NDArray;
  return new MaskedArray(data, mask, _defaultFillValueForDtype(toDType(dtype).name));
}

export function masked_all_like(a: MaskedArray | NDArray): MaskedArray {
  const data = a instanceof MaskedArray ? a._data : a;
  return masked_all(data.shape, data.dtype.name);
}

export function fix_invalid(a: MaskedArray | NDArray | ArrayLike, fill_value?: number): MaskedArray {
  const ma = masked_invalid(a, true);
  const fv = fill_value ?? (ma._fill_value as number);
  if (ma._mask !== false) {
    const flatMask = (ma._mask as NDArray).ravel();
    const n = flatMask.size;
    for (let i = 0; i < n; i++) {
      if (flatMask.item(i) === true) ma._data.flat.set(i, fv);
    }
  }
  return new MaskedArray(ma._data, ma._mask, fv);
}

// ---------------------------------------------------------------------------
// Mask utilities (D-204)
// ---------------------------------------------------------------------------

export function make_mask(
  m: MaskLike | NDArray | boolean, copy = false, shrink = true, dtype: DTypeLike = "bool",
): NDArray | false {
  if (m === false || m === null || m === undefined) return nomask;
  let arr: NDArray;
  if (m instanceof NDArray) arr = copy ? m.astype(dtype, { copy: true }) : m.astype(dtype);
  else if (m === true) arr = _array([true], { dtype: "bool" }) as NDArray;
  else if (Array.isArray(m)) arr = _array(m as boolean[], { dtype: "bool" }) as NDArray;
  else return nomask;
  if (shrink && _allFalse(arr)) return nomask;
  return arr;
}

export function make_mask_none(newshape: Shape | number, dtype: DTypeLike = "bool"): NDArray {
  const sh = typeof newshape === "number" ? [newshape] : [...newshape];
  return _zeros(sh, { dtype }) as NDArray;
}

export function make_mask_descr(_ndtype: unknown): "bool" { return "bool"; }

export function getmask(a: MaskedArray | NDArray): NDArray | false {
  if (a instanceof MaskedArray) return a._mask;
  return nomask;
}

export function getmaskarray(a: MaskedArray | NDArray): NDArray {
  if (a instanceof MaskedArray) {
    if (a._mask === false) return _zeros(a.shape, { dtype: "bool" }) as NDArray;
    return a._mask as NDArray;
  }
  return _zeros((a as NDArray).shape, { dtype: "bool" }) as NDArray;
}

export function getdata(a: MaskedArray | NDArray): NDArray {
  return a instanceof MaskedArray ? a._data : a;
}

export function filled(a: MaskedArray | NDArray, fill_value?: number | boolean): NDArray {
  return a instanceof MaskedArray ? a.filled(fill_value) : a;
}

export function is_masked(a: unknown): boolean {
  if (!(a instanceof MaskedArray)) return false;
  if (a._mask === false) return false;
  return !_allFalse(a._mask as NDArray);
}

export function is_mask(m: unknown): boolean {
  if (m === false) return true;
  return m instanceof NDArray && m.dtype.name === "bool";
}

export function mask_or(
  m1: NDArray | false | null, m2: NDArray | false | null, copy = false, shrink = true,
): NDArray | false {
  const a = m1 ?? false; const b = m2 ?? false;
  if (a === false && b === false) return nomask;
  if (a === false) return copy ? (b as NDArray).copy() : (b as NDArray);
  if (b === false) return copy ? (a as NDArray).copy() : (a as NDArray);
  const shape = (a as NDArray).shape;
  const result = _orNDMasks(a as NDArray, b as NDArray, shape);
  if (shrink && result !== false && _allFalse(result as NDArray)) return nomask;
  return result;
}

export function flatten_mask(mask: NDArray | boolean): NDArray | false {
  if (mask === false) return nomask;
  if (mask === true) return _array([true], { dtype: "bool" }) as NDArray;
  if (mask instanceof NDArray) return mask.ravel();
  return nomask;
}

export function shrink_mask(a: MaskedArray): MaskedArray {
  if (a._mask === false) return a;
  if (_allFalse(a._mask as NDArray)) return new MaskedArray(a._data, false, a._fill_value);
  return a;
}

// ---------------------------------------------------------------------------
// Arithmetic that propagates mask (D-205)
// ---------------------------------------------------------------------------

type MaOperand = MaskedArray | NDArray | ArrayLike | number | boolean;

function _binaryMa(a: MaOperand, b: MaOperand, op: (x: NDArray, y: NDArray) => NDArray): MaskedArray {
  const ma = _toMa(a); const mb = _toMa(b);
  let resultData: NDArray;
  try { resultData = op(ma._data, mb._data); }
  catch { resultData = _zeros(ma.shape, { dtype: ma._data.dtype.name }) as NDArray; }
  const outMask = _orMasks(
    _broadcastMask(ma._mask, resultData.shape),
    _broadcastMask(mb._mask, resultData.shape),
    resultData.shape,
  );
  return new MaskedArray(resultData, outMask, _defaultFillValueForDtype(resultData.dtype.name));
}

function _unaryMa(a: MaOperand, op: (x: NDArray) => NDArray): MaskedArray {
  const ma = _toMa(a);
  const resultData = op(ma._data);
  const outMask = ma._mask === false ? false : (ma._mask as NDArray).copy();
  return new MaskedArray(resultData, outMask, _defaultFillValueForDtype(resultData.dtype.name));
}

// arithmetic
export const add = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _add);
export const subtract = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _subtract);
export const multiply = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _multiply);
export const divide = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _divide);
export const true_divide = divide;
export const floor_divide = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _floorDivide);
export const power = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _power);
export const mod = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _mod);
export const remainder = mod;
export const fmod = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _fmod);
export const negative = (a: MaOperand): MaskedArray => _unaryMa(a, _negative);
export const absolute = (a: MaOperand): MaskedArray => _unaryMa(a, _abs);
export const abs = absolute;
export const fabs = absolute;

// comparisons
export const equal = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _equal);
export const not_equal = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _notEqual);
export const less = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _less);
export const less_equal = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _lessEqual);
export const greater = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _greater);
export const greater_equal = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _greaterEqual);

// math
export const sqrt = (a: MaOperand): MaskedArray => _unaryMa(a, _sqrt);
export const exp = (a: MaOperand): MaskedArray => _unaryMa(a, _exp);
export const log = (a: MaOperand): MaskedArray => _unaryMa(a, _log);
export const log2 = (a: MaOperand): MaskedArray => _unaryMa(a, _log2);
export const log10 = (a: MaOperand): MaskedArray => _unaryMa(a, _log10);
export const sin = (a: MaOperand): MaskedArray => _unaryMa(a, _sin);
export const cos = (a: MaOperand): MaskedArray => _unaryMa(a, _cos);
export const tan = (a: MaOperand): MaskedArray => _unaryMa(a, _tan);
export const arcsin = (a: MaOperand): MaskedArray => _unaryMa(a, _arcsin);
export const arccos = (a: MaOperand): MaskedArray => _unaryMa(a, _arccos);
export const arctan = (a: MaOperand): MaskedArray => _unaryMa(a, _arctan);
export const arctan2 = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _arctan2);
export const sinh = (a: MaOperand): MaskedArray => _unaryMa(a, _sinh);
export const cosh = (a: MaOperand): MaskedArray => _unaryMa(a, _cosh);
export const tanh = (a: MaOperand): MaskedArray => _unaryMa(a, _tanh);
export const arcsinh = (a: MaOperand): MaskedArray => _unaryMa(a, _arcsinh);
export const arccosh = (a: MaOperand): MaskedArray => _unaryMa(a, _arccosh);
export const arctanh = (a: MaOperand): MaskedArray => _unaryMa(a, _arctanh);
export const hypot = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _hypot);
export const ceil = (a: MaOperand): MaskedArray => _unaryMa(a, _ceil);
export const floor = (a: MaOperand): MaskedArray => _unaryMa(a, _floor);
export const conjugate = (a: MaOperand): MaskedArray => _unaryMa(a, (x) => x.conjugate());
export const angle = (a: MaOperand): MaskedArray => _unaryMa(a, _angle);

// bitwise
export const bitwise_and = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _bitwiseAnd);
export const bitwise_or = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _bitwiseOr);
export const bitwise_xor = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _bitwiseXor);
export const left_shift = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _leftShift);
export const right_shift = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _rightShift);
export const logical_and = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _logicalAnd);
export const logical_or = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _logicalOr);
export const logical_xor = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _logicalXor);
export const logical_not = (a: MaOperand): MaskedArray => _unaryMa(a, _logicalNot);

// max/min ufuncs
export const maximum = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _maximum);
export const minimum = (a: MaOperand, b: MaOperand): MaskedArray => _binaryMa(a, b, _minimum);

// ---------------------------------------------------------------------------
// Reductions (module-level) (D-206)
// ---------------------------------------------------------------------------

export const sum = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean; dtype?: DTypeLike } = {}): NDArray | MaskedArray =>
  _toMa(a).sum(opts);
export const prod = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean; dtype?: DTypeLike } = {}): NDArray | MaskedArray =>
  _toMa(a).prod(opts);
export const product = prod;
export const sometrue = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray =>
  _toMa(a).any(opts);
export const alltrue = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray =>
  _toMa(a).all(opts);
export const min = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray =>
  _toMa(a).min(opts);
export const amin = min;
export const max = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray =>
  _toMa(a).max(opts);
export const amax = max;
export const mean = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean; dtype?: DTypeLike } = {}): NDArray | MaskedArray =>
  _toMa(a).mean(opts);
export const var_ = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean; ddof?: number } = {}): NDArray | MaskedArray =>
  _toMa(a).var(opts);
export const std = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean; ddof?: number } = {}): NDArray | MaskedArray =>
  _toMa(a).std(opts);
export const all = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray =>
  _toMa(a).all(opts);
export const any = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray =>
  _toMa(a).any(opts);
export const argmin = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray =>
  _toMa(a).argmin(opts);
export const argmax = (a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray =>
  _toMa(a).argmax(opts);
export const cumsum = (a: MaOperand, opts: { axis?: number | null; dtype?: DTypeLike } = {}): MaskedArray =>
  _toMa(a).cumsum(opts);
export const cumprod = (a: MaOperand, opts: { axis?: number | null; dtype?: DTypeLike } = {}): MaskedArray =>
  _toMa(a).cumprod(opts);

export function count(a: MaskedArray | NDArray, _axis?: number | null): number {
  return _toMa(a).count();
}

export function count_masked(arr: MaskedArray | NDArray, _axis?: number | null): number {
  const ma = _toMa(arr);
  if (ma._mask === false) return 0;
  const flat = (ma._mask as NDArray).ravel();
  let cnt = 0;
  for (let i = 0; i < flat.size; i++) if (flat.item(i) === true) cnt++;
  return cnt;
}

export function ptp(a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray {
  const hi = max(a, opts);
  const lo = min(a, opts);
  return subtract(hi instanceof MaskedArray ? hi : _toMa(hi), lo instanceof MaskedArray ? lo : _toMa(lo));
}

export function anom(a: MaOperand, opts: { axis?: number | null } = {}): MaskedArray {
  const ma = _toMa(a);
  const m = mean(ma, { axis: opts.axis ?? null });
  return subtract(ma, m instanceof MaskedArray ? m : _toMa(m));
}
export const anomalies = anom;

export function average(
  a: MaOperand,
  opts: { axis?: number | null; weights?: NDArray | number[] | null; returned?: boolean } = {},
): MaskedArray | [MaskedArray, NDArray] {
  const ma = _toMa(a);
  const w = opts.weights;
  if (!w) {
    const result = mean(ma, { axis: opts.axis ?? null });
    const mResult = result instanceof MaskedArray ? result : _toMa(result);
    if (opts.returned) return [mResult, _array([ma.count()]) as NDArray];
    return mResult;
  }
  const wMa = _toMa(w instanceof NDArray ? w : _array(w as number[]) as NDArray);
  const num = sum(multiply(ma, wMa), { axis: opts.axis ?? null });
  const den = sum(wMa, { axis: opts.axis ?? null });
  const result = divide(num instanceof MaskedArray ? num : _toMa(num), den instanceof MaskedArray ? den : _toMa(den));
  if (opts.returned) return [result, den instanceof MaskedArray ? den._data : den as NDArray];
  return result;
}

export function median(a: MaOperand, opts: { axis?: number | null; keepdims?: boolean } = {}): NDArray | MaskedArray {
  const ma = _toMa(a);
  const fv = maximum_fill_value(ma) as number;
  const filled_ = ma.filled(fv);
  void opts;
  return filled_.mean({ axis: opts.axis ?? null, keepdims: opts.keepdims });
}

// ---------------------------------------------------------------------------
// Shape/manipulation (D-207 module-level)
// ---------------------------------------------------------------------------

export const reshape = (a: MaOperand, newshape: Shape): MaskedArray => _toMa(a).reshape(newshape);
export const ravel = (a: MaOperand): MaskedArray => _toMa(a).ravel();
export const squeeze = (a: MaOperand, axis?: number | readonly number[]): MaskedArray => _toMa(a).squeeze(axis);
export const transpose = (a: MaOperand, axes?: number[]): MaskedArray =>
  axes ? _toMa(a).transpose(...axes) : _toMa(a).transpose();
export function expand_dims(a: MaOperand, axis: number): MaskedArray {
  const ma = _toMa(a);
  const sh = [...ma.shape];
  const ax = axis < 0 ? ma.ndim + 1 + axis : axis;
  sh.splice(ax, 0, 1);
  return ma.reshape(sh);
}
export const repeat = (a: MaOperand, repeats: number | number[], axis?: number | null): MaskedArray =>
  _toMa(a).repeat(repeats, axis ?? null);
export const sort = (a: MaOperand, opts: { axis?: number; kind?: string } = {}): MaskedArray =>
  _toMa(a).sort(opts);
export const argsort = (a: MaOperand, opts: { axis?: number; kind?: string } = {}): NDArray =>
  _toMa(a).argsort(opts);
export const take = (a: MaOperand, indices: NDArray | number[], axis?: number | null): MaskedArray =>
  _toMa(a).take(indices, axis ?? null);
export const put = (a: MaskedArray, indices: NDArray | number[], values: MaOperand): void =>
  a.put(indices, values instanceof MaskedArray ? values : _toMa(values));
export const swapaxes = (a: MaOperand, axis1: number, axis2: number): MaskedArray =>
  _toMa(a).swapaxes(axis1, axis2);
export const shape = (a: MaOperand): number[] => _toMa(a).shape;
export const size = (a: MaOperand, axis?: number): number => {
  const ma = _toMa(a); return axis === undefined ? ma.size : (ma.shape[axis] ?? 0);
};
export const ndim = (a: MaOperand): number => _toMa(a).ndim;

export function resize(a: MaOperand, newshape: Shape | number): MaskedArray {
  const ma = _toMa(a);
  const sh = typeof newshape === "number" ? [newshape] : [...newshape];
  const n = sh.reduce((x, y) => x * y, 1);
  const flat = ma.ravel();
  const flatData = flat._data; const flatMask = flat._mask;
  const curN = flatData.size;
  const outData: number[] = [];
  const outMask: boolean[] = [];
  for (let i = 0; i < n; i++) {
    const j = i % curN;
    outData.push(flatData.item(j) as number);
    if (flatMask !== false) outMask.push((flatMask as NDArray).item(j) as boolean);
  }
  const newData = _array(outData, { dtype: ma._data.dtype.name }).reshape(sh) as NDArray;
  const newMask = flatMask === false ? false
    : (_array(outMask, { dtype: "bool" }).reshape(sh) as NDArray);
  return new MaskedArray(newData, newMask === false ? false : (_allFalse(newMask as NDArray) ? false : newMask), ma._fill_value);
}

export function diagonal(a: MaOperand, offset = 0, axis1 = 0, axis2 = 1): MaskedArray {
  const ma = _toMa(a);
  const newData = ma._data.diagonal({ offset, axis1, axis2 });
  const newMask = ma._mask === false ? false : (ma._mask as NDArray).diagonal({ offset, axis1, axis2 });
  return new MaskedArray(newData, newMask, ma._fill_value);
}

export function trace(a: MaOperand, offset = 0, axis1 = 0, axis2 = 1): NDArray | MaskedArray {
  return sum(diagonal(a, offset, axis1, axis2));
}

// ---------------------------------------------------------------------------
// Concatenate/stack (D-208)
// ---------------------------------------------------------------------------

function _catMasked(arrays: MaOperand[], axis = 0): MaskedArray {
  const mas = arrays.map(_toMa);
  const datas = mas.map((m) => m._data);
  const masks = mas.map((m) => m._mask);
  const catData = _concatenate(datas, axis);
  if (masks.every((m) => m === false)) return new MaskedArray(catData, false);
  const maskArrs = mas.map((m) =>
    m._mask === false ? (_zeros(m.shape, { dtype: "bool" }) as NDArray) : (m._mask as NDArray),
  );
  const catMask = _concatenate(maskArrs, axis);
  return new MaskedArray(catData, _allFalse(catMask) ? false : catMask);
}

export const concatenate = (arrays: MaOperand[], axis = 0): MaskedArray => _catMasked(arrays, axis);

export function vstack(tup: MaOperand[]): MaskedArray {
  const mas = tup.map((a) => { const m = _toMa(a); return m.ndim < 2 ? m.reshape([1, ...m.shape]) : m; });
  return _catMasked(mas, 0);
}
export const row_stack = vstack;

export function hstack(tup: MaOperand[]): MaskedArray {
  const mas = tup.map(_toMa);
  return _catMasked(mas, (mas[0]?.ndim ?? 1) === 1 ? 0 : 1);
}

export function dstack(tup: MaOperand[]): MaskedArray {
  const mas = tup.map((a) => {
    let m = _toMa(a);
    if (m.ndim === 1) m = m.reshape([1, m.size, 1]);
    else if (m.ndim === 2) m = m.reshape([m.shape[0]!, m.shape[1]!, 1]);
    return m;
  });
  return _catMasked(mas, 2);
}

export function stack(arrays: MaOperand[], axis = 0): MaskedArray {
  const expanded = arrays.map((a) => {
    const m = _toMa(a); const sh = [...m.shape]; sh.splice(axis, 0, 1); return m.reshape(sh);
  });
  return _catMasked(expanded, axis);
}

export function column_stack(tup: MaOperand[]): MaskedArray {
  const mas = tup.map((a) => { const m = _toMa(a); return m.ndim < 2 ? m.reshape([m.size, 1]) : m; });
  return _catMasked(mas, 1);
}

export function append(a: MaOperand, b: MaOperand, axis?: number): MaskedArray {
  if (axis === undefined) return _catMasked([_toMa(a).ravel(), _toMa(b).ravel()], 0);
  return _catMasked([a, b], axis);
}

export function atleast_1d(...arys: MaOperand[]): MaskedArray[] {
  return arys.map((a) => { const m = _toMa(a); return m.ndim === 0 ? m.reshape([1]) : m; });
}
export function atleast_2d(...arys: MaOperand[]): MaskedArray[] {
  return arys.map((a) => {
    const m = _toMa(a);
    if (m.ndim === 0) return m.reshape([1, 1]);
    if (m.ndim === 1) return m.reshape([1, m.size]);
    return m;
  });
}
export function atleast_3d(...arys: MaOperand[]): MaskedArray[] {
  return arys.map((a) => {
    const m = _toMa(a);
    if (m.ndim === 0) return m.reshape([1, 1, 1]);
    if (m.ndim === 1) return m.reshape([1, m.size, 1]);
    if (m.ndim === 2) return m.reshape([m.shape[0]!, m.shape[1]!, 1]);
    return m;
  });
}

export function hsplit(a: MaOperand, n: number): MaskedArray[] {
  const ma = _toMa(a);
  const axis = ma.ndim === 1 ? 0 : 1;
  const s = ma.shape[axis]!;
  const step = Math.floor(s / n);
  return Array.from({ length: n }, (_, i) => {
    const start = i * step; const end = i === n - 1 ? s : start + step;
    return ma.take(Array.from({ length: end - start }, (__, k) => k + start), axis);
  });
}

// ---------------------------------------------------------------------------
// Utility functions (D-209)
// ---------------------------------------------------------------------------

export function clip(a: MaOperand, a_min: number | null, a_max: number | null): MaskedArray {
  const ma = _toMa(a);
  return new MaskedArray(_clip(ma._data, a_min, a_max), ma._mask === false ? false : (ma._mask as NDArray).copy(), ma._fill_value);
}

export function where(condition: MaOperand, x?: MaOperand, y?: MaOperand): MaskedArray | NDArray {
  if (x === undefined || y === undefined) {
    const ma = _toMa(condition as MaOperand);
    return ma.filled(false).nonzero() as unknown as NDArray;
  }
  const cMa = _toMa(condition); const xMa = _toMa(x); const yMa = _toMa(y);
  const cond = cMa.filled(false);
  const n = cond.size;
  const outData: number[] = new Array<number>(n);
  const outMask: boolean[] = new Array<boolean>(n);
  const flatC = cond.ravel();
  const flatX = xMa._data.ravel(); const flatY = yMa._data.ravel();
  const flatXm = xMa._mask === false ? null : (xMa._mask as NDArray).ravel();
  const flatYm = yMa._mask === false ? null : (yMa._mask as NDArray).ravel();
  for (let i = 0; i < n; i++) {
    const c = flatC.item(i) as boolean;
    outData[i] = c ? (flatX.item(i % flatX.size) as number) : (flatY.item(i % flatY.size) as number);
    outMask[i] = c ? (flatXm ? (flatXm.item(i % (flatXm.size || 1)) as boolean) : false)
      : (flatYm ? (flatYm.item(i % (flatYm.size || 1)) as boolean) : false);
  }
  const sh = cond.shape;
  const resData = _array(outData, { dtype: "float64" }).reshape(sh) as NDArray;
  const resMask = _array(outMask, { dtype: "bool" }).reshape(sh) as NDArray;
  return new MaskedArray(resData, _allFalse(resMask) ? false : resMask);
}

export function nonzero(a: MaOperand): NDArray[] {
  return _toMa(a).filled(0).nonzero() as unknown as NDArray[];
}

export function compress(condition: NDArray | boolean[], a: MaOperand, axis?: number | null): MaskedArray {
  const ma = _toMa(a);
  const cond = condition instanceof NDArray ? condition : _array(condition as boolean[], { dtype: "bool" }) as NDArray;
  const idx: number[] = [];
  const flat = cond.ravel();
  for (let i = 0; i < flat.size; i++) if (flat.item(i) === true) idx.push(i);
  return ma.take(idx, axis ?? null);
}

export const round_ = (a: MaOperand, decimals = 0): MaskedArray => _unaryMa(a, (x) => x.round(decimals));
export const around = round_;
export const round = round_;

export const copy = (a: MaOperand): MaskedArray => _toMa(a).copy();

export function diag(a: MaOperand, k = 0): MaskedArray {
  const ma = _toMa(a);
  return new MaskedArray(_diag(ma._data, k), ma._mask === false ? false : _diag(ma._mask as NDArray, k), ma._fill_value);
}

export function diagflat(a: MaOperand, k = 0): MaskedArray {
  const ma = _toMa(a).ravel();
  return new MaskedArray(_diagflat(ma._data, k), ma._mask === false ? false : _diagflat(ma._mask as NDArray, k), ma._fill_value);
}

export function dot(a: MaOperand, b: MaOperand): MaskedArray {
  const ma = _toMa(a); const mb = _toMa(b);
  return new MaskedArray(_dot(ma.filled(0), mb.filled(0)), false);
}

export function inner(a: MaOperand, b: MaOperand): MaskedArray {
  const ma = _toMa(a); const mb = _toMa(b);
  return new MaskedArray(_inner(ma.filled(0), mb.filled(0)), false);
}
export const innerproduct = inner;

export function outer(a: MaOperand, b: MaOperand): MaskedArray {
  const ma = _toMa(a); const mb = _toMa(b);
  return new MaskedArray(_outer(ma.filled(0), mb.filled(0)), false);
}
export const outerproduct = outer;

export function ids(a: MaskedArray): [number, number] {
  return [a._data.nbytes, a._mask === false ? 0 : (a._mask as NDArray).nbytes];
}

export const isarray = (a: unknown): a is MaskedArray => a instanceof MaskedArray;
export const isMA = isarray;
export const isMaskedArray = isarray;

export function allequal(a: MaOperand, b: MaOperand, fill_value = true): boolean {
  const ma = _toMa(a); const mb = _toMa(b);
  const fa = ma.filled(fill_value ? 1 : 0); const fb = mb.filled(fill_value ? 1 : 0);
  const flatA = fa.ravel(); const flatB = fb.ravel();
  for (let i = 0; i < flatA.size; i++) if (flatA.item(i) !== flatB.item(i)) return false;
  return true;
}

export function allclose(a: MaOperand, b: MaOperand, masked_equal = true, rtol = 1e-5, atol = 1e-8): boolean {
  const ma = _toMa(a); const mb = _toMa(b);
  const fa = ma.filled(0); const fb = mb.filled(0);
  const flatA = fa.ravel(); const flatB = fb.ravel();
  const flatMa = ma._mask === false ? null : (ma._mask as NDArray).ravel();
  const flatMb = mb._mask === false ? null : (mb._mask as NDArray).ravel();
  for (let i = 0; i < flatA.size; i++) {
    const mA = flatMa ? (flatMa.item(i) as boolean) : false;
    const mB = flatMb ? (flatMb.item(i) as boolean) : false;
    if (mA && mB) { if (!masked_equal) return false; continue; }
    if (mA || mB) return false;
    if (Math.abs((flatA.item(i) as number) - (flatB.item(i) as number)) > atol + rtol * Math.abs(flatB.item(i) as number)) return false;
  }
  return true;
}

// creation
export const zeros = (shape: Shape | number, opts: { dtype?: DTypeLike } = {}): MaskedArray =>
  new MaskedArray(_zeros(typeof shape === "number" ? [shape] : [...shape], { dtype: opts.dtype }) as NDArray, false);
export const ones = (shape: Shape | number, opts: { dtype?: DTypeLike } = {}): MaskedArray =>
  new MaskedArray(_ones(typeof shape === "number" ? [shape] : [...shape], { dtype: opts.dtype }) as NDArray, false);
export const empty = (shape: Shape | number, opts: { dtype?: DTypeLike } = {}): MaskedArray =>
  new MaskedArray(_empty(typeof shape === "number" ? [shape] : [...shape], { dtype: opts.dtype }) as NDArray, false);
export const zeros_like = (a: MaskedArray | NDArray): MaskedArray => {
  const data = a instanceof MaskedArray ? a._data : a;
  return new MaskedArray(_zeros(data.shape, { dtype: data.dtype.name }) as NDArray, false);
};
export const ones_like = (a: MaskedArray | NDArray): MaskedArray => {
  const data = a instanceof MaskedArray ? a._data : a;
  return new MaskedArray(_ones(data.shape, { dtype: data.dtype.name }) as NDArray, false);
};
export const empty_like = (a: MaskedArray | NDArray): MaskedArray => {
  const data = a instanceof MaskedArray ? a._data : a;
  return new MaskedArray(_empty(data.shape, { dtype: data.dtype.name }) as NDArray, false);
};

export function identity(n: number, dtype: DTypeLike = "float64"): MaskedArray {
  const rows = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (__, j) => +(i === j)));
  return new MaskedArray(_array(rows, { dtype }) as NDArray, false);
}

export function arange(start: number, stop?: number, step = 1, opts: { dtype?: DTypeLike } = {}): MaskedArray {
  const s = stop === undefined ? 0 : start; const e = stop === undefined ? start : stop;
  const n = Math.max(0, Math.ceil((e - s) / step));
  const data = Array.from({ length: n }, (_, i) => s + i * step);
  return new MaskedArray(_array(data, { dtype: opts.dtype ?? "float64" }) as NDArray, false);
}

export function indices(dimensions: number[], dtype: DTypeLike = "int64"): MaskedArray {
  const ndims = dimensions.length;
  const n = dimensions.reduce((a, b) => a * b, 1);
  const totalShape = [ndims, ...dimensions];
  const out: number[][] = Array.from({ length: ndims }, (_, d) => {
    const arr: number[] = new Array<number>(n);
    const inner = dimensions.slice(d + 1).reduce((a, b) => a * b, 1);
    const outer_ = n / dimensions.slice(d).reduce((a, b) => a * b, 1);
    let idx = 0;
    for (let o = 0; o < outer_; o++) {
      for (let k = 0; k < (dimensions[d] ?? 0); k++) {
        for (let ii = 0; ii < inner; ii++) arr[idx++] = k;
      }
    }
    return arr;
  });
  const flatData = ([] as number[]).concat(...out);
  return new MaskedArray(_array(flatData, { dtype }).reshape(totalShape) as NDArray, false);
}

export const harden_mask = (a: MaskedArray): MaskedArray => { a.harden_mask(); return a; };
export const soften_mask = (a: MaskedArray): MaskedArray => { a.soften_mask(); return a; };

// notmasked / contiguous helpers
export function notmasked_edges(a: MaOperand, _axis?: number): [number, number] | null {
  const ma = _toMa(a);
  const flat = ma._mask === false ? null : (ma._mask as NDArray).ravel();
  const n = ma.size;
  let first = -1; let last = -1;
  for (let i = 0; i < n; i++) {
    if (!flat || !(flat.item(i) as boolean)) { if (first === -1) first = i; last = i; }
  }
  return first === -1 ? null : [first, last];
}

export function notmasked_contiguous(a: MaOperand, _axis?: number): [number, number][] {
  const ma = _toMa(a);
  const flat = ma._mask === false ? null : (ma._mask as NDArray).ravel();
  const n = ma.size;
  const runs: [number, number][] = [];
  let start = -1;
  for (let i = 0; i < n; i++) {
    const isMasked = flat ? (flat.item(i) as boolean) : false;
    if (!isMasked) { if (start === -1) start = i; }
    else { if (start !== -1) { runs.push([start, i - 1]); start = -1; } }
  }
  if (start !== -1) runs.push([start, n - 1]);
  return runs;
}

export const flatnotmasked_edges = notmasked_edges;
export const flatnotmasked_contiguous = notmasked_contiguous;

export function clump_masked(a: MaOperand): [number, number][] {
  const ma = _toMa(a);
  if (ma._mask === false) return [];
  const flat = (ma._mask as NDArray).ravel();
  const n = ma.size;
  const runs: [number, number][] = [];
  let start = -1;
  for (let i = 0; i < n; i++) {
    if (flat.item(i) as boolean) { if (start === -1) start = i; }
    else { if (start !== -1) { runs.push([start, i - 1]); start = -1; } }
  }
  if (start !== -1) runs.push([start, n - 1]);
  return runs;
}

export const clump_unmasked = notmasked_contiguous;

export function compress_nd(x: MaOperand, axis?: number | number[]): MaskedArray {
  const ma = _toMa(x);
  const axes = axis === undefined ? Array.from({ length: ma.ndim }, (_, i) => i)
    : typeof axis === "number" ? [axis] : axis;
  let result = ma;
  for (const ax of axes) {
    const keep: number[] = [];
    for (let i = 0; i < (result.shape[ax] ?? 0); i++) {
      const sl = result.take([i], ax);
      if (!is_masked(sl)) keep.push(i);
    }
    if (keep.length > 0) result = result.take(keep, ax);
  }
  return result;
}

export const compress_rows = (x: MaOperand): MaskedArray => compress_nd(x, 0);
export const compress_cols = (x: MaOperand): MaskedArray => compress_nd(x, 1);
export const compress_rowcols = (x: MaOperand): MaskedArray => compress_nd(x, [0, 1]);

export function mask_rows(a: MaOperand): MaskedArray {
  const ma = _toMa(a);
  if (ma._mask === false) return ma;
  const newMask = (ma._mask as NDArray).copy();
  const rows = ma.shape[0] ?? 0; const cols = ma.shape[1] ?? 0;
  for (let i = 0; i < rows; i++) {
    let any = false;
    for (let j = 0; j < cols; j++) if (newMask.item(i, j) as boolean) { any = true; break; }
    if (any) for (let j = 0; j < cols; j++) newMask.set([[i, i + 1], [j, j + 1]], true);
  }
  return new MaskedArray(ma._data, newMask, ma._fill_value);
}

export function mask_cols(a: MaOperand): MaskedArray {
  const ma = _toMa(a);
  if (ma._mask === false) return ma;
  const newMask = (ma._mask as NDArray).copy();
  const rows = ma.shape[0] ?? 0; const cols = ma.shape[1] ?? 0;
  for (let j = 0; j < cols; j++) {
    let any = false;
    for (let i = 0; i < rows; i++) if (newMask.item(i, j) as boolean) { any = true; break; }
    if (any) for (let i = 0; i < rows; i++) newMask.set([[i, i + 1], [j, j + 1]], true);
  }
  return new MaskedArray(ma._data, newMask, ma._fill_value);
}

export const mask_rowcols = (a: MaOperand): MaskedArray => mask_cols(mask_rows(a));
export const flatten_structured_array = (a: MaOperand): MaskedArray => _toMa(a).flatten();

export function ediff1d(ary: MaOperand, to_end?: number | number[] | null, to_begin?: number | number[] | null): MaskedArray {
  const ma = _toMa(ary).ravel();
  const flatD = ma._data.ravel(); const flatM = ma._mask === false ? null : (ma._mask as NDArray).ravel();
  const n = ma.size;
  const diff: number[] = []; const diffMask: boolean[] = [];
  for (let i = 1; i < n; i++) {
    diff.push((flatD.item(i) as number) - (flatD.item(i - 1) as number));
    diffMask.push((flatM ? (flatM.item(i) as boolean) : false) || (flatM ? (flatM.item(i - 1) as boolean) : false));
  }
  if (to_begin != null) { const arr = Array.isArray(to_begin) ? to_begin : [to_begin]; diff.unshift(...arr); diffMask.unshift(...arr.map(() => false)); }
  if (to_end != null) { const arr = Array.isArray(to_end) ? to_end : [to_end]; diff.push(...arr); diffMask.push(...arr.map(() => false)); }
  const data = _array(diff.length ? diff : [], { dtype: "float64" }) as NDArray;
  const maskArr = _array(diffMask, { dtype: "bool" }) as NDArray;
  return new MaskedArray(data, _allFalse(maskArr) ? false : maskArr);
}

export function diff(a: MaOperand, n = 1, axis = -1): MaskedArray {
  let ma = _toMa(a);
  for (let i = 0; i < n; i++) {
    const ax = axis < 0 ? ma.ndim + axis : axis;
    const s = ma.shape[ax]!;
    const a1 = ma.take(Array.from({ length: s - 1 }, (_, k) => k + 1), ax);
    const a0 = ma.take(Array.from({ length: s - 1 }, (_, k) => k), ax);
    ma = subtract(a1, a0);
  }
  return ma;
}

export function ndenumerate(a: MaOperand): Array<[number[], number | boolean | typeof masked]> {
  const ma = _toMa(a);
  const flat = ma._data.ravel();
  const flatM = ma._mask === false ? null : (ma._mask as NDArray).ravel();
  const n = flat.size;
  return Array.from({ length: n }, (_, i) => {
    const coords = _flatToCoords(i, ma.shape);
    const isMasked = flatM ? (flatM.item(i) as boolean) : false;
    return [coords, isMasked ? masked : (flat.item(i) as number | boolean)] as [number[], number | boolean | typeof masked];
  });
}

// set operations
function _c(a: MaOperand): NDArray { return _toMa(a).compressed(); }
export const intersect1d = (ar1: MaOperand, ar2: MaOperand): NDArray => _intersect1d(_c(ar1), _c(ar2));
export const setdiff1d = (ar1: MaOperand, ar2: MaOperand): NDArray => _setdiff1d(_c(ar1), _c(ar2));
export const setxor1d = (ar1: MaOperand, ar2: MaOperand): NDArray => _setxor1d(_c(ar1), _c(ar2));
export const union1d = (ar1: MaOperand, ar2: MaOperand): NDArray => _union1d(_c(ar1), _c(ar2));
export const unique = (a: MaOperand): NDArray => _unique(_c(a));
export const in1d = (ar1: MaOperand, ar2: MaOperand): MaskedArray =>
  new MaskedArray(_isin(_c(ar1), _c(ar2)), false);
export const isin = in1d;

export function putmask(a: MaskedArray, mask: NDArray | boolean[], values: MaOperand): void {
  const cond = mask instanceof NDArray ? mask : _array(mask as boolean[], { dtype: "bool" }) as NDArray;
  const vals = _toMa(values);
  const n = a.size;
  const flatC = cond.ravel(); const flatV = vals._data.ravel();
  const flatVm = vals._mask === false ? null : (vals._mask as NDArray).ravel();
  const currentMask = a._mask === false
    ? (_zeros(a.shape, { dtype: "bool" }) as NDArray) : (a._mask as NDArray).copy();
  const flatCurrMask = currentMask.ravel();
  for (let i = 0; i < n; i++) {
    if (flatC.item(i) === true) {
      a._data.flat.set(i, flatV.item(i % flatV.size) as number);
      const mv = flatVm ? (flatVm.item(i % (flatVm.size || 1)) as boolean) : false;
      flatCurrMask.set(i, mv);
    }
  }
  a._mask = _allFalse(currentMask) ? false : currentMask;
}

export function choose(a: MaOperand, choices: MaOperand[]): MaskedArray {
  const ma = _toMa(a); const cs = choices.map(_toMa);
  const flat = ma._data.ravel(); const n = flat.size;
  const out: number[] = []; const outMask: boolean[] = [];
  for (let i = 0; i < n; i++) {
    const idx = Math.floor(flat.item(i) as number);
    const choice = cs[idx]!; const cFlat = choice._data.ravel();
    const cmFlat = choice._mask === false ? null : (choice._mask as NDArray).ravel();
    out.push(cFlat.item(i % cFlat.size) as number);
    outMask.push(cmFlat ? (cmFlat.item(i % (cmFlat.size || 1)) as boolean) : false);
  }
  const sh = ma.shape;
  const resData = _array(out, { dtype: "float64" }).reshape(sh) as NDArray;
  const resMask = _array(outMask, { dtype: "bool" }).reshape(sh) as NDArray;
  return new MaskedArray(resData, _allFalse(resMask) ? false : resMask);
}

// Simple implementations of corrcoef/cov/convolve/correlate using filled data
export function corrcoef(x: MaOperand, y?: MaOperand): MaskedArray {
  const mx = _toMa(x); const fx = mx.filled(0);
  if (y) {
    const fy = _toMa(y).filled(0);
    const n = fx.size;
    const meanX = fx.mean().item() as number; const meanY = fy.mean().item() as number;
    const flatX = fx.ravel(); const flatY = fy.ravel();
    let sxx = 0; let syy = 0; let sxy = 0;
    for (let i = 0; i < n; i++) {
      const dx = (flatX.item(i) as number) - meanX; const dy = (flatY.item(i) as number) - meanY;
      sxx += dx * dx; syy += dy * dy; sxy += dx * dy;
    }
    const r = sxy / Math.sqrt(sxx * syy);
    return new MaskedArray(_array([[1, r], [r, 1]], { dtype: "float64" }) as NDArray, false);
  }
  const n = fx.size; const flat = fx.ravel();
  const m = flat.mean().item() as number;
  let s = 0;
  for (let i = 0; i < n; i++) { const d = (flat.item(i) as number) - m; s += d * d; }
  return new MaskedArray(_array([[1]], { dtype: "float64" }) as NDArray, false);
}

export function correlate(a: MaOperand, v: MaOperand, mode?: string): MaskedArray {
  const fa = _toMa(a).filled(0); const fv = _toMa(v).filled(0);
  const an = fa.size; const vn = fv.size;
  const flatA = fa.ravel(); const flatV = fv.ravel();
  let out: number[];
  if (!mode || mode === "valid") {
    const rn = an - vn + 1;
    out = Array.from({ length: Math.max(0, rn) }, (_, i) => {
      let s = 0; for (let j = 0; j < vn; j++) s += (flatA.item(i + j) as number) * (flatV.item(j) as number); return s;
    });
  } else {
    const rn = an + vn - 1;
    out = Array.from({ length: rn }, (_, i) => {
      let s = 0;
      for (let j = 0; j < vn; j++) { const k = i - j; if (k >= 0 && k < an) s += (flatA.item(k) as number) * (flatV.item(j) as number); }
      return s;
    });
  }
  return new MaskedArray(_array(out.length ? out : [], { dtype: "float64" }) as NDArray, false);
}

export function convolve(a: MaOperand, v: MaOperand, mode?: string): MaskedArray {
  const fa = _toMa(a).filled(0); const fv = _toMa(v).filled(0);
  const flatA = fa.ravel(); const flatV = fv.ravel();
  const an = flatA.size; const vn = flatV.size;
  const rn = an + vn - 1;
  const full = Array.from({ length: rn }, (_, i) => {
    let s = 0;
    for (let j = 0; j < vn; j++) { const k = i - j; if (k >= 0 && k < an) s += (flatA.item(k) as number) * (flatV.item(vn - 1 - j) as number); }
    return s;
  });
  let out: number[];
  if (mode === "same") out = full.slice(Math.floor((vn - 1) / 2), Math.floor((vn - 1) / 2) + an);
  else if (mode === "valid") out = full.slice(vn - 1, an);
  else out = full;
  return new MaskedArray(_array(out.length ? out : [], { dtype: "float64" }) as NDArray, false);
}

export function polyfit(x: MaOperand, y: MaOperand, deg: number): NDArray {
  return _polyfit(_toMa(x).filled(0), _toMa(y).filled(0), deg) as NDArray;
}

export function cov(m: MaOperand, y?: MaOperand): MaskedArray {
  const fm = _toMa(m).filled(0);
  if (y) {
    const fy = _toMa(y).filled(0);
    const n = fm.size;
    const flatM = fm.ravel(); const flatY = fy.ravel();
    const mMean = flatM.mean().item() as number; const yMean = flatY.mean().item() as number;
    let smm = 0; let syy = 0; let smy = 0;
    for (let i = 0; i < n; i++) {
      const dm = (flatM.item(i) as number) - mMean; const dy = (flatY.item(i) as number) - yMean;
      smm += dm * dm; syy += dy * dy; smy += dm * dy;
    }
    const f = 1 / (n - 1);
    return new MaskedArray(_array([[smm * f, smy * f], [smy * f, syy * f]], { dtype: "float64" }) as NDArray, false);
  }
  const flat = fm.ravel(); const n = flat.size;
  const m_ = flat.mean().item() as number;
  let s = 0;
  for (let i = 0; i < n; i++) { const d = (flat.item(i) as number) - m_; s += d * d; }
  return new MaskedArray(_array([[s / (n - 1)]], { dtype: "float64" }) as NDArray, false);
}

export function vander(x: MaOperand, N?: number, increasing?: boolean): MaskedArray {
  return new MaskedArray(_vander(_toMa(x).filled(0), N, { increasing }), false);
}

export const mr_ = (...arrays: MaOperand[]): MaskedArray =>
  concatenate(arrays.map((a) => _toMa(a).ravel()), 0);

/** mvoid — a masked void (masked scalar wrapper) */
export class mvoid {
  readonly _data: NDArray;
  readonly _mask: NDArray | false;
  constructor(data: NDArray, mask: NDArray | false) { this._data = data; this._mask = mask; }
}

export const bool_ = toDType("bool");

// ---------------------------------------------------------------------------
// np.ma namespace object
// ---------------------------------------------------------------------------

export const ma = {
  // types/classes
  MaskedArray,
  MaskError,
  MAError,
  MaskType,
  MaskedIterator,
  mvoid,
  bool_,

  // constants
  masked,
  masked_singleton,
  masked_print_option,
  nomask,

  // constructors
  masked_array,
  array,
  asarray,
  asanyarray,
  masked_where,
  masked_equal,
  masked_not_equal,
  masked_greater,
  masked_greater_equal,
  masked_less,
  masked_less_equal,
  masked_inside,
  masked_outside,
  masked_invalid,
  masked_values,
  masked_object,
  masked_all,
  masked_all_like,
  fix_invalid,

  // mask utilities
  make_mask,
  make_mask_none,
  make_mask_descr,
  getmask,
  getmaskarray,
  getdata,
  filled,
  is_masked,
  is_mask,
  mask_or,
  flatten_mask,
  shrink_mask,

  // fill value
  default_fill_value,
  maximum_fill_value,
  minimum_fill_value,
  set_fill_value,
  common_fill_value,

  // arithmetic / ufuncs
  add,
  subtract,
  multiply,
  divide,
  true_divide,
  floor_divide,
  power,
  mod,
  remainder,
  fmod,
  negative,
  absolute,
  abs,
  fabs,
  equal,
  not_equal,
  less,
  less_equal,
  greater,
  greater_equal,
  sqrt,
  exp,
  log,
  log2,
  log10,
  sin,
  cos,
  tan,
  arcsin,
  arccos,
  arctan,
  arctan2,
  sinh,
  cosh,
  tanh,
  arcsinh,
  arccosh,
  arctanh,
  hypot,
  ceil,
  floor,
  conjugate,
  angle,
  bitwise_and,
  bitwise_or,
  bitwise_xor,
  left_shift,
  right_shift,
  logical_and,
  logical_or,
  logical_xor,
  logical_not,
  maximum,
  minimum,

  // reductions
  sum,
  prod,
  product,
  sometrue,
  alltrue,
  min,
  amin,
  max,
  amax,
  mean,
  var: var_,
  std,
  all,
  any,
  argmin,
  argmax,
  cumsum,
  cumprod,
  count,
  count_masked,
  ptp,
  anom,
  anomalies,
  average,
  median,

  // shape/manip
  reshape,
  ravel,
  squeeze,
  transpose,
  expand_dims,
  repeat,
  sort,
  argsort,
  take,
  put,
  swapaxes,
  shape,
  size,
  ndim,
  resize,
  diagonal,
  trace,

  // concat/stack
  concatenate,
  vstack,
  row_stack,
  hstack,
  dstack,
  stack,
  column_stack,
  append,
  atleast_1d,
  atleast_2d,
  atleast_3d,
  hsplit,

  // utilities
  clip,
  where,
  nonzero,
  compress,
  round_,
  around,
  round,
  copy,
  diag,
  diagflat,
  dot,
  inner,
  innerproduct,
  outer,
  outerproduct,
  ids,
  isarray,
  isMA,
  isMaskedArray,
  allequal,
  allclose,
  harden_mask,
  soften_mask,
  notmasked_edges,
  notmasked_contiguous,
  flatnotmasked_edges,
  flatnotmasked_contiguous,
  clump_masked,
  clump_unmasked,
  compress_nd,
  compress_rows,
  compress_cols,
  compress_rowcols,
  mask_rows,
  mask_cols,
  mask_rowcols,
  flatten_structured_array,

  // creation
  zeros,
  ones,
  empty,
  zeros_like,
  ones_like,
  empty_like,
  identity,
  arange,
  indices,
  mr_,

  // misc
  ediff1d,
  diff,
  ndenumerate,
  intersect1d,
  setdiff1d,
  setxor1d,
  union1d,
  unique,
  in1d,
  isin,
  putmask,
  choose,
  corrcoef,
  correlate,
  convolve,
  polyfit,
  cov,
  vander,
} as const;
