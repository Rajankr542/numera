// P16-A public functions (D-191): TS-only quick wins.
// - Math/bool/sentinel constants (pi, e, inf, nan, euler_gamma, True_, False_,
//   PINF, NINF, PZERO, NZERO).
// - Dtype alias constants (double, single, half, int_, intc, intp, byte,
//   short, ubyte, ushort, uint, uintc, uintp, cdouble, csingle, longdouble,
//   clongdouble, long, ulong).
// - np.vectorize(fn, {otypes?, signature?}): element-wise JS callback wrapper.
// - np.shares_memory(a, b, {maxWork?}): delegates to mayShareMemory.
// - NDArray.prototype.cumsum / cumprod via declaration merging.

import {
  bool as dtBool,
  int8,
  uint8,
  int16,
  uint16,
  int32,
  uint32,
  int64,
  uint64,
  float16,
  float32,
  float64,
  complex64,
  complex128,
  dtype as toDType,
  type DTypeLike,
} from "./dtype.js";
import { ValueError } from "./errors.js";
import { array, asarray } from "./creation.js";
import { NDArray, type NestedArray } from "./ndarray.js";
import { add, multiply, type UfuncAccumulateOptions } from "./ufunc.js";
import { mayShareMemory } from "./creation.js";

// ---- Math / bool / sentinel constants (D-191) ----

/** NumPy pi: mathematical constant π. */
export const pi: number = Math.PI;
/** NumPy e: base of natural logarithm. */
export const e: number = Math.E;
/** NumPy inf / Inf / Infinity: positive infinity. */
export const inf: number = Infinity;
/** NumPy nan / NaN: not-a-number. */
export const nan: number = NaN;
/** NumPy euler_gamma: Euler–Mascheroni constant. */
export const euler_gamma: number = 0.5772156649015328606;
/** NumPy True_: Python True (JS true). */
export const True_: true = true;
/** NumPy False_: Python False (JS false). */
export const False_: false = false;
/** NumPy PINF: positive infinity (alias of inf). */
export const PINF: number = Infinity;
/** NumPy NINF: negative infinity. */
export const NINF: number = -Infinity;
/** NumPy PZERO: positive zero. */
export const PZERO: number = 0;
/** NumPy NZERO: negative zero. */
export const NZERO: number = -0;

// ---- Dtype alias constants (D-191) ----
// NumPy spelling → canonical DType singleton.

/** np.double: alias for float64. */
export const double = float64;
/** np.single: alias for float32. */
export const single = float32;
/** np.half: alias for float16. */
export const half = float16;
/** np.int_: default Python int (int64 on 64-bit). */
export const int_ = int64;
/** np.intc: C int (int32). */
export const intc = int32;
/** np.intp: pointer-sized int (int64 on 64-bit). */
export const intp = int64;
/** np.long: C long (int64 on 64-bit). */
export const long = int64;
/** np.byte: C signed char (int8). */
export const byte = int8;
/** np.short: C short (int16). */
export const short = int16;
/** np.ubyte: C unsigned char (uint8). */
export const ubyte = uint8;
/** np.ushort: C unsigned short (uint16). */
export const ushort = uint16;
/** np.uint: unsigned platform int (uint64 on 64-bit). */
export const uint = uint64;
/** np.uintc: C unsigned int (uint32). */
export const uintc = uint32;
/** np.uintp: pointer-sized unsigned int (uint64). */
export const uintp = uint64;
/** np.ulong: C unsigned long (uint64 on 64-bit). */
export const ulong = uint64;
/** np.cdouble: alias for complex128. */
export const cdouble = complex128;
/** np.csingle: alias for complex64. */
export const csingle = complex64;
/** np.longdouble: alias for float64 (no 80-bit float in JS). */
export const longdouble = float64;
/** np.clongdouble: alias for complex128 (no 80-bit complex in JS). */
export const clongdouble = complex128;

// ---- np.vectorize (D-191) ----

/** Options for np.vectorize. */
export interface VectorizeOptions {
  /**
   * Output dtype(s). If given, the first element is the result dtype.
   * Only a single output dtype is supported (consistent with JS callbacks).
   */
  otypes?: DTypeLike[];
  /**
   * NumPy generalised ufunc signature string.
   * Currently only `null` / `undefined` (scalar-in scalar-out) is supported;
   * any non-null value raises ValueError.
   */
  signature?: string | null;
}

/** The function returned by np.vectorize. */
export interface VectorizedFn {
  /** Apply fn element-wise over the inputs (broadcast together). */
  (...args: (NDArray | NestedArray)[]): NDArray;
  /** The original wrapped function. */
  readonly pyfunc: (...args: unknown[]) => unknown;
}

/**
 * NumPy vectorize: wraps `fn` so that it is applied element-wise over its
 * inputs, which are broadcast to a common shape. The output dtype is taken
 * from `otypes[0]` when given, or inferred from the first call.
 *
 * Only scalar-in / scalar-out signatures are supported (`signature` must be
 * `null` or omitted). A non-null `signature` raises `ValueError`.
 */
export function vectorize(fn: (...args: unknown[]) => unknown, opts: VectorizeOptions = {}): VectorizedFn {
  if (opts.signature != null) {
    throw new ValueError("vectorize: signature parameter is not supported (only scalar-in scalar-out)");
  }

  const outDtype = opts.otypes != null && opts.otypes.length > 0 ? toDType(opts.otypes[0]!) : null;

  const vfn = function (...inputs: (NDArray | NestedArray)[]): NDArray {
    if (inputs.length === 0) {
      throw new ValueError("vectorize: at least one input is required");
    }
    // Convert all inputs to NDArrays.
    const arrs = inputs.map((x) => (x instanceof NDArray ? x : array(x as NestedArray)));

    // Compute the broadcast shape.
    const ndim = Math.max(...arrs.map((a) => a.ndim));
    const shape: number[] = new Array<number>(ndim).fill(1);
    for (const a of arrs) {
      for (let i = 1; i <= a.ndim; i++) {
        const d = a.shape[a.ndim - i]!;
        const o = shape[ndim - i]!;
        if (d !== o && d !== 1 && o !== 1) {
          throw new ValueError(
            `vectorize: inputs could not be broadcast together with shapes ${arrs.map((a) => `(${a.shape.join(",")})`).join(" ")}`,
          );
        }
        if (o === 1) shape[ndim - i] = d;
      }
    }

    // Broadcast each input to the common shape.
    const size = shape.reduce((acc, d) => acc * d, 1);

    // Flatten each broadcast array to 1-D for element-wise iteration.
    const flat = arrs.map((a) => {
      // Create broadcast strides (0 for size-1 dimensions).
      const strides: number[] = new Array<number>(ndim).fill(0);
      for (let i = 1; i <= a.ndim; i++) {
        const d = a.shape[a.ndim - i]!;
        strides[ndim - i] = d === 1 ? 0 : a.strides[a.ndim - i]!;
      }
      // Walk in C order and collect items.
      const items: unknown[] = new Array<unknown>(size);
      for (let flat_idx = 0; flat_idx < size; flat_idx++) {
        // Compute multi-index for flat_idx in C order.
        const multiIdx: number[] = new Array<number>(ndim);
        let rem = flat_idx;
        for (let dim = ndim - 1; dim >= 0; dim--) {
          multiIdx[dim] = rem % shape[dim]!;
          rem = Math.floor(rem / shape[dim]!);
        }
        // Compute flat offset in the (possibly non-contiguous) array.
        let byteOff = a._native.offset();
        for (let dim = 0; dim < ndim; dim++) {
          byteOff += multiIdx[dim]! * strides[dim]!;
        }
        // We need scalar elements. Use item() on a slice — simplest is to use
        // the flat index in the broadcast result.
        // Actually, compute the per-dim index into the original array.
        const origIdx: number[] = new Array<number>(a.ndim);
        for (let i = 1; i <= a.ndim; i++) {
          const d = a.shape[a.ndim - i]!;
          origIdx[a.ndim - i] = d === 1 ? 0 : multiIdx[ndim - i]!;
        }
        items[flat_idx] = a.item(...origIdx);
      }
      return items;
    });

    // Call fn for each position.
    const results: unknown[] = new Array<unknown>(size);
    for (let i = 0; i < size; i++) {
      const args = flat.map((f) => f[i]);
      results[i] = fn(...args);
    }

    // Infer output dtype if not given.
    let resultDtype = outDtype;
    if (resultDtype === null && size > 0) {
      const first = results[0];
      if (typeof first === "boolean" || first instanceof dtBool.constructor) {
        resultDtype = dtBool;
      } else if (typeof first === "bigint") {
        resultDtype = int64;
      } else if (typeof first === "number") {
        resultDtype = float64;
      } else if (
        first !== null &&
        typeof first === "object" &&
        "re" in (first as object) &&
        "im" in (first as object)
      ) {
        resultDtype = complex128;
      } else {
        resultDtype = float64;
      }
    } else if (resultDtype === null) {
      resultDtype = float64;
    }

    // Build the result array.
    const nested = reshapeNested(results, shape);
    return array(nested as NestedArray, { dtype: resultDtype });
  } as VectorizedFn;

  // Attach pyfunc.
  (vfn as { pyfunc: unknown }).pyfunc = fn;
  return vfn;
}

/** Recursively reshape a flat results array into nested arrays matching `shape`. */
function reshapeNested(flat: unknown[], shape: number[]): unknown {
  if (shape.length === 0) return flat[0];
  if (shape.length === 1) return flat.slice(0, shape[0]);
  const inner = shape.slice(1);
  const innerSize = inner.reduce((a, b) => a * b, 1);
  const out: unknown[] = [];
  for (let i = 0; i < shape[0]!; i++) {
    out.push(reshapeNested(flat.slice(i * innerSize, (i + 1) * innerSize), inner));
  }
  return out;
}

// ---- np.shares_memory (D-191) ----

/** Options for shares_memory (maxWork is accepted but not used). */
export interface SharesMemoryOptions {
  /** Maximum number of work elements (NumPy 2: this is deprecated/ignored). */
  maxWork?: number;
}

/**
 * NumPy shares_memory: returns true if arrays `a` and `b` share any memory.
 * Delegates to the native buffer identity + byte-range overlap check.
 * `maxWork` is accepted for API compatibility but ignored.
 */
export function shares_memory(a: NDArray, b: NDArray, opts: SharesMemoryOptions = {}): boolean {
  void opts; // maxWork is not used
  return mayShareMemory(a, b);
}

// ---- NDArray.cumsum / cumprod (D-191) ----
// Declaration merging so these methods appear on NDArray.

/** Options for cumsum/cumprod. */
export interface CumOptions {
  /** Axis to accumulate along; null/omitted flattens first. */
  axis?: number | null;
  /** Result dtype. */
  dtype?: DTypeLike | null;
}

declare module "./ndarray.js" {
  interface NDArray {
    /** NumPy `a.cumsum(axis?, {dtype?})`: cumulative sum. */
    cumsum(axis?: number | null | CumOptions, opts?: CumOptions): NDArray;
    /** NumPy `a.cumprod(axis?, {dtype?})`: cumulative product. */
    cumprod(axis?: number | null | CumOptions, opts?: CumOptions): NDArray;
  }
}

function parseCumArgs(
  axis?: number | null | CumOptions,
  opts?: CumOptions,
): { axis?: number; dtype?: DTypeLike | null } {
  if (axis !== null && typeof axis === "object") {
    // called as cumsum({ axis: 0, dtype: ... })
    return { axis: axis.axis ?? undefined, dtype: axis.dtype };
  }
  return { axis: axis as number | undefined, dtype: opts?.dtype };
}

NDArray.prototype.cumsum = function (
  this: NDArray,
  axis?: number | null | CumOptions,
  opts?: CumOptions,
): NDArray {
  const { axis: ax, dtype: dt } = parseCumArgs(axis, opts);
  const src: NDArray = ax === undefined || ax === null ? this.ravel() : this;
  const accOpts: UfuncAccumulateOptions = {};
  if (ax !== undefined && ax !== null) accOpts.axis = ax;
  if (dt != null) accOpts.dtype = dt;
  return add.accumulate(src, accOpts);
};

NDArray.prototype.cumprod = function (
  this: NDArray,
  axis?: number | null | CumOptions,
  opts?: CumOptions,
): NDArray {
  const { axis: ax, dtype: dt } = parseCumArgs(axis, opts);
  const src: NDArray = ax === undefined || ax === null ? this.ravel() : this;
  const accOpts: UfuncAccumulateOptions = {};
  if (ax !== undefined && ax !== null) accOpts.axis = ax;
  if (dt != null) accOpts.dtype = dt;
  return multiply.accumulate(src, accOpts);
};

// ---- Top-level cumsum / cumprod (D-191) ----

/**
 * NumPy cumsum: cumulative sum along `axis` (or flattened if omitted/null).
 */
export function cumsum(a: NDArray | NestedArray, axis?: number | null, opts: { dtype?: DTypeLike | null } = {}): NDArray {
  const arr = a instanceof NDArray ? a : array(a as NestedArray);
  const src = axis === undefined || axis === null ? arr.ravel() : arr;
  const accOpts: UfuncAccumulateOptions = {};
  if (axis !== undefined && axis !== null) accOpts.axis = axis;
  if (opts.dtype != null) accOpts.dtype = opts.dtype;
  return add.accumulate(src, accOpts);
}

/**
 * NumPy cumprod: cumulative product along `axis` (or flattened if omitted/null).
 */
export function cumprod(a: NDArray | NestedArray, axis?: number | null, opts: { dtype?: DTypeLike | null } = {}): NDArray {
  const arr = a instanceof NDArray ? a : array(a as NestedArray);
  const src = axis === undefined || axis === null ? arr.ravel() : arr;
  const accOpts: UfuncAccumulateOptions = {};
  if (axis !== undefined && axis !== null) accOpts.axis = axis;
  if (opts.dtype != null) accOpts.dtype = opts.dtype;
  return multiply.accumulate(src, accOpts);
}

// ---- p16a spread object (D-191) ----

export const p16a = {
  // Math / sentinel constants
  pi,
  e,
  inf,
  nan,
  euler_gamma,
  True_,
  False_,
  PINF,
  NINF,
  PZERO,
  NZERO,
  // Dtype aliases
  double,
  single,
  half,
  int_,
  intc,
  intp,
  long,
  byte,
  short,
  ubyte,
  ushort,
  uint,
  uintc,
  uintp,
  ulong,
  cdouble,
  csingle,
  longdouble,
  clongdouble,
  // Functions
  vectorize,
  shares_memory,
  cumsum,
  cumprod,
} as const;
