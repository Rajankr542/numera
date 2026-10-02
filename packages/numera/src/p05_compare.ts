// P5 comparison / logical ufunc objects (D-080).
import { DTypeError } from "./errors.js";
import { array } from "./creation.js";
import { dtype as toDType } from "./dtype.js";
import { NDArray } from "./ndarray.js";
import {
  binaryUfunc,
  unaryUfunc,
  type ArrayLike,
  type BinaryUfunc,
  type Operand,
  type UfuncOptions,
  type UfuncReduceOptions,
  type UnaryUfunc,
} from "./ufunc.js";

const INT_RANGE: Record<string, readonly [bigint, bigint]> = {
  int8: [-128n, 127n],
  uint8: [0n, 255n],
  int16: [-32768n, 32767n],
  uint16: [0n, 65535n],
  int32: [-2147483648n, 2147483647n],
  uint32: [0n, 4294967295n],
  int64: [-(2n ** 63n), 2n ** 63n - 1n],
  uint64: [0n, 2n ** 64n - 1n],
};

// A JS integer outside the other operand's integer range is not weak for
// comparisons (NumPy 2 compares the exact value, D-080).
function exactScalar(v: Operand, other: Operand): Operand {
  if (!(other instanceof NDArray)) return v;
  const range = INT_RANGE[other.dtype.name];
  if (!range) return v;
  let big: bigint;
  if (typeof v === "bigint") big = v;
  else if (typeof v === "number" && Number.isInteger(v)) big = BigInt(v);
  else return v;
  if (big >= range[0] && big <= range[1]) return v;
  const [i64, u64] = [INT_RANGE.int64!, INT_RANGE.uint64!];
  if (big >= i64[0] && big <= i64[1]) return array(big, { dtype: "int64" });
  if (big >= u64[0] && big <= u64[1]) return array(big, { dtype: "uint64" });
  return array(Number(big), { dtype: "float64" });
}

// dtype= on a bool-output ufunc is NumPy's output signature: only bool.
function boolOpts(name: string, opts: UfuncOptions | undefined): UfuncOptions | undefined {
  if (!opts || opts.dtype === undefined || opts.dtype === null) return opts;
  if (toDType(opts.dtype).name !== "bool") {
    throw new DTypeError(`No loop matching the specified signature and casting was found for ufunc ${name}`);
  }
  const { dtype: _ignored, ...rest } = opts;
  return rest;
}

/** A binary ufunc whose output is bool (comparisons, logical ops; D-080). */
export function boolBinaryUfunc(op: string): BinaryUfunc {
  const base = binaryUfunc(op);
  const f = (a: Operand, b: Operand, opts?: UfuncOptions): NDArray =>
    base(exactScalar(a, b), exactScalar(b, a), boolOpts(op, opts));
  return Object.assign(f, {
    reduce: base.reduce,
    accumulate: base.accumulate,
    reduceat: base.reduceat,
    outer: (a: Operand, b: Operand, opts?: UfuncOptions): NDArray =>
      base.outer(exactScalar(a, b), exactScalar(b, a), boolOpts(op, opts)),
    at: base.at,
  });
}

/** NumPy equal: element-wise `a == b` (bool). */
export const equal: BinaryUfunc = boolBinaryUfunc("equal");
/** NumPy not_equal: element-wise `a != b` (bool). */
export const notEqual: BinaryUfunc = boolBinaryUfunc("notEqual");
/** NumPy less: element-wise `a < b` (bool; complex compares lexicographically). */
export const less: BinaryUfunc = boolBinaryUfunc("less");
/** NumPy less_equal: element-wise `a <= b` (bool). */
export const lessEqual: BinaryUfunc = boolBinaryUfunc("lessEqual");
/** NumPy greater: element-wise `a > b` (bool). */
export const greater: BinaryUfunc = boolBinaryUfunc("greater");
/** NumPy greater_equal: element-wise `a >= b` (bool). */
export const greaterEqual: BinaryUfunc = boolBinaryUfunc("greaterEqual");

/** NumPy logical_and: element-wise truth AND (bool; identity True). */
export const logicalAnd: BinaryUfunc = boolBinaryUfunc("logicalAnd");
/** NumPy logical_or: element-wise truth OR (bool; identity False). */
export const logicalOr: BinaryUfunc = boolBinaryUfunc("logicalOr");
/** NumPy logical_xor: element-wise truth XOR (bool; identity False). */
export const logicalXor: BinaryUfunc = boolBinaryUfunc("logicalXor");
/** NumPy logical_not: element-wise truth NOT (bool). */
export const logicalNot: UnaryUfunc = unaryUfunc("logicalNot");

/** Options for `np.all` / `np.any` (NumPy keywords, D-080). */
export interface AllAnyOptions {
  /** Axis or axes to reduce; `null`/omitted reduces all axes. */
  axis?: number | readonly number[] | null;
  /** Write the result into this array (cast to its dtype) and return it. */
  out?: NDArray | null;
  /** Keep reduced axes with length 1. */
  keepdims?: boolean;
  /** Bool mask: only true elements take part. */
  where?: ArrayLike | boolean | number;
}

function allAny(u: BinaryUfunc, a: ArrayLike, opts: AllAnyOptions): NDArray {
  const r: UfuncReduceOptions = { axis: opts.axis === undefined ? null : opts.axis };
  if (opts.out !== undefined) r.out = opts.out;
  if (opts.keepdims !== undefined) r.keepdims = opts.keepdims;
  if (opts.where !== undefined) r.where = opts.where;
  return u.reduce(a, r);
}

/** NumPy all: true if every element (along `axis`) is truthy. */
export const all = (a: ArrayLike, opts: AllAnyOptions = {}): NDArray => allAny(logicalAnd, a, opts);
/** NumPy any: true if some element (along `axis`) is truthy. */
export const any = (a: ArrayLike, opts: AllAnyOptions = {}): NDArray => allAny(logicalOr, a, opts);

declare module "./ndarray.js" {
  interface NDArray {
    /** NumPy ndarray.all (see `np.all`). */
    all(opts?: AllAnyOptions): NDArray;
    /** NumPy ndarray.any (see `np.any`). */
    any(opts?: AllAnyOptions): NDArray;
  }
}
NDArray.prototype.all = function (this: NDArray, opts: AllAnyOptions = {}): NDArray {
  return all(this, opts);
};
NDArray.prototype.any = function (this: NDArray, opts: AllAnyOptions = {}): NDArray {
  return any(this, opts);
};
