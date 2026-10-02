// P11 products (D-140): vdot, kron, cross, tensordot, multiDot, NDArray.dot.
// Compositions of the native dot / multiply kernels; no linalg.ts import (cycle).
import { addon } from "./addon.js";
import { array } from "./creation.js";
import { IndexError, LinAlgError, ValueError, wrapNative } from "./errors.js";
import { take } from "./indexing.js";
import { NDArray } from "./ndarray.js";
import { expandDims, moveAxis } from "./shape.js";
import { conjugate, multiply, subtract, type ArrayLike } from "./ufunc.js";
import { toArray } from "./p11_native.js";

const dot2 = (a: NDArray, b: NDArray): NDArray =>
  wrapNative(() => NDArray._wrap(addon.linalg.dot(a._native, b._native)));
const prod = (v: readonly number[]): number => v.reduce((p, x) => p * x, 1);
const isComplex = (a: NDArray): boolean => a.dtype.kind === "c";

const normAxis = (ax: number, nd: number, prefix = ""): number => {
  if (!Number.isInteger(ax) || ax < -nd || ax >= nd) {
    throw new IndexError(`${prefix}axis ${ax} is out of bounds for array of dimension ${nd}`);
  }
  return ax < 0 ? ax + nd : ax;
};

/** Dot product of two arrays flattened to 1-D; the first argument is conjugated. */
export function vdot(a: ArrayLike, b: ArrayLike): NDArray {
  const x = toArray(a).reshape([-1]);
  const y = toArray(b).reshape([-1]);
  if (x.size !== y.size) {
    throw new ValueError(`cannot reshape array of size ${y.size} into shape (${x.size},)`);
  }
  return dot2(isComplex(x) ? conjugate(x) : x, y);
}

/** Kronecker product: blocks `a[i…] * b`; result shape is the elementwise product of the shapes. */
export function kron(a: ArrayLike, b: ArrayLike): NDArray {
  const x0 = toArray(a);
  const y0 = toArray(b);
  const nda = x0.ndim;
  const ndb = y0.ndim;
  if (nda === 0 || ndb === 0) return multiply(x0, y0);
  const nd = Math.max(nda, ndb);
  const as = [...Array<number>(Math.max(0, ndb - nda)).fill(1), ...x0.shape];
  const bs = [...Array<number>(Math.max(0, nda - ndb)).fill(1), ...y0.shape];
  const range = (start: number, stop: number, step = 1): number[] => {
    const r: number[] = [];
    for (let i = start; i < stop; i += step) r.push(i);
    return r;
  };
  let x = ndb > nda ? expandDims(x0, range(0, ndb - nda)) : x0;
  let y = nda > ndb ? expandDims(y0, range(0, nda - ndb)) : y0;
  x = expandDims(x, range(1, nd * 2, 2));
  y = expandDims(y, range(0, nd * 2, 2));
  return multiply(x, y).reshape(as.map((v, i) => v * bs[i]!));
}

export interface CrossOptions {
  /** Axis of `a` holding the vectors. Default -1. */
  axisa?: number;
  /** Axis of `b` holding the vectors. Default -1. */
  axisb?: number;
  /** Axis of the result holding the vectors. Default -1. */
  axisc?: number;
  /** Overrides `axisa`, `axisb` and `axisc` together. */
  axis?: number | null;
}
const IDX_A = [1, 2, 0];
const IDX_B = [2, 0, 1];
/** Cross product of (arrays of) 3-element vectors (NumPy 2 rejects 2-element vectors). */
export function cross(a: ArrayLike, b: ArrayLike, opts: CrossOptions = {}): NDArray {
  let x = toArray(a);
  let y = toArray(b);
  let { axisa = -1, axisb = -1, axisc = -1 } = opts;
  if (opts.axis !== undefined && opts.axis !== null) axisa = axisb = axisc = opts.axis;
  if (x.ndim < 1 || y.ndim < 1) throw new ValueError("At least one array has zero dimension");
  x = moveAxis(x, normAxis(axisa, x.ndim, "axisa: "), -1);
  y = moveAxis(y, normAxis(axisb, y.ndim, "axisb: "), -1);
  const la = x.shape[x.ndim - 1];
  const lb = y.shape[y.ndim - 1];
  if (la !== 3 || lb !== 3) {
    throw new ValueError(
      `Both input arrays must be (arrays of) 3-dimensional vectors, but they are ${la} and ${lb} dimensional instead.`,
    );
  }
  const ia = array(IDX_A, { dtype: "int64" });
  const ib = array(IDX_B, { dtype: "int64" });
  const cp = subtract(
    multiply(take(x, ia, -1), take(y, ib, -1)),
    multiply(take(x, ib, -1), take(y, ia, -1)),
  );
  return moveAxis(cp, normAxis(axisc, cp.ndim, "axisc: "), -1);
}

/** `np.linalg.cross`: array-API form with a single `axis` (default -1). */
export const linalgCross = (x1: ArrayLike, x2: ArrayLike, opts: { axis?: number } = {}): NDArray =>
  cross(x1, x2, { axis: opts.axis ?? -1 });

export type TensordotAxes = number | readonly [number | readonly number[], number | readonly number[]];
export interface TensordotOptions {
  /** `N` (last N of `a` with first N of `b`) or `[axesA, axesB]`. Default 2. */
  axes?: TensordotAxes;
}
/** Sum of products over the given axes of `a` and `b`. */
export function tensordot(a: ArrayLike, b: ArrayLike, opts: TensordotOptions = {}): NDArray {
  const x = toArray(a);
  const y = toArray(b);
  const axes = opts.axes ?? 2;
  let axesA: number[];
  let axesB: number[];
  if (typeof axes === "number") {
    if (!Number.isInteger(axes)) throw new ValueError("tensordot axes must be an integer");
    axesA = [];
    axesB = [];
    for (let i = -axes; i < 0; i++) axesA.push(i);
    for (let i = 0; i < axes; i++) axesB.push(i);
  } else {
    const [p, q] = axes;
    axesA = typeof p === "number" ? [p] : [...p];
    axesB = typeof q === "number" ? [q] : [...q];
  }
  if (new Set(axesA).size !== axesA.length || new Set(axesB).size !== axesB.length) {
    throw new ValueError("duplicate axes are not allowed in tensordot");
  }
  const as = x.shape;
  const bs = y.shape;
  let equal = axesA.length === axesB.length;
  for (let k = 0; equal && k < axesA.length; k++) {
    const sa = as.at(axesA[k]!);
    const sb = bs.at(axesB[k]!);
    if (sa === undefined || sb === undefined) {
      throw new IndexError("tuple index out of range");
    }
    if (sa !== sb) equal = false;
    if (axesA[k]! < 0) axesA[k]! += x.ndim;
    if (axesB[k]! < 0) axesB[k]! += y.ndim;
  }
  if (!equal) throw new ValueError("shape-mismatch for sum");
  const notinA = as.map((_, i) => i).filter((i) => !axesA.includes(i));
  const notinB = bs.map((_, i) => i).filter((i) => !axesB.includes(i));
  const olda = notinA.map((i) => as[i]!);
  const oldb = notinB.map((i) => bs[i]!);
  const n2 = prod(axesA.map((i) => as[i]!));
  const at = x.transpose([...notinA, ...axesA]).reshape([prod(olda), n2]);
  const bt = y.transpose([...axesB, ...notinB]).reshape([n2, prod(oldb)]);
  return dot2(at, bt).reshape([...olda, ...oldb]);
}

/** Dot product of two or more arrays, choosing the cheapest multiplication order. */
export function multiDot(arrays: readonly ArrayLike[]): NDArray {
  const n = arrays.length;
  if (n < 2) throw new ValueError("Expecting at least two arrays.");
  const arrs = arrays.map(toArray);
  if (n === 2) return dot2(arrs[0]!, arrs[1]!);
  const nd0 = arrs[0]!.ndim;
  const ndl = arrs[n - 1]!.ndim;
  if (nd0 === 1) arrs[0] = arrs[0]!.reshape([1, -1]);
  if (ndl === 1) arrs[n - 1] = arrs[n - 1]!.reshape([-1, 1]);
  for (const a of arrs) {
    if (a.ndim !== 2) {
      throw new LinAlgError(`${a.ndim}-dimensional array given. Array must be two-dimensional`);
    }
  }
  const p = [...arrs.map((a) => a.shape[0]!), arrs[n - 1]!.shape[1]!];
  const m = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  const s = Array.from({ length: n }, () => new Array<number>(n).fill(0));
  for (let l = 1; l < n; l++) {
    for (let i = 0; i < n - l; i++) {
      const j = i + l;
      m[i]![j] = Infinity;
      for (let k = i; k < j; k++) {
        const q = m[i]![k]! + m[k + 1]![j]! + p[i]! * p[k + 1]! * p[j + 1]!;
        if (q < m[i]![j]!) {
          m[i]![j] = q;
          s[i]![j] = k;
        }
      }
    }
  }
  const run = (i: number, j: number): NDArray =>
    i === j ? arrs[i]! : dot2(run(i, s[i]![j]!), run(s[i]![j]! + 1, j));
  const r = run(0, n - 1);
  if (nd0 === 1 && ndl === 1) return r.reshape([]);
  if (nd0 === 1 || ndl === 1) return r.reshape([-1]);
  return r;
}

declare module "./ndarray.js" {
  interface NDArray {
    /** NumPy `a.dot(b)` (same as `np.dot(a, b)`). */
    dot(b: ArrayLike): NDArray;
  }
}
NDArray.prototype.dot = function (this: NDArray, b: ArrayLike): NDArray {
  return dot2(this, toArray(b));
};

const matmul2 = (a: NDArray, b: NDArray): NDArray =>
  wrapNative(() => NDArray._wrap(addon.linalg.matmul(a._native, b._native)));

const coreCheck = (
  name: string,
  sig: string,
  ops: readonly [NDArray, number][],
): void => {
  ops.forEach(([a, need], i) => {
    if (a.ndim < need) {
      throw new ValueError(
        `${name}: Input operand ${i} does not have enough dimensions (has ${a.ndim}, gufunc core with signature ${sig} requires ${need})`,
      );
    }
  });
};
const mismatch = (name: string, sig: string, op: number, dim: number, got: number, want: number): never => {
  throw new ValueError(
    `${name}: Input operand ${op} has a mismatch in its core dimension ${dim}, with gufunc signature ${sig} (size ${got} is different from ${want})`,
  );
};
const conjIf = (a: NDArray): NDArray => (isComplex(a) ? conjugate(a) : a);

export interface VecdotOptions {
  /** Axis holding the vectors (after broadcasting). Default -1. */
  axis?: number;
}
const VECDOT_SIG = "(n),(n)->()";
/** Vector dot product `sum(conj(x1) * x2)` along `axis`, broadcasting the other axes. */
export function vecdot(x1: ArrayLike, x2: ArrayLike, opts: VecdotOptions = {}): NDArray {
  let a = toArray(x1);
  let b = toArray(x2);
  coreCheck("vecdot", VECDOT_SIG, [[a, 1], [b, 1]]);
  const axis = opts.axis ?? -1;
  if (axis !== -1) {
    a = moveAxis(a, normAxis(axis, a.ndim), -1);
    b = moveAxis(b, normAxis(axis, b.ndim), -1);
  }
  const n = a.shape[a.ndim - 1]!;
  const m = b.shape[b.ndim - 1]!;
  if (n !== m) mismatch("vecdot", VECDOT_SIG, 1, 0, m, n);
  const r = matmul2(expandDims(conjIf(a), -2), expandDims(b, -1));
  return r.reshape(r.shape.slice(0, -2));
}

const MATVEC_SIG = "(m,n),(n)->(m)";
/** Matrix-vector product over the last axes, broadcasting the rest: `x1 @ x2[..., None]`. */
export function matvec(x1: ArrayLike, x2: ArrayLike): NDArray {
  const a = toArray(x1);
  const b = toArray(x2);
  coreCheck("matvec", MATVEC_SIG, [[a, 2], [b, 1]]);
  const n = a.shape[a.ndim - 1]!;
  const m = b.shape[b.ndim - 1]!;
  if (n !== m) mismatch("matvec", MATVEC_SIG, 1, 0, m, n);
  const r = matmul2(a, expandDims(b, -1));
  return r.reshape(r.shape.slice(0, -1));
}

const VECMAT_SIG = "(n),(n,m)->(m)";
/** Vector-matrix product `conj(x1)[..., None, :] @ x2`, broadcasting the batch axes. */
export function vecmat(x1: ArrayLike, x2: ArrayLike): NDArray {
  const a = toArray(x1);
  const b = toArray(x2);
  coreCheck("vecmat", VECMAT_SIG, [[a, 1], [b, 2]]);
  const n = a.shape[a.ndim - 1]!;
  const m = b.shape[b.ndim - 2]!;
  if (n !== m) mismatch("vecmat", VECMAT_SIG, 1, 0, m, n);
  const r = matmul2(expandDims(conjIf(a), -2), b);
  return r.reshape([...r.shape.slice(0, -2), r.shape[r.ndim - 1]!]);
}
