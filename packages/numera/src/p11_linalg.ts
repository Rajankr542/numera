// P11 np.linalg completion (D-140): cholesky, slogdet, svdvals, matrixPower, ...
// Spread into the `linalg` object by linalg.ts. Only imports lower layers
// (never linalg.ts) to avoid an import cycle.
import { addon } from "./addon.js";
import type { DTypeLike } from "./dtype.js";
import { DTypeError, IndexError, LinAlgError, ValueError, wrapNative } from "./errors.js";
import { diagonal, trace } from "./p08.js";
import { NDArray } from "./ndarray.js";
import type { ArrayLike } from "./ufunc.js";
import { native, toArray, w } from "./p11_native.js";

export interface CholeskyOptions {
  /** Return the upper factor `U` (`A = Uᴴ U`) instead of the lower `L`. Default `false`. */
  upper?: boolean;
}
/** Cholesky factor of a Hermitian positive-definite matrix (stack), D-142. */
export const cholesky = (a: ArrayLike, opts: CholeskyOptions = {}): NDArray =>
  w((n) => n.cholesky(toArray(a)._native, opts.upper ?? false));

export interface SlogdetResult {
  sign: NDArray;
  logabsdet: NDArray;
}
/** Sign and natural log of |det(a)| (NumPy `SlogdetResult`). */
export function slogdet(a: ArrayLike): SlogdetResult {
  return wrapNative(() => {
    const r = native().slogdet(toArray(a)._native);
    return { sign: NDArray._wrap(r.sign), logabsdet: NDArray._wrap(r.logabsdet) };
  });
}

/** Singular values in descending order (`svd(x, {computeUV: false}).S`). */
export const svdvals = (x: ArrayLike): NDArray =>
  wrapNative(() => NDArray._wrap(addon.linalg.svd(toArray(x)._native, false, false).S));

/** `a` raised to the integer power `n` (n < 0 inverts first). */
export function matrixPower(a: ArrayLike, n: number): NDArray {
  if (typeof n !== "number" || !Number.isSafeInteger(n)) {
    throw new DTypeError("exponent must be an integer");
  }
  return w((m) => m.matrixPower(toArray(a)._native, n));
}

export interface PinvOptions {
  /** Cutoff relative to the largest singular value (NumPy default `1e-15`); broadcasts to the batch. */
  rcond?: ArrayLike | number | null;
  /** Array-API spelling of `rcond`; `null` means `max(M, N) * eps`. Exclusive with `rcond`. */
  rtol?: ArrayLike | number | null;
  /** Assume `a` is Hermitian and use `eigh`. Default `false`. */
  hermitian?: boolean;
}
const EPS64 = 2.220446049250313e-16;
const EPS32 = 1.1920928955078125e-7;
/** Moore–Penrose pseudo-inverse (stack). */
export function pinv(a: ArrayLike, opts: PinvOptions = {}): NDArray {
  const x = toArray(a);
  let rcond: ArrayLike | number;
  if (opts.rcond !== undefined && opts.rcond !== null) {
    if (opts.rtol !== undefined) throw new ValueError("`rtol` and `rcond` can't be both set.");
    rcond = opts.rcond;
  } else if (opts.rtol !== undefined && opts.rtol !== null) {
    rcond = opts.rtol;
  } else if (opts.rtol === null) {
    const s = x.shape;
    const lowp = x.dtype.name === "float32" || x.dtype.name === "complex64";
    rcond = Math.max(s[s.length - 1] ?? 0, s[s.length - 2] ?? 0) * (lowp ? EPS32 : EPS64);
  } else {
    rcond = 1e-15;
  }
  return w((n) => n.pinv(x._native, toArray(rcond as ArrayLike)._native, opts.hermitian ?? false));
}

export interface MatrixRankOptions {
  /** Absolute threshold below which singular values count as zero. */
  tol?: ArrayLike | number | null;
  /** Relative threshold (times the largest singular value). Exclusive with `tol`. */
  rtol?: ArrayLike | number | null;
  /** Assume `A` is Hermitian (uses |eigenvalues|). Default `false`. */
  hermitian?: boolean;
}
/** Matrix rank via SVD (int64, batch shape). ndim < 2: 1 if any element is nonzero. */
export function matrixRank(A: ArrayLike, opts: MatrixRankOptions = {}): NDArray {
  const opt = (v: ArrayLike | number | null | undefined) =>
    v === undefined || v === null ? null : toArray(v as ArrayLike)._native;
  return w((n) => n.matrixRank(toArray(A)._native, opt(opts.tol), opt(opts.rtol), opts.hermitian ?? false));
}

/** Condition number of `x` in norm `p` (`null`/2/-2 via SVD; 1, -1, ±Infinity, "fro", "nuc" via inv). */
export const cond = (x: ArrayLike, p: NormOrder = null): NDArray =>
  w((n) => n.cond(toArray(x)._native, p));

export type NormOrder = number | "fro" | "nuc" | null;
const L = () => addon.linalg;
const inv = (a: NDArray): NDArray => wrapNative(() => NDArray._wrap(L().inv(a._native)));
const solve = (a: NDArray, b: NDArray): NDArray =>
  wrapNative(() => NDArray._wrap(L().solve(a._native, b._native)));
const outer = (a: NDArray, b: NDArray): NDArray =>
  wrapNative(() => NDArray._wrap(L().outer(a._native, b._native)));
const norm = (a: NDArray, o: { ord: NormOrder; axis: number[]; keepdims?: boolean }): NDArray =>
  wrapNative(() => NDArray._wrap(L().norm(a._native, o.ord, o.axis, o.keepdims ?? false)));

const normAxis = (ax: number, nd: number): number => {
  if (!Number.isInteger(ax) || ax < -nd || ax >= nd) {
    throw new IndexError(`axis ${ax} is out of bounds for array of dimension ${nd}`);
  }
  return ax < 0 ? ax + nd : ax;
};

export interface VectorNormOptions {
  /** Axis or axes to reduce (any number); `null` flattens. Default `null`. */
  axis?: number | readonly number[] | null;
  keepdims?: boolean;
  /** Vector norm order (number, ±Infinity). Default 2. */
  ord?: number;
}
/** Vector norm over one or several axes (array-API `vector_norm`). */
export function vectorNorm(x: ArrayLike, opts: VectorNormOptions = {}): NDArray {
  let a = toArray(x);
  const shape = [...a.shape];
  const ord = opts.ord ?? 2;
  if (typeof ord !== "number") throw new ValueError(`Invalid norm order '${String(ord)}' for vectors`);
  const axis = opts.axis ?? null;
  let axes: number[];
  if (axis === null) {
    a = a.reshape([-1]);
    axes = shape.map((_, i) => i);
  } else if (typeof axis === "number") {
    axes = [normAxis(axis, shape.length)];
  } else {
    axes = axis.map((v) => normAxis(v, shape.length));
    if (new Set(axes).size !== axes.length) throw new ValueError("repeated axis");
    const rest = shape.map((_, i) => i).filter((i) => !axes.includes(i));
    const size = axes.reduce((p, i) => p * shape[i]!, 1);
    a = a.transpose([...axes, ...rest]).reshape([size, ...rest.map((i) => shape[i]!)]);
  }
  const single = typeof axis === "number" ? axes[0]! : 0;
  let res = norm(a, { ord, axis: [single] });
  if (opts.keepdims) {
    for (const i of axes) shape[i] = 1;
    res = res.reshape(shape);
  }
  return res;
}

export interface MatrixNormOptions {
  keepdims?: boolean;
  /** Matrix norm order: "fro" (default), "nuc", 1, -1, 2, -2, ±Infinity. */
  ord?: NormOrder;
}
/** Matrix norm over the last two axes (array-API `matrix_norm`). */
export function matrixNorm(x: ArrayLike, opts: MatrixNormOptions = {}): NDArray {
  const a = toArray(x);
  if (a.ndim < 2) throw new IndexError(`axis -2 is out of bounds for array of dimension ${a.ndim}`);
  return norm(a, { ord: opts.ord ?? "fro", axis: [-2, -1], keepdims: opts.keepdims ?? false });
}

/** Swaps the last two axes (view). */
export function matrixTranspose(x: ArrayLike): NDArray {
  const a = toArray(x);
  if (a.ndim < 2) {
    throw new ValueError(`Input array must be at least 2-dimensional, but it is ${a.ndim}`);
  }
  return a.swapAxes(-1, -2);
}

/** Diagonals of the trailing matrices (`axis1=-2, axis2=-1`). */
export const linalgDiagonal = (x: ArrayLike, opts: { offset?: number } = {}): NDArray =>
  diagonal(x, { offset: opts.offset ?? 0, axis1: -2, axis2: -1 });

/** Traces of the trailing matrices (`axis1=-2, axis2=-1`). */
export const linalgTrace = (x: ArrayLike, opts: { offset?: number; dtype?: DTypeLike | null } = {}): NDArray =>
  trace(x, { offset: opts.offset ?? 0, axis1: -2, axis2: -1, dtype: opts.dtype ?? null });

/** Outer product of two 1-D arrays (array-API: no flattening). */
export function linalgOuter(x1: ArrayLike, x2: ArrayLike): NDArray {
  const a = toArray(x1);
  const b = toArray(x2);
  if (a.ndim !== 1 || b.ndim !== 1) {
    throw new ValueError(
      `Input arrays must be one-dimensional, but they are x1.ndim=${a.ndim} and x2.ndim=${b.ndim}.`,
    );
  }
  return outer(a, b);
}

const prod = (v: readonly number[]): number => v.reduce((p, x) => p * x, 1);

export interface TensorinvOptions {
  /** Number of leading indices in the inverse product. Default 2. */
  ind?: number;
}
/** Inverse of an N-d array w.r.t. `tensordot(…, ind)`; result shape `a.shape[ind:] + a.shape[:ind]`. */
export function tensorinv(a: ArrayLike, opts: TensorinvOptions = {}): NDArray {
  const x = toArray(a);
  const ind = opts.ind ?? 2;
  if (!Number.isInteger(ind) || ind <= 0) throw new ValueError("Invalid ind argument.");
  const s = x.shape;
  const invshape = [...s.slice(ind), ...s.slice(0, ind)];
  const p = prod(s.slice(ind));
  return inv(x.reshape([p, p === 0 ? 0 : x.size / p])).reshape(invshape);
}

export interface TensorsolveOptions {
  /** Axes of `a` moved to the end before solving. */
  axes?: readonly number[] | null;
}
/** Solves `tensordot(a, x, x.ndim) = b` for `x`. */
export function tensorsolve(a: ArrayLike, b: ArrayLike, opts: TensorsolveOptions = {}): NDArray {
  let x = toArray(a);
  const y = toArray(b);
  const an = x.ndim;
  if (opts.axes) {
    const all = Array.from({ length: an }, (_, i) => i);
    for (const k of opts.axes) {
      const kk = normAxis(k, an);
      all.splice(all.indexOf(kk), 1);
      all.push(kk);
    }
    x = x.transpose(all);
  }
  const oldshape = x.shape.slice(y.ndim);
  const p = prod(oldshape);
  if (x.size !== p * p) {
    throw new LinAlgError(
      "Input arrays must satisfy the requirement prod(a.shape[b.ndim:]) == prod(a.shape[:b.ndim])",
    );
  }
  return solve(x.reshape([p, p]), y.reshape([-1])).reshape(oldshape);
}

export const p11Linalg = {
  cholesky,
  slogdet,
  svdvals,
  matrixPower,
  pinv,
  matrixRank,
  cond,
  vectorNorm,
  matrixNorm,
  matrixTranspose,
  diagonal: linalgDiagonal,
  trace: linalgTrace,
  outer: linalgOuter,
  tensorinv,
  tensorsolve,
} as const;
