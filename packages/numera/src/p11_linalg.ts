// P11 np.linalg completion (D-140): cholesky, slogdet, svdvals, matrixPower, ...
// Spread into the `linalg` object by linalg.ts. Only imports lower layers
// (never linalg.ts) to avoid an import cycle.
import { addon } from "./addon.js";
import { DTypeError, ValueError, wrapNative } from "./errors.js";
import type { NormOrder } from "./linalg.js";
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

export const p11Linalg = { cholesky, slogdet, svdvals, matrixPower, pinv, matrixRank, cond } as const;
