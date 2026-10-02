// P11 np.linalg completion (D-140): cholesky, slogdet, svdvals, matrixPower, ...
// Spread into the `linalg` object by linalg.ts. Only imports lower layers
// (never linalg.ts) to avoid an import cycle.
import { addon } from "./addon.js";
import { DTypeError, wrapNative } from "./errors.js";
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

export const p11Linalg = { cholesky, slogdet, svdvals, matrixPower } as const;
