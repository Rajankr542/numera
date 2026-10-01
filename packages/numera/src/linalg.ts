import { addon } from "./addon.js";
import { array } from "./creation.js";
import { LinAlgError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import type { ArrayLike } from "./ufunc.js";

/**
 * Linear algebra (PLAN §19/§20, M8). Semantics: DECISIONS D-018.
 * Top-level `matmul`/`dot`/`inner`/`outer`, and the `linalg` namespace
 * mirroring `numpy.linalg`.
 */

const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));
const w = NDArray._wrap;
const wOpt = (n: Parameters<typeof NDArray._wrap>[0] | null): NDArray | null =>
  n === null ? null : w(n);

export const matmul = (a: ArrayLike, b: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.matmul(toArray(a)._native, toArray(b)._native)));
export const dot = (a: ArrayLike, b: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.dot(toArray(a)._native, toArray(b)._native)));
export const inner = (a: ArrayLike, b: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.inner(toArray(a)._native, toArray(b)._native)));
export const outer = (a: ArrayLike, b: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.outer(toArray(a)._native, toArray(b)._native)));

export const det = (a: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.det(toArray(a)._native)));
export const inv = (a: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.inv(toArray(a)._native)));
export const solve = (a: ArrayLike, b: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.solve(toArray(a)._native, toArray(b)._native)));

export interface EigResult {
  eigenvalues: NDArray;
  eigenvectors: NDArray;
}
/**
 * General eigen-decomposition; results are always complex (D-018). Complex
 * input is accepted and keeps its precision (D-043).
 */
export function eig(a: ArrayLike): EigResult {
  return wrapNative(() => {
    const r = addon.linalg.eig(toArray(a)._native);
    return { eigenvalues: w(r.eigenvalues), eigenvectors: w(r.eigenvectors) };
  });
}
/**
 * Hermitian (real: symmetric) eigen-decomposition using the lower triangle.
 * Eigenvalues are real and ascending; complex input is accepted (D-042).
 */
export function eigh(a: ArrayLike): EigResult {
  return wrapNative(() => {
    const r = addon.linalg.eigh(toArray(a)._native);
    return { eigenvalues: w(r.eigenvalues), eigenvectors: w(r.eigenvectors) };
  });
}
/** Eigenvalues only (LAPACK JOBVR='N', like NumPy), D-043. */
export const eigvals = (a: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.eigvals(toArray(a)._native)));
/** Eigenvalues only (LAPACK JOBZ='N', like NumPy), D-042. */
export const eigvalsh = (a: ArrayLike): NDArray =>
  wrapNative(() => w(addon.linalg.eigvalsh(toArray(a)._native)));

export interface SvdOptions {
  fullMatrices?: boolean;
  computeUV?: boolean;
}
export interface SvdResult {
  U: NDArray | null;
  S: NDArray;
  Vh: NDArray | null;
}
export function svd(a: ArrayLike, opts: SvdOptions = {}): SvdResult {
  return wrapNative(() => {
    const r = addon.linalg.svd(
      toArray(a)._native,
      opts.fullMatrices ?? true,
      opts.computeUV ?? true,
    );
    return { U: wOpt(r.U), S: w(r.S), Vh: wOpt(r.Vh) };
  });
}

export type QrMode = "reduced" | "complete" | "r";
export interface QrResult {
  Q: NDArray | null;
  R: NDArray;
}
export function qr(a: ArrayLike, mode: QrMode = "reduced"): QrResult {
  return wrapNative(() => {
    const r = addon.linalg.qr(toArray(a)._native, mode);
    return { Q: wOpt(r.Q), R: w(r.R) };
  });
}

export interface LstsqResult {
  x: NDArray;
  residuals: NDArray;
  rank: number;
  s: NDArray;
}
/** Least squares via SVD. `rcond` null/omitted = eps * max(M, N) (NumPy 2 default). */
export function lstsq(a: ArrayLike, b: ArrayLike, rcond: number | null = null): LstsqResult {
  return wrapNative(() => {
    const r = addon.linalg.lstsq(toArray(a)._native, toArray(b)._native, rcond);
    return { x: w(r.x), residuals: w(r.residuals), rank: r.rank, s: w(r.s) };
  });
}

export type NormOrder = number | "fro" | "nuc" | null;
export interface NormOptions {
  ord?: NormOrder;
  axis?: number | readonly [number, number] | null;
  keepdims?: boolean;
}
export function norm(a: ArrayLike, opts: NormOptions = {}): NDArray {
  const axis = opts.axis ?? null;
  return wrapNative(() =>
    w(
      addon.linalg.norm(
        toArray(a)._native,
        opts.ord ?? null,
        axis === null ? null : typeof axis === "number" ? [axis] : [...axis],
        opts.keepdims ?? false,
      ),
    ),
  );
}

/** Name of the active native backend ("accelerate" or "fallback"). */
export const backend = (): string => addon.linalg.backend();

/** @internal Testing hook: force the portable fallback backend or restore the default. */
export const _setBackend = (which: "fallback" | "default"): void =>
  wrapNative(() => addon.linalg._setBackend(which));

export const linalg = {
  matmul,
  det,
  inv,
  solve,
  eig,
  eigh,
  eigvals,
  eigvalsh,
  svd,
  qr,
  lstsq,
  norm,
  backend,
  _setBackend,
  LinAlgError,
} as const;
