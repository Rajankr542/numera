import { addon, type FftNorm } from "./addon.js";
import { array } from "./creation.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import type { ArrayLike } from "./ufunc.js";

/**
 * Discrete Fourier transforms mirroring `numpy.fft` (PLAN §24, M10).
 * Backed by vendored pocketfft; semantics in DECISIONS D-020.
 *
 * Each function takes NumPy's positional parameters, or a trailing
 * options object, e.g. `fft(a, 8)` or `fft(a, { n: 8, norm: "ortho" })`.
 * Complex results are complex64/complex128 NDArrays; read them with
 * `toTypedArray()` (interleaved re/im, D-008).
 */

export type { FftNorm };

export interface FftOptions {
  n?: number | null;
  axis?: number;
  norm?: FftNorm | null;
}
export interface FftNOptions {
  s?: number[] | null;
  axes?: number[] | null;
  norm?: FftNorm | null;
}

const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));
const w = NDArray._wrap;

function isOptions(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v) && !(v instanceof NDArray);
}

function checkNorm(norm: FftNorm | null | undefined): FftNorm | null {
  if (norm === null || norm === undefined) return null;
  if (norm !== "backward" && norm !== "ortho" && norm !== "forward") {
    throw new ValueError(
      `Invalid norm value ${String(norm)}; should be "backward", "ortho" or "forward".`,
    );
  }
  return norm;
}

function checkInt(v: unknown, what: string): number {
  if (typeof v !== "number" || !Number.isInteger(v)) {
    throw new ValueError(`${what} must be an integer`);
  }
  return v;
}

function checkInts(v: unknown, what: string): number[] | null {
  if (v === null || v === undefined) return null;
  if (!Array.isArray(v)) throw new ValueError(`${what} must be an array of integers`);
  return v.map((x) => checkInt(x, what));
}

type OneDName = "fft" | "ifft" | "rfft" | "irfft";
type NDName = "fftn" | "ifftn";

function oneD(
  name: OneDName,
  a: ArrayLike,
  n: number | null | undefined | FftOptions,
  axis: number,
  norm: FftNorm | null | undefined,
): NDArray {
  const o: FftOptions = isOptions(n) ? n : { n: n ?? null, axis, norm };
  const nn = o.n === null || o.n === undefined ? null : checkInt(o.n, "n");
  const ax = checkInt(o.axis ?? -1, "axis");
  const nm = checkNorm(o.norm);
  const x = toArray(a);
  return wrapNative(() => w(addon.fft[name](x._native, nn, ax, nm)));
}

function nD(
  name: NDName,
  a: ArrayLike,
  s: number[] | null | undefined | FftNOptions,
  axes: number[] | null | undefined,
  norm: FftNorm | null | undefined,
  defaultAxes: number[] | null,
): NDArray {
  const o: FftNOptions = isOptions(s) ? s : { s: s ?? null, axes, norm };
  const ss = checkInts(o.s, "s");
  const ax = checkInts(o.axes, "axes") ?? defaultAxes;
  const nm = checkNorm(o.norm);
  const x = toArray(a);
  return wrapNative(() => w(addon.fft[name](x._native, ss, ax, nm)));
}

/** 1-D complex DFT: `fft(a, n?, axis = -1, norm?)`. */
export const fft = (
  a: ArrayLike,
  n?: number | null | FftOptions,
  axis = -1,
  norm?: FftNorm | null,
): NDArray => oneD("fft", a, n, axis, norm);

/** 1-D inverse complex DFT: `ifft(a, n?, axis = -1, norm?)`. */
export const ifft = (
  a: ArrayLike,
  n?: number | null | FftOptions,
  axis = -1,
  norm?: FftNorm | null,
): NDArray => oneD("ifft", a, n, axis, norm);

/** 1-D DFT of real input; returns `n // 2 + 1` complex values. */
export const rfft = (
  a: ArrayLike,
  n?: number | null | FftOptions,
  axis = -1,
  norm?: FftNorm | null,
): NDArray => oneD("rfft", a, n, axis, norm);

/** Inverse of `rfft`; default `n = 2 * (m - 1)`. Returns a real array. */
export const irfft = (
  a: ArrayLike,
  n?: number | null | FftOptions,
  axis = -1,
  norm?: FftNorm | null,
): NDArray => oneD("irfft", a, n, axis, norm);

/** N-D complex DFT over `axes` (default: all, or the last `s.length`). */
export const fftn = (
  a: ArrayLike,
  s?: number[] | null | FftNOptions,
  axes?: number[] | null,
  norm?: FftNorm | null,
): NDArray => nD("fftn", a, s, axes, norm, null);

/** N-D inverse complex DFT. */
export const ifftn = (
  a: ArrayLike,
  s?: number[] | null | FftNOptions,
  axes?: number[] | null,
  norm?: FftNorm | null,
): NDArray => nD("ifftn", a, s, axes, norm, null);

/** 2-D complex DFT; `axes` defaults to `[-2, -1]`. */
export const fft2 = (
  a: ArrayLike,
  s?: number[] | null | FftNOptions,
  axes: number[] | null = [-2, -1],
  norm?: FftNorm | null,
): NDArray => nD("fftn", a, s, axes, norm, [-2, -1]);

/** 2-D inverse complex DFT; `axes` defaults to `[-2, -1]`. */
export const ifft2 = (
  a: ArrayLike,
  s?: number[] | null | FftNOptions,
  axes: number[] | null = [-2, -1],
  norm?: FftNorm | null,
): NDArray => nD("ifftn", a, s, axes, norm, [-2, -1]);

/** Sample frequencies for `fft` output: float64 array of length `n`. */
export const fftfreq = (n: number, d = 1.0): NDArray => {
  const nn = checkInt(n, "n");
  if (typeof d !== "number") throw new ValueError("d must be a number");
  return wrapNative(() => w(addon.fft.fftfreq(nn, d)));
};

/** Sample frequencies for `rfft` output: float64 array of length `n // 2 + 1`. */
export const rfftfreq = (n: number, d = 1.0): NDArray => {
  const nn = checkInt(n, "n");
  if (typeof d !== "number") throw new ValueError("d must be a number");
  return wrapNative(() => w(addon.fft.rfftfreq(nn, d)));
};

/** The `np.fft` namespace. */
export const fftModule = {
  fft,
  ifft,
  rfft,
  irfft,
  fft2,
  ifft2,
  fftn,
  ifftn,
  fftfreq,
  rfftfreq,
} as const;
