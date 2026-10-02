import { addon, nativeModule, type FftNorm, type NativeNDArray } from "./addon.js";
import { array } from "./creation.js";
import { DTypeError, ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import type { ArrayLike } from "./ufunc.js";

/**
 * Discrete Fourier transforms mirroring `numpy.fft` (PLAN §24, M10, P12).
 * Backed by vendored pocketfft; semantics in DECISIONS D-020 and D-150..D-152.
 *
 * Each function takes NumPy's positional parameters, or a trailing
 * options object, e.g. `fft(a, 8)` or `fft(a, { n: 8, norm: "ortho" })`.
 * Complex results are complex64/complex128 NDArrays; read them with
 * `toTypedArray()` (interleaved re/im, D-008). `out` writes the result into
 * an existing array and returns it (D-150).
 */

export type { FftNorm };

export interface FftOptions {
  n?: number | null;
  axis?: number;
  norm?: FftNorm | null;
  out?: NDArray | null;
}
export interface FftNOptions {
  s?: number[] | null;
  axes?: number[] | null;
  norm?: FftNorm | null;
  out?: NDArray | null;
}

type N = NativeNDArray;
interface P12Native {
  fft(a: N, n: number | null, axis: number, norm: string | null, out: N | null): N;
  ifft(a: N, n: number | null, axis: number, norm: string | null, out: N | null): N;
  rfft(a: N, n: number | null, axis: number, norm: string | null, out: N | null): N;
  irfft(a: N, n: number | null, axis: number, norm: string | null, out: N | null): N;
  hfft(a: N, n: number | null, axis: number, norm: string | null, out: N | null): N;
  ihfft(a: N, n: number | null, axis: number, norm: string | null, out: N | null): N;
  fftn(a: N, s: number[] | null, axes: number[] | null, norm: string | null, out: N | null): N;
  ifftn(a: N, s: number[] | null, axes: number[] | null, norm: string | null, out: N | null): N;
  rfftn(a: N, s: number[] | null, axes: number[] | null, norm: string | null, out: N | null): N;
  irfftn(a: N, s: number[] | null, axes: number[] | null, norm: string | null, out: N | null): N;
  shift(a: N, axes: number[] | null, inverse: boolean): N;
}
const native = nativeModule<P12Native>("p12");

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

function checkOut(out: unknown): NDArray | null {
  if (out === null || out === undefined) return null;
  if (!(out instanceof NDArray)) throw new DTypeError("return arrays must be of ArrayType");
  return out;
}

type OneDName = "fft" | "ifft" | "rfft" | "irfft" | "hfft" | "ihfft";
type NDName = "fftn" | "ifftn" | "rfftn" | "irfftn";

/** Returns `out` itself (same JS object) when given, else the new result. */
const finish = (res: N, out: NDArray | null): NDArray => out ?? w(res);

function oneD(
  name: OneDName,
  a: ArrayLike,
  n: number | null | undefined | FftOptions,
  axis: number,
  norm: FftNorm | null | undefined,
  out: NDArray | null | undefined,
): NDArray {
  const o: FftOptions = isOptions(n) ? n : { n: n ?? null, axis, norm, out };
  const nn = o.n === null || o.n === undefined ? null : checkInt(o.n, "n");
  const ax = checkInt(o.axis ?? -1, "axis");
  const nm = checkNorm(o.norm);
  const ot = checkOut(o.out);
  const x = toArray(a);
  return wrapNative(() => finish(native[name](x._native, nn, ax, nm, ot?._native ?? null), ot));
}

function nD(
  name: NDName,
  a: ArrayLike,
  s: number[] | null | undefined | FftNOptions,
  axes: number[] | null | undefined,
  norm: FftNorm | null | undefined,
  out: NDArray | null | undefined,
  defaultAxes: number[] | null,
): NDArray {
  const o: FftNOptions = isOptions(s) ? s : { s: s ?? null, axes, norm, out };
  const ss = checkInts(o.s, "s");
  const ax = checkInts(o.axes, "axes") ?? defaultAxes;
  const nm = checkNorm(o.norm);
  const ot = checkOut(o.out);
  const x = toArray(a);
  // fftn/ifftn over no axes return the input unchanged (NumPy), even with `out`.
  const noAxes = (ax ?? ss ?? x.shape).length === 0 && (name === "fftn" || name === "ifftn");
  return wrapNative(() => {
    const r = native[name](x._native, ss, ax, nm, ot?._native ?? null);
    return noAxes ? x : finish(r, ot);
  });
}

type OneD = (
  a: ArrayLike,
  n?: number | null | FftOptions,
  axis?: number,
  norm?: FftNorm | null,
  out?: NDArray | null,
) => NDArray;
type ND = (
  a: ArrayLike,
  s?: number[] | null | FftNOptions,
  axes?: number[] | null,
  norm?: FftNorm | null,
  out?: NDArray | null,
) => NDArray;

const make1 = (name: OneDName): OneD => (a, n, axis = -1, norm, out) => oneD(name, a, n, axis, norm, out);
const makeN = (name: NDName, def: number[] | null): ND => (a, s, axes, norm, out) =>
  nD(name, a, s, axes === undefined ? def : axes, norm, out, def);

/** 1-D complex DFT: `fft(a, n?, axis = -1, norm?, out?)`. */
export const fft: OneD = make1("fft");
/** 1-D inverse complex DFT: `ifft(a, n?, axis = -1, norm?, out?)`. */
export const ifft: OneD = make1("ifft");
/** 1-D DFT of real input; returns `n // 2 + 1` complex values. */
export const rfft: OneD = make1("rfft");
/** Inverse of `rfft`; default `n = 2 * (m - 1)`. Returns a real array. */
export const irfft: OneD = make1("irfft");
/** DFT of a Hermitian-symmetric signal (real spectrum); default `n = 2 * (m - 1)`. */
export const hfft: OneD = make1("hfft");
/** Inverse of `hfft`: `conj(rfft(a)) / n`; returns `n // 2 + 1` complex values. */
export const ihfft: OneD = make1("ihfft");

/** N-D complex DFT over `axes` (default: all, or the last `s.length`). */
export const fftn: ND = makeN("fftn", null);
/** N-D inverse complex DFT. */
export const ifftn: ND = makeN("ifftn", null);
/** 2-D complex DFT; `axes` defaults to `[-2, -1]`. */
export const fft2: ND = makeN("fftn", [-2, -1]);
/** 2-D inverse complex DFT; `axes` defaults to `[-2, -1]`. */
export const ifft2: ND = makeN("ifftn", [-2, -1]);
/** N-D DFT of real input (`rfft` over the last axis, `fft` over the others). */
export const rfftn: ND = makeN("rfftn", null);
/** Inverse of `rfftn`; the last length defaults to `2 * (m - 1)`. */
export const irfftn: ND = makeN("irfftn", null);
/** 2-D `rfftn`; `axes` defaults to `[-2, -1]`. */
export const rfft2: ND = makeN("rfftn", [-2, -1]);
/** 2-D `irfftn`; `axes` defaults to `[-2, -1]`. */
export const irfft2: ND = makeN("irfftn", [-2, -1]);

function shiftAxes(axes: number | number[] | null | undefined): number[] | null {
  if (axes === null || axes === undefined) return null;
  if (typeof axes === "number") return [checkInt(axes, "axes")];
  return checkInts(axes, "axes");
}

/** Moves the zero-frequency term to the centre: `roll` by `shape[ax] // 2` over `axes` (default all). */
export const fftshift = (x: ArrayLike, axes?: number | number[] | null): NDArray => {
  const ax = shiftAxes(axes);
  const a = toArray(x);
  return wrapNative(() => w(native.shift(a._native, ax, false)));
};

/** Inverse of `fftshift` (they differ for odd lengths). */
export const ifftshift = (x: ArrayLike, axes?: number | number[] | null): NDArray => {
  const ax = shiftAxes(axes);
  const a = toArray(x);
  return wrapNative(() => w(native.shift(a._native, ax, true)));
};

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
  rfft2,
  irfft2,
  rfftn,
  irfftn,
  hfft,
  ihfft,
  fftfreq,
  rfftfreq,
  fftshift,
  ifftshift,
} as const;
