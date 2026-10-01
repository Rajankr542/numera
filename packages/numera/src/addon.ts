import { createRequire } from "node:module";
import { existsSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

/** Handle to a native NDArray. Internal: not part of the public API. */
export interface NativeNDArray {
  shape(): number[];
  strides(): number[];
  offset(): number;
  dtype(): string;
  ndim(): number;
  size(): number;
  itemsize(): number;
  nbytes(): number;
  flags(): { cContiguous: boolean; fContiguous: boolean; ownData: boolean; writeable: boolean };
  toList(): unknown;
  toTypedArray(): ArrayBufferView;
  view(shape: number[], strides: number[], offset: number): NativeNDArray;
  reshape(shape: number[]): NativeNDArray;
  copy(): NativeNDArray;
  astype(dtype: string): NativeNDArray;
  sharesMemory(other: NativeNDArray): boolean;
  getItem(index: number[]): number | boolean | { re: number; im: number };
}

export interface NativeAddon {
  napiVersion: number;
  dtypes: Record<string, { itemsize: number; alignment: number; kind: string }>;
  promoteTypes(a: string, b: string): string;
  NativeNDArray: new (...args: never[]) => NativeNDArray;
  empty(shape: number[], dtype: string): NativeNDArray;
  zeros(shape: number[], dtype: string): NativeNDArray;
  fromNested(data: unknown, dtype: string): NativeNDArray;
  fromTypedArray(data: ArrayBufferView, shape: number[], dtype: string): NativeNDArray;
  fromFloat64(data: Float64Array, shape: number[], dtype: string): NativeNDArray;
  memoryStats(): { buffers: number; bytes: number };
  setComplexClass(ctor: new (re: number, im: number) => unknown): void;
  // M2 creation
  full(shape: number[], value: NativeNDArray): NativeNDArray;
  ones(shape: number[], dtype: string): NativeNDArray;
  arange(start: number, stop: number, step: number, dtype: string): NativeNDArray;
  linspace(start: number, stop: number, num: number, endpoint: boolean, dtype: string): NativeNDArray;
  eye(n: number, m: number, k: number, dtype: string): NativeNDArray;
  // M3 shape
  transpose(a: NativeNDArray, axes: number[]): NativeNDArray;
  squeeze(a: NativeNDArray, axes: number[] | undefined): NativeNDArray;
  expandDims(a: NativeNDArray, axes: number[]): NativeNDArray;
  swapaxes(a: NativeNDArray, axis1: number, axis2: number): NativeNDArray;
  moveaxis(a: NativeNDArray, source: number[], destination: number[]): NativeNDArray;
  ravel(a: NativeNDArray): NativeNDArray;
  flatten(a: NativeNDArray): NativeNDArray;
  // M4/M5 ufuncs and broadcasting
  broadcastShapes(shapes: number[][]): number[];
  broadcastTo(a: NativeNDArray, shape: number[]): NativeNDArray;
  binary(op: string, a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  unary(op: string, a: NativeNDArray): NativeNDArray;
  // P1 complex helpers (D-033)
  complexPart(a: NativeNDArray, imag: boolean): NativeNDArray;
  isComplexElementwise(a: NativeNDArray, wantComplex: boolean): NativeNDArray;
  // M6 indexing (encoding: see ndarray.ts encodeIndex)
  getIndex(a: NativeNDArray, index: NativeIndexItem[]): NativeNDArray;
  setIndex(a: NativeNDArray, index: NativeIndexItem[], value: NativeNDArray): void;
  nonzero(a: NativeNDArray): NativeNDArray[];
  take(a: NativeNDArray, indices: NativeNDArray, axis: number | null): NativeNDArray;
  where(cond: NativeNDArray, x: NativeNDArray, y: NativeNDArray): NativeNDArray;
  // M7 reductions (D-017)
  reduce(
    op: string,
    a: NativeNDArray,
    opts: {
      axis: number[] | null;
      keepdims: boolean;
      dtype: string | null;
      initial: number | null;
      ddof: number;
    },
  ): NativeNDArray;
  argReduce(isMax: boolean, a: NativeNDArray, axis: number | null, keepdims: boolean): NativeNDArray;
  // M8 linear algebra (D-018)
  linalg: NativeLinalg;
  // M9 random (D-019)
  random: { BitGenerator: NativeBitGeneratorCtor };
  // M10 FFT (D-020)
  fft: NativeFft;
}

export type FftNorm = "backward" | "ortho" | "forward";
export interface NativeFft {
  fft(a: NativeNDArray, n: number | null, axis: number, norm: string | null): NativeNDArray;
  ifft(a: NativeNDArray, n: number | null, axis: number, norm: string | null): NativeNDArray;
  rfft(a: NativeNDArray, n: number | null, axis: number, norm: string | null): NativeNDArray;
  irfft(a: NativeNDArray, n: number | null, axis: number, norm: string | null): NativeNDArray;
  fftn(a: NativeNDArray, s: number[] | null, axes: number[] | null, norm: string | null): NativeNDArray;
  ifftn(a: NativeNDArray, s: number[] | null, axes: number[] | null, norm: string | null): NativeNDArray;
  fftfreq(n: number, d: number): NativeNDArray;
  rfftfreq(n: number, d: number): NativeNDArray;
}

type N = NativeNDArray;
export type SeedMode = "seedseq" | "int" | "array";
export interface NativeBitGenerator {
  reseed(mode: SeedMode, words: number[]): void;
  random(size: number[], dtype: string): N;
  uniform(low: number, high: number, size: number[]): N;
  normal(loc: number, scale: number, size: number[], dtype: string): N;
  legacyNormal(loc: number, scale: number, size: number[]): N;
  integers(
    low: number | bigint,
    high: number | bigint,
    closed: boolean,
    size: number[],
    dtype: string,
    masked: boolean,
  ): N;
  shuffle(a: N): void;
  choiceIndices(popSize: number, size: number[], replace: boolean, shuffle: boolean): N;
}
export type NativeBitGeneratorCtor = new (
  kind: "pcg64" | "mt19937",
  mode: SeedMode,
  words: number[],
) => NativeBitGenerator;
export interface NativeLinalg {
  matmul(a: N, b: N): N;
  dot(a: N, b: N): N;
  inner(a: N, b: N): N;
  outer(a: N, b: N): N;
  solve(a: N, b: N): N;
  det(a: N): N;
  inv(a: N): N;
  eig(a: N): { eigenvalues: N; eigenvectors: N };
  eigh(a: N): { eigenvalues: N; eigenvectors: N };
  eigvalsh(a: N): N;
  svd(a: N, fullMatrices: boolean, computeUV: boolean): { U: N | null; S: N; Vh: N | null };
  qr(a: N, mode: string): { Q: N | null; R: N };
  lstsq(a: N, b: N, rcond: number | null): { x: N; residuals: N; rank: number; s: N };
  norm(a: N, ord: string | number | null, axis: number[] | null, keepdims: boolean): N;
  backend(): string;
  _setBackend(which: "fallback" | "default"): void;
}

/**
 * Wire format of one index item: number = integer, boolean = 0-d bool index,
 * null = newaxis, "..." = ellipsis, native array = advanced index,
 * object = slice.
 */
export type NativeIndexItem =
  | number
  | boolean
  | null
  | "..."
  | NativeNDArray
  | { start?: number | null; stop?: number | null; step?: number | null };

// Resolution order (DECISIONS D-007, D-026): NATIVPY_ADDON_PATH, then the
// prebuild bundled in the npm package, then the repo build.
function candidatePaths(): string[] {
  const here = dirname(fileURLToPath(import.meta.url));
  const paths: string[] = [];
  const env = process.env["NATIVPY_ADDON_PATH"];
  if (env) paths.push(resolve(env));
  // dist/ -> package root -> prebuilds/<platform>-<arch>/
  paths.push(
    resolve(here, "../prebuilds", `${process.platform}-${process.arch}`, "nativpy.node"),
  );
  // src/ or dist/ -> packages/nativpy -> repo root
  paths.push(resolve(here, "../../../build/Release/nativpy.node"));
  paths.push(resolve(here, "../../../build/Debug/nativpy.node"));
  return paths;
}

function loadAddon(): NativeAddon {
  const require = createRequire(import.meta.url);
  const tried = candidatePaths();
  for (const p of tried) {
    if (existsSync(p)) return require(p) as NativeAddon;
  }
  throw new Error(
    `numera: no prebuilt native addon for ${process.platform}-${process.arch}. ` +
      "Prebuilds ship for darwin-arm64, darwin-x64, linux-x64 and linux-arm64 (glibc >= 2.28); " +
      "elsewhere set NATIVPY_ADDON_PATH to a nativpy.node built for this platform. Tried:\n  " +
      tried.join("\n  "),
  );
}

export const addon: NativeAddon = loadAddon();
