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

/** np.seterr modes (D-054). */
export interface NativeErrState {
  divide?: string;
  over?: string;
  under?: string;
  invalid?: string;
}

/** Options for ufunc.reduce / ufunc.accumulate (D-052). `axis`: omitted = default, null = all. */
export interface NativeUfuncMethodOptions {
  axis?: number[] | null;
  dtype?: string;
  keepdims?: boolean;
  initial?: number;
  where?: NativeNDArray;
}

/** Ufunc `dtype=` / `casting=` (D-048) and `where=` mask (D-049). */
export interface NativeUfuncParams {
  dtype?: string;
  casting?: string;
  where?: NativeNDArray;
  /** D-050; validated natively. */
  order?: unknown;
}


export interface NativeAddon {
  napiVersion: number;
  dtypes: Record<string, { itemsize: number; alignment: number; kind: string }>;
  promoteTypes(a: string, b: string): string;
  canCast(from: string, to: string, casting: string): boolean;
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
  ravel(a: NativeNDArray, order?: string): NativeNDArray;
  flatten(a: NativeNDArray, order?: string): NativeNDArray;
  // P3-1 memory order (D-055)
  emptyOrder(shape: number[], dtype: string, order: string | undefined, zeroed: boolean): NativeNDArray;
  copyOrder(a: NativeNDArray, dtype: string | null, order: string | undefined): NativeNDArray;
  reshapeOrder(a: NativeNDArray, shape: number[], order: string | undefined): NativeNDArray;
  // M4/M5 ufuncs and broadcasting
  broadcastShapes(shapes: number[][]): number[];
  broadcastTo(a: NativeNDArray, shape: number[]): NativeNDArray;
  binary(op: string, a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  /** Writes into `out` (D-046); returns undefined. `params`: D-048. */
  binary(op: string, a: NativeNDArray, b: NativeNDArray, out: NativeNDArray, params?: NativeUfuncParams): undefined;
  binary(op: string, a: NativeNDArray, b: NativeNDArray, out: null, params: NativeUfuncParams): NativeNDArray;
  unary(op: string, a: NativeNDArray): NativeNDArray;
  unary(op: string, a: NativeNDArray, out: NativeNDArray, params?: NativeUfuncParams): undefined;
  unary(op: string, a: NativeNDArray, out: null, params: NativeUfuncParams): NativeNDArray;
  /** ufunc.reduce / ufunc.accumulate (D-052). Returns undefined when `out` is given. */
  ufuncMethod(
    method: "reduce" | "accumulate" | "reduceat" | "outer",
    op: string,
    a: NativeNDArray,
    out: NativeNDArray | null,
    opts: NativeUfuncMethodOptions | NativeUfuncParams,
    b?: NativeNDArray,
  ): NativeNDArray | undefined;
  /** np.geterr / np.seterr state (D-054). */
  getErr(): Required<NativeErrState>;
  setErr(state: NativeErrState): undefined;
  /** ufunc.at (P2-9): in place on `a`. */
  ufuncAt(op: string, a: NativeNDArray, indices: NativeNDArray[], b: NativeNDArray | null): undefined;
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
  // P13-3 (D-161): Generator discrete distributions
  binomial(n: number, p: number, size: number[]): N;
  negativeBinomial(n: number, p: number, size: number[]): N;
  poisson(lam: number, size: number[]): N;
  zipf(a: number, size: number[]): N;
  geometric(p: number, size: number[]): N;
  hypergeometric(good: number, bad: number, sample: number, size: number[]): N;
  logseries(p: number, size: number[]): N;
  // P13-2 (D-161): Generator continuous distributions
  standardExponential(size: number[]): N;
  exponential(scale: number, size: number[]): N;
  standardGamma(shape: number, size: number[]): N;
  gamma(shape: number, scale: number, size: number[]): N;
  beta(a: number, b: number, size: number[]): N;
  chisquare(df: number, size: number[]): N;
  f(dfnum: number, dfden: number, size: number[]): N;
  standardCauchy(size: number[]): N;
  pareto(a: number, size: number[]): N;
  weibull(a: number, size: number[]): N;
  power(a: number, size: number[]): N;
  laplace(loc: number, scale: number, size: number[]): N;
  gumbel(loc: number, scale: number, size: number[]): N;
  logistic(loc: number, scale: number, size: number[]): N;
  lognormal(mean: number, sigma: number, size: number[]): N;
  rayleigh(scale: number, size: number[]): N;
  standardT(df: number, size: number[]): N;
  noncentralChisquare(df: number, nonc: number, size: number[]): N;
  noncentralF(dfnum: number, dfden: number, nonc: number, size: number[]): N;
  wald(mean: number, scale: number, size: number[]): N;
  vonmises(mu: number, kappa: number, size: number[]): N;
  triangular(left: number, mode: number, right: number, size: number[]): N;
  // P13-4 (D-162): multivariate distributions
  multinomial(n: number, pvals: number[], size: number[]): N;
  dirichlet(alpha: number[], size: number[]): N;
  mvhgCount(colors: number[], nsample: number, size: number[]): N;
  mvhgMarginals(colors: number[], nsample: number, size: number[]): N;
  choiceP(popSize: number, size: number[], cdf: number[]): N;
  permuted(x: N, axis: number): N;
  // P13-5 (D-163): RandomState / legacy distributions
  getState(): bigint[];
  setState(words: bigint[]): void;
  bytes(length: number): Uint8Array;
  legacyStandardExponential(size: number[]): N;
  legacyExponential(scale: number, size: number[]): N;
  legacyStandardGamma(shape: number, size: number[]): N;
  legacyGamma(shape: number, scale: number, size: number[]): N;
  legacyBeta(a: number, b: number, size: number[]): N;
  legacyChisquare(df: number, size: number[]): N;
  legacyF(dfnum: number, dfden: number, size: number[]): N;
  legacyNoncentralChisquare(df: number, nonc: number, size: number[]): N;
  legacyNoncentralF(dfnum: number, dfden: number, nonc: number, size: number[]): N;
  legacyStandardCauchy(size: number[]): N;
  legacyStandardT(df: number, size: number[]): N;
  legacyPareto(a: number, size: number[]): N;
  legacyWeibull(a: number, size: number[]): N;
  legacyPower(a: number, size: number[]): N;
  legacyLognormal(mean: number, sigma: number, size: number[]): N;
  legacyRayleigh(mode: number, size: number[]): N;
  legacyWald(mean: number, scale: number, size: number[]): N;
  legacyVonmises(mu: number, kappa: number, size: number[]): N;
  legacyNegativeBinomial(n: number, p: number, size: number[]): N;
  legacyBinomial(n: number, p: number, size: number[]): N;
  legacyHypergeometric(good: number, bad: number, sample: number, size: number[]): N;
  legacyZipf(a: number, size: number[]): N;
  legacyGeometric(p: number, size: number[]): N;
  legacyLogseries(p: number, size: number[]): N;
  legacyChoiceP(cdf: N, size: number[]): N;
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
  eigvals(a: N): N;
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

/**
 * Per-milestone native function table (D-056): `addon.p03` ... `addon.p15`,
 * set by native/bindings/pNN_binding.cpp. Each milestone's TS file declares
 * the shape of its own table, e.g. `const native = nativeModule<P04Native>("p04")`.
 */
export function nativeModule<T>(name: string): T {
  const m = (addon as unknown as Record<string, unknown>)[name];
  if (m === undefined) throw new Error(`numera: native module '${name}' missing from the addon`);
  return m as T;
}
