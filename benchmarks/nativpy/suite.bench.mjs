// PLAN §38–§43 benchmark suite for nativpy. NumPy counterpart:
// benchmarks/numpy/suite_bench.py (same case names, same inputs, same method).
//
// Run: pnpm build && pnpm bench:suite   then   pnpm bench:compare
// Env: NATIVPY_BENCH_LARGE=1 adds the 10M / 2048x2048 sizes (slow, memory heavy).
//      NATIVPY_BENCH_FILTER=<substring> runs only matching case names.
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import np from "../../packages/numera/dist/index.js";

const LARGE = process.env.NATIVPY_BENCH_LARGE === "1";
const FILTER = process.env.NATIVPY_BENCH_FILTER ?? "";
const VEC = LARGE ? [1_000, 100_000, 1_000_000, 10_000_000] : [1_000, 100_000, 1_000_000];
const MAT = LARGE ? [32, 128, 512, 1024, 2048] : [32, 128, 512, 1024];
const DECOMP = LARGE ? [32, 128, 512, 1024] : [32, 128, 512];
const REPEATS = 7;
const MIN_SAMPLE_MS = 5; // each sample runs the op enough times to last >= this

/**
 * Median over REPEATS samples of the per-call time. Each sample repeats the
 * call `inner` times, calibrated once so that a sample lasts >= MIN_SAMPLE_MS
 * (the same method as suite_bench.py). GC runs between samples, not inside.
 */
function bench(fn) {
  fn(); // warm-up (JIT, first-touch of inputs)
  let inner = 1;
  for (;;) {
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < inner; i++) fn();
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    if (ms >= MIN_SAMPLE_MS || inner >= 1 << 20) break;
    inner *= ms <= 0 ? 16 : Math.min(16, Math.max(2, Math.ceil(MIN_SAMPLE_MS / ms)));
  }
  const times = [];
  for (let r = 0; r < REPEATS; r++) {
    globalThis.gc?.();
    const t0 = process.hrtime.bigint();
    for (let i = 0; i < inner; i++) fn();
    times.push(Number(process.hrtime.bigint() - t0) / 1e6 / inner);
  }
  times.sort((a, b) => a - b);
  return { median_ms: times[Math.floor(REPEATS / 2)], min_ms: times[0], inner };
}

const results = [];
function run(category, name, shape, fn) {
  const full = `${category}/${name}`;
  if (FILTER && !full.includes(FILTER)) return;
  const r = bench(fn);
  results.push({ category, name, shape, ...r });
  console.log(`${full.padEnd(44)} ${JSON.stringify(shape).padEnd(14)} ${r.median_ms.toFixed(4).padStart(11)} ms`);
}
function notImplemented(category, name, shape) {
  const full = `${category}/${name}`;
  if (FILTER && !full.includes(FILTER)) return;
  results.push({ category, name, shape, not_implemented: true });
  console.log(`${full.padEnd(44)} ${JSON.stringify(shape).padEnd(14)}   not implemented`);
}

const rng = np.random.defaultRng(12345);

// ---- vectors: creation, memory, elementwise, reductions, views, fft, random ----
for (const n of VEC) {
  const s = [n];
  const a = rng.random([n]);
  const b = rng.random([n]);
  const a32 = a.astype("float32");
  const b32 = b.astype("float32");
  const ai = rng.integers(0, 1000, [n], "int32");
  const idx = rng.integers(0, n, [n]);
  const list = n <= 1_000_000 ? Array.from(a.toTypedArray()) : null;
  const typed = a.toTypedArray();

  run("creation", "zeros", s, () => np.zeros([n]));
  run("creation", "empty", s, () => np.empty([n]));
  run("creation", "ones", s, () => np.ones([n]));
  run("creation", "arange", s, () => np.arange(n));
  if (list) run("creation", "array(list)", s, () => np.array(list));
  run("memory", "fromTypedArray", s, () => np.fromTypedArray(typed));
  run("memory", "copy", s, () => a.copy());
  run("memory", "toTypedArray", s, () => a.toTypedArray());
  run("memory", "astype(float32)", s, () => a.astype("float32"));

  run("elementwise", "add f64", s, () => np.add(a, b));
  run("elementwise", "add f32", s, () => np.add(a32, b32));
  run("elementwise", "add i32", s, () => np.add(ai, ai));
  run("elementwise", "multiply f64", s, () => np.multiply(a, b));
  run("elementwise", "multiply scalar f64", s, () => np.multiply(a, 2.5));
  run("elementwise", "divide f64", s, () => np.divide(a, b));
  run("elementwise", "sqrt f64", s, () => np.sqrt(a));
  run("elementwise", "exp f64", s, () => np.exp(a));
  run("elementwise", "log f64", s, () => np.log(a));
  run("elementwise", "abs f64", s, () => np.abs(a));

  // P4 math ufuncs (V-P4)
  run("math", "sin f64", s, () => np.sin(a));
  run("math", "cos f64", s, () => np.cos(a));
  run("math", "tan f64", s, () => np.tan(a));
  run("math", "arcsin f64", s, () => np.arcsin(a));
  run("math", "arccos f64", s, () => np.arccos(a));
  run("math", "arctan f64", s, () => np.arctan(a));
  run("math", "arctan2 f64", s, () => np.arctan2(a, b));
  run("math", "sinh f64", s, () => np.sinh(a));
  run("math", "cosh f64", s, () => np.cosh(a));
  run("math", "tanh f64", s, () => np.tanh(a));
  run("math", "arcsinh f64", s, () => np.arcsinh(a));
  run("math", "arccosh f64", s, () => np.arccosh(a));
  run("math", "arctanh f64", s, () => np.arctanh(a));
  run("math", "hypot f64", s, () => np.hypot(a, b));
  run("math", "deg2rad f64", s, () => np.deg2rad(a));
  run("math", "rad2deg f64", s, () => np.rad2deg(a));
  run("math", "degrees f64", s, () => np.degrees(a));
  run("math", "radians f64", s, () => np.radians(a));
  run("math", "exp2 f64", s, () => np.exp2(a));
  run("math", "expm1 f64", s, () => np.expm1(a));
  run("math", "log2 f64", s, () => np.log2(a));
  run("math", "log10 f64", s, () => np.log10(a));
  run("math", "log1p f64", s, () => np.log1p(a));
  run("math", "logaddexp f64", s, () => np.logaddexp(a, b));
  run("math", "logaddexp2 f64", s, () => np.logaddexp2(a, b));
  run("math", "cbrt f64", s, () => np.cbrt(a));
  run("math", "square f64", s, () => np.square(a));
  run("math", "reciprocal f64", s, () => np.reciprocal(a));
  run("math", "floor f64", s, () => np.floor(a));
  run("math", "ceil f64", s, () => np.ceil(a));
  run("math", "trunc f64", s, () => np.trunc(a));
  run("math", "rint f64", s, () => np.rint(a));
  run("math", "positive f64", s, () => np.positive(a));
  run("math", "fix f64", s, () => np.fix(a));
  run("math", "round f64", s, () => np.round(a));
  run("math", "clip f64", s, () => np.clip(a, 0.1, 0.9));
  run("math", "fmod f64", s, () => np.fmod(a, b));
  run("math", "floatPower f64", s, () => np.floatPower(a, b));
  run("math", "sign f64", s, () => np.sign(a));
  run("math", "heaviside f64", s, () => np.heaviside(a, 0.5));
  run("math", "maximum f64", s, () => np.maximum(a, b));
  run("math", "minimum f64", s, () => np.minimum(a, b));
  run("math", "fmax f64", s, () => np.fmax(a, b));
  run("math", "fmin f64", s, () => np.fmin(a, b));
  run("math", "fabs f64", s, () => np.fabs(a));
  run("math", "gcd i32", s, () => np.gcd(ai, ai));
  run("math", "lcm i32", s, () => np.lcm(ai, ai));
  run("math", "copysign f64", s, () => np.copysign(a, b));
  run("math", "nextafter f64", s, () => np.nextafter(a, b));
  run("math", "spacing f64", s, () => np.spacing(a));
  run("math", "ldexp f64", s, () => np.ldexp(a, ai));
  run("math", "signbit f64", s, () => np.signbit(a));
  run("math", "divmod f64", s, () => np.divmod(a, b));
  run("math", "modf f64", s, () => np.modf(a));
  run("math", "frexp f64", s, () => np.frexp(a));
  run("math", "sinc f64", s, () => np.sinc(a));
  run("math", "i0 f64", s, () => np.i0(a));
  run("math", "unwrap f64", s, () => np.unwrap(a));
  run("math", "nanToNum f64", s, () => np.nanToNum(a));
  run("math", "realIfClose f64", s, () => np.realIfClose(a));
  run("math", "ndarray.clip f64", s, () => a.clip(0.1, 0.9));
  run("math", "ndarray.round f64", s, () => a.round());
  run("math", "ndarray.conjugate f64", s, () => a.conjugate());

  run("reduction", "sum f64", s, () => np.sum(a));
  run("reduction", "sum f32", s, () => np.sum(a32));
  run("reduction", "sum i32", s, () => np.sum(ai));
  run("reduction", "mean f64", s, () => np.mean(a));
  run("reduction", "max f64", s, () => np.max(a));
  run("reduction", "argmax f64", s, () => np.argmax(a));
  run("reduction", "std f64", s, () => np.std(a));

  run("view", "slice [::2]", s, () => a.slice([null, null, 2]));
  run("view", "reshape", s, () => a.reshape([n / 1000, 1000]));
  run("slicing", "take(idx)", s, () => np.take(a, idx));

  notImplemented("sorting", "sort f64", s);

  run("fft", "fft f64", s, () => np.fft.fft(a));
  run("fft", "rfft f64", s, () => np.fft.rfft(a));
  // P1-5e complex FFT input: c = a + i*b
  const c = np.add(a, np.multiply(b, np.complex(0, 1)));
  const c64 = c.astype("complex64");
  run("fft", "fft c128", s, () => np.fft.fft(c));
  run("fft", "fft c64", s, () => np.fft.fft(c64));
  run("fft", "ifft c128", s, () => np.fft.ifft(c));

  run("random", "rng.random", s, () => rng.random([n]));
  run("random", "rng.standardNormal", s, () => rng.standardNormal([n]));
  run("random", "rng.integers", s, () => rng.integers(0, 1000, [n]));

run("random2", "rng.normal f64", s, () => np.random.defaultRng(0).normal(0, 1, [n]));
  run("random2", "rng.uniform f64", s, () => np.random.defaultRng(0).uniform(0, 1, [n]));
  run("random2", "rng.exponential f64", s, () => np.random.defaultRng(0).exponential(1, [n]));
  run("random2", "rng.gamma f64", s, () => np.random.defaultRng(0).gamma(2, 1, [n]));
  run("random2", "rng.beta f64", s, () => np.random.defaultRng(0).beta(2, 5, [n]));
  run("random2", "rng.chisquare f64", s, () => np.random.defaultRng(0).chisquare(3, [n]));
  run("random2", "rng.f f64", s, () => np.random.defaultRng(0).f(5, 2, [n]));
  run("random2", "rng.hypergeometric i64", s, () => np.random.defaultRng(0).hypergeometric(20, 10, 5, [n]));
  run("random2", "rng.weibull f64", s, () => np.random.defaultRng(0).weibull(1.5, [n]));
  run("random2", "rng.pareto f64", s, () => np.random.defaultRng(0).pareto(3, [n]));
  run("random2", "rng.lognormal f64", s, () => np.random.defaultRng(0).lognormal(0, 1, [n]));
  run("random2", "rng.laplace f64", s, () => np.random.defaultRng(0).laplace(0, 1, [n]));
  run("random2", "rng.gumbel f64", s, () => np.random.defaultRng(0).gumbel(0, 1, [n]));
  run("random2", "rng.logistic f64", s, () => np.random.defaultRng(0).logistic(0, 1, [n]));
  run("random2", "rng.rayleigh f64", s, () => np.random.defaultRng(0).rayleigh(1, [n]));
  run("random2", "rng.power f64", s, () => np.random.defaultRng(0).power(2, [n]));
  run("random2", "rng.standardNormal f64", s, () => np.random.defaultRng(0).standardNormal([n]));
  run("random2", "rng.standardExponential f64", s, () => np.random.defaultRng(0).standardExponential([n]));
  run("random2", "rng.standardGamma f64", s, () => np.random.defaultRng(0).standardGamma(2, [n]));
  run("random2", "rng.standardT f64", s, () => np.random.defaultRng(0).standardT(5, [n]));
  run("random2", "rng.standardCauchy f64", s, () => np.random.defaultRng(0).standardCauchy([n]));
  run("random2", "rng.vonmises f64", s, () => np.random.defaultRng(0).vonmises(0, 1, [n]));
  run("random2", "rng.wald f64", s, () => np.random.defaultRng(0).wald(1, 1, [n]));
  run("random2", "rng.triangular f64", s, () => np.random.defaultRng(0).triangular(0, 0.5, 1, [n]));
  run("random2", "rng.poisson i64", s, () => np.random.defaultRng(0).poisson(3, [n]));
  run("random2", "rng.binomial i64", s, () => np.random.defaultRng(0).binomial(10, 0.5, [n]));
  run("random2", "rng.geometric i64", s, () => np.random.defaultRng(0).geometric(0.5, [n]));
  run("random2", "rng.negativeBinomial i64", s, () => np.random.defaultRng(0).negativeBinomial(5, 0.5, [n]));
  run("random2", "rng.logseries i64", s, () => np.random.defaultRng(0).logseries(0.9, [n]));
  run("random2", "rng.zipf i64", s, () => np.random.defaultRng(0).zipf(2, [n]));
  run("random2", "rng.choice i64", s, () => np.random.defaultRng(0).choice(n, [n]));
  run("random2", "rng.permutation i64", s, () => np.random.defaultRng(0).permutation(n));
run("p16a", "cumsum i32", s, () => np.cumsum(ai));
  run("p16a", "cumsum f64", s, () => np.cumsum(a));
  run("p16a", "cumprod f64", s, () => np.cumprod(a));
  run("p16a", "ndarray.cumsum f64", s, () => a.cumsum());
  run("p16a", "ndarray.cumprod f64", s, () => a.cumprod());
  run("p16a", "shares_memory", [1], () => np.sharesMemory(a, b));

run("linalg2", "vdot f64", [1], () => np.vdot(a, b));
  run("linalg2", "kron f64", [n * n], () => np.kron(a, b));
  run("linalg2", "vector_norm f64", [1], () => np.linalg.vectorNorm(a));
  run("linalg2", "cross f64", [3], () => np.cross(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])));
  run("linalg2", "outer f64", [n, n], () => np.linalg.outer(a, b));
run("hfft",     "hfft f64",     [n],    () => np.fft.hfft(a));
  run("ihfft",    "ihfft f64",    [n],    () => np.fft.ihfft(a));
  run("fftshift", "fftshift f64", [n],    () => np.fft.fftshift(a));
  run("ifftshift","ifftshift f64",[n],    () => np.fft.ifftshift(a));
run("io", "bartlett f64", [n], () => np.bartlett(n));
  run("io", "blackman f64", [n], () => np.blackman(n));
  run("io", "hamming f64", [n], () => np.hamming(n));
  run("io", "hanning f64", [n], () => np.hanning(n));
  run("io", "kaiser f64", [n], () => np.kaiser(n, 8.6));
  run("io", "polyval f64", [n], () => np.polyval(np.array([1.0, 2.0, 3.0]), a));
  run("io", "polymul f64", [4], () => np.polymul(np.array([1.0, 2.0]), np.array([1.0, 3.0])));
  run("io", "polyadd f64", [3], () => np.polyadd(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])));
  run("io", "polysub f64", [3], () => np.polysub(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0])));
  run("io", "polyder f64", [2], () => np.polyder(np.array([1.0, 0.0, 0.0])));
  run("io", "polyint f64", [3], () => np.polyint(np.array([2.0, 0.0])));
  run("io", "polydiv f64", [2], () => np.polydiv(np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0])));
  run("io", "poly f64", [3], () => np.poly(np.array([1.0, 2.0, 3.0])));
  run("io", "roots f64", [2], () => np.roots(np.array([1.0, -6.0, 11.0, -6.0])));
  run("io", "base_repr", [1], () => np.baseRepr(255, 16));
  run("io", "binary_repr", [1], () => np.binaryRepr(10));
run("emath", "emath.sqrt f64", s, () => np.emath.sqrt(a));
  run("emath", "emath.log f64", s, () => np.emath.log(a));
  run("emath", "emath.log2 f64", s, () => np.emath.log2(a));
  run("emath", "emath.log10 f64", s, () => np.emath.log10(a));
  run("emath", "emath.arcsin f64", s, () => np.emath.arcsin(a));
  run("emath", "emath.arccos f64", s, () => np.emath.arccos(a));
  run("emath", "emath.arctanh f64", s, () => np.emath.arctanh(a));
  run("emath", "emath.power f64", s, () => np.emath.power(a, 2));
  run("emath", "emath.logn f64", s, () => np.emath.logn(10, a));

run("sorting", "sort f64", s, () => np.sort(a));
  run("sorting", "sort i32", s, () => np.sort(ai));
  run("sorting", "argsort f64", s, () => np.argsort(a));
  run("sorting", "partition f64", s, () => np.partition(a, Math.floor(n/2)));
  run("sorting", "searchsorted f64", s, () => np.searchsorted(a, 0.5));
  run("sorting", "unique f64", s, () => np.unique(a));
  run("sorting", "isin f64", s, () => np.isin(a, b));
  run("sorting", "lexsort f64", s, () => np.lexsort([a, b]));
  run("sorting", "ediff1d i32", s, () => np.ediff1d(ai));

run("logic", "equal f64", s, () => np.equal(a, b));
  run("logic", "notEqual f64", s, () => np.notEqual(a, b));
  run("logic", "less f64", s, () => np.less(a, b));
  run("logic", "lessEqual f64", s, () => np.lessEqual(a, b));
  run("logic", "greater f64", s, () => np.greater(a, b));
  run("logic", "greaterEqual f64", s, () => np.greaterEqual(a, b));
  run("logic", "logicalAnd f64", s, () => np.logicalAnd(a, b));
  run("logic", "logicalOr f64", s, () => np.logicalOr(a, b));
  run("logic", "logicalXor f64", s, () => np.logicalXor(a, b));
  run("logic", "logicalNot f64", s, () => np.logicalNot(a));
  run("logic", "isnan f64", s, () => np.isnan(a));
  run("logic", "isinf f64", s, () => np.isinf(a));
  run("logic", "isfinite f64", s, () => np.isfinite(a));
  run("logic", "isposinf f64", s, () => np.isposinf(a));
  run("logic", "isneginf f64", s, () => np.isneginf(a));
  run("logic", "all f64", s, () => np.all(a));
  run("logic", "any f64", s, () => np.any(a));
  run("logic", "isclose f64", s, () => np.isclose(a, b));
  run("logic", "bitwiseAnd i32", s, () => np.bitwiseAnd(ai, ai));
  run("logic", "bitwiseOr i32", s, () => np.bitwiseOr(ai, ai));
  run("logic", "bitwiseXor i32", s, () => np.bitwiseXor(ai, ai));
  run("logic", "invert i32", s, () => np.invert(ai));
  run("logic", "leftShift i32", s, () => np.leftShift(ai, ai));
  run("logic", "rightShift i32", s, () => np.rightShift(ai, ai));
  run("logic", "bitwiseCount i32", s, () => np.bitwiseCount(ai));
  run("logic", "packbits u8", [Math.floor(n/8)], () => np.packbits(np.array(new Uint8Array(Math.floor(n/8)))));
  run("logic", "unpackbits u8", [Math.floor(n/8)], () => np.unpackbits(np.array(new Uint8Array(Math.floor(n/8)))));
run("manip", "concatenate f64", [n * 2], () => np.concatenate([a, b]));
  run("manip", "stack f64", [2, n], () => np.stack([a, b]));
  run("manip", "repeat f64", [n * 2], () => np.repeat(a, 2));
  run("manip", "flip f64", s, () => np.flip(a));
  run("manip", "roll f64", s, () => np.roll(a, 10));
  run("manip", "tile f64", [n * 2], () => np.tile(a, 2));
  run("manip", "split f64", s, () => np.split(a, 2));
run("creation2", "logspace f64", [50], () => np.logspace(0, 3));
  run("creation2", "geomspace f64", [50], () => np.geomspace(1, 1000));
  run("creation2", "astype f64->f32", s, () => np.astype(a, "float32"));
  run("creation2", "tri f64", s, () => np.tri(m));
  run("creation2", "tril f64", s, () => np.tril(A));
  run("creation2", "triu f64", s, () => np.triu(A));
  run("creation2", "diag extract", [m], () => np.diag(A));
  run("creation2", "diag construct", s, () => np.diag(np.arange(m)));
  run("creation2", "vander f64", s, () => np.vander(np.arange(m)));
  run("creation2", "meshgrid f64", [2, m], () => np.meshgrid(np.arange(m), np.arange(m)));
run("indexing2", "take f64", s, () => np.take(a, idx));
  run("indexing2", "compress f64", s, () => np.compress(np.greater(a, 0.5), a));
  run("indexing2", "argwhere f64", s, () => np.argwhere(np.greater(a, 0.5)));
  run("indexing2", "flatnonzero f64", s, () => np.flatnonzero(a));
  run("indexing2", "countNonzero f64", [], () => np.countNonzero(a));
  run("indexing2", "ravelMultiIndex i32", s, () => np.ravelMultiIndex([idx, idx], [n, n]));
run("stats", "median f64", s, () => np.median(a));
  run("stats", "quantile f64", s, () => np.quantile(a, 0.5));
  run("stats", "nansum f64", s, () => np.nansum(a));
  run("stats", "nanmean f64", s, () => np.nanmean(a));
  run("stats", "cumsum f64", s, () => np.cumsum(a));
  run("stats", "cumprod f64", s, () => np.cumprod(a));
  run("stats", "diff f64", [n - 1], () => np.diff(a));
  run("stats", "histogram f64", [10], () => np.histogram(a));
  run("stats", "bincount i32", [1001], () => np.bincount(np.abs(ai)));
  run("stats", "digitize f64", s, () => np.digitize(a, np.linspace(0, 1, 100)));
  run("stats", "interp f64", s, () => np.interp(a, np.linspace(0, 1, 100), np.linspace(0, 1, 100)));
  run("stats", "correlate f64", [2 * 1000 - 1], () => np.correlate(a.slice([[0, 1000]]), a.slice([[0, 1000]]), "full"));
  run("stats", "trapezoid f64", [], () => np.trapezoid(a));
  run("stats", "cov f64", [m, m], () => np.cov(A));
  run("stats", "corrcoef f64", [m, m], () => np.corrcoef(A));
  run("stats", "gradient f64", [m], () => np.gradient(A.slice([0])));
}

// ---- matrices: broadcasting, transpose, axis reductions, matmul, fft2 ----
for (const m of MAT) {
  const s = [m, m];
  const A = rng.random([m, m]);
  const B = rng.random([m, m]);
  const A32 = A.astype("float32");
  const B32 = B.astype("float32");
  const row = rng.random([m]);
  const col = rng.random([m, 1]);

  run("broadcast", "add(matrix, row)", s, () => np.add(A, row));
  run("broadcast", "add(matrix, col)", s, () => np.add(A, col));
  run("transpose", "A.T (view)", s, () => A.T);
  run("transpose", "A.T.copy()", s, () => A.T.copy());
  run("transpose", "add(A.T, B)", s, () => np.add(A.T, B));
  run("reduction", "sum axis=0", s, () => np.sum(A, { axis: 0 }));
  run("reduction", "sum axis=1", s, () => np.sum(A, { axis: 1 }));
  run("slicing", "A[1:-1, ::2]", s, () => A.slice([[1, -1], [null, null, 2]]));
  run("matmul", "matmul f64", s, () => np.matmul(A, B));
  run("matmul", "matmul f32", s, () => np.matmul(A32, B32));
  if (m <= 512) {
    const Ai = A.astype("int32");
    run("matmul", "matmul i32", s, () => np.matmul(Ai, Ai));
  }
  // P1-4 complex matmul (D-035/D-036): C = A + i*B
  const C = np.add(A, np.multiply(B, np.complex(0, 1)));
  const C64 = C.astype("complex64");
  run("matmul", "matmul c128", s, () => np.matmul(C, C));
  run("matmul", "matmul c64", s, () => np.matmul(C64, C64));
  run("fft", "fft2 f64", s, () => np.fft.fft2(A));
}

// ---- decompositions (well-conditioned inputs) ----
for (const m of DECOMP) {
  const s = [m, m];
  const A = np.add(rng.random([m, m]), np.multiply(np.eye(m), m));
  const S = np.add(A, A.T);
  const b = rng.random([m]);
— paste inside `for (const m of MAT) { ... }`
  run("manip", "flip2d f64", s, () => np.flip(A));
  run("manip", "rot90 f64", s, () => np.rot90(A));
  run("manip", "pad f64", [m + 4, m + 4], () => np.pad(A, 2));
  run("manip", "vstack f64", [m * 2, m], () => np.vstack([A, B]));
  run("manip", "hstack f64", [m, m * 2], () => np.hstack([A, B]));
  run("manip", "concatenate2d f64", [m * 2, m], () => np.concatenate([A, B], { axis: 0 }));
run("indexing2", "diagonal2d f64", [m], () => np.diagonal(A));
  run("indexing2", "trace f64", [], () => np.trace(A));
  run("indexing2", "takeAlongAxis f64", s, () => np.takeAlongAxis(A, np.argmax(A, {axis: 1}).reshape([m, 1]), 1));

run("linalg2", "tensordot f64", s, () => np.tensordot(A, B, { axes: 1 }));
  run("linalg2", "einsum matmul f64", s, () => np.einsum("ij,jk->ik", A, B));
  run("linalg2", "multi_dot f64", s, () => np.linalg.multiDot([A, B]));
  run("linalg2", "cholesky f64", s, () => np.linalg.cholesky(np.add(np.dot(A, np.transpose(A)), np.eye(m, { dtype: "float64" }))));
  run("linalg2", "cond f64", [1], () => np.linalg.cond(A));
  run("linalg2", "pinv f64", s, () => np.linalg.pinv(A));
  run("linalg2", "slogdet f64", [1], () => np.linalg.slogdet(A));
  run("linalg2", "svdvals f64", [m], () => np.linalg.svdVals(A));
  run("linalg2", "matrix_rank f64", [1], () => np.linalg.matrixRank(A));
  run("linalg2", "matrix_norm fro f64", [1], () => np.linalg.matrixNorm(A, { ord: "fro" }));
  run("linalg2", "vecdot f64", [m], () => np.linalg.vecdot(A, B));
run("rfft2",   "rfft2 f64",    [m, m], () => np.fft.rfft2(A));
  run("irfft2",  "irfft2 f64",   [m, m], () => np.fft.irfft2(np.fft.rfft2(A)));
  run("rfftn",   "rfftn f64",    [m, m], () => np.fft.rfftn(A));
  run("irfftn",  "irfftn f64",   [m, m], () => np.fft.irfftn(np.fft.rfftn(A)));
run("io", "polyfit f64", [3], () => np.polyfit(np.arange(m, { dtype: "float64" }), np.arange(m, { dtype: "float64" }), 2));

run("random2", "rng.multivariate_normal f64", [m, m], () => {
    const mean = np.zeros([m]);
    const cov = np.eye(m);
    return np.random.defaultRng(0).multivariateNormal(mean, cov, [m]);
  });
  run("random2", "rng.dirichlet f64", [m, m], () => {
    const alpha = np.ones([m]);
    return np.random.defaultRng(0).dirichlet(alpha, m);
  });
  run("random2", "rng.multinomial i64", [m, m], () => {
    const pvals = np.divide(np.ones([m]), m);
    return np.random.defaultRng(0).multinomial(100, pvals, m);
  });
run("p16a", "cumsum2d f64", s, () => np.cumsum(A, { axis: 0 }));
  run("p16a", "cumprod2d f64", s, () => np.cumprod(A, { axis: 1 }));

  run("linalg", "inv", s, () => np.linalg.inv(A));
  run("linalg", "solve", s, () => np.linalg.solve(A, b));
  run("linalg", "det", s, () => np.linalg.det(A));
  run("linalg", "svd", s, () => np.linalg.svd(A));
  run("linalg", "qr", s, () => np.linalg.qr(A));
  run("linalg", "eigh", s, () => np.linalg.eigh(S));
  if (m <= 512) run("linalg", "eig", s, () => np.linalg.eig(A));
  // P1-5 complex linalg (D-038–D-044): CA = A + i*A.T (diagonally dominant),
  // CH = CA + CA^H (Hermitian), cb = (1+i)*b
  const CA = np.add(A, np.multiply(A.T, np.complex(0, 1)));
  const CH = np.add(CA, np.conj(CA).T);
  const cb = np.multiply(b, np.complex(1, 1));
  run("linalg", "inv c128", s, () => np.linalg.inv(CA));
  run("linalg", "solve c128", s, () => np.linalg.solve(CA, cb));
  run("linalg", "det c128", s, () => np.linalg.det(CA));
  run("linalg", "svd c128", s, () => np.linalg.svd(CA));
  run("linalg", "qr c128", s, () => np.linalg.qr(CA));
  run("linalg", "eigh c128", s, () => np.linalg.eigh(CH));
  if (m <= 512) run("linalg", "eig c128", s, () => np.linalg.eig(CA));
}

// ---- api coverage (D-032): every implemented callable timed at a small size ----
{
  const n = 1_000;
  const v = rng.random([n]);
  const w = rng.random([n]);
  const M = np.add(rng.random([32, 32]), np.multiply(np.eye(32), 32));
  const S = np.add(M, M.T);
  const bvec = rng.random([32]);
  const vs = [n];
  const ms = [32, 32];
  const rs = new np.random.RandomState(1);
  np.random.seed(1);
  const api = (name, shape, fn) => run("api", name, shape, fn);

  api("asarray", vs, () => np.asarray(v));
  api("full", vs, () => np.full([n], 2.5));
  api("fullLike", vs, () => np.fullLike(v, 2.5));
  api("zerosLike", vs, () => np.zerosLike(v));
  api("onesLike", vs, () => np.onesLike(v));
  api("emptyLike", vs, () => np.emptyLike(v));
  api("identity", ms, () => np.identity(32));
  api("linspace", vs, () => np.linspace(0, 1, n));
  api("mayShareMemory", vs, () => np.mayShareMemory(v, w));
  api("copy", ms, () => np.copy(M));
  api("ascontiguousarray", ms, () => np.ascontiguousarray(M.T));
  api("asfortranarray", ms, () => np.asfortranarray(M));
  api("promoteTypes", [], () => np.promoteTypes("int32", "float32"));
  api("canCast", [], () => np.canCast("int64", "float32", "same_kind"));
  api("broadcastShapes", [], () => np.broadcastShapes([32, 1], [1, 32]));
  api("broadcastTo", ms, () => np.broadcastTo(bvec, [32, 32]));
  api("subtract", vs, () => np.subtract(v, w));
  api("power", vs, () => np.power(v, 2.0));
  api("mod", vs, () => np.mod(v, 0.3));
  api("floorDivide", vs, () => np.floorDivide(v, 0.3));
  api("geterr", [], () => np.geterr());
  api("seterr", [], () => np.seterr(np.geterr()));
  api("negative", vs, () => np.negative(v));
  api("reshape", vs, () => np.reshape(v, [10, 100]));
  api("ravel", ms, () => np.ravel(M));
  api("squeeze", vs, () => np.squeeze(np.reshape(v, [1, n])));
  api("expandDims", vs, () => np.expandDims(v, 0));
  api("transpose", ms, () => np.transpose(M));
  api("swapaxes", ms, () => np.swapAxes(M, 0, 1));
  api("moveaxis", ms, () => np.moveAxis(M, 0, 1));
  api("where", vs, () => np.where(v, v, w));
  api("nonzero", vs, () => np.nonzero(v));
  api("min", vs, () => np.min(v));
  api("amin", vs, () => np.amin(v));
  api("amax", vs, () => np.amax(v));
  api("argmin", vs, () => np.argmin(v));
  api("prod", vs, () => np.prod(v));
  api("var", vs, () => np.var(v));
  api("dot", vs, () => np.dot(v, w));
  api("inner", vs, () => np.inner(v, w));
  api("outer", [100, 100], () => np.outer(v.slice([[0, 100]]), w.slice([[0, 100]])));

  api("ndarray.argmin", vs, () => v.argmin());
  api("ndarray.min", vs, () => v.min());
  api("ndarray.prod", vs, () => v.prod());
  api("ndarray.var", vs, () => v.var());
  api("ndarray.flatten", ms, () => M.flatten());
  api("ndarray.ravel", ms, () => M.ravel());
  api("ndarray.item", [], () => v.item(0));
  api("ndarray.squeeze", vs, () => np.reshape(v, [1, n]).squeeze());
  api("ndarray.sum", vs, () => v.sum());
  api("ndarray.mean", vs, () => v.mean());
  api("ndarray.std", vs, () => v.std());
  api("ndarray.max", vs, () => v.max());
  api("ndarray.argmax", vs, () => v.argmax());
  const B1 = np.reshape(v, [1, n]);
  api("ndarray.squeeze (view)", vs, () => B1.squeeze());
  api("ndarray.swapaxes", ms, () => M.swapAxes(0, 1));
  api("ndarray.transpose", ms, () => M.transpose());

  api("linalg.matmul", ms, () => np.linalg.matmul(M, M));
  api("linalg.norm", ms, () => np.linalg.norm(M));
  api("linalg.eigvals", ms, () => np.linalg.eigvals(M));
  api("linalg.eigvalsh", ms, () => np.linalg.eigvalsh(S));
  api("linalg.lstsq", ms, () => np.linalg.lstsq(M, bvec));

  const V = np.fft.fft(v);
  api("fft.ifft", vs, () => np.fft.ifft(V));
  api("fft.irfft", vs, () => np.fft.irfft(np.fft.rfft(v)));
  api("fft.fftn", ms, () => np.fft.fftn(M));
  api("fft.ifftn", ms, () => np.fft.ifftn(M));
  api("fft.ifft2", ms, () => np.fft.ifft2(M));
  api("fft.fftfreq", vs, () => np.fft.fftfreq(n));
  api("fft.rfftfreq", vs, () => np.fft.rfftfreq(n));

  api("rng.uniform", vs, () => rng.uniform(0, 1, [n]));
  api("rng.normal", vs, () => rng.normal(0, 1, [n]));
  api("rng.choice", vs, () => rng.choice(n, [n]));
  api("rng.permutation", vs, () => rng.permutation(n));
  api("rng.shuffle", vs, () => rng.shuffle(v));

  api("rs.seed", [], () => rs.seed(1));
  api("rs.random", vs, () => rs.random([n]));
  api("rs.random_sample", vs, () => rs.randomSample([n]));
  api("rs.rand", vs, () => rs.rand(n));
  api("rs.randn", vs, () => rs.randn(n));
  api("rs.standard_normal", vs, () => rs.standardNormal([n]));
  api("rs.normal", vs, () => rs.normal(0, 1, [n]));
  api("rs.uniform", vs, () => rs.uniform(0, 1, [n]));
  api("rs.randint", vs, () => rs.randint(0, 1000, [n]));
  api("rs.choice", vs, () => rs.choice(n, [n]));
  api("rs.permutation", vs, () => rs.permutation(n));
  api("rs.shuffle", vs, () => rs.shuffle(w));

  api("np.random.seed", [], () => np.random.seed(1));
  api("np.random.random", vs, () => np.random.random([n]));
  api("np.random.random_sample", vs, () => np.random.randomSample([n]));
  api("np.random.rand", vs, () => np.random.rand(n));
  api("np.random.randn", vs, () => np.random.randn(n));
  api("np.random.standard_normal", vs, () => np.random.standardNormal([n]));
  api("np.random.normal", vs, () => np.random.normal(0, 1, [n]));
  api("np.random.uniform", vs, () => np.random.uniform(0, 1, [n]));
  api("np.random.randint", vs, () => np.random.randint(0, 1000, [n]));
  api("np.random.choice", vs, () => np.random.choice(n, [n]));
  api("np.random.permutation", vs, () => np.random.permutation(n));
  api("np.random.shuffle", vs, () => np.random.shuffle(w));
  // P1 complex (D-033): a = v + i*w
  {
    const a = np.add(v, np.multiply(w, np.complex(0, 1)));
    api("complex.add", vs, () => np.add(a, a));
    api("complex.multiply", vs, () => np.multiply(a, a));
    api("complex.divide", vs, () => np.divide(a, a));
    api("complex.abs", vs, () => np.abs(a));
    api("complex.sqrt", vs, () => np.sqrt(a));
    api("complex.exp", vs, () => np.exp(a));
    api("complex.log", vs, () => np.log(a));
    api("complex.power", vs, () => np.power(a, 3));
    api("real", vs, () => np.real(a));
    api("imag", vs, () => np.imag(a));
    api("conj", vs, () => np.conj(a));
    api("conjugate", vs, () => np.conjugate(a));
    api("ndarray.conj", vs, () => a.conj());
    api("angle", vs, () => np.angle(a));
    api("iscomplex", vs, () => np.iscomplex(a));
    api("isreal", vs, () => np.isreal(a));
    api("iscomplexobj", vs, () => np.iscomplexobj(a));
    api("isrealobj", vs, () => np.isrealobj(a));
    // P1 complex reductions (D-034)
    api("complex.sum", vs, () => np.sum(a));
    api("complex.prod", vs, () => np.prod(a));
    api("complex.mean", vs, () => np.mean(a));
    api("complex.min", vs, () => np.min(a));
    api("complex.max", vs, () => np.max(a));
    api("complex.argmin", vs, () => np.argmin(a));
    api("complex.argmax", vs, () => np.argmax(a));
    api("complex.var", vs, () => np.var(a));
    api("complex.std", vs, () => np.std(a));
    const cm = a.reshape([10, n / 10]);
    api("complex.sum axis=0", vs, () => np.sum(cm, { axis: 0 }));
    api("complex.sum axis=1", vs, () => np.sum(cm, { axis: 1 }));
    // P1 complex matmul family (D-035/D-036): gemm, gemv and dotu paths
    const b = np.add(w, np.multiply(v, np.complex(0, 1)));
    const CM = np.add(M, np.multiply(S, np.complex(0, 1)));
    const cv = b.slice([[0, 32]]);
    api("complex.dot", vs, () => np.dot(a, b));
    api("complex.inner", vs, () => np.inner(a, b));
    api("complex.outer", [100, 100], () => np.outer(a.slice([[0, 100]]), b.slice([[0, 100]])));
    api("complex.matmul", ms, () => np.matmul(CM, CM));
    api("complex.matmul matvec", ms, () => np.matmul(CM, cv));
    // P1-5 complex linalg API cases (D-042–D-044); CH is Hermitian
    const CH = np.add(CM, np.conj(CM).T);
    api("complex.linalg.norm", ms, () => np.linalg.norm(CM));
    api("complex.linalg.eigvals", ms, () => np.linalg.eigvals(CM));
    api("complex.linalg.eigvalsh", ms, () => np.linalg.eigvalsh(CH));
    api("complex.linalg.lstsq", ms, () => np.linalg.lstsq(CM, cv));
  }
}

const out = join(dirname(fileURLToPath(import.meta.url)), "..", "reports", "nativpy-suite.json");
mkdirSync(dirname(out), { recursive: true });
writeFileSync(
  out,
  JSON.stringify(
    {
      node: process.version,
      platform: `${process.platform}-${process.arch}`,
      linalg_backend: np.linalg.backend(),
      large: LARGE,
      repeats: REPEATS,
      min_sample_ms: MIN_SAMPLE_MS,
      results,
    },
    null,
    1,
  ),
);
console.log(`wrote ${out}`);

