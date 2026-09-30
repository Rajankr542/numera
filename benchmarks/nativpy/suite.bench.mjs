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

  run("random", "rng.random", s, () => rng.random([n]));
  run("random", "rng.standardNormal", s, () => rng.standardNormal([n]));
  run("random", "rng.integers", s, () => rng.integers(0, 1000, [n]));
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
  run("fft", "fft2 f64", s, () => np.fft.fft2(A));
}

// ---- decompositions (well-conditioned inputs) ----
for (const m of DECOMP) {
  const s = [m, m];
  const A = np.add(rng.random([m, m]), np.multiply(np.eye(m), m));
  const S = np.add(A, A.T);
  const b = rng.random([m]);
  run("linalg", "inv", s, () => np.linalg.inv(A));
  run("linalg", "solve", s, () => np.linalg.solve(A, b));
  run("linalg", "det", s, () => np.linalg.det(A));
  run("linalg", "svd", s, () => np.linalg.svd(A));
  run("linalg", "qr", s, () => np.linalg.qr(A));
  run("linalg", "eigh", s, () => np.linalg.eigh(S));
  if (m <= 512) run("linalg", "eig", s, () => np.linalg.eig(A));
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

