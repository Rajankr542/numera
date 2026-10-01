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
  run("linalg", "inv", s, () => np.linalg.inv(A));
  run("linalg", "solve", s, () => np.linalg.solve(A, b));
  run("linalg", "det", s, () => np.linalg.det(A));
  run("linalg", "svd", s, () => np.linalg.svd(A));
  run("linalg", "qr", s, () => np.linalg.qr(A));
  run("linalg", "eigh", s, () => np.linalg.eigh(S));
  if (m <= 512) run("linalg", "eig", s, () => np.linalg.eig(A));
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
  api("promoteTypes", [], () => np.promoteTypes("int32", "float32"));
  api("broadcastShapes", [], () => np.broadcastShapes([32, 1], [1, 32]));
  api("broadcastTo", ms, () => np.broadcastTo(bvec, [32, 32]));
  api("subtract", vs, () => np.subtract(v, w));
  api("power", vs, () => np.power(v, 2.0));
  api("mod", vs, () => np.mod(v, 0.3));
  api("floorDivide", vs, () => np.floorDivide(v, 0.3));
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

