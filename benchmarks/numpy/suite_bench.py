#!/usr/bin/env python3
"""NumPy counterpart of benchmarks/nativpy/suite.bench.mjs (PLAN §38-§43).

Same case names, same inputs and the same timing method: median of REPEATS
samples, each sample repeating the call enough times to last >= MIN_SAMPLE_MS.

Run: pnpm bench:suite:numpy   (NATIVPY_BENCH_LARGE=1 / NATIVPY_BENCH_FILTER=...)
"""
import gc
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

LARGE = os.environ.get("NATIVPY_BENCH_LARGE") == "1"
FILTER = os.environ.get("NATIVPY_BENCH_FILTER", "")
VEC = [1_000, 100_000, 1_000_000, 10_000_000] if LARGE else [1_000, 100_000, 1_000_000]
MAT = [32, 128, 512, 1024, 2048] if LARGE else [32, 128, 512, 1024]
DECOMP = [32, 128, 512, 1024] if LARGE else [32, 128, 512]
REPEATS = 7
MIN_SAMPLE_MS = 5.0


def bench(fn):
    fn()
    inner = 1
    while True:
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        ms = (time.perf_counter_ns() - t0) / 1e6
        if ms >= MIN_SAMPLE_MS or inner >= 1 << 20:
            break
        inner *= 16 if ms <= 0 else min(16, max(2, int(-(-MIN_SAMPLE_MS // ms))))
    times = []
    for _ in range(REPEATS):
        gc.collect()
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        times.append((time.perf_counter_ns() - t0) / 1e6 / inner)
    times.sort()
    return {"median_ms": times[REPEATS // 2], "min_ms": times[0], "inner": inner}


results = []


def run(category, name, shape, fn):
    full = f"{category}/{name}"
    if FILTER and FILTER not in full:
        return
    r = bench(fn)
    results.append({"category": category, "name": name, "shape": shape, **r})
    print(f"{full:<44} {json.dumps(shape, separators=(',', ':')):<14} {r['median_ms']:11.4f} ms")


rng = np.random.default_rng(12345)

for n in VEC:
    s = [n]
    a = rng.random(n)
    b = rng.random(n)
    a32 = a.astype(np.float32)
    b32 = b.astype(np.float32)
    ai = rng.integers(0, 1000, n, dtype=np.int32)
    idx = rng.integers(0, n, n)
    lst = a.tolist() if n <= 1_000_000 else None
    raw = a.tobytes()

    run("creation", "zeros", s, lambda: np.zeros(n))
    run("creation", "empty", s, lambda: np.empty(n))
    run("creation", "ones", s, lambda: np.ones(n))
    run("creation", "arange", s, lambda: np.arange(n))
    if lst is not None:
        run("creation", "array(list)", s, lambda: np.array(lst))
    # nativpy copies a JS TypedArray into native memory (D-010).
    run("memory", "fromTypedArray", s, lambda: np.frombuffer(raw, dtype=np.float64).copy())
    run("memory", "copy", s, lambda: a.copy())
    run("memory", "toTypedArray", s, lambda: a.tobytes())
    run("memory", "astype(float32)", s, lambda: a.astype(np.float32))

    run("elementwise", "add f64", s, lambda: np.add(a, b))
    run("elementwise", "add f32", s, lambda: np.add(a32, b32))
    run("elementwise", "add i32", s, lambda: np.add(ai, ai))
    run("elementwise", "multiply f64", s, lambda: np.multiply(a, b))
    run("elementwise", "multiply scalar f64", s, lambda: np.multiply(a, 2.5))
    run("elementwise", "divide f64", s, lambda: np.divide(a, b))
    run("elementwise", "sqrt f64", s, lambda: np.sqrt(a))
    run("elementwise", "exp f64", s, lambda: np.exp(a))
    run("elementwise", "log f64", s, lambda: np.log(a))
    run("elementwise", "abs f64", s, lambda: np.abs(a))

    run("reduction", "sum f64", s, lambda: np.sum(a))
    run("reduction", "sum f32", s, lambda: np.sum(a32))
    run("reduction", "sum i32", s, lambda: np.sum(ai))
    run("reduction", "mean f64", s, lambda: np.mean(a))
    run("reduction", "max f64", s, lambda: np.max(a))
    run("reduction", "argmax f64", s, lambda: np.argmax(a))
    run("reduction", "std f64", s, lambda: np.std(a))

    run("view", "slice [::2]", s, lambda: a[::2])
    run("view", "reshape", s, lambda: a.reshape(n // 1000, 1000))
    run("slicing", "take(idx)", s, lambda: np.take(a, idx))

    run("sorting", "sort f64", s, lambda: np.sort(a))

    run("fft", "fft f64", s, lambda: np.fft.fft(a))
    run("fft", "rfft f64", s, lambda: np.fft.rfft(a))

    run("random", "rng.random", s, lambda: rng.random(n))
    run("random", "rng.standardNormal", s, lambda: rng.standard_normal(n))
    run("random", "rng.integers", s, lambda: rng.integers(0, 1000, n))

for m in MAT:
    s = [m, m]
    A = rng.random((m, m))
    B = rng.random((m, m))
    A32 = A.astype(np.float32)
    B32 = B.astype(np.float32)
    row = rng.random(m)
    col = rng.random((m, 1))

    run("broadcast", "add(matrix, row)", s, lambda: np.add(A, row))
    run("broadcast", "add(matrix, col)", s, lambda: np.add(A, col))
    run("transpose", "A.T (view)", s, lambda: A.T)
    # nativpy's copy() is always C order; NumPy's ndarray.copy() defaults to 'C' too.
    run("transpose", "A.T.copy()", s, lambda: A.T.copy())
    # nativpy ufunc results are always C-contiguous (D-014); NumPy's order='K'
    # would keep F order for this input, so force C for equal work.
    run("transpose", "add(A.T, B)", s, lambda: np.add(A.T, B, order="C"))
    run("reduction", "sum axis=0", s, lambda: np.sum(A, axis=0))
    run("reduction", "sum axis=1", s, lambda: np.sum(A, axis=1))
    run("slicing", "A[1:-1, ::2]", s, lambda: A[1:-1, ::2])
    run("matmul", "matmul f64", s, lambda: np.matmul(A, B))
    run("matmul", "matmul f32", s, lambda: np.matmul(A32, B32))
    if m <= 512:
        Ai = A.astype(np.int32)
        run("matmul", "matmul i32", s, lambda: np.matmul(Ai, Ai))
    run("fft", "fft2 f64", s, lambda: np.fft.fft2(A))

for m in DECOMP:
    s = [m, m]
    A = rng.random((m, m)) + np.eye(m) * m
    S = A + A.T
    b = rng.random(m)
    run("linalg", "inv", s, lambda: np.linalg.inv(A))
    run("linalg", "solve", s, lambda: np.linalg.solve(A, b))
    run("linalg", "det", s, lambda: np.linalg.det(A))
    run("linalg", "svd", s, lambda: np.linalg.svd(A))
    run("linalg", "qr", s, lambda: np.linalg.qr(A))
    run("linalg", "eigh", s, lambda: np.linalg.eigh(S))
    if m <= 512:
        run("linalg", "eig", s, lambda: np.linalg.eig(A))

# ---- api coverage (D-032): every implemented callable timed at a small size ----
n = 1_000
v = rng.random(n)
w = rng.random(n)
M = rng.random((32, 32)) + np.eye(32) * 32
S = M + M.T
bvec = rng.random(32)
vs = [n]
ms = [32, 32]
rs = np.random.RandomState(1)
np.random.seed(1)


def api(name, shape, fn):
    run("api", name, shape, fn)


api("asarray", vs, lambda: np.asarray(v))
api("full", vs, lambda: np.full(n, 2.5))
api("fullLike", vs, lambda: np.full_like(v, 2.5))
api("zerosLike", vs, lambda: np.zeros_like(v))
api("onesLike", vs, lambda: np.ones_like(v))
api("emptyLike", vs, lambda: np.empty_like(v))
api("identity", ms, lambda: np.identity(32))
api("linspace", vs, lambda: np.linspace(0, 1, n))
api("mayShareMemory", vs, lambda: np.may_share_memory(v, w))
api("promoteTypes", [], lambda: np.promote_types("int32", "float32"))
api("broadcastShapes", [], lambda: np.broadcast_shapes((32, 1), (1, 32)))
api("broadcastTo", ms, lambda: np.broadcast_to(bvec, (32, 32)))
api("subtract", vs, lambda: np.subtract(v, w))
api("power", vs, lambda: np.power(v, 2.0))
api("mod", vs, lambda: np.mod(v, 0.3))
api("floorDivide", vs, lambda: np.floor_divide(v, 0.3))
api("negative", vs, lambda: np.negative(v))
api("reshape", vs, lambda: np.reshape(v, (10, 100)))
api("ravel", ms, lambda: np.ravel(M))
api("squeeze", vs, lambda: np.squeeze(np.reshape(v, (1, n))))
api("expandDims", vs, lambda: np.expand_dims(v, 0))
api("transpose", ms, lambda: np.transpose(M))
api("swapaxes", ms, lambda: np.swapaxes(M, 0, 1))
api("moveaxis", ms, lambda: np.moveaxis(M, 0, 1))
api("where", vs, lambda: np.where(v, v, w))
api("nonzero", vs, lambda: np.nonzero(v))
api("min", vs, lambda: np.min(v))
api("amin", vs, lambda: np.amin(v))
api("amax", vs, lambda: np.amax(v))
api("argmin", vs, lambda: np.argmin(v))
api("prod", vs, lambda: np.prod(v))
api("var", vs, lambda: np.var(v))
api("dot", vs, lambda: np.dot(v, w))
api("inner", vs, lambda: np.inner(v, w))
api("outer", [100, 100], lambda: np.outer(v[:100], w[:100]))

api("ndarray.argmin", vs, lambda: v.argmin())
api("ndarray.min", vs, lambda: v.min())
api("ndarray.prod", vs, lambda: v.prod())
api("ndarray.var", vs, lambda: v.var())
api("ndarray.flatten", ms, lambda: M.flatten())
api("ndarray.ravel", ms, lambda: M.ravel())
api("ndarray.item", [], lambda: v.item(0))
api("ndarray.squeeze", vs, lambda: np.reshape(v, (1, n)).squeeze())
api("ndarray.sum", vs, lambda: v.sum())
api("ndarray.mean", vs, lambda: v.mean())
api("ndarray.std", vs, lambda: v.std())
api("ndarray.max", vs, lambda: v.max())
api("ndarray.argmax", vs, lambda: v.argmax())
B1 = np.reshape(v, (1, n))
api("ndarray.squeeze (view)", vs, lambda: B1.squeeze())
api("ndarray.swapaxes", ms, lambda: M.swapaxes(0, 1))
api("ndarray.transpose", ms, lambda: M.transpose())

api("linalg.matmul", ms, lambda: np.linalg.matmul(M, M))
api("linalg.norm", ms, lambda: np.linalg.norm(M))
api("linalg.eigvals", ms, lambda: np.linalg.eigvals(M))
api("linalg.eigvalsh", ms, lambda: np.linalg.eigvalsh(S))
api("linalg.lstsq", ms, lambda: np.linalg.lstsq(M, bvec))

V = np.fft.fft(v)
api("fft.ifft", vs, lambda: np.fft.ifft(V))
api("fft.irfft", vs, lambda: np.fft.irfft(np.fft.rfft(v)))
api("fft.fftn", ms, lambda: np.fft.fftn(M))
api("fft.ifftn", ms, lambda: np.fft.ifftn(M))
api("fft.ifft2", ms, lambda: np.fft.ifft2(M))
api("fft.fftfreq", vs, lambda: np.fft.fftfreq(n))
api("fft.rfftfreq", vs, lambda: np.fft.rfftfreq(n))

api("rng.uniform", vs, lambda: rng.uniform(0, 1, n))
api("rng.normal", vs, lambda: rng.normal(0, 1, n))
api("rng.choice", vs, lambda: rng.choice(n, n))
api("rng.permutation", vs, lambda: rng.permutation(n))
api("rng.shuffle", vs, lambda: rng.shuffle(v))

api("rs.seed", [], lambda: rs.seed(1))
api("rs.random", vs, lambda: rs.random(n))
api("rs.random_sample", vs, lambda: rs.random_sample(n))
api("rs.rand", vs, lambda: rs.rand(n))
api("rs.randn", vs, lambda: rs.randn(n))
api("rs.standard_normal", vs, lambda: rs.standard_normal(n))
api("rs.normal", vs, lambda: rs.normal(0, 1, n))
api("rs.uniform", vs, lambda: rs.uniform(0, 1, n))
api("rs.randint", vs, lambda: rs.randint(0, 1000, n))
api("rs.choice", vs, lambda: rs.choice(n, n))
api("rs.permutation", vs, lambda: rs.permutation(n))
api("rs.shuffle", vs, lambda: rs.shuffle(w))

api("np.random.seed", [], lambda: np.random.seed(1))
api("np.random.random", vs, lambda: np.random.random(n))
api("np.random.random_sample", vs, lambda: np.random.random_sample(n))
api("np.random.rand", vs, lambda: np.random.rand(n))
api("np.random.randn", vs, lambda: np.random.randn(n))
api("np.random.standard_normal", vs, lambda: np.random.standard_normal(n))
api("np.random.normal", vs, lambda: np.random.normal(0, 1, n))
api("np.random.uniform", vs, lambda: np.random.uniform(0, 1, n))
api("np.random.randint", vs, lambda: np.random.randint(0, 1000, n))
api("np.random.choice", vs, lambda: np.random.choice(n, n))
api("np.random.permutation", vs, lambda: np.random.permutation(n))
api("np.random.shuffle", vs, lambda: np.random.shuffle(w))
# P1 complex (D-033): a = v + i*w
a = v + w * 1j
api("complex.add", vs, lambda: np.add(a, a))
api("complex.multiply", vs, lambda: np.multiply(a, a))
api("complex.divide", vs, lambda: np.divide(a, a))
api("complex.abs", vs, lambda: np.abs(a))
api("complex.sqrt", vs, lambda: np.sqrt(a))
api("complex.exp", vs, lambda: np.exp(a))
api("complex.log", vs, lambda: np.log(a))
api("complex.power", vs, lambda: np.power(a, 3))
api("real", vs, lambda: np.real(a))
api("imag", vs, lambda: np.imag(a))
api("conj", vs, lambda: np.conj(a))
api("conjugate", vs, lambda: np.conjugate(a))
api("ndarray.conj", vs, lambda: a.conj())
api("angle", vs, lambda: np.angle(a))
api("iscomplex", vs, lambda: np.iscomplex(a))
api("isreal", vs, lambda: np.isreal(a))
api("iscomplexobj", vs, lambda: np.iscomplexobj(a))
api("isrealobj", vs, lambda: np.isrealobj(a))


def blas_name():
    try:
        cfg = np.show_config(mode="dicts")
        return cfg["Build Dependencies"]["blas"]["name"]
    except Exception:  # pragma: no cover - older NumPy
        return "unknown"


out = Path(__file__).resolve().parents[1] / "reports" / "numpy-suite.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "numpy": np.__version__,
    "python": platform.python_version(),
    "platform": platform.platform(),
    "blas": blas_name(),
    "large": LARGE,
    "repeats": REPEATS,
    "min_sample_ms": MIN_SAMPLE_MS,
    "results": results,
}, indent=1))
print(f"wrote {out}")

