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

