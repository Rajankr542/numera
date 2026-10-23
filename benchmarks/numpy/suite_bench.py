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

    # P4 math ufuncs (V-P4)
    run("math", "sin f64", s, lambda: np.sin(a))
    run("math", "cos f64", s, lambda: np.cos(a))
    run("math", "tan f64", s, lambda: np.tan(a))
    run("math", "arcsin f64", s, lambda: np.arcsin(a))
    run("math", "arccos f64", s, lambda: np.arccos(a))
    run("math", "arctan f64", s, lambda: np.arctan(a))
    run("math", "arctan2 f64", s, lambda: np.arctan2(a, b))
    run("math", "sinh f64", s, lambda: np.sinh(a))
    run("math", "cosh f64", s, lambda: np.cosh(a))
    run("math", "tanh f64", s, lambda: np.tanh(a))
    run("math", "arcsinh f64", s, lambda: np.arcsinh(a))
    run("math", "arccosh f64", s, lambda: np.arccosh(a))
    run("math", "arctanh f64", s, lambda: np.arctanh(a))
    run("math", "hypot f64", s, lambda: np.hypot(a, b))
    run("math", "deg2rad f64", s, lambda: np.deg2rad(a))
    run("math", "rad2deg f64", s, lambda: np.rad2deg(a))
    run("math", "degrees f64", s, lambda: np.degrees(a))
    run("math", "radians f64", s, lambda: np.radians(a))
    run("math", "exp2 f64", s, lambda: np.exp2(a))
    run("math", "expm1 f64", s, lambda: np.expm1(a))
    run("math", "log2 f64", s, lambda: np.log2(a))
    run("math", "log10 f64", s, lambda: np.log10(a))
    run("math", "log1p f64", s, lambda: np.log1p(a))
    run("math", "logaddexp f64", s, lambda: np.logaddexp(a, b))
    run("math", "logaddexp2 f64", s, lambda: np.logaddexp2(a, b))
    run("math", "cbrt f64", s, lambda: np.cbrt(a))
    run("math", "square f64", s, lambda: np.square(a))
    run("math", "reciprocal f64", s, lambda: np.reciprocal(a))
    run("math", "floor f64", s, lambda: np.floor(a))
    run("math", "ceil f64", s, lambda: np.ceil(a))
    run("math", "trunc f64", s, lambda: np.trunc(a))
    run("math", "rint f64", s, lambda: np.rint(a))
    run("math", "positive f64", s, lambda: np.positive(a))
    run("math", "fix f64", s, lambda: np.fix(a))
    run("math", "round f64", s, lambda: np.round(a))
    run("math", "clip f64", s, lambda: np.clip(a, 0.1, 0.9))
    run("math", "fmod f64", s, lambda: np.fmod(a, b))
    run("math", "floatPower f64", s, lambda: np.float_power(a, b))
    run("math", "sign f64", s, lambda: np.sign(a))
    run("math", "heaviside f64", s, lambda: np.heaviside(a, 0.5))
    run("math", "maximum f64", s, lambda: np.maximum(a, b))
    run("math", "minimum f64", s, lambda: np.minimum(a, b))
    run("math", "fmax f64", s, lambda: np.fmax(a, b))
    run("math", "fmin f64", s, lambda: np.fmin(a, b))
    run("math", "fabs f64", s, lambda: np.fabs(a))
    run("math", "gcd i32", s, lambda: np.gcd(ai, ai))
    run("math", "lcm i32", s, lambda: np.lcm(ai, ai))
    run("math", "copysign f64", s, lambda: np.copysign(a, b))
    run("math", "nextafter f64", s, lambda: np.nextafter(a, b))
    run("math", "spacing f64", s, lambda: np.spacing(a))
    run("math", "ldexp f64", s, lambda: np.ldexp(a, ai))
    run("math", "signbit f64", s, lambda: np.signbit(a))
    run("math", "divmod f64", s, lambda: np.divmod(a, b))
    run("math", "modf f64", s, lambda: np.modf(a))
    run("math", "frexp f64", s, lambda: np.frexp(a))
    run("math", "sinc f64", s, lambda: np.sinc(a))
    run("math", "i0 f64", s, lambda: np.i0(a))
    run("math", "unwrap f64", s, lambda: np.unwrap(a))
    run("math", "nanToNum f64", s, lambda: np.nan_to_num(a))
    run("math", "realIfClose f64", s, lambda: np.real_if_close(a))
    run("math", "ndarray.clip f64", s, lambda: a.clip(0.1, 0.9))
    run("math", "ndarray.round f64", s, lambda: a.round())
    run("math", "ndarray.conjugate f64", s, lambda: a.conjugate())

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
    # P1-5e complex FFT input: c = a + i*b
    c = a + b * 1j
    c64 = c.astype(np.complex64)
    run("fft", "fft c128", s, lambda: np.fft.fft(c))
    run("fft", "fft c64", s, lambda: np.fft.fft(c64))
    run("fft", "ifft c128", s, lambda: np.fft.ifft(c))

    run("random", "rng.random", s, lambda: rng.random(n))
    run("random", "rng.standardNormal", s, lambda: rng.standard_normal(n))
    run("random", "rng.integers", s, lambda: rng.integers(0, 1000, n))

run("sorting", "sort f64", s, lambda: np.sort(a))
    run("sorting", "sort i32", s, lambda: np.sort(ai))
    run("sorting", "argsort f64", s, lambda: np.argsort(a))
    run("sorting", "partition f64", s, lambda: np.partition(a, n//2))
    run("sorting", "searchsorted f64", s, lambda: np.searchsorted(a, 0.5))
    run("sorting", "unique f64", s, lambda: np.unique(a))
    run("sorting", "isin f64", s, lambda: np.isin(a, b))
    run("sorting", "lexsort f64", s, lambda: np.lexsort([a, b]))
    run("sorting", "ediff1d i32", s, lambda: np.ediff1d(ai))

run("logic", "equal f64", s, lambda: np.equal(a, b))
    run("logic", "notEqual f64", s, lambda: np.not_equal(a, b))
    run("logic", "less f64", s, lambda: np.less(a, b))
    run("logic", "lessEqual f64", s, lambda: np.less_equal(a, b))
    run("logic", "greater f64", s, lambda: np.greater(a, b))
    run("logic", "greaterEqual f64", s, lambda: np.greater_equal(a, b))
    run("logic", "logicalAnd f64", s, lambda: np.logical_and(a, b))
    run("logic", "logicalOr f64", s, lambda: np.logical_or(a, b))
    run("logic", "logicalXor f64", s, lambda: np.logical_xor(a, b))
    run("logic", "logicalNot f64", s, lambda: np.logical_not(a))
    run("logic", "isnan f64", s, lambda: np.isnan(a))
    run("logic", "isinf f64", s, lambda: np.isinf(a))
    run("logic", "isfinite f64", s, lambda: np.isfinite(a))
    run("logic", "isposinf f64", s, lambda: np.isposinf(a))
    run("logic", "isneginf f64", s, lambda: np.isneginf(a))
    run("logic", "all f64", s, lambda: np.all(a))
    run("logic", "any f64", s, lambda: np.any(a))
    run("logic", "isclose f64", s, lambda: np.isclose(a, b))
    run("logic", "bitwiseAnd i32", s, lambda: np.bitwise_and(ai, ai))
    run("logic", "bitwiseOr i32", s, lambda: np.bitwise_or(ai, ai))
    run("logic", "bitwiseXor i32", s, lambda: np.bitwise_xor(ai, ai))
    run("logic", "invert i32", s, lambda: np.invert(ai))
    run("logic", "leftShift i32", s, lambda: np.left_shift(ai, ai))
    run("logic", "rightShift i32", s, lambda: np.right_shift(ai, ai))
    run("logic", "bitwiseCount i32", s, lambda: np.bitwise_count(ai))
    run("logic", "packbits u8", [n//8], lambda: np.packbits(np.zeros(n//8, dtype=np.uint8)))
    run("logic", "unpackbits u8", [n//8], lambda: np.unpackbits(np.zeros(n//8, dtype=np.uint8)))
run("manip", "concatenate f64", [n * 2], lambda: np.concatenate([a, b]))
    run("manip", "stack f64", [2, n], lambda: np.stack([a, b]))
    run("manip", "repeat f64", [n * 2], lambda: np.repeat(a, 2))
    run("manip", "flip f64", s, lambda: np.flip(a))
    run("manip", "roll f64", s, lambda: np.roll(a, 10))
    run("manip", "tile f64", [n * 2], lambda: np.tile(a, 2))
    run("manip", "split f64", s, lambda: np.split(a, 2))
run("creation2", "logspace f64", [50], lambda: np.logspace(0, 3))
    run("creation2", "geomspace f64", [50], lambda: np.geomspace(1, 1000))
    run("creation2", "astype f64->f32", s, lambda: a.astype(np.float32))
    run("creation2", "tri f64", s, lambda: np.tri(m))
    run("creation2", "tril f64", s, lambda: np.tril(A))
    run("creation2", "triu f64", s, lambda: np.triu(A))
    run("creation2", "diag extract", [m], lambda: np.diag(A))
    run("creation2", "diag construct", s, lambda: np.diag(np.arange(m)))
    run("creation2", "vander f64", s, lambda: np.vander(np.arange(m)))
    run("creation2", "meshgrid f64", [2, m], lambda: np.meshgrid(np.arange(m), np.arange(m)))
run("indexing2", "take f64", s, lambda: np.take(a, idx))
    run("indexing2", "compress f64", s, lambda: np.compress(a > 0.5, a))
    run("indexing2", "argwhere f64", s, lambda: np.argwhere(a > 0.5))
    run("indexing2", "flatnonzero f64", s, lambda: np.flatnonzero(a))
    run("indexing2", "countNonzero f64", [], lambda: np.count_nonzero(a))
    run("indexing2", "ravelMultiIndex i32", s, lambda: np.ravel_multi_index([idx, idx], [n, n]))
run("stats", "median f64", s, lambda: np.median(a))
    run("stats", "quantile f64", s, lambda: np.quantile(a, 0.5))
    run("stats", "nansum f64", s, lambda: np.nansum(a))
    run("stats", "nanmean f64", s, lambda: np.nanmean(a))
    run("stats", "cumsum f64", s, lambda: np.cumsum(a))
    run("stats", "cumprod f64", s, lambda: np.cumprod(a))
    run("stats", "diff f64", [n - 1], lambda: np.diff(a))
    run("stats", "histogram f64", [10], lambda: np.histogram(a))
    run("stats", "bincount i32", [1001], lambda: np.bincount(np.abs(ai)))
    run("stats", "digitize f64", s, lambda: np.digitize(a, np.linspace(0, 1, 100)))
    run("stats", "interp f64", s, lambda: np.interp(a, np.linspace(0, 1, 100), np.linspace(0, 1, 100)))
    run("stats", "correlate f64", [2 * 1000 - 1], lambda: np.correlate(a[:1000], a[:1000], "full"))
    run("stats", "trapezoid f64", [], lambda: np.trapezoid(a))
    run("stats", "cov f64", [m, m], lambda: np.cov(A))
    run("stats", "corrcoef f64", [m, m], lambda: np.corrcoef(A))
    run("stats", "gradient f64", [m], lambda: np.gradient(A[0]))

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
    # P1-4 complex matmul (D-035/D-036): C = A + i*B
    C = A + B * 1j
    C64 = C.astype(np.complex64)
    run("matmul", "matmul c128", s, lambda: np.matmul(C, C))
    run("matmul", "matmul c64", s, lambda: np.matmul(C64, C64))
    run("fft", "fft2 f64", s, lambda: np.fft.fft2(A))

for m in DECOMP:
    s = [m, m]
    A = rng.random((m, m)) + np.eye(m) * m
    S = A + A.T
    b = rng.random(m)
— paste inside `for m in MAT:` block
    run("manip", "flip2d f64", s, lambda: np.flip(A))
    run("manip", "rot90 f64", s, lambda: np.rot90(A))
    run("manip", "pad f64", [m + 4, m + 4], lambda: np.pad(A, 2))
    run("manip", "vstack f64", [m * 2, m], lambda: np.vstack([A, B]))
    run("manip", "hstack f64", [m, m * 2], lambda: np.hstack([A, B]))
    run("manip", "concatenate2d f64", [m * 2, m], lambda: np.concatenate([A, B], axis=0))
run("indexing2", "diagonal2d f64", [m], lambda: np.diagonal(A))
    run("indexing2", "trace f64", [], lambda: np.trace(A))
    run("indexing2", "takeAlongAxis f64", s, lambda: np.take_along_axis(A, np.argmax(A, axis=1).reshape(m, 1), 1))

    run("linalg", "inv", s, lambda: np.linalg.inv(A))
    run("linalg", "solve", s, lambda: np.linalg.solve(A, b))
    run("linalg", "det", s, lambda: np.linalg.det(A))
    run("linalg", "svd", s, lambda: np.linalg.svd(A))
    run("linalg", "qr", s, lambda: np.linalg.qr(A))
    run("linalg", "eigh", s, lambda: np.linalg.eigh(S))
    if m <= 512:
        run("linalg", "eig", s, lambda: np.linalg.eig(A))
    # P1-5 complex linalg (D-038–D-044): CA = A + i*A.T (diagonally dominant),
    # CH = CA + CA^H (Hermitian), cb = (1+i)*b
    CA = A + A.T * 1j
    CH = CA + CA.conj().T
    cb = b * (1 + 1j)
    run("linalg", "inv c128", s, lambda: np.linalg.inv(CA))
    run("linalg", "solve c128", s, lambda: np.linalg.solve(CA, cb))
    run("linalg", "det c128", s, lambda: np.linalg.det(CA))
    run("linalg", "svd c128", s, lambda: np.linalg.svd(CA))
    run("linalg", "qr c128", s, lambda: np.linalg.qr(CA))
    run("linalg", "eigh c128", s, lambda: np.linalg.eigh(CH))
    if m <= 512:
        run("linalg", "eig c128", s, lambda: np.linalg.eig(CA))

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
api("copy", ms, lambda: np.copy(M))
api("ascontiguousarray", ms, lambda: np.ascontiguousarray(M.T))
api("asfortranarray", ms, lambda: np.asfortranarray(M))
api("promoteTypes", [], lambda: np.promote_types("int32", "float32"))
api("canCast", [], lambda: np.can_cast("int64", "float32", "same_kind"))
api("broadcastShapes", [], lambda: np.broadcast_shapes((32, 1), (1, 32)))
api("broadcastTo", ms, lambda: np.broadcast_to(bvec, (32, 32)))
api("subtract", vs, lambda: np.subtract(v, w))
api("power", vs, lambda: np.power(v, 2.0))
api("mod", vs, lambda: np.mod(v, 0.3))
api("floorDivide", vs, lambda: np.floor_divide(v, 0.3))
api("geterr", [], lambda: np.geterr())
api("seterr", [], lambda: np.seterr(**np.geterr()))
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
# P1 complex reductions (D-034)
api("complex.sum", vs, lambda: np.sum(a))
api("complex.prod", vs, lambda: np.prod(a))
api("complex.mean", vs, lambda: np.mean(a))
api("complex.min", vs, lambda: np.min(a))
api("complex.max", vs, lambda: np.max(a))
api("complex.argmin", vs, lambda: np.argmin(a))
api("complex.argmax", vs, lambda: np.argmax(a))
api("complex.var", vs, lambda: np.var(a))
api("complex.std", vs, lambda: np.std(a))
cm = np.reshape(a, (10, n // 10))
api("complex.sum axis=0", vs, lambda: np.sum(cm, axis=0))
api("complex.sum axis=1", vs, lambda: np.sum(cm, axis=1))
# P1 complex matmul family (D-035/D-036): gemm, gemv and dotu paths
b = w + v * 1j
CM = M + S * 1j
cv = b[:32]
api("complex.dot", vs, lambda: np.dot(a, b))
api("complex.inner", vs, lambda: np.inner(a, b))
api("complex.outer", [100, 100], lambda: np.outer(a[:100], b[:100]))
api("complex.matmul", ms, lambda: np.matmul(CM, CM))
api("complex.matmul matvec", ms, lambda: np.matmul(CM, cv))
# P1-5 complex linalg API cases (D-042–D-044); CH is Hermitian
CH = CM + CM.conj().T
api("complex.linalg.norm", ms, lambda: np.linalg.norm(CM))
api("complex.linalg.eigvals", ms, lambda: np.linalg.eigvals(CM))
api("complex.linalg.eigvalsh", ms, lambda: np.linalg.eigvalsh(CH))
api("complex.linalg.lstsq", ms, lambda: np.linalg.lstsq(CM, cv))


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

