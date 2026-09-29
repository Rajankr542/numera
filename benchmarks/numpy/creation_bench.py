#!/usr/bin/env python3
"""NumPy counterpart of benchmarks/nativpy/creation.bench.mjs (PLAN §65)."""
import gc
import json
import platform
import time
from pathlib import Path

import numpy as np

SIZES = [1_000, 100_000, 1_000_000]
REPEATS = 7


def bench(fn):
    fn()
    times = []
    for _ in range(REPEATS):
        gc.collect()
        t0 = time.perf_counter_ns()
        fn()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    times.sort()
    return {"median_ms": times[len(times) // 2], "min_ms": times[0]}


results = []
for n in SIZES:
    lst = [i * 0.5 for i in range(n)]
    arr = np.array(lst)
    cases = {
        "zeros(float64)": lambda: np.zeros(n),
        "array(list float64)": lambda: np.array(lst),
        "fromTypedArray(float64)": lambda: np.frombuffer(arr.tobytes(), dtype=np.float64).copy(),
        "astype(float32)": lambda: arr.astype(np.float32),
        "copy": lambda: arr.copy(),
        "toTypedArray": lambda: arr.tobytes(),
    }
    for name, fn in cases.items():
        r = bench(fn)
        results.append({"name": name, "n": n, **r})
        print(f"{name:<26} n={n:>8}  median {r['median_ms']:.3f} ms")

out = Path(__file__).resolve().parents[1] / "reports" / "numpy-creation.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"numpy": np.__version__, "python": platform.python_version(),
                           "platform": platform.platform(), "results": results}, indent=1))
print(f"wrote {out}")
