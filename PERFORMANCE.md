# PERFORMANCE

No performance claims are made yet. The numbers below are a single local run
(macOS arm64, Node 22.7.0, NumPy 2.5.3), median of 7 repetitions after a
warm-up, from `pnpm bench` / `pnpm bench:numpy`. Raw JSON is written to
`benchmarks/reports/` (git-ignored).

## Creation / conversion baseline (n = 1,000,000, float64, median ms)

| Operation | nativpy | NumPy | Ratio (nativpy / NumPy) |
|-----------|--------:|------:|------:|
| `zeros` | 0.644 | 0.070 | 9.2× slower |
| `array(list)` | 57.4 | 18.2 | 3.1× slower |
| `fromTypedArray` / `frombuffer().copy()` | 0.695 | 0.360 | 1.9× slower |
| `astype(float32)` | 2.31 | 0.153 | 15× slower |
| `copy` | 0.752 | 0.152 | 4.9× slower |
| `toTypedArray` / `tobytes` | 0.433 | 0.148 | 2.9× slower |

nativpy is slower on every operation measured. M1 aimed for correctness, not
speed. These causes are likely but **not yet profiled**:
- `zeros` zero-fills memory eagerly; NumPy uses `calloc`, which gets lazily zeroed pages.
- `copy`/`astype` go through the generic per-element strided loop in
  `for_each_element`. There are no contiguous fast paths (`memcpy`,
  vectorized casts) yet.
- `array(list)` walks nested JS arrays element by element through Node-API.

Treat these as the baseline for the M4 kernel work (PLAN §65). Re-run and
update this file after each optimization, and profile before claiming causes.
