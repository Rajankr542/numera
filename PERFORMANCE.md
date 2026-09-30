# PERFORMANCE

No performance claims are made yet. The numbers below are a single local run
(macOS arm64, Node 22.7.0, NumPy 2.5.3), median of 7 repetitions after a
warm-up, from `pnpm bench` / `pnpm bench:numpy`. Raw JSON is written to
`benchmarks/reports/` (git-ignored).

## Creation / conversion (n = 1,000,000, float64, median ms)

| Operation | nativpy M1 | nativpy now | NumPy | now / NumPy |
|-----------|-----------:|------------:|------:|------:|
| `zeros` | 0.644 | 0.012 | 0.070 | 0.17× (see note) |
| `array(list)` | 57.4 | 3.22 | 18.2 | 0.18× |
| `fromTypedArray` / `frombuffer().copy()` | 0.695 | 0.622 | 0.360 | 1.7× slower |
| `astype(float32)` | 2.31 | 0.396 | 0.153 | 2.6× slower |
| `copy` | 0.752 | 0.607 | 0.152 | 4.0× slower |
| `toTypedArray` / `tobytes` | 0.433 | 0.452 | 0.148 | 3.1× slower |

These are single local runs, so they carry noise. The NumPy column was not
re-run; it is the M1 baseline.

Profiling done for these changes (scratch C++ micro-benchmarks, D-013):
- **`zeros`**: the old core zero-filled with aligned `new` + `memset`. It now
  uses `calloc`, so large blocks get lazily zeroed pages from the OS. This
  measures only the allocation: the page-fault cost moves to the first write,
  just as with NumPy's calloc-based `zeros`. The comparison is fair, but it
  does not mean end-to-end work is 5× faster.
- **`array(list)`**: the cost was one Node-API call per element, about 51 ms.
  Rectangular all-number input is now flattened in JS into a Float64Array and
  converted in one native pass. The dtype and range rules are unchanged
  (D-009). Mixed, bool or bigint input still takes the per-element path.
- **`astype`**: the generic N-d iterator prevented auto-vectorization. There
  is now a C-contiguous flat loop (about 6× faster). The remaining gap is
  likely the extra checks in `cast_value`; this is not profiled.
- **`copy`**: it already used `memcpy` in M1 (an earlier note here said it
  didn't; that was wrong). Most of the remaining cost is page faults on the
  freshly allocated destination. A caching buffer pool would address this
  but is deferred (D-013).
- **`toTypedArray` / `fromTypedArray`**: these are dominated by a copy into
  new memory (D-010 requires copies). They are not yet profiled further.

Earlier versions of this file blamed `copy`'s cost on a missing `memcpy`
path and gave the wrong reason for slow `zeros`; both are corrected above.

## Ufuncs (M4/M5) (n = 1,000,000, float64, median ms, same run)

| Operation | nativpy | NumPy | nativpy / NumPy |
|-----------|--------:|------:|------:|
| `add(a, b)` | 0.792 | 0.280 | 2.8× slower |
| `multiply(a, 2.5)` | 0.685 | 0.175 | 3.9× slower |
| `add(ones(1000,1000), arange(1000))` broadcast | 0.674 | 0.398 | 1.7× slower |
| `add(a.T, a.T)` → C-order (NumPy + `ascontiguousarray`) | 1.033 | 0.699 | 1.5× slower |
| `sqrt(a)` | 0.740 | 0.303 | 2.4× slower |
| `exp(a)` | 2.072 | 1.883 | 1.1× slower |

These have not been profiled. `add` and `multiply` land close to `copy`
(0.62 ms), which suggests, but does not prove, that allocating and
page-faulting the destination dominates rather than the loop itself (the
same open question as `copy`, D-013). The loops are plain scalar C++ with no
explicit SIMD. `exp` is compute-bound and close to parity. Profile before
optimizing.

Treat these as the baseline for the M4 kernel work (PLAN §65). Re-run and
update this file after each optimization, and profile before claiming causes.

## M11 baseline: full suite (2026-09-29, before any M11 optimization)

`pnpm bench:suite && pnpm bench:suite:numpy && pnpm bench:compare` writes
`benchmarks/reports/comparison.json` (PLAN §41 records). Environment: Apple M2 Pro
(10 logical CPUs), 16 GB, macOS arm64, Node v22.7.0, Python 3.12.5, NumPy 2.5.3.
Both sides use Accelerate. Method: median of 7 samples, each ≥ 5 ms of repeated
calls, with the same inputs on both sides. Result: 170 comparable cases,
geometric-mean `numpy_ms / nativpy_ms` = **0.285**, 15 faster and 155 slower.
`sort` is not implemented.

Representative rows (median ms):

| Operation | Shape | nativpy | NumPy | |
|---|---|---:|---:|---|
| `add f64` | 1e5 | 0.122 | 0.020 | 6.2× slower |
| `add f64` | 1e6 | 1.329 | 0.259 | 5.1× slower |
| `copy` | 1e6 | 1.307 | 0.130 | 10× slower |
| `sum f64` | 1e6 | 1.114 | 0.180 | 6.2× slower |
| `max f64` | 1e6 | 1.315 | 0.094 | 14× slower |
| `sum axis=0` | 1024² | 3.056 | 0.160 | 19× slower |
| `A.T` (view) | any | 1.1e-3 | 7e-5 | 15× slower (per-call binding overhead) |
| `matmul f32` | 128² | 0.339 | 0.005 | 66× slower |
| `matmul f64` | 1024² | 5.167 | 3.204 | 1.6× slower |
| `array(list)` | 1e6 | 2.755 | 18.14 | 6.6× faster |
| `svd` / `eig` / `inv` | 512² | ≈ | ≈ | parity (same Accelerate LAPACK) |
| `fft` / `random` | 1e6 | | | 1.1–1.4× slower |

### Profiling findings (evidence, not guesses)

1. **Kernel vs. end-to-end.** A C++ harness linked against `libnativpy_core.a`
   (no JS) measures `add f64` at 20 µs (1e5) and 247 µs (1e6), close to NumPy's
   20 / 259 µs. From JS the same call costs 70 µs / 2.4 ms. The loop is not
   the bottleneck for elementwise ops.
2. **Where the rest goes: deferred release plus page faults.** `memoryStats()`
   shows 1000 synchronous `add` calls leave 1002 buffers (765 MB) alive. They
   are freed only after the event loop turns (N-API finalizers run in a
   deferred task), even after `gc()`. Every fresh result buffer is therefore
   new memory from the OS. `process.resourceUsage()` measured 49 minor faults
   per call at 1e5 and 490 at 1e6, which is one per 16 KB page. The equivalent
   C++ pattern with immediate `free` has ~0.1 faults per iteration.
   `AdjustExternalMemory` additionally forces a Mark-Compact every ~9 calls.
   With it disabled, `empty(1e5)` drops from 11.6 → 2.3 µs, but `copy`/`add`
   barely move, so faults dominate, not GC. Disabling it is not an option:
   V8 would never learn about native memory.
3. **Reductions are slow in the kernel itself.** The C++ harness measures
   `sum f64` at 1e6 as 983 µs (NumPy 180) and `max` as 1280 µs (NumPy 94).
   `prepare()` always materializes a transposed `astype` copy of the input,
   then runs a scalar loop with a per-element `switch`. `sum` is a serial
   dependency chain, which is also less accurate than NumPy's pairwise sum.
4. **matmul f32 128²**: 0.34 ms vs 5 µs. Not yet profiled.
5. **Fixed per-call overhead** is ~1 µs even for views (`A.T`, `reshape`,
   `slice`), vs ~0.1 µs in NumPy. It dominates every 1K-element case.


## M11 step 1: reduction kernels (D-021)

Same environment and method as the baseline. Run with
`NATIVPY_BENCH_FILTER=reduction pnpm bench:suite && NATIVPY_BENCH_FILTER=reduction pnpm bench:suite:numpy && pnpm bench:compare`.

Across the 29 reduction cases, the geometric-mean `numpy_ms / nativpy_ms` went
from **0.237 to 0.950** (17 faster than NumPy, 12 slower). Results are
bit-identical to NumPy on contiguous float input: see D-021 and the exact
differential cases.

| Operation | Shape | before (ms) | after (ms) | NumPy (ms) | speedup |
|---|---|---:|---:|---:|---:|
| `sum f64` | 1e6 | 1.114 | 0.110 | 0.174 | 10.2× |
| `sum f32` | 1e6 | 0.994 | 0.084 | 0.156 | 11.9× |
| `mean f64` | 1e6 | 1.014 | 0.108 | 0.176 | 9.4× |
| `std f64` | 1e6 | 2.341 | 0.300 | 0.661 | 7.8× |
| `max f64` | 1e6 | 1.315 | 0.186 | 0.093 | 7.1× |
| `sum i32` | 1e6 | 0.235 | 0.207 | 0.194 | 1.1× |
| `sum axis=0` | 1024² | 3.056 | 0.154 | 0.160 | 19.8× |
| `sum axis=1` | 1024² | 0.944 | 0.116 | 0.186 | 8.1× |
| `argmax f64` | 1e6 | 4.312 | 0.363 | 0.645 | 11.9× (step 1b, below) |

Where the time went:

- **Copy removed.** `prepare()` now reads C-contiguous input in place when the
  reduced axes trail. For leading axes it sweeps columns instead of
  transposing.
- **`sum`/`mean`/`var`/`std`.** Pairwise summation (8 accumulators) breaks the
  serial add chain.
- **`max`/`min`.** A 16-lane compare-select loop vectorizes. The first attempt
  kept a `bool` NaN flag inside the loop and measured 434 µs at 1e6 in the C++
  kernel harness. Clang reported the loop as not vectorized. Accumulating `v - v`
  instead brought the harness to ~190 µs (186 µs end to end).
- **Regression found and fixed during the loop.** Hoisting the per-element `switch` out of
  the generic fold into a shared lambda made `sum i32` noticeably slower. The
  exact figure was not recorded. Dispatching the op outside the element loop
  restored 0.21 ms at 1e6 (baseline 0.235).
- **FMA contraction** is disabled for `reduce.cpp` (`-ffp-contract=off`, D-021).
  Kernel timings did not change: `sum f64` 108 µs, `max f64` 191 µs,
  `sum axis=0` 145 µs.

Remaining gaps, with profiling evidence still needed before claiming causes:

- `max f64` is 2× NumPy, which likely uses dedicated NEON `fmax` kernels. Not yet
  attempted, since the code base has no SIMD intrinsics and that would need a
  decision.
- `argmax` was 6.3× slower (copy plus scalar path). Fixed in step 1b.
- 1e3 cases are 1.2–1.7× slower, which is consistent with finding 5
  (per-call overhead).

### M11 step 1b: argmin/argmax (D-021)

Contiguous rows now use the vectorized `minmax_row` value search, then scan
for the first index. Same filter and method as step 1:

| Shape | before (ms) | after (ms) | NumPy (ms) |
|---|---:|---:|---:|
| 1e3 | — | 1.41e-3 | 1.24e-3 |
| 1e5 | — | 0.048 | 0.064 |
| 1e6 | 4.015 | 0.363 | 0.645 |

Reduction geo-mean (29 cases) went from 0.950 to **1.131** (16 faster, 13 slower).
The 18 new exact differential cases (ties, NaN, ±0, ±inf, axis=0/1) all pass.

## M11 step 2: matmul wrapper copies (D-022)

The baseline row `matmul f32 128²` (0.339 ms, "66× slower") **did not reproduce**.
A matmul-only suite run before this change gave 0.014 ms (NumPy 0.005). The cause
of the baseline outlier was not identified; the most likely candidate is interference
from earlier cases in the full run (deferred buffer release, finding 2), but that
is unverified. C++ probe at 128² f32: `cblas_sgemm` 5.6 µs, `linalg::matmul`
9.3 µs before and 4.9 µs after. `NATIVPY_BENCH_FILTER=matmul` suite (median ms):

| Case | before | after | NumPy |
|---|---:|---:|---:|
| f32 32² | 3.56e-3 | 2.40e-3 | 1.14e-3 |
| f32 128² | 0.014 | 0.011 | 5.24e-3 |
| f64 128² | 0.030 | 0.025 | 0.015 |
| f32 512² | 0.290 | 0.200 | 0.124 |
| f64 1024² | 4.993 | 4.187 | 3.405 |

Matmul geo-mean (11 cases) went from 0.561 to 0.679. At 512² f32 JS takes 200 µs while the C++
wrapper takes 107 µs; the rest is consistent with fresh-buffer page faults
(finding 2), which is not yet fixed.

## M11 step 3: GC-time buffer release (D-023)

The addon is now built with `NAPI_EXPERIMENTAL`, so Node runs our basic
finalizer synchronously during GC instead of in a later event-loop task
(finding 2). Probe (`packages/nativpy/test/memory.test.ts` scenario): 1000 sync `add` 1e5 then `gc()`: live buffers
**1001 → 1** (the stable build, checked with `-DNATIVPY_NAPI_EXPERIMENTAL=OFF`,
still shows 1001). Probe on Node 22.7, median µs per call:

| Case | stable N-API | experimental | page faults/call |
|---|---:|---:|---:|
| add 1e5 | 67–85 | 20–23 | 49 → 5 |
| copy 1e5 | 62–80 | 13 | |
| add 1e6 | 1007–1121 | 273–304 | 490 → 25 |

Node 18.20 / 20.17 runs of the same probe were ~2.5× slower than Node 22 (add 1e5 68 µs)
even with fewer faults. The synchronous-finalizer gain is therefore
Node-version dependent. Only Node 22 is measured in the suite.

Full suite (170 comparable cases, same machine, Node 22.7), geo-mean
`numpy_ms / nativpy_ms` went from **0.285 to 0.628** (60 faster, 110 slower). This includes D-021/D-022.
By category (baseline → now): reduction 0.237 → 1.179, broadcast 0.180 → 1.076,
matmul 0.319 → 0.874, elementwise 0.194 → 0.608, memory 0.167 → 0.463,
creation 0.239 → 0.398, transpose 0.196 → 0.260, slicing 0.145 → 0.153,
view 0.096 → 0.098, fft 0.757 → 0.903, linalg 0.963 → 0.995, random 0.774 → 0.925.

Regression: `creation/zeros 1e6` went from 0.049 to 0.104 ms (NumPy 0.065). The likely cause
(unverified): before, freed memory was never reused within the loop, so `calloc`
always got fresh, lazily zeroed OS pages. Now it reuses freed memory and must
zero it. `ones 1e6` improved from 0.601 to 0.176 ms for the same reason in reverse.

Still slow: 1e3 and view/slicing cases (0.1–0.3), which are dominated by fixed
per-call cost (finding 5).

