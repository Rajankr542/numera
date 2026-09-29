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

Treat these as the baseline for the M4 kernel work (PLAN §65). Re-run and
update this file after each optimization, and profile before claiming causes.
