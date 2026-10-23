# ROADMAP

Milestones follow `docs/plan/PLAN.md` §86. Status: ✅ done · 🟡 partial · ⬜ not started.

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| 0 | Project infrastructure | 🟡 | CMake, cmake-js addon, TS (strict, ESM), vitest, C++ test harness, ASan/UBSan, NumPy harness, CI workflow. npm packaging + `pnpm release` added (D-026). Published as `@cyfora/numera@1.0.0` on npm (D-028). Releases are manual via `pnpm release`; the GitHub Actions release workflow was removed (D-031, supersedes D-029's automatic channels). |
| 1 | NDArray | ✅ | MemoryBuffer (aligned, refcounted), DType (14 dtypes), shape, byte strides, offset, ownership, bounds-checked views, NumPy nocopy reshape. |
| 2 | Creation | ✅ | PLAN M2 list complete: `array`, `asarray`, `zeros`, `ones`, `empty`, `full`, `arange`, `linspace`, `eye` (+ `identity`, `*Like`, `fromTypedArray`). D-012. |
| 3 | Shape | ✅ | PLAN M3 list complete: `reshape`, `transpose`/`.T`, `squeeze`, `expandDims`, `ravel`, `flatten` (+ `swapAxes`, `moveAxis`). All views except `flatten` / non-viewable `ravel`. `concatenate`/`stack` are not in PLAN M3; they are not started. |
| 4 | Arithmetic | ✅ | PLAN M4 list complete: `add`, `subtract`, `multiply`, `divide`, `power`, `mod`, `abs`, `sqrt`, `exp`, `log` (+ `floorDivide`, `negative`). All 12 real dtypes, NEP 50 scalars. D-014. |
| 5 | Broadcasting | ✅ | Rules (`broadcastShapes`), iterator (`BroadcastPlan`, dim coalescing), tests (C++, vitest, differential); `broadcastTo` views. |
| 6 | Indexing | ✅ | `get`/`slice`/`set`: integer, slice (negative/reverse), ellipsis, newaxis, integer-array, boolean, mixed advanced indexing; `nonzero`, `take`, `where`. D-015. |
| 7 | Reductions | ✅ | PLAN M7 list complete: `sum`, `mean`, `min`, `max`, `argmin`, `argmax`, `prod`, `std`, `var` with `axis` (int/list/none), `keepdims`, `dtype`, `initial`, `ddof`; NDArray methods. D-017. |
| 8 | Linear algebra | ✅ | `matmul`, `dot`, `inner`, `outer`; `linalg.det`/`inv`/`solve`/`eig`/`eigh`/`eigvals`/`eigvalsh`/`svd`/`qr`/`lstsq`/`norm`. Batched. Accelerate backend on macOS, portable fallback elsewhere; both tested. D-018. |
| 9 | Random | ✅ | Bit-exact with NumPy: `defaultRng` (PCG64 + SeedSequence) Generator and the legacy MT19937 `RandomState`/global functions: `rand`, `randn`, `random`, `uniform`, `normal`, `randint`/`integers`, `choice`, `shuffle`, `permutation`. Scalar parameters only. D-019. |
| 10 | FFT | ✅ | PLAN M10 list complete: `np.fft.fft`/`ifft`/`fft2`/`ifft2`/`fftn`/`ifftn`/`rfft`/`irfft` (+ `fftfreq`/`rfftfreq`), backed by vendored pocketfft (NumPy's pinned commit). Verified against NumPy within tolerance. Not in PLAN M10 and not started: `rfftn`/`irfftn`/`hfft`/`fftshift`/`out=`. D-020. |
| 11–14 | → release | 🟡 | M11 in progress: benchmark suite and baseline done. D-021 reductions (incl. argmin/argmax), D-022 matmul copies and D-023 GC-time buffer release are done. Suite geo-mean went from 0.285 to 0.628 of NumPy speed. See PERFORMANCE.md. |

## NumPy parity (D-032, runs before the rest of M11–M14)
The full function list and order are in `docs/plan/NUMPY_PARITY.md`, milestones P0–P15.
Every function needs a native kernel, unit and edge tests, NumPy differential cases,
and a benchmark case in both suites.

| Milestone | Status |
|---|---|
| P0 coverage tooling (`pnpm api:coverage` / `api:check` in CI) | ✅ API coverage is 16.0% (170/1060) after P1 (14.9%, 158/1060, at P0), and every implemented callable is benchmarked in both suites |
| P1 complex numbers (D-008, D-033) | ✅ Complex64/complex128 work in every existing path: conversion, ufuncs plus `real`/`imag`/`conj`/`angle`/`iscomplex*`/`isreal*`, reductions (D-034), `matmul`/`dot`/`inner`/`outer` (D-035–D-037), all of `linalg` on both backends (D-038–D-044), and FFT with `np.Complex` input. Verified by the differential groups `complex_ufuncs`, `complex_reductions`, `complex_matmul`, `complex_linalg` and `complex_fft`; benchmarked in both suites. Complex `sort` waits for `sort` itself (P9); complex trig and comparisons wait for P4/P5. |
| P2 ufunc machinery | 🟡 built, unverified | P2-1…P2-10 all built (D-045–D-054). P3-1 built (D-055). `reduce`/`accumulate`/`outer`/`reduceat`/`at` and `errstate`/`seterr` are build-first; V-P2 (NumPy differential + bench) not yet done. API coverage 16.1% (171/1060) |
| P3 layout + core array API | 🟡 built, unverified | P3-1 ✅ (order= for creation/reshape/copy). P3-2…P3-5 (NDArray methods, iterators, dtype introspection, printing) not yet built — see "Remaining build work" in `TASK_SLICES.md`. |
| P4 math ufuncs | 🟡 built, unverified | All 75 entries built (D-070–D-079). V-P4 (differential + bench) not yet done. |
| P5 comparison/logic/bitwise | 🟡 built, unverified | All 38 entries built (D-080–D-089). V-P5 not yet done. |
| P6 array manipulation | 🟡 built, unverified | All 46 entries built (D-090–D-099). V-P6 not yet done. |
| P7 creation + grids | 🟡 built, unverified | All 24 entries built (D-100–D-109). V-P7 not yet done. |
| P8 indexing extras | 🟡 built, unverified | All 24 entries built (D-110–D-119). V-P8 not yet done. |
| P9 sorting/searching/sets | 🟡 built, unverified | All 23 entries built (D-120–D-129). V-P9 not yet done. |
| P10 statistics + NaN reductions | 🟡 built, unverified | All 39 entries built (D-130–D-139). V-P10 not yet done. |
| P11 linalg completion | 🟡 built, unverified | All 29 entries built (D-140–D-149). V-P11 not yet done. |
| P12 FFT completion | 🟡 built, unverified | All 8 entries built (D-150–D-159). V-P12 not yet done. |
| P13 random completion | 🟡 built, unverified | All 115 entries built (D-160–D-169). V-P13 not yet done. |
| P14 I/O + extra dtypes | 🟡 built, unverified | P14-1…P14-7 built (D-170–D-173); P14-8…P14-11 (datetime, str_, structured, rec) not yet built. V-P14 not yet done. |
| P15 submodules | 🟡 built, unverified | All 23 entries built (D-180–D-189). V-P15 not yet done. |
| P16A constants/vectorize | 🟡 built, unverified | All 7 entries built (D-191–D-199). V-P16A not yet done. |
| P16B np.ma | 🟡 built, unverified | 206 entries built (D-200–D-209); 6 names still missing (applyAlongAxis, applyOverAxes, compressed, frombuffer, fromflex, fromfunction). V-P16B not yet done. |
| P16C np.strings + np.char | 🟡 built, unverified | 99 entries built (D-210–D-213). V-P16C not yet done. |
| P16D datetime64 | 🟡 built, unverified | 8 entries built (D-220–D-229). V-P16D not yet done. |
| P16E np.rec + cleanup | 🟡 built, unverified | 5 entries built (D-230–D-239). V-P16E not yet done. |

**Overall API coverage: 99.1% (1045/1054) built. ~821 callables still on bench-exempt lists (unverified against NumPy).**

Work is done in small slices (see `docs/plan/TASK_SLICES.md`). **Current focus: V phase** — add NumPy differential tests and benchmarks for each built milestone, highest-value first (V-P4 → V-P5 → V-P9 → …).

## Next steps
1. **V phase** (see `TASK_SLICES.md`): add differential groups and benchmarks for P2-11 through P16E in priority order. Fix any correctness bugs found. Clear the 821 bench-exempt entries one V slice at a time.
2. **Remaining build work** (P3-2…P3-5, P14-8…P14-11, `multivariateNormal`, 6 ma gaps) — see `TASK_SLICES.md`.
3. **M11 performance** (after V phase establishes a full baseline):
   - Per-call binding overhead: ~700 ns per result (ObjectWrap construction). Evaluate `napi_create_external` handles (needs a decision).
   - `zeros 1e6` regression after D-023 (calloc on reused memory).
   - `max f64` is 2× NumPy (NEON `fmax` kernel, needs a decision).
   - Node 18/20 are ~2.5× slower than Node 22; profile before making version claims.
4. **Branch cleanup**: all 18 `pNN` branches are merged (0 commits ahead of main). Delete them and their worktrees after confirming nothing is needed.
