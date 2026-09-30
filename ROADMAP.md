# ROADMAP

Milestones follow `docs/plan/PLAN.md` §86. Status: ✅ done · 🟡 partial · ⬜ not started.

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| 0 | Project infrastructure | 🟡 | CMake, cmake-js addon, TS (strict, ESM), vitest, C++ test harness, ASan/UBSan, NumPy harness, CI workflow. No prebuilt binaries yet. |
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
| 11–14 | → release | 🟡 | M11 in progress: benchmark suite and baseline done. Reduction kernels done (D-021): reduction geo-mean went from 0.237 to 0.950 of NumPy speed. See PERFORMANCE.md. |

## Next steps
1. M11 per PLAN §86. Next targets, from PERFORMANCE.md findings:
   - Buffer release and page faults for elementwise ops. Needs a decision first: pool, dispose or post-finalizer (D-013).
   - Per-call binding overhead.
   - `max` at 2× NumPy (NEON would need a decision).
2. FFT follow-ups: `rfftn`/`irfftn`/`rfft2`/`irfft2`, `hfft`/`ihfft`, `fftshift`/`ifftshift` (D-020). Benchmark against NumPy before making any FFT performance claims.
3. Random follow-ups: `choice(p=...)`, broadcast array parameters, more distributions (D-019).
4. Complex element conversion (D-008). This unblocks complex ufuncs and complex linalg.
5. Benchmark matmul/solve/svd across the Accelerate and fallback backends and NumPy before making any performance claims (PERFORMANCE.md).
6. Profile ufunc and `copy` costs (destination allocation vs. loop, PERFORMANCE.md) before optimizing.
7. Buffer pool for `copy`-heavy workloads (deferred, D-013). It needs its own decision entry.
8. Pairwise summation for float16, cast (`dtype=`) and middle-axis reductions, if NumPy's buffered order is needed there too (D-021 covers contiguous f32/f64).
