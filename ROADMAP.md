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
| 9–14 | → release | ⬜ | M9 is next. |

## Next steps
1. M9 per PLAN §86.
2. Complex element conversion (D-008). This unblocks complex ufuncs and complex linalg.
3. Benchmark matmul/solve/svd across the Accelerate and fallback backends and NumPy before making any performance claims (PERFORMANCE.md).
4. Profile ufunc and `copy` costs (destination allocation vs. loop, PERFORMANCE.md) before optimizing.
5. Buffer pool for `copy`-heavy workloads (deferred, D-013). It needs its own decision entry.
6. Pairwise float summation for reductions, if exact NumPy float sums are needed (D-017).
