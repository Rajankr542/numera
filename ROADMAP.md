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
| 6–14 | Indexing → release | ⬜ | M6 indexing is next. |

## Next steps
1. M6 indexing: integer, slicing, negative, reverse, ellipsis, boolean, advanced.
2. Profile ufunc and `copy` costs (destination allocation vs. loop, PERFORMANCE.md) before optimizing.
3. Complex element conversion (currently `NotImplementedError`, D-008), which unblocks complex ufuncs.
4. Buffer pool for `copy`-heavy workloads (deferred, D-013). It needs its own decision entry.
