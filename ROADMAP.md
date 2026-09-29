# ROADMAP

Milestones follow `docs/plan/PLAN.md` §86. Status: ✅ done · 🟡 partial · ⬜ not started.

| # | Milestone | Status | Notes |
|---|-----------|--------|-------|
| 0 | Project infrastructure | 🟡 | CMake, cmake-js addon, TS (strict, ESM), vitest, C++ test harness, ASan/UBSan, NumPy harness, CI workflow. No prebuilt binaries yet. |
| 1 | NDArray | ✅ | MemoryBuffer (aligned, refcounted), DType (14 dtypes), shape, byte strides, offset, ownership, bounds-checked views, NumPy nocopy reshape. |
| 2 | Creation | 🟡 | `array`, `empty`, `zeros`, `fromTypedArray` done. `asarray`, `ones`, `full`, `arange`, `linspace`, `eye` not started. |
| 3 | Shape | 🟡 | `reshape` (view when possible, D-011). transpose/squeeze/expand_dims/concatenate/stack not started. |
| 4–14 | Arithmetic → release | ⬜ | Blocked on M2/M3, per PLAN §92 ("only after this works reliably"). |

## Next steps
1. Finish M2 creation functions (`ones`, `full`, `arange`, `linspace`, `eye`, `asarray`), each with differential cases.
2. Add basic slicing (`a.get(...)`) so views can be made without `asStrided`.
3. M3 shape functions.
4. Complex element conversion (currently `NotImplementedError`, D-008).
