# PROGRESS

## 2026-09-29 — PLAN §92 first task: end-to-end native NDArray

Done:
- C++ core (`native/core`): `MemoryBuffer`, `DType` + promotion table, shape/stride
  utilities, `NDArray` with views, copy, astype (unsafe casting), NumPy
  `_attempt_nocopy_reshape`, extent-based `may_share_memory`.
- Node-API bindings (`native/bindings`): `NDArray` wrapper; C++ errors mapped to
  typed JS errors (D-006).
- TypeScript package (`packages/nativpy`): `np.array/empty/zeros/fromTypedArray`,
  dtype objects, `promoteTypes`, `mayShareMemory`, `lib.stride_tricks.asStrided`.
- Tests, verified locally (macOS arm64, Node 22.7, NumPy 2.5.3):
  - `pnpm test:native`: C++ unit tests pass.
  - `pnpm test:asan`: C++ unit tests pass under ASan+UBSan, no reports.
  - `pnpm test`: 34 vitest tests pass.
  - `pnpm test:diff`: 601 NumPy differential cases pass (creation, astype, views,
    reshape, strided reshape, promote_types).
  - `pnpm typecheck` clean.

Issues the differential tests caught and I fixed (D-011): allocation strides for
empty arrays, reshape copying when NumPy returns a view, and `mayShareMemory`
semantics for empty or disjoint views.

Known gaps: see COMPATIBILITY.md. CI workflow is written but has not run yet
(no remote).
