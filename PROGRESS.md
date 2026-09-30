# PROGRESS

## 2026-09-29 — M8 linear algebra (D-018)

Done:
- C++ `native/linalg/` (a separate static library, `nativpy_linalg`; core does not depend on it):
  - `backend.hpp`: the `Backend`/`Routines<T>` interface.
  - `fallback_backend.cpp`: portable GEMM, LU, Householder QR, Jacobi eigh/SVD, and Hessenberg + shifted-QR eig.
  - `accelerate_backend.cpp`: Accelerate BLAS/LAPACK (`ACCELERATE_NEW_LAPACK`), with workspace queries and 32-bit dimension checks.
  - `linalg.{hpp,cpp}`: NDArray-level `matmul`/`dot`/`inner`/`outer`/`det`/`inv`/`solve`/`eig`/`eigh`/`svd`/`qr`/`lstsq`/`norm`. These handle batching, dtype rules and column-major conversion.
- CMake: `NATIVPY_LINALG_BACKEND=auto|accelerate|fallback`; links `-framework Accelerate` on macOS.
- Bindings: `native/bindings/linalg_binding.cpp` exposes `exports.linalg`, plus an internal `_setBackend` test hook.
- TS: `src/linalg.ts` exports top-level `matmul`/`dot`/`inner`/`outer` and the `np.linalg` namespace (plus `eigvals`, `eigvalsh`, `backend()`, `LinAlgError`).
- Tests:
  - `tests/native/test_linalg.cpp` runs every case on both backends.
  - `packages/nativpy/test/linalg.test.ts`.
  - The new differential group `linalg` (408 cases) runs on both backends.
- Verification (macOS arm64, NumPy 2.5.3):
  - `pnpm build` passes.
  - `pnpm test:native` and `pnpm test:asan` pass (50 cases, ASan+UBSan clean).
  - A `-DNATIVPY_LINALG_BACKEND=fallback` build passes and links no LAPACK symbols.
  - `pnpm test`: 66 tests pass.
  - `pnpm test:diff`: 2875 pass, including 408 linalg cases × 2 backends.
- Not done: no benchmarks yet, so no performance claims. Complex linalg is not implemented. Divergences are listed in COMPATIBILITY.md.

## 2026-09-29 — M7 reductions (D-017)

Done:
- C++ `native/core/reduce.{hpp,cpp}`:
  - `reduce` covers sum/prod/min/max/mean/var/std; `arg_reduce` covers argmin/argmax.
  - The input is transposed so reduced axes come last, then copied
    contiguously in the work dtype, and each row is folded.
  - var/std use two passes with `ddof`.
  - NaN propagates; min/max order signed zeros like NumPy.
- Bindings: `reduce` and `argReduce` in `ops_binding.cpp`.
- TS: `packages/nativpy/src/reduce.ts` adds `np.sum/prod/min/max/amin/amax/mean/var/std/argmin/argmax`,
  plus the matching NDArray methods.
- Tests:
  - `tests/native/test_reduce.cpp` (4 cases)
  - `packages/nativpy/test/reduce.test.ts` (6 tests)
  - differential group `reduce` (813 cases)
- Divergences (COMPATIBILITY.md):
  - Float sums are sequential, where NumPy sums pairwise, so float sum/prod/mean/var/std are compared with a tolerance.
  - Results are always C-contiguous.
- Verification: `pnpm build`, `pnpm typecheck`, `pnpm test:native` (45 cases),
  `pnpm test:asan` pass; `pnpm test` 60 tests; `pnpm test:diff` 2058 cases.

## 2026-09-29 — Read-only views (D-016)

Fixed a bug where `set` wrote through `broadcastTo` views into the source
array. NumPy raises `ValueError: assignment destination is read-only`.
- C++: `NDArray` has a `writeable` flag. `view()` inherits it, `broadcast_to`
  clears it, and allocations and copies are writeable. `set_index` and the
  element setters call `check_writeable()`.
- Binding: `flags.writeable` reports the real flag (it was hard-coded `true`).
- Tests: a C++ case in `test_indexing.cpp`, a vitest case, and a new
  differential group `writeable` (27 op chains). Each chain checks the layout,
  `flags.writeable`, and whether a write succeeds or raises NumPy's message.
- COMPATIBILITY.md: verified rows for indexing and `flags.writeable`.
- Verification: `pnpm build`, `pnpm typecheck`, `pnpm test:native`,
  `pnpm test:asan` pass; `pnpm test` 54 tests; `pnpm test:diff` 1245 cases.

## 2026-09-29 — M6 indexing

Done:
- C++ `native/core/indexing.{hpp,cpp}`:
  - `slice_indices`, a port of Python `slice.indices`.
  - `get_index` builds basic views. It then gathers advanced indices using
    NumPy placement rules: adjacent vs. separated indices, bool arrays turned
    into nonzero, and 0-d bools.
  - `set_index` broadcasts the value, casts unsafely and stages overlapping
    sources through a copy.
  - `nonzero`, `take` and `where`.
- Bindings: `getIndex`, `setIndex`, `nonzero`, `take` and `where` in
  `ops_binding.cpp`, plus `NDArrayWrap::is_ndarray`.
- TS:
  - `NDArray.get`, `.slice` and `.set`.
  - `np.newaxis` and `np.ellipsis`.
  - `np.nonzero`, `np.take` and `np.where`.
  - Semantics are in D-015.
- Tests:
  - `tests/native/test_indexing.cpp`.
  - `packages/nativpy/test/indexing.test.ts`, which covers the PLAN §9/§13/§57
    examples.
  - A differential `indexing` group with 103 cases: 63 getitem expressions
    and 40 setitem cases across 4 dtypes, including error classes.
- Verification (macOS arm64, Node 22.7, NumPy 2.5.3):
  - `pnpm build` and `pnpm typecheck` are clean.
  - `pnpm test:native` passes, and `pnpm test:asan` passes (ASan+UBSan clean).
  - `pnpm test`: 53 tests pass.
  - `pnpm test:diff`: 1218 cases pass.
- Not measured: indexing performance. The gather precomputes a per-element
  offset table (O(n) int64 memory); no claims are made.

## 2026-09-29 — M4 arithmetic ufuncs, M5 broadcasting

Done:
- C++ `native/core/broadcast.{hpp,cpp}`: `broadcast_shapes`, `broadcast_to`
  (zero-stride view), `BroadcastPlan` with dim coalescing and an inner-loop
  runner. `native/core/ufunc.{hpp,cpp}` + `ufunc_kernels.hpp`: 7 binary and 5
  unary ops, NumPy loop-dtype resolution, and contiguous/scalar fast paths.
  Float `mod`/`floorDivide` port `npy_divmod`. Semantics are in D-014.
- TS `ufunc.ts`: the ops, NEP 50 number scalars, nested-list operands,
  `broadcastShapes`, `broadcastTo`.
- Tests: `tests/native/test_ufunc.cpp`, `packages/nativpy/test/ufunc.test.ts`,
  and a differential `ufuncs` group (231 cases).
- Benchmarks: ufunc cases were added to `pnpm bench` / `pnpm bench:numpy`.
  Numbers are in PERFORMANCE.md, not profiled yet. `add` is 2.8× slower than
  NumPy at 1M elements.
- Verification (macOS arm64, Node 22.7, NumPy 2.5.3):
  - `pnpm build` and `pnpm typecheck` are clean.
  - `pnpm test:native` passes, and `pnpm test:asan` passes (ASan+UBSan clean).
  - `pnpm test`: 47 tests pass.
  - `pnpm test:diff`: 1115 cases pass.

## 2026-09-29 — M2 creation, M3 shape, first measured optimizations

Done:
- C++ `native/core/creation.{hpp,cpp}`: `full`, `ones`, `arange` (NumPy fill
  semantics), `linspace`, `eye`. `native/core/shape_ops.{hpp,cpp}`:
  `transpose`, `squeeze`, `expand_dims`, `swapaxes`, `moveaxis`, `ravel`,
  `flatten`, `normalize_axes`. Bindings are in `native/bindings/ops_binding.cpp`.
- TS: `asarray`, `ones`, `full`, `arange`, `linspace`, `eye`, `identity`,
  `zerosLike`/`onesLike`/`emptyLike`/`fullLike`, `shape.ts` functions, and the
  `NDArray` methods `transpose`, `T`, `squeeze`, `swapAxes`, `ravel`, `flatten`.
  Semantics are recorded in D-012.
- Performance (D-013, PERFORMANCE.md): calloc-backed zero buffers, a
  contiguous `astype` loop, and bulk `fromFloat64` for `np.array(list)`.
  Median at 1M float64 elements: `array(list)` 57.4 → 3.2 ms, `astype` 2.31 →
  0.40 ms, `zeros` 0.64 → 0.012 ms (allocation only). Corrected the wrong
  cause notes in PERFORMANCE.md.
- Verification (macOS arm64, Node 22.7, NumPy 2.5.3):
  - `pnpm build` passes.
  - `pnpm test:native` and `pnpm test:asan` pass (ASan+UBSan clean).
  - `pnpm test`: 42 tests pass.
  - `pnpm test:diff`: 884 cases pass, including new `ranges` (219) and
    `shape_ops` (64) groups.

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
