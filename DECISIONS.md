# Architectural Decision Log

Format: ID — title — status — date. Newest last.

## D-001 — Node-API via `node-addon-api` — Accepted — 2026-09-29
Use Node-API (PLAN §6) through the header-only C++ wrapper `node-addon-api`,
`NAPI_VERSION=8`, C++ exceptions enabled. Raw C++ exceptions are caught in the
binding layer and converted into typed JS errors (D-006). No direct V8 usage.

## D-002 — Build: CMake + Ninja driven by `cmake-js` — Accepted — 2026-09-29
The addon is built from the root `CMakeLists.txt` with `cmake-js`. The same
CMake project builds a static core library `nativpy_core` (no Node dependency)
plus C++ unit tests, so the core can be tested/sanitized without Node.
CMake/Ninja for development are installed into `.venv` from PyPI
(`python/requirements.txt`); end users will get prebuilt binaries later (M13).

## D-003 — Memory model — Accepted — 2026-09-29
- `MemoryBuffer` owns a 64-byte-aligned allocation (RAII, non-copyable).
- `NDArray` holds `std::shared_ptr<MemoryBuffer>`, `DType`, `shape`,
  `strides` and `offset`. **Strides and offset are in bytes**, matching NumPy
  (`ndarray.strides`), so negative/zero strides and non-contiguous views are
  representable.
- Views share the buffer (`shared_ptr` refcount); the buffer lives as long as
  any view. JS wrappers release their reference on GC finalization.
- Every view constructor validates that all reachable byte addresses lie in
  `[0, buffer.size)`; out-of-bounds views are rejected (safety over NumPy's
  unchecked `as_strided`).

## D-004 — Default dtype inference for `np.array` — Accepted — 2026-09-29
Mirror NumPy: all-boolean input → `bool`; all JS numbers integral (and within
int64 safe range) → `int64`; otherwise `float64`; `bigint` input → `int64`.
Empty input → `float64`. Because JS has one number type, a number counts as an
integer only if `Number.isSafeInteger(v)` and it is not `-0`; so `[2**60]` and
`[-0]` infer `float64` (NumPy gives int64 for the Python int `2**60`; this is
a documented divergence). Explicit `dtype` always wins. `bigint` values beyond
int64 require an explicit `dtype: "uint64"` (NumPy would infer uint64).

## D-005 — 64-bit integer conversion to JS — Accepted — 2026-09-29
`toArray()` returns JS `number` for all real dtypes, including int64/uint64
(exact only up to 2^53; documented lossy). `toTypedArray()` returns
`BigInt64Array`/`BigUint64Array` for exact access. float16 is stored as IEEE
binary16 bits and converted in C++; `toTypedArray()` for float16 returns a
`Uint16Array` of raw bits (Node 22 has no stable `Float16Array`).

## D-006 — Error model — Accepted — 2026-09-29
C++ throws `nativpy::Error` subclasses carrying an `ErrorKind`
(Shape, DType, Index, Broadcast, Value, Memory, NotImplemented). The binding
layer converts them into JS errors with `name` set to the kind and a `code`
property; TS re-exports matching `instanceof`-able classes (`errors.ts`).

## D-007 — Package / module format — Accepted — 2026-09-29
Single public package `packages/nativpy` (PLAN §49), ESM output. CommonJS is
deferred (PLAN §50 "where practical"). The addon loader resolves, in order:
`NATIVPY_ADDON_PATH`, then the repo-local `build/Release/nativpy.node`.
Prebuilt platform packages come in M13.

## D-008 — Complex dtypes in M1 — Accepted — 2026-09-29
complex64/complex128 exist in the dtype system (size, alignment, promotion
metadata) and can be allocated/zero-filled, but element conversion to/from JS
raises `NotImplementedError` until a JS complex representation is decided.

## D-009 — Value conversion in `np.array(data, {dtype})` — Accepted — 2026-09-29
JS numbers have no int/float distinction, so conversion follows NumPy's
behaviour for Python scalars: for integer target dtypes, integral values out
of the dtype range raise `ValueError` (NumPy: `OverflowError`), NaN/±Inf raise
`ValueError`, and fractional values are truncated toward zero (float cast).
`bigint` values are range-checked the same way. Array-to-array `astype` uses
NumPy "unsafe" casting with the deterministic float→int rule in `cast.hpp`
(NaN→0, saturate for 32/64-bit, wrap via int32 for 8/16-bit — equal to NumPy
on arm64; NumPy's result is platform-dependent/UB for these inputs).

## D-010 — Minimal M1 public surface — Accepted — 2026-09-29
Public in M1: `np.array`, `np.empty`, `np.zeros`, dtype objects,
`np.promoteTypes`, `np.mayShareMemory` (NumPy
`may_share_memory` bounds semantics, see D-011), and
`np.lib.stride_tricks.asStrided(a, shape, strides, offset?)`; the last one is
bounds-checked, unlike NumPy. NDArray exposes `shape, strides, ndim, size,
dtype, itemSize, nbytes, flags, reshape, copy, astype, item, toArray,
toTypedArray, toString`. `toTypedArray()` always returns a new C-order copy.

## D-011 — Stride/overlap semantics found by differential tests — Accepted — 2026-09-29
Found by the NumPy differential tests (NumPy 2.5.3):
- **Allocation strides**: a newly allocated array (`empty/zeros/copy/astype`)
  with any zero-size dimension gets all-zero strides, matching NumPy.
  `c_contiguous_strides` (used by reshape) keeps the `max(dim,1)` rule.
- **Reshape**: returns a view whenever the existing strides allow it (NumPy's
  `_attempt_nocopy_reshape` algorithm, C order). It copies only when no view
  is possible.
- **`mayShareMemory`**: NumPy `may_share_memory` bounds semantics. Returns
  true iff the two arrays use the same buffer, both are non-empty, and their
  byte extents `[low, high)` overlap. This supersedes the same-buffer wording
  in D-010.
