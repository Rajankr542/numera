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


## D-012 — M2 creation / M3 shape semantics — Accepted — 2026-09-29
- `full(shape, v, {dtype})` converts `v` with `np.array(v, {dtype})`, so the
  D-009 rules apply: out-of-range values raise ValueError, as in NumPy's
  OverflowError. Fractional fills into integer dtypes also go through D-009,
  which truncates (as NumPy does). Without `dtype`, the dtype is inferred
  from `v` (D-004).
- `arange(start, stop?, step?, {dtype})` defaults to int64 when all arguments
  are safe integers, otherwise float64. Values follow NumPy's `fill` rule:
  `v0 + i*(v1-v0)`, computed in the target dtype. step 0, a non-finite length,
  or a bool result with more than 2 elements raise ValueError.
- `linspace` computes in float64, floors values for integer dtypes, then
  casts (NumPy ≥ 2.0). Negative `num` raises ValueError.
- `transpose/squeeze/expandDims/swapAxes/moveAxis` always return views.
  `ravel` returns a view when reshape(-1) can; `flatten` always copies.
  Repeated axes raise ValueError; out-of-range axes raise IndexError
  (NumPy's AxisError subclasses both). `expandDims` is implemented with
  reshape, so its strides match NumPy exactly.
- `asarray(x)` returns `x` itself when it is an NDArray of the requested
  dtype.

## D-013 — Allocation and bulk-conversion fast paths — Accepted — 2026-09-29
- Zero-filled buffers come from `std::calloc` with 64 bytes of slack, aligned
  to 64 bytes by hand; the raw pointer is kept for `free`. Large calloc
  blocks get lazily zeroed pages from the OS, so `zeros` no longer touches
  every page. Non-zeroed buffers use `std::malloc` the same way. Alignment
  (64 B) and ownership (one RAII MemoryBuffer per allocation) do not change.
- A caching buffer pool is **deferred**. It would hide page-fault cost for
  `copy`, but it keeps memory alive beyond what V8 knows about and makes the
  live-buffer instrumentation harder to read. It needs a separate decision.
- `np.array(list)`: a single JS pass flattens rectangular, all-`number` input
  into a Float64Array. Native `fromFloat64` then converts it with the same
  `store_number` rule (D-009) as the per-element path. Any other input (bool,
  bigint, ragged, sparse, complex target) falls back to `fromNested`.
- `astype` has a C-contiguous flat loop that uses the same `cast_value` as
  the strided loop.

## D-014 — Broadcasting and arithmetic ufuncs (M4/M5) — Accepted — 2026-09-29
- **Broadcasting** is a single native subsystem (`native/core/broadcast.hpp`).
  `BroadcastPlan` holds the output shape and each operand's byte strides
  (0 on broadcast dims). It merges compatible adjacent dims and runs an
  inner loop over the last dim, like NumPy inner loops. Every ufunc uses it.
  An incompatible shape raises `BroadcastError`; NumPy raises `ValueError`.
  Public helpers: `broadcastShapes` and `broadcastTo` (a zero-stride view).
- **Ufunc results** are always new C-contiguous arrays. NumPy's default
  `order='K'` can return F-ordered results for F-ordered inputs; values and
  shape match, strides may not. `out=`, `where=`, `casting=` and `dtype=` are
  not supported yet.
- **Loop dtype** follows NumPy's type resolution:
  - Binary ops use `promote_types(a, b)`.
  - `divide` sends bool and int inputs to float64.
  - `power`, `mod` and `floorDivide` send bool to int8.
  - `sqrt`, `exp` and `log` send bool/int8/uint8 to float16, int16/uint16 to
    float32 and wider ints to float64.
  - For bool: `add` is logical OR and `multiply` is logical AND.
    `subtract` and `negative` on bool raise `DTypeError` (NumPy
    `TypeError`).
  - float16 is computed in float32 and rounded once, like NumPy.
- **Integer edge cases** have no undefined behaviour:
  - `add`, `subtract`, `multiply` and `power` wrap (modular).
  - `x // 0` and `x % 0` give 0. NumPy gives the same values, plus a
    RuntimeWarning.
  - `MIN // -1` gives MIN.
  - A negative integer exponent raises `ValueError`, as in NumPy.
  - Float `mod` and `floorDivide` port NumPy's `npy_divmod`.
- **JS number scalars** follow NumPy 2's weak scalar promotion (NEP 50):
  - An integer-valued number keeps an int or float array's dtype.
    Bool arrays go to int64.
  - A non-integer number keeps a float array's dtype. Int and bool arrays go
    to float64.
  - A scalar that doesn't fit the array dtype raises `ValueError` (NumPy
    `OverflowError`, D-009).
- **Complex operands** raise `NotImplementedError` because their values can't
  be read back or verified yet (D-008).
- **Transcendentals** (`sqrt`, `exp`, `log`, `power` on floats) call the
  platform libm. NumPy has its own SIMD kernels, so results may differ by a
  few ULP. The differential tests use a relative tolerance for these
  functions only. Every other op must match exactly.


## D-015 — Indexing API and semantics (M6) — Accepted — 2026-09-29
- **The API has no `a[...]` syntax, because JS can't overload it.**
  - `a.get(...specs)` works like NumPy `a[s0, s1, ...]`.
  - `a.slice(specs)` takes a list of per-axis specs (PLAN §13).
    - A flat list of 1–3 numbers or nulls is read as a single slice tuple, so
      `a.slice([0, 5])` means `a[0:5]` (PLAN §13 example).
    - For an integer index on axis 0 alone, use `get`.
  - `a.set(specs, value)` works like `a[...] = value`. A single non-tuple spec
    may be passed bare.
- **Spec encoding:**
  - A number is an integer index.
  - `null` is `:`.
  - `[start, stop, step]` is a slice; any entry may be `null`.
  - `np.ellipsis` (`"..."`) is `...`.
  - `np.newaxis` (`"newaxis"`) adds an axis.
  - A boolean is a 0-d bool index.
  - An `NDArray` (integer or bool dtype) is an advanced index.
  - Nested JS lists are not accepted as index arrays; wrap them in
    `np.array`. This avoids ambiguity with slice tuples.
- **`get` always returns an `NDArray`.** A full integer index such as
  `a.get(1, 2)` gives a 0-d copy (NumPy returns a scalar, which also does not
  share memory). `a.get(1, 2, np.ellipsis)` gives a 0-d view, as in NumPy.
  Use `item()` for a JS scalar.
- **Basic indexing** (integers, slices, newaxis, ellipsis) returns a view.
  Its shape, strides and offset match NumPy exactly.
- **Advanced indexing** implements NumPy's rules:
  - Index arrays broadcast together.
  - Integer scalars count as advanced indices when any array index is present.
  - If the advanced indices are adjacent, the broadcast dims replace them in
    place. If they are separated by a slice, ellipsis or newaxis, the
    broadcast dims go first.
  - A bool array is treated as `nonzero(mask)`.
  - A 0-d bool adds a length-1 or length-0 axis.
  - The result is always a new C-contiguous array. NumPy's strides can differ
    because of internal transposes, so differential tests compare shape,
    values and non-sharing only.
- **Assignment:**
  - `value` broadcasts to the selection shape and is cast unsafely to the
    target dtype, as in NumPy `setitem`.
  - A broadcast mismatch raises `BroadcastError`; NumPy raises `ValueError`.
  - Overlapping sources are staged through a copy.
  - With repeated advanced indices the last write wins, as in NumPy.
  - Nested JS values are converted with the target dtype. Out-of-range JS
    numbers raise `ValueError`, like `np.array(..., dtype)` (D-009).
- **Errors:**
  - An out-of-bounds index, too many indices, more than one ellipsis, a
    bool-shape mismatch, a non-integer index array, or index arrays whose
    shapes don't broadcast all raise `IndexError`.
  - A zero slice step raises `ValueError`.
- **`nonzero`, `take` and `where`:**
  - `nonzero` returns int64 arrays and raises `ValueError` on 0-d input.
  - `take(a, idx, axis?)` flattens when `axis` is omitted. Negative indices
    wrap; out-of-bounds indices raise `IndexError`. `mode=` is not supported.
  - `where(c, x, y)` uses `promote_types(x, y)` and broadcasts all three
    arguments. `where(c)` is `nonzero(c)`.
  - JS number scalars in `where` are not weak (NEP 50). They are inferred as
    int64/float64 arrays.

## D-016 — Writeable flag and read-only views — Accepted — 2026-09-29
- `NDArray` carries a `writeable` flag. It was previously reported as always
  `true`, which let `set` write through `broadcastTo` views into the source.
- Semantics (verified against NumPy 2.5.3):
  - Freshly allocated arrays and every copy (`copy`, `astype`, copying
    `reshape`/`ravel`/`flatten`, advanced-index results, ufunc outputs) are
    writeable.
  - `broadcast_to` returns a read-only view.
  - Every view (`view`/`asStrided`, basic indexing, `transpose`, `squeeze`,
    `expandDims`, view `reshape`, ...) inherits the flag of its base.
- Any write into a read-only array (`set_index`, element setters) raises
  `ValueError` "assignment destination is read-only", as NumPy does.
- `flags.writeable` reports the real flag. Setting the flag from JS
  (`setflags`) is not supported yet.

