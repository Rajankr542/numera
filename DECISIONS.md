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
Prebuilt platform packages come in M13 (superseded in part by D-026: bundled prebuilds).

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
- **Ufunc results** were always new C-contiguous arrays until D-050. NumPy's
  result layout (`order=`, default 'K') is now matched on the 12 element-wise
  ufuncs (see D-050). Native `out=` landed in D-046, `dtype=`/`casting=` in
  D-048 and `where=` in D-049.
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


## D-017 — Reductions (M7) — Accepted — 2026-09-29
- C++ `native/core/reduce.{hpp,cpp}` provides `sum`, `prod`, `mean`, `min`,
  `max`, `var`, `std`, `argmin` and `argmax`.
- Options:
  - `axis`: none, an int, or a list.
  - `keepdims`.
  - `dtype`: sum/prod/mean/var/std only.
  - `initial`: sum/prod/min/max only.
  - `ddof`: var/std only.
  - `argmin`/`argmax` take a single int axis or none.
- Result dtypes follow NumPy 2.5.3:
  - sum/prod: bool and signed ints give int64, unsigned ints give uint64,
    floats keep their dtype.
  - mean/var/std: bool and ints give float64, floats keep their dtype.
    float16 accumulates in float32.
  - min/max keep the input dtype. argmin/argmax give int64.
- Results are always an `NDArray`, 0-d when every axis is reduced; use
  `item()` for a JS scalar.
- Floats are summed sequentially in the accumulator type. NumPy uses pairwise
  summation, so float sum/prod/mean/var/std can differ by a few ULP. The
  differential tests use a relative tolerance for those five only; integer
  results and min/max/arg* must match exactly.
- float16 sum/prod accumulate in float32 and round once at the end.
- Signed zero: min returns -0.0 and max returns +0.0 regardless of order
  (matches NumPy).
- Results are always C-contiguous (like D-014); NumPy keeps the input memory
  order for non-contiguous inputs. Values, dtype and shape match.
- JS API: `np.sum/prod/min/max/amin/amax/mean/var/std/argmin/argmax` and the
  matching NDArray methods, with an options object. `var` is a reserved word,
  so the named ESM export is `variance` (`np.var` on the default export).
- NaN propagates through min/max. argmin/argmax return the index of the
  first NaN. Ties return the first index.
- Errors:
  - Empty min/max without `initial`, and empty argmin/argmax, raise
    `ValueError`.
  - An out-of-range axis raises `IndexError` (D-012); a repeated axis raises
    `ValueError`.
  - Empty mean/var return NaN, with no warning.
  - Complex input raises `NotImplementedError` (D-008).

## D-018 — Linear algebra backend and semantics (M8) — Accepted — 2026-09-29
- Code in `native/linalg/`, behind the `LinalgBackend` interface
  (`backend.hpp`, PLAN §20/§45). `native/core` does not depend on it.
  - macOS: `AccelerateBackend` (Accelerate BLAS/LAPACK, `ACCELERATE_NEW_LAPACK`).
  - Everywhere else: `FallbackBackend`, portable C++ (GEMM loop, partial-pivot
    LU, Householder QR, Jacobi eigh/SVD, Hessenberg + shifted QR for eig).
  - OpenBLAS/MKL backends are M11 work.
  - CMake option `NATIVPY_LINALG_BACKEND=auto|accelerate|fallback`.
  - The C++ tests run both backends. JS tests switch backends through the
    internal `np.linalg._setBackend("fallback" | "default")` hook (not public
    API) and run the differential cases on both.
- `matmul`/`dot`/`inner`/`outer`:
  - Result dtype comes from `promote_types`. Shapes follow the NumPy gufunc
    `(n?,k),(k,m?)->(n?,m?)` with broadcast batch dims; `dot` sums over the
    last axis of `a` and the second-to-last axis of `b`.
  - float32/float64 use backend GEMM. Integer/bool/float16 use an exact loop
    in the result dtype (integers wrap, bool is an OR of ANDs, float16
    accumulates in float32).
  - A shape mismatch raises `ShapeError` (NumPy: `ValueError`). A mismatch in
    the batch dimensions raises `BroadcastError` (D-014).
- `det`/`inv`/`solve`/`eig`/`eigh`/`svd`/`qr`/`lstsq`:
  - float32 stays float32; bool and integer inputs compute in float64.
  - float16 raises `DTypeError` (NumPy: `TypeError`).
  - Leading dimensions are batched.
  - Singular matrices, non-convergence, NaN/inf input to eig, and
    non-square or <2-D input raise `LinAlgError` (new `ErrorKind::LinAlg`,
    matching `numpy.linalg.LinAlgError`).
- `eig` always returns complex eigenvalues and eigenvectors (NumPy 2):
  complex64 for float32 input, otherwise complex128. Read them with
  `toTypedArray()` (interleaved re/im) until complex element conversion lands
  (D-008).
- Decompositions return plain objects: `{eigenvalues, eigenvectors}`,
  `{U, S, Vh}`, `{Q, R}`, `{x, residuals, rank, s}`.
  - Signs and phases of eigenvectors and singular vectors depend on the
    backend (as in LAPACK). Tests check reconstruction and orthogonality,
    not raw vectors.
- `norm`:
  - `ord`: none, 'fro', 'nuc', ±Infinity, ±1, ±2, or any p for vectors.
  - `axis`: an int or a pair; `keepdims` is supported.
  - Result dtype: float32/float16 inputs keep their dtype; everything else
    gives float64.
- Complex *input* to linalg raises `NotImplementedError` (D-008). Superseded
  by D-035–D-044: every linalg function now accepts complex input.

## D-019 — Random module (M9) — Accepted — 2026-09-29
- Code lives in `native/random/` as the `nativpy_random` static lib, which
  depends on core. The algorithms are ported from the NumPy 2.x sources
  (BSD-3):
  - `SeedSequence`.
  - `PCG64`: XSL-RR 128/64, needs `__uint128_t`.
  - `MT19937`: legacy integer and array seeding.
- Goal: bit-exact streams vs. NumPy for the implemented methods. Every
  claimed method is checked by differential tests.
- `np.random.defaultRng(seed)` is `Generator(PCG64(SeedSequence(seed)))`.
  - The seed can be a non-negative integer (number or bigint) or an array of
    them.
  - An omitted seed means OS entropy (128 bits from `node:crypto`
    `randomFillSync`; the `globalThis.crypto` global is absent on Node 18).
- `Generator` methods use NumPy's algorithms:
  - `random`: 53-bit double, or 24-bit float32.
  - `uniform`.
  - `standardNormal` / `normal`: ziggurat.
  - `integers`: Lemire rejection, buffered 8/16-bit draws, `endpoint`.
  - `choice`, `shuffle`, `permutation`.
- The legacy global `np.random.*` functions use a `RandomState(MT19937)`
  singleton, as in NumPy:
  - `seed`, `rand`, `randn`, `random` / `randomSample`, `uniform`.
  - `normal`: polar Box–Muller with the cached second gauss.
  - `randint`: masked rejection, default dtype int64.
  - `choice`, `shuffle`, `permutation`.
- Signatures follow NumPy's positional order, e.g.
  `normal(loc?, scale?, size?)`. A single options object also works:
  `normal({loc, scale, size})`. Without `size`, a scalar comes back as a JS
  number (int64/uint64 follow D-005).
- In M9, distribution parameters are scalars only. Array (broadcast)
  parameters raise `NotImplementedError`.
- `shuffle` works in place on a writeable NDArray: axis 0 for `RandomState`,
  any `axis` for `Generator`. `permutation` returns a new array.
- Invalid arguments raise `ValueError` with NumPy's messages where practical.
- Float32 output is only available through `standardNormal`, as in NumPy.
  Asking native `normal` for float32 with a non-default loc/scale raises
  `ValueError`; it never silently drops the parameters.
- An unseeded `RandomState` is array-seeded with 624 words of OS entropy. The
  stream isn't reproducible, the same as in NumPy.

## D-020 — FFT module (M10) — Accepted — 2026-09-29
- Backend: vendored header-only pocketfft (C++ branch, BSD-3), pinned to
  commit `33ae5dc9`, which is the one NumPy 2.x vendors. It lives in
  `third_party/pocketfft/` with its LICENSE.
  - Built with `POCKETFFT_NO_MULTITHREADING`, like NumPy (threading is M11).
  - Code is in `native/fft/` as the `nativpy_fft` static lib, which depends
    on core.
  - The 1-D loops mirror NumPy's `_pocketfft_umath.cpp`: per-lane plan exec
    with zero-pad or truncate to `n`, and FFTpack packing for rfft/irfft.
    Results are therefore expected to match NumPy to within rounding.
- API:
  - `np.fft.fft/ifft/rfft/irfft(a, n?, axis=-1, norm?)`.
  - `fft2/ifft2(a, s?, axes=[-2,-1], norm?)`.
  - `fftn/ifftn(a, s?, axes?, norm?)`.
  - `fftfreq(n, d=1)` and `rfftfreq(n, d=1)`.
  - Parameters can be passed positionally or as a trailing options object
    `{n|s, axis|axes, norm}`.
  - Multi-axis transforms apply 1-D transforms from the last axis in `axes`
    to the first, like NumPy's `_raw_fftnd`.
- Dtypes (NumPy 2):
  - float16/float32/complex64 compute in float32; everything else computes
    in float64.
  - `fft`/`ifft`/`rfft` return complex64 or complex128.
  - `irfft` returns the real compute dtype (float16 input → float16, as
    NumPy does).
  - `rfft` on complex input raises `DTypeError` (NumPy: `TypeError`).
- `norm`: `"backward"` (default, also null/undefined), `"ortho"`,
  `"forward"`. The factor is computed as NumPy does, in
  `result_type(a.real.dtype, 1.0)`. For float16 input, `n`, the `sqrt` and
  the reciprocal are therefore each rounded to float16, and the transform
  itself still runs in float32. Any other value raises `ValueError`.
- Validation order follows NumPy: an explicit `n < 1` is reported before a
  bad axis.
- `fftfreq`/`rfftfreq`:
  - Return float64.
  - `n == 0` or `d == 0` raises `ValueError` (NumPy: `ZeroDivisionError`).
  - `fftfreq` with negative `n` raises `ValueError`.
  - `rfftfreq` with negative `n` returns an empty array, as NumPy does.
- `fftn(a, axes=[])` returns the input array unchanged. It is the same
  native array, although the JS wrapper may be a different object.
- Errors:
  - `n < 1`, or a zero-length transform axis with default `n`, raises
    `ValueError` "Invalid number of FFT data points (n) specified."
  - An out-of-range axis raises `IndexError` (D-012).
  - `s` and `axes` of different lengths raise `ValueError`.
  - `s` without `axes` uses the last `len(s)` axes. NumPy deprecates this
    and warns; we accept it without a warning.
  - `s[i] = -1` means "use the input length".
  - Repeated axes are allowed; the axis is transformed twice, as in NumPy.
- Results are new C-contiguous arrays. Complex values are read through
  `toTypedArray()` (interleaved re/im) until D-008 is lifted; complex
  input is built with `fromTypedArray(..., {dtype: "complex128"})`.
- `rfftn`/`irfftn`/`hfft`/`ihfft`/`fftshift`/`out=` are follow-ups; they
  are not in PLAN §24/M10.


## D-021 — Reduction kernels (M11) — Accepted — 2026-09-29
- Motivation: PERFORMANCE.md "M11 baseline", finding 3. In the kernel alone,
  `sum`/`max` over 1e6 float64 are 5–14× slower than NumPy.
- No input copy when possible. If the reduced axes are already the
  trailing axes, the input is C-contiguous and no cast is needed, the rows are
  read in place. Only other layouts materialize the rearranged copy, as before.
- Float `sum` (and `mean`, `var`/`std` accumulation) uses NumPy's pairwise
  summation (`pairwise_sum` in `loops_utils.h.src`). Blocks of < 8 are
  sequential; blocks of ≤ 128 use 8 interleaved partial sums combined as
  `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))`; larger blocks split at
  `n/2` rounded down to a multiple of 8. Verified to match `np.sum` bit-for-bit
  on contiguous float64 input (sizes 5 to 1e5). This replaces D-017's
  sequential sum and improves accuracy. The differential tolerance for float
  sums is unchanged; it is not tightened in this step.
- Amendment (P1-3b, 2026-10-01): NumPy drops size-1 axes before choosing the
  loop order. So a reduction counts as "trailing" (pairwise inner loop) when the
  non-unit axes stay in their original order. For example, `(n, 1)` over axis 0
  is pairwise, but `(n, 2)` over axis 0 is sequential. Checked bit-for-bit for
  float64 and complex128 over 9 shape/axis cases. Complex sums use NumPy's
  complex pairwise variant (4 complex accumulators over interleaved re/im).
- Integer sums are exact and unchanged. `prod` stays sequential.
- `min`/`max` use a branch-free multi-lane loop for float/int types that
  keeps NaN propagation and the signed-zero rule (-0.0 < +0.0) from D-017.
- The public API, result dtypes and error behavior are unchanged.
- Amendments during implementation, each measured or verified against NumPy 2.5.3:
  - **Leading-axis order.** NumPy sums pairwise only along the inner loop.
    When reducing over leading axes (e.g. `axis=0` of a C-contiguous matrix)
    it accumulates sequentially, row by row. Pairwise is therefore applied only
    when the reduced axes are the trailing axes. Otherwise `sum`/`mean`/`var`/`std`
    stay sequential, and the result is bit-exact in both cases.
  - **Column sweep.** When all reduced axes are leading, `sum`/`prod`/`min`/`max`/
    `mean`/`var`/`std` read the untransposed input as `n × rows` and stream each
    line into per-output accumulators. This has the same per-output order as
    before, without the transposed copy. `argmin`/`argmax` keep the old path.
  - **min/max NaN detection.** A `bool` flag in the lane loop blocked
    vectorization (428 µs vs 87 µs at 1e6 f64 in a probe). Lanes instead
    accumulate `v - v`, which is NaN for NaN or ±inf input. A hit triggers an exact
    rescan that returns the first NaN; ±inf-only input just costs one extra
    pass. The signed-zero rule is applied by a rescan only when the result is 0.
  - **Contraction.** NumPy rounds `d*d` in the variance (and every product)
    separately. Clang's default `-ffp-contract=on` may fuse them into an FMA,
    which changes the bits. `CMakeLists.txt` therefore compiles `reduce.cpp` with
    `-ffp-contract=off` (GCC/Clang). MSVC builds are not covered and not verified.
    This was verified bit-exact with no kernel slowdown.
  - **argmin/argmax.** Contiguous rows reuse the vectorized `minmax_row` to
    find the extreme value, then run a second scan for its first index (the
    first NaN if the value is NaN). `==` treats ±0.0 as equal, so the first
    zero of either sign wins, the same as NumPy's first-strictly-better scan.
    Cost: at most two passes over the row. bool and float16 keep the scalar loop.
- Verification: 77 exact (not tolerance-based) differential cases in
  `d021_cases()`. Pre-D-021 code fails 34 of them.

## D-022 — matmul wrapper copies (M11) — Accepted — 2026-09-29
- Evidence: the baseline `matmul f32 128²` figure (0.339 ms, "66×") did not
  reproduce. A matmul-only suite run gives 0.014 ms (NumPy 0.005). A C++ probe
  measured `cblas_sgemm` at 5.6 µs and `linalg::matmul` at 9.3 µs. The gap is
  wrapper work: both operands always went through `astype` (a full copy) and
  the output was `zeros` although gemm overwrites every element.
- Change: an operand that is already C-contiguous, of the loop dtype and not
  broadcast is passed to gemm in place (the NDArray shares the buffer, which is
  read-only here). The output is `empty`: Accelerate gemm (β=0), the fallback
  gemm (fills first), the k=0 path (fills) and `loop_gemm` (assigns) all write
  every element. `m*n == 0` returns before any kernel runs.
- Results are unchanged: the same kernel sees the same data.

## D-023 — Synchronous (GC-time) buffer release via experimental N-API (M11) — Accepted — 2026-09-29
- Problem: PERFORMANCE.md finding 2. Node-API addons built for a stable
  `NAPI_VERSION` have their finalizers deferred to a later event-loop task. A
  synchronous loop keeps every result buffer alive (1000 × `add` 1e5 left 1002
  buffers live), so each new result is fresh OS memory: 49 page faults per call
  at 1e5.
- Options considered:
  1. Caching buffer pool (D-013 deferred it). It hides faults but keeps memory
     V8 does not track, and it does not fix the deferred frees.
  2. Explicit `dispose()`. This changes the API and does nothing for code that
     does not call it. It may be added later as an extra.
  3. **Chosen:** build the addon with `NAPI_EXPERIMENTAL`. For modules that
     declare the experimental API version, Node (≥ 18.x/20.x, fully in 22) runs
     *basic* finalizers synchronously inside GC, so the buffer is freed right away
     and malloc reuses it.
- Safety: our only finalizer is `~NDArrayWrap`. It frees the `shared_ptr` and
  calls `napi_adjust_external_memory`, which takes a `node_api_basic_env` and
  is allowed during GC. `NODE_ADDON_API_REQUIRE_BASIC_FINALIZERS` is defined,
  so node-addon-api fails the build (`static_assert`) if a finalizer ever needs
  a full env.
- Cost / risk: the addon depends on experimental Node-API. Its ABI is not
  covered by Node-API stability guarantees, so prebuilt binaries (M13) must be
  tested per supported Node major. The CMake option `NATIVPY_NAPI_EXPERIMENTAL`
  (default ON) builds the stable `NAPI_VERSION=8` addon when OFF. Node 16 loads
  the module but gains nothing (outside `engines >= 18`).
- Scope: memory is still released only when V8 collects the wrapper. This is
  not deterministic release.

## D-024 — Per-call binding overhead (M11) — Accepted (partial) — 2026-09-29
- Evidence (`sample` of a `transpose` 32² loop, Node 22.7, experimental N-API):
  - core `transpose` alone takes 374 ns and the JS call about 1.18 µs.
  - Of the native samples, about 700 ns sits under `NDArrayWrap::create` → `napi_new_instance` →
    `ObjectWrap` ctor/`napi_wrap`. The GC side (`Reference::WeakCallback`,
    `GlobalHandles::Release`, `~ObjectWrap`) is also visible.
  - Argument conversion (`arg_ints`, `napi_get_array_length`) is small.
  - The TS layer adds ≈0 (`np` A.T vs raw addon).
- Change: the construction guard was a string tag (a new JS string per call, plus
  `Utf8Value()` copy and compare). It is now a per-env token object checked by
  identity (`StrictEquals`). Direct JS construction is still rejected, including
  with the old tag string, and a test covers this. Measured: transpose 1185 → ~1035 ns,
  reshape view 1196 → ~1100 ns.
- Not done: the remaining cost is intrinsic to one `ObjectWrap` (constructor call,
  wrap, weak ref) per result. Going below it would need a different handle model,
  for example `napi_create_external` handles with a plain JS prototype, or batching
  many ops per native call. That changes the binding architecture and needs its own
  decision and prototype. Deferred; it is recorded in ROADMAP.


## D-025 — Portable CI: FP contraction and platform-dependent references — Accepted — 2026-09-30
- Context: the first Linux CI run (GCC 13, ubuntu-latest) found issues that macOS/Clang did not show.
- Build: GCC `-Wshadow` flagged the `ErrorKind::Shape` enumerator against `using Shape`.
  `shape.hpp` now includes `error.hpp` before declaring `Shape`. GCC `-Wconversion`
  flagged `pairwise_sum` instantiated for small ints; that path is now `if constexpr` float-only.
- FP contraction: GCC defaults to `-ffp-contract=fast` and fused `a*b+c` into FMA on
  aarch64. That broke bit-exact arange/linspace/random against NumPy's Linux wheels.
  Non-Apple, non-MSVC builds now use `-ffp-contract=off` globally. Apple keeps Clang's
  default (`on`), which matches the macOS wheels. Forcing `off` on macOS broke 9
  arange/random cases. `reduce.cpp` stays `off` everywhere (D-021).
- eig differential test: the real parts of conjugate pairs are equal only up to
  rounding. Sorting by the exact real part could swap the pairs. The sort key now
  compares real parts within the test tolerance first. The eigenvalues were already correct.
- min/max signed zero: when the input has both +0 and -0, the sign NumPy returns depends
  on its SIMD path (x86 baseline differs from AVX2/NEON). Those cases (`zero_rule`)
  assert the D-017 rule (min -> -0.0, max -> +0.0), not the host NumPy's bits.
- Verified: macOS arm64 (Clang), plus Docker ubuntu:24.04 arm64 and amd64 (emulated) with GCC 13.3.

## D-026 — npm distribution: bundled stable-ABI prebuilds + local release command (M13) — Accepted — 2026-09-30
- Goal: `npm install nativpy` works without a compiler, CMake, Ninja or Python.
  Python/NumPy are development-only (differential tests; pip supplies cmake/ninja).
- Layout: one package `nativpy` that ships `dist/` plus
  `prebuilds/<process.platform>-<process.arch>/nativpy.node`. The loader (D-007)
  now resolves: `NATIVPY_ADDON_PATH` → bundled prebuild → repo `build/{Release,Debug}`.
  Per-platform `optionalDependencies` packages were considered. They make the
  install smaller, but every release has to publish N packages in lockstep, which
  is harder to do from one local command. We can revisit this if the package grows too large.
- ABI: published prebuilds are built with `NATIVPY_NAPI_EXPERIMENTAL=OFF`
  (stable `NAPI_VERSION=8`). One binary per platform then loads on every
  Node ≥ 18. The cost is that D-023's GC-time buffer release is lost in
  published builds (finalizers run after the event loop turns). Source/dev builds keep D-023.
  Verified: the stable addon passes the unit (83) and differential (3677) suites.
- Linux prebuilds are built in Docker `quay.io/pypa/manylinux_2_28_{x86_64,aarch64}`
  (AlmaLinux 8, glibc 2.28, GCC 14) with `-static-libstdc++ -static-libgcc`
  (`NATIVPY_STATIC_RUNTIME`). They need only libc/libm (max symbol `GLIBC_2.27`),
  so they load on Ubuntu 20.04+, Debian 10+ and RHEL 8+. A first attempt on
  `node:22-bookworm` needed glibc 2.36 (`arc4random`, `_dl_find_object`), which
  would have excluded Ubuntu 22.04. Its GCC 12 also hit a libstdc++ `-Wrestrict`
  false positive (GCC PR 105329), so CMake adds `-Wno-restrict` for GCC < 13 only.
  vitest's esbuild crashes under QEMU, so the emulated arch (linux-x64 on Apple
  Silicon) runs only the smoke test. The native arch runs the full vitest suite.
  musl (Alpine) and Windows are not provided yet.
- Release: `pnpm release [patch|minor|major]` (`scripts/release.mjs`) runs locally.
  It checks npm auth (`npm login --auth-type=web` opens the browser), bumps the
  version when the current one is already on npm, builds, tests and smoke-tests
  the tarball, publishes, then commits and tags `vX.Y.Z`.


## D-027 — Package `numera` (`@rajankr542/numera` on GitHub Packages); CI release to GitHub Packages + GitHub Release (M13) — Accepted — 2026-09-30
- npm name: **`numera`**. npm rejected the unscoped `nativpy` (E403 "too
  similar to existing package natives"). `numera` was unregistered at the time of
  this decision, but it is close to the popular `numeral`, so npm's
  similarity check may reject it too. In that case the fallback is to publish
  `@rajankr542/numera` on npm as well, which always passes. The package directory
  moved from `packages/nativpy` to `packages/numera`. The repo, addon file
  (`nativpy.node`), `NATIVPY_*` env vars and API names are unchanged. This
  supersedes the `nativpy` name in D-026.
- GitHub Packages name: **`@rajankr542/numera`**. Its npm registry accepts only
  names scoped to the repository owner. `scripts/ci-pack.mjs --name` packs the
  scoped variant. It rewrites `name` only for the duration of `npm pack` and
  always restores `package.json`, so the committed name stays `numera`.
- A manually triggered workflow (`.github/workflows/release.yml`, `workflow_dispatch`)
  builds, publishes and marks releases. It is **optional**. The maintainer's
  primary path for npmjs.org is the local `pnpm release` (D-026), run by hand
  from their own machine with browser login and a 2FA prompt. The local script
  also detects npm's "too similar" E403 and prints the scoped-name fix. The
  workflow runs only on `main`. Inputs: `bump`
  (`current|patch|minor|major`), `registry` (`github|npm|both`, default `github`)
  and `dry_run`. GitHub Packages uses the built-in `GITHUB_TOKEN`
  (`packages: write`). npmjs.org needs an `NPM_TOKEN` secret (granular access
  token, so no 2FA prompt). The local `pnpm release` (D-026) stays available.
- Prebuilds come from native runners, one job per platform: `ubuntu-24.04`
  (linux-x64) and `ubuntu-24.04-arm` (linux-arm64) each run the existing
  `scripts/build-prebuilds.mjs` in manylinux_2_28 Docker, with the full vitest suite
  running natively. `macos-14` builds darwin-arm64 (unit + smoke tested) and
  darwin-x64 (cross-compiled, not executed in CI). macOS prebuilds pin
  `MACOSX_DEPLOYMENT_TARGET=13.3`, the minimum for the `ACCELERATE_NEW_LAPACK`
  symbols. Without it the binary required the build host's macOS.
- One publish job assembles all four prebuilds. `scripts/ci-pack.mjs` packs the
  tarball, checks that it holds all four prebuilds, installs it into a clean project
  and runs the smoke test on linux-x64. The job then publishes, commits the version
  as `github-actions[bot]`, tags `vX.Y.Z`, pushes, and creates a GitHub Release
  marked *latest* with the tarball attached and generated notes.
- A version whose tag already exists is rejected, never overwritten.


## D-028 — npm package name `@cyfora/numera` (M13) — Accepted — 2026-09-30
- npm rejected the unscoped `numera` too (E403 "too similar to existing
  package", as anticipated in D-027). The npm package is now **`@cyfora/numera`**.
  npm does not apply its similarity check to scoped names. The scope is the `cyfora`
  npm organization, which is chosen over the personal `@rajankr542` so the package
  name is not tied to one account. Publishing requires that the org exists
  and that the publishing user is a member with publish rights.
  `publishConfig.access: "public"` keeps it free and public.
- The import specifier becomes `@cyfora/numera`; the API is unchanged. The
  directory stays `packages/numera`. The root workspace package (private) stays
  `numera`.
- `scripts/release.mjs` checks the scope before building. If the scope is not the
  logged-in user, `npm org ls <scope> <user>` must succeed and show a
  membership. Otherwise the release stops at once with instructions to create
  the org or ask for an invite, instead of failing at `npm publish` after
  the full build.
- GitHub Packages still requires the repo-owner scope. The workflow strips any
  existing scope before adding the owner's, so the name is `@rajankr542/numera`, not
  `@rajankr542/@cyfora/numera`. This supersedes the npm name in D-027.
- Outcome (2026-09-30): `@cyfora/numera@1.0.0` was published publicly from the
  local `pnpm release`, by npm account `cyfora`, which owns the `@cyfora` scope.
  The scope equals the logged-in user, so the org preflight is skipped; it still
  applies if someone else publishes. Tag `v1.0.0` is on `3d98b89`.


## D-029 — Automatic branch-channel releases; self-contained npm package (M13) — Accepted — 2026-09-30
- **Trigger.** `.github/workflows/release.yml` runs on every push to `main`,
  `beta` and `alpha` (Markdown/docs/benchmarks-only pushes are ignored), plus
  `workflow_dispatch` with `bump=auto|patch|minor|major`, `registry` and
  `dry_run`. Channel = branch: `main` → npm dist-tag `latest`, `beta` → `beta`,
  `alpha` → `alpha`. Concurrency is one run per branch, queued rather than
  cancelled.
- **Version from Conventional Commits** (`scripts/next-version.mjs`, unit
  tested). It looks at the commits since the newest stable tag `vX.Y.Z` merged into HEAD.
  `type!:` or a `BREAKING CHANGE:` footer → major, `feat` → minor,
  `fix`/`perf`/`revert` → patch. Anything else (docs, chore, test, ci, build,
  refactor, style, merge commits, `chore(release)`) → **no release**; the
  prebuild and publish jobs are skipped. `latest` publishes `X.Y.Z`. beta/alpha
  publish `X.Y.Z-<channel>.N`, where N is one past the highest number already
  used in git tags (all branches) or on npm for that base, so numbers are
  never reused even after a failed run.
- **Publishing auth:** repository secret `NPM_TOKEN`, an npm granular access
  token with read+write on `@cyfora/numera` and "bypass 2FA", so CI never
  prompts. npm trusted publishing (OIDC) was not chosen: it requires
  `repository.url` to match the (private) GitHub repo, and that conflicts
  with the self-contained rule below. Granular write tokens expire; renewing
  the secret is a maintainer task. `npm publish --access public --tag <channel>`.
- **Git state:** a stable release commits the version to `package.json`
  (`chore(release): …`, skipped by the workflow's `if`) and pushes it with tag
  `vX.Y.Z`. Prereleases push only the tag, so `package.json` on `main` always
  shows the last stable version and beta/alpha never diverge from it. A GitHub
  Release is created with the tarballs: `--latest` for main, `--prerelease`
  for beta/alpha. GitHub Packages is opt-in (`registry` input), no longer the default.
- **Self-contained package.** The repository is private but the package is
  public, so the tarball must not point at anything a user cannot open.
  `scripts/stage-package.mjs` writes a package README (the root README up to
  "Development", without the GitHub Packages, ROADMAP, source-build and
  PERFORMANCE parts), a `COMPATIBILITY.md` without the internal "Decision"
  column, and `LICENSE`. It also rewrites dist comments that cite
  PLAN/DECISIONS. Then it **fails** if anything repo-only is left (D-NNN,
  PLAN, ROADMAP, github.com, `pnpm` dev commands, "source build").
  `package.json` has no `repository`/`homepage`/`bugs`. `files` excludes
  `dist/**/*.map`, since the maps point at `src/`, which is not shipped.
  `scripts/ci-pack.mjs` checks all of this on the packed tarball. The addon's
  "no prebuild" error no longer tells users to run `pnpm build:native`.
- `scripts/release.mjs` (local, interactive fallback) uses the same staging.


## D-030 — API reference site shipped in the package as the npm homepage; npm-only publishing (M13) — Accepted — 2026-09-30
- **What.** A single-page, lodash-style API reference: a sidebar with search
  and categories, and for each function its signature, arguments, return value
  and an example. Source: `docs/site/api.mjs` (content) and
  `scripts/build-docs.mjs` (renderer). `pnpm docs` writes
  `packages/numera/docs/index.html`, one self-contained HTML file with no
  external requests (no fonts or scripts from a CDN). It is generated and
  gitignored, like the staged README.
- **Hosting.** The repository is private and there is no GitHub Pages, so the
  page ships inside the npm tarball (`files` includes `docs`) and is served by
  unpkg. unpkg serves `.html` as `text/html`; jsDelivr serves it as
  `text/plain`, so it can't be used. At deploy time
  `scripts/set-homepage.mjs` runs `npm pkg set homepage=...` with
  `https://unpkg.com/<name>@<version>/docs/index.html`, so each npm version
  links to the docs for that version. Both `release.yml` and
  `scripts/release.mjs` call it. `ci-pack.mjs` allows `homepage` only when it
  is exactly that URL; `repository` and `bugs` are still rejected (D-029).
- **Examples are tested.** `test/docs_site.test.ts` runs every documented
  example against the real addon and checks each `// => value` result comment.
  It also checks that every function on the default export is documented, so
  the docs can't drift from the API.
- **npm only.** GitHub Packages publishing (D-027) is removed. There is no
  `registry` workflow input, no `@rajankr542/numera` pack/publish steps, no
  `.npmrc` instructions in the README, and no `ci-pack.mjs --name`. GitHub
  Releases (tag + tarballs) remain.


## D-031 — Manual releases only; GitHub Actions release workflow removed (M13) — Accepted — 2026-09-30
- **Decision.** Releases are run by hand from the maintainer's machine with
  `pnpm release` / `release:minor` / `release:major` / `release:dry`
  (`scripts/release.mjs`, D-026). `.github/workflows/release.yml` is deleted,
  and so is everything that only it used:
  - `scripts/ci-pack.mjs`;
  - `scripts/next-version.mjs`, which computed the version from Conventional
    Commits;
  - `test/release_version.test.ts`, its unit tests.

  The test workflow `.github/workflows/ci.yml` stays.
- **Supersedes** the CI parts of D-027 (the manual `workflow_dispatch` release)
  and D-029:
  - the branch-channel trigger (`main`/`beta`/`alpha` → `latest`/`beta`/`alpha`);
  - versions from Conventional Commits;
  - the `NPM_TOKEN` secret;
  - prebuilds on native runners;
  - GitHub Releases created by the bot.

  It also supersedes the `release.yml` mention in D-030.
- **What changes for releases.**
  - Only the `latest` dist-tag. No beta/alpha channels.
  - The version is a patch/minor/major bump chosen on the command line. If the
    current version isn't on npm yet, it is published as is.
  - Auth is `npm login` in the browser, plus a 2FA OTP prompt or `--otp`.
  - All four prebuilds are built on the maintainer's Mac: darwin locally, and
    Linux in `manylinux_2_28` Docker.
  - No GitHub Release is created. The `vX.Y.Z` tag is pushed with `--push` or
    `git push --follow-tags`.
- **Checks kept.** The self-contained tarball checks from `ci-pack.mjs` (D-029,
  D-030) moved into `release.mjs` (`checkTarball`). Before publishing, it checks
  that the tarball has:
  - every built prebuild;
  - README/LICENSE/COMPATIBILITY and `docs/index.html`;
  - no source maps;
  - no `repository`/`bugs`;
  - `homepage` exactly `https://unpkg.com/<name>@<version>/docs/index.html`.

  The smoke test in a clean project still runs. `stage-package.mjs` and
  `set-homepage.mjs` are unchanged.
- **Why.** The maintainer chose to release manually. The workflow never ran on
  GitHub, and it needed a long-lived npm write token that bypasses 2FA.

## D-032 — NumPy parity programme P0–P15 and its scope — Accepted — 2026-09-30
- **Decision.** The function-completion milestones in `docs/plan/NUMPY_PARITY.md`
  (P0–P15) run before the rest of M11–M14. Every function follows that
  document's section 2 Definition of Done, which includes a benchmark case in
  **both** `benchmarks/nativpy/suite.bench.mjs` and `benchmarks/numpy/suite_bench.py`.
- **Scope defaults** (the maintainer said "start working on everything" without
  answering the open questions, so these are the proposed defaults; they can be
  revisited):
  - a. No `object_` dtype. `vectorize`/`frompyfunc` will wrap JS callbacks
    (a slow path, documented as such).
  - b. `longdouble`/`clongdouble` will alias float64/complex128, as on MSVC.
    This is a documented divergence.
  - c. Deprecated APIs (`matrix`, `bmat`, `asmatrix`, `char.chararray`, `rec`)
    are implemented last (P15), or not at all.
  - d. Build/introspection/Python-only names (`show_config`, `show_runtime`,
    `get_include`, `f2py`, `ctypeslib`, `test`, `info`, `typing`,
    `from_dlpack`, `frompyfunc` objects, `nested_iters` …) are excluded from the
    coverage denominator. Each exclusion and its reason is listed in
    `api/exclusions.json`.
- **Coverage tool (P0, PLAN §37).**
  - `python/api_inventory.py` writes `api/numpy-api.json`. This is the public
    NumPy surface of the pinned reference version, grouped by surface: `np`,
    `linalg`, `fft`, `random`, `Generator`, `RandomState`, `ndarray`, `ma`,
    `polynomial`, `strings`, `emath`, `testing`, `rec`, `char`. The file is
    checked in, so coverage does not change when the local NumPy changes.
  - `scripts/api-coverage.mjs` loads the built package and matches names
    ignoring case and `_`, so `floor_divide` matches `floorDivide` and
    `moveaxis` matches `moveAxis`. It writes `api/coverage.json` and prints
    per-surface percentages.
  - **Benchmark coverage** is found statically. Each suite is scanned for call
    sites: `np.<name>(`, `np.linalg.<name>(`, `np.fft.<name>(`,
    `rng.<name>(`/`rs.<name>(`, and `.<method>(` for ndarray methods. An
    implemented name counts as benchmarked only when **both** suites reference
    it. This is a heuristic; it is documented and cheap.
  - `--check` (run in CI) fails in any of these cases:
    - API coverage falls below `api/coverage-baseline.json`;
    - an implemented name lacks a benchmark on either side and is not in the
      baseline's `benchIgnore` list, which must only shrink;
    - an alias in `api/aliases.json` points to a name numera doesn't export.
- **Aliases.** Names that JS can't use, or that numera spells differently, are
  mapped explicitly in `api/aliases.json`, e.g. `ix_` → `ix`, `r_` → `r`.



## D-033 — JS representation of complex numbers (P1, supersedes the open part of D-008) — Accepted — 2026-09-30
- **Value type.** `np.Complex` is a frozen class with `re` and `im` number fields.
  - `np.complex(re, im = 0)` is the factory.
  - `toArray()`, `item()` and `toString()` on complex64/complex128 arrays return
    `Complex` instances. complex64 values are widened to double, like float32.
- **Input.** `np.array`, `np.full` and `set` accept `Complex` instances and plain
  `{re, im}` objects as elements.
  - Inference: any complex element makes the array complex128 (NumPy: Python
    `complex` → complex128).
  - Real numbers, bools and bigints stored into a complex dtype get `im = 0`.
  - Storing a complex element into a real dtype raises `DTypeError` (numera's
    `TypeError`). This is
    stricter than NumPy, which raises `TypeError` for `complex` → `int` but
    only warns (ComplexWarning) for `complex` → `float`. numera chooses the
    error so that data is never lost silently.
- **Kernels** follow NumPy's complex loops, using `std::complex` with the
  formulas NumPy uses where they differ from the C++ library:
  - `abs` returns the real dtype. It uses NumPy's SIMD `cabsolute` formula:
    inf wins over nan, then `max * sqrt(fma(min/max, min/max, 1))`. This is
    the path NumPy takes on NEON/AVX, so results match bit-for-bit.
  - `divide` uses Smith's algorithm, as in NumPy's `nc_quot`.
  - `sqrt` and `log` are ports of NumPy's npymath `npy_csqrt` (msun,
    Algorithm 312) and `npy_clog` (CPython). A libc++ `std::sqrt(-1+0j)`
    returns `6e-17+1j` rather than `1j`, which a relative tolerance cannot
    absorb. `exp` uses `std::exp`. Libm results may still differ by a few ULP
    (D-014 tolerance), e.g. when a platform's NumPy uses the system `csqrt`.
  - `power` with a complex exponent uses `std::pow`. Integer exponents are
    squared repeatedly, as NumPy does for small integers.
  - `mod` and `floorDivide` raise `DTypeError`, as NumPy raises `TypeError`.
- **Reductions.** `sum`, `prod` and `mean` support complex. `min`, `max`,
  `argmin` and `argmax` use NumPy's lexicographic order (re first, then im;
  NaN propagates). `var` and `std` return real values: the mean of
  `|x - mean|²`.
- **Staging.** P1 lands in this order:
  1. conversion;
  2. ufuncs plus `real`, `imag`, `conj`, `angle`, `iscomplex`, `isreal`,
     `iscomplexobj`, `isrealobj`;
  3. reductions;
  4. matmul;
  5. complex linalg.
  Each step is tested against NumPy before the next one starts. Any path that
  isn't done yet keeps raising `NotImplementedError`, never a wrong value.

## D-034 — Complex var/std (P1-3d) — Accepted — 2026-10-01
- Follows NumPy `_var`: the mean is `sum / n` in the complex dtype (D-033
  division); then `|x - mean|² = re² + im²` in the matching real dtype, with
  each square rounded before the add (no FMA); then the same pairwise or
  sequential summation as a real reduction (D-021); then a float64 divide by
  `max(n - ddof, 0)`, cast back, and `sqrt` for std.
- The result is real: float32 for complex64, float64 for complex128.
- `dtype=` different from the input, when either side is complex, raises
  `NotImplementedError`. NumPy's intermediate casts in that case are
  dtype-specific; they're deferred, not approximated.
- Exactness: bit-exact vs NumPy where real reductions are (D-021: trailing or
  leading reduced axes). Non-adjacent multi-axis (e.g. `axis=(0, 2)`) is within
  1 ulp, the same as real `sum`/`var` there (D-017 tolerance).
- Complex `prod` (P1-3f) is compared with a tolerance, not bit-for-bit. Its
  rounding depends on FMA contraction in complex multiply, which is
  build-specific: NumPy 2.5.3's arm64 complex64 loop fuses one product, and a
  24-element product differed by 1 ulp in one component. `sum`/`mean`/`min`/
  `max`/`arg*` stay exact on the `complex_reductions` group; `var`/`std` are
  exact on its large random inputs and use the D-017 tolerance on small ones.


## D-035 — Complex matmul, portable kernel (P1-4a) — Accepted — 2026-10-01
- complex64/complex128 `matmul` (and so `dot`/`inner`/`outer`, which go
  through `matmul_2d`) use a loop in `linalg.cpp`, the same structure as
  NumPy's non-BLAS `matmul` inner loop. Each output starts at `+0+0j` and adds
  the terms `(ar*br - ai*bi, ar*bi + ai*br)` in increasing `k` order. There is
  no C99 Annex G NaN recovery, so `inf*1 -> inf+nanj` and
  `(inf+infj)*1j -> nan+nanj`, as in NumPy. Sums of `-0` products are `+0`,
  and `k=0` gives `+0+0j`.
- Result dtype comes from `promote_types` (e.g. float64 @ complex64 ->
  complex128).
- Exactness: compared with a tolerance (D-018), not bit-for-bit. NumPy calls
  `cblas_cgemm`/`zgemm` for complex matmul. Probing NumPy 2.x + Accelerate on
  arm64 found that k=1 results equal `fma(ar,br,-(ai*bi))`/`fma(ar,bi,ai*br)`,
  but for k>1 neither an unfused nor any simple FMA emulation reproduces
  NumPy. Its strided non-BLAS loop also differs from an unfused emulation,
  because of FP contraction in the wheel build. Summation order and FMA use
  belong to the BLAS, so no portable kernel can match them bit-for-bit.
- P1-4b sends complex to the backend's `?gemm` (Accelerate `cblas_cgemm`/
  `zgemm`). This loop stays as the fallback backend path.

## D-036 — Complex matmul on Accelerate: NumPy's BLAS dispatch (P1-4b) — Accepted — 2026-10-01
- `Routines<T>::cgemm` (row-major, no conjugation) is part of the backend
  interface. The non-BLAS loop from D-035 moved to `noblas_cgemm` in
  `backend.hpp`. It is the whole fallback implementation, and the BLAS backend
  also uses it where NumPy does.
- The Accelerate backend follows NumPy's `@TYPE@_matmul` loop selection
  (`matmul.c.src`) for C-contiguous operands, which is all `matmul_2d` passes:
  - any of `m`, `k`, `n` is 0: non-BLAS loop;
  - `m == n == 1` (scalar_out): `cblas_?dotu_sub`. NumPy's double re-sum of
    a single chunk is exact, so the result is the BLAS value;
  - `k == 1` and (`m == 1` or `n == 1`) (scalar_vec): non-BLAS loop;
  - `m == 1` (vector @ matrix): `cblas_?gemv(RowMajor, Trans, k, n, B)`;
  - `n == 1` (matrix @ vector): `cblas_?gemv(ColMajor, Trans, k, m, A, lda=k)`;
  - `k == 1` (column @ row): non-BLAS loop;
  - otherwise `cblas_?gemm`. NumPy's `syrk` branch requires `ip1 == ip2`
    with transposed strides, which can't happen for our contiguous copies.
- The branch choice matters for results. Accelerate's `zgemm` on a 1×1×1
  `inf * 1` returns `nan+nanj`, while `zdotu` and NumPy return `inf+nanj`.
- Exactness, measured with NumPy 2.x + Accelerate on arm64: every branch
  (gemm, large gemm, both gemv shapes, dotu with k = 13 and 200, scalar_vec,
  column @ row, batched) is bit-identical for complex64 and complex128. Other
  BLAS builds may round differently, so the differential suite keeps the
  D-018 tolerance. The fallback backend stays tolerance-level (D-035).

## D-037 — Complex product differential cases (P1-4d) — Accepted — 2026-10-01
- New differential group `complex_matmul` (262 cases): `matmul`/`dot`/`inner`/
  `outer` on complex64/complex128 over the same shape set as D-018, plus the
  D-036 dispatch shapes (1×1·1×1, 1×k·k×1, column·row, 1×1·1×n), batched and
  broadcast stacks, N-D `dot`, empty operands, every real dtype mixed with
  either complex width (both operand orders), mixed complex64/complex128, 0-d
  `dot` operands, non-finite values, and core/batch/0-d errors.
- Quarter-step inputs make every product exact, so those cases compare values
  exactly (`toEqual`, signed zeros and NaN included). Larger random operands
  (gemm, both gemv shapes, dotu, batched) use the D-018 tolerance, scaled by
  max|expected|: 2e-5 for complex64, 1e-12 for complex128.
- With non-finite input NumPy's BLAS path and its own non-BLAS loop can give
  different results. A 1×2·2×2 `gemv` with `inf` gives `nan+nanj` on
  Accelerate, while NumPy's loop gives `inf+nanj`. The default backend must
  match NumPy (BLAS). The fallback backend must match NumPy's non-BLAS loop,
  which the generator records as `expected_noblas` by running `np.matmul` on
  inner-stride-2 views of the same data. On the exact cases both backends are
  therefore checked bit-for-bit against NumPy, never against each other.
- For zero-size results NumPy reports strides `(0, 0)` from some product paths.
  We always return C strides, which also hold for zero-size arrays, so strides
  are compared only for non-empty results (the real linalg group never
  compared them, D-018).

## D-038 — Complex det (P1-5a) — Accepted — 2026-10-01
- `det` accepts complex64/complex128 and returns the same dtype. All other
  decompositions still reject complex input with `NotImplementedError` until
  their own slice lands.
- `Routines<T>::cgetrf` (column-major, in place, 0-based pivots) is part of
  the backend interface. Accelerate calls `cgetrf_`/`zgetrf_`. The fallback
  reuses its partial-pivot LU and, like LAPACK's `icamax`, picks the pivot
  by |re| + |im|.
- The value follows NumPy's `umath_linalg` `det` exactly: `sign` starts at ±1
  from the pivot parity and is multiplied by `u_ii / |u_ii|` (`|.|` is
  `hypot`) using the unfused `mult` formula. The result is
  `mult(sign, exp(Σ log|u_ii|) + 0j)`. If
  `getrf` reports a singular matrix (info > 0), `sign = 0` and
  `logdet = -inf`, so the result is `0+0j`. A 0×0 matrix gives `1+0j`.
- Precision: NumPy's `det` always runs the complex128 kernel
  (`signature='D->D'`, from `_commonType`) and casts to the result dtype
  afterwards. We do the same, so complex64 input is factored with `zgetrf`
  and rounded to complex64 once at the end. Probe: computing complex64 det
  in single precision matched NumPy in 39/70 random cases; the complex128
  path then cast matched 120/120. NumPy's real `det` also runs in float64
  (`'d->d'`) and then casts float32 results, whereas ours keeps float32
  (D-018). That is a separate follow-up.
- Real `det` keeps its direct product of the diagonal (D-018). Switching it
  to the slogdet form would be a separate decision.
- Exactness (NumPy 2.x + Accelerate, arm64, 70 random matrices n = 1–40,
  both widths): the default backend is bit-identical to NumPy in 70/70,
  because it uses the same `zgetrf` and the same post-processing. The
  fallback LU uses a different elimination order: complex128 is within
  ~2e-14 relative, so it is tolerance-level (D-018). Its complex64 results
  matched after the final rounding.

## D-039 — Complex inv/solve (P1-5b) — Accepted — 2026-10-01
- `inv` and `solve` accept complex64/complex128.
- Result dtype follows NumPy's `_commonType`: complex if any operand is
  complex, and double width if any operand is float64/complex128 or is an
  integer/bool. So `solve(complex64, float32)` is complex64, while
  `solve(complex64, int8)` and `solve(complex64, float64)` are complex128.
  float16 still raises `DTypeError`.
- Like `det` (D-038), complex input is computed in complex128 (NumPy's
  `'D->D'`/`'DD->D'` signatures) and the result is cast once at the end.
- `Routines<T>::cgesv(n, nrhs, a, b)` (column-major, A overwritten by its LU,
  B by X) is part of the backend interface. Accelerate calls `cgesv_`/`zgesv_`
  with `lda = ldb = max(n, 1)`, as NumPy's `init_gesv` does. The fallback runs
  its partial-pivot LU (|re| + |im| pivoting) and then forward/back
  substitution.
- `inv` solves `A X = I`, the same `gesv` call NumPy makes. If `info > 0` the
  call raises `LinAlgError("Singular matrix")`, as for real input. NaN input
  does not raise, also as in NumPy.
- Real `inv`/`solve` are unchanged: float32 stays float32 (D-018).
- Exactness (NumPy 2.x + Accelerate, arm64; 210 random cases: `inv`, `solve`
  with an n×3 b, and `solve` with a vector b, for n = 1–40 at both widths):
  the default backend was bit-identical to NumPy in 210/210 cases. The
  fallback LU is tolerance-level (D-018): its complex128 results are within
  ~2e-14 relative, and its complex64 results matched after the final
  rounding.

## D-040 — Complex qr (P1-5c) — Accepted — 2026-10-01
- `qr` accepts complex64/complex128 in every mode ('reduced', 'complete',
  'r'). Q and R keep the input's complex dtype (NumPy `_commonType`).
- Like `det`/`inv`/`solve` (D-038/D-039), complex input is computed in
  complex128 (NumPy's `'D->D'`/`'DD->D'` signatures) and the result is cast
  once at the end. Real `qr` is unchanged: float32 stays float32 (D-018).
- `Routines<T>::cgeqrf(m, n, a, tau)` and `cungqr(m, cols, k, q, tau)` are
  part of the backend interface. They take the same column-major layout as
  `geqrf`/`orgqr`. Accelerate calls `cgeqrf_`/`zgeqrf_` and
  `cungqr_`/`zungqr_` with lda = m and the queried optimal workspace.
- The fallback follows LAPACK's `zlarfg`/`zgeqr2`/`zung2r`:
  `beta = -sign(|(alpha, x)|, Re alpha)` is real, so R's diagonal is real
  (and usually negative), as in NumPy. `tau = 0` only when `x = 0` and
  `Im alpha = 0`. The reflector `H^H = I - conj(tau) v v^H` is applied to the
  trailing columns, and Q = H(1)…H(k) is built by applying `I - tau v v^H`
  to the identity, starting from the last reflector.
- NaN input does not raise, as in NumPy (Q and R contain NaN).
- Exactness (NumPy 2.x + Accelerate, arm64; 60 random m×n cases, m, n = 1–29,
  both widths, all three modes): the default backend was bit-identical to
  NumPy in 60/60 cases. The fallback is tolerance-level (D-018). It was
  bit-identical in 30/60 cases, and its largest absolute difference was ~5e-15.

## D-041 — Complex svd (P1-5d) — Accepted — 2026-10-01
- `svd` accepts complex64/complex128 with every `fullMatrices`/`computeUV`
  combination. U and Vh keep the input's complex dtype. S is real:
  float64 for complex128 input, float32 for complex64 input (NumPy
  `_realType`).
- Like D-038–D-040, complex input is computed in complex128 (NumPy's
  `'D->DdD'`/`'D->d'` signatures) and the results are cast once at the end.
  Real `svd` is unchanged: float32 stays float32 (D-018).
- `Routines<T>::cgesdd(m, n, a, s, u, vt, full)` is part of the backend
  interface. It uses the same column-major layout and job selection as
  `gesdd`, but `s` is real. Accelerate calls `cgesdd_`/`zgesdd_` with the
  queried optimal `lwork`, `iwork` = 8·min(m,n), and LAPACK's documented
  `lrwork` minimum: 7·mn for job 'N', else
  max(5·mn² + 5·mn, 2·mx·mn + 2·mn² + mn).
- The fallback reuses the one-sided Jacobi SVD. For the complex rotation of
  columns p and q with γ = u_pᴴu_q, column q of both U and V is first scaled
  by conj(γ/|γ|), which makes γ real; then the real rotation is applied.
  Wide input works on B = Aᴴ. Then A = V_B Σ U_Bᴴ, so U = V_B and
  Vh = U_Bᴴ. The basis completion (`full_matrices`, rank deficiency) uses
  Hermitian inner products. The real fallback path is unchanged.
- As in NumPy's `svd_wrapper`, any non-finite input element raises
  `LinAlgError("SVD did not converge")` before LAPACK is called. This
  explicit check also applies to real input (both backends already raised
  there).
- With k = min(m, n) = 0 and `fullMatrices`, U and Vh are identity
  matrices, as in NumPy.
- Singular vectors are unique only up to a unit phase per pair. The
  fallback is checked by reconstruction and unitarity (D-018), not by
  element-wise comparison of U/Vh.
- Exactness (NumPy 2.x + Accelerate, arm64; 60 random m×n cases, m, n = 1–29,
  both dtypes, full and reduced): with the default backend, S was
  bit-identical in 60/60 cases and U/Vh in 57/60. The other 3 differ by at
  most 5.6e-17 absolute, in near-zero entries. Both sides are deterministic
  and the cause is not known; giving it the exact queried `lwork` made no
  difference. With the fallback, S was bit-identical in 32/60 cases, largest
  |ΔS| 5e-14.

## D-042 — Complex eigh/eigvalsh, native eigvalsh (P1-5e.1) — Accepted — 2026-10-01
- `eigh`/`eigvalsh` accept complex64/complex128 (Hermitian, lower triangle,
  UPLO='L'). Eigenvalues are real and ascending: float32 for complex64,
  float64 for complex128. Eigenvectors keep the input's complex dtype.
- Like D-038–D-041, complex input is computed in complex128 (NumPy's
  `'D->dD'`/`'D->d'` signatures) and the results are cast once at the end.
  Real input is unchanged: float32 stays float32 (D-018).
- `eigvalsh` becomes its own native entry point (`linalg::eigvalsh`). It
  calls the eigen routine with JOBZ='N', as NumPy's `eigvalsh` gufunc does.
  This affects real input too. LAPACK's values-only path is a different
  algorithm (`?sterf` rather than `?stedc`), and in a NumPy probe its
  results differed bitwise from `eigh(a).eigenvalues` in 29/31 random real
  and 29/62 random complex matrices. Before this change, `eigvalsh` returned
  `eigh(a).eigenvalues`, so it could not be bit-identical to NumPy.
- Backend interface: `syevd(n, a, w, vectors)` gains the `vectors` flag, and
  `cheevd(n, complex* a, T* w, vectors)` is new. Accelerate calls
  `ssyevd_`/`dsyevd_`/`cheevd_`/`zheevd_` with queried `lwork`/`lrwork`/
  `liwork`. The fallback is the cyclic Jacobi method. For complex input, each
  pair (p, q) is first phase-aligned: row/column q of A and column q of V are
  scaled by conj(a_pq)/|a_pq|, which makes a_pq real; then the real rotation
  is applied. The diagonal's imaginary part is ignored, as in LAPACK. The
  fallback always computes vectors, and `vectors` only changes what is
  returned. The real fallback path is unchanged.
- As in NumPy, non-finite input is not pre-checked; it goes to the backend.
  With Accelerate the results match NumPy: a NaN diagonal gives [nan, 1]
  with identity vectors, and an Inf in the lower triangle gives all-NaN
  values and vectors. The Jacobi fallback raises
  `LinAlgError("Eigenvalues did not converge")` in that case, which is a
  backend difference already present for real input. Any backend failure
  (info > 0) raises that error.
- Eigenvectors are unique only up to a unit phase per column, so the
  fallback is checked by reconstruction (A·V = V·Λ) and unitarity, not
  element-wise.
- Exactness (NumPy 2.x + Accelerate, arm64; 60 random non-Hermitian n×n
  inputs, n = 1–29, 15 per dtype; only the lower triangle matters):
  - Default backend, complex128/complex64/float64: `eigh` values,
    `eigvalsh` values and eigenvectors are bit-identical in 15/15 cases each.
  - Default backend, float32: 0/15. NumPy's `_commonType` computes float32
    input in float64 ('d->dd') and casts the result; nativpy keeps D-018's
    float32 compute. The largest |Δw| is 7.6e-6. This existing D-018
    difference affects every real float32 decomposition. It is not changed
    here and is recorded as an open item.
  - Fallback: values within 6.6e-14 for complex128/float64 (complex64 values
    bit-identical 15/15 after the complex128 compute and cast); the largest
    reconstruction error is 2.6e-14.

## D-043 — Complex eig/eigvals, native eigvals (P1-5e.2) — Accepted — 2026-10-01
- `eig`/`eigvals` accept complex64/complex128. Eigenvalues and eigenvectors
  keep the input's complex dtype (complex64 -> complex64, complex128 ->
  complex128). Eigenvectors are unit 2-norm columns. Real input is unchanged
  (complex64 for float32, else complex128; D-018).
- As in D-038–D-042, complex input is computed in complex128 (NumPy's
  `'D->DD'`/`'D->D'` signatures) and cast once at the end.
- `eigvals` becomes its own native entry point (`linalg::eigvals`), calling
  geev with JOBVR='N' as NumPy's `eigvals` gufunc does. This affects real
  input too. A probe run in NumPy itself (n = 40–200) found `eigvals(a)` !=
  `eig(a).eigenvalues` bitwise in 2/5 float64 and 2/5 complex128 cases
  (n >= 160), so deriving `eigvals` from `eig` could not match NumPy.
- Backend interface: `geev(n, a, w, v)` accepts `v == nullptr` (values only);
  new `cgeev(n, complex* a, w, v)`. Accelerate calls
  `sgeev_`/`dgeev_`/`cgeev_`/`zgeev_` with JOBVL='N', ld = max(n, 1), a
  queried `lwork` and (complex) `rwork` of 2n, as NumPy's `init_geev`. The
  fallback reuses its complex Hessenberg + shifted QR solver, now templated
  on real or complex input; it skips back-substitution for values only.
- Non-finite input raises `LinAlgError("Array must not contain infs or
  NaNs")` before the backend (NumPy `_assert_finite`). Backend failure
  (info > 0) raises `LinAlgError("Eigenvalues did not converge")`.
- Exactness (NumPy 2.x + Accelerate, arm64; 45 random n×n inputs,
  n = 1–12, 20, 32, 48; 15 per dtype):
  - Default backend, complex128 and complex64: `eig` values, `eigvals`
    values and eigenvectors bit-identical in 15/15 each.
  - Default backend, float64: values and `eigvals` 15/15; eigenvectors
    10/15, the rest within 1 ulp (≤ 2.2e-16). The same 5 cases differ on the
    previous commit, so this is not new; the cause is not yet investigated.
  - Fallback: not bit-identical (different algorithm). Reconstruction error
    ≤ 1.7e-14 (complex128/float64) and 9.5e-7 (complex64); vectors match up
    to phase.

## D-044 — Complex lstsq and norm (P1-5e.3) — Accepted — 2026-10-01
- `lstsq` accepts complex64/complex128 `a` and/or `b`. The result type follows
  NumPy `_commonType` (`complex_result`, D-039): complex64 only if every
  operand is float32 or complex64, else complex128. `x` has that complex
  dtype. `residuals` and `s` are real: float32 for complex64, else float64
  (NumPy's `'DDd->Ddid'`). Complex input is computed in complex128 and cast
  once at the end (as D-038–D-043).
- Same algorithm as the real path (SVD, D-018), now generic over the element
  type: x = V · diag(1/s) · Uᴴ · b over singular values above
  `rcond · s_max`. Residuals are Σ|b − A·x|² per column, only when
  rank == N and M > N. NumPy uses `?gelsd`, so results agree within
  tolerance, not bitwise. Real input is unchanged.
- `norm` accepts complex input; the result is real (float32 for complex64,
  else float64).
  - Every ord works on |x| = hypot(re, im) (NumPy: `abs(x)`). The 2-norm
    paths (default, 'fro', vector ord=2) therefore square |x| instead of
    NumPy's `(x.conj() * x).real`, which can differ by about 1 ulp per
    element.
  - 'nuc' and matrix ±2 use the complex SVD's singular values (D-041).
  - Values match NumPy within tolerance (different summation order).


## D-045 — Casting rules and `canCast` (P2-1) — Accepted — 2026-10-01
- Native `enum class Casting { No, Equiv, Safe, SameKind, Unsafe }` and
  `can_cast(from, to, casting)` in `native/core/dtype.hpp`. All later ufunc
  `out=`/`casting=` checks (P2-2 onward) use this function, so no other code
  duplicates the casting rules.
- Rules for the 14 supported dtypes (all native byte order). A check against
  NumPy 2.5.3 `np.can_cast` over all 5 × 14 × 14 combinations found 0
  mismatches:
  - `no` and `equiv`: `from == to`. They differ only for byte order, which
    numera does not support.
  - `safe`: `promote_types(from, to) == to`.
  - `same_kind`: `safe`, or kind(from) ≤ kind(to) in the order
    b < u < i < f < c. For example float64 → float32 and int64 → int8 are
    allowed, but int64 → uint8 and float → int are not.
  - `unsafe`: always true.
- Public `np.canCast(from, to, casting = "safe")`:
  - `from` is a `DTypeLike` or an `NDArray`, whose dtype is used. Value-based
    casting does not apply, as in NumPy 2.
  - JS number, boolean and bigint arguments raise `DTypeError`, like NumPy 2's
    `TypeError` for Python scalars (NEP 50).
  - An unknown casting name raises `ValueError`, as in NumPy.
  - An unknown dtype raises `DTypeError` (NumPy `TypeError`).
- `canCast` comes forward from P3 (dtype introspection) because P2 needs it.


## D-046 — Native ufunc `out=` (P2-2) — Accepted — 2026-10-01
- New overloads `binary(op, a, b, out)` and `unary(op, a, out)` in
  `native/core/ufunc.hpp` write into `out` and return it, a view of the same
  buffer. The existing overloads without `out` are unchanged.
- Checks run in NumPy 2.5.3's order (checked by hand), so a call with several
  faults raises the same error as NumPy:
  1. `out` is read-only: `ValueError` "output array is read-only".
  2. Loop dtype resolution (D-014), using only the inputs. A missing loop
     raises `DTypeError`.
  3. The loop dtype must cast to `out.dtype` under `same_kind` (D-045),
     otherwise `DTypeError` (NumPy `UFuncTypeError`, a `TypeError`):
     "Cannot cast ufunc 'add' output from float64 to int64 with casting rule
     'same_kind'".
  4. Shapes: `broadcast_shapes(inputs..., out.shape)` must equal
     `out.shape`. Inputs may broadcast up to a larger `out`; `out` itself
     never broadcasts.
     - Incompatible shapes raise `BroadcastError` "operands could not be
       broadcast together with shapes ...", listing `out` last.
     - A smaller `out` raises `BroadcastError` "non-broadcastable output
       operand with shape X doesn't match the broadcast shape Y".
     - NumPy raises `ValueError` for both; `BroadcastError` follows D-014.
  5. Value checks, such as integer power with a negative exponent.
- The kernel computes in the loop dtype. If `out.dtype` equals it, results go
  straight into `out` through the broadcast plan, so strided and zero-stride
  `out` both work. Otherwise results go into a temporary array that is then
  cast into `out` with `copy_into`. As in NumPy, int8 + int8 into an int16
  `out` wraps in int8.
- Overlap: when writing straight into `out`, an input that may share memory
  with `out` (D-011) is copied first. The exception is an input that is
  exactly the same view as `out` (same buffer, offset, dtype and aligned
  strides); element-wise in-place is safe there. This reproduces NumPy's
  "as if the inputs were copied" results.
- `assign` moves out of `indexing.cpp` and becomes the public
  `copy_into(dst, src)` in `broadcast.hpp`, used by both indexing and ufuncs.
  It broadcasts `src`, casts unsafely, and stages through a copy when the
  two overlap.


## D-047 — TS `{ out }` option on the element-wise ufuncs (P2-3) — Accepted — 2026-10-01
- The 12 ufuncs `add subtract multiply divide power mod floorDivide abs
  negative sqrt exp log` take a final optional `opts: UfuncOptions`
  (`{ out?: NDArray | null }`). `conjugate` and `angle` are unchanged here.
- With `out`, the native D-046 overloads run and the function returns the
  same `out` JS object (`np.add(a, b, { out }) === out`), as NumPy returns
  `out` itself.
- `out: undefined` or `null` means "allocate a new result", like NumPy
  `out=None`.
- An `out` that is not an `NDArray` (nested list, scalar, JS array) raises
  `DTypeError`, matching NumPy's `TypeError` "return arrays must be of
  ArrayType". NumPy's one-element tuple form `out=(arr,)` is not accepted.
- `out` takes no part in dtype resolution. JS scalar operands stay weak
  relative to the other operand (NEP 50), and the loop result is then cast
  to `out.dtype` under `same_kind` (D-046).



## D-048 — Ufunc `dtype=` and `casting=` (P2-4) — Accepted — 2026-10-01
- Native `struct UfuncParams { std::optional<DType> dtype; Casting casting =
  Casting::SameKind; }` and overloads `binary(op, a, b[, out], params)` and
  `unary(op, a[, out], params)`. The existing overloads keep their behaviour
  because they pass the default params.
- Loop selection:
  - No `dtype`: the D-014 loop, as before.
  - `dtype=D`: the loop whose output is D, with inputs of dtype D. The one
    exception is `abs` of a complex input with D float32/float64: it uses the
    complex64/complex128 → D loop (NumPy `F->f`/`D->d`) when the input casts
    *safely* to that complex dtype. Otherwise it uses the D → D loop, so
    complex128 with D float32 needs `unsafe` and drops the imaginary part, as
    in NumPy.
  - `conjugate`/`angle` are not exposed with options. Natively, `conjugate`
    has D → D loops for every D except bool, and `angle` has no `dtype=` loop.
  - With D, a missing loop raises `DTypeError` "No loop matching the specified
    signature and casting was found for ufunc X" (NumPy `TypeError`). Missing
    loops:
    - D bool: `power`/`mod`/`floorDivide`.
    - D not float or complex: `divide`/`sqrt`/`exp`/`log`.
    - D complex: `mod`/`floorDivide`/`abs`.
    - D bool for `subtract`/`negative` gives the existing boolean messages.
- `casting` checks run in NumPy 2.5.3 order (checked by hand): read-only
  `out` → loop resolution → each input `can_cast(in, loop_in, casting)` →
  output `can_cast(loop_out, out.dtype, casting)` (only with `out`) → shapes →
  values. Errors are `DTypeError` (NumPy `UFuncTypeError`):
  - inputs: "Cannot cast ufunc 'add' input 0 from int8 to int64 with casting
    rule 'no'". Unary ufuncs say just "input".
  - output: the D-046 message, now with the actual rule name.
- Casting names are parsed natively: an unknown name raises `ValueError`, as
  in NumPy.
- TS: `UfuncOptions` gains `dtype?: DTypeLike | null` and
  `casting?: Casting` (default `"same_kind"`) on the same 12 ufuncs as D-047.
  If `dtype` is given, a JS scalar operand is made weakly relative to D rather
  than to the other operand (so `add(int8Arr, 1000, {dtype: "int16"})` works
  and `add(int8Arr, 1.5, {dtype: "int8"})` fails the input cast, as in
  NumPy).


## D-049 — Ufunc `where=` mask (P2-5) — Accepted — 2026-10-01
- Native: `UfuncParams` gains `std::optional<NDArray> where`. The mask must
  have dtype bool. Any other dtype raises `DTypeError` "Cannot cast array data
  from int64 to bool according to the rule 'safe'", because NumPy casts
  `where` to bool with the `safe` rule and only bool → bool is safe.
- Shapes: the mask takes part in broadcasting like an extra input.
  - Without `out`: the result shape is `broadcast(a, b, where)`.
  - With `out`: `out.shape` must equal `broadcast(a, b, where, out)`.
  - The broadcast error lists the mask's shape after the inputs, and the
    `out` message reports the broadcast shape including the mask, as in
    NumPy.
- Semantics: only elements where the mask is true are written. With `out`,
  the other elements of `out` are left unchanged. Without `out`, NumPy
  leaves them uninitialized; numera fills them with **zeros** so results are
  deterministic. Zero is one of the values NumPy may happen to leave, so this
  is a stricter guarantee than NumPy's, not a contradiction (COMPATIBILITY).
- The value check for integer `power` with a negative exponent only applies
  to elements where the mask is true. NumPy 2.5.3 does the same.
- Error order (NumPy 2.5.3, checked by hand): read-only `out` → mask dtype →
  loop resolution → input casts → output cast → shapes (including the mask)
  → values (masked).
- Implementation: when a mask is given, the loop runs over the full
  broadcast shape into a fresh temporary of the loop dtype. A masked copy
  (`masked_copy_into`, unsafe cast) then writes the true positions into
  `out`, or into a zeroed result. The masked copy copies the mask first when
  it overlaps the destination, so a mask that aliases `out` gives NumPy's
  result. This costs an extra pass and computes masked-out elements
  (harmless: the integer kernels are total). Making the loops themselves
  masked is left to the P2-7 registry. No performance claims are made.
- TS: `UfuncOptions.where?: ArrayLike | boolean | number`.
  - `undefined` and `true` mean no mask.
  - An `NDArray` mask is passed through unchanged, so a non-bool dtype
    raises `DTypeError`.
  - Nested lists and JS scalars are converted with
    `array(where, {dtype: "bool"})`, like NumPy's handling of Python lists
    and scalars (`[1.5, 0]` → `[true, false]`).
  - `null` is not accepted. NumPy treats `where=None` as `False`, which is
    a trap, and numera refuses to copy it.



## D-050 — Ufunc `order=` and NumPy result layout (P2-6) — Accepted — 2026-10-01
- Supersedes the D-014 rule "ufunc results are always C-contiguous". Results
  of the element-wise ufuncs now get NumPy's memory layout (strides), not
  only its shape and values. Default `order` is `'K'`, as in NumPy.
- Native: `enum class Order { C, F, A, K }` and `UfuncParams::order`
  (default K). Names are parsed natively (case-insensitive one letter, like
  NumPy). Anything else raises `ValueError` "order must be one of 'C', 'F',
  'A', or 'K' (got 'X')". A non-string raises `DTypeError` (NumPy
  `TypeError`).
- Layout of a freshly allocated result (port of NumPy 2.5.3
  `try_trivial_single_output_loop` + `NpyIter` `npyiter_find_best_axis_ordering`
  + `npyiter_new_temp_array`):
  1. **Trivial path** (no `where`; in input order, each cast input that is
     0-d or 1-d with <= 8192 (NPY_BUFSIZE) elements becomes a contiguous cast
     copy, and any other cast input disables this path): every non-0-d input
     has the same shape; every input with ndim >= 2 is C- or F-contiguous with
     the same flags (and matching `order` if it is C or F). Then the result is
     F-contiguous if the flags are F-only and C-contiguous otherwise.
  2. Otherwise `order` picks the layout: `C` → C-order, `F` → F-order, `A` →
     F if every operand (inputs, using the copies from step 1, and the `where`
     mask) is F-contiguous, else C.
     `K` → a stable insertion sort of the axes by absolute stride (smallest
     innermost) over all operands. Strides are taken as 0 on dims where the
     operand has extent 1, operands with a 0 stride on either axis don't vote,
     and C order wins ties and conflicts.
  3. A zero-size result has all-zero strides (NumPy, same as
     `allocation_strides`).
- With `out=`, `order` doesn't affect the result: `out` is returned as it
  is. Internal temporaries (the `where` path, cast to a different `out` dtype)
  use the same layout, so the loop runs over memory that matches its inputs.
- Error order: NumPy 2.5.3 reports, in order, read-only `out` → `casting`
  name → `where` dtype → `order` name → loop resolution, then as D-049.
  numera parses `order` in the binding along with `casting` (whose name is
  already parsed there, D-048). So when a call has several errors at once, a
  bad `order`/`casting` name is reported before a read-only `out` or a
  non-bool `where`. The error class for each single error matches NumPy. The
  `order` name is validated even when `out` is given.
- Verification: differential group `ufunc_order` (916 cases: random
  C/F/permuted/reversed/stepped/size-1/0-d/zero-size operands, input casts,
  `dtype=`, `where=`, unary and binary, all four orders, invalid names)
  compares strides and flags as well as values. A one-off run of the same
  generator with 7 more seeds (6,492 cases) also passed.
- `np.dot`/`np.inner` with a 0-d operand (native `linalg.cpp`, which uses
  `binary(Multiply)`) follow NumPy 2.5.3, probed on 672 + 1,473 random cases:
  both operands are first cast to the result dtype with
  `PyArray_NewLikeArray(KEEPORDER)` strides (C/F if the source is, else axes
  stably sorted by |stride|). Then for float/complex dtypes with ndim <= 2
  the result is C-order (cblas_matrixproduct); otherwise it is multiply's
  'K' layout. `outer` is unchanged (it reshapes to C-contiguous (n,1)/(1,m)
  views, giving C like NumPy).
- TS: `UfuncOptions.order?: UfuncOrder | null` (`"C" | "F" | "A" | "K"`,
  either case) on the same 12 ufuncs. The value goes straight to the
  native side, which does all the validation. `conjugate`/`angle` take no
  options, so they always use 'K', which matches NumPy's default.
- `angle` with `deg=true` multiplies by a scalar afterwards, which keeps the
  'K' layout.


## D-051 — Table-driven native ufunc registry (P2-7) — Accepted — 2026-10-01

**Context.** The `BinaryOp`/`UnaryOp` enums and their `switch` statements
spread each ufunc's definition over several functions in
`native/core/ufunc.cpp`: name, type resolver, `dtype=` loop selection,
per-dtype kernels and input checks. P2-8/P2-9 (`reduce`/`accumulate`/
`outer`/`at`) need to look up a ufunc's identity and loops by name.

**Decision.**
- New `native/core/ufunc_registry.{hpp,cpp}`. Each ufunc is one immutable
  `Ufunc` record with these fields:
  - `name`, `nin` (1 or 2).
  - `identity` (`std::optional<double>`: add 0, multiply 1, none for the
    others, as in NumPy).
  - `resolve`, the default type resolver (input dtypes → loop `{in, out}`
    dtypes).
  - `resolve_dtype`, the `dtype=` loop selection (D-048).
  - `check`, optional validation of the cast inputs before the loop runs
    (power's negative-integer-exponent error).
  - `loops[kNumDTypes]`, type-erased strided loop function pointers indexed
    by the loop input dtype (null means no loop).
- `find_ufunc(name)` returns `const Ufunc*` (null if unknown). The registry is
  a static table. `get(BinaryOp)` and `get(UnaryOp)` return the record, so
  the enums stay as aliases for existing C++ callers. The public C++ API
  (`binary`, `unary`, `*_result_dtype`, `*_op_from_name`) is unchanged.
- The binding looks ufuncs up by name through the registry.
- The generic drivers (out/where/dtype/casting/order, D-046…D-050) talk only
  to the record. No op-specific `switch` is left in the drivers.
- This is a pure refactor. Error messages, result dtypes, layouts and values
  do not change, and every existing test must stay green with no edits.


## D-052 — ufunc.reduce / ufunc.accumulate (P2-8) — Accepted — 2026-10-01
- Native `ufunc_reduce` / `ufunc_accumulate` in `native/core/ufunc_methods.{hpp,cpp}`,
  driven only by the D-051 `Ufunc` record (binary ufuncs only; unary raises
  `ValueError: reduce only supported for binary functions`).
- JS: `np.add.reduce(a, {axis, dtype, out, keepdims, initial, where})` and
  `np.add.accumulate(a, {axis, dtype, out})` as properties on the 7 binary
  ufunc functions (NumPy rejects keepdims/initial/where on accumulate; so do we).
- Loop dtype: `dtype=d` → the ufunc's dtype= resolver (D-048) on `(a, d)`;
  else with `out` → the normal resolver on `(out.dtype, a.dtype)`; else
  add/multiply use the sum/prod rule (bool and signed ints → int64, unsigned →
  uint64) and every other ufunc its normal resolver on `(a, a)`. Inputs and
  `out` are cast `unsafe` (NumPy reduce takes no `casting=`).
- Axis rules:
  - reduce: default 0; an int, a list, `null` (all axes) or `[]` (no reduction,
    result is the cast input). More than one axis is allowed only for ufuncs with
    an identity (add, multiply); others raise `ValueError: reduction operation
    '<name>' is not reorderable, so at most one axis may be specified`.
  - accumulate: a single int axis (default 0); `null` or a list raise
    `ValueError: accumulate does not allow multiple axes`; 0-d input raises
    `DTypeError` (NumPy `TypeError: cannot accumulate on a scalar`, D-009).
  - Out-of-range axes raise `IndexError`; duplicates raise `ValueError` (D-017).
- Empty reductions: the identity (or `initial`) fills the result; with neither,
  a reduced length of 0 with a non-empty result raises `ValueError: zero-size
  array to reduction operation <name> which has no identity`.
- `where=` (reduce only, bool, broadcast to `a`) needs an identity or `initial`;
  a non-bool mask raises `DTypeError`.
- Seeding: with one axis and no `initial`, the first element seeds the
  accumulator (NumPy); otherwise `initial` or the identity does.
- Evaluation (build-first, P2-8): sequential `acc = op(acc, a[k])` through the
  registry's strided loop (accumulator view with stride 0 on reduced axes).
  Exact for integers and every non-reorderable ufunc. NumPy's pairwise float
  `add`, float16 accumulator precision, and the per-buffer round-trip through
  `out`'s dtype are **not** reproduced yet; they are scheduled for P2-8v
  together with the differential group and benchmarks.
- Results are built in a fresh C-contiguous buffer and copied into `out` last,
  so `out` overlapping the input is safe. `out` must have exactly the result
  shape. NumPy's keep-order result strides are deferred to P2-8v.
- Integer power reductions apply the negative-exponent check (D-051 `check`).

## D-053 — ufunc.outer / reduceat / at (P2-9) — Accepted — 2026-10-02
- Native `ufunc_outer`, `ufunc_reduceat`, `ufunc_at` in `native/core/ufunc_methods.cpp`,
  driven by the D-051 record. Build-first (see `TASK_SLICES.md`): NumPy differential
  checks are deferred to the V phase.
- `outer`: `a` is viewed as `a.shape + (1,)*b.ndim` and passed to the generic
  binary driver, so `out`/`dtype`/`casting`/`where`/`order` behave as for the
  call form. Unary ufuncs raise `ValueError`.
- `reduceat(a, indices, {axis, dtype, out})`: segment `i` reduces
  `a[indices[i]:indices[i+1]]` (last segment runs to the end); when
  `indices[i] >= indices[i+1]` the result is `a[indices[i]]` (NumPy rule).
  Loop dtype follows D-052. Indices outside `[0, len)` raise `IndexError`.
- `at(a, indices, b?)`: JS has no tuple index, so `indices` is a number, an
  index array/list (axis 0), or an array of NDArrays (one per leading axis,
  broadcast together). Repeated indices apply repeatedly (unbuffered). All
  indices are validated before `a` is modified. Unary ufuncs get `.at` too.
  Slices/boolean masks as `at` indices are not supported yet.

## D-054 — np.seterr / np.geterr / np.errstate (P2-10) — Accepted — 2026-10-02
- Native `native/core/fp_errors.{hpp,cpp}`: a thread-local `ErrState`
  (NumPy defaults divide/over/invalid = warn, under = ignore) and an RAII
  `FpScope` that clears the C99 FP exception flags (`<cfenv>`) on entry and
  reads them after the loop. Nested scopes (e.g. reduceat → reduce) defer to
  the outermost one.
- Wired into the generic `binary`/`unary` drivers (D-051) and the public
  `ufunc_reduce`/`accumulate`/`reduceat`/`at` entry points. Other native paths
  (`sum`, linalg, fft, ...) do not report FP errors yet.
- Integer kernels raise the flags explicitly, as NumPy does: `x // 0` and
  `x % 0` → divide; `MIN // -1` → over.
- Modes: `ignore`, `warn` (Node `process.emitWarning(msg, "RuntimeWarning")`),
  `raise` (new `FloatingPointError`, `ErrorKind::FloatingPoint`) and `print`
  (stdout). NumPy's `call`/`log` and `seterrcall` are not supported.
  Messages follow NumPy: "divide by zero encountered in divide", "overflow ...",
  "underflow ...", "invalid value ..."; categories are checked in NumPy's order.
  "raise" throws after the loop, so `out` may already be written (as in NumPy).
- Warnings are queued natively and emitted by the binding wrapper
  (`translate_errors`) after a successful call.
- JS: `np.seterr({all?, divide?, over?, under?, invalid?})` returns the old
  state, `np.geterr()`, and `np.errstate(settings, fn)` (synchronous callback
  instead of Python's context manager; restored in `finally`).
- Relies on the compiler not reordering FP ops across the flag reads (no
  `-ffast-math`); Clang/GCC defaults are fine for this.

## D-055 — Memory order for creation, copies and reshaping (P3-1) — Accepted — 2026-10-02
- Native `native/core/layout.{hpp,cpp}` reuses the D-050 `Order` enum and
  `keep_order_axes` (now public in `ufunc.hpp`). Build-first; NumPy
  differential checks are deferred to the V phase.
- Creation (`empty/zeros/ones/full`): `order` "C" (default) or "F"; anything
  else raises `ValueError("only 'C' or 'F' order is permitted")`.
- `np.array(ndarray)`/`np.copy`/`astype`/`*Like` default to "K" (keep the
  input's stride order, as NumPy); `NDArray.copy()` defaults to "C" as NumPy's
  `ndarray.copy`. "A" means F when F- and not C-contiguous.
- `reshape`/`ravel`/`flatten` take `{order}`; F works by reversing axes around
  a C reshape. `reshape` rejects "K". `ravel("K")` reads in memory order by
  sorting axes by |stride| without flipping negative strides (NumPy behaviour),
  and returns a view when that order is contiguous.
- JS has no keyword arguments, so `reshape(3, 2, {order})` accepts a trailing
  options object.
- `astype(dt, {order, copy})`: `copy: false` returns `this` when dtype and
  layout already match.
- `ascontiguousarray`/`asfortranarray` return at least 1-d (NumPy 2 behaviour).

## D-056 — Parallel milestone layout (P3–P15) — Accepted — 2026-10-02
- P3–P15 are built in parallel on branches `pNN` (git worktrees), then merged
  into `main`. To keep merges conflict-free, each milestone has its own files:
  - native bindings `native/bindings/pNN_binding.cpp` (exposed as `addon.pNN`,
    helpers in `binding_utils.hpp`); new native sources under `native/*/`
    (CMake globs `native/core`, `native/linalg`, `native/random`, `native/fft`,
    `native/bindings` and `tests/native`, so no CMake edits are needed);
  - TS `packages/numera/src/pNN.ts` (exported object `pNN` is spread into
    `np`; `export *` from index.ts), extra modules `pNN_*.ts` if large;
  - docs `docs/site/parts/pNN.mjs`; bench exemptions `api/bench-exempt/pNN.json`;
    aliases/exclusions `api/aliases.d/pNN.json`, `api/exclusions.d/pNN.json`;
  - slice table `docs/plan/slices/pNN.md`;
  - tests `packages/numera/test/pNN_*.test.ts`, `tests/native/test_pNN_*.cpp`.
- New ufuncs go in per-family tables: P4 `native/core/ufunc_math.cpp`, P5
  `native/core/ufunc_logic.cpp`, using the shared loop templates in
  `native/core/ufunc_loops.hpp` (moved out of `ufunc_registry.cpp`, no
  behaviour change). TS ufunc objects use the exported `binaryUfunc`/`unaryUfunc`.
- NDArray methods added outside P3 use TS declaration merging in the
  milestone's file instead of editing `ndarray.ts`.
- Decision numbers are reserved per milestone: P3 D-060–D-069, P4 D-070–D-079,
  … P15 D-180–D-189 (D-0(10·N+30)…).
- `DECISIONS.md`, `PROGRESS.md`, `COMPATIBILITY.md` are append-only on milestone
  branches (`merge=union` in `.gitattributes`); `ROADMAP.md`,
  `api/coverage*.json` and `TASK_SLICES.md` are updated only on `main`.
- Build-first rule (TASK_SLICES.md) is unchanged: no NumPy-compatibility or
  performance claim until the V phase.

## D-110 — Indexing extras API (P8) — Accepted — 2026-10-02
- Native kernels in `native/core/p08_indexing.{hpp,cpp}`, exposed as
  `addon.p08`; TS in `p08.ts` (and `take` in `indexing.ts`, owned by P8).
- `take(a, indices, axis?, {mode})` (also `take(a, indices, {axis, mode})`);
  `mode` is `"raise"` (default; negative indices wrap once), `"wrap"` or
  `"clip"` (clip maps negatives to 0, as NumPy). A non-empty take from an
  empty axis raises `IndexError`. Unknown modes raise `ValueError`.
- `put(a, ind, v, {mode})`, `putmask(a, mask, values)`, `place(arr, mask, vals)`
  and `putAlongAxis(arr, indices, values, axis)` mutate in place (write through
  non-contiguous views via flat C order) and return `undefined`; read-only
  targets raise `ValueError`.
- `choose(a, choices, {mode})`, `select(condlist, choicelist, {default})`,
  `compress(condition, a, {axis})`, `extract`, `piecewise(x, condlist, funclist)`
  (funclist entries are JS callbacks `(x: NDArray) => ArrayLike` or constants;
  callbacks run in JS on the masked selection, the selection and assignment
  are native).
- Results are always `NDArray`s, as D-017: `countNonzero(a)` without `axis`
  returns a 0-d int64 array (NumPy returns a Python int); `ravelMultiIndex` of
  scalars returns a 0-d int64 array; `unravelIndex` returns `NDArray[]` (one
  int64 array per dimension, like NumPy's tuple).
- `diagonal(a, {offset, axis1, axis2})` returns a **read-only view** (NumPy ≥
  1.9); `trace(a, {offset, axis1, axis2, dtype})` sums that view with the D-017
  sum dtype rules.
- NDArray methods `choose compress diagonal nonzero put take trace` are added
  by declaration merging in `p08.ts`.
- `nested_iters` stays excluded (api/exclusions.json).

## D-111 — Value casting in P8 setters and selectors — Accepted — 2026-10-02
- `put` / `putAlongAxis`: values cast unsafely to the target dtype (NumPy).
- `putmask` / `place`: an `NDArray` of values must cast to the target under
  `"safe"` casting, else `DTypeError` (NumPy `TypeError`). Nested JS arrays and
  scalars are converted with the target dtype first (like Python lists).
- `choose`: result dtype is the promotion of the array/nested choices; JS
  number/boolean scalar choices are weak (NEP 50, as D-014). `select`:
  `choicelist` entries are converted like `np.asarray` (strong), `default` is
  weak, as NumPy 2. NumPy wraps an out-of-range weak Python int silently
  (`int8` with 300 → 44); numera raises `ValueError` (documented divergence).
  `select` conditions must be bool arrays (`DTypeError`).
## D-100 — P7 creation helpers: JS encodings for slices and index helpers — Accepted — 2026-10-02
- JS has no `__getitem__`, so NumPy's index-trick objects are functions:
  `np.mgrid(...slices)`, `np.ogrid(...slices)`, `np.r_(...items)`,
  `np.c_(...items)`, `np.s_(...specs)`, `np.indexExp(...specs)`, `np.ix_(...seqs)`.
- `mgrid`/`ogrid` take slice tuples `[start, stop, step?]` (`start` may be
  `null` = 0). NumPy's complex step `5j` ("number of points, stop inclusive") is
  written as a complex step `np.complex(0, 5)` (any `{re, im}`; the count is
  `trunc(|step|)` as in NumPy). One slice returns a 1-D array (NumPy
  `mgrid[0:5]`); several return the dense `(N, ...)` array (`mgrid`) or an array
  of sparse arrays (`ogrid`). A zero step raises `ValueError` (Python
  `ZeroDivisionError`); a missing stop raises `ValueError` (Python `TypeError`).
- `r_`/`c_` items are arrays/nested lists (data), JS scalars (weak, NEP 50,
  as in ufuncs D-014) or strings. Because a JS list `[1, 4]` is data, a slice
  is written in Python syntax as a string containing `:` — `"1:4"`, `"::2"`,
  `"0:1:5j"` (complex step = linspace count). A first string without `:` is a
  directive (`"0,2"`, `"-1"`, `"1,2,0"`) with NumPy's meaning. The matrix
  directives `"r"`/`"c"` raise `NotImplementedError` (numera has no matrix class).
- `s_(...specs)` / `indexExp(...specs)` return index specs usable with
  `a.get(...)` / `a.slice(...)` (D-015). They also translate Python slice
  strings (`"1:3"` → `[1, 3, null]`, `":"` → `[null, null, null]`,
  `"..."` → `np.ellipsis`); every other spec is passed through unchanged. As in NumPy, `s_` with one spec returns that spec,
  `indexExp` always returns an array.
- `meshgrid(...xi, {indexing, sparse, copy})`, `ix_` and `indices(dims,
  {dtype, sparse})` return JS arrays of NDArrays where NumPy returns tuples.
  `meshgrid` with `copy: false` and `sparse: false` returns read-only
  broadcast views (D-016; NumPy returns writeable views with a deprecation
  warning on write).
- `np.astype(x, dtype, {copy})` (array API) accepts only an `NDArray`
  (others raise `DTypeError`, NumPy `TypeError`) and calls `x.astype`.

## D-101 — P7 frombuffer/fromstring/fromiter semantics — Accepted — 2026-10-02
- `frombuffer(buffer, {dtype, count, offset})` accepts an `ArrayBuffer`,
  `SharedArrayBuffer` or any `ArrayBufferView` (TypedArray, DataView, Buffer;
  its byte window is used). Data is **copied** (as `fromTypedArray`, PLAN §31),
  so the result is writeable and does not share memory with the buffer; NumPy
  returns a view (read-only for `bytes`). Error messages and checks follow
  NumPy (`offset` range, multiple of element size, buffer smaller than requested).
- `fromstring(string, {dtype, count, sep})` supports text mode only (non-empty
  `sep`); an empty/missing `sep` raises `ValueError` ("The binary mode of
  fromstring is removed, use frombuffer instead"), as NumPy 2. The parser is a
  native port of NumPy's `array_from_text` / `swab_separator` /
  `fromstr_skip_separator` and the per-dtype `fromstr` functions: integers via
  CPython `PyOS_strtol`/`PyOS_strtoul` (base 10, 64-bit `long` as on LP64, then
  truncated to the element width), floats via `NumPyOS_ascii_strtod` (POSIX
  inf/nan forms, decimal only), complex via NumPy's `CDOUBLE_fromstr` (including
  its quirks, e.g. `"j"` reads as `-1j`). Divergence: when `count` exceeds the
  number of items, NumPy returns uninitialized trailing elements; numera raises
  `ValueError` ("string is smaller than requested size").
- `fromiter(iterable, dtype, count = -1)`: `dtype` is required (as NumPy);
  values convert with the `np.array` rules (D-009). A short iterator with
  `count >= 0` raises `ValueError` ("iterator too short: Expected N but iterator
  had only M items."); a longer one is truncated.
## D-080 — Bool-output ufuncs: comparisons and logical ops (P5-1, P5-2) — Accepted — 2026-10-02
- New rows in `native/core/ufunc_logic.cpp` (registry names in camelCase, like
  `floorDivide`): `equal notEqual less lessEqual greater greaterEqual`,
  `logicalAnd logicalOr logicalXor logicalNot`. The loop takes the promoted input
  dtype and writes bool (`LoopTypes{promote(a, b), Bool}`); new bool-output loop
  templates live in `ufunc_logic.cpp`.
- Complex inputs compare lexicographically like NumPy's `CLT`/`CLE` macros
  (`(xr < yr && xi == xi && yi == yi) || (xr == yr && xi < yi)`). As in NumPy,
  the complex ordered comparisons raise the FP "invalid" flag when a NaN takes
  part in an ordered compare (`np.seterr` then reports "invalid value encountered
  in less"); real float comparisons and `equal`/`notEqual` stay quiet.
- `dtype=` on a call means NumPy's output signature: only `bool` is accepted
  (anything else raises `DTypeError: No loop matching ...`). For binary calls the
  TS wrapper validates it and then uses the default loop, so
  `equal([1], [1.5], {dtype: "bool"})` compares in float64 like NumPy.
- JS integer scalars outside an integer array's range (e.g. `less(int8Array, 1000)`)
  are not weak for the bool-output binary ufuncs: they become int64, uint64 or
  float64 arrays, so the comparison is value-exact like NumPy 2 (no overflow error).
- Logical ops take the truthiness of the promoted input dtype (same values as
  NumPy's bool cast). Divergence: NumPy's logical ufuncs accept mixed input dtypes
  under `casting: "no"/"equiv"/"safe"`; numera checks the cast to the promoted
  dtype, so e.g. `logicalAnd(int64, float64, {casting: "no"})` raises.
- uint64 vs int64 comparisons use the float64 loop (exact only up to 2^53);
  NumPy has dedicated mixed loops. Documented divergence.
- Reductions (`ufunc_methods.cpp`, D-052 loop dtype rule extended): when the
  resolved loop's input and output dtypes differ (a bool-output ufunc), a ufunc
  with an identity (`logicalAnd` True, `logicalOr`/`logicalXor` False) reduces in
  bool (inputs cast to bool, as NumPy's logical reduce); without an identity
  (comparisons) only bool input works, anything else raises
  `DTypeError: No loop matching ...` (NumPy behaviour, e.g. `equal.reduce(int)`).
- Identities are filled through an int64 scalar when integral, so `-1` fills
  uint32/uint64 accumulators with all ones (needed by `bitwiseAnd`, D-081).
- `np.all`/`np.any` (and NDArray methods) are `logicalAnd.reduce`/`logicalOr.reduce`
  with NumPy's defaults (axis `null` = all axes; `axis`, `keepdims`, `where`, `out`).

## D-081 — isnan/isinf/isfinite/isnat/isposinf/isneginf, isscalar (P5-3) — Accepted — 2026-10-02
- `isnan isinf isfinite` are unary bool-output ufuncs in `ufunc_logic.cpp`
  (D-080 templates) with loops for every dtype: integers and bool give the
  constant answer, complex is NaN/inf if either part is, finite if both are.
  `dtype=` accepts only `bool`.
- `isnat` is a registry ufunc whose resolver always raises `DTypeError`
  ("ufunc 'isnat' is only defined for np.datetime64 and np.timedelta64."),
  since numera has no datetime dtypes yet; it starts working when P14 adds them.
- `isposinf`/`isneginf` are NumPy functions, not ufuncs: `np.isposinf(x, {out})`.
  Natively they are registry rows (`isposinf`/`isneginf`) whose resolver rejects
  complex input with NumPy's `TypeError` message (as `DTypeError`); real dtypes
  get bool loops.
- `isscalar(x)`: true for JS `number`, `boolean`, `bigint`, `string` and complex
  scalars (`Complex` / `{re, im}`); false for `NDArray` (including 0-d), arrays,
  `null`/`undefined` and other objects. Matches NumPy for the Python analogues;
  numera has no NumPy scalar types (D-005).

## D-082 — Bitwise ufuncs (P5-4) — Accepted — 2026-10-02
- Rows in `ufunc_logic.cpp`: `bitwiseAnd bitwiseOr bitwiseXor` (bool and integer
  loops; identities -1 → all ones / True, 0, 0), `invert` (bool: logical not),
  `leftShift rightShift` (integer loops; bool promotes to int8, like NumPy's
  `??`-less loop table), `bitwiseCount` (integer/bool input, `uint8` output,
  popcount of `|x|` as NumPy).
- Shift semantics follow NumPy's `npy_lshift`/`npy_rshift`: a shift count that is
  `>=` the bit width or negative gives 0 (`rightShift` of a negative signed value
  gives -1), so there is no C++ undefined behaviour.
- Float/complex inputs (or a promotion to float, e.g. uint64 with int64) raise
  `DTypeError: ufunc '<name>' not supported for the input types, ...` (NumPy
  `TypeError`). `dtype=` must be bool or an integer dtype (integers only for
  shifts; `uint8` only for `bitwiseCount`).
- JS aliases mirror NumPy's identical objects: `bitwiseNot = bitwiseInvert = invert`,
  `bitwiseLeftShift = leftShift`, `bitwiseRightShift = rightShift`.

## D-083 — isclose / allclose / arrayEqual / arrayEquiv (P5-5) — Accepted — 2026-10-02
- Native kernel `isclose(a, b, rtol, atol, equal_nan)` in `native/core/p05_logic.cpp`
  (binding `addon.p05.isclose`). Loop dtype as NumPy: `b` is made inexact
  (`result_type(b, 1.0)`: bool/int → float64), then promoted with `a`.
  Result: `|a - b| <= atol + rtol * |b|` and `isfinite(b)`, or `a == b`; with
  `equal_nan`, also true where both are NaN (complex: either part NaN). float16
  is evaluated in float32 (NumPy rounds each step to float16; documented
  divergence, differences only at the tolerance boundary).
- `rtol`/`atol` are JS numbers (NumPy also accepts arrays; not supported yet).
  Non-finite tolerances report "One of rtol or atol is not valid, atol: ..., rtol: ..."
  through the `np.seterr` "invalid" mode, as NumPy.
- `isclose` returns an NDArray (0-d for scalar inputs, D-017 style), `allclose`
  a JS boolean (`all(isclose(...))`).
- `arrayEqual(a1, a2, {equalNan})` and `arrayEquiv(a1, a2)` return JS booleans
  and are composed in TS from the native `equal`/`isnan`/`all` kernels: inputs
  that cannot be converted give `false`; `arrayEqual` requires equal shapes,
  `arrayEquiv` broadcastable shapes (else `false`).

## D-084 — packbits / unpackbits (P5-6) — Accepted — 2026-10-02
- Native kernels in `native/core/p05_logic.cpp` (`addon.p05.packbits/unpackbits`).
  The input is copied C-contiguous (`packbits`: cast to bool, so any nonzero is
  a 1 bit) and processed as (outer, axis, inner) blocks; the output is `uint8`.
- `axis` default `null` flattens (NumPy `axis=None`); 0-d input acts like shape (1,).
  Out-of-range axes raise `IndexError` (NumPy `AxisError`).
- `packbits` input must be bool/integer (`DTypeError` "Expected an input array of
  integer or boolean data type"); `unpackbits` input must be `uint8`.
- `bitorder` is `"big"` (default) or `"little"`; other values raise `ValueError`
  with NumPy's messages ("'order' must be either 'little' or 'big'" for
  `packbits`, "'order' must begin with 'l' or 'b'" for `unpackbits`). numera
  accepts only the two full words, not NumPy's prefix match.
- `unpackbits` `count`: `null` → all bits; `>= 0` keeps that many (zero-padded
  past the end); negative drops `-count` bits, ValueError
  "-count larger than number of elements" when too negative.
## D-120 — np.sort / np.argsort and NDArray.sort/argsort (P9-1) — Accepted — 2026-10-02
- Native kernels in `native/core/p09_sorting.{hpp,cpp}` cover every dtype. The
  default kind is a port of NumPy's scalar introsort (`quicksort_`/`aquicksort_`:
  median-of-3, insertion sort for ranges ≤ 16, heapsort when the depth limit
  `2*msb(n)` runs out). Stable kinds use `std::stable_sort` with the same
  comparator; any stable sort gives the same output.
- Comparators follow NumPy's `npy::*_tag::less`: NaN sorts last (also in
  descending order); complex compares lexicographically with NumPy's NaN rules;
  float16 compares on bit patterns with ±0 equal.
- `kind`: the first letter is matched case-insensitively (q/h → quicksort,
  m/s → stable). Other values raise ValueError with NumPy's message. As in
  NumPy 2.5, heapsort runs the quicksort kernel. Passing `kind` together with
  `stable` or `descending` raises ValueError. `descending: true` uses NumPy's
  `greater` comparator.
- `axis`: default -1; `null` flattens (C order). 0-d input with an integer axis
  raises IndexError (AxisError analogue, D-012). `NDArray.sort` sorts in place,
  requires a writeable array and an integer axis; `axis: null` raises DTypeError
  (NumPy TypeError).
- argsort returns int64.

## D-121 — partition / argpartition (P9-2) — Accepted — 2026-10-02
- Port of NumPy's scalar `introselect_` (dumb_select when `kth-low < 3`, the
  max-scan for `kth == n-1` on inexact types, median-of-3 with
  median-of-medians-of-5 fallback, and the pivot stack across several kth).
  On arm64 NumPy also uses this scalar code, so results are intended to match
  element for element (to be verified in the V phase).
- `kth`: integer or list of integers (number[] / int NDArray). Booleans raise
  ValueError "Booleans unacceptable as partition index"; non-integers raise
  DTypeError "Partition index must be integer"; an empty list raises DTypeError;
  ndim > 1 raises ValueError. Negative kth adds the axis length; out of range
  raises ValueError `kth(=k) out of bounds (n)` when the array is not empty.
  kth values are sorted before use.
- `kind` must be exactly "introselect". `NDArray.partition` works in place.

## D-122 — lexsort, searchsorted, sortComplex (P9-3) — Accepted — 2026-10-02
- `lexsort(keys, {axis = -1})`: keys is a list of arrays or a 2-D+ array (one key
  per row). The last key is primary. Uses a stable merge of argsort passes
  (stable sort by each key from first to last). Error messages follow NumPy;
  `axis: null` raises DTypeError.
- `searchsorted(a, v, {side, sorter})`: `a` must be 1-D. The comparison dtype is
  the common dtype of `a` and `v` (JS scalars are weak, as in ufuncs); both are
  cast to it. Port of NumPy's batched binary search, using `less` for left
  and `less_equal` for right. Side and sorter errors use NumPy's messages. Scalar
  `v` gives a 0-d int64 result.
- `sortComplex(a)`: sorts along the last axis and returns complex64 for
  int8/uint8/int16/uint16 and complex128 for other real dtypes; complex input
  keeps its dtype.

## D-123 — unique and the Array-API unique_* functions (P9-4) — Accepted — 2026-10-02
- `unique(a, {returnIndex, returnInverse, returnCounts, axis, equalNan = true, sorted = true})`
  follows NumPy's `_unique1d`. It sorts, using a stable argsort when indices are
  requested, and masks adjacent duplicates. With `equalNan`, all NaNs, and for
  complex every NaN-containing value, form one group. When any return flag is
  set the result is an object `{values, indices?, inverse?, counts?}`;
  otherwise it is the values array.
- `sorted: false`: NumPy uses a hash table whose output order is an
  implementation detail. nativpy always returns sorted values (allowed by
  NumPy's documentation, "may be sorted in practice").
- `axis`: rows after `moveaxis(axis, 0)` are compared lexicographically using the
  element sort order. `equalNan` does not apply to rows, as in NumPy.
- `uniqueAll/uniqueCounts/uniqueInverse/uniqueValues` return
  `{values, indices, inverseIndices, counts}` subsets (camelCase of NumPy's
  namedtuple fields), with `equalNan: false`.

## D-124 — Set functions, isin, ediff1d (P9-5) — Accepted — 2026-10-02
- `intersect1d`, `union1d`, `setdiff1d`, `setxor1d` follow NumPy's
  `_arraysetops_impl` algorithms on top of the P9 sort/unique kernels. Inputs
  are promoted to a common dtype first (NumPy's concatenate rule).
  `intersect1d(..., {returnIndices: true})` returns `{values, indices1, indices2}`.
- `isin(element, testElements, {assumeUnique, invert, kind})` returns a bool array
  of `element`'s shape. kind is checked as in NumPy (null/"sort"/"table"; table
  only for bool/int input). Every kind gives the same result; the implementation
  sorts the test values and uses binary search.
- `ediff1d(a, {toEnd, toBegin})`: flattens; the result has the input dtype; bool
  input raises DTypeError; toBegin/toEnd must be castable to it with
  `same_kind` casting, otherwise DTypeError with NumPy's message.
## D-090 — P6 array manipulation: API shape and layouts (P6) — Accepted — 2026-10-02
- Kernels live in `native/core/p06_manip.{hpp,cpp}` (`addon.p06`); TS in
  `packages/numera/src/p06*.ts`. Build-first: NumPy differential and bench cases
  come in the V phase.
- Keyword arguments become a trailing options object; the most common keyword
  may also be passed positionally: `concatenate(arrays, axis | {axis, dtype,
  casting, out})` (`axis: null` flattens), `stack(arrays, axis | {...})`,
  `split(a, sectionsOrIndices, axis?)`, `repeat(a, repeats, axis?)`,
  `roll(a, shift, axis?)`, `flip(a, axis?)`.
- `concatenate`/`stack` result layout follows NumPy's multi-sorted stride
  permutation over the inputs (F inputs give an F result); `casting` defaults to
  "same_kind" and is checked for every input (`DTypeError`, NumPy `TypeError`);
  `out` and `dtype` together raise `DTypeError`.
- Functions that return a tuple in NumPy (`split` family, `unstack`,
  `broadcastArrays`, `atleast*d` with several inputs) return a JS array of
  NDArrays. `atleast1d/2d/3d(x)` with one argument returns one NDArray.
- `block` takes nested JS arrays whose leaves are NDArrays or scalars, and is
  implemented as NumPy's `_block_concatenate` (TS recursion over the native
  concatenate kernel), with NumPy's depth-mismatch and empty-list errors.
- Split pieces, `unstack`, `flip`, `rot90`, `rollaxis`, `permuteDims`,
  `matrixTranspose`, `atleast*d`, `trimZeros` return views; everything else
  returns new arrays.
- `delete` is a JS reserved word: it is exported as `np.delete` and as the
  named export `delete` (implemented as `del`).

## D-091 — P6 tile / repeat (P6) — Accepted — 2026-10-02
- `tile(a, reps)` and `repeat(a, repeats, axis?)` are native copy kernels;
  results are always new C-contiguous arrays (`tile` with all-ones reps still
  copies, like NumPy).
- `repeats` is an integer or an integer list (length 1 broadcasts); non-integer
  repeats raise `DTypeError` (NumPy `TypeError` under "safe" casting), negative
  values and length mismatches raise `ValueError` with NumPy's messages.
- `NDArray.prototype.repeat(repeats, axis?)` is added by declaration merging in
  `p06_tile.ts`.

## D-092 — P6 pad (P6) — Accepted — 2026-10-02
- `pad(a, padWidth, mode?, options?)` follows NumPy `_arraypad_impl`: the
  result has `a`'s dtype, layout F if `a` is F- and not C-contiguous (else C),
  and axes are padded in order, each over the region already padded on earlier
  axes. The native kernel processes every 1-d lane along the current axis with
  NumPy's per-mode rules (iterative reflect/symmetric/wrap, `stat_length`
  clipping, integer stats rounded half-to-even, `reflect_type: "odd"`).
- Options use camelCase keys: `constantValues`, `endValues`, `statLength`,
  `reflectType`; `padWidth`/values accept NumPy's scalar, pair, per-axis-pairs
  and `{axis: width}` forms. Unsupported keys for a mode and unknown modes raise
  `ValueError`; a non-integer `padWidth` raises `DTypeError`.
- Divergences (build-first, checked in the V phase): `linear_ramp` and `mean` /
  `median` are computed in float64 (complex128) and cast once, while NumPy may
  compute in the array's float dtype; `mode: "empty"` fills with zeros instead
  of leaving memory uninitialised.
- A JS function as `mode` is called NumPy-style as `fn(vector, [before, after],
  axis, options)` on writable 1-d views of a zero-padded result.

## D-093 — P6 append / insert / delete / resize / trimZeros (P6) — Accepted — 2026-10-02
- `insert(arr, obj, values, axis?)` and `delete(arr, obj, axis?)` follow NumPy
  `_function_base_impl`: `axis` omitted/null flattens; `obj` is an integer, an
  integer or boolean list/array, or a slice written `{start, stop, step}` (JS
  has no slice syntax; a plain array is always a list of indices). TS
  normalises `obj` to result positions / a keep mask; the native kernels
  `insert_along` / `delete_along` move the data. Results are new arrays, F
  order if `arr` is F- and not C-contiguous, else C (also for array `obj` to
  `delete`, where NumPy's advanced-indexing layout may differ). `values` are
  cast unsafely to `arr.dtype`, as in NumPy's item assignment.
- `append(arr, values, axis?)` is `concatenate` (flattening both when `axis`
  is omitted), so it promotes dtypes.
- `np.resize(a, shape)` cycles the flattened data (zeros if `a` is empty).
  `NDArray.prototype.resize(shape, {refcheck})` resizes in place like
  `ndarray.resize`: the array must own C/F-contiguous, writeable data; the data
  is truncated or zero-extended in memory order and the wrapper switches to the
  new buffer (the method returns `undefined`). With `refcheck` (default true)
  it raises `ValueError` while another live NDArray (e.g. a view not yet
  garbage-collected) shares the buffer; with `refcheck: false` existing views
  keep the old data instead of dangling.
- `trimZeros(filt, trim="fb", axis?)` trims every selected axis to the bounding
  box of nonzero elements (NumPy 2.2+ N-d behaviour) and returns a view.

## D-094 — P6 conversions and metadata helpers (P6) — Accepted — 2026-10-02
- `copyto(dst, src, {casting = "same_kind", where})` casts every element under
  `casting` (`DTypeError` otherwise; NumPy `TypeError`), broadcasts `src` and
  `where` to `dst` (`BroadcastError` / `ValueError`) and checks writeability.
  JS scalars follow NEP 50 weak-scalar rules: a JS number is weak int64 if it
  is a safe integer, else weak float64; booleans are bool; a weak integer must
  fit `dst`'s integer dtype (`ValueError`, NumPy `OverflowError`) and may not
  go into bool except under "unsafe". A non-bool `where` array raises
  `DTypeError`; nested JS lists for `where` are converted to bool first.
- `asanyarray` is `asarray` (no subclasses). `asarrayChkfinite` raises
  `ValueError("array must not contain infs or NaNs")` using a native scan.
- `require(a, dtype?, requirements?)` accepts NumPy's flag letters/names
  (C/C_CONTIGUOUS/CONTIGUOUS, F/F_CONTIGUOUS/FORTRAN, A/ALIGNED, W/WRITEABLE,
  O/OWNDATA, E/ENSUREARRAY) as a string of letters or a list; unknown flags
  raise `ValueError` (NumPy `KeyError`). Returns the input unchanged when it
  already satisfies them.
- `broadcastArrays(...arrays)` returns writeable broadcast views (NumPy marks
  them writeable with a FutureWarning; writes through zero strides alias).
  The read-only flag is not set, matching current NumPy results.
- `shape`, `size(a, axis?)`, `ndim`, `isfortran` accept any array-like.

## D-150 — `out=` for every `np.fft` transform (P12-1) — Accepted — 2026-10-02
- Every transform (`fft ifft rfft irfft hfft ihfft fft2 ifft2 fftn ifftn rfft2
  irfft2 rfftn irfftn`) accepts `out` as a positional parameter after `norm`
  or in the trailing options object (`{ n|s, axis|axes, norm, out }`). With
  `out` the function writes into it and returns that same `NDArray` object.
- Compute precision (supersedes the D-020 "float16/float32/complex64 compute in
  float32" rule for every transform, with or without `out`): numera now picks the
  pocketfft loop the way NumPy's ufunc type resolution does (checked with
  `_pocketfft_umath.*.resolve_dtypes`). The factor `fct` is 1 (a Python int) when
  the effective norm is "backward" (after the inverse swap), otherwise it is
  rounded to `result_type(a.real.dtype, 1.0)` (float16 / float32 / float64).
  The float32 loop runs only when the input already has the loop's single dtype
  (complex64 for `fft`/`ifft`/`irfft`, float32 for `rfft`) **and** `fct` is a
  float32 scalar; everything else (float16, float32 real input to `fft`, `norm`
  giving `fct = 1`, integers, bool) runs the float64 loop with the rounded `fct`.
  The loop result is then cast to the result dtype (unchanged from D-020:
  complex64/complex128; `irfft` float16/float32/float64) or to `out`.
- Each 1-D step follows NumPy's `_raw_fft`, and with `out` writes into it:
  - `out.ndim` must equal the input's and `out.shape[axis]` the output length,
    otherwise `ValueError` "output array has wrong shape.".
  - The result dtype must cast to `out.dtype` under `same_kind`, otherwise
    `DTypeError` "Cannot cast ufunc 'fft' output from complex128 to float64 with
    casting rule 'same_kind'" (NumPy: `UFuncTypeError`).
  - A read-only `out` raises `ValueError` "output array is read-only".
  - The other dimensions broadcast from the result to `out` (NumPy broadcasts the
    input loop dimensions); a mismatch raises `BroadcastError` (NumPy: `ValueError`).
  - `out` may alias the input (`fft(c, { out: c })`).
- Multi-axis transforms pass `out` to every 1-D step, as NumPy does
  (`fftn`/`fft2`/`rfftn`/`rfft2`: each step writes into `out` and the next step
  reads from it, so intermediate results are rounded to `out.dtype`, and an `s`
  that changes a non-final shape makes `out` fail the shape check, as in NumPy).
  `irfftn`/`irfft2` pass `out` only to the final `irfft` step. `fftn(a, {axes: []})`
  returns `a` and leaves `out` untouched.
- `s[i] = -1` resolves to the *original* input length along `axes[i]` (NumPy);
  the M10 `fftn` used the current length, which differed for repeated axes.
- Implementation: `native/fft/p12_fft.{hpp,cpp}` has its own pocketfft lane
  kernels (ported from M10 with an explicit loop type), exposed as `addon.p12`.
  `fft.ts` routes every transform through it; the M10 `addon.fft` transform
  bindings stay in place but are no longer called from TS.

## D-151 — hfft/ihfft and real N-D transforms (P12-2) — Accepted — 2026-10-02
- `hfft(a, n?, axis=-1, norm?, out?)` = `irfft(conj(a), n, axis, swap(norm))`, default
  `n = 2 * (m - 1)`; `ihfft(a, n?, axis=-1, norm?, out?)` = `conj(rfft(a, n, axis,
  swap(norm)))` (conjugated in place in `out`), default `n = m`. `swap` exchanges
  "backward" and "forward". dtypes follow `irfft`/`rfft` (hfft float16 → float16,
  complex64 → float32; ihfft float16/float32 → complex64; complex input to
  `ihfft` raises `DTypeError` as `rfft` does).
- `rfftn(a, s?, axes?, norm?, out?)`: `rfft` over the last axis in `axes`, then
  `fft` over the others from last to first. `irfftn`: `ifft` over all axes but
  the last (from first to last), then `irfft` over the last. `rfft2`/`irfft2`
  are the same with `axes` default `[-2, -1]`.
- `s`/`axes` follow `_cook_nd_args`: no `s` and no `axes` = all axes; `s`
  without `axes` = the last `len(s)` axes (no deprecation warning, D-020);
  `s[i] = -1` = the input length; for `irfftn` without `s` the last length is
  `2 * (m - 1)`. An empty `axes` raises `IndexError` (NumPy: `IndexError` "list
  index out of range"); 0-d input raises `IndexError`.

## D-152 — fftshift / ifftshift (P12-3) — Accepted — 2026-10-02
- `fftshift(x, axes?)` / `ifftshift(x, axes?)`: `axes` is `null`/omitted (all
  axes), a number, or a number array (repeated axes add up, as `np.roll`). The
  shift is `shape[ax] // 2` (`ifftshift`: its negative). Any dtype; the result
  is a new C-contiguous array of the input dtype, computed by a native roll
  (`addon.p12.roll`).
- An empty `axes` list returns a copy. 0-d input with default or empty axes raises
  `ValueError` (NumPy fails in `np.roll` with "not enough values to unpack").
  Out-of-range axes raise `IndexError` (D-012).
## D-070 — P4 math ufuncs: registry rows and type resolution — Accepted — 2026-10-02
- Every P4 element-wise ufunc that NumPy defines as a ufunc is a row in
  `native/core/ufunc_math.cpp` (D-051/D-056) and a TS object from
  `binaryUfunc`/`unaryUfunc`, so it gets `out/where/dtype/casting/order` and
  `reduce/accumulate/outer/at`. Identities follow NumPy (`hypot` 0,
  `logaddexp`/`logaddexp2` -inf, `gcd` 0, others none).
- Type resolution mirrors NumPy's loop lists (`np.<ufunc>.types`):
  - float-only ufuncs (trig, exp/log family, `arctan2 hypot deg2rad ...`):
    integer/bool inputs use the smallest safe float (`float_for`: 8-bit →
    float16, 16-bit → float32, wider → float64); binary ones promote the two
    float loop types. Complex loops exist exactly where NumPy has them
    (e.g. `sin`, `exp2`, `log10`, `square`, `sign`; not `arctan2`, `cbrt`,
    `deg2rad`); otherwise complex input raises `DTypeError`.
  - `dtype=` accepts only dtypes NumPy has a loop for (float, or inexact when
    a complex loop exists); others raise the "No loop matching" `DTypeError`.
- Array-API aliases (`asin` … `atan2`, `pow`) are the same JS objects.
- The long-double (`g`/`G`) and object loops have no numera dtype and are out
  of scope. libm results are not claimed bit-identical to NumPy (D-014).

## D-071 — P4 np.round / around / fix — Accepted — 2026-10-02
- `np.round(a, decimals = 0, { out })` (alias `around`, method `a.round`) is a
  TS composition like NumPy's `PyArray_Round`: floats/complex use `rint` for
  `decimals = 0`, else `rint(a * 10**d) / 10**d` (or `* 10**-d` for negative
  `d`), so results carry the same floating-point error as NumPy. Integer
  inputs return a copy for `decimals >= 0`; negative `decimals` round through
  float64 and cast back to the input dtype (wrapping like NumPy, e.g. int8
  127 → -126). `bool` with `decimals = 0` gives float16 (`rint`), with other
  decimals raises `DTypeError` (NumPy's casting error). Non-integer `decimals`
  raise `DTypeError`.
- `np.fix` is `np.trunc` (NumPy 2.5 deprecates `fix` in favour of `trunc` and
  returns the same values and dtypes).
- `positive` has no bool loop (NumPy raises `UFuncTypeError`; numera `DTypeError`).

## D-072 — P4 arithmetic: aliases, clip, extrema — Accepted — 2026-10-02
- `remainder`, `trueDivide`, `pow`, `absolute` are the existing `mod`,
  `divide`, `power`, `abs` objects (NumPy aliases), so `np.remainder === np.mod`.
- New registry rows: `fmod` (C remainder; integer `x % 0` → 0 with the
  divide-by-zero FP flag, `MIN % -1` → 0), `float_power` (float64/complex128
  loops only), `sign` (NaN → NaN; complex `z/|z|` with NumPy 2's special
  cases; no bool loop), `heaviside`, `fabs`, `maximum minimum fmax fmin`
  (all dtypes incl. bool and complex lexicographic order; `maximum`/`minimum`
  propagate NaN, `f*` ignore it; signed zeros: max → +0, min → −0).
  NumPy gives `maximum`/`minimum` no identity, so their `.reduce` on empty
  input raises, as in NumPy.
- `np.clip(a, min?, max?, { out })` and `a.clip` compose `minimum(maximum(a,
  min), max)` like NumPy's `_clip` (NaN propagates; `min > max` → `max`;
  both bounds omitted → `positive(a)`, which raises for bool like NumPy).
  NumPy's dedicated `clip` ufunc (and its `out`/`where` keywords beyond `out`)
  is not exposed as a ufunc object.
- `a.conjugate()` is added by declaration merging (same as `a.conj()`).

## D-073 — P4 float bits, gcd/lcm and multi-output ufuncs — Accepted — 2026-10-02
- Registry rows: `copysign`, `nextafter`, `spacing` (float loops; float16 uses
  NumPy's bit-pattern `npy_half_nextafter`/`npy_half_spacing`, whose spacing
  steps toward +inf), `ldexp` (x picks the float loop; the exponent must be
  bool/int/uint ≤ 32 bits or int64 — NumPy's `fi`/`fl` loops — and is cast
  to the float type and clamped to ±100000 before `std::ldexp`), `signbit`
  (float in, bool out), `gcd` (identity 0) and `lcm` (integer loops only,
  modular on magnitudes like NumPy).
- `divmod`, `modf`, `frexp` have two outputs, which the single-output
  registry cannot express. They use a dedicated path
  (`native/core/p04_multi.cpp`, `addon.p04.multi`) and return
  `[NDArray, NDArray]` in TS. They support `out: [o1 | null, o2 | null]`,
  `dtype`, `casting`, not `where`/`order` or ufunc methods (`reduce` etc.).
  Loops: divmod uses NumPy's floor_divide/remainder kernels (bool → int8);
  modf/frexp are float-only (ints → `float_for`); the frexp exponent is int32
  (0 for inf/nan).

## D-074 — P4 special functions i0, sinc, unwrap, nanToNum, realIfClose — Accepted — 2026-10-02
- NumPy implements these in Python, not as ufuncs; numera exposes them as
  plain functions (no `out`/`where`/ufunc methods) with the numerics in
  `native/core/p04_special.cpp` (`addon.p04`).
- `i0` uses NumPy's Cephes Chebyshev coefficients; `sinc` uses NumPy's
  `where(x, x, eps)` substitution for zeros (1e-20 for complex). Both use
  NumPy's output dtypes (`i0`: float16/float32 kept, other real inputs
  float64, complex raises `DTypeError`; `sinc`: ints → float64).
- `unwrap` follows NumPy's algorithm including the integer path (integer
  input and integer period keep the dtype) and bool `diff` = `not_equal`.
  float16 input is computed in float32 and cast back (NumPy computes in
  float16, so the last bit may differ). Complex input raises `DTypeError`.
- `nanToNum` accepts scalar `nan`/`posinf`/`neginf` only (NumPy also allows
  arrays); `copy: false` needs an NDArray and modifies it in place.
- `realIfClose(x, tol = 100)` returns the `real` view (NumPy also returns
  `a.real`).

## D-140 — P11 linear algebra completion: API and dtypes — Accepted — 2026-10-02
- Native kernels in `native/linalg/p11_linalg.{hpp,cpp}` (`addon.p11`); TS in
  `p11*.ts`; the new `np.linalg` names are added to the `linalg` object in
  `linalg.ts` (owned by P11).
- Results are always `NDArray`s (D-017): `matrixRank`, `cond`, `vdot`,
  `multiDot` of two vectors, ... return 0-d arrays where NumPy returns scalars.
  `slogdet` returns `{ sign, logabsdet }` (NumPy `SlogdetResult`).
- New decompositions (`cholesky`, `slogdet`, the singular-matrix-tolerant
  inverse used by `cond`) compute in float64 / complex128 and cast to NumPy's
  `_commonType` result type at the end (float32 for float32, complex64 for
  complex64, float64 for bool/int; float16 → `DTypeError`), as NumPy's
  `'d->d'` / `'D->D'` signatures. Functions composed from the existing
  `svd`/`eigh`/`inv`/`norm` (`pinv`, `matrixRank`, `svdvals`, `cond` with
  p = None/±2, `matrixPower` with n < 0) keep those functions' dtype rules (D-018).
- Keyword arguments are option objects: `cholesky(a, {upper})`,
  `pinv(a, {rcond, hermitian, rtol})` (`rtol: null` = API-standard default
  `max(M, N) * eps`; `rcond` and `rtol` together → `ValueError`),
  `matrixRank(A, {tol, hermitian, rtol})`, `cond(x, p?)`,
  `vectorNorm(x, {axis, keepdims, ord})`, `matrixNorm(x, {keepdims, ord})`,
  `linalg.diagonal(x, {offset})`, `linalg.trace(x, {offset, dtype})`,
  `tensorinv(a, {ind})`, `tensorsolve(a, b, {axes})`,
  `cross(a, b, {axisa, axisb, axisc, axis})`, `linalg.cross(x1, x2, {axis})`,
  `vecdot(x1, x2, {axis})`, `tensordot(a, b, {axes})` (also positional `axes`).
  `rcond`/`tol`/`rtol` accept a number or an array that broadcasts against the stack.
- `matrixPower(a, n)`: `n` must be a safe integer (`DTypeError`, NumPy
  `TypeError`). `n = 0` gives the identity in `a`'s dtype, `n < 0` inverts first.
- `vecdot`/`matvec`/`vecmat` are plain functions (not ufunc objects; no
  `out`/`where`/`reduce`); they conjugate the vector argument for complex
  input like NumPy (`vecdot`: x1, `vecmat`: the vector) and broadcast batch dims.
- `NDArray.dot(b)` is added by declaration merging (= `np.dot(a, b)`).
- `np.matrix_transpose` is P6's (reorder family); P11 adds only
  `linalg.matrixTranspose`. Batched `lstsq` is not added: NumPy 2.5.3 still
  rejects stacked `a`.

## D-141 — P11 einsum / einsumPath — Accepted — 2026-10-02
- `einsum(subscripts, ...operands, opts?)` and the sublist form
  `einsum(op0, sub0, op1, sub1, ..., [outSub], opts?)` (sublists are arrays of
  integers 0–51 and `np.ellipsis`). A trailing plain object (not an array,
  NDArray or complex value) is the options object `{ optimize }`.
- Parsing (explicit `->`, implicit output = labels seen once in sorted order,
  `...` broadcasting, error messages) and path search (`greedy`, `optimal`,
  explicit `["einsum_path", [i, j], ...]`, `[name, memoryLimit]`) are ports of
  NumPy's `einsumfunc.py`, done in TS (shape bookkeeping only).
  `einsumPath` returns `[path, report]` with NumPy's report text.
- The numerics run in C++ (`p11_linalg.cpp::einsum`): repeated labels in an
  operand take a strided diagonal view; size-1 dimensions broadcast (stride 0);
  labels used by only one operand are summed out by a product with a ones
  vector; each pairwise contraction is a transpose/reshape to
  `(batch, M, K) @ (batch, K, N)` on the existing `matmul` (backend GEMM for
  float32/float64/complex, the exact loop otherwise). Operands are cast to the
  promoted dtype first, so integers wrap and bool is OR-of-AND as NumPy's
  einsum. Without `optimize` the operands are still contracted pairwise left to
  right (results agree with NumPy within floating-point rounding).
- The result is always a new array; NumPy may return a view (`'ii->i'`,
  `'ij->ji'`). No `out=`, `dtype=`, `order=`, `casting=`.

## D-142 — Cholesky on both backends without changing the backend interface — Accepted — 2026-10-02
- `Routines<T>` (backend.hpp) has no `potrf`. To avoid editing shared backend
  files on the P11 branch, `p11_linalg.cpp` dispatches on
  `active_backend().name()`: `"accelerate"` calls Accelerate `dpotrf_`/`zpotrf_`
  (compiled under `NATIVPY_HAVE_ACCELERATE`), every other backend uses a
  portable column Cholesky (lower: Cholesky–Banachiewicz; upper: L of Aᴴ then
  Lᴴ). A non-positive or NaN pivot raises `LinAlgError("Matrix is not positive
  definite")`. Only the lower (upper with `upper: true`) triangle is read; the
  other triangle of the result is zero.
## D-060 — NDArray methods fill/tolist/tobytes/view/byteswap/setflags/base/mT/flat (P3-2) — Accepted — 2026-10-02
- Native kernels in `native/core/p03_methods.{hpp,cpp}`, bound in `p03_binding.cpp`.
  Build-first: NumPy differential and bench cases are deferred to the V phase.
- `base` (NumPy identity semantics, `a[1:][1:].base === a`): the native addon keeps a
  per-env map buffer → owning `NDArrayWrap*` (plain C++ map, inserted when an owning
  array is wrapped, erased in its GC-time destructor without any N-API call, D-023).
  When a non-owning result is wrapped, the owner's JS handle is stored as a hidden
  property on the view's handle, so a view keeps its base alive, as in NumPy.
  (JS `WeakRef`s were rejected: they keep every array alive until the end of a
  synchronous job, which defeats D-023.)
  The TS `NDArray` stores itself on its native handle so `base` returns the same
  object. Arrays that own their data, and views whose buffer has no wrapped owner
  (e.g. `imag` of a real array), have `base === null`. Shared-file edits (minimal):
  `ndarray.hpp` (`set_writeable`) and `ndarray_binding.{hpp,cpp}` (owner map,
  `base()`/`setWriteable()` handle methods).
- `setflags({write})`: `write: false` always works; `write: true` raises
  `ValueError("cannot set WRITEABLE flag to True of this array")` when the array's
  owner is read-only (NumPy rule). `align`/`uic` are not supported (`NotImplementedError`).
- `view(dtype?)`: NumPy 2 rules — same itemsize: dtype swap; 0-d with a different
  itemsize, a non-contiguous last axis (unless its length is 1 or the array is empty),
  or a last-axis byte size not divisible by the new itemsize raise `ValueError` with
  NumPy's messages. Views inherit writeability.
- `tobytes({order})` returns a `Uint8Array` copy. `"C"`/`"K"` → C order (NumPy's
  tobytes treats K as C), `"F"` → F, `"A"` → F if F- and not C-contiguous.
- `byteswap({inplace})`: per-element byte reversal (complex: per component); the copy
  keeps the input layout (K). In place on a read-only array raises
  `ValueError("array to be byte-swapped is read-only")`.
- `fill(value)`: JS scalar (or size-1 array) cast like `np.array(value, {dtype})`
  (D-009: out-of-range ints / NaN to int raise `ValueError`, floats truncate).
- `tolist()` is `toArray()` (D-005 applies).
- `mT`: swaps the last two axes; `ndim < 2` raises
  `ValueError("matrix transpose with ndim < 2 is undefined")`.
- `flat` returns a new `FlatIter` (NumPy `flatiter`): `get(i)` (number → JS scalar;
  slice tuple, int array or bool mask → 1-D/shape-of-index copy), `set(i, value)`
  (values repeat cyclically, as NumPy), `base`, `index`, `coords`, `length`, `copy()`,
  `toArray()`, and the iterator protocol (yields JS scalars in C order, read live).
  `a.flat = v` assigns cyclically to every element.
- `astype(dt, {order, copy, casting})`: `casting` (default `"unsafe"`) is checked
  with `canCast`; a disallowed cast raises `DTypeError` ("Cannot cast array data
  from dtype('float64') to dtype('int32') according to the rule 'safe'").

## D-061 — ndindex / ndenumerate / nditer (P3-3) — Accepted — 2026-10-02
- Native `native/core/p03_iter.{hpp,cpp}` computes the iteration plan: broadcast
  shape, the axis order (outermost → innermost) and per-axis flips. Orders:
  `"C"` row-major, `"F"` column-major, `"A"` F if every operand is F-contiguous
  else C, `"K"` (default) NumPy's memory order: axes sorted by |stride|
  (`keep_order_axes`, D-050) and an axis is flipped when no operand has a positive
  stride on it and at least one has a negative stride (NumPy
  `npyiter_flip_negative_strides`). TS walks the plan and builds 0-d read-only views.
- `np.nditer(op | op[], {flags, order, opFlags})` returns an `NDIter` (JS has no
  context manager or Python iterator object; it is a JS iterable). Each step
  yields a 0-d read-only `NDArray` view (one operand) or an array of them.
  Supported flags: `multi_index`, `c_index`, `f_index`, `zerosize_ok`
  (camelCase spellings accepted). `buffered`, `external_loop`, `reduce_ok`,
  `refs_ok`, `ranged`, `delay_bufalloc`, `grow_inner`, `copy_if_overlap`,
  `common_dtype` raise `NotImplementedError`; unknown flags raise `ValueError`
  (NumPy message). `opFlags` other than `readonly` raise `NotImplementedError`.
- Members: `shape`, `ndim`, `nop`, `itersize`, `iterindex` (get/set), `multiIndex`
  (get/set, coordinates in operand space even when an axis is flipped), `index`
  (C or F flat index of the broadcast shape), `finished`, `value`, `operands`,
  `iternext()`, `reset()`. Errors follow NumPy: missing `multi_index` / index
  → `ValueError`, out-of-range `iterindex`/`multiIndex` → `IndexError`, zero-size
  operands without `zerosize_ok` → `ValueError`, broadcast mismatch → `BroadcastError`.
- `np.ndindex(...shape)` (numbers or one shape array) and `np.ndenumerate(a)` are
  generators of `number[]` / `[number[], scalar]` in C order (scalars are JS values,
  D-005).

## D-062 — dtype introspection: finfo, iinfo, resultType, minScalarType, issubdtype, isdtype, commonType, mintypecode, abstract dtypes (P3-4) — Accepted — 2026-10-02
- `np.finfo(dt | array)` / `np.iinfo(dt | array)` return frozen `FInfo` / `IInfo`
  objects computed natively (`native/core/p03_dtypes.cpp`, `std::numeric_limits`
  plus binary16 constants). `FInfo` has NumPy's fields in camelCase (`bits, eps,
  epsneg, max, min, tiny, smallestNormal, smallestSubnormal, resolution,
  precision, iexp, nexp, nmant, machep, negep, minexp, maxexp, dtype`); float
  values are JS numbers holding the dtype-rounded value (e.g. float32
  `resolution` is `float32(1e-6)`). Complex dtypes report their component
  dtype. `IInfo` has `bits, min, max, dtype, kind`; `min`/`max` are numbers
  (lossy above 2^53, D-005) and `minExact`/`maxExact` are exact bigints.
  Wrong kinds raise `ValueError` with NumPy's messages.
- `np.resultType(...args)`: DTypes, dtype names and arrays (0-d included) are
  strong and promoted with `promoteTypes`; JS scalars are weak (NEP 50, the same
  rules as ufunc operands, D-014): the strongest scalar kind (bool < int < float
  < complex) only raises the result's kind (int → int64, float → float64,
  complex → complex64 for float16/float32, complex128 otherwise), never its size.
  No argument raises `ValueError("at least one array or dtype is required")`.
- `np.minScalarType(x)`: arrays with ndim > 0 return their dtype. 0-d arrays and
  JS scalars (converted like `np.array(x)`) are value-based, natively: smallest
  unsigned integer for values ≥ 0, smallest signed for negative ones; floats
  → float16 when non-finite or in (−65000, 65000), float32 in (−3.4e38, 3.4e38),
  else float64; complex → complex64 when both parts are in (−3.4e38, 3.4e38).
  JS numbers count as integers only when they are safe integers (|x| < 2^53);
  other numbers are floats (use a bigint for larger integers). Bigints that fit
  no integer dtype (NumPy `object`) raise `ValueError`.
- Abstract dtypes `np.generic, number, integer, signedinteger, unsignedinteger,
  inexact, floating, complexfloating` are frozen `AbstractDType` objects
  (`name`, `parent`); NumPy's scalar-type hierarchy, with `bool` directly under
  `generic`. They are only meaningful for `issubdtype`.
- `np.issubdtype(a, b)`: `a`, `b` are DTypeLike or AbstractDType. Concrete `b` →
  equality; abstract `b` → `a` (or its kind's abstract type) descends from `b`.
- `np.isdtype(dtype, kind)`: `dtype` must be a `DType` (strings raise
  `DTypeError`, as NumPy's TypeError); `kind` is a DType, one of NumPy's kind
  names, or an array of them. AbstractDType kinds raise `DTypeError`.
- `np.commonType(...arrays)` returns a `DType` (NumPy returns a scalar type):
  integers count as float64; bool raises `DTypeError`; no argument → float16.
- `np.mintypecode(typechars, typeset = "GDFgdf", default = "d")`: NumPy's
  algorithm with its type characters (int64 = `l`, uint64 = `L`). Strings are
  iterated per character; array entries use their dtype's character.

## D-063 — Array printing: dragon4, array2string/arrayRepr/arrayStr, print options, toString = repr (P3-5) — Accepted — 2026-10-02
- `native/core/p03_print.cpp` ports NumPy's Dragon4 (`dragon4.c`: the
  `Dragon4` digit generator in unique/exact mode with total/fraction-length
  cutoffs, `FormatPositional`, `FormatScientific`, trim modes `k . 0 -`) on a
  small arbitrary-precision integer. Values are decomposed in their own dtype
  (binary16/32/64), so float16/float32 print their shortest unique digits.
- `np.formatFloatPositional(x, {precision, unique, fractional, trim, sign,
  padLeft, padRight, minDigits})` and `np.formatFloatScientific(x, {precision,
  unique, trim, sign, padLeft, expDigits, minDigits})`: `x` is a JS number
  (float64) or a 0-d/size-1 NDArray (its float dtype; ints and bool convert to
  float64; complex raises `DTypeError`). NumPy's argument errors map to
  `ValueError`/`DTypeError` (for its TypeErrors).
- Element formatting (NumPy `FloatingFormat`, `ComplexFloatingFormat`,
  `IntegerFormat`, `BoolFormat`, summarization corners) runs natively and
  returns the C-order strings of the summarized array; TS does the line layout
  (`_formatArray`/`_extendLine`) and the repr/str wrappers, which only
  concatenate strings.
- `np.array2string(a, {maxLineWidth, precision, suppressSmall, separator,
  prefix, suffix, formatter, threshold, edgeitems, sign, floatmode, legacy})`,
  `np.arrayRepr(a, {maxLineWidth, precision, suppressSmall})`,
  `np.arrayStr(a, {...})` follow NumPy 2.x defaults (legacy=False). `formatter`
  is an object of JS callbacks keyed by NumPy's names (`all`, `bool`, `int`,
  `float`, `complexfloat`, `int_kind`, `float_kind`, `complex_kind`; camelCase
  accepted); callbacks get the JS scalar (`item()`) and must return a string.
  `legacy` other than `false` raises `NotImplementedError`.
- Print options are module state: `np.setPrintoptions(opts)` (`precision,
  threshold, edgeitems, linewidth, suppress, nanstr, infstr, sign, floatmode,
  formatter, legacy, overrideRepr`; `formatter`/`overrideRepr` reset on every
  call like NumPy; `undefined`/`null` = unchanged), `np.getPrintoptions()`
  (a copy), and `np.printoptions(opts, fn)` — NumPy's context manager becomes a
  callback: options apply while `fn()` runs and are restored afterwards (also
  on throw); returns `fn`'s result.
- `NDArray.toString()` is `np.arrayRepr(this)` (NumPy `repr`), e.g.
  `array([1, 2])`; the old JSON-like `array([1,2], dtype=int64)` form is gone.
  0-d `arrayStr` uses NumPy scalar `str` rules (float16/32/64 positional below
  1e3/1e6/1e16 and ≥ 1e-4, complex `(a+bj)`).
## D-170 — P14 NPY / NPZ binary I/O (P14) — Accepted — 2026-10-02
- `.npy` encoding/decoding lives in C++ (`native/core/p14_npy.{hpp,cpp}`):
  header v1.0 (v2.0 when the header exceeds 65535 bytes), byte-for-byte the
  layout of `numpy.lib.format.write_array` (sorted dict keys, 21-digit growth
  padding, 64-byte alignment). Descriptors `|b1 |i1 |u1 <i2 <u2 <i4 <u4 <i8 <u8
  <f2 <f4 <f8 <c8 <c16`; loading also accepts `>`/`=` byte orders (big-endian
  data is byte-swapped into a native array), header versions 1/2/3, and
  `fortran_order: True` (the result is F-contiguous). Any other descriptor
  (strings, structured, object) raises `DTypeError` (object: `ValueError`, as
  NumPy with `allow_pickle=False`). There is no pickle support at all.
- File arguments: a path (`string` or `URL`) reads/writes the file with Node
  `fs`; `save`/`savez*` append `.npy`/`.npz` like NumPy. Passing `null` as the
  file returns the encoded bytes as a `Buffer` instead of writing; `load`
  accepts a `Buffer`/`Uint8Array`/`ArrayBuffer` of file contents.
- `load` returns an `NDArray` for `.npy` data and an `NpzFile` for zip data
  (`files`, `get(name)` with or without the `.npy` suffix, `keys()`,
  `entries()`, iteration over names, `close()`; NumPy's `npz[key]` mapping
  syntax is `npz.get(key)`). Members are decoded lazily on `get`.
- `savez(file, ...arrays)` / `savezCompressed`: positional arrays are named
  `arr_0, arr_1, ...`; a trailing plain object (not an NDArray, nested list or
  complex-like `{re, im}`) maps names to arrays (NumPy keyword arguments).
  The zip container is written in TS like Python's `zipfile` with
  `force_zip64=True` (fixed 1980-01-01 timestamp, mode 0o600), so `savez`
  output is byte-identical to NumPy's; `savezCompressed` uses Node's
  `zlib.deflateRawSync` (level 6) whose deflate stream may differ in bytes
  from CPython's zlib but decodes to the same members. CRC-32 is native.
- Unsupported: `mmap_mode` (raises `ValueError` unless null), `allow_pickle`,
  `fix_imports`, `encoding`, `max_header_size` (headers of any size are
  parsed). Truncated or empty input raises `ValueError` (NumPy `EOFError` for
  an empty file).

## D-171 — P14 text I/O: loadtxt, savetxt, genfromtxt, fromregex, fromfile, tofile (P14) — Accepted — 2026-10-02
- Text sources: a path (`string` or file `URL`), the contents as a
  `Buffer`/`Uint8Array` (UTF-8), or an array of lines (`string[]`, like
  NumPy's list-of-lines input). A plain string is always a path. Text sinks
  (`savetxt`) take a path or `null` (returns the text as a string).
- `loadtxt` is parsed in C++ (`native/core/p14_text.cpp`) with the rules of
  NumPy's C reader: single-character (UTF-8) `delimiter` or whitespace runs,
  one or more `comments` strings, optional `quotechar` (doubled quote escapes),
  `skiprows` (raw lines), `usecols` (negative allowed, per row), `maxRows`,
  `ndmin`, `unpack`; numeric dtypes only; integers must be plain decimal,
  floats accept `inf`/`nan` but no hex or `_`, bool parses an integer. Error
  messages follow NumPy's. `converters`, `encoding`, structured and string
  dtypes are not supported. Lines end at `\n` (a trailing `\r` is dropped).
- `savetxt` formatting is native: Python `%`-formatting for `d i u o x X e E
  f F g G s` with flags `-+ 0#`, width, precision and ignored `h l L`; `%s`
  of a float prints NumPy's scalar `str` (shortest round-trip digits, the
  float16/32/64 positional/scientific cut-offs). `%r`, `%c`, `%a` and `*`
  raise `ValueError`. Booleans are accepted by `%o %x %X` (NumPy raises
  `TypeError` because `np.bool` has no `__index__`).
- `genfromtxt` is implemented in TS on top of the same splitting rules as
  NumPy's `LineSplitter` (comment split, whitespace or delimiter or fixed
  widths, autostrip) and the `StringConverter` semantics for an explicit
  numeric `dtype`: Python `int()/float()/complex()` syntax, `str2bool` for
  bool, loose mode (unconvertible → filling value) or strict mode
  (`loose: false`), `missingValues`/`fillingValues` as a value, list or
  `{column: value}` map, `skipHeader`, `skipFooter`, `usecols`, `maxRows`,
  `invalidRaise`, `ndmin`, `unpack`. `dtype: null` (type inference), `names`,
  `converters` and `usemask` are not supported. Int64 cells are parsed as
  BigInt so they keep full precision.
- `fromregex(file, regexp, dtype)`: `dtype` is a list of `[name, dtype]`
  fields (NumPy requires a structured dtype); the result is a plain object
  `{name: NDArray}` (one 1-D array per field) since structured arrays are not
  available. A JS `RegExp` (flags kept, `g` added) or a pattern string.
- `fromfile(file, {dtype, count, sep, offset})`: binary mode (`sep` empty)
  reads like `frombuffer`; text mode parses like `fromstring`. 
  `NDArray.tofile(file, {sep, format})` writes raw C-order bytes, or text
  items joined by `sep` using Python-scalar formatting (float repr, complex
  repr), `format` applied with `%`. `file` may be `null` to return a Buffer.

## D-172 — P14 baseRepr/binaryRepr and window functions (P14) — Accepted — 2026-10-02
- `baseRepr(number, base = 2, padding = 0)` and `binaryRepr(num, {width})`
  follow NumPy's pure-Python algorithms exactly (including the gh-8679
  two's-complement boundary rule and the "Insufficient bit width" error).
  They are string utilities, so they are implemented in TS on `bigint`.
  `number`/`bigint` integers are accepted, and so is a 0-d integer NDArray;
  a non-integral number raises `TypeError` (Python `operator.index`).
  NumPy's `base_repr` truncates a float with `int()`; numera raises
  `TypeError` instead.
- `bartlett blackman hamming hanning kaiser(M[, beta])` are native
  (`native/core/p14_window.cpp`) and evaluate NumPy's formulas in the same
  operation order: `n = arange(1-M, M, 2)` (or `arange(0, M)` for kaiser),
  `M` may be any real number, results are float64. `kaiser` uses NumPy's
  Chebyshev `i0` coefficients, so the results match NumPy to the last bit
  except where libm `cos`/`exp` differ by an ulp.

## D-173 — P14 legacy polynomials: poly, poly1d, polyadd ... roots (P14) — Accepted — 2026-10-02
- The `np.poly*` functions follow NumPy's `numpy/lib/_polynomial_impl.py`
  step by step, composed from existing native ops (ufuncs, `vander`,
  `linalg.lstsq`, `linalg.eigvals`, `linalg.inv`). Two loops are new native
  kernels in `native/core/p14_poly.cpp`: full 1-D convolution (`polymul`,
  `poly`; result dtype `promote_types`, integer products wrap, bool is
  or-of-ands) and long division (`polydiv`, computed in the inexact dtype
  of `u[0] + v[0]`; float16 is computed in float32 and rounded back; the
  remainder is trimmed while `|r[0]| <= 1e-8`, NumPy's `allclose(r[0], 0)`).
- `roots` builds NumPy's companion matrix and returns real values when the
  input is real and every imaginary part is 0 (NumPy's
  `_to_real_if_imag_zero`), otherwise complex.
- `polyfit(x, y, deg, {rcond, full, w, cov})` returns the coefficients, or
  NumPy's tuples as JS arrays: `full` gives `[c, residuals, rank, s, rcond]`,
  `cov` gives `[c, V]`. The rank warning is a Node `RankWarning` warning.
- `poly1d` is a class. Python operators become methods: `call(x)` (`p(x)`),
  `add sub mul div pow neg equals`; `get(power)`/`set(power, v)` are
  `p[k]`/`p[k] = v`; `length` is `len(p)` (the order); `coeffs`, `c`,
  `coef`, `coefficients`, `order`, `o`, `roots`, `r`, `variable`;
  `integ(m, k)`, `deriv(m)`; iteration over the coefficients; `toString()`
  is NumPy's `str(p)` (with the superscript line). `div` by a polynomial
  returns `[q, r]`. Every `np.poly*` function returns a `poly1d` when any
  input is one, as NumPy does.

## D-190 — P16 sub-milestones for 98% NumPy coverage — Accepted — 2026-10-02

P16 closes the gap between ~74% (after P3–P15) and 98% of the 1,060 tracked
NumPy names. Five parallel branches on the same file-ownership model as P3–P15
(D-056):

- **P16-A** (`p16a`, D-191–199): TS-only quick wins — `np.vectorize`,
  `np.shares_memory`, dtype alias constants (`np.double`, `np.int_`, `np.pi`,
  `np.e`, `np.inf`, `np.nan`, `np.True_/False_`, etc.), and the two missing
  NDArray methods (`cumsum`/`cumprod` via declaration merging).
- **P16-B** (`p16b`, D-200–209): `np.ma` masked arrays — MaskedArray class,
  masked constant, constructors, ufunc wrappers, reductions, MA methods,
  utilities. ~154 missing names. Pure TS + optional thin C++ for inner loops.
- **P16-C** (`p16c`, D-210–219): `np.strings` module (46 names) and
  `np.char` module (52 names). Both use a JS-string object-array representation
  (an NDArray whose elements are JS strings), no new C++ DType needed for the
  initial implementation. Pure TS loops are fast enough for typical workloads.
- **P16-D** (`p16d`, D-220–229): `datetime64`/`timedelta64` and busday
  functions. Stored as int64 NDArray with unit metadata in the TS layer. Core
  arithmetic in C++; DType enum gets two new opaque entries (dt64, td64) that
  are rejected by all existing ufunc loops with a clean DTypeError.
- **P16-E** (`p16e`, D-230–239): `np.rec` (9 names) — lightweight structured
  record arrays as plain JS objects with named NDArray fields; `asmatrix`/`bmat`
  exclusions updated; remaining exclusions/aliases pass.

Reserved decision ranges: A 191–199, B 200–209, C 210–219, D 220–229, E 230–239.
File ownership: each branch owns `packages/numera/src/p16[abcde]*.ts`,
`native/bindings/p16[abcde]_binding.cpp`, `native/core/p16[abcde]_*.{hpp,cpp}`,
`tests/native/test_p16[abcde]_*.cpp`, `packages/numera/test/p16[abcde]_*.test.ts`,
`docs/site/parts/p16[abcde].mjs`, `api/bench-exempt/p16[abcde].json`,
`api/aliases.d/p16[abcde].json`, `api/exclusions.d/p16[abcde].json`,
`docs/plan/slices/p16[abcde].md`. P16-B also owns `packages/numera/src/ma.ts`.
Append-only: DECISIONS.md, PROGRESS.md, COMPATIBILITY.md.

## D-191 — P16-A constants, dtype aliases, vectorize, shares_memory, cumsum/cumprod (P16-A) — Accepted — 2026-10-02
- Math constants `pi`, `e`, `inf`, `nan`, `euler_gamma`, `PINF`, `NINF`,
  `PZERO`, `NZERO` are plain JS primitive number exports.
- Bool sentinels `True_` = `true`, `False_` = `false` (JS booleans; no scalar
  class hierarchy).
- Dtype aliases re-export existing `DType` singletons from `dtype.ts`:
  `int_/intp/long` → int64, `intc` → int32, `byte` → int8, `short` → int16,
  `uint/uintp/ulong` → uint64, `uintc` → uint32, `ubyte` → uint8,
  `ushort` → uint16, `double/longdouble` → float64, `single` → float32,
  `half` → float16, `cdouble/clongdouble` → complex128, `csingle` → complex64.
  No new DType instances are created; aliases share singletons.
- `vectorize(fn, {otypes?, signature?})`: pure TS loop.  Inputs are broadcast
  to the common shape by walking multi-indices.  `otypes[0]` forces output
  dtype; otherwise inferred from the JS type of the first result
  (boolean → bool, bigint → int64, number → float64, {re,im} → complex128).
  `signature` must be `null`/omitted; a non-null value raises `ValueError`.
  The returned function has a `.pyfunc` property.
- `shares_memory(a, b, {maxWork?})`: delegates to `mayShareMemory` (native
  `sharesMemory`). `maxWork` is accepted and ignored.
- `cumsum`/`cumprod` (both `np.*` functions and `NDArray.prototype.*` methods):
  implemented via `add.accumulate` / `multiply.accumulate`. Null/omitted axis
  flattens the array first (NumPy behaviour).
## D-220 — P16-D datetime64/timedelta64 storage and arithmetic — Accepted — 2026-10-02

`datetime64` and `timedelta64` are stored as int64 NDArray with unit metadata
carried on a TS subclass (`DatetimeArray` / `TimedeltaArray`). This avoids
adding new entries to the C++ DType enum (which would require touching 38+
dispatch sites) and keeps arithmetic entirely in TypeScript.

Key design choices:
- `DatetimeArray extends NDArray`: wraps an `int64` NDArray + `unit: string`.
  The stored integers are epoch offsets in the given unit (same as NumPy's
  internal representation).
- `TimedeltaArray extends NDArray`: same pattern — int64 values + unit string.
- **NaT**: represented as `BigInt(-9223372036854775808n)` (int64 min), matching
  NumPy's sentinel. Checked with a constant `NAT_VALUE`.
- **Unit hierarchy**: 'Y' > 'M' > 'W' > 'D' > 'h' > 'm' > 's' > 'ms' > 'us' > 'ns'
  > 'ps' > 'fs' > 'as'. Conversions where numerics are exact use BigInt
  multiplication/division; Y and M remain in their abstract units (NumPy
  forbids Y/M ↔ finer unit conversion).
- **Arithmetic**: `datetime + timedelta → datetime` (same unit if compatible,
  else error), `datetime - datetime → timedelta`, `timedelta ± timedelta →
  timedelta`. Units must be compatible (no Y/M mixing with sub-day units).
- **Busday functions**: `is_busday`, `busday_count`, `busday_offset` are
  implemented in pure TS (inner loops iterate over `days-since-epoch`). C++
  helpers only needed if perf profiling shows a bottleneck; a pure TS loop
  over typical arrays (≤10k dates) runs in <10 ms on modern hardware.
- **`np.array()`** integration: P16-D adds a pre-check in `p16d.ts` —
  if the user passes `dtype: "datetime64[D]"` (or similar strings), the input
  is routed to `DatetimeArray.from()`. Automatic inference from ISO-8601
  strings is also supported.
## D-210 — P16-C StringArray: JS-string object-array representation — Accepted — 2026-10-02

`np.strings` and `np.char` operate on arrays of strings. Rather than adding a
new C++ DType (which would require touching the native dtype registry, all
ufunc loops, `toArray`, `toTypedArray`, etc.), P16-C uses a pure-TypeScript
`StringArray` class that holds strings in a flat JS `string[]` plus a `shape`
number array. This is intentionally analogous to how NDArray holds numeric data
in a native buffer.

- `StringArray` is the public return type for all string-valued `np.strings`
  operations. It is not an `NDArray` subclass; it does not participate in the
  numeric ufunc machinery.
- `StringArray` implements: `shape`, `ndim`, `size`, `dtype` (returns
  `"str_"` or `"bytes_"` string), `flat` (iterator over elements),
  `toArray()` (returns nested JS string arrays matching shape), `toString()`,
  and all string operation methods.
- Module-level functions in `np.strings` accept `StringArray | string[] |
  readonly string[]` (treated as 1-D) or nested string arrays as input.
  They return `StringArray` for string results and `NDArray` (bool or int64)
  for comparisons and numeric results.
- `np.char` is an alias namespace exposing the same functions (NumPy `np.char`
  is deprecated but widely used). `np.char.array(data)` constructs a
  `StringArray` from a nested JS string array.
- Performance: string operations are inherently O(string-length × array-size)
  and JS string primitives are already fast enough for typical workloads
  (benchmarks deferred to V phase per build-first policy).

## D-211 — P16-C np.strings surface (46 names) — Accepted — 2026-10-02

All 46 `np.strings` names are implemented in `packages/numera/src/p16c.ts`.
The `strings` sub-object is exposed as `np.strings` (not spread into `np`
itself, matching NumPy's `np.strings.add(...)` calling convention).

## D-212 — P16-C np.char surface (52 names) — Accepted — 2026-10-02

`np.char` mirrors `np.strings` for the 46 shared names and adds `array`,
`asarray`, `join`, `slice`, and `compare_chararrays`. The names `bytes_`,
`str_`, `character`, `chararray`, `ndarray`, `narray`, `asnarray`,
`array_function_dispatch`, `set_module`, and `strings_multiply/partition/
rpartition` are excluded (implementation internals or class objects not
meaningful in JS — see `api/exclusions.d/p16c.json`).

## D-213 — P16-C divergences — Accepted — 2026-10-02

- `encode` / `decode`: NumPy encodes/decodes bytes_ arrays using Python codec
  names. In JS there is no equivalent bytes_ DType; `encode` and `decode` are
  implemented as identity stubs that return the input StringArray unchanged and
  emit a runtime warning. This is recorded in COMPATIBILITY.md.
- `mod` (`%`-formatting): NumPy's `np.strings.mod` applies Python `%`-style
  formatting. numera implements it with JS template-literal-style `%s/%d/%f`
  substitution (sufficient for the common case). Differences for `%r`, `%c`,
  `%x`, etc. are documented.
- `translate`: NumPy's `str.translate(table)` takes a mapping of Unicode
  ordinals. numera implements it with a JS Map<string, string> (character →
  replacement); `null` values delete characters. This is compatible for the
  common ASCII subset but not for 32-bit code-point mappings. Documented
  divergence.
## D-230 — P16E np.rec record arrays — Accepted — 2026-10-02

`np.rec` is implemented as a lightweight pure-TypeScript module (no new C++ DType).

**Data model:** A `recarray` instance wraps a plain `Record<string, NDArray>` map of
named fields. Attribute access (`rec.x`) returns the NDArray for field `x`.
All fields must have compatible shapes (broadcast-compatible leading dims).

**API:**
- `recarray(shape, {names, formats})` — creates zero-filled columns (each column is
  `np.zeros(shape, {dtype})` where dtype is parsed from the format string).
- `record` — type alias; a 0-d record is just a length-1 recarray with `shape=[]`.
- `fromarrays(arrayList, {names, formats, titles, byteorder, aligned})` — builds a
  recarray from a list of equal-length NDArrays; `names` is required unless `formats`
  contains `name:fmt` entries.
- `fromrecords(recList, {names, formats})` — builds columns from a list of rows
  (each row is an array of scalars or nested arrays, one per field).
- `format_parser(formats, names, titles, {aligned, byteorder})` — parses a NumPy
  dtype string or an array of format strings into a list of `{name, dtype}` pairs.
  Recognized format chars: `b/B` int8/uint8, `h/H` int16/uint16, `i/I` int32/uint32,
  `l/L/q/Q` int64/uint64, `e` float16, `f` float32, `d` float64, `F` complex64,
  `D` complex128. Prefix N for fixed-size arrays (e.g. `3f` → float32[3]).
- `fromfile(file, dtype, shape)` — not implemented in this milestone (file I/O);
  raises `NotImplementedError`. Listed as excluded in `api/exclusions.d/p16e.json`.
- `fromstring(string, dtype, shape)` — not implemented in this milestone; raises
  `NotImplementedError`. Listed as excluded.
- `array(obj, {names, formats, ...})` — flexible constructor: delegates to
  `fromarrays` when `obj` is an array of NDArrays, to `fromrecords` when it is
  an array of rows, or clones an existing `recarray`.
- `find_duplicate(list)` — returns values that appear more than once in the input
  list (pure TS, mirrors `np.rec.find_duplicate`).

**np surface:**
- `recarray` is exported from `p16e` and added to the `np` object so that the
  `np.recarray` missing entry disappears.
- `shares_memory(a, b, {maxWork})` — pure-TS check: compares the underlying
  `ArrayBuffer` references of two NDArrays (same buffer → true). No C++.
  This covers the `np.shares_memory` missing entry (P16-A has not added it yet
  at time of writing; if P16-A merges first and already exports it, our entry
  is a no-op duplicate and the later-spreading milestone wins harmlessly).

**Exclusions (api/exclusions.d/p16e.json):**
- `rec.fromfile` / `rec.fromstring` — deferred (binary file/buffer I/O for rec)
- `np.asmatrix` / `np.bmat` / `np.matrix` — matrix class is excluded (D-032 c)
- `np.bytes_` / `np.str_` — dtype constants; covered by `np.char` module exclusions
- dtype aliases (`byte`, `short`, `int_`, `intc`, `intp`, `long`, `ubyte`, `ushort`,
  `uint`, `uintc`, `uintp`, `ulong`, `half`, `single`, `double`, `cdouble`, `csingle`) —
  these are C-type aliases that map to existing numera dtypes; excluded (D-032 b).
- math constants (`pi`, `e`, `nan`, `inf`, `euler_gamma`) — these are numeric constants
  that P16-A owns; recorded as aliases to JS globals here so coverage counts them.
- `np.False_` / `np.True_` — P16-A constants; added as aliases to `false`/`true`.
- `np.vectorize` — P16-A class; added as exclusion pending P16-A merge.
- All `np` functions owned by other P16 branches (busday*, datetime*, nan*,
  histogram*, ptp, quantile, percentile, etc.) remain in missing for now.

**Compatibility:** `recarray` instances do not support NumPy's `view()` casting
or structured-dtype memory layout. Recorded in COMPATIBILITY.md.
## D-200 — np.ma MaskedArray class design — Accepted — 2026-10-02
Pure TypeScript implementation; no new C++ needed. A `MaskedArray` wraps two
NDArrays: `_data` (any dtype) and `_mask` (bool NDArray or the `nomask`
sentinel). The `nomask` sentinel is `false` (a scalar bool, matching NumPy's
`np.ma.nomask is False`). The `masked` singleton is a 0-d MaskedArray with
data=0 and mask=true (it acts as a "masked scalar" sentinel). All operations
that produce a scalar result return a MaskedArray (or the `masked` singleton)
rather than a plain number, to preserve masked semantics.

## D-201 — np.ma nomask and masked constants — Accepted — 2026-10-02
`nomask` is exported as `false as const` (matching `np.ma.nomask is False`).
`masked` is a singleton `MaskedArray` with shape `()`, `_data = array([0])`,
`_mask = array([true])`. `masked_singleton` is an alias. `masked_print_option`
is a configurable string defaulting to "--"; it is mutable via `.set(s)`.

## D-202 — MaskError, MAError, MaskedIterator — Accepted — 2026-10-02
`MaskError` extends `NativpyError` (from errors.ts); exported as `np.ma.MaskError`.
`MAError` is an alias for `MaskError`. `MaskedIterator` iterates over the elements of
a `MaskedArray`, yielding either element values or the `masked` singleton for masked
positions; it implements the JS iterator protocol.

## D-203 — np.ma constructors — Accepted — 2026-10-02
`masked_array(data, mask?, fill_value?, dtype?, copy?)`: primary constructor.
`array` is an alias for `masked_array`. `asarray(data)` returns a MaskedArray view
(copy=false). All `masked_*` predicates call `masked_where` internally.
`masked_invalid` masks NaN and Inf (uses isNaN/isFinite). `masked_values` uses
`np.isclose` semantics. `fix_invalid` replaces invalid (NaN/Inf) with fill_value.

## D-204 — np.ma mask utilities — Accepted — 2026-10-02
`make_mask(m, copy?, shrink?, dtype?)`: converts input to a bool mask NDArray.
`make_mask_none(shape)`: returns a false-filled bool NDArray.
`make_mask_descr(ndtype)`: for structured dtypes, returns corresponding bool dtype
(for simple dtypes, returns bool). `getmask(a)`: returns the mask or `nomask`.
`getmaskarray(a)`: always returns a bool NDArray (expands nomask to all-false).
`getdata(a)`: returns the underlying NDArray. `filled(a, fill_value?)`: returns a
plain NDArray with masked elements replaced by fill_value. `is_masked(a)`: true if
any element is masked. `is_mask(m)`: true if m is a valid mask (bool NDArray or false).
`mask_or(m1, m2)`: element-wise OR of two masks. `flatten_mask(m)`: flattens a
structured mask to bool. `shrink_mask(a)`: replaces a uniform-false mask with nomask.

## D-205 — np.ma arithmetic propagates mask — Accepted — 2026-10-02
Binary ops: if either operand is masked at position i, output is masked there.
Unary ops: preserve the mask unchanged. All NumPy ufuncs on np.ma inputs delegate
to the plain ufunc on `.data` and combine masks. Division by masked zero is masked
(not an error). Power: 0**negative is masked. The `fill_value` for arithmetic
results is derived from `default_fill_value` of the result dtype.

## D-206 — np.ma reductions skip masked — Accepted — 2026-10-02
Reductions use `filled(a, fill_identity)` before delegating to the plain NDArray
reduction, where `fill_identity` is: 0 for sum/cumsum, 1 for prod/cumprod, +Inf for
min, -Inf for max, false for all (fill with true, treat unmasked only), true for any
(fill with false). Result mask: scalar result is masked only when ALL elements are
masked. When axis= is given, an output position is masked when all contributing
inputs were masked.

## D-207 — np.ma shape/manipulation methods — Accepted — 2026-10-02
MaskedArray exposes the same shape-manipulation surface as NDArray: reshape,
ravel, flatten, transpose, T, squeeze, expand_dims, repeat, take, put, sort,
argsort, swapaxes. Each delegates to the plain NDArray operation on `_data` and
applies the same operation to `_mask` (or keeps `nomask` if mask is `nomask`).

## D-208 — np.ma concatenate/stack — Accepted — 2026-10-02
concatenate/vstack/hstack/dstack/stack gather both `.data` and `.mask` from all
inputs, run the plain NDArray operation on each, and return a MaskedArray.
If all input masks are `nomask`, the result mask is `nomask`.

## D-209 — np.ma utility functions — Accepted — 2026-10-02
Functions that do not neatly fit other slices (clip, where, anom, average, etc.)
delegate to their plain-np counterparts on `.filled()` data and combine masks.
`default_fill_value`: returns the NumPy default fill value for the dtype
(bool=true, int=999999, float=1e20, complex=1e20+0j, object='N/A').
`maximum_fill_value`/`minimum_fill_value`: return dtype max/min.
`common_fill_value(a,b)`: returns shared fill value or masked.
`ids(a)`: returns [id(data), id(mask)] as a tuple (JS: [data buffer id, mask buffer id]).

## D-130 — Statistics API layout (P10) — Accepted — 2026-10-02
- Native kernels in `native/core/p10_*.{hpp,cpp}` (binding `addon.p10`); TS in
  `p10.ts` / `p10_*.ts`, plus `reduce.ts` (owned by P10). Build-first: NumPy
  differential cases and benchmarks come in the V phase.
- Keyword arguments are options objects (`np.quantile(a, q, {axis, method,
  keepdims, weights})`); results are always `NDArray`s (0-d instead of NumPy
  scalars, as D-017). Tuple results are JS arrays: `histogram` → `[hist, edges]`,
  `histogram2d` → `[H, xedges, yedges]`, `histogramdd` → `[H, edges[]]`,
  `average(..., {returned: true})` → `[avg, sumOfWeights]`, multi-axis
  `gradient` → `NDArray[]`.
- NDArray methods `cumsum cumprod ptp` are added by declaration merging in `p10.ts`.
- Functions with NumPy's `out=` that P10 supports: `sum prod min max mean var
  std` (np.* functions), `cumsum cumprod cumulativeSum cumulativeProd nancumsum
  nancumprod`. `out` must have the exact result shape; values are cast unsafely
  (NumPy reductions). Other `out=` / `overwrite_input=` keywords are not
  accepted (listed in COMPATIBILITY.md).

## D-131 — Quantiles (P10) — Accepted — 2026-10-02
- `quantile/percentile/median` and their nan-variants follow NumPy 2.5
  `_quantile`/`_median` exactly: the 13 `method=` names, NumPy's virtual
  index formulas, `_get_indexes` clamping, `fix_gamma`, and `_lerp`
  (`a + d*t`, replaced by `b - d*(1-t)` where `t >= 0.5`). The difference
  `b - a` is taken in the input dtype, so integer inputs wrap as in NumPy.
- Selection uses `std::nth_element` on a per-slice copy (no global sort).
- Result dtype: discrete methods (`inverted_cdf closest_observation lower
  higher nearest`, and `linear` with an integer `q` array) keep the input
  dtype; the others use `result_type(a, q)` where a JS number `q` is weak
  (NEP 50; float32 input stays float32) and a list/NDArray `q` is strong.
  Median uses the `mean` dtype rules (int → float64).
- A slice containing NaN gives NaN; empty input raises `IndexError` (NumPy),
  except `median`, which gives NaN like `mean`. Complex input raises `DTypeError`
  (NumPy `TypeError`); complex `median` raises `NotImplementedError` (numera).
- `weights=` only with `method: "inverted_cdf"` (NumPy rule), same shape as
  `a` or 1-d/nd matching the reduced axes; negative weights → `ValueError`.

## D-132 — Cumulative ops, diff, ptp (P10) — Accepted — 2026-10-02
- `cumsum/cumprod` (axis default: flattened) and `cumulativeSum/
  cumulativeProd` (array API: axis required for ndim > 1, `includeInitial`)
  run `add/multiply.accumulate` natively (D-052 dtype rules = NumPy sum/prod).
- `nancumsum/nancumprod` replace NaN by 0/1 first. `diff` uses `subtract`
  (bool: `!=`), `prepend`/`append` are converted with `np.array` (strong).
- `ptp = max - min` in the input dtype (integers wrap, bool raises `DTypeError`).

## D-133 — NaN reductions (P10) — Accepted — 2026-10-02
- Follow NumPy `_nanfunctions_impl`: NaN replaced by 0 (sum, mean, var), 1
  (prod), ±inf (min/max/argmin/argmax) and the count of non-NaN values per
  slice is used as divisor. All-NaN slices: `nanmin/nanmax/nanmean/nanvar/
  nanstd/nanmedian/nanquantile` → NaN (NumPy also warns; numera does not),
  `nanargmin/nanargmax` → `ValueError("All-NaN slice encountered")`.
  Non-float inputs call the plain reduction.

## D-134 — Histograms, bincount, digitize, interp (P10) — Accepted — 2026-10-02
- `histogram` follows NumPy's two algorithms: uniform bins (int/string `bins`)
  use the computed-index-plus-correction method with `bincount`-style float64
  weight accumulation per 65536 block; explicit edges use sorted cumulative
  sums (`searchsorted` inclusive on the last edge). The bin estimators are
  NumPy's formulas computed in float64. Complex weights raise
  `NotImplementedError`.
- `histogramdd` sample: an `NDArray` is `(N, D)` (1-d → `(N, 1)`); a JS array is
  a list of D coordinate arrays (NumPy's array_like interpretation).
- `interp` ports NumPy's `arr_interp` (binary search, slope, the NaN fallbacks);
  complex `fp` handled per component.

## D-135 — correlate / convolve / gradient / cov (P10) — Accepted — 2026-10-02
- `correlate(a, v, mode="valid")` computes `c_k = Σ a[n+k]·conj(v[n])` with
  NumPy's mode windows (`same`: left pad `min/2`); `convolve(a, v,
  mode="full")` = correlate with reversed `v` and no conjugation. Sums run
  sequentially (NumPy may use BLAS dot, so the last bits can differ).
- `gradient` ports NumPy (interior central differences, edge_order 1/2,
  scalar or coordinate spacing; uniform coordinates reduce to a scalar that is
  strong, i.e. computed in float64).
- `cov/corrcoef/average/trapezoid` are composed from native ufuncs,
  reductions and `matmul` in TS following NumPy's code.

## D-136 — Reduction where= / out= (P10) — Accepted — 2026-10-02
- `np.sum/prod/min/max/mean/var/std` accept `where` (bool mask broadcast to
  `a`) and `out`. Masked-out elements are replaced by the identity (sum 0,
  prod 1) or by `initial` (min/max, which require `initial` with `where`, as
  NumPy); mean/var/std divide by the masked count. Implemented natively
  (`p10_reduce`). The NDArray methods (`a.sum()`) live in `ndarray.ts` (P3)
  and do not get `where`/`out` on this branch.

## D-137 — average / cov / gradient errors and placement (P10) — Accepted — 2026-10-02
- `average/cov/corrcoef/gradient/trapezoid` are native (`p10_stats`), not
  composed in TS (refines D-135). `cov` uses a sequential conjugated dot
  product instead of BLAS `dot`, so the last bits may differ.
- Python exception mapping: `ZeroDivisionError` (weights sum to zero, `average`
  with an empty slice) → `ValueError`; `TypeError` → `DTypeError`;
  `RuntimeError` (bad `fweights`/`aweights` shape) → `ValueError`. The cov
  "Degrees of freedom <= 0" warning is a Node `RuntimeWarning`.
- `gradient(f, ...spacings, {axis, edgeOrder})`: a trailing plain object is the
  options; a JS-number spacing is a weak scalar (keeps float32), a 0-d array or
  uniform coordinates are strong (NumPy semantics).

## D-180 — np.emath (P15-1) — Accepted — 2026-10-02
- `np.emath` is a namespace object on `np` (from `p15.ts`), with the 9 NumPy
  names `sqrt log log2 log10 logn power arccos arcsin arctanh`. Only the
  namespace is a named export (its members would clash with top-level names).
- Whole-array promotion, as `numpy.lib._scimath_impl`: if any real element is
  out of domain (`x < 0` for sqrt/log*, `|x| > 1` for the inverse trig), the
  whole array is converted to complex (`complex64` for int8/uint8/int16/uint16/
  float32, else `complex128`, i.e. float16 → complex128 like `_tocomplex`).
  In-domain real input uses the ufunc's float loop (bool/int8/uint8 → float16,
  int16/uint16 → float32). Native kernel `native/core/p15_emath.cpp`
  (P4's `log2`/`arcsin`/... ufuncs are not on this branch): complex sqrt/log
  are NumPy's ports from `complex_kernels.hpp`, complex log2/log10 are
  `clog(z) * log2(e)` / `* log10(e)` (NumPy's npy_clog2 formula), complex
  arccos/arcsin/arctanh use the C99 libm `cacos/casin/catanh` (as NumPy).
  FP errors follow `np.seterr` (D-054) under the emath function's name.
- `logn(n, x)` = `log(x) / log(n)` and `power(x, p)` = `np.power` after the
  same fixups; NumPy's `_fix_int_lt_zero` (`p * 1.0`) gives float64 for integer
  `p`. Results are always arrays (0-d for scalars), no out/where options.

## D-181 — np.testing (P15-2) — Accepted — 2026-10-02
- `np.testing` is a namespace object; `AssertionError` (a `NativpyError`
  subclass, code `NATIVPY_ASSERTION`) is also a named export. Names are
  camelCase (`assert_array_equal` → `assertArrayEqual`); `assert_` keeps its
  trailing underscore. Python keyword arguments become an options object
  (`errMsg`, `verbose`, `strict`, `rtol`, `atol`, `equalNan`, `decimal`,
  `significant`, `maxulp`, `dtype`); `assertArrayAlmostEqualNulp(x, y, nulp)`
  keeps a positional `nulp`.
- Messages follow NumPy 2.x (`build_err_msg`, `assert_array_compare`):
  header, mismatch count/percent, up to 5 mismatching indices, max
  absolute/relative difference, and `array_repr`s. The reprs come from a private
  formatter (`p15_format.ts`) modelled on NumPy's arrayprint (floatmode
  "maxprec", 75 columns, summarization above 1000 elements). JS numbers that
  are integers print as Python ints (`1`, not `1.0`) because JS has no separate
  float scalar type.
- `assertRaises(ErrorClass, fn, ...args)` / `assertRaisesRegex` take a JS
  callback and return the caught error (no context-manager form).
  `assertWarns(type | null, fn, ...args)` / `assertNoWarnings(fn, ...args)` watch
  `process.emitWarning` while `fn` runs synchronously (numera's `np.seterr` "warn"
  path). `assertStringEqual` produces `-`/`+` line diffs without difflib's `?`
  hint lines.
- Excluded from coverage (`api/exclusions.d/p15.json`): Python/NumPy build flags,
  unittest/nose classes, warnings-module context managers, gc/proc/exec/thread
  helpers, `test`.

## D-182 — np.polynomial (P15-3) — Accepted — 2026-10-02
- `np.polynomial` exposes the six classes (`Polynomial`, `Chebyshev`,
  `Legendre`, `Laguerre`, `Hermite`, `HermiteE`), `setDefaultPrintstyle`, and
  the per-basis modules `np.polynomial.{polynomial, chebyshev, legendre,
  laguerre, hermite, hermite_e}`. Module functions keep NumPy's prefixed
  names (`chebadd`, `lagval`, `herme2poly`, `chebdomain`, ...).
- Series arithmetic (add/sub/mul/mulx/div/pow/val/der/int/vander/companion/
  fromroots, conversions to/from the power basis) is a native kernel
  (`native/core/p15_polynomial.cpp`) over 1-D float64/complex128
  coefficients, following NumPy's algorithms and operation order. Coefficients
  are always float64 or complex128 (NumPy also keeps object arrays). Only 1-D
  coefficient arrays are supported: there is no `axis` argument for der/int and
  no `tensor` argument for val. The multidimensional `*val2d/3d`, `*grid2d/3d`,
  `*vander2d/3d` and Gauss quadrature/weight helpers are not provided yet.
- Fitting uses `np.linalg.lstsq` with NumPy's column scaling. Roots are the
  `np.linalg.eigvals` of the companion matrix, sorted, and made real when the
  input is real and every imaginary part is 0 (NumPy's power, Laguerre and
  Hermite modules). A rank-deficient fit emits `process.emitWarning(...,
  "RankWarning")`.
- Python operators map to methods: `add sub rsub mul truediv floordiv mod
  divmod pow neg pos equals`. `p.call(x)` evaluates; `p.call(q)` with a
  series composes. `toString()` is `str()` (unicode, or ascii after
  `setDefaultPrintstyle("ascii")`) and `repr()` is `repr()`. `fit(x, y, deg,
  { domain, window, rcond, w, symbol })` returns the series, and `fitFull`
  also returns the diagnostics. `fromroots`/`basis` take an options object,
  `integ(m, k, lbnd)` and `deriv(m)` are positional.
- Class default domain/window are float64 static `defaultDomain` /
  `defaultWindow` (`Cls.domain` getters return copies). Division by a zero
  series throws `ValueError` (NumPy: ZeroDivisionError).
- `polynomial.test` is excluded from coverage.

## D-160 — P13 bit generators, SeedSequence and state (P13-1) — Accepted — 2026-10-02
- `native/random/p13_bitgen.{hpp,cpp}` (namespace `nativpy::random::p13`) has its
  own `SeedSeq` (entropy, spawn key, pool size) and `MT19937`, `PCG64`,
  `PCG64DXSM`, `Philox` and `SFC64`, ported from NumPy 2.x. Each one derives from
  `StatefulBitGen : random::BitGen`, so the D-019 helpers (`bounded_integers`,
  `random_doubles`, `shuffle`, the ziggurat normal, ...) work on all of them.
  State is read and written as uint64 word vectors in a fixed per-kind layout.
  The D-019 `addon.random.BitGenerator` stays unchanged. `random.ts` moves to
  `addon.p13.BitGen`, and the old streams stay identical; the existing tests
  verify this.
- TS API (camelCase of NumPy):
  - `new np.random.SeedSequence(entropy?, {spawnKey, poolSize})` with
    `entropy`, `spawnKey`, `poolSize`, `nChildrenSpawned`, `generateState(n,
    dtype)` and `spawn(n)`.
  - `new np.random.PCG64(seed?)`, and likewise `PCG64DXSM`, `SFC64`, `MT19937`,
    `Philox(seed?, {counter, key})`. `seed` may be a SeedSequence. Each one has a
    `state` getter/setter that takes NumPy-shaped plain objects (uint64 values as
    bigint, MT19937 `key` as a uint32 `NDArray`), plus `seedSeq`, `spawn(n)` and
    `randomRaw(size?)`.
  - `np.random.BitGenerator` is the abstract base. It cannot be constructed, as
    in NumPy.
  - `new np.random.Generator(bitGenerator)`, `rng.bitGenerator`, `rng.spawn(n)`,
    `rng.bytes(n)` (returns a `Uint8Array`, NumPy `bytes`).
- Errors: Python `TypeError`/`OverflowError` have no numera kind. They map to
  JS `TypeError`/`RangeError`, as in D-019's existing range checks; the message
  text matches NumPy.

## D-161 — Generator distributions and broadcasting (P13-2, P13-3) — Accepted — 2026-10-02
- Scalar kernels are ported line by line from NumPy's `distributions.c`,
  `random_hypergeometric.c` and `logfactorial.c` into
  `native/random/p13_distributions.cpp`, so the bit streams match.
  `(int64_t)` casts of out-of-range doubles saturate and map NaN to 0, which
  is what arm64 does (on x86 such casts are UB/INT64_MIN). Only parameters
  NumPy rejects reach those casts.
- A generic native engine (`p13_binding.cpp`) mirrors NumPy's `cont` / `disc`
  / `cont_f`:
  - parameters are converted to float64 (int64 for binomial `n` and the
    hypergeometric arguments, with the `safe`-cast TypeError for float arrays
    and truncation for float scalars);
  - with any non-0-d parameter, the array constraint checks run and the output
    shape is `size` or the broadcast shape, filled in C order;
  - NumPy's exact shape-mismatch and "Output size ... is not compatible" errors;
  - otherwise the scalar constraint checks run, and the call returns a JS
    number when `size` is omitted.
- `out=` is supported where NumPy has it (`random`, `standardNormal`,
  `standardExponential`, `standardGamma`), with NumPy's dtype/contiguity/size
  checks.

## D-162 — Multivariate samplers, choice(p), permuted (P13-4) — Accepted — 2026-10-02
- `multinomial` (broadcast `n` × `pvals[..., :]`), `dirichlet` (beta
  stick-breaking when max(alpha) < 0.1), and `multivariateHypergeometric`
  (`count` / `marginals`) run natively with NumPy's algorithms and errors.
- `multivariateNormal` computes the factorization with `np.linalg`
  (svd/eigh/cholesky; LAPACK-backed, D-018), then
  `mean + z @ factor.T` (legacy: `z @ (sqrt(s)[:, None] * v) + mean`). Results
  are bit-exact only to the extent that the decompositions match NumPy's
  LAPACK. Tests compare with a 1e-12 tolerance and the code makes no
  bit-exactness claim. `checkValid: "warn"` emits a Node `RuntimeWarning`.
- `choice(p=)` follows NumPy: kahan-sum validation, cdf + `searchsorted(right)`,
  and the no-replace loop with `unique(returnIndex)`.
- `permuted(x, {axis, out})`: copy, then a Fisher–Yates pass per lane (or over
  the flattened array for `axis: null`), as NumPy does.

## D-163 — RandomState legacy distributions and module functions (P13-5, P13-6) — Accepted — 2026-10-02
- The `legacy::` kernels port `legacy-distributions.c`: polar gauss with the
  cached value, the legacy gamma/beta/exponential, and the legacy
  binomial/zipf/geometric/logseries/vonmises/rayleigh. They reuse the
  Generator kernels where NumPy does (laplace/gumbel/logistic/triangular/
  uniform/poisson).
- `RandomState.getState({legacy = true})` returns
  `["MT19937", key: NDArray<uint32>, pos, hasGauss, gauss]`. `legacy: false`
  returns the dict shape. `setState` accepts both and uses NumPy's error
  messages. `RandomState(seed)` also accepts a BitGenerator (a non-MT19937
  generator cannot be reseeded).
- `RandomState()` without a seed fills the key from SeedSequence(OS entropy)
  with `key[0] = 0x80000000`, as NumPy does. D-019 used array seeding instead;
  neither is reproducible.
- `randomIntegers(low, high)` is `randint(low, high + 1)` and emits NumPy's
  DeprecationWarning through `process.emitWarning`. `ranf` and `sample` are
  aliases of `randomSample`, and `bytes` uses NumPy's uint32 draw.
- Every distribution is also exposed as an `np.random.*` function on the
  global RandomState, together with `getState`, `setState` and `bytes`.

## D-231 — Platform-independent ufunc differential cases — Accepted — 2026-10-02
- A float → integer cast that is out of range (or NaN/inf) is undefined in C, so
  NumPy's results for it differ by platform. x86 `cvtt*` gives INT_MIN or wraps,
  arm64 saturates, and NumPy's SIMD and scalar loops can disagree too. The
  `ufunc_dtype_casting` and `ufunc_where` generators now list those result
  positions as `cast_ub` (`python/generators/cast_ub.py`). The tests skip only
  those positions. All other elements are still compared exactly, and numera
  keeps the fixed D-009 rule for them.
- Approximate cases record the `loop` dtype. The tolerance follows the loop's
  precision, not the dtype of `out` (for example a float32 loop writing into a
  float64 `out`).
- `expected_noblas` (D-037) is now computed by writing out NumPy's non-BLAS
  complex loop in the generator. Strided views no longer force NumPy's own loop
  (OpenBLAS accepts any stride), so the old approach recorded per-CPU BLAS
  results. It is recorded for every exact 1-D/2-D product (not only
  non-finite inputs) whenever it differs from the local BLAS result, because
  OpenBLAS kernels vary with the CPU (AVX2/AVX-512) of the machine that
  generates the cases.
- `ufunc_out` approximate cases also record `loop`. For example, `exp(int16)`
  runs in float32 even with a float64 `out`. Each tolerance is the coarser of
  the loop's and `out`'s.
- Verified: case files generated with NumPy 2.5.3 on linux-x86_64 and on
  linux-aarch64 both pass against the GCC 13 Linux build. macOS cases pass
  against the macOS build.

## D-240 — Public repository links; API reference hosted at numera.cyfora.in — Accepted — 2026-10-02
- **Context.** The GitHub repository `Rajankr542/numera` is now public. The
  "self-contained package" rule of D-029 and the unpkg docs hosting of D-030
  existed only because it was private.
- **Package metadata.** `packages/numera/package.json` now commits
  `homepage: https://numera.cyfora.in`,
  `repository: git+https://github.com/Rajankr542/numera.git` (directory
  `packages/numera`) and `bugs: …/issues`, so npm shows the repo, issues and
  docs links. `scripts/set-homepage.mjs` (publish-time unpkg homepage) is
  removed. `checkTarball` in `scripts/release.mjs` requires these exact values.
- **Docs hosting.** The API reference is no longer shipped in the tarball
  (`files` drops `docs`; `checkTarball` rejects `docs/`). `pnpm docs` writes
  `docs-dist/index.html` (gitignored), which the maintainer deploys to
  `numera.cyfora.in` (external host, not GitHub Pages). The page links to
  GitHub and npm and has a canonical URL; it still loads nothing external.
  Docs are not versioned: the site shows the latest release.
- **Package README.** Relative links (`./COMPATIBILITY.md`, `./ROADMAP.md`,
  `./PERFORMANCE.md`, `#development`) are rewritten to absolute GitHub URLs, and
  a "Links" section (docs, source, issues) is appended. ROADMAP/Performance and
  the source-build hint are kept. `assertSelfContained` now only rejects
  internal decision references (D-NNN, PLAN, DECISIONS), dev commands and
  relative Markdown links, and only README/COMPATIBILITY are checked; `dist/`
  comment stripping stays best-effort.
- **GitHub Release.** With `--push`, `pnpm release` creates a GitHub Release for
  `vX.Y.Z` via `gh release create` (npm link, install line, docs link plus
  generated notes). If `gh` is missing or fails, it prints the manual URL;
  npm publishing is not affected. Supersedes the "no GitHub Release" part of D-031.
- **Supersedes** the private-repo parts of D-029, the hosting part of D-030 and
  the `homepage`/`repository`/`bugs` checks of D-031.

## D-241 — README layout, CONTRIBUTING.md and a curated npm README — Accepted — 2026-10-02
- **Context.** The root README mixed user docs with maintainer release notes,
  and the npm README was that file cut at "## Development", so it was long and
  repo-centric.
- **Root README** follows a short landing-page layout: badges, the Cyfora
  logo (`docs/images/cyfora-logo-{light,dark}.png`, `<picture>` for dark
  mode), a feature list, nav links, install, quick start, a compact feature
  tour, and links to Contributing/License. Badge numbers come only from
  verified sources (`api/coverage.json`); no performance claims
  (PERFORMANCE.md stays the only source).
- **CONTRIBUTING.md** (new) holds the development setup, test commands, PR
  and commit conventions, the differential-test workflow and the maintainer
  release procedure previously in the README.
- **npm README** is generated by `packageReadme()` from the root README:
  the text up to `## Contributing` (was `## Development`), with relative image
  and doc links rewritten to absolute GitHub URLs (`raw.githubusercontent.com`
  for images, since npm cannot render repo-relative images). Supersedes the
  README-cut rule of D-240; the Links section and `assertSelfContained` are kept.
