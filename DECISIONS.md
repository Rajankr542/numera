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
