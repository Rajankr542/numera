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
- Complex *input* to linalg raises `NotImplementedError` (D-008).

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
  - An omitted seed means OS entropy (128 bits from `crypto.getRandomValues`).
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

