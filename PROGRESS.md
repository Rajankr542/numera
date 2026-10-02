# PROGRESS

## 2026-10-02 — P2-8: ufunc.reduce / ufunc.accumulate, build-first (D-052)
- Native: new `native/core/ufunc_methods.{hpp,cpp}` with `ufunc_reduce` / `ufunc_accumulate`. Both run only through the D-051 registry record (loops, identity, resolvers, power check), with no per-op code apart from the add/multiply sum/prod dtype rule.
- Binding: `ufuncMethod(method, name, a, out | null, opts)`.
- TS: the 7 binary ufuncs now carry `.reduce(a, {axis, dtype, out, keepdims, initial, where})` and `.accumulate(a, {axis, dtype, out})`. New exported types: `UfuncReduceOptions`, `UfuncAccumulateOptions`.
- Tests: new `tests/native/test_ufunc_methods.cpp` (9 cases) and a vitest block (4 cases).
- Verified: `pnpm build`, `test` (271), `test:native`, `test:asan`, `typecheck` and `api:check` pass.
- Not yet done (P2-8v): NumPy differential group, pairwise float add / float16 accumulator bit-exactness, NumPy keep-order result strides, out-dtype buffer round-trip, benchmarks. No NumPy-compatibility or performance claims are made for this slice.
- Environment: I recreated `.venv` and removed the stale `build*/` caches, which pointed at the old `nativpy` path. Native builds need `SDKROOT=$(xcrun --sdk macosx --show-sdk-path)` on this machine because the CommandLineTools SDK 27 fails to link.

## 2026-10-01 — P2-7: table-driven native ufunc registry (D-051)
- Native: new `native/core/ufunc_registry.{hpp,cpp}`. Each of the 14 ufuncs is now one `constexpr` `Ufunc` record with these fields:
  - name, `nin` and `identity` (add 0, multiply 1).
  - The default type resolver and the `dtype=` resolver.
  - An optional input check (power's negative exponent).
  - Per-dtype strided loop tables, built at compile time.
- The generic `binary(const Ufunc&, …)`/`unary(const Ufunc&, …)` drivers handle out/where/dtype/casting/order (D-046…D-050). They contain no op-specific `switch`.
- `ufunc.cpp` keeps the enum API (`binary`/`unary`, `*_result_dtype`, `*_op_from_name`) as thin aliases over the registry.
- The binding finds ufuncs by name with `find_ufunc`.
- Pure refactor: no existing test was changed. Error messages, dtypes, layouts and values are unchanged.
- Tests: 2 new native test cases cover lookup, identity, loop availability, and record-based vs enum calls.
- Verified: `pnpm build`, `test` (267), `test:native`, `test:asan`, `test:diff` (10420 passed), `typecheck` and `api:check` all pass.
- Benchmarks: I ran `bench:suite` once on the old code and once on the new code, on this machine. Elementwise timings are within run-to-run noise with no consistent direction. Most differ by under 4%; the largest gap is about 13%, on a roughly 25 µs case (multiply scalar at 100k), and there the new code was faster. No speedup is claimed.

## 2026-10-01 — P2-6: ufunc `order=` and NumPy result layout (D-050)
- Native: `Order {C,F,A,K}`, `UfuncParams::order` (default K), and `ufunc_result_strides` (ports of NumPy's trivial-loop check, `npyiter_find_best_axis_ordering` and the 'A' rule).
  - `NDArray::empty_strided` allocates results and temporaries with that layout. The `where=` path keeps the layout as well.
  - Strides come from the inputs before any cast. This includes NumPy's contiguous copy of cast 0-d and short (<= 8192) 1-d inputs.
  - `dot`/`inner` with a 0-d operand now match NumPy's layout as well: inputs are cast with keep-order strides, and BLAS dtypes with ndim <= 2 give C order.
- Binding: `order` is parsed in `ops_binding` (case-insensitive). Bad names raise `ValueError` with NumPy's message, and non-strings raise `DTypeError`.
- TS: `UfuncOptions.order?: UfuncOrder | null`, with the `UfuncOrder` type exported.
- Supersedes the D-014 "always C-contiguous" rule. The `sub_transposed` special case in the differential runner is gone; it now compares exactly. `conjugate`/`angle` match NumPy too.
- Tests:
  - New differential group `ufunc_order` with 951 cases. It compares strides and flags as well as values.
  - One-off runs, not committed: 6,492 extra order cases (7 seeds) and 1,473 dot/inner cases, all passing.
  - 1 native test case and 6 numera tests.
- Verified on Node 22: `pnpm build`, `test` (267), `test:native`, `test:asan`, `test:diff` (10420 passed), `typecheck` and `api:check` all pass. `test` also passes on Node 18.20.4 and 20.17.0.
- Limitation (documented in D-050): when several errors happen in one call, a bad `order`/`casting` name is reported before a read-only `out` or a bad `where` dtype. NumPy checks those first. No performance claims are made; the layout computation is O(nd² · nops) per call.

## 2026-10-01 — P2-5: ufunc `where=` mask (D-049)
- Native: `UfuncParams::where` holds an optional bool mask, used by both binary and unary ufuncs.
  - The mask broadcasts like an extra input: it can expand the result, and it is checked against `out`.
  - When a mask is given, the loop writes to a temporary. `masked_copy_into` then copies the true positions into `out`, or into a zeroed result.
  - Integer `power` checks for negative exponents only at masked-in positions.
  - Error order matches NumPy 2.5.3: read-only `out` → mask dtype → loop → casts → shapes → values.
- TS: `UfuncOptions.where` accepts an NDArray (must be bool), a nested list, a boolean or a number.
  - Lists and scalars are converted to bool, like NumPy.
  - `null` is rejected.
- Tests:
  - 813 new NumPy differential cases (`ufunc_where`), 55 of them errors.
  - 3 native test cases and 5 numera tests.
- Verified: `pnpm build`, `test`, `test:native`, `test:asan`, `test:diff` (9469 passed) and `api:check` all pass.
  - `tsc` adds no new errors. The 3 type errors in `complex.test.ts` predate this slice.
- Limitation: the masked path computes every element and then makes an extra pass to apply the mask. It has not been benchmarked, so no performance claims are made.


## 2026-10-01 — P2-4: ufunc `dtype=` and `casting=` (D-048)

- Native: `struct UfuncParams { optional<DType> dtype; Casting casting; }`
  and overloads `binary(op, a, b, const NDArray* out, params)` /
  `unary(op, a, out, params)` in `ufunc.hpp`. The old overloads forward to
  them with the defaults, so they behave as before.
  - With `dtype=D`, the D → D loop runs. The one exception is `abs` of
    complex input with D float32/float64, which uses the complex → real loop
    when the input casts safely to it (NumPy `F->f` / `D->d`).
  - Missing loops raise `DTypeError` "No loop matching the specified
    signature and casting was found for ufunc X". The list of missing loops
    comes from an exhaustive NumPy probe over 14 dtypes × 12 ufuncs.
  - Input casts are checked with `can_cast(in, loop_in, casting)` and
    produce NumPy's messages ("input 0", "input 1", or just "input" for
    unary ufuncs). The `out` cast now uses the given casting rule instead of
    a fixed `same_kind`. Checks run in NumPy's order: read-only `out` →
    loop → inputs → `out` → shape → values.
- Binding: `binary`/`unary` take an optional params object
  `{ dtype?, casting? }`. The shared `parse_casting` now lives in
  `dtype_binding`, and `canCast` uses it too.
- TS: `UfuncOptions` gains `dtype?: DTypeLike | null` and
  `casting?: Casting`. With `dtype`, JS scalars are weak relative to that
  dtype, as in NumPy: `add(int8Arr, 1000, {dtype: "int16"})` works, and
  `add(int8Arr, 1.5, {dtype: "int8"})` raises the input-cast error.
- Tests:
  - 4 new C++ test cases in `test_ufunc.cpp`.
  - 5 new vitest tests (block `ufunc dtype= / casting=`).
  - New differential group `ufunc_dtype_casting`
    (`python/generators/ufunc_dtype_casting_cases.py`) with 2198 cases, 803
    of which expect errors. It covers:
    - every ufunc × 7 input dtypes × 11 loop dtypes × {same_kind, unsafe}
    - `casting=` alone (5 rules) over mixed dtypes
    - `out` × `casting` × `dtype`
    - weak JS scalars with `dtype=`
    NumPy's `OverflowError` for an out-of-range Python int maps to
    `ValueError` (D-009).
- Verification:
  - `pnpm build`: clean.
  - `pnpm test`: 256 tests pass.
  - `pnpm test:diff`: 8656 tests pass.
  - `pnpm test:native` and `pnpm test:asan`: pass.
  - `pnpm api:check`: passes.
  - Type check: 3 pre-existing errors in `complex.test.ts`, none new.
- Not done: bench cases for the kwargs (planned for P2-11), `signature=`, and
  tuple `dtype`.

Next: P2-5 (`where=`).

## 2026-10-01 — P2-3: TS `{ out }` option on the ufuncs (D-047)

- `add subtract multiply divide power mod floorDivide abs negative sqrt exp
  log` take a final `{ out?: NDArray | null }` (exported type `UfuncOptions`).
  With `out` set, they call the D-046 native overloads and return the same
  `out` object. `null` or `undefined` allocates a new result, and a non-NDArray
  `out` raises `DTypeError`.
- Binding (`ops_binding.cpp`): `binary`/`unary` take an optional trailing
  `out` and return `undefined` when it is given.
- New vitest block `ufunc out=` in `ufunc.test.ts` with 4 tests: identity,
  casts and wraparound, strided/in-place/overlapping `out`, and errors.
- New differential group `ufunc_out`, generated by
  `python/generators/ufunc_out_cases.py`, with 347 cases:
  - a matrix of input dtype × `out` dtype for add/divide/abs/sqrt
  - all 12 ops into wider and narrower `out` dtypes
  - broadcasting, 0-d and empty arrays, weak scalars
  - strided, reversed, transposed and zero-stride `out`
  - overlap between inputs and `out`
  - read-only `out` and the order of errors
  The whole base buffer is compared against NumPy 2.5.3, so writes outside
  `out` would also be caught.
- Verification:
  - `pnpm build` and `pnpm typecheck` are clean.
  - `pnpm test`: 251 tests pass.
  - `pnpm test:diff`: 6458 tests pass.
  - `pnpm api:check` passes.
- Environment note: the build here needs
  `SDKROOT=$(xcode-select -p)/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk`.
  With the default, `xcrun` picks a broken CommandLineTools SDK. The repository
  itself is unchanged.

## 2026-10-01 — P2-2: native `out=` for `binary`/`unary` (D-046)

- New native overloads `binary(op, a, b, out)` and `unary(op, a, out)`
  (`native/core/ufunc.{hpp,cpp}`). The kernels now write into any `out` of
  the loop dtype; the overloads without `out` are unchanged and reuse them.
  When `out.dtype` differs from the loop dtype, the result goes into a
  temporary array that is then cast into `out`.
- The checks run in NumPy's order: read-only, then loop resolution, then
  the `same_kind` cast, then shape, then value checks. A second NumPy probe
  confirmed this order for calls with several faults.
- Overlap: an input that may share memory with `out` is copied first, unless
  it is exactly the same view. Results match NumPy for shifted, reversed,
  partially overlapping and broadcast-input cases.
- `assign` moved from `indexing.cpp` to the public `copy_into` in
  `broadcast.hpp`. Indexing setitem now uses it, with no behaviour change.
- No JS binding yet; that is P2-3.
- New C++ tests: `ufunc: out= dtype casts and broadcast`,
  `ufunc: out= strided and overlapping` and `ufunc: out= errors in NumPy
  order`. Expected values come from NumPy 2.5.3.
- Verification:
  - `pnpm test:native`: 88 cases, 0 failed.
  - `pnpm test:asan` is clean.
  - `pnpm test`: 247 tests pass.
  - `pnpm test:diff`: 6111 tests pass.

## 2026-10-01 — P2 sliced; P2-1: casting rules and `canCast` (D-045)

- P2 (ufunc machinery) is split into P2-1 to P2-11 in
  `docs/plan/TASK_SLICES.md`.
- Before slicing, NumPy 2.5.3's `out=` behaviour was checked by hand. Findings
  that P2-2/P2-3 must reproduce:
  - The loop dtype comes from the inputs only. For example, int8 + int8 into
    an int16 `out` wraps to -56. The result is then cast to `out` under
    `same_kind`; failures raise `UFuncTypeError`, which is a `TypeError`.
  - `out` must have exactly the broadcast shape. A smaller `out` raises
    `ValueError`.
  - A read-only `out` raises `ValueError`.
  - Overlapping operands behave as if the inputs were copied first.
  - The ufunc returns the `out` object itself.
- P2-1 adds native `Casting` and `can_cast` (`native/core/dtype.{hpp,cpp}`),
  which later P2 slices build on, and a public
  `np.canCast(from, to, casting = "safe")`.
  - Before implementing, the rules were checked against `np.can_cast` over
    all 5 rules × 14 × 14 dtypes: 0 mismatches.
  - New tests:
    - C++ test `dtype: can_cast rules`.
    - vitest cases in `views.test.ts`.
    - Differential group `casting` with 987 cases: the full table, the
      default rule, array arguments and bad rule names.
  - Bench case `api/canCast` added to both suites, plus a docs site entry.
- Verification:
  - `pnpm test`: 247 tests pass.
  - `pnpm test:diff`: 6111 tests pass.
  - `pnpm test:native` and `pnpm test:asan` pass.
  - `pnpm typecheck` is clean.
  - `pnpm api:check` passes. API coverage is 16.1% (171/1060), and 0
    callables are missing a benchmark.

## 2026-10-01 — P1-5e.6: complex bench cases; P1 complete

P1 (complex numbers, D-033) is done. Every existing path accepts complex64 and
complex128: conversion, ufuncs, reductions, the matmul family, all of
`linalg` on both backends, and FFT. Each path is checked against NumPy by a
differential group: `complex`, `complex_ufuncs`, `complex_reductions`,
`complex_matmul`, `complex_linalg` and `complex_fft`.
- Benchmarks, added to both suites with the same names and inputs:
  - `linalg/{inv,solve,det,svd,qr,eigh,eig} c128` at 32², 128² and 512²
    (`eig` only up to 512², as for real input). The inputs are
    CA = A + i·Aᵀ (diagonally dominant), the Hermitian CH = CA + CAᴴ for
    `eigh`, and cb = (1+i)·b.
  - `fft/{fft c128, fft c64, ifft c128}` at 1e3, 1e5 and 1e6 elements.
  - API cases `complex.linalg.norm`/`eigvals`/`eigvalsh`/`lstsq` at 32².
- Smoke run with `NATIVPY_BENCH_FILTER=c128` on one machine (M2 Pro,
  Accelerate, NumPy 2.5.3, median ms, nativpy vs NumPy):
  - `fft c128` 1e6: 9.96 vs 9.92.
  - `svd c128` 512²: 63.5 vs 68.4. `eig c128` 512²: 339 vs 352.
  - `solve c128` 512²: 4.08 vs 2.96. `det c128` 512²: 3.78 vs 2.53.
    These are the slowest cases, possibly the complex128 copy/promotion
    around `zgetrf`/`zgesv`. Not profiled yet.
  - The geometric mean over the 31 filtered cases was 0.958.
  These are single runs, not a performance claim, so they were not added to
  PERFORMANCE.md.
- `pnpm api:check` passes: 0 callables without a benchmark, and API coverage
  is 16% (170/1060).
- Checked by hand (temporary probe test, removed afterwards):
  - `where`, `take`, boolean indexing, `set`, `broadcastTo`, `full`, and
    `zeros`/`ones`/`empty`/`eye` with a complex dtype all work.
  - `mod`/`floorDivide` raise `DTypeError` (D-033).
  - `arange`/`linspace` with complex arguments raise `ValueError`. This is
    already listed as not supported in COMPATIBILITY.md.
- Still not in P1, by plan:
  - Complex `sort` waits for `sort` itself (P9).
  - Complex trig/comparison ufuncs wait for those ufuncs (P4/P5).
  - Complex versions of the linalg functions that don't exist yet
    (`pinv`, `cholesky`, …) belong to P11.
- Docs: ROADMAP (P1 ✅, coverage 16%), TASK_SLICES (P1-5e ✅), and the
  COMPATIBILITY "Not implemented" note on complex trig/comparisons, which no
  longer points to a P1 step.

Next: split P2 (ufunc machinery: `out=`, `where=`, `dtype=`, `casting=`,
`order=`, `ufunc.reduce/accumulate/...`, `errstate`) into slices in
`docs/plan/TASK_SLICES.md`, then start P2-1.

## 2026-10-01 — P1-5e.5: FFT accepts np.Complex input end to end

Checked that every complex FFT path works from JS `Complex` values in to
`Complex` values out. No library code changed: the existing paths
(`np.array` of nested Complex → pocketfft → `toArray()`) were already correct.
- New differential group `complex_fft`, with 120 cases from NumPy 2.5.3.
  - Unlike the `fft` group, which builds input with `fromTypedArray`, input
    is built with `np.array` from nested `np.complex` lists, or from plain
    `{ re, im }` objects.
  - Dtypes: inferred, complex64 and complex128.
  - Also covers mixed number/bool/complex lists, plus transposed, reversed
    and step-2 views (asserted non-contiguous).
  - Covers `fft`/`ifft`/`irfft`/`fft2`/`ifft2`/`fftn`/`ifftn` with
    `n`/`axis`/`norm`/`s`/`axes`.
  - Edge cases: NaN input (NaN/inf must match exactly), and the errors from
    `rfft` (DTypeError), 0-d input (IndexError) and empty input (ValueError).
  - Output is read through `toArray()`, and every complex element must be a
    `Complex`.
- Sanity check: with the tolerance set to −1, 115 of the 121 tests fail.
  The 6 that still pass are the error cases and the coverage check.
- Vitest: `np.fft` "accepts np.Complex input end to end".
- Checks: `pnpm test:diff` (5124), `pnpm test` (245), and `pnpm typecheck`
  all pass.

Next: P1-5e.6, bench cases, docs updates, and marking P1 ✅.


## 2026-10-01 — P1-5e.4: differential complex_linalg group

New differential group `complex_linalg`, with 211 cases from NumPy 2.5.3, run
on both backends (422 tests).
- Coverage: complex64/complex128 `det`, `inv`, `solve`, `eig`, `eigvals`,
  `eigh`, `eigvalsh` (Hermitian input), `svd` (full/reduced/values only),
  `qr` (reduced/complete/r), `lstsq` (1-D and 2-D b, rank-deficient), and
  `norm` (every vector/matrix ord, axis, keepdims). Also mixed real/complex
  `solve`/`lstsq` promotion, and errors (non-square, singular, NaN for eig,
  'fro' on a vector).
- Comparison: dtype and shape are exact. Values are within 1e-10 (complex128)
  or 2e-4 (complex64) relative to max|expected|. Eigenvalues are compared
  after sorting. Eigen/singular vectors and Q are checked by reconstruction
  and unitarity, and `qr(mode='r')` by |R|, because signs and phases depend
  on the backend.
- Sanity check: with the tolerance set to 0, 148 of the 422 tests fail, so
  the comparisons do run against real values.
- Fix: COMPATIBILITY.md lines added in P1-5e.3 contained internal references
  (D-044, P1-5e.4) that the package-docs check rejects. That broke
  `release_stage.test.ts`, which went unnoticed in P1-5e.3. Reworded.
- Checks: `pnpm test:diff` (5003), `pnpm test` (244), and `pnpm typecheck`
  all pass.

Next: P1-5e.5, FFT accepts `np.Complex` input end to end.


## 2026-10-01 — P1-5e.3: complex lstsq and norm (D-044)

`linalg.lstsq` and `linalg.norm` accept complex64/complex128.
- `lstsq`: result type follows `_commonType` (complex64 only if every operand
  is float32/complex64). `x` is complex; `residuals` and `s` are real (float32
  for complex64). The real SVD route is now generic: x = V·diag(1/s)·Uᴴ·b;
  residuals Σ|b − A·x|². Computed in complex128, cast once.
- `norm`: every ord works on |x|; 'nuc' and matrix ±2 use the complex SVD.
  Result float32 for complex64, else float64.
- Values checked against NumPy 2.5.3 within tolerance (not bitwise; NumPy
  uses `?gelsd` and a different summation order).
- Tests: C++ `linalg: complex lstsq/norm (D-044)`; vitest `complex
  linalg.lstsq / norm`. Both run on both backends. The earlier "lstsq rejects
  complex" guards are removed, since no linalg entry point rejects complex now.
- Checks: `pnpm typecheck`, `pnpm test:native` (84), `pnpm test:asan`,
  `pnpm test` (244), `pnpm test:diff` (4580) all pass.

Next: P1-5e.4, differential `complex_linalg` group.

## 2026-10-01 — P1-5e.2: complex eig/eigvals (D-043)

`linalg.eig`/`eigvals` accept complex64/complex128. Results keep the input's
complex dtype, and eigenvectors are unit-norm columns.
- Backend: `geev` accepts `v == nullptr` (values only, JOBVR='N'); new
  `cgeev`. Accelerate calls `cgeev_`/`zgeev_` with a queried `lwork` and
  `rwork` of 2n, as NumPy's `init_geev`. The fallback's complex Hessenberg +
  shifted QR is templated on real/complex input.
- `eigvals` is now a native values-only call, for real input too. A NumPy
  self-probe (n = 40–200) shows `eigvals` != `eig().eigenvalues` bitwise in
  2/5 cases at n >= 160.
- Complex input runs in complex128 and is cast at the end, as in NumPy.
- Probe against NumPy 2.x + Accelerate (arm64), 45 random n×n inputs
  (n = 1–12, 20, 32, 48; 15 per dtype):
  - Default backend, complex128 and complex64: `eig` values, `eigvals` and
    eigenvectors bit-identical in 15/15 cases each.
  - Default backend, float64: values 15/15; eigenvectors 10/15, the rest
    within 1 ulp. The same 5 cases also differ on the previous commit
    (existing difference, not investigated).
  - Fallback: reconstruction error ≤ 1.7e-14 (complex128/float64) and
    9.5e-7 (complex64).
- Tests: C++ `linalg: complex eig/eigvals (D-043)`; vitest `complex
  linalg.eig / eigvals`. Both run on both backends. The "still rejects
  complex" guards now use `lstsq`.
- Checks: `pnpm typecheck`, `pnpm test:native` (83), `pnpm test:asan`,
  `pnpm test` (240), `pnpm test:diff` (4580) all pass.

Next: P1-5e.3, complex `lstsq` and `norm`.


## 2026-10-01 — P1-5e.1: complex eigh/eigvalsh (D-042)

`linalg.eigh`/`eigvalsh` accept complex64/complex128 (Hermitian, lower
triangle). Eigenvalues are real (float32 for complex64, float64 for
complex128); eigenvectors keep the complex dtype.
- Backend: `syevd` gains a `vectors` (JOBZ) flag; new `cheevd`. Accelerate
  calls `ssyevd_`/`dsyevd_`/`cheevd_`/`zheevd_` with queried
  `lwork`/`lrwork`/`liwork`, as NumPy's `init_evd` does. The fallback Jacobi
  is generalised to Hermitian input: each pair is phase-aligned, then given
  the real rotation. The real path is unchanged.
- `eigvalsh` is now a native values-only call (JOBZ='N'), for real input too.
  Before, it returned `eigh(a).eigenvalues`. A probe run in NumPy itself
  shows those values differ bitwise from NumPy's `eigvalsh` in most cases.
- Complex input runs in complex128 and is cast at the end, as in NumPy.
- Probe against NumPy 2.x + Accelerate (arm64), 60 random n×n inputs
  (n = 1–29, 15 per dtype):
  - Default backend: `eigh` values, `eigvalsh` values and eigenvectors are
    bit-identical in 15/15 cases each for complex128, complex64 and float64.
  - float32: 0/15 bit-identical (max |Δw| 7.6e-6). NumPy computes float32
    input in float64; nativpy computes it in float32 (D-018). This is an
    existing difference, recorded as an open item in COMPATIBILITY.md.
  - Fallback: within 6.6e-14 (complex128/float64); largest reconstruction
    error 2.6e-14.
- Tests: C++ `linalg: complex eigh/eigvalsh (D-042)`; vitest `complex
  linalg.eigh / eigvalsh`. Both run on both backends. The "still rejects
  complex" guards now use `eig`.
- Checks: `pnpm typecheck`, `pnpm test:native`, `pnpm test:asan`,
  `pnpm test` (237), `pnpm test:diff` (4580) all pass.

Next: P1-5e.2, complex `eig`/`eigvals` (`?geev`).

## 2026-10-01 — P1 step 5: complex linalg (in progress, D-038–D-041)

P1-5d done: `linalg.svd` accepts complex64/complex128 with every
`fullMatrices`/`computeUV` combination. U and Vh keep the complex dtype; S is
real (float32 for complex64, float64 for complex128).
- New backend routine `cgesdd`: Accelerate `cgesdd_`/`zgesdd_`, with
  LAPACK's documented `lrwork`. The fallback reuses the one-sided Jacobi SVD:
  it phase-aligns each column pair, uses Aᴴ for wide input, and completes
  the basis with Hermitian inner products. The real path is unchanged.
- As in NumPy, complex input runs in complex128 and is cast at the end.
  Non-finite input (real or complex) raises `LinAlgError("SVD did not
  converge")` before LAPACK is called, matching NumPy's `svd_wrapper`.
- Measured against NumPy 2.x + Accelerate (arm64) on 60 random m×n cases
  (m, n = 1–29; both dtypes; full and reduced):
  - Default backend: S bit-identical 60/60; U and Vh bit-identical 57/60.
    The other 3 differ by at most 5.6e-17 absolute, in near-zero entries.
    The cause is not known. Our output is deterministic, and NumPy's is too.
  - Fallback: S bit-identical 32/60, largest |ΔS| 5e-14. Vectors are checked
    by reconstruction and unitarity only, since they are unique up to phase
    (D-018).
- Tests: C++ `linalg: complex svd (D-041)`; vitest `complex linalg.svd`. Both
  run on both backends.
- Checks: `pnpm test:native`, `pnpm test:asan`, `pnpm test` (233),
  `pnpm test:diff` (4580) all pass.

Next: P1-5e completes P1, in six sub-steps (see TASK_SLICES.md). It starts
with complex `eigh`/`eigvalsh`.

## 2026-10-01 — P1-5c (D-040)

P1-5c done: `linalg.qr` accepts complex64/complex128 in all modes
('reduced', 'complete', 'r').
- New backend routines `cgeqrf`/`cungqr`: Accelerate `cgeqrf_`/`zgeqrf_` and
  `cungqr_`/`zungqr_`. The fallback is a complex Householder QR using
  LAPACK's `zlarfg` convention, so R's diagonal is real.
- As in NumPy, complex input runs in complex128 and is cast at the end.
- Measured against NumPy 2.x + Accelerate (arm64) on 60 random m×n cases
  (m, n = 1–29; both widths; all modes). The default backend was
  bit-identical 60/60. The fallback was bit-identical 30/60, and its largest
  absolute difference was ~5e-15.
- Tests: C++ `linalg: complex qr (D-040)`; vitest `complex linalg.qr`. Both
  run on both backends.
- Checks: `pnpm test:native`, `pnpm test:asan`, `pnpm test` (229),
  `pnpm test:diff` (4580) all pass.

Next: P1-5d, complex `svd`. (Done; see above.)

P1-5b done: `linalg.inv` and `linalg.solve` accept complex64/complex128.
- New backend routine `cgesv`: Accelerate `cgesv_`/`zgesv_` with
  `ld = max(n, 1)`, as in NumPy; the fallback runs LU, then substitution.
- Matches NumPy's `_commonType`: complex input runs in complex128, then is
  cast. The result is complex64 only when every operand is float32/complex64.
- Measured against NumPy 2.x + Accelerate (arm64) on 210 random cases (`inv`,
  matrix `solve` and vector `solve`; n = 1–40; both widths). The default
  backend was bit-identical 210/210. The fallback backend's complex128
  results are within ~2e-14 relative.
- Tests: C++ `linalg: complex inv/solve (D-039)`; vitest `complex
  linalg.inv / solve`. Both run on both backends.

P1-5a done: `linalg.det` accepts complex64/complex128.
- New backend routine `cgetrf`: Accelerate `cgetrf_`/`zgetrf_`; the fallback
  reuses its LU with |re| + |im| pivoting.
- Value formula ported from NumPy's `umath_linalg` (sign × exp(Σ log|u_ii|)).
  As in NumPy, every complex det runs in complex128 and is then cast.
- Measured against NumPy 2.x + Accelerate (arm64) on 70 random matrices
  (n = 1–40, both widths). The default backend was bit-identical 70/70. A
  first version that worked in single precision for complex64 matched only
  39/70, and that probe is how the complex128 rule was found. The fallback
  backend's complex128 results are within ~2e-14 relative.
- Tests: C++ `linalg: complex det (D-038)` on both backends; vitest
  `complex linalg.det` on both backends.
- Checks: `pnpm test:native`, `pnpm test:asan`, `pnpm test` (218),
  `pnpm test:diff` (4580), `pnpm typecheck` and root `tsc` all pass.
- Not yet: differential cases and benchmarks (now part of P1-5e).

Next: P1-5c, complex `qr`.

## 2026-10-01 — P1 step 4: complex matmul family (done, D-035–D-037)

Done (slices P1-4a to P1-4e):
- `matmul`/`dot`/`inner`/`outer` accept complex64 and complex128, with no
  conjugation. P1-4a added the portable kernel (D-035). P1-4b made the
  Accelerate backend choose the same BLAS routine as NumPy (D-036).
- P1-4c: 14 vitest cases, run on both backends.
- P1-4d: new differential group `complex_matmul` with 262 cases, run on both
  backends (D-037).
  - Exactly representable inputs are compared bit-for-bit: dispatch shapes,
    batched/broadcast stacks, empty operands, all 12 real dtypes mixed with
    either complex width, non-finite values and errors.
  - Large random gemm/gemv/dotu cases use the D-018 tolerance.
  - With inf/NaN input, the fallback backend is checked against NumPy's own
    non-BLAS loop (`expected_noblas`).

Verification (P1-4d):
- `pnpm test:diff`: 4580 cases pass (was 4055); 525 of them are new
  `complex_matmul` checks, i.e. 262 cases × 2 backends plus a backend check.
- `pnpm test`: 213 tests pass. `pnpm test:native` passes. `pnpm typecheck`
  and the root `tsc` pass.
- No C++ changed in P1-4c/P1-4d, so ASan was last run in P1-4b.

P1-4e (bench): both suites gained `matmul c128`/`matmul c64` at 32²–1024² and
API cases `complex.dot`/`inner`/`outer`/`matmul`/`matmul matvec`. A smoke run
with a filter on one machine (Accelerate, median ms) gave nativpy vs NumPy:
c128 matmul 1024² 14.6 vs 15.8, 32² 0.0037 vs 0.0029; complex.dot on 1000
elements 0.0018 vs 0.0007. These are single runs, not a performance claim;
they were not added to PERFORMANCE.md. `pnpm api:check` passes (0 callables
without a benchmark).

Next: P1-5a, complex `det`.

## 2026-10-01 — P1 step 3: complex reductions (D-034)

Done (slices P1-3a to P1-3g):
- **`sum`/`prod`** on complex64 and complex128. Summation order follows D-021
  (pairwise along the trailing axis, sequential along a leading one). `prod`
  uses the project's complex multiply, so `(inf+0j)²` gives `nan+nanj` as in
  NumPy.
- **`mean`** keeps the complex dtype and divides with D-033 division.
- **`min`/`max`/`argmin`/`argmax`** order by real part, then imaginary part.
  A NaN in either part propagates, and the first NaN wins.
- **`var`/`std`** follow NumPy's `_var` and return the real dtype (float32 for
  complex64, float64 for complex128). `ddof` is supported.
- A `dtype=` that differs from the input, when either side is complex, raises
  `NotImplementedError` for `var`/`std` (D-034). Real input reduced into a
  complex dtype works for `sum`/`prod`/`mean`.
- `reject_complex()` was removed from `reduce.cpp`.
- Benchmarks: 11 complex reduction cases added to both suites
  (`complex.sum` ... `complex.std`, `complex.sum axis=0/1`). Report JSON files
  were not regenerated; no performance claims are made.

Verification:
- `pnpm test:native` and `pnpm test:asan` (ASan + UBSan) pass.
- `pnpm test`: 199 tests pass, including 7 new complex reduction cases.
- `pnpm test:diff`: 4055 cases pass, including 254 new `complex_reductions`
  cases.
  - `sum`/`mean`/`min`/`max`/`arg*` are compared exactly. `var`/`std` are
    exact on the 1000-element random inputs (both summation orders); on the
    small cases they use the real-float tolerance (D-017).
  - Complex `prod` uses rtol 1e-5 (complex64) or 1e-12 (complex128). NumPy's
    arm64 complex64 loop fuses one multiply, which gave a 1-ulp difference (D-034).
- `pnpm typecheck` and `pnpm api:check` pass. API coverage is still 16% (170/1060).
- Also fixed 7 strict-mode type errors in test files that predate this step
  (root `tsconfig.json`).

Known gaps:
- Complex `var` over non-adjacent axes (e.g. `axis=(0, 2)`) is within 1 ulp of
  NumPy, the same as real `sum`/`var` (D-017).
- `out=`, `where=` and the `nan*` reductions are not implemented, for real or
  complex.

Next: P1 step 4, the matmul family.

## 2026-09-30 — P1 step 2: complex ufuncs (D-033)

Done:
- **Kernels** in the new `native/core/complex_kernels.hpp`, following NumPy's
  algorithms:
  - `cdiv` uses Smith's algorithm (`nc_quot`);
  - `cpow` ports `npy_cpow`, with exact small-integer powers and `0^b` rules;
  - `cabs` uses the SIMD `cabsolute` formula;
  - `csqrt` ports msun `npy_csqrt`;
  - `clog` ports CPython `npy_clog`.
  `exp` uses `std::exp`.
- **Ufuncs.** `add`, `subtract`, `multiply`, `divide`, `power`, `negative`,
  `sqrt`, `exp` and `log` now work on complex64 and complex128.
  - `abs` and the new `angle` return the real dtype.
  - `mod` and `floorDivide` raise `DTypeError`.
- **New API** (+10 names): `np.real`, `np.imag`, `np.conj`, `np.conjugate`,
  `np.angle(z, deg)`, `np.iscomplex`, `np.isreal`, `np.iscomplexobj`,
  `np.isrealobj`, plus `NDArray.real`, `.imag` and `.conj()`.
  - `real` and `imag` are strided views that share the buffer and can be
    written through.
  - For a real array, `imag` is a read-only zeros array, as in NumPy.
- **Weak complex scalars (NEP 50).** A JS `{re, im}` operand gives complex64
  with float16, float32 and complex64 arrays, and complex128 otherwise.
- Updated existing tests that expected complex ufuncs to throw
  `NotImplementedError`. They now check the supported behaviour and the
  `DTypeError` for `mod` and `floorDivide`.

Verification:
- `pnpm test`: 192 tests pass.
- `pnpm test:diff`: 3801 cases pass, including 79 new `complex_ufuncs` cases.
  - They are exact for add, subtract, negative, conjugate, real, imag and the
    predicates.
  - The libm-based ops use rtol 1e-14 for complex128 and 1e-6 for complex64.
  - inf, nan and signed-zero categories must match exactly.
- `pnpm test:native`: 71 cases pass, including 4 new ones in
  `test_complex.cpp`.
- `pnpm test:asan` (ASan + UBSan) passes.
- `pnpm typecheck` and `pnpm api:check` pass. API coverage is 16%
  (170/1060), and every callable is benchmarked in both suites.
- Build note: this machine's Command Line Tools SDK is broken (`tapi`
  "unknown architecture"). Builds were run with
  `SDKROOT=$(xcrun --show-sdk-path)` pointing at the Xcode SDK.

Next: P1 step 3, complex reductions.

## 2026-09-30 — P0 NumPy API coverage tooling (D-032)

Done:
- **D-032 recorded.** It covers the parity programme and scope defaults a–d:
  - no `object_` dtype;
  - `longdouble` aliases float64;
  - deprecated APIs go last;
  - build/introspection names are excluded, each with a reason.
- **`python/api_inventory.py`** writes the checked-in `api/numpy-api.json`
  (NumPy 2.5.3, 1168 names across 14 surfaces).
- **Exclusions and aliases.** `api/exclusions.json` has 91 entries, each with a
  reason. Two of them are wildcards for `testing` constants and classes, so 108
  names are excluded in total. `api/aliases.json` is empty: matching ignores
  case and `_`.
- **Coverage tool.** `scripts/api-coverage.mjs` and `scripts/api-coverage-lib.mjs`
  measure API coverage against the built package. They also check benchmark
  coverage by scanning both suites for call sites.
  - `--check` fails if the implemented count drops, if a new callable lacks a
    benchmark in either suite, or if an alias target is missing.
  - It runs in CI as `pnpm api:check`.
- **Benchmarks.** An `api` category was added to both suites. It covers 92
  cases, so all 158 implemented names are benchmarked; `benchIgnore` is empty.
- **Bug fix.** `benchmarks/compare.mjs` read `packages/nativpy/package.json`,
  which the rename had removed. It now reads `packages/numera`.

Verification:
- `pnpm test`: 173 tests pass, including 5 new ones in `api_coverage.test.ts`.
- `pnpm typecheck` passes.
- `pnpm api:check` passes.
- Both suites ran with `NATIVPY_BENCH_FILTER=api/`, and `compare.mjs` paired all
  92 `api/*` rows.

Measured: API coverage is **14.9% (158/1060)** after 108 exclusions.

Next: P1, complex numbers (D-008).

## 2026-09-30 — Manual releases only; release workflow removed (D-031)

Done:
- Deleted `.github/workflows/release.yml` and the files only it used:
  `scripts/ci-pack.mjs`, `scripts/next-version.mjs` and
  `test/release_version.test.ts`. `ci.yml` (tests on push/PR) is kept.
- The tarball checks from `ci-pack.mjs` now run in `scripts/release.mjs`
  (`checkTarball`), before the smoke test and the publish:
  - prebuilds;
  - README/LICENSE/COMPATIBILITY/`docs/index.html`;
  - no `.map` files;
  - no `repository`/`bugs`;
  - the exact unpkg `homepage`.
- README "Publishing a release" now describes the manual `pnpm release` flow
  only. `release_stage.test.ts` checks that the workflow is gone and that the
  README has no workflow/`NPM_TOKEN` instructions.

Not verified: a real `pnpm release` / `release:dry` run after this change (needs
Docker and npm login). The `checkTarball` logic is the same as the
`ci-pack.mjs` checks that passed earlier today.

## 2026-09-30 — API reference site as npm homepage; npm-only publishing (D-030)

Done:
- Lodash-style API reference: `docs/site/api.mjs` (75 entries in 12 categories:
  signature, arguments, returns, example) rendered by `scripts/build-docs.mjs`
  (`pnpm docs`) into one self-contained `packages/numera/docs/index.html`
  (~97 KB, no external requests, sidebar search with `/` shortcut, dark mode,
  mobile menu). Staged by `stage-package.mjs` and shipped via `files: ["docs"]`.
- Deploy sets `homepage` to `https://unpkg.com/<name>@<version>/docs/index.html`
  (`scripts/set-homepage.mjs` → `npm pkg set`) in `release.yml` and
  `scripts/release.mjs`; not committed. `ci-pack.mjs` requires exactly that URL
  and `docs/index.html` in the tarball.
- Removed GitHub Packages publishing (workflow `registry` input, GPR pack/publish,
  `packages: write`, `ci-pack.mjs --name`) and the README `.npmrc` section.
- `test/docs_site.test.ts` runs every example against the addon and checks each
  `// =>` result; also checks every public function is documented. A deliberately
  wrong result (det 5 → 6) fails it.

Verified: `pnpm typecheck`; `pnpm test` 174/174; staged `npm pack --dry-run`
includes `docs/index.html` and the homepage; headless Chrome render.
Not verified: live unpkg serving of this package (only after the next publish).

## 2026-09-30 — Automatic branch-channel releases; self-contained package (D-029)

Done:
- `.github/workflows/release.yml` runs on push to `main` / `beta` / `alpha`,
  publishing to npm dist-tag `latest` / `beta` / `alpha` with the `NPM_TOKEN`
  secret (no prompts). The version comes from Conventional Commits
  (`scripts/next-version.mjs`); docs/chore/test-only pushes skip the build and
  publish. Prereleases are `X.Y.Z-<channel>.N` with N past every git tag and npm
  version already taken. Stable releases commit `package.json`; prereleases
  only tag. GitHub Release: latest vs pre-release.
- Self-contained tarball: `scripts/stage-package.mjs` writes the package README,
  COMPATIBILITY.md and LICENSE, and strips PLAN/DECISIONS references from dist
  comments. It fails on leftovers. Source maps are excluded, and `package.json`
  has no repository/homepage/bugs. `ci-pack.mjs` checks the packed tarball.
  The addon's "no prebuild" error no longer tells users to run `pnpm build:native`.
- Verified locally: 11 new vitest cases (`release_version`, `release_stage`) pass.
  actionlint (with shellcheck) on the workflows passes. `npm pack --dry-run` gives
  30 files, 0 `.map`, with README/LICENSE/COMPATIBILITY. Staged dist passes
  `node --check` and the smoke test. `next-version.mjs` against the real repo:
  no release (only docs/chore since `v1.0.0`).

Not verified: the workflow has not run on GitHub yet. It needs the `NPM_TOKEN`
secret and the `beta`/`alpha` branches. The linux-arm64 job's `ubuntu-24.04-arm`
runner must be available to this private repo; if not, that job fails before
anything is published.

## 2026-09-30 — First npm release: `@cyfora/numera@1.0.0`

Done:
- **Published** `@cyfora/numera@1.0.0` to npmjs.org (public, `latest`) with the
  local `pnpm release`, by npm account `cyfora`. Tag `v1.0.0` (on `3d98b89`) is
  pushed to `origin`.
- Verified from the registry: the tarball has 46 files, 7.1 MB unpacked, with all four
  prebuilds (darwin-arm64/x64, linux-x64/arm64). `npm install @cyfora/numera@1.0.0`
  in a clean project imports it and passes the smoke test on darwin-arm64 (Node
  22.7.0). `linalg.inv` / `det` run there too. The Linux and darwin-x64 prebuilds
  were tested at build time (Docker / packing) and have not been checked from
  the published package on those platforms.
- Not done: no GitHub Release or GitHub Packages publish yet. The optional
  workflow has never run. The next CI release must use `bump=patch` or higher,
  because `v1.0.0` exists.

## 2026-09-30 — npm name `@cyfora/numera` (D-028)

Done:
- npm rejected the unscoped `numera` as well ("too similar"). The npm package
  is now **`@cyfora/numera`**, published under the `cyfora` npm org.
  GitHub Packages stays `@rajankr542/numera`. The workflow now strips the npm
  scope before adding the owner scope.
- `pnpm release` checks up front, before the build, that the logged-in npm user
  belongs to the package's scope (or that the scope is the user's own). The
  first publish was done as npm user `cyfora`, the scope owner.

## 2026-09-30 — GitHub Packages + manual GitHub Actions release (D-027)

Done:
- `npm publish nativpy` was rejected by npm with E403: "Package name too similar
  to existing package natives". The npm package is now **`numera`**. It is
  published to GitHub Packages as **`@rajankr542/numera`**, because the owner
  scope is required there. The import name changes; the API does not. `numera`
  may still hit npm's similarity check (close to the popular `numeral`). The
  fallback is `@rajankr542/numera` on npm too (D-027).
- Release path: npm releases are published **manually from the local machine**
  with `pnpm release`. The Actions workflow is optional (GitHub Packages +
  GitHub Release). `release.mjs` now explains npm's "too similar" E403 and
  points to the scoped name. `package.json` repository URLs point to
  `Rajankr542/numera` (the GitHub repo was renamed).
- `.github/workflows/release.yml` (manual `workflow_dispatch`, `main` only):
  prebuilds on native runners, pack + smoke test (`scripts/ci-pack.mjs`, with
  `--name` to pack under the scoped name for GitHub Packages), publish to
  GitHub Packages and/or npmjs.org, commit + tag `vX.Y.Z`, and a GitHub Release
  marked latest with the tarballs attached.
- macOS prebuilds now pin `MACOSX_DEPLOYMENT_TARGET=13.3`. The earlier ones
  required the build host's macOS 26.5. 13.3 is the floor set by the
  `ACCELERATE_NEW_LAPACK` symbols.
- Verified locally: actionlint clean; `ci-pack.mjs` packs all 4 prebuilds
  for both `numera` and `@rajankr542/numera`, and each installed package passes
  the smoke test; typecheck and 83 unit tests pass. The workflow itself has not
  run on GitHub yet, and `numera` has not yet been accepted by npm.


## 2026-09-30 — npm packaging and local release command (D-026)
- `packages/nativpy` ships `dist/` + `prebuilds/<platform>-<arch>/nativpy.node`.
  The loader checks the bundled prebuild before the repo build. Keywords,
  repository metadata, LICENSE and README are included in the package.
- Prebuilds use the stable Node-API (`NATIVPY_NAPI_EXPERIMENTAL=OFF`). The
  stable addon passes 83 unit + 3677 differential tests (macOS arm64).
- `pnpm prebuilds` (`scripts/build-prebuilds.mjs`): darwin-arm64 and darwin-x64
  built locally. linux-x64/arm64 are built in Docker `manylinux_2_28` with
  `NATIVPY_STATIC_RUNTIME` and need only glibc ≥ 2.27. linux-arm64 (native) passes
  81 vitest tests (2 skipped). linux-x64 (QEMU, where esbuild crashes) passes the smoke test.
  The linux-arm64 prebuild also passed the smoke test on Debian 11/Node 18 and Debian 12/Node 20.
  Fixed along the way: GCC 12 `-Wrestrict` false positive (`-Wno-restrict`
  on GCC < 13), stale corepack keys (install pnpm via npm), and the build script
  now auto-starts Docker Desktop and falls back to the Xcode SDK.
- Package version set to 1.0.0.
- `pnpm release` (`scripts/release.mjs`): npm web login → version bump →
  build/test → prebuilds → pack + smoke test → publish → commit + tag.
  Not run end-to-end here, because it needs the maintainer's npm login.
- Verified: an `npm pack` tarball (1.15 MB) installed into a clean folder and
  passed `scripts/smoke-test.mjs` on Node 18.20, 20.20, 22.7 and 24.21 (darwin-arm64).
- Local toolchain note: on this machine a fresh CMake configure selects the
  CommandLineTools MacOSX27.0 SDK, and the linker rejects it
  ("unknown architecture arm64e.x1"). `SDKROOT=<Xcode MacOSX26.5.sdk>` works around it.

## 2026-09-29 — M11 step 4: per-call overhead (D-024, partial)

Done:
- The native construction guard now uses an identity token instead of a string tag. transpose 32² went from 1185 to ~1035 ns and reshape view from 1196 to ~1100 ns.
- New test: direct `new NativeNDArray(...)` is rejected, including with the old tag string.
- Profiled with `sample`: the remaining ~700 ns per result is `napi_new_instance`, `napi_wrap` and weak refs. The handle-model change is deferred (D-024).
- Verification: `pnpm test` 83 pass; `pnpm test:diff` 3677 pass; native 67 pass; ASan clean.

## 2026-09-29 — M11 step 3: GC-time buffer release (D-023)

Done:
- Addon built with `NAPI_EXPERIMENTAL` + `NODE_ADDON_API_REQUIRE_BASIC_FINALIZERS` (CMake option `NATIVPY_NAPI_EXPERIMENTAL`, default ON). `~NDArrayWrap` now runs synchronously during GC.
- New test `packages/nativpy/test/memory.test.ts`: a child process with `--expose-gc` runs 1000 sync adds, then `gc()`, and expects ≤ 2 live buffers. Verified to **fail** on the stable build (1001 live) and pass on the experimental one.
- Stress probe with views outliving parents, temporaries and forced GC: values stay valid, and live buffers drop to 0 after release.
- Full suite geo-mean went from 0.285 to 0.628 vs NumPy (D-021..D-023 combined). See PERFORMANCE.md "M11 step 3". Known regression: `zeros 1e6` went from 0.049 to 0.104 ms.
- Verification: `pnpm test` 82 pass; `pnpm test:diff` 3677 pass; native 67 cases pass; ASan clean.

Notes:
- cmake-js `--CD...` flags did not reconfigure an existing `build/` cache. Toggling the option needs `cmake -S . -B build -DNATIVPY_NAPI_EXPERIMENTAL=OFF`.
- Experimental Node-API is not ABI-stable. Prebuilds (M13) must be tested per Node major.

## 2026-09-29 — M11 step 2: matmul wrapper copies (D-022)

Done:
- `matmul_2d` uses contiguous, same-dtype, non-broadcast operands in place, and the output is `empty` instead of `zeros`.
- The baseline "matmul f32 128² 66× slower" figure did not reproduce (0.014 ms in isolation). This is recorded in PERFORMANCE.md.
- New native test (D-022): an offset view, a transposed operand, a broadcast batch and dirty-heap `empty` output for float/int/k=0, run on both backends.
- Verification: native tests 67 pass; ASan clean; `pnpm test` 81 pass; `pnpm test:diff` 3677 pass.
- Matmul geo-mean went from 0.561 to 0.679. f32 128² went from 0.014 to 0.011 ms (NumPy 0.005).

## 2026-09-29 — M11 step 1: reduction kernels (D-021)

Done:
- Benchmark suite, NumPy mirror and comparison report added (committed separately). Baseline is recorded in PERFORMANCE.md.
- `native/core/reduce.cpp`:
  - No input copy for C-contiguous input when the reduced axes trail.
  - Column sweep, with no transpose, when the reduced axes lead.
  - NumPy pairwise float sum; `mean`/`var`/`std` use it too.
  - 16-lane vectorizable `min`/`max` with NaN and signed-zero rescans.
  - Op dispatch moved out of the element loops.
- Tests:
  - New native test case "D-021 fast paths": NaN at lane seed, body and tail; ±inf; signed zero; int lanes; pairwise block regimes; column sweep vs. transposed path for all ops.
  - 77 new exact differential cases (`d021_cases()` in the generator). The pre-change kernel fails 34 of them.
  - Long differential labels are now truncated and hashed.
- Results (see PERFORMANCE.md "M11 step 1"):
  - Reduction geo-mean went from 0.237 to 0.950 of NumPy speed.
  - `sum f64` 1e6: 1.114 → 0.110 ms (NumPy 0.174).
  - `max f64` 1e6: 1.315 → 0.186 ms (NumPy 0.093).
  - `sum axis=0` 1024²: 3.056 → 0.154 ms (NumPy 0.160).
- Verification (macOS arm64, NumPy 2.5.3):
  - Native build passes.
  - `pnpm build:ts` passes.
  - `pnpm test:native` and `pnpm test:asan` pass (66 cases, ASan+UBSan clean).
  - `pnpm test`: 81 tests pass.
  - `pnpm test:diff`: 3659 tests pass.

Notes:
- Environment: a fresh cmake-js configure picked up a broken CommandLineTools 27.0 SDK and failed to link. The build works when `--CDCMAKE_OSX_SYSROOT=<Xcode MacOSX26.5.sdk>` is passed. This is local to this machine; no repo change was made.
- Not done: `max` is 2× NumPy.
- Step 1b: `argmin`/`argmax` reuse the vectorized value search. 1e6 f64 went from 4.015 to 0.363 ms (NumPy 0.645). 18 new exact diff cases; `pnpm test:diff` 3677 pass; native, ASan and unit tests pass.

## 2026-09-29 — M10 FFT (D-020)

Done:
- Vendored pocketfft (`third_party/pocketfft/`, BSD-3), pinned to the same commit NumPy 2.x uses (`33ae5dc9`).
- C++ `native/fft/` (static lib `nativpy_fft`):
  - `fft.{hpp,cpp}`: per-lane c2c/r2c/c2r loops ported from NumPy's `_pocketfft_umath.cpp` (zero-pad/truncate to `n`, FFTpack packing for rfft/irfft).
  - N-D composition following `_raw_fftnd`/`_cook_nd_args`; NumPy's dtype, `norm` factor and error rules, including float16 factors rounded to half precision.
  - `fftfreq`/`rfftfreq`.
- Bindings: `native/bindings/fft_binding.cpp` exposes `exports.fft`.
- TS: `src/fft.ts` provides `np.fft.{fft,ifft,rfft,irfft,fft2,ifft2,fftn,ifftn,fftfreq,rfftfreq}`. Arguments can be positional (NumPy order) or a trailing options object. Also available as the named export `fft`.
- Tests:
  - `tests/native/test_fft.cpp` (7 cases vs. a naive DFT: pad/truncate, norms, even/odd rfft/irfft, dtypes, axes, strided input, fftn, errors, freqs).
  - `packages/nativpy/test/fft.test.ts` (7 tests).
  - A new differential group `fft` with 697 cases.
- Differential tests found two bugs, both now fixed:
  - Error precedence: NumPy checks an explicit `n` before the axis.
  - float16 `ortho`/`forward` factors: NumPy computes them in half precision.
- Verification (macOS arm64, NumPy 2.5.3):
  - `pnpm build` passes.
  - `pnpm typecheck` is clean.
  - `pnpm test:native` and `pnpm test:asan` pass (65 cases, ASan+UBSan clean).
  - `pnpm test`: 81 tests pass.
  - `pnpm test:diff`: 3582 tests pass.
- Not done (see COMPATIBILITY.md):
  - `rfftn`/`irfftn`, `hfft`, `fftshift` and `out=`.
  - No FFT benchmarks, so no performance claims.

## 2026-09-29 — M9 random (D-019)

Done:
- C++ `native/random/` (its own static lib, `nativpy_random`):
  - `bitgen.{hpp,cpp}`: `SeedSequence`, `PCG64`, and `MT19937` with legacy integer and array seeding.
  - `distributions.{hpp,cpp}`: 53-bit and 24-bit floats, ziggurat normal (float64/float32), and the legacy polar gauss with its cache.
  - Also in `distributions`: Lemire and masked bounded integers with NumPy's buffered 8/16/32-bit draws and bool bit-buffering; `shuffle`; and Generator `choice` indices (Floyd + hash set, partial tail shuffle).
- Bindings: `native/bindings/random_binding.cpp` exposes `exports.random.BitGenerator`.
- TS: `src/random.ts` provides `defaultRng`/`Generator`, `RandomState`, and the `np.random` namespace with the legacy global functions. Positional NumPy argument order or an options object both work.
- Fixed: float32 `normal` silently ignored `loc`/`scale`. It now raises `ValueError` for non-default values; NumPy only exposes float32 through `standard_normal`.
- PLAN examples `normal([n])` changed to `normal({ size: [n] })`. In NumPy order the first positional argument is `loc`.
- Tests:
  - `tests/native/test_random.cpp` (8 cases, NumPy reference values).
  - `packages/nativpy/test/random.test.ts`.
  - A new differential group `random`: 10 seeded streams, 30 or 21 chained calls each. The test requires **exact** equality.
- Verification (macOS arm64, NumPy 2.5.3):
  - `pnpm build` passes.
  - `pnpm typecheck` is clean.
  - `pnpm test:native` and `pnpm test:asan` pass (58 cases, ASan+UBSan clean).
  - `pnpm test`: 74 tests pass.
  - `pnpm test:diff`: 2885 tests pass.
- Build note: on this machine cmake-js needs
  `SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX26.5.sdk`
  and `.venv/bin` on `PATH`.
- Not done (see COMPATIBILITY.md):
  - `choice(p=...)`.
  - Broadcast array parameters.
  - Other distributions.
  - No benchmarks, so no performance claims.

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

## P8 — Indexing extras (branch p08) — 2026-10-02
- Native `native/core/p08_indexing.{hpp,cpp}` (binding `addon.p08`), TS
  `p08.ts` plus `take` `mode=` in `indexing.ts` (D-110, D-111):
  `takeAlongAxis putAlongAxis put putmask place choose compress extract select
  piecewise argwhere flatnonzero countNonzero ravelMultiIndex unravelIndex
  diagonal trace`, `take` `mode=`, NDArray `choose compress diagonal nonzero put
  take trace`. `nested_iters` stays excluded.
- Checks: `pnpm build`, `pnpm test` (322 tests), `pnpm test:native`,
  `pnpm test:asan` (ASan+UBSan clean), `pnpm typecheck` and `pnpm api:check` pass;
  API coverage 16.7% → 19.0% (201/1060). Benchmarks and NumPy differential cases
  are deferred to the V phase (bench-exempt in `api/bench-exempt/p08.json`).
## 2026-10-02 — P7 creation and grids (D-100, D-101)

Done: `logspace`, `geomspace`, `tri`, `tril`, `triu`, `diag`, `diagflat`, `vander`,
`indices`, `meshgrid`, `mgrid`, `ogrid`, `ix_`, `trilIndices`, `triuIndices`,
`trilIndicesFrom`, `triuIndicesFrom`, `diagIndices`, `diagIndicesFrom`,
`maskIndices`, `fillDiagonal` (wrap), `fromfunction`, `fromiter`, `frombuffer`,
`fromstring` (text mode, port of NumPy's parser), `r_`, `c_`, `s_`, `indexExp`,
`np.astype`. Kernels in `native/core/p07_creation.cpp` (`addon.p07`).
Verified: `pnpm build`, `pnpm test`, `pnpm test:native`, `pnpm api:check`, `pnpm test:asan`.
Benchmarks are not written yet; the callables are listed in `api/bench-exempt/p07.json`.
Divergences: see COMPATIBILITY.md "P7 creation and grids divergences".
## 2026-10-02 — P5: comparison, logic and bitwise (branch p05)

Done (D-080–D-084):
- Comparison ufuncs `equal notEqual less lessEqual greater greaterEqual` and
  logical `logicalAnd logicalOr logicalXor logicalNot` with `bool` output
  (complex compared lexicographically, reduce/accumulate with NumPy identities);
  `all`/`any` and `NDArray.all()/any()`.
- `isnan isinf isfinite isnat isposinf isneginf`, `isscalar`.
- Bitwise `bitwiseAnd bitwiseOr bitwiseXor invert` (`bitwiseNot`/`bitwiseInvert`
  are the same object), `leftShift rightShift` (`bitwiseLeftShift`/`bitwiseRightShift`
  aliases), `bitwiseCount`.
- `isclose allclose arrayEqual arrayEquiv` (native isclose kernel).
- `packbits unpackbits` (`axis`, `bitorder`, `count`).
- Shared file touched (kept small): `native/core/ufunc_methods.cpp` (reduce loop
  dtype for bool-output ufuncs, integer identity fill via int64).
- Checks run on macOS arm64 with NumPy 2.5.3: `pnpm build`, `pnpm test`
  (325 tests), `pnpm test:native`, `pnpm test:asan` (ASan+UBSan clean),
  `pnpm test:diff` (10420 cases, none for P5 yet), `pnpm typecheck` and
  `pnpm api:check` all pass. Expected values in the P5 unit tests were checked
  by hand against NumPy.
- Not done: NumPy differential cases for P5, because the generator
  (`python/generators/generate_cases.py`) is a shared file; benchmarks (all
  P5 names are listed in `api/bench-exempt/p05.json`); array-valued
  `rtol`/`atol` in `isclose`.
## 2026-10-02 — P9 sorting, searching and set functions (D-120–D-124)

Done:
- Native kernels in `native/core/p09_sort_kernels.hpp`, `p09_sorting.cpp` and `p09_sets.cpp`
  for every dtype (bool, ints, float16/32/64, complex):
  - ports of NumPy's introsort and introselect;
  - `std::stable_sort` for the stable kinds;
  - batched binary search;
  - `_unique1d`-style unique for 1-D input and along an axis.
- `sort argsort sortComplex partition argpartition lexsort searchsorted unique
  uniqueAll uniqueCounts uniqueInverse uniqueValues intersect1d union1d setdiff1d
  setxor1d isin ediff1d`, plus the methods `NDArray.sort/argsort/partition/argpartition/searchsorted`.
- Verified locally: `pnpm build`, `pnpm test` (324), `pnpm test:native`, `pnpm api:check`.
  `partition`/`argpartition` reproduce NumPy's exact output on the reference vector.
- Not done: benchmarks (names listed in `api/bench-exempt/p09.json`) and NumPy
  differential cases for P9.
## 2026-10-02 — P6 array manipulation, build-first (D-090–D-094)
- Native: `native/core/p06_manip.{hpp,cpp}` has kernels for concatenate, split views, tile, repeat, resize (copying and in-place), pad (a per-lane kernel for every mode), insert_along, delete_along, trim_zeros, flip, roll, copyto and all_finite. Binding: `addon.p06` (`native/bindings/p06_binding.cpp`).
- TS (`packages/numera/src/p06*.ts`): concatenate/concat, stack, vstack, hstack, dstack, columnStack, block, unstack; split, arraySplit, hsplit, vsplit, dsplit; tile, repeat; pad; append, insert, delete, resize, trimZeros; flip, fliplr, flipud, roll, rollaxis, rot90, permuteDims, matrixTranspose; atleast1d/2d/3d, broadcastArrays; asanyarray, asarrayChkfinite, require, copyto; shape, size, ndim, isfortran; applyAlongAxis, applyOverAxes. Two NDArray methods are added: `repeat`, and `resize`, which changes the array in place.
- Tests: `tests/native/test_p06_manip.cpp` and `packages/numera/test/p06_*.test.ts`. Every example in the docs entries (`docs/site/parts/p06.mjs`) runs as part of the tests.
- Verified: `pnpm build`, `test` (352), `test:native`, `test:asan`, `typecheck` and `api:check` pass. All P6 np/ndarray names count in `api:coverage`.
- Not yet done (V phase): NumPy differential groups and benchmarks; P6 names are listed in `api/bench-exempt/p06.json`. No NumPy-compatibility or performance claims are made for P6.

## P12 — FFT completion (D-150..D-152) — 2026-10-02
- `np.fft.hfft ihfft rfftn irfftn rfft2 irfft2 fftshift ifftshift` (`axes` as
  number/list/null), and `out=` (positional or options) for all 14 transforms.
  Native: `native/fft/p12_fft.{hpp,cpp}` (pocketfft lane kernels, NumPy loop
  precision, `out` checks, roll), `addon.p12`; TS in `fft.ts` (`fftModule`).
- Shared file touched (minimal): `packages/numera/test/fft.test.ts` (expected key list).
- Checks (macOS arm64, NumPy 2.5.3): `pnpm build`, `pnpm test` (424), `pnpm test:native`,
  `pnpm test:asan`, `pnpm test:diff` (10420), `pnpm typecheck`, `pnpm api:check`
  all pass. An ad-hoc NumPy comparison (1530 cases) was bit-exact.
- Not done: committed NumPy differential cases and benchmarks for the new names
  (listed in `api/bench-exempt/p12.json`).
## P4 — Math ufuncs (branch p04, 2026-10-02)
- About 75 public names: trig/hyperbolic (with Array-API aliases), exp/log,
  rounding (`round/around/fix`, `a.round`), arithmetic (`fmod floatPower sign
  heaviside maximum minimum fmax fmin fabs clip`, `a.clip`, `a.conjugate`,
  aliases `remainder trueDivide pow absolute`), float bits (`copysign
  nextafter spacing ldexp signbit`), `gcd lcm`, the two-output ufuncs
  `divmod modf frexp`, and `i0 sinc unwrap nanToNum realIfClose`
  (D-070–D-074).
- Verified locally: `pnpm build`, `pnpm test` (343 tests), `pnpm test:native`,
  `pnpm test:asan`, `pnpm api:check` all pass. NumPy differential and
  benchmark cases are deferred to the V phase (`api/bench-exempt/p04.json`).

## P11 — Linear algebra completion (branch p11) — 2026-10-02
- Native `native/linalg/p11_linalg.{hpp,cpp}` (binding `addon.p11`): cholesky
  (Accelerate `?potrf` + portable fallback, D-142), slogdet, matrix_power, pinv,
  matrix_rank, cond, and the einsum contraction core (D-141). TS `p11_linalg.ts`,
  `p11_products.ts`, `p11_einsum.ts` (D-140, D-141).
- `np.linalg`: `cholesky slogdet svdvals matrixPower pinv matrixRank cond
  vectorNorm matrixNorm matrixTranspose diagonal trace outer tensorinv
  tensorsolve cross tensordot multiDot vecdot`. `np`: `vdot kron cross tensordot
  vecdot matvec vecmat einsum einsumPath`. NDArray `dot`.
- Checks: `pnpm build`, `pnpm test` (420 tests, both linalg backends),
  `pnpm test:native`, `pnpm test:asan` (ASan+UBSan clean), `pnpm typecheck`,
  `pnpm api:check` pass. An ad-hoc einsum/einsumPath comparison against NumPy
  2.5.3 (22 expressions, greedy and optimal) matched results, paths and reports.
  Benchmarks and checked-in NumPy differential cases are deferred to the V phase
  (`api/bench-exempt/p11.json`).
## P3 — NDArray methods, iteration, dtype introspection, printing (branch p03) — 2026-10-02
- P3-2 (D-060) `fill`, `tolist`, `tobytes`, `view(dtype)`, `byteswap`, `setflags`, `base`, `mT`, `flat`, `astype({copy, casting})`.
- P3-3 (D-061) `ndindex`, `ndenumerate`, read-only `nditer`.
- P3-4 (D-062) `finfo`, `iinfo`, `resultType`, `minScalarType`, `issubdtype`, `isdtype`, `commonType`, `mintypecode`, abstract dtypes.
- P3-5 (D-063) native Dragon4 port; `formatFloatPositional`/`formatFloatScientific`,
  `array2string`, `arrayRepr`, `arrayStr`, `setPrintoptions`/`getPrintoptions`/`printoptions`;
  `NDArray.toString()` is now NumPy's repr.
- Verified locally (macOS arm64, Node 22.7, NumPy 2.5.3): `pnpm build`, `pnpm test`
  (362 tests), `pnpm test:native`, `pnpm test:asan`, `pnpm test:diff`, `pnpm api:check`,
  `pnpm typecheck` pass. Ad-hoc comparisons against NumPy (not yet in `test:diff`): 3000 random
  `format_float_*` calls and 600 random `array2string`/`repr` calls (float16/32/64,
  complex, ints, bool, all print options) gave identical output.
- Gaps: NumPy differential cases for P3 are not in `test:diff` yet; no benchmarks
  (bench-exempt); `legacy` print modes, writable `nditer`, `nditer` buffering flags.
## 2026-10-02 — P14 extra dtypes and I/O, partial (D-170–D-173)
- I/O: `save load savez savezCompressed` (.npy v1/v2/v3 and .npz with stored/deflate), `loadtxt savetxt` (native tokenizer and formatter), `genfromtxt fromregex`, `fromfile` and `NDArray.tofile`.
- Misc: `baseRepr binaryRepr`; native windows `bartlett blackman hamming hanning kaiser`; legacy polynomials `poly poly1d polyadd polyder polydiv polyfit polyint polymul polysub polyval roots` (native convolve/polydiv).
- Tests: `tests/native/test_p14_*.cpp`, `packages/numera/test/p14_*.test.ts`, docs examples in `docs/site/parts/p14.mjs`. Verified: `pnpm build`, `test`, `test:native`, `api:check`, `test:asan`.
- Remaining: datetime64/timedelta64 (P14-8..10, D-174/D-175; this needs a core DType enum change touching ~38 dispatch sites), str_/bytes_/np.strings/structured (P14-11), and NumPy differential cases and benchmarks (names are in `api/bench-exempt/p14.json`).
