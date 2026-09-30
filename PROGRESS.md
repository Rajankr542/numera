# PROGRESS

## 2026-09-30 — npm name `@cyfora/numera` (D-028)

Done:
- npm rejected the unscoped `numera` as well ("too similar"). The npm package
  is now **`@cyfora/numera`**, published under the `cyfora` npm org.
  GitHub Packages stays `@rajankr542/numera`. The workflow now strips the npm
  scope before adding the owner scope.
- `pnpm release` checks up front, before the build, that the logged-in npm user
  belongs to the `@cyfora` org. At the time of this change the `cyfora` scope did
  **not exist on npm** (registry: "Scope not found"), so it must be created at
  npmjs.com/org/create before the first publish.

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
