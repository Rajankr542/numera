# COMPATIBILITY

Reference: NumPy 2.5.3. "Verified" means the behaviour is covered by the NumPy
differential tests (`pnpm test:diff`).

## Supported (verified)
| Feature | Notes |
|---------|-------|
| `np.array` dtype inference | bool / int64 / float64 rules, see D-004 and divergences below |
| `np.array(data, {dtype})` for all 12 real dtypes | out-of-range ints raise `ValueError` (NumPy `OverflowError`) |
| `np.zeros` / `np.empty` | shape, strides (incl. zero strides for empty arrays), flags |
| `astype` | unsafe casting; float→int only verified for in-range values |
| `reshape` | view when layout allows, else copy; `-1` inference |
| strided views | `asStrided`; shape, strides, flags, values, `copy()` strides |
| `mayShareMemory` | NumPy `may_share_memory` bounds semantics |
| `promoteTypes` | full 14×14 table incl. complex |
| `ones` / `full` / `eye` | all real dtypes (`ones` shape/strides also verified for complex); `full` dtype inference |
| `arange` | int/float args, negative steps, empty ranges; int8/int32/float16/float32/float64 targets; bool ≤ 2 elements |
| `linspace` | `endpoint`, `num` 0/1, integer targets (floor, NumPy ≥ 2) |
| `transpose` / `squeeze` / `expandDims` / `swapAxes` / `moveAxis` / `ravel` / `flatten` | shape, strides, flags, values, view-vs-copy, error class; on contiguous and transposed inputs |
| `add` `subtract` `multiply` `divide` `power` `mod` `floorDivide` | all 12 real dtypes plus mixed pairs; result dtype, NaN/inf/-0, integer wrap, `x//0`, `x%0`; `power` on floats within rtol (D-014) |
| `abs` `negative` `sqrt` `exp` `log` | all 12 real dtypes; `sqrt`/`exp`/`log` within rtol (float16 1e-3, float32 1e-6, float64 1e-14) |
| broadcasting | `broadcastShapes`, `broadcastTo`, broadcast results incl. 0-d and zero-size; NEP 50 number scalars |
| indexing (`get`/`slice`/`set`) | integer, negative, slice (incl. reverse/clamped/empty), ellipsis, newaxis, integer-array, boolean mask, 0-d bool, mixed advanced; result dtype/shape/values, view-vs-copy, basic-view strides, error class; setitem on int32/float64/uint8/bool with broadcast, cast and repeated indices (D-015) |
| `flags.writeable` | `broadcastTo` views read-only; views inherit (basic index, transpose, expandDims, squeeze, swapAxes, moveAxis, view reshape, asStrided); copies writeable; writing a read-only view raises `ValueError` "assignment destination is read-only" (D-016) |
| reductions (`sum`/`prod`/`min`/`max`/`mean`/`var`/`std`/`argmin`/`argmax`) | 12 real dtypes × axis none/0/1/-1/tuple/keepdims on (2,3,4); transposed inputs; NaN, signed zero, ties, empty arrays, 0-d, `initial`, `ddof`, `dtype`, bad/repeated axis. Exact dtype/shape/strides; exact values except float sum/prod/mean/var/std (relative tolerance) (D-017) |
| complex reductions (same 9 functions) | 254 cases: complex64/complex128 × the same axis/keepdims set on (2,3,4); transposed inputs; lexicographic ties, NaN in either part, ±inf, signed zeros, empty arrays, 0-d, `initial`, `ddof`, bad axis, real input with a complex `dtype=`; 1000-element random inputs on trailing (pairwise) and leading (sequential) axes. Exact except complex `prod` (rtol 1e-5 / 1e-12) and the small-case `var`/`std` (real float results, same tolerance as real); `var`/`std` are exact on the large random inputs (D-034) |
| `matmul`/`dot`/`inner`/`outer` | float64/float32/int32/int8/uint16/bool (plus mixed int/float, float16, int8 wraparound) × 1-D/2-D/batched/broadcast/empty/mismatched shapes. Exact values for integer and bool; relative tolerance for floats (D-018) |
| `linalg.det`/`inv`/`solve`/`eig`/`eigh`/`svd`/`qr`/`lstsq`/`norm` | float64/float32/int/bool inputs, batched stacks, singular and empty matrices, rectangular and rank-deficient matrices, NaN input to `eig`, float16 and non-square errors; every `norm` ord and axis/keepdims. dtype and shape exact. Values checked by tolerance (det, inv, solve, eigenvalues, S, lstsq, norm). Vectors checked by reconstruction and orthogonality: A·V=V·Λ, U·S·Vh=A, Q·R=A with upper-triangular R. Run on **both** Accelerate and fallback backends (D-018) |
| random: `defaultRng` Generator (`random`, `uniform`, `standardNormal`, `normal`, `integers`, `choice`, `shuffle`, `permutation`) and legacy `seed`/`RandomState` (`rand`, `randn`, `random`/`randomSample`, `standardNormal`, `normal`, `uniform`, `randint`, `choice`, `shuffle`, `permutation`) | **Bit-exact** streams (exact equality) for int, bigint and array seeds, including chained calls that continue one stream; float32; int8/int16/uint8/uint32/int64/uint64/bool `integers`/`randint`; `endpoint`; choice with and without replacement (Floyd, partial-shuffle and `shuffle=False` paths); array populations; 1-D and 2-D shuffle, including Generator `axis=1`; bound/size/scale errors (D-019) |
| `fft.fft`/`ifft`/`rfft`/`irfft`/`fft2`/`ifft2`/`fftn`/`ifftn`/`fftfreq`/`rfftfreq` | 697 cases: float64/float32/float16/int32/uint8/bool/complex128/complex64 inputs; lengths 1, odd, even, prime and power of two; `n` pad/truncate/0/-1; every axis incl. out of range; all three `norm` values plus a bad one, including float16 half-precision factors; zero-size and 0-d inputs; `s`/`axes` combinations (-1 entries, repeated axes, empty `axes`, `s` without `axes`, length mismatch). dtype and shape exact; values within 1e-12 (float64), 2e-6 (float32) and 2e-3 (float16) relative to max\|x\| (D-020) |

## Documented divergences
| Behaviour | NumPy | nativpy | Decision |
|-----------|-------|---------|----------|
| `np.array([2**60])` | int64 | float64 (JS number is not safe int) | D-004 |
| `np.array([-0])` | int64 (Python int 0) | float64 (`-0` kept) | D-004 |
| bigint > int64 max without dtype | uint64 | `ValueError` | D-004 |
| `as_strided` out of bounds | reads arbitrary memory | `ValueError` | D-010 |
| out-of-range int input | `OverflowError` | `ValueError` | D-009 |
| float→int cast out of range / NaN | platform-dependent | platform-dependent (not tested) | D-009 |
| int64/uint64 `toArray()` | exact Python ints | JS numbers, exact only to 2^53; use `toTypedArray()` | D-005 |
| float16 `toTypedArray()` | — | raw bits as `Uint16Array` | D-005 |
| `full(shape, -1.5, {dtype: uint*})` | unchecked cast | `ValueError` | D-009/D-012 |
| invalid axis | `AxisError` | `IndexError` (repeated axis: `ValueError`) | D-012 |
| incompatible broadcast shapes | `ValueError` | `BroadcastError` | D-014 |
| ufunc result layout for F-ordered inputs | F order (`order='K'`) | always C-contiguous | D-014 |
| `subtract`/`negative` on bool, unsupported loops | `TypeError` | `DTypeError` | D-014 |
| number scalar out of the array dtype's range | `OverflowError` | `ValueError` | D-009/D-014 |
| `sqrt`/`exp`/`log`/float `power` | NumPy SIMD kernels | platform libm (may differ by a few ULP) | D-014 |
| integer `//0`, `%0` | 0 + RuntimeWarning | 0, no warning | D-014 |
| `a[1, 2]` (full integer index) | NumPy scalar | 0-d `NDArray` copy via `get`; `item()` for a JS scalar | D-015 |
| advanced-index result strides | may be non-C (internal transposes) | always C-contiguous copy (same shape/values) | D-015 |
| setitem value broadcast mismatch | `ValueError` | `BroadcastError` | D-015 |
| nested JS list as an index array | list → array index | not accepted; wrap in `np.array` | D-015 |
| `where(c, x, y)` with JS number scalars | weak (NEP 50) | inferred int64/float64 arrays | D-015 |
| `astype` on non-C-contiguous input | keeps layout (`order='K'`) | always C-contiguous (same values) | D-011 |
| `setflags(write=...)` | supported | not implemented; flag is read-only from JS | D-016 |
| float `sum`/`prod`/`mean`/`var`/`std` | pairwise summation | Bit-exact with NumPy for C-contiguous float32/float64 input, with no cast and no `initial`, when the reduced axes are all trailing (pairwise) or all leading (sequential). Verified on AArch64. Other cases may differ by a few ULP: float16, dtype casts, `initial`, mixed/middle axes and non-contiguous input. `prod` is sequential on both sides. | D-017, D-021 |
| complex `sum`/`mean`/`var`/`std` | pairwise summation | Same exactness rules as float (D-021). `var`/`std` return float32/float64. Complex `prod` may differ by 1 ulp: NumPy's arm64 complex64 loop fuses one multiply (FMA) | D-034 |
| complex `min`/`max`/`argmin`/`argmax` | lexicographic | Same: real part, then imaginary part; a NaN in either part propagates and the first NaN wins | D-034 |
| complex `var`/`std` with a different `dtype=` | dtype-specific casts | `NotImplementedError` | D-034 |
| reduction result layout for non-C inputs | keeps input order | always C-contiguous (same values) | D-017 |
| full reduction result | NumPy scalar | 0-d `NDArray`; `item()` for a JS scalar | D-017 |
| empty `mean`/`var`/`std` | NaN + RuntimeWarning | NaN, no warning | D-017 |
| `var` export name | `np.var` | `np.var` on default export; named export `variance` | D-017 |
| matmul/dot/inner/solve core-dimension mismatch | `ValueError` | `ShapeError` (batch mismatch: `BroadcastError`) | D-018 |
| linalg on float16 | `TypeError` | `DTypeError` | D-018 |
| eigenvector / singular-vector signs and phases | LAPACK (OpenBLAS) | backend-dependent (Accelerate or fallback); same subspaces | D-018 |
| decomposition result | named tuple | plain object (`{eigenvalues, eigenvectors}`, `{U, S, Vh}`, `{Q, R}`, `{x, residuals, rank, s}`) | D-018 |
| `svd`/`qr` options | `full_matrices=`, `compute_uv=`, `mode=` | `{fullMatrices, computeUV}`, `qr(a, mode)` | D-018 |
| float matmul / decompositions | OpenBLAS | Accelerate or fallback loops; may differ by rounding | D-018 |
| random with no seed | OS entropy via SeedSequence | OS entropy (`crypto.getRandomValues`); the legacy global state is array-seeded with 624 entropy words (not reproducible either way) | D-019 |
| random distribution parameters | broadcast array `loc`/`scale`/`low`/`high` | scalars only; arrays raise `NotImplementedError` | D-019 |
| random scalar results | NumPy scalar | JS number/boolean (D-005 applies to int64 > 2^53) | D-019 |
| `rfft` on complex input | `TypeError` | `DTypeError` | D-020 |
| `fftfreq`/`rfftfreq` with `n == 0` or `d == 0` | `ZeroDivisionError` | `ValueError` | D-020 |
| `fftn` with `s` but no `axes` | `DeprecationWarning` | accepted silently (same result) | D-020 |
| FFT complex results | complex ndarray | complex64/complex128 NDArray; elements read as `np.Complex` | D-020, D-033 |
| complex scalars | Python `complex` | frozen `np.Complex {re, im}`; `{re, im}` objects accepted as input | D-033 |
| complex value into a real array | `TypeError` (int) / `ComplexWarning`, imag dropped (float) | `DTypeError` for every real dtype (`astype` still drops imag, like NumPy) | D-033 |
| `mod` / `floorDivide` on complex | `TypeError` | `DTypeError` | D-033 |
| complex `sqrt`/`exp`/`log`/`power`/`abs`/`angle` | platform libm / npymath | NumPy's npymath algorithms (`npy_csqrt`, `npy_clog`, `npy_cpow`, SIMD `cabsolute`); `exp`, `pow` and `atan2` come from the C++ library; libm results may differ by a few ULP | D-014, D-033 |
| `imag` of a real array | read-only zeros array | same (read-only zeros) | D-033 |

## Not implemented
- Reduction keywords `out=`, `where=`; `nansum`/`nanmean` etc.; `argmin`/`argmax` with axis tuples (NumPy doesn't support them either).
- Complex ufuncs beyond `add`, `subtract`, `multiply`, `divide`, `power`, `negative`, `abs`, `sqrt`, `exp`, `log`, `conjugate` and `angle`. There are no complex trig functions or comparison ufuncs yet (P1 step 2, D-033).
- Linalg: complex inputs (`NotImplementedError`); `pinv`, `matrix_rank`, `matrix_power`, `cholesky`, `slogdet`, `cond`, `tensordot`, `einsum`, `vdot`, `kron`; batched `lstsq`; `out=` parameters; `eigh(UPLO='U')` (only the lower triangle is used). The `@` operator is not available in JS; use `np.matmul`.
- Ufunc keywords `out=`, `where=`, `casting=`, `dtype=`, `order=`; `NDArray` operator methods.
- `take` `mode=`/`out=`; field (structured) indexing; `put`, `putmask`, `choose`, `compress`.
- Random: `choice(p=...)`, other distributions (`exponential`, `gamma`, `binomial`, `poisson`, ...), `permuted`, `bytes`, `spawn`, `get_state`/`set_state`, other bit generators (Philox, SFC64), and `float32` `normal` with non-default loc/scale (NumPy has no such API either).
- FFT: `rfftn`/`irfftn`/`rfft2`/`irfft2`, `hfft`/`ihfft`, `fftshift`/`ifftshift`, `out=`; `fftfreq` `device=`.
- Everything from PLAN M11 onward (see ROADMAP.md). Not yet supported: `order='F'`, `arange`/`linspace` with complex arguments, `linspace` `retstep`/`axis`.
