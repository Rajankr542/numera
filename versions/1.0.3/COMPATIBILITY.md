# COMPATIBILITY

Reference: NumPy 2.5.3. "Verified" means the behaviour is covered by the NumPy
differential tests (the project's NumPy differential test suite).

## Supported (verified)
| Feature | Notes |
|---------|-------|
| `np.array` dtype inference | bool / int64 / float64 rules; see divergences below |
| `np.array(data, {dtype})` for all 12 real dtypes | out-of-range ints raise `ValueError` (NumPy `OverflowError`) |
| `np.zeros` / `np.empty` | shape, strides (incl. zero strides for empty arrays), flags |
| `astype` | unsafe casting; float→int only verified for in-range values |
| `reshape` | view when layout allows, else copy; `-1` inference |
| strided views | `asStrided`; shape, strides, flags, values, `copy()` strides |
| `mayShareMemory` | NumPy `may_share_memory` bounds semantics |
| `promoteTypes` | full 14×14 table incl. complex |
| `canCast` | all 5 casting rules × 14×14 dtype pairs (980 cases), default `"safe"`, array arguments (dtype only, no value-based casting), unknown rule → `ValueError` |
| `ones` / `full` / `eye` | all real dtypes (`ones` shape/strides also verified for complex); `full` dtype inference |
| `arange` | int/float args, negative steps, empty ranges; int8/int32/float16/float32/float64 targets; bool ≤ 2 elements |
| `linspace` | `endpoint`, `num` 0/1, integer targets (floor, NumPy ≥ 2) |
| `transpose` / `squeeze` / `expandDims` / `swapAxes` / `moveAxis` / `ravel` / `flatten` | shape, strides, flags, values, view-vs-copy, error class; on contiguous and transposed inputs |
| `add` `subtract` `multiply` `divide` `power` `mod` `floorDivide` | all 12 real dtypes plus mixed pairs; result dtype, NaN/inf/-0, integer wrap, `x//0`, `x%0`; `power` on floats within rtol |
| `abs` `negative` `sqrt` `exp` `log` | all 12 real dtypes; `sqrt`/`exp`/`log` within rtol (float16 1e-3, float32 1e-6, float64 1e-14) |
| broadcasting | `broadcastShapes`, `broadcastTo`, broadcast results incl. 0-d and zero-size; NEP 50 number scalars |
| indexing (`get`/`slice`/`set`) | integer, negative, slice (incl. reverse/clamped/empty), ellipsis, newaxis, integer-array, boolean mask, 0-d bool, mixed advanced; result dtype/shape/values, view-vs-copy, basic-view strides, error class; setitem on int32/float64/uint8/bool with broadcast, cast and repeated indices |
| `flags.writeable` | `broadcastTo` views read-only; views inherit (basic index, transpose, expandDims, squeeze, swapAxes, moveAxis, view reshape, asStrided); copies writeable; writing a read-only view raises `ValueError` "assignment destination is read-only" |
| reductions (`sum`/`prod`/`min`/`max`/`mean`/`var`/`std`/`argmin`/`argmax`) | 12 real dtypes × axis none/0/1/-1/tuple/keepdims on (2,3,4); transposed inputs; NaN, signed zero, ties, empty arrays, 0-d, `initial`, `ddof`, `dtype`, bad/repeated axis. Exact dtype/shape/strides; exact values except float sum/prod/mean/var/std (relative tolerance) |
| complex reductions (same 9 functions) | 254 cases: complex64/complex128 × the same axis/keepdims set on (2,3,4); transposed inputs; lexicographic ties, NaN in either part, ±inf, signed zeros, empty arrays, 0-d, `initial`, `ddof`, bad axis, real input with a complex `dtype=`; 1000-element random inputs on trailing (pairwise) and leading (sequential) axes. Exact except complex `prod` (rtol 1e-5 / 1e-12) and the small-case `var`/`std` (real float results, same tolerance as real); `var`/`std` are exact on the large random inputs |
| `matmul`/`dot`/`inner`/`outer` | float64/float32/int32/int8/uint16/bool (plus mixed int/float, float16, int8 wraparound) × 1-D/2-D/batched/broadcast/empty/mismatched shapes. Exact values for integer and bool; relative tolerance for floats |
| complex `matmul`/`dot`/`inner`/`outer` | 262 cases on **both** backends: complex64/complex128 × the same shapes plus every BLAS-dispatch shape, batched/broadcast stacks, N-D `dot`, empty operands, all 12 real dtypes mixed with either complex width, complex64·complex128, 0-d `dot`, inf/NaN, core/batch/0-d errors. Exact values (exactly representable inputs); large random gemm/gemv/dotu cases within 2e-5 / 1e-12 relative to max\|x\| |
| `linalg.det`/`inv`/`solve`/`eig`/`eigh`/`svd`/`qr`/`lstsq`/`norm` | float64/float32/int/bool inputs, batched stacks, singular and empty matrices, rectangular and rank-deficient matrices, NaN input to `eig`, float16 and non-square errors; every `norm` ord and axis/keepdims. dtype and shape exact. Values checked by tolerance (det, inv, solve, eigenvalues, S, lstsq, norm). Vectors checked by reconstruction and orthogonality: A·V=V·Λ, U·S·Vh=A, Q·R=A with upper-triangular R. Run on **both** Accelerate and fallback backends. Complex64/complex128 input to every function (`complex_linalg` group, 211 cases): square, batched, 1×1 and 0×0 matrices, Hermitian `eigh`/`eigvalsh`, rectangular and rank-deficient `svd`/`qr`/`lstsq`, every `norm` ord, mixed real/complex promotion, singular/NaN/non-square errors. Eigenvalues compared after sorting; vectors by reconstruction and unitarity (A·V=V·Λ, U·S·Vh=A, Q·R=A, Qᴴ·Q=I) and `qr(mode='r')` by \|R\|, since signs and phases depend on the backend |
| random: `defaultRng` Generator (`random`, `uniform`, `standardNormal`, `normal`, `integers`, `choice`, `shuffle`, `permutation`) and legacy `seed`/`RandomState` (`rand`, `randn`, `random`/`randomSample`, `standardNormal`, `normal`, `uniform`, `randint`, `choice`, `shuffle`, `permutation`) | **Bit-exact** streams (exact equality) for int, bigint and array seeds, including chained calls that continue one stream; float32; int8/int16/uint8/uint32/int64/uint64/bool `integers`/`randint`; `endpoint`; choice with and without replacement (Floyd, partial-shuffle and `shuffle=False` paths); array populations; 1-D and 2-D shuffle, including Generator `axis=1`; bound/size/scale errors |
| `fft.fft`/`ifft`/`rfft`/`irfft`/`fft2`/`ifft2`/`fftn`/`ifftn`/`fftfreq`/`rfftfreq` | 697 cases: float64/float32/float16/int32/uint8/bool/complex128/complex64 inputs; lengths 1, odd, even, prime and power of two; `n` pad/truncate/0/-1; every axis incl. out of range; all three `norm` values plus a bad one, including float16 half-precision factors; zero-size and 0-d inputs; `s`/`axes` combinations (-1 entries, repeated axes, empty `axes`, `s` without `axes`, length mismatch). dtype and shape exact; values within 1e-12 (float64), 2e-6 (float32) and 2e-3 (float16) relative to max\|x\|. Complex input from JS (`complex_fft` group, 120 cases): nested `np.complex` lists and plain `{ re, im }` objects (inferred, complex64, complex128), mixed number/bool/complex lists, transposed, reversed and step-2 views, `n`/`axis`/`norm`/`s`/`axes`, NaN input, 0-d and empty input; results read back through `toArray()` as `Complex` values |

## Documented divergences
| Behaviour | NumPy | numera |
|-----------|-------|---------|
| `np.array([2**60])` | int64 | float64 (JS number is not safe int) |
| `np.array([-0])` | int64 (Python int 0) | float64 (`-0` kept) |
| bigint > int64 max without dtype | uint64 | `ValueError` |
| `as_strided` out of bounds | reads arbitrary memory | `ValueError` |
| out-of-range int input | `OverflowError` | `ValueError` |
| float→int cast out of range / NaN | platform-dependent | platform-dependent (not tested) |
| int64/uint64 `toArray()` | exact Python ints | JS numbers, exact only to 2^53; use `toTypedArray()` |
| float16 `toTypedArray()` | — | raw bits as `Uint16Array` |
| `full(shape, -1.5, {dtype: uint*})` | unchecked cast | `ValueError` |
| invalid axis | `AxisError` | `IndexError` (repeated axis: `ValueError`) |
| incompatible broadcast shapes | `ValueError` | `BroadcastError` |
| ufunc result layout (strides) | `order='K'` default; `'C'`/`'F'`/`'A'` | same: NumPy's trivial-loop and `NpyIter` layout rules for every element-wise ufunc, incl. `conjugate`/`angle` (default 'K'), and `dot`/`inner` with a scalar |
| ufunc `out=` shape mismatch | `ValueError` | `BroadcastError` |
| ufunc `out=` cast not `same_kind`, non-array `out` | `UFuncTypeError` / `TypeError` | `DTypeError` |
| ufunc `out=(arr,)` tuple form | accepted | not accepted; pass `{ out: arr }` |
| ufunc `dtype=`/`casting=` input or output cast refused, no matching loop | `UFuncTypeError` / `TypeError` | `DTypeError` |
| ufunc `signature=` / `dtype=` as a tuple | accepted | not supported; `dtype` is a single dtype |
| ufunc `where=` without `out`: masked-out elements | uninitialized | zero |
| ufunc `where=None` | treated as `False` | rejected (`DTypeError`) |
| `subtract`/`negative` on bool, unsupported loops | `TypeError` | `DTypeError` |
| number scalar out of the array dtype's range | `OverflowError` | `ValueError` |
| `sqrt`/`exp`/`log`/float `power` | NumPy SIMD kernels | platform libm (may differ by a few ULP) |
| integer `//0`, `%0` | 0 + RuntimeWarning | 0, no warning |
| `a[1, 2]` (full integer index) | NumPy scalar | 0-d `NDArray` copy via `get`; `item()` for a JS scalar |
| advanced-index result strides | may be non-C (internal transposes) | always C-contiguous copy (same shape/values) |
| setitem value broadcast mismatch | `ValueError` | `BroadcastError` |
| nested JS list as an index array | list → array index | not accepted; wrap in `np.array` |
| `where(c, x, y)` with JS number scalars | weak (NEP 50) | inferred int64/float64 arrays |
| `astype` on non-C-contiguous input | keeps layout (`order='K'`) | always C-contiguous (same values) |
| `setflags(write=...)` | supported | not implemented; flag is read-only from JS |
| float `sum`/`prod`/`mean`/`var`/`std` | pairwise summation | Bit-exact with NumPy for C-contiguous float32/float64 input, with no cast and no `initial`, when the reduced axes are all trailing (pairwise) or all leading (sequential). Verified on AArch64. Other cases may differ by a few ULP: float16, dtype casts, `initial`, mixed/middle axes and non-contiguous input. `prod` is sequential on both sides. |
| complex `sum`/`mean`/`var`/`std` | pairwise summation | Same exactness rules as float. `var`/`std` return float32/float64. Complex `prod` may differ by 1 ulp: NumPy's arm64 complex64 loop fuses one multiply (FMA) |
| complex `matmul`/`dot`/`inner` | BLAS where NumPy's dispatch picks it, else NumPy's own loop | Accelerate backend: same routine choice as NumPy, bit-identical on arm64. Fallback backend: always the non-BLAS loop, so with inf/NaN input it matches NumPy's non-BLAS result (e.g. `inf+nanj`) where Accelerate's `gemv`/`gemm` give `nan+nanj`; finite results within tolerance |
| complex `det` | `zgetrf` in complex128, then cast | Same: complex64 input is computed in complex128 and rounded once; sign/log-magnitude formula of `umath_linalg`. Accelerate backend bit-identical on arm64 in random probes; fallback LU within tolerance |
| complex `inv`/`solve` | `zgesv` in complex128, then cast; complex64 result only if every operand is float32/complex64 | Same routine, dtype rule and single final rounding. Accelerate backend bit-identical on arm64 in random probes; fallback LU within tolerance |
| complex `qr` | `zgeqrf`/`zungqr` in complex128, then cast; R has a real diagonal | Same routines and single final rounding, all modes. Accelerate backend bit-identical on arm64 in random probes; fallback Householder (`zlarfg` convention) within tolerance |
| complex `svd` | `zgesdd` in complex128, then cast; S real (`_realType`); non-finite input raises before LAPACK | Same routine, dtypes and NaN check. Accelerate backend: S bit-identical in random probes on arm64, U/Vh bit-identical in 57/60 (others ≤6e-17 apart). Fallback (complex Jacobi) matches S within tolerance; vectors match up to phase, checked by reconstruction |
| complex `eigh`/`eigvalsh` | `zheevd` (UPLO='L') in complex128, then cast; eigenvalues real; `eigvalsh` uses JOBZ='N' | Same routine, JOBZ and dtypes; `eigvalsh` is now a native values-only call for real input too. Accelerate backend: values and vectors bit-identical in random probes on arm64 (complex128/complex64/float64). Fallback (complex Jacobi) within tolerance; vectors up to phase. Inf input: the fallback raises `LinAlgError` where LAPACK returns NaN |
| complex `eig`/`eigvals` | `zgeev` (JOBVL='N') in complex128, then cast; `eigvals` uses JOBVR='N' | Same routine, JOBVR and dtypes (complex64 in -> complex64 out); `eigvals` is now a native values-only call for real input too. Accelerate backend: complex128/complex64 values, `eigvals` and vectors bit-identical in a 15-case random probe each on arm64. Float64 eigenvectors match in 10/15; the other 5 are within 1 ulp, as before this change. Fallback (complex shifted QR) within tolerance; vectors up to phase |
| real float32 `det`/`inv`/`solve`/`eig`/`eigh`/`svd`/`qr`/`lstsq` | computed in float64 (`_commonType`), result cast to float32 | computed in float32, so results are not expected to be bit-identical. Measured only for `eigh` so far: max \|Δw\| 7.6e-6 in a 15-case probe. Open item |
| complex `min`/`max`/`argmin`/`argmax` | lexicographic | Same: real part, then imaginary part; a NaN in either part propagates and the first NaN wins |
| complex `var`/`std` with a different `dtype=` | dtype-specific casts | `NotImplementedError` |
| reduction result layout for non-C inputs | keeps input order | always C-contiguous (same values) |
| full reduction result | NumPy scalar | 0-d `NDArray`; `item()` for a JS scalar |
| empty `mean`/`var`/`std` | NaN + RuntimeWarning | NaN, no warning |
| `var` export name | `np.var` | `np.var` on default export; named export `variance` |
| matmul/dot/inner/solve core-dimension mismatch | `ValueError` | `ShapeError` (batch mismatch: `BroadcastError`) |
| linalg on float16 | `TypeError` | `DTypeError` |
| eigenvector / singular-vector signs and phases | LAPACK (OpenBLAS) | backend-dependent (Accelerate or fallback); same subspaces |
| decomposition result | named tuple | plain object (`{eigenvalues, eigenvectors}`, `{U, S, Vh}`, `{Q, R}`, `{x, residuals, rank, s}`) |
| `svd`/`qr` options | `full_matrices=`, `compute_uv=`, `mode=` | `{fullMatrices, computeUV}`, `qr(a, mode)` |
| float matmul / decompositions | OpenBLAS | Accelerate or fallback loops; may differ by rounding |
| random with no seed | OS entropy via SeedSequence | OS entropy (`node:crypto` `randomFillSync`); the legacy global state is array-seeded with 624 entropy words (not reproducible either way) |
| random distribution parameters | broadcast array `loc`/`scale`/`low`/`high` | scalars only; arrays raise `NotImplementedError` |
| random scalar results | NumPy scalar | JS number/boolean (the `toArray` 2^53 limit applies to int64 > 2^53) |
| `rfft` on complex input | `TypeError` | `DTypeError` |
| `fftfreq`/`rfftfreq` with `n == 0` or `d == 0` | `ZeroDivisionError` | `ValueError` |
| `fftn` with `s` but no `axes` | `DeprecationWarning` | accepted silently (same result) |
| FFT complex results | complex ndarray | complex64/complex128 NDArray; elements read as `np.Complex` |
| complex scalars | Python `complex` | frozen `np.Complex {re, im}`; `{re, im}` objects accepted as input |
| complex value into a real array | `TypeError` (int) / `ComplexWarning`, imag dropped (float) | `DTypeError` for every real dtype (`astype` still drops imag, like NumPy) |
| `mod` / `floorDivide` on complex | `TypeError` | `DTypeError` |
| complex `sqrt`/`exp`/`log`/`power`/`abs`/`angle` | platform libm / npymath | NumPy's npymath algorithms (`npy_csqrt`, `npy_clog`, `npy_cpow`, SIMD `cabsolute`); `exp`, `pow` and `atan2` come from the C++ library; libm results may differ by a few ULP |
| `imag` of a real array | read-only zeros array | same (read-only zeros) |
| `kind: "heapsort"` | heapsort | introsort (same result; heapsort only as the depth-limit fallback) |
| `kind: "mergesort"`/`"stable"` | timsort/radix sort | `std::stable_sort` (same stable order) |
| `NDArray.sort({axis: null})`, `lexsort` with `axis: null` | `TypeError` | `DTypeError` |
| sort/partition axis out of range | `AxisError` | `IndexError` |
| `unique` with multiple-return flags | tuple | object `{values, indices?, inverse?, counts?}` |
| `unique` of equal signed zeros (`[0, -0]`) with no flags | hash path; which zero is kept is unspecified | sort path; keeps the first zero in stable sorted order |
| `unique({sorted: false})` | hash-table order | sorted order |
| `uniqueAll`/`uniqueCounts`/`uniqueInverse` | named tuples (`inverse_indices`) | objects (`inverseIndices`) |
| `intersect1d({returnIndices: true})` | tuple | object `{values, indices1, indices2}` |
| `isin({kind: "table"})` | lookup table | sort + binary search (same result; errors identical) |
| `ediff1d` incompatible `toBegin`/`toEnd`, bool input | `TypeError` | `DTypeError` |
| P4 long-double / object loops | `g`, `G`, `O` loops | not available (no such dtypes) |
| P4 libm-based ufuncs (trig, exp/log, ...) | platform libm / SIMD | C++ `std::` libm; may differ by a few ULP |
| `divmod` / `modf` / `frexp` | ufunc objects (`where=`, `.reduce`, ...) | functions returning `[NDArray, NDArray]`; only `out`, `dtype`, `casting` |
| `np.clip` | dedicated `clip` ufunc | `minimum(maximum(a, min), max)`, `out` only |
| `nan_to_num` replacements | scalars or arrays | scalars only |
| `unwrap` on float16 | computed in float16 | computed in float32, cast back (last bit may differ) |
| P11 `slogdet`, `cond`, `matrix_rank`, `vdot`, `einsum` scalar results | NumPy scalars / `SlogdetResult` namedtuple | 0-d `NDArray`; `slogdet` returns `{ sign, logabsdet }` |
| P11 keyword arguments (`cholesky(upper)`, `pinv(rcond, hermitian, rtol)`, `matrix_rank(tol, hermitian, rtol)`, `tensorinv(ind)`, `tensorsolve(axes)`, `cross(axisa…)`, `vecdot(axis)`, `tensordot(axes)`, `einsum(optimize)`) | keywords | trailing options object |
| `cholesky`, `slogdet`, `pinv`, `cond` internal precision | LAPACK in the input precision (float32 stays float32) | compute in float64/complex128, then cast to the NumPy result dtype |
| `vecdot`/`matvec`/`vecmat` | gufuncs with `out=`, `axes=`, `dtype=` | plain functions (`vecdot` takes `axis` only) |
| `einsum` result | may be a view (`'ii->i'`, `'ij->ji'`); `out=`, `dtype=`, `order=`, `casting=` | always a new array; those keywords are not supported |
| `einsum` without `optimize` | single n-ary C loop (`c_einsum`) | pairwise left-to-right contractions over `matmul`; results equal up to float rounding |
| `einsumPath` report for `...` subscripts | ellipsis letters from Python set order ("may vary") | the highest unused letters (`z`, `y`, …) |
| `einsumPath` unknown path name | `KeyError` / `TypeError` | `ValueError` |
| `a.flat` | `flatiter` object, `a.flat[i]` | `FlatIter` with `get(i)`/`set(i, v)`, iterable; `a.flat = v` assigns cyclically |
| `a.flat[i]` for an integer | NumPy scalar | JS scalar |
| `a.tobytes()` | Python `bytes` | `Uint8Array` copy |
| `a.setflags(align=, uic=)` | supported | `NotImplementedError` (only `write`) |
| `a.fill(300)` on int8 | `OverflowError` | `ValueError` |
| `a.astype(dt, casting=)` disallowed | `TypeError` | `DTypeError` |
| `np.nditer` | full iterator (buffering, writable operands, context manager) | read-only `NDIter`, flags `multi_index`/`c_index`/`f_index`/`zerosize_ok`; others raise `NotImplementedError` |
| `np.ndenumerate` values | NumPy scalars | JS scalars |
| `np.finfo`/`np.iinfo` fields | NumPy scalars of the dtype; snake_case | JS numbers (camelCase); `iinfo` also has exact bigint `minExact`/`maxExact` |
| `np.minScalarType(1e5)` | `float32` (Python float) | `uint32`: JS cannot tell `1e5` from `100000`, so safe integers count as integers |
| `np.minScalarType` for integers beyond 64 bits | `object` | `ValueError` |
| `np.commonType` | returns a scalar type | returns a `DType` |
| `np.printoptions(...)` | context manager (`with`) | callback form `np.printoptions(opts, fn)`; options restored after `fn` |
| print option `legacy` | `'1.13'`, `'1.21'`, `'1.25'`, `'2.1'`, `'2.2'` or `False` | only `false`; others raise `NotImplementedError` |
| `formatter` callables in `array2string`/print options | Python callables; keys incl. `str_kind`, `numpystr`, `datetime`, `object` | JS callbacks for `all`, `bool`, `int`, `float`, `complexfloat`, `int_kind`, `float_kind`, `complex_kind` |
| `formatFloatPositional`/`formatFloatScientific` argument `TypeError`s | `TypeError` | `DTypeError` |
| `String(a)` / `a.toString()` | n/a (`repr(a)`) | NumPy `repr` text |
| `np.emath` with scalar input | NumPy scalar | 0-d array |
| `np.testing` messages for JS integer-valued numbers | `1.0` (Python float) | `1` (JS has one number type) |
| `np.testing.assertRaises` / `assertWarns` | context manager or callable | callable only; warnings are Node `process.emitWarning` warnings |
| `np.testing.assertStringEqual` diff | difflib with `?` hint lines | `-`/`+` lines only |
| `np.polynomial` coefficient dtype / dimensionality | any dtype incl. object; N-D `c` with `axis`/`tensor` | float64/complex128, 1-D only; no `*val2d/3d`, `*grid*`, `*vander2d/3d`, `*gauss`, `*weight` yet |
| `np.polynomial` operators and division by a zero series | Python operators; `ZeroDivisionError` | methods (`add`, `mul`, `floordiv`, `pow`, `call`, ...); `ValueError` |

## Not implemented
- Reduction keywords `out=`, `where=`; `nansum`/`nanmean` etc.; `argmin`/`argmax` with axis tuples (NumPy doesn't support them either).
- Complex ufuncs beyond `add`, `subtract`, `multiply`, `divide`, `power`, `negative`, `abs`, `sqrt`, `exp`, `log`, `conjugate` and `angle`. Trig/hyperbolic functions and comparison ufuncs are not implemented for any dtype yet; they will accept complex input when they land.
- Linalg (every implemented function accepts complex input): `pinv`, `matrix_rank`, `matrix_power`, `cholesky`, `slogdet`, `cond`, `tensordot`, `einsum`, `vdot`, `kron`; batched `lstsq`; `out=` parameters; `eigh(UPLO='U')` (only the lower triangle is used). The `@` operator is not available in JS; use `np.matmul`.
- `NDArray` operator methods; ufunc keywords on `conjugate`/`angle` (the other 12 element-wise ufuncs support `out=`, `where=`, `casting=`, `dtype=` and `order=`).
- `take` `out=`; field (structured) indexing.
- Random: `choice(p=...)`, other distributions (`exponential`, `gamma`, `binomial`, `poisson`, ...), `permuted`, `bytes`, `spawn`, `get_state`/`set_state`, other bit generators (Philox, SFC64), and `float32` `normal` with non-default loc/scale (NumPy has no such API either).
- FFT: `rfftn`/`irfftn`/`rfft2`/`irfft2`, `hfft`/`ihfft`, `fftshift`/`ifftshift`, `out=`; `fftfreq` `device=`.
- Not yet supported: `order='F'` for array creation/`reshape`/`astype` (ufuncs support it), `arange`/`linspace` with complex arguments, `linspace` `retstep`/`axis`.

## P8 indexing extras (build-first, not yet differential-verified)
Implemented natively: `take` `mode=` (raise/wrap/clip),
`takeAlongAxis`, `putAlongAxis`, `put` (mode), `putmask`, `place`, `choose`
(mode), `compress`, `extract`, `select`, `piecewise`, `argwhere`, `flatnonzero`,
`countNonzero` (axis, keepdims), `ravelMultiIndex` (mode, order),
`unravelIndex` (order), `diagonal` (read-only view), `trace` (dtype), and the
NDArray methods `choose compress diagonal nonzero put take trace`. This
supersedes the "`take` `mode=`; `put`, `putmask`, `choose`, `compress`" item
under "Not implemented" (`take` `out=` is still missing). NumPy differential
cases come in the V phase.

| Feature | NumPy | numera |
|---------|-------|--------|
| `countNonzero(a)` without axis, `ravelMultiIndex` of scalars | Python int / `np.int64` | 0-d int64 `NDArray` |
| `unravelIndex` | tuple of arrays | `NDArray[]` |
| `choose` with an out-of-range weak int scalar (e.g. 300 with `int8`) | wraps silently | `ValueError` |
| `putAlongAxis(..., axis=null)` on a non-contiguous array | raises (writes to a read-only copy) | writes through in flat C order |
| `piecewise` callbacks | Python callables | JS callbacks on `x[cond]` |
| `nested_iters` | iterator objects | excluded (api/exclusions.json) |
## P7 creation and grids divergences
| Behaviour | NumPy | numera |
|---|---|---|
| `mgrid`/`ogrid`/`r_`/`c_`/`s_`/`index_exp` | index-trick objects (`np.mgrid[0:3, 0:1:5j]`) | functions taking `[start, stop, step]` tuples (complex step = point count) or slice strings (`"1:4"`, `"0:1:5j"`) |
| `r_`/`c_` matrix directives `"r"`/`"c"` | return `np.matrix` | `NotImplementedError` (no matrix class) |
| `meshgrid`/`ix_`/`indices(sparse)`/`ogrid` results | tuple (`ogrid`: list) | JS array of NDArrays |
| `logspace`/`geomspace` bounds | array-like `start`/`stop`/`base`, `axis=` | scalar (real or complex) `start`/`stop`/`base` only, no `axis` |
| `np.astype` on non-arrays | `TypeError` | `DTypeError` |
| `tril`/`triu` on 0-d input | `ValueError` | `DTypeError` |
| `frombuffer` result | read-only view sharing the buffer | owned, writeable copy; native byte order only |
| `fromstring` binary mode (`sep=""`) | removed (`ValueError`) | same `ValueError` |
| `fromstring` with `count` larger than the data | `DeprecationWarning`, short array | `ValueError: string is smaller than requested size` |
| `fromiter` without `dtype` | `TypeError` | `dtype` is a required positional argument |
## P5 — comparison, logic and bitwise

| Area | NumPy | numera |
|---|---|---|
| logical ufuncs with `casting=` | mixed dtypes accepted under `casting="no"` (inputs go through the bool loop) | casting is checked against the promoted dtype, so mixed inputs under `"no"` raise `DTypeError` |
| `uint64` vs `int64` comparisons | exact comparison | compared via `float64` (can differ for values above 2^53) |
| `dtype=` on comparison/logical/classification ufuncs | output signature, only `bool` | same; other dtypes raise `DTypeError` |
| `isnat` | works for datetime64/timedelta64 | always raises `DTypeError` (no datetime dtypes yet) |
| `isscalar` | true for Python and NumPy scalars | true for JS number/boolean/bigint/string and complex scalars; false for every `NDArray`, including 0-d |
| bitwise ufuncs on float / `uint64`+`int64` mixes | `TypeError` | `DTypeError` |
| `isclose` with `float16` | each step rounded to float16 | evaluated in float32 (can differ only right at the tolerance boundary) |
| `isclose` `rtol`/`atol` | scalars or arrays | JS numbers only |
| `allclose` / `arrayEqual` / `arrayEquiv` result | Python `bool` | JS `boolean` |
| `bitorder=` | prefix match (`"l..."`, `"b..."`) | only `"big"` / `"little"`; anything else raises `ValueError` |
| `packbits`/`unpackbits` bad axis | `AxisError` | `IndexError` |
- P9: `unique({sorted: false})` unsorted order, `isin` table algorithm, benchmarks and NumPy differential cases for sorting/set functions.
## P6 array manipulation divergences (build-first, unverified by differential tests)

| Behaviour | NumPy | numera |
|---|---|---|
| `np.delete` name | `delete` | named export `delete` (implemented as `del`, a reserved word in JS) |
| slice arguments to `insert`/`delete` | `slice(a, b, c)` | `{start, stop, step}` object |
| `pad` option names | `constant_values`, `end_values`, `stat_length`, `reflect_type` | `constantValues`, `endValues`, `statLength`, `reflectType` |
| `pad` `linear_ramp`/`mean`/`median` precision | computed in the array's float dtype | computed in float64 (complex128), cast once |
| `pad` mode `empty` | uninitialised padding | zero padding |
| non-integer `pad_width` / `repeats` | `TypeError` | `DTypeError` |
| `ndarray.resize` refcheck | Python refcount | count of live native arrays sharing the buffer (views not yet garbage-collected count) |
| `delete` with an index array | advanced-indexing result layout | F if input F- and not C-contiguous, else C |
| `copyto` casting / overflow errors | `TypeError` / `OverflowError` | `DTypeError` / `ValueError` |
| `require` unknown flag | `KeyError` | `ValueError` |
| `broadcast_arrays` writeable views | writeable with `FutureWarning` | writeable, no warning |
| `asanyarray` | keeps subclasses | same as `asarray` (no subclasses) |

## P12 — FFT completion (build-first, not yet differential-verified)
New: `np.fft.hfft ihfft rfftn irfftn rfft2 irfft2 fftshift ifftshift`, and `out`
for every transform. The compute precision now follows NumPy's ufunc loop
selection (for example, float32 real input to `fft` runs the float64 loop). Ad-hoc
checks against NumPy 2.5.3 were bit-exact; the committed differential cases are
still to be written.

| Behaviour | NumPy | numera |
|-----------|-------|---------|
| FFT `out=` cast not `same_kind` | `UFuncTypeError` | `DTypeError` |
| FFT `out=` other dims not broadcastable | `ValueError` | `BroadcastError` |
| `out=` given as a non-array | `TypeError` | `DTypeError` |
| `rfftn`/`irfftn` with `axes=[]` | `IndexError` (list index out of range) | `IndexError` |
| `ihfft`/`rfftn` on complex input | `TypeError` | `DTypeError` |
| `fftshift`/`ifftshift` of a 0-d array | `ValueError` from `np.roll` | `ValueError` |
| `s` without `axes` (`rfftn`, `irfftn`) | `DeprecationWarning` | accepted, no warning |

## P11 linear algebra completion
Implemented on both linalg backends: `linalg.cholesky` (upper),
`slogdet`, `svdvals`, `matrixPower`, `pinv`, `matrixRank`, `cond`, `vectorNorm`,
`matrixNorm`, `matrixTranspose`, `diagonal`, `trace`, `outer`, `tensorinv`,
`tensorsolve`, `cross`, `tensordot`, `multiDot`, `vecdot`; `np.vdot`, `kron`,
`cross`, `tensordot`, `vecdot`, `matvec`, `vecmat`, `einsum`, `einsumPath`;
NDArray `dot`. This supersedes the "`pinv`, `matrix_rank`, `matrix_power`,
`cholesky`, `slogdet`, `cond`, `tensordot`, `einsum`, `vdot`, `kron`" part of
the Linalg item under "Not implemented". Batched `lstsq` (NumPy 2.5.3 rejects
stacks too), `out=` and `eigh(UPLO='U')` are still missing. Divergences are
listed in "Documented divergences".
| `load` of an empty file | `EOFError` | `ValueError` |
| `savezCompressed` output bytes | zlib deflate | Node zlib deflate (content equal, bytes may differ) |
| `allow_pickle` / `mmap_mode` in `load` | object arrays via pickle, memory maps | not supported (`ValueError`) |
| `savetxt` `%x` on bool | `TypeError` | accepted (0/1) |
| `fromfile` with `offset` past the end | `ValueError` (negative dimensions) | empty array |
| `fromregex` result | structured array | `{name: NDArray}` object |
| `genfromtxt` `dtype=None`, `names`, `converters`, `usemask` | supported | not supported |
| `base_repr` of a float | truncates | `TypeError` |
| `kaiser` window | NumPy i0 | same Chebyshev coefficients, may differ by 1 ulp (libm) |
| poly1d operators | `p(x) p+q p*q p/q p**n p[k]` | methods `call add mul div pow get/set` |
| `polyfit` rank warning | `RankWarning` class | Node warning named `RankWarning` |

## P16-D datetime64/timedelta64/busday
| Feature | NumPy | numera |
|---------|-------|--------|
| datetime64/timedelta64 dtype | C-level DType in the dtype enum, accepted by ufuncs | TS `DatetimeArray`/`TimedeltaArray` subclass wrapping int64 NDArray + unit string; rejected by all ufunc loops |
| Arithmetic on datetime arrays | `a + b`, `a - b` operator overloads | Not yet implemented (arithmetic must be done via explicit offset conversions) |
| `np.array(["2023-01-15"], dtype="datetime64[D]")` | Creates datetime64 array | Not wired into `array()`; use `datetime64()` directly |
| `busday_offset` NaT input | Returns NaT | Returns NaT sentinel (same) |
| `datetime_as_string` timezone offset output | Can display timezone offsets | All output is UTC (timezone parameter accepted but ignored) |
| `np.strings.encode(a, encoding)` | encodes str_ to bytes_ with given codec | identity stub — no bytes_ DType in JS; returns input unchanged |
| `np.strings.decode(a, encoding)` | decodes bytes_ to str_ with given codec | identity stub — returns input unchanged |
| `np.strings.mod(a, values)` | full Python `%`-format (all conversion types) | subset only: `%s %d %i %o %u %x %X %e %E %f %F %g %G %%` |
| `np.strings.translate(a, table)` | `table` is dict keyed by ordinal (`int`) | `table` is `Map<string, string\|null>` (char→char/null) — ASCII-compatible only |
| `np.rec.recarray` memory model | C-contiguous structured buffer, single allocation | JS object with separate named NDArray columns |
| `np.rec.recarray` field access | `r.x` returns a view into the struct buffer | `r.x` returns the column NDArray directly |
| `np.rec.fromfile` / `fromstring` | reads structured binary data from file/buffer | raises `NotImplementedError` (deferred) |
| `np.shares_memory` `maxWork` | may be approximate for high-dim overlapping views | always exact (native `sharesMemory`) |
| `np.ptp` | deprecated peak-to-peak function | excluded (removed in NumPy 2.x) |
| `np.asmatrix` / `np.bmat` / `np.matrix` | matrix class | excluded (c) |
## P16B: np.ma masked arrays
`np.ma` is now implemented in pure TypeScript. Known divergences:
| NumPy behaviour | numera behaviour |
|---|---|
| `ma.masked` is a 0-d `MaskedArray` with dtype float64 | `ma.masked` is a symbolic object sentinel |
| `ma.nomask` is `False` (Python bool) | `ma.nomask` is `false` (JS boolean) |
| `MaskedArray` inherits `ndarray` | `MaskedArray` wraps `NDArray` |
| `ma.fill_value` defaults vary by dtype | defaults are float64-based (`1e20`, `max float64`) |
| `ma.compress_nd(x, ndmin)` accepts `ndmin` | not supported (ignored) |
| `ma.convolve` mode `'same'` uses central portion | identical |
| `ma.corrcoef` rowvar parameter | `rowvar=True` default only |
| `ma.cov` full parameter set | `rowvar`, `bias`, `ddof` supported |
| Structured arrays / record arrays in ma | not supported |
| `ma.flatten_structured_array` | returns identity (not structured) |

## P10 statistics & NaN-reductions (build-first, not yet differential-verified)
Implemented natively: `median`, `percentile`, `quantile`,
`nanmedian`, `nanpercentile`, `nanquantile`, `cumsum`, `cumprod`,
`cumulativeSum`, `cumulativeProd`, `nancumsum`, `nancumprod`, `diff`, `ptp`,
`nansum`, `nanprod`, `nanmean`, `nanvar`, `nanstd`, `nanmin`, `nanmax`,
`nanargmin`, `nanargmax`, `average`, `cov`, `corrcoef`, `gradient`,
`trapezoid`, `histogram`, `histogramBinEdges`, `histogram2d`, `histogramdd`,
`bincount`, `digitize`, `interp`, `correlate`, `convolve`, and `where=`/`out=`
for `np.sum/prod/min/max/mean/var/std`. NumPy differential cases and benchmarks
come in the V phase.

| Feature | NumPy | numera |
|---------|-------|--------|
| `histogram` rightmost bin | half-open `[edge, +∞)` → closed on right | closed on both sides for the last bin; matches NumPy exactly |
| `bincount([])` | `zeros(0, int64)` | `zeros(0, int64)` (empty input is safe; no error) |
| `histogram` with identical range endpoints (`range=[v,v]`) | `RuntimeWarning`, all counts 0 | `ValueError` (zero-width range is never meaningful) |
| `convolve([], [])` | `array([], dtype=float64)` | `array([], dtype=float64)` — matches |
| `np.min/max` with `where=` and no `initial=` | `ValueError` | `ValueError` (same) |
| `where=` masked-out elements in reduction output | uninitialized | zero |
| `nanargmin`/`nanargmax` on all-NaN slice | `ValueError` | `ValueError` (same) |
| `quantile` `method='inverted_cdf'` with `weights=` | supported in NumPy ≥ 2.0 | supported |
