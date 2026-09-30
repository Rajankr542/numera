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
| `matmul`/`dot`/`inner`/`outer` | float64/float32/int32/int8/uint16/bool (plus mixed int/float, float16, int8 wraparound) × 1-D/2-D/batched/broadcast/empty/mismatched shapes. Exact values for integer and bool; relative tolerance for floats (D-018) |
| `linalg.det`/`inv`/`solve`/`eig`/`eigh`/`svd`/`qr`/`lstsq`/`norm` | float64/float32/int/bool inputs, batched stacks, singular and empty matrices, rectangular and rank-deficient matrices, NaN input to `eig`, float16 and non-square errors; every `norm` ord and axis/keepdims. dtype and shape exact. Values checked by tolerance (det, inv, solve, eigenvalues, S, lstsq, norm). Vectors checked by reconstruction and orthogonality: A·V=V·Λ, U·S·Vh=A, Q·R=A with upper-triangular R. Run on **both** Accelerate and fallback backends (D-018) |

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
| float `sum`/`prod`/`mean`/`var`/`std` | pairwise summation | sequential summation (may differ by a few ULP) | D-017 |
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

## Not implemented
- Reduction keywords `out=`, `where=`; `nansum`/`nanmean` etc.; complex reductions (`NotImplementedError`); `argmin`/`argmax` with axis tuples (NumPy doesn't support them either).
- Complex element read/write (`toArray`, `item`, `array([...], complex)`): `NotImplementedError` (D-008). Ufuncs on complex operands raise `NotImplementedError`.
- Linalg: complex inputs (`NotImplementedError`); `pinv`, `matrix_rank`, `matrix_power`, `cholesky`, `slogdet`, `cond`, `tensordot`, `einsum`, `vdot`, `kron`; batched `lstsq`; `out=` parameters; `eigh(UPLO='U')` (only the lower triangle is used). The `@` operator is not available in JS; use `np.matmul`.
- Ufunc keywords `out=`, `where=`, `casting=`, `dtype=`, `order=`; `NDArray` operator methods.
- `take` `mode=`/`out=`; field (structured) indexing; `put`, `putmask`, `choose`, `compress`.
- Everything from PLAN M8 onward (see ROADMAP.md). Not yet supported: `order='F'`, `arange`/`linspace` with complex arguments, `linspace` `retstep`/`axis`.
