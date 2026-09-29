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

## Not implemented
- Complex element read/write (`toArray`, `item`, `array([...], complex)`): `NotImplementedError` (D-008). Ufuncs on complex operands raise `NotImplementedError`.
- Ufunc keywords `out=`, `where=`, `casting=`, `dtype=`, `order=`; `NDArray` operator methods.
- Everything from PLAN M6 onward (see ROADMAP.md). Not yet supported: `order='F'`, `arange`/`linspace` with complex arguments, `linspace` `retstep`/`axis`.
