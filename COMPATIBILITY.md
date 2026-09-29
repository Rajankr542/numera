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

## Not implemented
- Complex element read/write (`toArray`, `item`, `array([...], complex)`): `NotImplementedError` (D-008).
- Everything from PLAN M2 onward not listed above (see ROADMAP.md).
