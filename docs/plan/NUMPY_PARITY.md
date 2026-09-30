# NumPy API parity plan (P-milestones)

Status: **Accepted** (D-032, 2026-09-30; scope defaults a–d adopted). Reference: NumPy 2.5.3.
P0 ✅ done. The live numbers come from `pnpm api:coverage`, which writes `api/coverage.json`.
This extends PLAN §86. It does not replace it. The P-milestones below come
**before** the rest of M11–M14 (optimization, compatibility, packaging, release),
so the feature set is complete first.

## 1. Measured gap (2026-09-30)

Found by diffing `dir()` of NumPy 2.5.3 against numera's exports
(`/tmp/np_api_inventory.py`, `/tmp/api_gap.py`; these become `scripts/api-coverage` in P0).

| Surface | NumPy | numera has | Missing |
|---|---:|---:|---:|
| `np.*` ufuncs | 106 | 13 | 93 |
| `np.*` functions | 283 | 41 | 242 |
| `np.linalg` | 31 | 13 | 18 |
| `np.fft` | 18 | 10 | 8 |
| `Generator` methods | 44 | 8 | 36 |
| `RandomState` methods | 49 | 12 | 37 |
| `ndarray` methods/attrs | ~70 | ~25 | 45 |
| `np.ma`, `np.polynomial`, `np.strings`/`np.char`, `np.rec`, `np.emath`, `np.testing` | ~420 | 0 | all |

numera covers only about 14% of the top-level `np.*` callables today.

## 2. Definition of Done per function (unchanged: PLAN §87 + the user's condition)

A function is done only when it has all of the following:
1. A C++ kernel in `native/` (TS only validates and orchestrates, per AGENTS.md).
2. A TS API in camelCase that mirrors NumPy (PLAN §76).
3. vitest unit tests and edge-case tests: empty, 0-d, NaN/Inf, every dtype, bad args.
4. NumPy differential cases in `python/generators/generate_cases.py`, replayed by `tests/differential`.
5. **A benchmark**: the same case name in `benchmarks/nativpy/suite.bench.mjs` **and**
   `benchmarks/numpy/suite_bench.py`. Pure metadata/validation functions (`isscalar`,
   `ndim`, `can_cast`, …) get one "tiny" case, so every function has a timing.
6. A docs entry in `docs/site/api.mjs` with an executed example, plus a COMPATIBILITY.md row.
7. ASan/UBSan clean. CI green.

## 3. Scope rules (need decisions)

| Item | Proposal |
|---|---|
| `object_` dtype, `frompyfunc`, `vectorize` | `vectorize`/`frompyfunc` wrap JS callbacks (slow path, documented). No `object_` dtype (D-032a). |
| `longdouble`/`clongdouble` | Alias float64/complex128 as on MSVC; documented divergence (D-032b). |
| `np.matrix`, `bmat`, `asmatrix`, `np.char.chararray`, `np.rec` | Deprecated in NumPy. Lowest priority (P15), or declared out of scope. |
| `np.testing` | Ship `assertAllclose`/`assertArrayEqual` family only. |
| Build/introspection (`show_config`, `get_include`, `f2py`, `ctypeslib`, `test`, `info`) | Out of scope. Excluded from the coverage denominator, and each exclusion is listed. |
| `save`/`load`/`savez` (`.npy`/`.npz`) | In scope: byte-compatible with NumPy files, verified both ways. |

## 4. Foundations (must land first; many functions are blocked on them)

**P0 — Coverage tooling.** Add `scripts/api-coverage.mjs` + `python/api_inventory.py`,
a checked-in `api-coverage.json` and exclusion list, and a CI step that fails if
coverage drops (PLAN §37). Also add a benchmark-coverage check: every exported
function must appear in both bench suites.

**P1 — Complex numbers (D-008).** JS representation (`{re, im}` or `np.complex(re, im)`),
`toArray`/`item`/`array()` for complex types, and complex support in every existing ufunc,
reduction, matmul, linalg and FFT path. Unblocks `angle`, `conj`, `real`, `imag`,
`iscomplex`, complex `eig`/`sort`.

**P2 — Ufunc machinery.** `out=`, `where=`, `dtype=`, `casting=` and `order=`, plus
`ufunc.reduce/accumulate/reduceat/outer/at` and `np.errstate`/`seterr`
(FP-exception flags). A table-driven native ufunc registry replaces the fixed
`BinaryOp`/`UnaryOp` enums in `native/core/ufunc.hpp`.

**P3 — Layout and core array API.** `order='F'`/`'K'` for creation, copy and ravel. NDArray methods:
`fill`, `view(dtype)`, `tobytes`, `tolist`, `byteswap`, `setflags`, `flat`/`nditer`/`ndindex`/`ndenumerate`,
`real`/`imag`/`T`/`mT`, `base`. Printing: `array2string`, `arrayRepr`, `setPrintoptions`,
`formatFloatPositional/Scientific`. Dtype introspection: `finfo`, `iinfo`, `canCast`,
`resultType`, `minScalarType`, `issubdtype`, `isdtype`, `commonType`, `mintypecode`.

## 5. Function milestones (every item follows the section 2 DoD)

**P4 — Ufuncs, math (≈60).** Trig/inverse/hyperbolic: `sin cos tan arcsin arccos arctan arctan2 sinh cosh tanh arcsinh arccosh arctanh hypot`
(plus the array-API aliases `asin acos atan atan2 asinh acosh atanh`), `deg2rad rad2deg degrees radians`, `unwrap`, `sinc`.
Exp/log: `exp2 expm1 log2 log10 log1p logaddexp logaddexp2 cbrt square reciprocal`.
Rounding: `floor ceil trunc rint round/around fix`.
Arithmetic: `remainder fmod divmod modf trueDivide floatPower pow positive sign heaviside`,
`maximum minimum fmax fmin clip`.
Float bits: `copysign nextafter spacing frexp ldexp signbit`.
Integer: `gcd lcm`.
Complex: `conj/conjugate angle real imag`.
Special: `i0 nanToNum realIfClose`.
libm results get per-function ULP tolerances, as in D-014.

**P5 — Comparison, logic, bitwise (≈35).** `equal notEqual less lessEqual greater greaterEqual`,
`logicalAnd/Or/Not/Xor`, `isnan isinf isfinite isnat isneginf isposinf`,
`isclose allclose arrayEqual arrayEquiv`, `all any` (with axis),
`iscomplex isreal iscomplexobj isrealobj isscalar`,
`bitwiseAnd/Or/Xor/Not/Invert/Count leftShift rightShift` (plus the `bitwise*` aliases), `packbits unpackbits`.

**P6 — Array manipulation (≈55).** Join: `concatenate/concat stack vstack hstack dstack columnStack block unstack`.
Split: `split arraySplit hsplit vsplit dsplit`.
Repeat and pad: `tile repeat pad`.
Edit: `append insert delete resize trimZeros`.
Reorder: `flip fliplr flipud roll rollaxis rot90 permuteDims matrixTranspose`.
Dimensions: `atleast1d/2d/3d broadcastArrays`.
Conversions: `ascontiguousarray asfortranarray asanyarray asarrayChkfinite require copyto`.
Metadata: `shape size ndim isfortran`.
Apply: `applyAlongAxis applyOverAxes`.

**P7 — Creation and grids (≈30).** `logspace geomspace meshgrid mgrid ogrid indices ix_`,
`diag diagflat tri tril triu vander fromfunction fromiter frombuffer fromstring`,
`trilIndices triuIndices(+From) diagIndices(+From) maskIndices fillDiagonal`,
`rC_ sC_ indexExp` (helper objects instead of `np.r_`), `copy`, `astype`.

**P8 — Indexing extras (≈20).** `takeAlongAxis putAlongAxis put putmask place choose compress extract select piecewise`,
`argwhere flatnonzero countNonzero ravelMultiIndex unravelIndex diagonal trace`,
`take` `mode=`, and `nestedIters`.

**P9 — Sorting, searching, sets (≈20).** `sort argsort` (quicksort, stable and heapsort kinds; NaN sorts last),
`partition argpartition lexsort searchsorted sortComplex`,
`unique` (returnIndex/Inverse/Counts, axis), `uniqueAll/Counts/Inverse/Values`,
`intersect1d union1d setdiff1d setxor1d isin ediff1d`.

**P10 — Statistics and NaN reductions (≈40).** `median percentile quantile` (all 13 `method=`s), `average ptp`,
`cumsum cumprod cumulativeSum cumulativeProd diff gradient`, `cov corrcoef`,
`histogram histogram2d histogramdd histogramBinEdges bincount digitize interp`,
`correlate convolve trapezoid`,
`nansum nanprod nanmean nanvar nanstd nanmin nanmax nanargmin nanargmax nanmedian nanpercentile nanquantile nancumsum nancumprod`,
and reduction `where=`/`out=`.

**P11 — Linear algebra completion (≈25).** `cholesky slogdet pinv matrixRank matrixPower cond`,
`multiDot tensordot tensorinv tensorsolve`, `svdvals vectorNorm matrixNorm`,
`linalg.cross/diagonal/trace/outer/vecdot/matrixTranspose`,
`np.vdot np.kron np.cross np.einsum np.einsumPath`,
the `matvec vecmat vecdot` ufuncs, batched `lstsq`, and complex versions of everything (after P1).

**P12 — FFT completion (8).** `rfft2 irfft2 rfftn irfftn hfft ihfft fftshift ifftshift`, and `out=`.

**P13 — Random completion (≈45).** Array (broadcast) parameters, `choice(p=)`, `size=`/`dtype=`/`out=`.
Distributions: `beta binomial chisquare dirichlet exponential f gamma geometric gumbel hypergeometric laplace
logistic lognormal logseries multinomial multivariateHypergeometric multivariateNormal negativeBinomial
noncentralChisquare noncentralF pareto poisson power rayleigh standardCauchy standardExponential
standardGamma standardT triangular vonmises wald weibull zipf`, plus `bytes permuted spawn`.
Bit generators: `MT19937 PCG64 PCG64DXSM Philox SFC64 SeedSequence`; RandomState `getState setState`.
Every stream is bit-exact with NumPy for both Generator and RandomState (the D-019 standard).

**P14 — Extra dtypes and I/O.**
- Datetime: `datetime64`/`timedelta64` with `busdayCount busdayOffset isBusday datetimeAsString datetimeData`.
- Strings: `str_`/`bytes_` dtypes and the `np.strings` module (46 functions).
- Structured/void: fields, `np.rec`.
- NPY files: `save load savez savezCompressed` (`.npy`/`.npz`, byte-compatible).
- Text files: `loadtxt savetxt genfromtxt fromfile tofile fromregex`.
- Misc: `baseRepr binaryRepr`; legacy polynomials `poly poly1d polyadd polyder polydiv polyfit polyint polymul polysub polyval roots`;
  window functions `bartlett blackman hamming hanning kaiser`.

**P15 — Submodules.** `np.polynomial` (6 series classes: Polynomial, Chebyshev, Legendre, Laguerre, Hermite, HermiteE),
`np.ma` masked arrays (≈210), `np.emath` (9), `np.testing` asserts, and `matrix` (if kept in scope).

## 6. Order and exit criteria

- The order is P0 → P1 → P2 → P3, because they are foundations. P4–P13 are mostly independent after that.
  P14–P15 come last. Only then do M11 (optimization), M12, M13 and M14 resume.
- A P-milestone is done when:
  - `scripts/api-coverage` shows every function in it as implemented;
  - every function has differential cases and a bench case on both sides;
  - `pnpm test`, `test:native`, `test:diff` and `test:asan` are green;
  - ROADMAP, PROGRESS, COMPATIBILITY and the docs site are updated.
- The PLAN §88 gate applies: at least 90% coverage of the *selected* surface, with every exclusion listed.

## 7. Size estimate (rough, for planning only)

- About 900 public names in total, or about 550 excluding `np.ma` and the `np.char`/`np.strings` duplicates.
- P1–P3 are the riskiest, because they change core internals (complex values, the ufunc registry, memory order).
- P4, P5 and P9 are mostly mechanical once P2's registry exists.
- P13 and P14 (bit-exact RNG, datetime, strings) cost the most per function.

