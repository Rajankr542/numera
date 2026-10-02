# Changelog

All notable changes to `@cyfora/numera`. NumPy coverage figures count the public NumPy 2.5.3 names that numera tracks. The [NumPy name index](/reference/numpy-index.html) of each version lists them.

## 1.0.3

Documentation-only release. The API is unchanged from 1.0.2.

- Rewritten README with the Cyfora logo, and a shorter README for npm.
- New `CONTRIBUTING.md`.

## 1.0.2

Coverage went from 15% to 99.1% of the tracked NumPy API (from 158 to 1045 of 1054 names).

### Universal functions
- All NumPy ufunc options: `out`, `where`, `dtype`, `casting` and `order`.
- Ufunc methods: `reduce`, `accumulate`, `reduceat`, `outer` and `at`.
- `seterr`, `geterr` and `errstate`, with real floating-point exception flags (`RuntimeWarning` or `FloatingPointError`).
- Trigonometric, hyperbolic, exponential/logarithmic, rounding, float-bit (`frexp`, `ldexp`, `nextafter`, …), `gcd`/`lcm`, `divmod`/`modf`, comparison, logical and bitwise ufuncs, and more. The new total is 106 ufunc names.

### Arrays
- Casting rules and `canCast`; `astype` with `casting`, `copy` and `order`.
- `finfo`, `iinfo`, `resultType`, `minScalarType`, `issubdtype`, `isdtype` and the abstract dtype hierarchy.
- NumPy-style printing: `toString()` is now `repr`; `array2string`, `arrayRepr`, `arrayStr` and print options.
- Memory order (`C`/`F`/`A`/`K`) for creation, copy, reshape and ravel.
- `NDArray` methods `fill`, `tolist`, `tobytes`, `view`, `byteswap`, `setflags`, `base`, `mT`, `flat`, `round`, `resize`, `repeat`, `dot`, `tofile`; `ndindex`, `ndenumerate` and `nditer`.

### Routines
- Array creation: `logspace`, `geomspace`, `tri*`, `diag*`, `vander`, `meshgrid`/`mgrid`/`ogrid`, `from*`, `r_`/`c_`/`s_`.
- Shape manipulation: `concatenate`/`stack`/`block`/`split` families, `tile`, `repeat`, `pad` (every mode), `flip`, `roll`, `rot90`, `append`/`insert`/`delete`, `applyAlongAxis` and more.
- Indexing routines: `take`/`put` (with `AlongAxis` variants), `choose`, `compress`, `select`, `piecewise`, `argwhere`, `ravelMultiIndex`/`unravelIndex`, …
- Sorting, searching, `unique` and set operations.
- Statistics: `quantile`, `percentile`, `median`, `average`, `cov`, `corrcoef`, `histogram*`, `bincount`, `correlate`, `convolve`, `gradient`, `trapezoid`, every `nan*` reduction, and `where=`/`out=` on reductions.
- Linear algebra: `cholesky`, `slogdet`, `pinv`, `matrixRank`, `cond`, `matrixPower`, `einsum`, `tensordot`, `kron`, `cross`, `vecdot`, `multiDot` and more, plus complex support across `np.linalg`.
- FFT: `out=` for every transform, `hfft`/`ihfft`, `rfft2`/`rfftn` and their inverses, `fftshift`/`ifftshift`.
- Random: all bit generators (`PCG64`, `PCG64DXSM`, `MT19937`, `Philox`, `SFC64`, `SeedSequence`), every `Generator` and `RandomState` distribution except `multivariate_normal`, and flat `np.random.*` functions.
- File I/O: `.npy`/`.npz` (`save`, `load`, `savez`, `savezCompressed`), `loadtxt`, `savetxt`, `genfromtxt`, `fromregex`, `fromfile`.
- New modules: `np.ma` (masked arrays), `np.strings` and `np.char`, `np.rec`, `np.emath`, `np.polynomial` (all six series classes), legacy `poly1d` and `np.poly*`, `np.testing`, and `datetime64`/`timedelta64` with the business-day functions.

### Fixes
- Linux/GCC 13 portability fixes in the native core.
- Node 18: OS entropy for random seeding now uses `node:crypto`.

## 1.0.1

- First public release under the `@cyfora/numera` name, with the API reference shipped inside the package.
- n-dimensional arrays with 14 numeric dtypes, NumPy dtype inference and promotion, views and strides.
- Element-wise arithmetic, comparisons, `sqrt`/`exp`/`log`/`abs` and other basic ufuncs with broadcasting.
- Reductions (`sum`, `prod`, `mean`, `var`, `std`, `min`, `max`, `argmin`, `argmax`) with `axis` and `keepdims`.
- Indexing with `get`, `slice` and `set`.
- `np.linalg` (`matmul`, `det`, `inv`, `solve`, `eig`, `eigh`, `svd`, `qr`, `lstsq`, `norm`), `np.fft` and `np.random` with NumPy-identical streams.
- Prebuilt binaries for macOS and Linux on x64 and arm64.
