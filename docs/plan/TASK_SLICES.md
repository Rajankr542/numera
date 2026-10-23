# Task slices (small units of work)

Each parity milestone (P0–P15, `NUMPY_PARITY.md`) is split into slices small
enough to finish in one short agent session. Ask for **one slice at a time**,
e.g. "do P1-3a".

## Slice rules
- A slice changes at most 1–2 source files plus their tests (about 300 lines of diff).
- Large files are written in several small edits (≤ 6000 characters each).
- Every slice ends green on the checks that apply to it (`pnpm build`, `pnpm test`,
  `pnpm test:native`, `pnpm test:diff`, `pnpm test:asan`), then gets its own commit.
- Docs (ROADMAP/PROGRESS/COMPATIBILITY/DECISIONS) are a separate slice at
  the end of each step, not repeated in every slice.
- Start a new conversation after each slice or two.

## P1 — Complex numbers (D-033) ✅
All steps are done (conversion, ufuncs, reductions, matmul family, linalg, FFT input). P1 closed with P1-5e.6.

### Step 3: reductions (`native/core/reduce.cpp`)
| Slice | Work |
|---|---|
| P1-3a ✅ | `sum`, `prod` on complex64/128: remove the `reject_complex` calls on those paths and add complex accumulators. C++ test. |
| P1-3b ✅ | `mean` on complex. C++ test. |
| P1-3c ✅ | `min`/`max`/`argmin`/`argmax`, using lexicographic order (re, then im) with NaN propagation. C++ test. |
| P1-3d ✅ | `var`/`std` return the real dtype (mean of `\|x−mean\|²`), with `ddof`. C++ test. |
| P1-3e ✅ | TS/binding pass-through and vitest cases for 3a–3d. |
| P1-3f ✅ | Differential group `complex_reductions` (axis, keepdims, empty, nan/inf). |
| P1-3g ✅ | Bench cases in both suites, then ROADMAP/PROGRESS/COMPATIBILITY updates. Commit. |

### Step 4: matmul family
| Slice | Work |
|---|---|
| P1-4a ✅ | Complex `matmul` kernel (portable path) + C++ test (D-035). |
| P1-4b ✅ | Accelerate path (`cblas_cgemm`/`zgemm`, plus NumPy's gemv/dotu dispatch) + C++ test (D-036). |
| P1-4c ✅ | `dot`, `inner`, `outer` on complex (no conjugation, as in NumPy); vitest on both backends. |
| P1-4d ✅ | Differential `complex_matmul` cases (batched, mixed real/complex promotion), both backends (D-037). |
| P1-4e ✅ | Bench cases + docs. Commit. |

### Step 5: complex linalg (one function per slice, both backends)
| Slice | Work |
|---|---|
| P1-5a ✅ | `det`: fallback, then Accelerate (`cgetrf`/`zgetrf`) (D-038). |
| P1-5b ✅ | `inv`, `solve` (`?gesv`) (D-039). |
| P1-5c ✅ | `qr` (`?geqrf`/`?ungqr`) (D-040). |
| P1-5d ✅ | `svd` (`?gesdd`). Singular values are real (D-041). |
| P1-5e ✅ | Rest of P1, combined into one slice (formerly P1-5e–5j): (1) ✅ `eigh`/`eigvalsh` (`?heevd`), with real eigenvalues (D-042); (2) ✅ `eig`/`eigvals` on complex input (`?geev`, D-043); (3) ✅ `lstsq` and `norm` (complex magnitudes, D-044); (4) ✅ a differential `complex_linalg` group, comparing results up to sign/phase where needed; (5) ✅ an FFT check that the complex paths accept `np.Complex` input end to end; (6) ✅ bench cases, ROADMAP/PROGRESS/COMPATIBILITY updates, and marking P1 ✅. Commit after each sub-step. |

Each 5x slice: C++ kernel + C++ test first, then binding/TS + vitest, all in
the same slice only if the diff stays small; otherwise split into 5x-1 / 5x-2.
P1-5e is larger than the ~300-line slice rule allows. Its sub-steps (1)–(6)
are done in order, and each one is green and committed before the next
starts.

## P2 — Ufunc machinery
Scope (`NUMPY_PARITY.md`): `out=`, `where=`, `dtype=`, `casting=`, `order=`,
`ufunc.reduce/accumulate/reduceat/outer/at`, `errstate`/`seterr`, and a
table-driven native ufunc registry. The slices below apply to the 12 existing
ufuncs (`add` … `angle`). Later ufunc families (P4/P5) are written to the same
machinery.

| Slice | Content |
|-------|---------|
| P2-1 ✅ | Casting rules (D-045): native `Casting` + `can_cast` (`no`/`equiv`/`safe`/`same_kind`/`unsafe`), public `np.canCast` (moved forward from P3 because `out=`/`casting=` need it). C++ test, vitest, differential `casting` group, bench case. |
| P2-2 ✅ | Native `out=` (D-046) for `binary`/`unary`: result broadcast into `out` (`out` shape must equal the broadcast shape), `same_kind` output cast, read-only check, inputs copied first when they overlap `out` (except identical in-place). C++ tests. |
| P2-3 ✅ | TS `{ out }` option on all 12 ufuncs (returns `out` itself), vitest, and differential group `ufunc_out` (dtype casts, broadcast, strided/overlapping `out`, errors). |
| P2-4 ✅ | `casting=` and `dtype=` (loop-dtype override with input casting, D-048). C++ + TS + differential cases. |
| P2-5 ✅ | `where=` mask (with and without `out`; masked-out elements keep `out`, or are zero without `out`, D-049). C++ + TS + differential cases. |
| P2-6 ✅ | `order=` (`'C'`/`'F'`/`'A'`/`'K'`) for ufunc results, including NumPy's `'K'` stride order. Replaces the D-014 C-contiguous divergence (D-050). |
| P2-7 ✅ | Table-driven native ufunc registry replacing the `BinaryOp`/`UnaryOp` enums (name → loops per dtype, identity, type resolver). Pure refactor: every existing test stays green (D-051). |
| P2-8 ✅ | `ufunc.reduce` and `ufunc.accumulate` (`np.add.reduce(a, {axis, dtype, out, keepdims, initial, where})`), routed through the registry. **Build-first:** native implementation + binding + TS API, verified by C++ unit tests and vitest only. |
| P2-8v | Verification pass for P2-8: NumPy differential group (`ufunc_reduce`), bit-exactness fixes (pairwise float `add`, float16 accumulation), and bench cases. |
| P2-9 ✅ | `ufunc.outer`, `ufunc.reduceat`, `ufunc.at`. |
| P2-10 ✅ | `errstate`/`seterr`/`geterr` (`divide`/`over`/`under`/`invalid` → `ignore`/`warn`/`raise`) via native FP-exception flags plus the integer divide-by-zero path. |
| P2-11 | Bench cases for every new kwarg/method in both suites, ROADMAP/PROGRESS/COMPATIBILITY updates, and marking P2 ✅. |

## Build-first order (user direction, 2026-10-02)
Milestones P2-9 … P15 are implemented **build-first**: native code + binding +
TS API, verified by `pnpm build`, `pnpm test`, `pnpm test:native` and
`pnpm test:asan`. NumPy differential groups, bit-exactness fixes and bench
cases are deferred to one **V phase** after P15 (V-P2 … V-P15, including P2-8v
and P2-11). No NumPy-compatibility or performance claim is made for a slice
until its V slice is done. `api:check` must stay green, so new callables are
listed in the bench-exemption list until V adds their bench cases.

## P3 — Layout and core array API (build-first)
| Slice | Content |
|-------|---------|
| P3-1 ✅ | `order='C'/'F'` for creation (`empty/zeros/ones/full/*Like`, `array`), `copy({order})` incl. `'K'`/`'A'`, `ravel/flatten/reshape({order})`, `ascontiguousarray`, `asfortranarray`, `np.copy`. |
| P3-2 | NDArray methods: `fill`, `tolist`, `tobytes({order})`, `view(dtype)`, `byteswap`, `setflags({write})`, `base`, `mT`, `flat` (1-D iterator view), `astype({copy})`. |
| P3-3 | Iteration helpers: `ndindex`, `ndenumerate`, `nditer` (read-only, C/F/K order, multi-operand broadcast; no buffering/external loop). |
| P3-4 | Dtype introspection: `finfo`, `iinfo`, `resultType`, `minScalarType`, `issubdtype`, `isdtype`, `commonType`, `mintypecode`. |
| P3-5 | Printing: `array2string`, `arrayRepr`, `arrayStr`, `setPrintoptions`/`getPrintoptions`/`printoptions`, `formatFloatPositional`/`Scientific` (NumPy dragon4 shortest repr), and `toString()` using them. |

## P4–P15
Slice each milestone just before it starts, using the same rules. As a rule of
thumb, one ufunc family, one function group, or one kwarg (`out=`, `where=`, …)
per slice.

---

## V phase — verification, NumPy differential tests, and benchmarks

All P2-9…P16E builds are **unverified**: they pass `pnpm test` (vitest) but have
no NumPy comparison cases and no benchmark entries. The V phase adds both.
**No NumPy-compatibility or performance claim covers a slice until its V slice is
green.**

### V-phase rules
- Each V slice adds differential generator cases to
  `python/generators/generate_cases.py` **and** a `main()` entry in that file,
  then runs `pnpm test:diff` and fixes any mismatches found.
- Each V slice also adds matching cases to `benchmarks/nativpy/suite.bench.mjs`
  **and** `benchmarks/numpy/suite_bench.py`, then removes those names from the
  relevant `api/bench-exempt/pNN.json` file.
- A V slice may fix bugs found during comparison; every fix gets its own commit
  with a D-number recorded in `DECISIONS.md` (range D-242+).
- After both checks are green the slice gets a single commit
  `test(verify): <area> differential + bench (V-PNN, D-24x)`.
- `PROGRESS.md` is updated at the end of each V slice.

### V-phase order (highest-value first)

| Slice | Area | Bench-exempt entries to clear | D-numbers |
|-------|------|-------------------------------|-----------|
| V-P2  | Ufunc machinery: `reduce`/`accumulate`/`outer`/`reduceat`/`at`, `errstate`/`seterr`; bench for every ufunc kwarg added in P2 (P2-8v + P2-11) | bench-exempt entries in `p03.json` that cover P2 ufunc kwargs | D-242 |
| V-P3  | Core array API: `fill`, `tolist`, `tobytes`, `view`, `byteswap`, `ndindex`, `ndenumerate`, `nditer`, `finfo`/`iinfo`/`resultType`, printing; `order=` creation paths | `api/bench-exempt/p03.json` (20 entries) | D-243 |
| V-P4  | Math ufuncs: all 75 trig/exp/round/arithmetic/float-bit/special functions (sin, cos, exp2, floor, heaviside, sinc, …) | `api/bench-exempt/p04.json` (75 entries) | D-244 |
| V-P5  | Comparison, logic, bitwise: equal/less/logicalAnd/isnan/bitwiseAnd/isclose/packbits (38 entries) | `api/bench-exempt/p05.json` (38 entries) | D-245 |
| V-P6  | Array manipulation: concatenate/stack/split/pad/insert/roll/applyAlongAxis/… (46 entries) | `api/bench-exempt/p06.json` (46 entries) | D-246 |
| V-P7  | Creation/grids: logspace/geomspace/tri/meshgrid/indices/fromfunction/r\_/… (24 entries) | `api/bench-exempt/p07.json` (24 entries) | D-247 |
| V-P8  | Indexing extras: take/put/choose/compress/piecewise/ravelMultiIndex/diagonal/… (24 entries) | `api/bench-exempt/p08.json` (24 entries) | D-248 |
| V-P9  | Sorting/searching/sets: sort/argsort/partition/lexsort/unique/isin/… (23 entries) | `api/bench-exempt/p09.json` (23 entries) | D-249 |
| V-P10 | Statistics/NaN reductions: median/quantile/cumsum/nansum/average/histogram/correlate/… (39 entries) | `api/bench-exempt/p10.json` (39 entries) | D-250 |
| V-P11 | Linalg completion: cholesky/pinv/cond/vdot/kron/cross/tensordot/einsum/vecdot/… (29 entries) | `api/bench-exempt/p11.json` (29 entries) | D-251 |
| V-P12 | FFT completion: hfft/rfftn/rfft2/fftshift/out= (8 entries) | `api/bench-exempt/p12.json` (8 entries) | D-252 |
| V-P13 | Random completion: all distributions for Generator/RandomState/module-level (115 entries) | `api/bench-exempt/p13.json` (115 entries) | D-253 |
| V-P14 | I/O + extras: save/load/loadtxt/genfromtxt/poly/windows (28 entries) | `api/bench-exempt/p14.json` (28 entries) | D-254 |
| V-P15 | Submodules: emath/testing/polynomial/ma (23 entries) | `api/bench-exempt/p15.json` (23 entries) | D-255 |
| V-P16A | Constants, vectorize, cumsum/cumprod methods (7 entries) | `api/bench-exempt/p16a.json` (7 entries) | D-256 |
| V-P16B | np.ma full surface (206 entries) | `api/bench-exempt/p16b.json` (206 entries) | D-257 |
| V-P16C | np.strings + np.char (99 entries) | `api/bench-exempt/p16c.json` (99 entries) | D-258 |
| V-P16D | datetime64/timedelta64/busday (8 entries) | `api/bench-exempt/p16d.json` (8 entries) | D-259 |
| V-P16E | np.rec + cleanup (5 entries) | `api/bench-exempt/p16e.json` (5 entries) | D-260 |

**Total bench-exempt entries to clear: ~821 across all V slices.**

### After V phase
Once all V slices are green:
1. Run the full benchmark suite (`pnpm bench:suite && pnpm bench:suite:numpy && pnpm bench:compare`) and update `PERFORMANCE.md`.
2. Update `ROADMAP.md` milestones P3–P15 from ⬜ to ✅.
3. Drop any unneeded parallel milestone branches (`pNN`) with `git branch -d`.
4. Begin M11 performance work (per-call overhead, `zeros` regression, `max` NEON, Node 18/20 profile — see `ROADMAP.md` "Next steps").

### Remaining build work (not yet started)
These slices were deferred from the build-first pass and still need implementation:

| Slice | Work |
|-------|------|
| P3-2 | NDArray methods: `fill`, `tolist`, `tobytes({order})`, `view(dtype)`, `byteswap`, `setflags({write})`, `base`, `mT`, `flat`, `astype({copy})` |
| P3-3 | Iteration helpers: `ndindex`, `ndenumerate`, `nditer` (read-only) |
| P3-4 | Dtype introspection: `finfo`, `iinfo`, `resultType`, `minScalarType`, `issubdtype`, `isdtype`, `commonType`, `mintypecode` |
| P3-5 | Printing: `array2string`, `arrayRepr`, `arrayStr`, `setPrintoptions`/`getPrintoptions`/`printoptions`, `toString()` |
| P14-8 | datetime64/timedelta64 dtype in the C++ core enum; all `dispatch_dtype` sites raise `DTypeError` for new dtypes |
| P14-9 | datetime arithmetic/comparison, `datetimeAsString`, `datetimeData` |
| P14-10 | `busdayCount`, `busdayOffset`, `isBusday` |
| P14-11 | `str_`/`bytes_` dtypes, `np.strings`, structured/void, `np.rec` full C++ path |
| multivariate_normal | `Generator.multivariateNormal`, `RandomState.multivariateNormal`, `np.random.multivariateNormal` (uses existing linalg Cholesky/SVD) |
| ma gaps | `ma.applyAlongAxis`, `ma.applyOverAxes`, `ma.compressed`, `ma.frombuffer`, `ma.fromflex`, `ma.fromfunction` |
