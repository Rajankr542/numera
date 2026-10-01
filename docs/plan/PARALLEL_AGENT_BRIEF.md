# Brief for milestone agents (D-056)

You implement **one** NumPy-parity milestone `PNN` of the numera project
(NumPy semantics, C++20 native core, TypeScript API; Node-API addon).

## Where you work
- Your git worktree: `/Users/rajankumar/Desktop/Rajan/numera/.worktrees/pNN`, branch `pNN`.
- **Every** shell command must run with `workdir` set to your worktree, and every
  file path you read/edit/write must be an absolute path inside it. Never edit
  files in `/Users/rajankumar/Desktop/Rajan/numera/` itself (that is `main`) or in
  another milestone's worktree. Never run `git push`, `git merge`, `git rebase` or
  `git checkout <other branch>`.
- Before any build/test command: `source scripts/dev-env.sh` (pins the macOS SDK and
  the Python venv with cmake/ninja/numpy; caps build parallelism). Example:
  `source scripts/dev-env.sh && pnpm build`.
- Read first: `AGENTS.md` (incl. "Parallel milestones"), `docs/plan/NUMPY_PARITY.md`
  (your milestone's scope and the Definition of Done), `docs/plan/TASK_SLICES.md`
  (build-first rule), `DECISIONS.md` entries D-051 (ufunc registry), D-055, D-056.

## Build-first rule
Native C++ kernel + binding + TS API, verified by `pnpm build`, `pnpm test`,
`pnpm test:native`, and `pnpm api:check`. Run `pnpm test:asan` at least at the end
of the milestone (and after any memory-sensitive native change). NumPy
differential cases and bench cases are deferred to the later V phase, so do not
claim NumPy compatibility or performance. If NumPy behaviour is unclear, check it
with the venv's numpy (`python3 -c "import numpy as np; ..."`) and match it.

## Files you own (and only these)
- `docs/plan/slices/pNN.md`: your slice table and status. **Write it first** and
  keep it current; it is the hand-off state if your session is restarted.
- `native/bindings/pNN_binding.cpp` → JS `addon.pNN` (helpers in
  `native/bindings/binding_utils.hpp`: `fn`, `arr`, `arg_int(s)`, `wrap`, ...).
- New native sources: `native/core/pNN_<area>.{hpp,cpp}` (or `native/linalg/`,
  `native/random/`, `native/fft/` for those libraries). CMake globs these
  directories, so do not edit `CMakeLists.txt`. Namespace `nativpy` (as the existing code).
- New C++ tests: `tests/native/test_pNN_<area>.cpp` (harness: `test_harness.hpp`,
  `TEST_CASE`, `CHECK`, `CHECK_EQ`, `CHECK_THROWS_KIND`).
- TS: `packages/numera/src/pNN.ts` (export const `pNN = { ...public functions }`,
  which index.ts spreads into the default `np`; also add named `export`s of your
  functions/types from `pNN.ts` because index.ts does `export * from "./pNN.js"`).
  Split into `pNN_<area>.ts` modules if large and re-export from `pNN.ts`.
  Read native functions with `const native = nativeModule<YourInterface>("pNN")`
  from `./addon.js`.
- New NDArray methods: TS declaration merging in your files, e.g.
  `declare module "./ndarray.js" { interface NDArray { sort(opts?: SortOptions): NDArray } }`
  plus `NDArray.prototype.sort = function (...) {...}`. (P3 edits `ndarray.ts` directly.)
- New ufuncs: P4 adds rows to `native/core/ufunc_math.cpp`, P5 to
  `native/core/ufunc_logic.cpp`, using the templates in `native/core/ufunc_loops.hpp`
  (`binary_table`, `unary_table`, `same`, `float_for`, ...), and builds the TS
  objects with `binaryUfunc(name)` / `unaryUfunc(name)` from `ufunc.ts` so they get
  `out/where/dtype/casting/order` and `reduce/accumulate/outer/at`. Ufuncs whose
  output dtype differs from the input (comparisons → bool, etc.) may need new loop
  templates; add them to your own file, or minimally to `ufunc_loops.hpp`.
- Vitest: `packages/numera/test/pNN_<area>.test.ts`.
- Docs: `docs/site/parts/pNN.mjs` (`categories` with entries `{name, sig, desc,
  args?, returns, example}`; an entry joins an existing category when the `id`
  matches: creation, ndarray, shape, indexing, math, reduce, linalg, fft, random,
  dtype, utilities, errors). **Every public name added to `np` (and to
  `np.linalg`/`np.fft`/`np.random`, `Generator`/`RandomState`) must appear in the
  docs**, and every example line `expr; // => <json>` is executed by
  `packages/numera/test/docs_site.test.ts`. Keep examples short and exact.
- `api/bench-exempt/pNN.json`: a JSON array of the strings `pnpm api:check` reports
  as "no benchmark in both suites" for **your** names (e.g. `"np.sinh [nativpy] [numpy]"`).
  `pnpm api:check` / `pnpm api:coverage` rewrite `api/coverage.json`; **never commit
  it** (`git checkout api/coverage.json` before committing).
- `api/aliases.d/pNN.json`, `api/exclusions.d/pNN.json`: `{ "<surface>": { "numpy_name": "jsName" } }`
  only when a name does not match after ignoring case/underscores (e.g. NumPy `var` → `variance`).
- Append-only: `DECISIONS.md` (use **only** your reserved numbers, see below; record
  public API / compat decisions *before* implementing them), `PROGRESS.md` (one
  entry per finished milestone at the end of the file is fine), `COMPATIBILITY.md`.
- Files owned by milestones (edit only if you are that milestone):
  `ndarray.ts` P3, `ufunc.ts` P4 (P5 may only add exports), `shape.ts` P6,
  `creation.ts` P7, `indexing.ts` P8, `reduce.ts` P10, `linalg.ts` P11, `fft.ts` P12,
  `random.ts` P13. You may extend an owned existing module (e.g. P11 adds to the
  `linalg` object in `linalg.ts`, P12 to `fftModule`, P13 to `Generator`/`RandomState`/`random`).
- Do not edit: `index.ts`, `CMakeLists.txt`, `ROADMAP.md`, `TASK_SLICES.md`,
  `api/coverage.json`, `api/coverage-baseline.json`, other milestones' files. If a tiny
  change to a shared file is unavoidable (e.g. a missing export in `ufunc.ts`), keep it
  minimal and mention it in the commit message.
- Do not redefine a name that already exists in `np` from a different file; instead
  extend it in its owning module, or (if you do not own it) leave it and note it in
  your slice file.

## Reserved decision numbers
P3 D-060–069, P4 D-070–079, P5 D-080–089, P6 D-090–099, P7 D-100–109,
P8 D-110–119, P9 D-120–129, P10 D-130–139, P11 D-140–149, P12 D-150–159,
P13 D-160–169, P14 D-170–179, P15 D-180–189.

## NDArray methods by milestone
P3: `fill tolist tobytes view byteswap setflags base mT flat`, `astype({copy})`.
P4: `clip round conjugate`. P5: `all any`. P6: `repeat resize`.
P8: `choose compress diagonal nonzero put take trace`. P9: `sort argsort partition
argpartition searchsorted`. P10: `cumsum cumprod ptp`. P11: `dot`. P14: `tofile`.

## Dependencies on other milestones
Milestones run in parallel; other milestones' new functions are **not** available
on your branch. If you need one (e.g. `concatenate`, `sort`, `isnan`), write a
private native helper in your own files rather than waiting. Existing `main`
functionality (P0–P2, P3-1) is available.

## How to work
1. Read the brief, then plan slices (each about 300 lines of diff, one function
   family per slice) into `docs/plan/slices/pNN.md` and commit it.
2. Per slice: C++ kernel + C++ test → binding → TS API + vitest → docs entry →
   bench-exempt entries → `pnpm build && pnpm test && pnpm test:native && pnpm api:check`
   all green → `git add` your files (not `api/coverage.json`) → conventional commit
   `feat(<area>): ... (PNN-k, D-xxx)` → mark the slice ✅ in the slice file.
3. Code style: C++20, RAII, `-Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror`;
   do not leak C++ exceptions (use `throw_error(ErrorKind::..., msg)` and
   `translate_errors`); TS strict, camelCase NumPy names, options objects for kwargs.
   Every function: edge cases (empty, 0-d, NaN/Inf, each dtype, bad args) in tests.
4. Never fake an implementation, never weaken or skip tests, never claim something
   is done without running the checks.
5. If your context is getting long, finish and commit the current slice, update the
   slice file with exactly what is left, and stop. A fresh agent will continue from it.

## Final report (your last message, at most ~250 words)
Slices done (with commit hashes), what is left (if anything), checks run and their
results (build/test/test:native/api:check/test:asan), and any edits to shared files.
