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

## P1 — Complex numbers (D-033)
Steps 1–4 are done (conversion, ufuncs, reductions, matmul family). Slices left:

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
| P1-5e | Rest of P1, combined into one slice (formerly P1-5e–5j): (1) ✅ `eigh`/`eigvalsh` (`?heevd`), with real eigenvalues (D-042); (2) ✅ `eig`/`eigvals` on complex input (`?geev`, D-043); (3) ✅ `lstsq` and `norm` (complex magnitudes, D-044); (4) ✅ a differential `complex_linalg` group, comparing results up to sign/phase where needed; (5) ✅ an FFT check that the complex paths accept `np.Complex` input end to end; (6) bench cases, ROADMAP/PROGRESS/COMPATIBILITY updates, and marking P1 ✅. Commit after each sub-step. |

Each 5x slice: C++ kernel + C++ test first, then binding/TS + vitest, all in
the same slice only if the diff stays small; otherwise split into 5x-1 / 5x-2.
P1-5e is larger than the ~300-line slice rule allows. Its sub-steps (1)–(6)
are done in order, and each one is green and committed before the next
starts.

## P2–P15
Slice each milestone just before it starts, using the same rules. As a rule of
thumb, one ufunc family, one function group, or one kwarg (`out=`, `where=`, …)
per slice.
