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
Steps 1–2 are done (conversion, ufuncs). Slices left:

### Step 3: reductions (`native/core/reduce.cpp`)
| Slice | Work |
|---|---|
| P1-3a ✅ | `sum`, `prod` on complex64/128: remove the `reject_complex` calls on those paths and add complex accumulators. C++ test. |
| P1-3b ✅ | `mean` on complex. C++ test. |
| P1-3c | `min`/`max`/`argmin`/`argmax`, using lexicographic order (re, then im) with NaN propagation. C++ test. |
| P1-3d | `var`/`std` return the real dtype (mean of `\|x−mean\|²`), with `ddof`. C++ test. |
| P1-3e | TS/binding pass-through and vitest cases for 3a–3d. |
| P1-3f | Differential group `complex_reductions` (axis, keepdims, empty, nan/inf). |
| P1-3g | Bench cases in both suites, then ROADMAP/PROGRESS/COMPATIBILITY updates. Commit. |

### Step 4: matmul family
| Slice | Work |
|---|---|
| P1-4a | Complex `matmul` kernel (portable path) + C++ test. |
| P1-4b | Accelerate path (`cblas_cgemm`/`zgemm`) + C++ test. |
| P1-4c | `dot`, `inner`, `outer` on complex (no conjugation, as in NumPy); vitest. |
| P1-4d | Differential `complex_matmul` cases (batched, mixed real/complex promotion). |
| P1-4e | Bench cases + docs. Commit. |

### Step 5: complex linalg (one function per slice, both backends)
| Slice | Work |
|---|---|
| P1-5a | `det`: fallback, then Accelerate (`cgetrf`/`zgetrf`). |
| P1-5b | `inv`, `solve` (`?gesv`). |
| P1-5c | `qr` (`?geqrf`/`?ungqr`). |
| P1-5d | `svd` (`?gesdd`). Singular values are real. |
| P1-5e | `eigh`/`eigvalsh` (`?heevd`). Eigenvalues are real. |
| P1-5f | `eig`/`eigvals` on complex input (`?geev`). |
| P1-5g | `lstsq`, `norm` (complex magnitudes). |
| P1-5h | Differential `complex_linalg` group (results compared up to sign/phase where needed). |
| P1-5i | FFT check: confirm the complex paths accept `np.Complex` input end to end. |
| P1-5j | Bench cases + ROADMAP/PROGRESS/COMPATIBILITY; mark P1 ✅. Commit. |

Each 5x slice: C++ kernel + C++ test first, then binding/TS + vitest, all in
the same slice only if the diff stays small; otherwise split into 5x-1 / 5x-2.

## P2–P15
Slice each milestone just before it starts, using the same rules. As a rule of
thumb, one ufunc family, one function group, or one kwarg (`out=`, `where=`, …)
per slice.
