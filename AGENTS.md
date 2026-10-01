# AGENTS.md — Rules for AI/human contributors

Canonical specification: `docs/plan/PLAN.md`. Do not rewrite it unnecessarily.

## Before any task
1. Read this file and the current task slice in `docs/plan/TASK_SLICES.md`.
2. Read only the source files directly touched by that slice.
3. Run `git status` to see what is already in progress.
4. Only read `docs/plan/PLAN.md`, `ROADMAP.md`, `PROGRESS.md`, `ARCHITECTURE.md`, or `DECISIONS.md` if the task involves an architectural decision or you are unfamiliar with the area being changed.

## Task loop (PLAN §83)
Read task → inspect → design → implement → compile → unit tests → NumPy
differential tests → edge cases → sanitizers → benchmarks (if perf-sensitive)
→ review diff → update docs → update ROADMAP/PROGRESS/COMPATIBILITY/DECISIONS → commit.

## Hard rules
- Never fake implementation or compatibility.
- Never remove/disable/weaken tests to make them pass.
- Never claim completion without verification evidence.
- Never make unsupported performance claims.
- Numerical work happens in C++ (`native/`); TypeScript validates, orchestrates, and exposes ergonomics.
- Do not leak raw C++ exceptions to JS; translate to typed errors (`native/bindings/error_binding.cpp`).
- Record architectural decisions (public API, memory model, ABI, compat behavior, package layout) in `DECISIONS.md` *before* implementing them.
- Keep the repository buildable after every meaningful change. Use small conventional commits.

## Environment setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
export PATH="$PWD/.venv/bin:$PATH"   # provides cmake, ninja, numpy
pnpm install
pnpm build          # native (cmake-js) + TypeScript
pnpm test           # vitest unit/integration tests
pnpm test:native    # C++ unit tests via CTest
pnpm test:diff      # NumPy differential tests
pnpm test:asan      # C++ tests under ASan+UBSan
```

## Code conventions
- C++20, RAII, no raw ownership, const-correct, `-Wall -Wextra -Wpedantic -Werror`.
- Namespace `numera`; headers `.hpp`; files snake_case.
- TypeScript strict mode, ESM output; camelCase API names mirroring NumPy (PLAN §76).

## Parallel milestones (D-056)
P3–P15 are developed on branches `pNN` in separate worktrees. On a milestone
branch, only touch files you own: `native/bindings/pNN_binding.cpp`, new files
under `native/` and `tests/native/` (prefix them with your area, e.g.
`native/core/sorting.cpp`, `tests/native/test_p09_sorting.cpp`),
`packages/numera/src/pNN*.ts`, `packages/numera/test/pNN_*.test.ts`,
`docs/site/parts/pNN.mjs`, `api/bench-exempt/pNN.json`, `api/aliases.d/pNN.json`,
`api/exclusions.d/pNN.json`, `docs/plan/slices/pNN.md`. P4 also owns
`native/core/ufunc_math.cpp`, P5 `native/core/ufunc_logic.cpp`, P3 owns
`packages/numera/src/ndarray.ts`. Append only to `DECISIONS.md` (your reserved
D-number range), `PROGRESS.md`, `COMPATIBILITY.md`. Do not edit `ROADMAP.md`,
`TASK_SLICES.md`, `api/coverage*.json`, `index.ts`, `CMakeLists.txt` or other
milestones' files; if a shared file must change, keep the edit minimal and say
so in the commit message.
