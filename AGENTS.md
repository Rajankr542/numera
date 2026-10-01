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
