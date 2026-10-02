# Contributing to numera

Thanks for your interest in numera! Issues and pull requests are welcome.

- **Bugs and feature requests:** [open an issue](https://github.com/Rajankr542/numera/issues).
  For a bug, include your OS, CPU architecture, Node version, the numera
  version, a minimal snippet, and what NumPy returns for the same input.
- **Pull requests:** keep them small and focused, one function family or fix
  per PR.

## Ground rules

These come from [AGENTS.md](./AGENTS.md), which applies to human and AI
contributors alike:

- **Numerical work happens in C++** (`native/`). TypeScript validates input,
  orchestrates and exposes the NumPy-style API.
- **Never fake compatibility.** Behaviour is "verified" only when it is covered
  by the NumPy differential tests. Document every divergence in
  [COMPATIBILITY.md](./COMPATIBILITY.md).
- **Never remove, disable or weaken tests** to make them pass.
- **No unsupported performance claims.** Numbers go in
  [PERFORMANCE.md](./PERFORMANCE.md) with how they were measured.
- **C++ exceptions never reach JS**: translate them to typed errors
  (`native/bindings/error_binding.cpp`).
- **Architectural decisions** (public API, memory model, ABI, compatibility
  behaviour, package layout) are recorded in [DECISIONS.md](./DECISIONS.md)
  *before* they are implemented.

## Development setup

You need Node ≥ 18, [pnpm](https://pnpm.io) 8, Python 3 and a C++20 compiler
(Xcode command line tools on macOS, GCC 13+ or Clang on Linux). Python is used
only to install `cmake` and `ninja` through pip and to generate the NumPy
reference cases. Users of the npm package never need it.

```bash
git clone https://github.com/Rajankr542/numera.git && cd numera
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
export PATH="$PWD/.venv/bin:$PATH"   # cmake, ninja, numpy (dev only)
pnpm install
pnpm build                            # native addon (cmake-js) + TypeScript
```

A source build in `build/Release/` is picked up automatically. The addon is
found in this order: `NATIVPY_ADDON_PATH`, the prebuild bundled in the package
(`prebuilds/<platform>-<arch>/nativpy.node`), then the local source build
(D-007, D-026). Source builds use experimental Node-API, which frees memory
sooner (D-023).

## Tests

| Command | What it runs |
| --- | --- |
| `pnpm test` | TypeScript unit/integration tests (vitest) through the real addon |
| `pnpm test:native` | C++ unit tests (CTest) |
| `pnpm test:diff` | NumPy differential tests: regenerates cases with your installed NumPy, then replays them |
| `pnpm test:asan` | C++ tests under AddressSanitizer + UBSan |
| `pnpm typecheck` | `tsc --noEmit` on the package |
| `pnpm api:check` | NumPy API and benchmark coverage; fails on regressions |
| `pnpm bench` / `pnpm bench:suite` | numera benchmarks (`bench:numpy`, `bench:suite:numpy` for NumPy, `bench:compare` to compare) |
| `pnpm docs` | builds the API reference into `docs-dist/` |

CI runs all of these on Ubuntu and macOS with Node 18, 20 and 22.

## Making a change

The usual loop for a new function or fix:

1. **Design.** If the change touches public API or compatibility behaviour, add
   a `D-NNN` entry to [DECISIONS.md](./DECISIONS.md) first.
2. **C++ kernel** in `native/` plus a C++ test in `tests/native/`.
3. **Binding** in `native/bindings/`.
4. **TypeScript API** in `packages/numera/src/` with vitest tests in
   `packages/numera/test/`. Names are camelCase and mirror NumPy.
5. **Differential cases** in `python/generators/` so the behaviour is checked
   against real NumPy.
6. **Docs entry** in `docs/site/parts/`. Every documented example is executed
   by `packages/numera/test/docs_site.test.ts`, so keep examples real.
7. **Update** [COMPATIBILITY.md](./COMPATIBILITY.md) and
   [PROGRESS.md](./PROGRESS.md).
8. Run `pnpm build && pnpm typecheck && pnpm test && pnpm test:native && pnpm test:diff && pnpm api:check`.

### Code conventions

- C++20, RAII, no raw ownership, const-correct, builds cleanly with
  `-Wall -Wextra -Wpedantic -Werror`. Namespace `numera`, `.hpp` headers,
  snake_case file names.
- TypeScript strict mode, ESM output.
- Small [conventional commits](https://www.conventionalcommits.org), e.g.
  `feat(linalg): add pinv (D-123)` or `fix(native): ...`. Keep the repository
  buildable after every commit.

The design is described in [ARCHITECTURE.md](./ARCHITECTURE.md) and
[DECISIONS.md](./DECISIONS.md); the full specification is in
[docs/plan/PLAN.md](./docs/plan/PLAN.md).

## Publishing a release (maintainers)

Releases are **manual**, run by a maintainer from their own machine (D-031).
There is no release workflow in GitHub Actions. `@cyfora/numera` is published
to npmjs.org only, under the `latest` dist-tag.

```bash
pnpm release:dry      # rehearse everything; nothing is published, committed or tagged
pnpm release          # patch bump (1.0.0 → 1.0.1), or publish the current version if it is not on npm yet
pnpm release:minor    # 1.0.0 → 1.1.0
pnpm release:major    # 1.0.0 → 2.0.0
```

Options (after `--`, e.g. `pnpm release -- --otp 123456 --push`):
- `--otp <code>`: npm 2FA one-time password (or set `NPM_OTP`). Without it
  you're prompted when npm asks.
- `--push`: push the release commit and tag (`git push --follow-tags`), then
  create the GitHub Release for the tag with the GitHub CLI (`gh`, logged in).
  Without `--push`, the script prints the link to create it by hand.
- `--targets darwin-arm64,linux-x64`: build only these prebuilds. The tarball
  check then expects only these, so use it only for rehearsals.
- `--allow-dirty`: skip the clean-working-tree check.

`pnpm release` checks that the git tree is clean. If you're not logged in to
npm, it opens `npm login` in the browser. It checks that your npm user can
publish to `@cyfora`. It then chooses the version and runs build, typecheck and
unit tests. It builds all four prebuilds: macOS locally, and Linux in
`manylinux_2_28` Docker. So you need macOS with Xcode, and Docker Desktop.
Next it packs the tarball and checks it: all prebuilds, no source maps, no
`docs/`, and the `homepage`/`repository`/`bugs` links. It smoke-tests the
tarball in a clean project and publishes with `--access public`. Finally it
commits `chore(release): @cyfora/numera@X.Y.Z` and tags `vX.Y.Z`. If any step
fails, `package.json` is restored.

**What ships.** `scripts/stage-package.mjs` writes the package README (the root
README up to "Contributing", with relative links rewritten to this GitHub
repository and a Links section), a `COMPATIBILITY.md` without decision IDs, and
`LICENSE`. It strips PLAN/DECISIONS references from `dist/` comments and fails
if any are left in the README or COMPATIBILITY.md. Source maps are excluded
(D-029, D-240, D-241).

**Docs site.** The API reference is hosted at
[numera.cyfora.in](https://numera.cyfora.in) and is not shipped in the
package. `pnpm docs` builds it from `docs/site/api.mjs` into `docs-dist/`
(one self-contained `index.html`); deploy that directory to the host after each
release.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](./LICENSE).
