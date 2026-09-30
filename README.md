# numera

**NumPy semantics. Native performance. JavaScript/TypeScript API.**

numera is an n-dimensional array library for Node.js. It follows NumPy's API
and behaviour. The numerical work runs in a C++20 core loaded as a Node-API
addon, and TypeScript provides a typed, NumPy-style API on top of it.

> **Status: `1.0.0`, early but usable.** Not all of NumPy is implemented yet. See
> [COMPATIBILITY.md](./COMPATIBILITY.md) for what is verified against NumPy,
> and [ROADMAP.md](./ROADMAP.md) for what is planned.

## Installation

```bash
npm install @cyfora/numera   # or: pnpm add @cyfora/numera / yarn add @cyfora/numera
```

It's also published to
[GitHub Packages](https://github.com/Rajankr542/numera/packages) as
`@rajankr542/numera` (GitHub requires the owner scope there). To install from
there, add this to your project's `.npmrc`. GitHub Packages needs a token with
`read:packages` even for reads:

```ini
@rajankr542:registry=https://npm.pkg.github.com
//npm.pkg.github.com/:_authToken=${GITHUB_TOKEN}
```

and then run `npm install @rajankr542/numera`.

That's all. The package ships a precompiled C++ addon for your platform, so you
**don't** need a compiler, CMake or Python. When you `import "@cyfora/numera"`, it
loads the matching binary and runs the numerical work in native code.

| Platform | Architectures | Node.js |
| --- | --- | --- |
| macOS 13.3+ | arm64 (Apple Silicon), x64 (Intel) | ≥ 18 |
| Linux (glibc ≥ 2.28: Ubuntu 20.04+, Debian 10+, RHEL 8+) | x64, arm64 | ≥ 18 |

Windows and Alpine/musl Linux have no prebuilt binaries yet. On those, use a
[source build](#development).

## Quick start

```ts
import np from "@cyfora/numera";

const a = np.array([[1, 2, 3], [4, 5, 6]]);
a.shape;          // [2, 3]
a.dtype.name;     // "int64"
a.toArray();      // [[1, 2, 3], [4, 5, 6]]
String(a);        // "array([[1,2,3],[4,5,6]], dtype=int64)"

np.add(a, 10).toArray();           // [[11, 12, 13], [14, 15, 16]]
a.sum({ axis: 0 }).toArray();      // [5, 7, 9]
np.mean(a, { axis: 1 }).toArray(); // [2, 5]
```

Named imports also work, e.g. `import { array, zeros, linalg } from "@cyfora/numera";`.

## Samples

### Creating arrays

```ts
np.zeros([2, 2]).toArray();                // [[0, 0], [0, 0]]
np.ones([3], { dtype: "float32" });
np.full([2, 2], 7);
np.arange(0, 10, 2).toArray();             // [0, 2, 4, 6, 8]
np.linspace(0, 1, 5).toArray();            // [0, 0.25, 0.5, 0.75, 1]
np.eye(3, undefined, { dtype: "int32" });  // 3×3 identity
np.fromTypedArray(new Float32Array([1, 2, 3])).dtype.name; // "float32"
```

Supported dtypes: `bool`, `int8`–`int64`, `uint8`–`uint64`, `float16`,
`float32`, `float64`, `complex64`, `complex128`.

### Shapes, views and indexing

```ts
const b = np.arange(12).reshape([3, 4]);
b.T.shape;                                   // [4, 3]  (view, no copy)
b.ravel().shape;                             // [12]

b.get(1).toArray();                          // b[1]      -> [4, 5, 6, 7]
b.get(1, 2).item();                          // b[1, 2]   -> 6
b.slice([[0, 2], [null, null, 2]]).toArray();// b[0:2, ::2] -> [[0, 2], [4, 6]]
np.arange(5).slice([null, null, -1]).toArray(); // [::-1] -> [4, 3, 2, 1, 0]
b.get(np.newaxis, np.ellipsis).shape;        // b[None, ...] -> [1, 3, 4]

const c = np.zeros([3]);
c.set([[0, 2]], [7, 8]);                     // c[0:2] = [7, 8]
c.toArray();                                 // [7, 8, 0]
```

A slice is written as a tuple `[start, stop, step]`, where `null` means
"omitted". Basic indexing returns views. Integer-array and boolean-mask
indexing return copies, as in NumPy.

### Math and broadcasting

```ts
np.multiply(a, np.array([1, 0, -1])).toArray(); // [[1, 0, -3], [4, 0, -6]]
np.sqrt(np.array([1, 4, 9])).toArray();         // [1, 2, 3]
np.where(np.array([true, false, true]), np.array([1, 2, 3]), 0).toArray(); // [1, 0, 3]
```

Available functions: `add`, `subtract`, `multiply`, `divide`, `power`, `mod`,
`floorDivide`, `abs`, `negative`, `sqrt`, `exp`, `log`, `broadcastTo` and
`broadcastShapes`.

### Reductions

```ts
a.sum().item();        // 21
a.max({ axis: 1 });    // [3, 6]
a.argmax().item();     // 5
np.std(a, { ddof: 1 });
```

Also available: `prod`, `min`, `mean`, `var`, `argmin`. They accept `axis`
(a number or an array of numbers), `keepdims`, `dtype`, `initial` and `ddof`
where NumPy supports them.

### Linear algebra

```ts
const A = np.array([[3, 1], [1, 2]]);
np.linalg.solve(A, np.array([9, 8])).toArray(); // [2, 3]
np.matmul(A, A).toArray();                      // [[10, 5], [5, 5]]
np.linalg.det(A).item();                        // 5
np.linalg.inv(A);
const { eigenvalues, eigenvectors } = np.linalg.eigh(A);
```

Also available: `dot`, `inner`, `outer`, and in `linalg`: `eig`, `eigvals`,
`eigvalsh`, `svd`, `qr`, `lstsq` and `norm`. All of them work on batched
(stacked) inputs.

### Random numbers (same streams as NumPy)

```ts
const rng = np.random.defaultRng(42);
rng.random([3]).toArray();      // [0.7739560485559633, 0.4388784397520523, 0.8585979199113825]
rng.integers(0, 10, [5]);
rng.normal(0, 1, [2, 2]);

np.random.seed(0);              // legacy RandomState API
np.random.rand(2, 3);
```

For the same seed, numera produces the same numbers as NumPy's
`default_rng(42)` (PCG64) and `np.random.seed` (MT19937). This is verified
bit-for-bit by the differential tests.

### FFT

```ts
const spec = np.fft.fft(np.array([1, 0, 0, 0]));
spec.dtype.name;                 // "complex128"
spec.toTypedArray();             // Float64Array of interleaved [re, im, re, im, ...]
np.fft.rfftfreq(8);
```

Also available: `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`, `ifftn` and
`fftfreq`. Complex elements can't be converted with `toArray()` yet, so use
`toTypedArray()`.

### Errors

Native errors come through as typed JavaScript errors:

```ts
try {
  np.add(np.zeros([2, 3]), np.zeros([4]));
} catch (e) {
  e instanceof np.BroadcastError; // true
}
```

The error classes are `NativpyError` (the base class), `ValueError`,
`ShapeError`, `BroadcastError`, `DTypeError`, `IndexError`, `LinAlgError`,
`MemoryError` and `NotImplementedError`.

## How it works

```
your code ──► TypeScript API (validation, NumPy-style ergonomics)
                   │  Node-API
                   ▼
              C++20 core (arrays, ufuncs, reductions, linalg, random, FFT)
```

Array data lives in native memory, and the JavaScript side holds handles to
it. Calls such as `np.add` or `np.linalg.solve` run as loops in C++, not as
per-element JavaScript. The addon is found automatically, in this order:
`NATIVPY_ADDON_PATH`, then the prebuild bundled in the package
(`prebuilds/<platform>-<arch>/nativpy.node`), then a local source build in
`build/Release/` (D-007, D-026). Published prebuilds use the stable Node-API,
so one binary works on every Node ≥ 18. Source builds use experimental
Node-API, which frees memory sooner (D-023).

## Differences from NumPy

numera aims to match NumPy, but some behaviour differs because of
JavaScript. For example, a full integer index returns a 0-d array (call
`.item()` to get a number), and 64-bit integers are only exact up to 2^53 in
`toArray()` (use `toTypedArray()` for full precision). Every difference is
listed in [COMPATIBILITY.md](./COMPATIBILITY.md).

## Performance

Benchmarks against NumPy are in [PERFORMANCE.md](./PERFORMANCE.md). Some
operations are faster than NumPy and others are slower. The numbers come from
single local runs and are not general claims.

## Development

Only needed if you want to **work on numera itself**, or use it on a platform
without a prebuild. Python is used here only to install `cmake` and `ninja`
through pip, and to generate the NumPy reference cases for the differential
tests. Users of the npm package never need it.

```bash
git clone https://github.com/Rajankr542/numera.git && cd numera
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
export PATH="$PWD/.venv/bin:$PATH"   # cmake, ninja, numpy (dev only)
pnpm install
pnpm build          # native (cmake-js) + TypeScript
pnpm test           # unit/integration tests (vitest)
pnpm test:native    # C++ unit tests (CTest)
pnpm test:diff      # NumPy differential tests (requires numpy)
pnpm test:asan      # C++ tests under ASan + UBSan
pnpm bench          # benchmarks
```

### Publishing a release (maintainers)

Releases are **automatic from GitHub Actions** (D-029). Push to a release
branch and the workflow publishes `@cyfora/numera` to npm, with no prompts:

| Branch | Version | npm dist-tag | Install |
|--------|---------|--------------|---------|
| `main` | `X.Y.Z` | `latest` | `npm install @cyfora/numera` |
| `beta` | `X.Y.Z-beta.N` | `beta` | `npm install @cyfora/numera@beta` |
| `alpha` | `X.Y.Z-alpha.N` | `alpha` | `npm install @cyfora/numera@alpha` |

The version comes from the [Conventional Commits](https://www.conventionalcommits.org)
since the last stable tag:

| Commits since the last `vX.Y.Z` | Next version |
|---------------------------------|--------------|
| `fix:` / `perf:` / `revert:` | patch (`1.0.0` → `1.0.1`) |
| `feat:` | minor (`1.0.0` → `1.1.0`) |
| `feat!:`, `fix!:`, or a `BREAKING CHANGE:` footer | major (`1.0.0` → `2.0.0`) |
| only `docs:` / `chore:` / `test:` / `ci:` / `build:` / `refactor:` / `style:` | **no release** |

A push that only changes `*.md`, `docs/` or `benchmarks/` does not start the
workflow. Run `node scripts/next-version.mjs latest` (or `beta` / `alpha`) to
see what the next push would release.

Each release builds all four prebuilds on native runners, runs the unit tests,
and smoke-tests the packed tarball in a clean project. Then it publishes with
`--tag <channel>` and pushes tag `vX.Y.Z[-channel.N]`, and creates a GitHub
Release: marked latest for `main`, a pre-release for `beta`/`alpha`. A stable
release also commits the new version to `package.json` (`chore(release): …`).
Prereleases only tag, so `package.json` on `main` stays at the last stable
version.

**One-time setup:**

1. On npmjs.com (as `cyfora`), go to *Access Tokens → Generate New Token →
   Granular Access Token*. Give it **Read and write** on `@cyfora/numera`, tick
   **Bypass two-factor authentication**, and set an expiry. Renew the token
   before it expires.
2. In the GitHub repo, go to *Settings → Secrets and variables → Actions → New
   repository secret*. Name it **`NPM_TOKEN`** and paste the token.
3. Create the prerelease branches: `git push origin main:beta main:alpha`.

**Manual run:** Actions → **Release** → *Run workflow* on a release branch.
Inputs:
- `bump`: `auto` (from the commits), or force `patch` / `minor` / `major`.
- `registry`: `npm` (default), `github` (`@rajankr542/numera` on GitHub
  Packages) or `both`.
- `dry_run`: build and test everything, but don't publish, push or create a release.

**What ships.** The repository is private, so the package must stand on its
own. `scripts/stage-package.mjs` writes a package README (this file up to
"Development", with no links into the repo) and a `COMPATIBILITY.md` without
decision IDs. It also strips PLAN/DECISIONS references from `dist/` comments.
It fails if anything repo-only is left. Source maps are excluded, and
`package.json` has no `repository`/`homepage`/`bugs`.

**Locally (fallback):** `pnpm release:dry` rehearses the whole release.
`pnpm release` / `release:minor` / `release:major` publish from your machine to
`@latest`. This path opens `npm login` in the browser and prompts for the 2FA
code if your account uses 2FA. It needs macOS with Xcode, and Docker Desktop
for the Linux binaries, which are built in `manylinux_2_28`. It commits
`chore(release)` and tags `vX.Y.Z`; push with `git push --follow-tags` or pass
`--push`. The CI then skips that commit.

Contributor rules are in [AGENTS.md](./AGENTS.md). The design is described in
[ARCHITECTURE.md](./ARCHITECTURE.md) and [DECISIONS.md](./DECISIONS.md).

## License

MIT
