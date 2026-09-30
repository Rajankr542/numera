# nativpy

**NumPy semantics. Native performance. JavaScript/TypeScript API.**

nativpy is an n-dimensional array library for Node.js. It follows NumPy's API
and behaviour. The numerical work runs in a C++20 core loaded as a Node-API
addon, and TypeScript provides a typed, NumPy-style API on top of it.

> **Status: early development (`0.0.x`).** The API is incomplete and may
> change. See [COMPATIBILITY.md](./COMPATIBILITY.md) for what is verified
> against NumPy, and [ROADMAP.md](./ROADMAP.md) for what is planned.

## Installation

> **Prebuilt binaries are not available yet** (roadmap milestone M13). Until
> then, nativpy has to be built from source, which needs a C++20 compiler,
> CMake and Ninja.

```bash
git clone https://github.com/Rajankr542/nativpy.git
cd nativpy
python3 -m venv .venv && .venv/bin/pip install -r python/requirements.txt
export PATH="$PWD/.venv/bin:$PATH"   # provides cmake, ninja, numpy
pnpm install
pnpm build                            # C++ addon + TypeScript
```

Requirements: Node.js ≥ 18, and macOS or Linux (the CI covers both).

## Quick start

```ts
import np from "nativpy";

const a = np.array([[1, 2, 3], [4, 5, 6]]);
a.shape;          // [2, 3]
a.dtype.name;     // "int64"
a.toArray();      // [[1, 2, 3], [4, 5, 6]]
String(a);        // "array([[1,2,3],[4,5,6]], dtype=int64)"

np.add(a, 10).toArray();           // [[11, 12, 13], [14, 15, 16]]
a.sum({ axis: 0 }).toArray();      // [5, 7, 9]
np.mean(a, { axis: 1 }).toArray(); // [2, 5]
```

Named imports also work, e.g. `import { array, zeros, linalg } from "nativpy";`.

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

For the same seed, nativpy produces the same numbers as NumPy's
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
per-element JavaScript. The addon file (`build/Release/nativpy.node`) is found
automatically. To use one from somewhere else, set
`NATIVPY_ADDON_PATH=/path/to/nativpy.node` (D-007).

## Differences from NumPy

nativpy aims to match NumPy, but some behaviour differs because of
JavaScript. For example, a full integer index returns a 0-d array (call
`.item()` to get a number), and 64-bit integers are only exact up to 2^53 in
`toArray()` (use `toTypedArray()` for full precision). Every difference is
listed in [COMPATIBILITY.md](./COMPATIBILITY.md).

## Performance

Benchmarks against NumPy are in [PERFORMANCE.md](./PERFORMANCE.md). Some
operations are faster than NumPy and others are slower. The numbers come from
single local runs and are not general claims.

## Development

```bash
pnpm build          # native (cmake-js) + TypeScript
pnpm test           # unit/integration tests (vitest)
pnpm test:native    # C++ unit tests (CTest)
pnpm test:diff      # NumPy differential tests (requires numpy)
pnpm test:asan      # C++ tests under ASan + UBSan
pnpm bench          # benchmarks
```

Contributor rules are in [AGENTS.md](./AGENTS.md). The design is described in
[ARCHITECTURE.md](./ARCHITECTURE.md) and [DECISIONS.md](./DECISIONS.md).

## License

MIT
