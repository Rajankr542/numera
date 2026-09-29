# nativpy

## Native Numerical Computing for JavaScript/TypeScript

**Working package name:** `nativpy`

**Tagline:**

> NumPy semantics. Native performance. JavaScript/TypeScript API.

---

# 1. Project Mission

Build a production-quality numerical-computing library for Node.js and TypeScript that provides a NumPy-inspired and progressively NumPy-compatible API while executing computationally expensive operations in native C++.

The project must not be a simple JavaScript implementation of mathematical functions.

The target architecture is:

```text
JavaScript / TypeScript
        │
        ▼
      nativpy
        │
        ▼
Node-API / N-API
        │
        ▼
       C++
        │
 ┌──────┼────────┐
 ▼      ▼        ▼
SIMD   BLAS    LAPACK
 │      │        │
 └──────┼────────┘
        ▼
       CPU
```

The project should eventually provide:

* multidimensional arrays
* NumPy-style dtype system
* shape and stride semantics
* views without unnecessary copying
* slicing
* broadcasting
* vectorized operations
* reductions
* statistics
* random number generation
* linear algebra
* FFT
* sorting
* searching
* indexing
* mathematical functions
* numerical utilities
* native CPU acceleration
* optional SIMD
* multithreading where appropriate
* TypeScript definitions
* Node.js support
* automated NumPy differential testing
* performance benchmarking
* documentation
* examples
* npm publishing
* prebuilt native binaries
* CI/CD
* semantic versioning

The project must be designed so that GPU acceleration or WebAssembly can potentially be added later without redesigning the core API.

---

# 2. Primary Objective

The long-term objective is:

> Allow JavaScript/TypeScript developers to perform serious numerical and scientific computing without needing Python for the numerical layer.

Example:

```ts
import np from "nativpy";

const A = np.array([
  [1, 2],
  [3, 4]
]);

const B = np.array([
  [5, 6],
  [7, 8]
]);

const C = np.matmul(A, B);

console.log(C.toArray());
```

Expected:

```text
[
  [19, 22],
  [43, 50]
]
```

The implementation should execute the expensive computation natively.

---

# 3. Important Engineering Principle

Do NOT attempt to implement NumPy by writing hundreds of JavaScript functions.

The implementation must be divided into:

```text
Public API
    ↓
TypeScript layer
    ↓
Native bindings
    ↓
C++ numerical core
    ↓
Optimized numerical backend
```

JavaScript should primarily:

* validate API-level arguments where appropriate
* construct objects
* expose ergonomic APIs
* manage high-level orchestration

C++ should perform:

* numerical computation
* memory-intensive operations
* vectorized operations
* matrix operations
* reductions
* numerical kernels
* dtype conversions
* broadcasting calculations
* indexing calculations
* native memory operations

---

# 4. Non-Goals for Initial Release

Do not attempt to implement all of the following in v0.1:

* neural networks
* automatic differentiation
* GPU execution
* distributed computing
* CUDA
* training LLMs
* deep-learning layers
* pandas equivalent
* plotting
* dataframe functionality

These can be future projects.

The initial project is a numerical computing foundation.

---

# 5. Technology Stack

## Primary languages

### TypeScript

Used for:

* public API
* type definitions
* developer ergonomics
* documentation examples
* high-level wrappers

### C++

Used for:

* ndarray implementation
* memory management
* numerical kernels
* native execution
* SIMD
* backend integration

### Python

Used ONLY as a:

* NumPy reference
* differential testing environment
* benchmark comparison environment
* test-data generator

Python must NOT become a runtime dependency of the npm package.

---

# 6. Native Binding Layer

Use Node-API / N-API.

Do not depend directly on unstable Node/V8 APIs unless absolutely necessary.

The native binding must be designed to remain compatible across Node.js versions where Node-API provides compatibility.

Preferred structure:

```text
native/
├── bindings/
│   ├── addon.cpp
│   ├── ndarray_binding.cpp
│   ├── dtype_binding.cpp
│   └── error_binding.cpp
```

The public TypeScript API must not expose C++ implementation details.

---

# 7. Repository Structure

Create the following structure:

```text
nativpy/
│
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
│
├── AGENTS.md
├── ROADMAP.md
├── ARCHITECTURE.md
├── COMPATIBILITY.md
├── PERFORMANCE.md
├── DECISIONS.md
│
├── package.json
├── pnpm-workspace.yaml
├── tsconfig.json
├── CMakeLists.txt
│
├── packages/
│   └── nativpy/
│       ├── src/
│       │   ├── index.ts
│       │   ├── ndarray.ts
│       │   ├── dtype.ts
│       │   ├── linalg.ts
│       │   ├── random.ts
│       │   ├── fft.ts
│       │   └── errors.ts
│       ├── test/
│       └── package.json
│
├── native/
│   ├── core/
│   │   ├── ndarray.cpp
│   │   ├── ndarray.hpp
│   │   ├── dtype.cpp
│   │   ├── dtype.hpp
│   │   ├── shape.cpp
│   │   ├── shape.hpp
│   │   ├── strides.cpp
│   │   ├── strides.hpp
│   │   ├── memory.cpp
│   │   └── memory.hpp
│   │
│   ├── kernels/
│   │   ├── arithmetic/
│   │   ├── comparison/
│   │   ├── reduction/
│   │   ├── transcendental/
│   │   └── indexing/
│   │
│   ├── linalg/
│   ├── random/
│   ├── fft/
│   ├── simd/
│   ├── threading/
│   └── bindings/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── compatibility/
│   ├── differential/
│   ├── edge-cases/
│   └── memory/
│
├── python/
│   ├── reference/
│   ├── generators/
│   ├── comparator/
│   └── requirements.txt
│
├── benchmarks/
│   ├── numpy/
│   ├── nativpy/
│   ├── datasets/
│   └── reports/
│
├── examples/
│   ├── basics/
│   ├── matrix/
│   ├── statistics/
│   ├── image/
│   └── performance/
│
├── docs/
│   ├── getting-started.md
│   ├── ndarray.md
│   ├── dtype.md
│   ├── indexing.md
│   ├── broadcasting.md
│   ├── mathematics.md
│   ├── linalg.md
│   ├── random.md
│   ├── fft.md
│   ├── performance.md
│   └── numpy-compatibility.md
│
├── scripts/
│   ├── build-native.ts
│   ├── test-compatibility.ts
│   ├── benchmark.ts
│   ├── generate-bindings.ts
│   └── release.ts
│
└── .github/
    └── workflows/
        ├── test.yml
        ├── compatibility.yml
        ├── benchmark.yml
        └── release.yml
```

---

# 8. Core NDArray Design

This is the most important component.

The internal representation must conceptually contain:

```cpp
class NDArray {
public:

    void* data;

    DType dtype;

    std::vector<int64_t> shape;

    std::vector<int64_t> strides;

    int64_t offset;

    std::shared_ptr<MemoryBuffer> buffer;

    size_t ndim() const;

    size_t size() const;
};
```

The implementation must support:

* arbitrary dimensions
* arbitrary strides
* views
* slicing
* transposition
* reshaping
* contiguous and non-contiguous arrays
* offsets
* shared memory ownership

---

# 9. Memory Model

Avoid unnecessary memory copies.

Example:

```ts
const a = np.arange(12).reshape([3, 4]);
const b = a.slice([null, [1, 3]]);
```

`b` should preferably reference the same underlying memory as `a`.

Conceptually:

```text
A:

0  1  2  3
4  5  6  7
8  9 10 11

B:

1  2
5  6
9 10
```

B should be represented using:

```text
same buffer
different offset
different shape
different strides
```

unless NumPy semantics require a copy.

---

# 10. DType System

Implement a strong dtype abstraction.

Initial dtypes:

```text
bool
int8
uint8
int16
uint16
int32
uint32
int64
uint64
float16
float32
float64
complex64
complex128
```

Future:

```text
bfloat16
```

Dtype must determine:

* item size
* alignment
* arithmetic implementation
* casting rules
* comparison behavior
* reduction behavior

Example:

```ts
const a = np.array([1, 2, 3], { dtype: np.float32 });
```

Expose:

```ts
a.dtype
a.itemSize
```

---

# 11. Array Creation API

Implement:

```ts
np.array()
np.asarray()
np.zeros()
np.ones()
np.empty()
np.full()
np.arange()
np.linspace()
np.logspace()
np.eye()
np.identity()
np.diag()
```

Examples:

```ts
const a = np.array([1, 2, 3]);

const b = np.zeros([3, 4]);

const c = np.ones([2, 2], {
  dtype: np.float32
});

const d = np.arange(0, 10, 2);

const e = np.linspace(0, 1, 100);
```

---

# 12. Shape Operations

Implement:

```ts
reshape()
resize()
ravel()
flatten()
squeeze()
expandDims()
transpose()
swapAxes()
moveAxis()
```

Where NumPy semantics require a view, return a view.

Where NumPy semantics require a copy, return a copy.

Do not simply optimize for convenience.

---

# 13. Indexing

Implement progressively.

## Basic indexing

```ts
a.get(0)
a.get(1, 2)
```

## Slicing

Support:

```ts
a.slice(...)
```

Examples:

```ts
a.slice([0, 5]);

a.slice([
  [1, 4],
  [2, 6]
]);
```

## Reverse slicing

```ts
a.slice([
  [null, null, -1]
]);
```

## Ellipsis

Support:

```text
...
```

## Integer-array indexing

## Boolean indexing

## Advanced indexing

The final implementation should follow NumPy semantics as closely as possible.

---

# 14. Broadcasting

Broadcasting is a mandatory core feature.

Example:

```ts
const a = np.ones([3, 4]);

const b = np.array([1, 2, 3, 4]);

const c = np.add(a, b);
```

Result:

```text
shape = [3, 4]
```

Implement broadcasting as a reusable native subsystem.

Do not duplicate broadcasting logic across every operation.

Create:

```cpp
BroadcastPlan
```

containing:

```text
output shape
input shapes
input strides
broadcast dimensions
iteration metadata
```

---

# 15. Universal Functions

Implement vectorized operations.

Arithmetic:

```ts
np.add()
np.subtract()
np.multiply()
np.divide()
np.floorDivide()
np.mod()
np.power()
```

Unary:

```ts
np.abs()
np.sign()
np.negate()
np.sqrt()
np.square()
np.exp()
np.log()
np.log10()
np.log2()
```

Trigonometric:

```ts
np.sin()
np.cos()
np.tan()
np.arcsin()
np.arccos()
np.arctan()
```

Hyperbolic:

```ts
np.sinh()
np.cosh()
np.tanh()
```

Rounding:

```ts
np.floor()
np.ceil()
np.round()
np.trunc()
```

---

# 16. Comparison Operations

Implement:

```ts
np.equal()
np.notEqual()
np.less()
np.lessEqual()
np.greater()
np.greaterEqual()
```

Logical:

```ts
np.logicalAnd()
np.logicalOr()
np.logicalNot()
np.logicalXor()
```

---

# 17. Reductions

Implement:

```ts
np.sum()
np.prod()
np.mean()
np.min()
np.max()
np.argmin()
np.argmax()
np.std()
np.var()
```

Support:

```ts
axis
keepdims
dtype
initial
```

where appropriate.

Example:

```ts
const a = np.array([
  [1, 2],
  [3, 4]
]);

np.sum(a);
np.sum(a, { axis: 0 });
np.sum(a, { axis: 1 });
```

---

# 18. Statistics

Implement:

```ts
np.mean()
np.median()
np.average()
np.std()
np.var()
np.percentile()
np.quantile()
np.cumsum()
np.cumprod()
np.cumulativeSum()
```

Match NumPy numerical behavior as closely as possible.

---

# 19. Linear Algebra

Create:

```ts
np.linalg
```

Implement:

```ts
np.linalg.matmul()
np.linalg.dot()
np.linalg.inner()
np.linalg.outer()
np.linalg.norm()
np.linalg.det()
np.linalg.inv()
np.linalg.solve()
np.linalg.lstsq()
np.linalg.eig()
np.linalg.eigh()
np.linalg.svd()
np.linalg.qr()
```

Prefer optimized native libraries.

Do not write naïve C++ implementations when high-quality BLAS/LAPACK implementations are available.

---

# 20. Matrix Multiplication

Support:

```ts
np.matmul(A, B)
```

and optionally:

```ts
A.matmul(B)
```

The implementation should select an optimized backend.

Potential backends:

```text
OpenBLAS
BLAS
LAPACK
Accelerate
MKL
custom SIMD kernels
```

The backend must be abstracted.

Example:

```text
LinearAlgebraBackend
├── OpenBLASBackend
├── AccelerateBackend
├── MKLBackend
└── FallbackBackend
```

---

# 21. SIMD

Implement CPU-specific optimization where appropriate.

Potential targets:

```text
x86_64:
    SSE
    AVX
    AVX2
    AVX-512

ARM64:
    NEON
```

The runtime should detect supported instructions.

Never assume AVX-512 exists.

Provide safe fallback kernels.

Example:

```text
Runtime CPU detection
        │
        ├── AVX-512 → AVX-512 kernel
        ├── AVX2    → AVX2 kernel
        ├── SSE     → SSE kernel
        ├── NEON    → NEON kernel
        └── scalar  → portable kernel
```

---

# 22. Multithreading

Investigate native thread pools for expensive operations.

Do NOT create a thread per operation.

Create a reusable thread pool.

Parallelize operations when array sizes justify it.

For small arrays, single-threaded execution may be faster.

Benchmark the threshold.

---

# 23. Random Module

Create:

```ts
np.random
```

Implement:

```ts
np.random.rand()
np.random.randn()
np.random.random()
np.random.uniform()
np.random.normal()
np.random.randint()
np.random.choice()
np.random.shuffle()
np.random.permutation()
```

Implement a deterministic seeded RNG.

Example:

```ts
const rng = np.random.defaultRng(42);

const a = rng.normal([1000]);
const b = rng.uniform([1000]);
```

The initial goal is deterministic reproducibility within nativpy.

Do not claim bit-for-bit NumPy RNG compatibility unless specifically implemented and verified.

---

# 24. FFT

Create:

```ts
np.fft
```

Implement:

```ts
np.fft.fft()
np.fft.ifft()
np.fft.fft2()
np.fft.ifft2()
np.fft.fftn()
np.fft.ifftn()
np.fft.rfft()
np.fft.irfft()
np.fft.fftfreq()
```

Use an optimized native backend where appropriate.

---

# 25. Sorting and Searching

Implement:

```ts
np.sort()
np.argsort()
np.partition()
np.argpartition()
np.searchsorted()
np.argmax()
np.argmin()
```

---

# 26. Set Operations

Implement:

```ts
np.unique()
np.intersect1d()
np.union1d()
np.setdiff1d()
np.setxor1d()
```

---

# 27. NaN / Infinity Semantics

Explicitly test:

```text
NaN
+Infinity
-Infinity
-0
+0
```

Operations must define behavior correctly.

Test:

```ts
np.isnan()
np.isinf()
np.isfinite()
```

---

# 28. Error Handling

Native errors must be translated into meaningful JavaScript errors.

Examples:

```text
ShapeError
DTypeError
IndexError
BroadcastError
ValueError
MemoryError
NotImplementedError
```

Do not leak raw C++ exceptions to JavaScript.

---

# 29. TypeScript API

Expose strongly typed APIs.

Example:

```ts
const a = np.array([1, 2, 3]);

a.shape;
a.ndim;
a.size;
a.dtype;
a.strides;
```

Methods:

```ts
a.reshape(...)
a.transpose(...)
a.astype(...)
a.copy()
a.slice(...)
a.sum()
a.mean()
a.min()
a.max()
```

Functions:

```ts
np.add()
np.multiply()
np.matmul()
np.mean()
```

Both styles may be supported where useful.

---

# 30. Conversion API

Support:

```ts
a.toArray()
a.toTypedArray()
a.toBuffer()
a.toString()
```

Example:

```ts
const a = np.array([
  [1, 2],
  [3, 4]
]);

console.log(a.toArray());
```

Do not make `toArray()` the internal representation.

It is a conversion operation.

---

# 31. TypedArray Interoperability

Support:

```ts
np.fromTypedArray()
```

Example:

```ts
const data = new Float32Array([1, 2, 3, 4]);

const a = np.fromTypedArray(data, [2, 2]);
```

Where safe, avoid unnecessary copying.

Clearly document ownership/lifetime rules.

---

# 32. Buffer Interoperability

Support Node.js:

```ts
Buffer
ArrayBuffer
SharedArrayBuffer
TypedArray
```

where safe and appropriate.

---

# 33. Zero-Copy Principle

The implementation should follow:

> Copy only when required by semantics or safety.

Every unexpected copy in a performance-sensitive path should be considered a potential bug.

Add instrumentation in development builds to detect expensive copies.

---

# 34. NumPy Differential Testing

This is one of the most important project components.

Python NumPy is the reference implementation.

Create:

```text
python/reference/
python/generators/
python/comparator/
```

For each operation:

```text
Generate input
      ↓
Run NumPy
      ↓
Serialize result
      ↓
Run nativpy
      ↓
Serialize result
      ↓
Compare
```

Compare:

* dtype
* shape
* values
* NaN
* infinity
* signed zero where applicable
* exceptions
* view/copy behavior where testable
* ordering
* axis semantics

---

# 35. Property-Based Testing

Use generated random cases.

Generate:

```text
1D
2D
3D
4D
5D
```

with random:

```text
shapes
dtypes
values
strides
slices
axes
broadcasting combinations
```

Example generated test:

```text
A shape = [3, 1, 5]
B shape = [1, 4, 5]

operation = multiply

dtype A = float32
dtype B = float32
```

Run both implementations.

---

# 36. Compatibility Matrix

Maintain:

```text
COMPATIBILITY.md
```

Example:

```text
Category                  Status

array                     ✅
zeros                     ✅
ones                      ✅
arange                    ✅
reshape                   ✅
transpose                 ✅
broadcasting              ✅
slicing                   🟡
advanced indexing         🟡
float16                   🟡
complex                   🟡
linalg                    🟡
random                    🟡
fft                       🟡
```

Never claim 100% compatibility until it has been demonstrated by tests.

---

# 37. Automated API Coverage

Create a script that tracks:

```text
NumPy public API
        vs
nativpy implemented API
```

Generate:

```text
API coverage: 63.4%
```

This should be part of CI.

---

# 38. Performance Benchmarking

Benchmark against NumPy.

Do NOT compare:

```text
JavaScript loop
vs
NumPy vectorized operation
```

That is meaningless.

Compare:

```text
nativpy native operation
vs
NumPy equivalent operation
```

on equivalent hardware.

---

# 39. Benchmark Categories

Measure:

```text
array creation
memory allocation
elementwise addition
elementwise multiplication
broadcasting
reductions
matrix multiplication
transpose
reshape
slicing
sorting
FFT
SVD
inverse
random generation
```

---

# 40. Benchmark Sizes

Test:

```text
tiny
small
medium
large
very large
```

Example:

```text
1K
10K
100K
1M
10M
100M
```

For matrices:

```text
32x32
128x128
512x512
1024x1024
2048x2048
4096x4096
```

---

# 41. Performance Metrics

Record:

```text
execution time
throughput
memory usage
allocation count
copy count
CPU utilization
thread count
```

Produce JSON:

```json
{
  "operation": "matmul",
  "shape": [2048, 2048],
  "numpy_ms": 91.3,
  "nativpy_ms": 89.7,
  "relative_performance": 1.0178
}
```

Do not make unsupported claims.

---

# 42. Performance Goal

The target is not to blindly claim:

> nativpy is 100% as fast as NumPy.

Instead:

> Match or approach NumPy performance for major computational workloads while providing a JavaScript/TypeScript API.

Where the same native backend is used, aim for minimal wrapper overhead.

---

# 43. Memory Benchmarks

Measure:

```text
allocation
copy
view creation
reshape
transpose
slice
```

Example:

```text
reshape 100M elements

NumPy:
    X ms
    no copy

nativpy:
    Y ms
    no copy
```

---

# 44. Kernel Fusion

Investigate optional operation fusion.

Example:

```ts
np.relu(
  np.add(
    np.matmul(A, B),
    bias
  )
);
```

Potentially execute as a fused native operation.

Do not implement this prematurely.

First establish correct semantics.

---

# 45. Backend Abstraction

Create:

```text
backend/
├── scalar/
├── simd/
├── blas/
├── lapack/
└── platform/
```

The core ndarray implementation must not directly depend on one numerical backend.

---

# 46. Platform Support

Initial target:

```text
Linux x64
macOS x64
macOS ARM64
Windows x64
```

Later:

```text
Linux ARM64
Windows ARM64
```

CI must test supported platforms.

---

# 47. Prebuilt Native Packages

Do not require every npm user to install:

```text
C++
CMake
Ninja
BLAS
LAPACK
```

for normal usage.

The package should eventually distribute prebuilt binaries.

Potential model:

```text
nativpy
nativpy-linux-x64
nativpy-linux-arm64
nativpy-darwin-x64
nativpy-darwin-arm64
nativpy-win32-x64
```

Use platform-specific optional dependencies where appropriate.

The installation experience should be:

```bash
npm install nativpy
```

followed by:

```ts
import np from "nativpy";
```

without requiring users to compile C++.

---

# 48. Source Build Fallback

Provide:

```bash
npm install nativpy --build-from-source
```

or an equivalent documented mechanism.

The build process should detect:

```text
compiler
CMake
architecture
OS
CPU features
```

and fail with useful instructions.

---

# 49. Package Design

Main package:

```text
nativpy
```

Potential future packages:

```text
@nativpy/core
@nativpy/blas
@nativpy/lapack
@nativpy/fft
@nativpy/benchmark
@nativpy/autograd
@nativpy/nn
```

Do not split into many packages until there is a demonstrated need.

---

# 50. ESM and CommonJS

Support modern Node.js ESM.

Prefer:

```ts
import np from "nativpy";
```

Also provide CommonJS compatibility where practical:

```js
const np = require("nativpy");
```

---

# 51. README Requirements

README must contain:

1. What nativpy is
2. Why it exists
3. Installation
4. Quick start
5. Basic examples
6. Performance philosophy
7. Native architecture
8. NumPy compatibility
9. Platform support
10. API documentation link
11. Benchmarks
12. Roadmap
13. Contributing
14. License

---

# 52. README Sample

Example:

```ts
import np from "nativpy";

const a = np.array([
  [1, 2],
  [3, 4]
]);

const b = np.array([
  [5, 6],
  [7, 8]
]);

const result = np.matmul(a, b);

console.log(result.toArray());
```

Output:

```text
[
  [19, 22],
  [43, 50]
]
```

---

# 53. Machine Learning Example

Provide:

```ts
const X = np.random.normal([1000, 10]);
const weights = np.random.normal([10, 1]);

const predictions = np.matmul(X, weights);

console.log(predictions.shape);
```

---

# 54. Broadcasting Example

```ts
const matrix = np.ones([1000, 3]);

const bias = np.array([1, 2, 3]);

const result = np.add(matrix, bias);
```

Expected:

```text
result.shape === [1000, 3]
```

---

# 55. Statistics Example

```ts
const data = np.random.normal([1_000_000]);

console.log("mean:", np.mean(data));
console.log("std:", np.std(data));
console.log("min:", np.min(data));
console.log("max:", np.max(data));
```

---

# 56. Linear Algebra Example

```ts
const A = np.array([
  [3, 1],
  [1, 2]
]);

const b = np.array([9, 8]);

const x = np.linalg.solve(A, b);

console.log(x.toArray());
```

---

# 57. Image Processing Example

Show how users could represent image data:

```ts
const image = np.zeros([1080, 1920, 3], {
  dtype: np.uint8
});

const red = image.slice([
  [null, null],
  [null, null],
  0
]);
```

Later integrate with image libraries rather than building image codecs into nativpy.

---

# 58. Test Requirements

Every public operation must have:

```text
unit test
edge case test
NumPy differential test
```

Performance-sensitive operations must additionally have:

```text
benchmark
```

---

# 59. Required Edge Cases

Test:

```text
empty arrays
zero-dimensional arrays
one-element arrays
negative dimensions
invalid dimensions
large dimensions
NaN
Infinity
negative zero
integer overflow
floating-point precision
broadcasting mismatch
invalid axes
duplicate axes
negative axes
out-of-range indexing
empty slices
reverse slices
non-contiguous views
```

---

# 60. Memory Safety

Native code must be tested for:

```text
use-after-free
double-free
buffer overflow
out-of-bounds indexing
dangling views
invalid dtype casts
race conditions
```

Use sanitizers:

```text
AddressSanitizer
UndefinedBehaviorSanitizer
ThreadSanitizer
```

where supported.

---

# 61. C++ Quality

Use:

```text
C++20 or newer where justified
```

Follow:

```text
RAII
smart pointers
const correctness
exception safety
no raw ownership
```

Avoid unnecessary global state.

---

# 62. Formatting

Use:

```text
clang-format
ESLint
Prettier
```

CI must reject formatting violations.

---

# 63. Static Analysis

Use:

```text
clang-tidy
compiler warnings
TypeScript strict mode
```

Compile C++ with strong warnings.

Warnings should not be silently ignored.

---

# 64. CI Pipeline

Every pull request must run:

```text
install
↓
TypeScript build
↓
C++ build
↓
unit tests
↓
integration tests
↓
compatibility tests
↓
sanitizers
↓
lint
```

Performance benchmarks can run separately for pull requests and nightly builds.

---

# 65. Nightly Compatibility Testing

Nightly CI should run a large generated test suite.

Example:

```text
100,000+ generated cases
```

Eventually:

```text
1,000,000+
```

Compare against NumPy.

Store failures as artifacts.

---

# 66. Fuzz Testing

Use fuzzing for:

```text
indexing
broadcasting
dtype conversion
shape operations
slicing
memory handling
```

Every discovered bug must become a regression test.

---

# 67. Release Process

Use semantic versioning:

```text
0.x.y
```

during development.

Eventually:

```text
1.0.0
```

only when:

* API is stable
* compatibility suite is mature
* native binaries are reliable
* CI is comprehensive
* documentation is complete

---

# 68. Release Checklist

Before npm release:

```text
[ ] tests pass
[ ] compatibility tests pass
[ ] benchmarks generated
[ ] native binaries built
[ ] README updated
[ ] CHANGELOG updated
[ ] version updated
[ ] package contents inspected
[ ] npm pack tested
[ ] installation tested on clean machines
[ ] examples tested
[ ] Git tag created
```

---

# 69. npm Package Validation

Run:

```bash
npm pack
```

Inspect the resulting tarball.

Verify:

```text
README
LICENSE
package.json
native binaries
TypeScript declarations
runtime JS
```

No unnecessary source/test artifacts should be shipped.

---

# 70. Clean Installation Test

Test:

```bash
mkdir test-install
cd test-install

npm init -y
npm install ../nativpy-package.tgz
```

Then:

```ts
import np from "nativpy";

console.log(
  np.matmul(
    np.array([[1, 2]]),
    np.array([[3], [4]])
  ).toArray()
);
```

Expected:

```text
[[11]]
```

---

# 71. Documentation

Every public API must have:

```text
description
signature
parameters
return value
exceptions
examples
NumPy equivalent
notes on copies/views
```

Example:

```text
np.reshape(array, shape)

NumPy equivalent:
numpy.reshape(array, shape)
```

---

# 72. NumPy Compatibility Documentation

Maintain a page:

```text
docs/numpy-compatibility.md
```

Organize:

```text
Implemented
Partially implemented
Different behavior
Not implemented
```

Never hide incompatibilities.

---

# 73. Compatibility Policy

The project must distinguish:

### API compatibility

Does the function exist?

### Semantic compatibility

Does it behave like NumPy?

### Numerical compatibility

Are the results sufficiently close?

### Performance compatibility

Is performance comparable?

These are separate metrics.

---

# 74. Numerical Comparison Rules

Do not always require:

```text
exact equality
```

for floating-point calculations.

Use appropriate tolerances.

For example:

```text
absolute tolerance
relative tolerance
dtype-specific tolerance
```

However, exact equality should be used where mathematically/semantically appropriate.

---

# 75. Benchmark Honesty

Never publish only favorable benchmarks.

Publish:

```text
fast
medium
slow
memory-heavy
small-array
large-array
```

results.

Explain hardware and software versions.

Example:

```text
CPU:
AMD Ryzen ...

Node:
...

Python:
...

NumPy:
...

nativpy:
...
```

---

# 76. API Design Principle

Prefer familiar NumPy naming.

For example:

```ts
np.zeros()
np.ones()
np.arange()
np.reshape()
np.transpose()
np.sum()
np.mean()
np.matmul()
```

Do not invent different names without a strong reason.

---

# 77. Fluent API

Optional convenience:

```ts
const result = np
  .array(data)
  .reshape([100, 10])
  .multiply(2)
  .mean();
```

But the functional API remains primary:

```ts
np.mean(
  np.multiply(
    np.array(data),
    2
  )
);
```

---

# 78. Internal Iterator

Build a generalized native iterator capable of iterating:

```text
contiguous arrays
strided arrays
broadcasted arrays
views
```

This should become the foundation for most element-wise kernels.

---

# 79. Kernel API

Create a reusable kernel abstraction:

```cpp
template<typename T>
void binary_kernel(
    const NDArray& a,
    const NDArray& b,
    NDArray& output,
    BinaryOp op
);
```

Support:

```text
contiguous fast path
strided path
broadcast path
SIMD path
parallel path
```

---

# 80. Optimization Hierarchy

Every operation should conceptually choose:

```text
1. Specialized fast path
2. SIMD path
3. Multithreaded path
4. Generic strided path
5. Scalar fallback
```

Correctness comes before optimization.

---

# 81. Benchmark-Driven Optimization

Never optimize based purely on intuition.

For every optimization:

```text
baseline
↓
change
↓
benchmark
↓
verify correctness
↓
compare memory
↓
keep/revert
```

---

# 82. Agent Development Rules

The AI coding agent must follow:

1. Read `AGENTS.md`.
2. Read `ROADMAP.md`.
3. Read `ARCHITECTURE.md`.
4. Never skip tests.
5. Never claim a task is complete without verification.
6. Never remove a failing test just to make CI pass.
7. Never weaken compatibility checks to hide failures.
8. Never introduce a performance regression without documenting it.
9. Commit working milestones.
10. Update documentation when APIs change.
11. Update compatibility matrix when functionality changes.
12. Update benchmarks when performance-sensitive code changes.

---

# 83. Agent Task Loop

For each task:

```text
Read task
    ↓
Inspect repository
    ↓
Understand existing implementation
    ↓
Design solution
    ↓
Implement
    ↓
Compile
    ↓
Run unit tests
    ↓
Run NumPy comparison
    ↓
Run edge cases
    ↓
Run benchmark if relevant
    ↓
Review diff
    ↓
Update docs
    ↓
Commit
```

If a test fails:

```text
Do not bypass the test.

Investigate root cause.
Fix implementation.
Run test again.
```

---

# 84. Agent Checkpointing

Maintain:

```text
PROGRESS.md
```

Example:

```text
Current milestone:
NDArray core

Completed:
- memory buffer
- shape
- strides
- dtype
- reshape

Current:
- slicing

Blocked:
- complex dtype

Next:
- view lifetime tests
```

The agent must update this file after major milestones.

---

# 85. Git Strategy

Use small commits.

Examples:

```text
feat(core): add ndarray memory model
feat(core): add dtype system
feat(core): add shape and stride support
feat(core): implement reshape views
feat(indexing): add basic slicing
feat(ops): add elementwise arithmetic
test(compat): add broadcasting differential tests
perf(simd): optimize float32 addition
```

Avoid giant commits containing unrelated changes.

---

# 86. Milestone Roadmap

## Milestone 0 — Project infrastructure

Deliver:

```text
repository
build system
CI
CMake
TypeScript
Node-API
testing framework
Python NumPy harness
```

---

## Milestone 1 — NDArray

Deliver:

```text
buffer
dtype
shape
strides
offset
size
ndim
memory ownership
views
```

---

## Milestone 2 — Creation

Deliver:

```text
array
asarray
zeros
ones
empty
full
arange
linspace
eye
```

---

## Milestone 3 — Shape

Deliver:

```text
reshape
transpose
squeeze
expandDims
ravel
flatten
```

---

## Milestone 4 — Arithmetic

Deliver:

```text
add
subtract
multiply
divide
power
mod
abs
sqrt
exp
log
```

---

## Milestone 5 — Broadcasting

Deliver:

```text
broadcast rules
broadcast iterator
broadcast tests
```

---

## Milestone 6 — Indexing

Deliver:

```text
integer indexing
slicing
negative indexing
reverse slicing
ellipsis
boolean indexing
advanced indexing
```

---

## Milestone 7 — Reductions

Deliver:

```text
sum
prod
mean
min
max
std
var
argmin
argmax
```

---

## Milestone 8 — Linear algebra

Deliver:

```text
dot
matmul
norm
solve
inv
det
svd
eig
qr
```

---

## Milestone 9 — Random

Deliver:

```text
RNG
seed
normal
uniform
rand
randn
randint
choice
shuffle
permutation
```

---

## Milestone 10 — FFT

Deliver:

```text
fft
ifft
fft2
ifft2
fftn
ifftn
rfft
irfft
```

---

## Milestone 11 — Optimization

Deliver:

```text
SIMD
thread pool
BLAS
LAPACK
memory optimization
kernel optimization
```

---

## Milestone 12 — Compatibility

Build:

```text
large NumPy compatibility suite
API coverage tool
property testing
fuzzing
```

---

## Milestone 13 — Packaging

Deliver:

```text
prebuilt binaries
platform packages
npm installation
source build
```

---

## Milestone 14 — Production release

Deliver:

```text
documentation
examples
benchmarks
security policy
contribution guide
release automation
npm publish
GitHub release
```

---

# 87. Definition of Done

A feature is NOT complete when the code compiles.

A feature is complete only when:

```text
[ ] implementation exists
[ ] TypeScript API exists
[ ] C++ implementation exists
[ ] unit tests exist
[ ] edge-case tests exist
[ ] NumPy comparison exists
[ ] documentation exists
[ ] example exists
[ ] benchmark exists if performance-sensitive
[ ] CI passes
```

---

# 88. Final Quality Gates

Before version 1.0:

```text
API coverage:
target ≥ 90% of selected NumPy surface

Semantic compatibility:
target ≥ 99% for supported APIs

Differential tests:
millions of cases

Memory:
no known memory safety violations

Platforms:
Linux x64
macOS ARM64
macOS x64
Windows x64

Installation:
npm install works without local compiler

Performance:
major vectorized operations approach native NumPy performance
```

Do not claim 100% compatibility until it has actually been demonstrated.

---

# 89. Example Final Usage

```ts
import np from "nativpy";

const X = np.random.normal([10_000, 100]);

const weights = np.random.normal([100, 10]);

const bias = np.zeros([10]);

const predictions = np.add(
  np.matmul(X, weights),
  bias
);

console.log(predictions.shape);
```

Expected:

```text
[10000, 10]
```

The computation should execute primarily inside the native C++ layer.

---

# 90. Future Architecture

The architecture should allow:

```text
nativpy
   │
   ├── CPU backend
   │     ├── scalar
   │     ├── SIMD
   │     ├── BLAS
   │     └── multithreading
   │
   ├── GPU backend
   │     ├── CUDA
   │     ├── WebGPU
   │     └── Metal
   │
   └── WASM backend
         └── WASM SIMD
```

Future projects:

```text
@nativpy/autograd
@nativpy/nn
@nativpy/vision
@nativpy/transformers
```

---

# 91. Final Agent Instruction

You are the primary software engineer responsible for implementing `nativpy`.

Do not attempt to implement the entire project in one uncontrolled change.

Work milestone by milestone.

Before starting each milestone:

1. Read the architecture.
2. Inspect existing code.
3. Identify dependencies.
4. Create a concrete implementation plan.
5. Implement the smallest correct increment.
6. Compile.
7. Test.
8. Compare against NumPy.
9. Benchmark where appropriate.
10. Document.
11. Commit.
12. Update progress.

Correctness has priority over performance.

Performance has priority over convenience.

NumPy behavior is the reference specification for supported functionality.

Never fake compatibility.

Never disable tests simply because the implementation currently fails them.

Never report a feature as complete unless it has been independently verified.

The ultimate goal is to create a serious, publishable, open-source numerical computing library for JavaScript/TypeScript with native C++ execution.

---

# 92. First Task

Do NOT start implementing the entire library.

First create:

```text
AGENTS.md
ARCHITECTURE.md
ROADMAP.md
COMPATIBILITY.md
PERFORMANCE.md
DECISIONS.md
```

Then initialize:

```text
TypeScript project
CMake project
Node-API native addon
Python NumPy reference environment
test infrastructure
benchmark infrastructure
CI
```

Then implement only:

```text
NDArray
MemoryBuffer
DType
Shape
Strides
Offset
```

Create the first differential tests.

The first successful milestone should prove:

```text
JavaScript
    ↓
TypeScript
    ↓
Node-API
    ↓
C++
    ↓
Native NDArray
    ↓
correct result
```

Only after this works reliably should the project proceed to arithmetic operations.

---

# 93. Success Criteria

The project succeeds when a developer can run:

```bash
npm install nativpy
```

and then:

```ts
import np from "nativpy";

const A = np.random.normal([2000, 2000]);
const B = np.random.normal([2000, 2000]);

const C = np.matmul(A, B);

console.log(C.shape);
```

without installing Python, NumPy, CMake, a C++ compiler, or other development dependencies.

The numerical computation should execute through the native backend.

The project should provide transparent documentation showing:

```text
API compatibility
numerical correctness
memory behavior
platform support
performance benchmarks
known differences
```

The project must be transparent about what is and is not NumPy-compatible.

---

# 94. Final Philosophy

nativpy is not intended to be:

> "NumPy syntax implemented in JavaScript."

It is intended to be:

> **"A native numerical computing engine exposed through a JavaScript/TypeScript API, with NumPy as its behavioral reference."**

The core principles are:

```text
Native computation
        +
NumPy-inspired/compatible semantics
        +
Zero unnecessary copies
        +
SIMD/BLAS/LAPACK
        +
Automated differential testing
        +
Transparent benchmarking
        +
Excellent TypeScript ergonomics
        =
nativpy
```
