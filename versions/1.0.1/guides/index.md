# Introduction

> **Note**: This is the documentation for numera **{{version}}**, an early release that implements {{implemented}} of the {{total}} tracked NumPy names ({{coverage}}%). Use the version menu to switch to a newer release.

## What numera is

numera is an n-dimensional array library for Node.js. It follows NumPy's API and behaviour: the same dtypes and promotion rules, the same broadcasting, and the same results from `np.linalg`, `np.fft` and seeded random generators. The numerical loops run in a C++20 core loaded as a Node-API addon. The TypeScript layer on top validates arguments, translates errors and gives every function a typed signature.

<div class="features">
<div><h3>NumPy semantics</h3><p>Dtype inference, type promotion, broadcasting, views and strides all follow NumPy. Differential tests check them against real NumPy.</p></div>
<div><h3>Native core</h3><p>Array data lives in native memory, and loops run in C++. On macOS, linear algebra uses Apple Accelerate.</p></div>
<div><h3>Typed API</h3><p>TypeScript declarations ship with the package. Errors are typed classes such as <code>ShapeError</code> and <code>LinAlgError</code>.</p></div>
<div><h3>Reproducible random numbers</h3><p><code>np.random.defaultRng(seed)</code> and <code>np.random.seed(seed)</code> produce the same numbers as NumPy.</p></div>
</div>

## A first look

```js
import np from "@cyfora/numera";

const a = np.array([[1, 2, 3], [4, 5, 6]]);
a.shape;                        // => [2, 3]
a.dtype.name;                   // => "int64"
np.add(a, 10);                  // => [[11, 12, 13], [14, 15, 16]]
a.sum({ axis: 0 });             // => [5, 7, 9]
np.mean(a, { axis: 1 });        // => [2, 5]
np.linalg.solve([[3, 1], [1, 2]], [9, 8]); // => [2, 3]
```

In these docs, a `// =>` comment shows the value of the expression. For an `NDArray`, that is `.toArray()`. The examples in these guides were run against the published {{version}} package.

## What 1.0.1 includes

- Array creation, `reshape`/`transpose`/`squeeze` and the other basic shape functions, and broadcasting.
- 13 element-wise functions: `add`, `subtract`, `multiply`, `divide`, `floorDivide`, `mod`, `power`, `negative`, `abs`, `sqrt`, `exp`, `log` and `matmul`.
- Reductions: `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `argmin`, `argmax`.
- Indexing with `get`, `slice` and `set`, plus `where`, `nonzero` and `take`.
- `np.linalg`, `np.fft` and `np.random`.

Comparison and logical functions, trigonometry, ufunc options (`out`, `where`) and methods (`reduce`, `outer`, …), sorting, statistics and the other NumPy modules came in later releases. See the [changelog](/changelog.html).

## Where to go next

- [Installation](/guides/installation.html)
- [Quickstart](/guides/quickstart.html)
- [Coming from NumPy](/guides/numpy-migration.html)
- [API reference](/reference/index.html) and the [NumPy name index](/reference/numpy-index.html)
- [NumPy compatibility](/compatibility.html)
