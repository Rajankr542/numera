# Introduction

## What numera is

numera is an n-dimensional array library for Node.js. It follows NumPy's API and behaviour closely, so code and knowledge carry over: the same dtypes and promotion rules, the same broadcasting, the same results from `np.linalg`, `np.fft` and seeded random generators. The numerical loops run in a C++20 core loaded as a Node-API addon. The TypeScript layer on top validates arguments and translates errors, and it gives every function a typed signature.

<div class="features">
<div><h3>NumPy semantics</h3><p>{{implemented}} of the {{total}} NumPy {{numpy}} names numera tracks are implemented ({{coverage}}%). Behaviour is checked against real NumPy by thousands of differential test cases.</p></div>
<div><h3>Native speed</h3><p>Array data lives in native memory, and loops run in C++ instead of per-element JavaScript. On macOS, linear algebra uses Apple Accelerate.</p></div>
<div><h3>Typed API</h3><p>Every function has TypeScript declarations. Keyword arguments become typed option objects, and errors are typed classes such as <code>ShapeError</code> and <code>LinAlgError</code>.</p></div>
<div><h3>Reproducible random numbers</h3><p><code>np.random.defaultRng(seed)</code> and <code>np.random.seed(seed)</code> produce the same numbers as NumPy, bit for bit.</p></div>
<div><h3>Broad coverage</h3><p>{{ufuncs}} ufuncs with <code>out</code>, <code>where</code> and ufunc methods, plus <code>linalg</code>, <code>fft</code>, <code>random</code>, <code>ma</code>, <code>strings</code>, <code>polynomial</code>, <code>emath</code>, <code>testing</code>, file I/O and datetimes.</p></div>
<div><h3>No build step</h3><p>Prebuilt binaries ship for macOS and Linux on x64 and arm64. You do not need a compiler, CMake or Python.</p></div>
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

In these docs, a `// =>` comment shows the value of the expression. For an `NDArray`, that is `.toArray()`. The test suite runs every example in the API reference and every guide example.

## Where to go next

- [Installation](/guides/installation.html): supported platforms and package managers.
- [Quickstart](/guides/quickstart.html): a short tour of the main features.
- [Coming from NumPy](/guides/numpy-migration.html): a translation table from Python to numera.
- [API reference](/reference/index.html): every function with its parameters, return value and an example.
- [NumPy name index](/reference/numpy-index.html): look up any NumPy name and find its numera equivalent.
- [NumPy compatibility](/compatibility.html): what is verified, and every documented difference.
