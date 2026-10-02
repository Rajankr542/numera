# Performance and memory

## How numera runs

Array data lives in native memory that the C++ core allocates. A call such as `np.add(a, b)` does three things: the TypeScript layer validates the arguments, the C++ core runs the loop over native memory, and a new `NDArray` wrapper is returned. Elements never pass through JavaScript one by one. Only `toArray()`, `toTypedArray()` and `item()` copy data back into JS values.

This has two consequences:

- **Large arrays benefit.** Each call has a fixed cost (argument checks plus the Node-API boundary), and the loops run in C++. That fixed cost is small next to the work on a million elements.
- **Small arrays and many tiny calls don't.** For arrays of a few elements, the per-call cost dominates. Avoid writing per-element loops in JavaScript that call numera once per element. Express the computation as whole-array operations instead.

```js
const x = np.linspace(0, 1, 5);
// One call over the whole array, instead of a JS loop over elements:
np.sum(np.multiply(x, x)).item(); // => 1.875
```

## Benchmarks

The repository has a benchmark suite that runs the same cases on numera and NumPy (`pnpm bench`). The numbers come from single local runs on one machine (Apple M2 Pro, macOS, Node 22.7, NumPy 2.5.3), so treat them as indications rather than general claims. Across the 170 comparable cases in the last recorded full-suite run, numera was **slower than NumPy overall**: 60 cases were faster and 110 slower. In those runs:

- contiguous float reductions such as `sum` and `mean` on 10^6 elements were faster than NumPy;
- element-wise ufuncs on 10^6 elements were slower than NumPy, though compute-heavy ones such as `exp` were close to parity;
- small (10^3) arrays and view/slicing operations were clearly slower, because the fixed per-call cost dominates.

The full tables, including the method and every case, are in `PERFORMANCE.md` in the [repository](https://github.com/Rajankr542/numera).

## Linear algebra backends

On macOS, `np.linalg` and large float matrix products use Apple Accelerate. On Linux they use numera's portable C++ implementation, which is correct but not tuned like an optimized BLAS. `np.linalg.backend()` reports the active backend.

## Memory

Native buffers are released when the garbage collector frees the `NDArray` that owns them. Views keep their base array alive. You don't need to free arrays by hand.

Because native memory is invisible to V8's heap accounting, a program that creates many large temporary arrays in a tight loop can hold more memory than the JS heap size suggests until a collection runs. Writing into an existing array with `out` avoids allocating a new result each time:

```js
const acc = np.zeros(3);
for (let i = 0; i < 4; i++) np.add(acc, [1, 2, 3], { out: acc });
acc; // => [4, 8, 12]
```

`np.memoryStats()` reports the number of live native buffers and their total size in bytes. It is meant for debugging leaks:

```js
const s = np.memoryStats();
typeof s.buffers; // => "number"
typeof s.bytes;   // => "number"
```
