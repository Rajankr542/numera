# Quickstart

This page is a short tour of numera {{version}}. It assumes you have [installed](/guides/installation.html) the package.

```js norun
import np from "@cyfora/numera";
```

## Create arrays

```js
np.array([[1, 2], [3, 4]]);          // => [[1, 2], [3, 4]]
np.zeros([2, 3]);                    // => [[0, 0, 0], [0, 0, 0]]
np.ones([3], { dtype: "int32" });    // => [1, 1, 1]
np.full([2, 2], 7);                  // => [[7, 7], [7, 7]]
np.arange(0, 10, 2);                 // => [0, 2, 4, 6, 8]
np.linspace(0, 1, 5);                // => [0, 0.25, 0.5, 0.75, 1]
np.eye(2);                           // => [[1, 0], [0, 1]]
np.arange(6).reshape(2, 3);          // => [[0, 1, 2], [3, 4, 5]]
```

Shapes are arrays of numbers: `np.zeros([2, 3])` corresponds to NumPy's `np.zeros((2, 3))`. Options such as `dtype` go in a trailing object.

## Inspect an array

```js
const a = np.array([[1, 2, 3], [4, 5, 6]]);
a.shape;        // => [2, 3]
a.ndim;         // => 2
a.size;         // => 6
a.dtype.name;   // => "int64"
a.toArray();    // => [[1, 2, 3], [4, 5, 6]]
String(a);      // => "array([[1,2,3],[4,5,6]], dtype=int64)"
```

> **Note**: In 1.0.1, `toString()` prints a compact form that always includes the dtype. NumPy-style `repr` formatting arrived in 1.0.2.

## Element-wise math

JavaScript has no operator overloading, so arithmetic is done with functions. They broadcast and follow NumPy's type promotion.

```js
const x = np.array([1, 4, 9]);
np.sqrt(x);                       // => [1, 2, 3]
np.add(x, 1);                     // => [2, 5, 10]
np.multiply(x, [1, 0, -1]);       // => [1, 0, -9]
np.power(2, [1, 2, 3]);           // => [2, 4, 8]
np.divide(x, 2);                  // => [0.5, 2, 4.5]
np.mod(x, 2);                     // => [1, 0, 1]
```

## Broadcasting

```js
const col = np.array([[0], [10], [20]]);   // shape [3, 1]
const row = np.array([1, 2, 3]);           // shape [3]
np.add(col, row);  // => [[1, 2, 3], [11, 12, 13], [21, 22, 23]]
```

## Reductions

```js
const m = np.array([[1, 2, 3], [4, 5, 6]]);
m.sum().item();                       // => 21
np.sum(m, { axis: 0 });               // => [5, 7, 9]
m.max({ axis: 1 });                   // => [3, 6]
np.mean(m, { axis: 1, keepdims: true }).shape; // => [2, 1]
np.std([1, 2, 3, 4]).item();          // => 1.118033988749895
np.argmax([1, 5, 3]).item();          // => 1
```

Reductions return an `NDArray`, even when the result is a single value. Call `.item()` to get a JavaScript number.

## Indexing and slicing

```js
const b = np.arange(12).reshape(3, 4);
b.get(1);                             // => [4, 5, 6, 7]
b.get(1, 2).item();                   // => 6
b.slice([[0, 2], [null, null, 2]]);   // => [[0, 2], [4, 6]]

const c = np.zeros(4);
c.set([[0, 2]], [7, 8]);              // c[0:2] = [7, 8]
c;                                    // => [7, 8, 0, 0]
```

See [Indexing and slicing](/guides/indexing.html).

## Linear algebra and FFT

```js
const A = np.array([[3, 1], [1, 2]]);
np.matmul(A, A);                      // => [[10, 5], [5, 5]]
np.linalg.det(A).item();              // => 5
np.linalg.solve(A, [9, 8]);           // => [2, 3]
np.linalg.eigh([[2, 1], [1, 2]]).eigenvalues; // => [1, 3]
np.fft.irfft(np.fft.rfft([1, 2, 3, 4])); // => [1, 2, 3, 4]
```

## Random numbers

```js
const rng = np.random.defaultRng(42);
rng.random(3);       // => [0.7739560485559633, 0.4388784397520523, 0.8585979199113825]
rng.integers(0, 10, [5]); // => [0, 6, 2, 0, 5]
```

These are the same numbers that `np.random.default_rng(42)` produces in NumPy.
