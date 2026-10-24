# Coming from NumPy

numera keeps NumPy's names, semantics and results. Most of the differences come from JavaScript itself: it has no keyword arguments, no operator overloading and no slice syntax. This page lists the translation rules. The [compatibility page](/compatibility.html) lists every behavioural difference.

## Translation table

| NumPy (Python) | numera (JavaScript / TypeScript) |
| --- | --- |
| `import numpy as np` | `import np from "@cyfora/numera"` |
| `np.zeros((2, 3))` | `np.zeros([2, 3])` |
| `np.array([1, 2], dtype=np.float32)` | `np.array([1, 2], { dtype: "float32" })` or `{ dtype: np.float32 }` |
| `np.expand_dims(a, 0)` | `np.expandDims(a, 0)` |
| `np.linalg.matrix_rank(a)` | `np.linalg.matrixRank(a)` |
| `np.random.default_rng(42)` | `np.random.defaultRng(42)` |
| `a.sum(axis=0, keepdims=True)` | `a.sum({ axis: 0, keepdims: true })` |
| `a + b`, `a * 2`, `a ** 2` | `np.add(a, b)`, `np.multiply(a, 2)`, `np.power(a, 2)` |
| `a @ b` | `np.matmul(a, b)` or `a.dot(b)` |
| `a > 0` | `np.greater(a, 0)` |
| `a & b`, `~a` | `np.logicalAnd(a, b)` / `np.bitwiseAnd(a, b)`, `np.logicalNot(a)` / `np.invert(a)` |
| `a[1, 2]` | `a.get(1, 2)`, or `a.item(1, 2)` for a JS number |
| `a[0:2, ::2]` | `a.slice([[0, 2], [null, null, 2]])` |
| `a[..., 0]`, `a[None]` | `a.get(np.ellipsis, 0)`, `a.get(np.newaxis)` |
| `a[mask]` | `a.get(mask)` |
| `a[0:2] = 5` | `a.set([[0, 2]], 5)` |
| `a.shape`, `a.dtype`, `a.T` | `a.shape`, `a.dtype`, `a.T` (the same) |
| `a.tolist()` | `a.toArray()` (or `a.tolist()`) |
| `None` | `null` |
| `np.nan`, `np.inf` | `NaN`, `Infinity` (or `np.nan`, `np.inf`) |
| `1 + 2j` | `np.complex(1, 2)` |
| `with np.errstate(divide="ignore"):` | `np.errstate({ divide: "ignore" }, () => { ... })` |

## Naming

Function and method names use camelCase: `expand_dims` becomes `expandDims`, and `nan_to_num` becomes `nanToNum`. Option keys are camelCase as well (`fullMatrices`, `returnCounts`). Names that are already a single word are unchanged, for example `sum`, `matmul`, `linalg.svd` and `fft.rfft`. The [NumPy name index](/reference/numpy-index.html) lists the numera name of every NumPy function. Search for the Python name there or press <kbd>/</kbd>.

```js
np.expandDims(np.array([1, 2]), 0).shape;        // => [1, 2]
np.nanToNum([NaN, 1]);                 // => [0, 1]
np.linalg.matrixRank([[1, 2], [2, 4]]).item(); // => 1
```

## Keyword arguments

Positional arguments stay positional. Keyword arguments go into an options object, which is always the last argument:

```js
const a = np.array([[1, 2], [3, 4]]);
np.sum(a, { axis: 0 });                       // => [4, 6]
np.sum(a, { axis: 1, keepdims: true });       // => [[3], [7]]
np.std([1, 2, 3, 4], { ddof: 1 }).item();     // => 1.2909944487358056
np.concatenate([[1, 2], [3]]);                // => [1, 2, 3]
```

Some functions also accept a common option positionally, as NumPy does. For example, `np.concatenate(arrays, 1)` and `np.sort(a, 0)` both take the axis. The signature in the API reference shows which forms each function accepts.

## Operators

JavaScript cannot overload `+`, `*`, `>` or `@`, so every operator is a function call. These are NumPy's own ufuncs, so `np.add(a, b)` behaves exactly like `a + b` in NumPy:

```js
const x = np.array([1, 2, 3]);
np.add(x, 1);                   // => [2, 3, 4]
np.subtract(10, x);             // => [9, 8, 7]
np.multiply(x, x);              // => [1, 4, 9]
np.divide(x, 2);                // => [0.5, 1, 1.5]
np.floorDivide(x, 2);           // => [0, 1, 1]
np.mod(x, 2);                   // => [1, 0, 1]
np.power(x, 2);                 // => [1, 4, 9]
np.equal(x, 2);                 // => [false, true, false]
np.matmul([[1, 2], [3, 4]], [[1], [1]]); // => [[3], [7]]
```

Every ufunc accepts NumPy's `out`, `where`, `dtype` and `casting` options. Binary ufuncs also have the `reduce`, `accumulate`, `reduceat`, `outer` and `at` methods. See [Universal functions](/reference/ufuncs.html).

## Return values

- **Array results are `NDArray`s.** Use `toArray()` for nested JS arrays, `toTypedArray()` for a flat typed array, or `item()` for a single element.
- **Scalar results are 0-d arrays.** In NumPy, `np.sum(a)` returns a NumPy scalar. In numera it returns a 0-d `NDArray`, so call `.item()` to get a JS number. A full integer index such as `a.get(1, 2)` also gives a 0-d array.
- **Several results become an object or an array.** NumPy's named tuples become objects with the same field names, for example `np.linalg.eigh(a).eigenvalues` and `np.linalg.svd(a).S`. Plain tuples become JS arrays, for example `np.nonzero(a)`.

```js
np.sum([1, 2, 3]).item();                                  // => 6
np.array([[1, 2], [3, 4]]).get(1, 0).item();              // => 3
np.linalg.svd([[3, 0], [0, 4]]).S;                         // => [4, 3]
np.nonzero([0, 1, 0, 2]).map((i) => i.toArray());          // => [[1, 3]]
```

## Inputs

Most functions that take an `array_like` in NumPy accept an `NDArray`, a nested JS array of numbers, booleans or bigints, or a scalar.

> **Note**: In this release, a few shape functions accept only an `NDArray`: `reshape`, `transpose`, `squeeze`, `ravel`, `swapAxes`, `moveAxis`, `expandDims` and the `*Like` creators (`zerosLike`, `onesLike`, …). Wrap plain arrays with `np.array(...)` first, or use the method form, e.g. `np.array(data).reshape(2, 3)`.

JS numbers are *weak* scalars, like Python numbers under NEP 50: they don't widen the dtype of an array.

```js
np.add(np.array([1, 2], { dtype: "int8" }), 1).dtype.name;   // => "int8"
np.add(np.array([1, 2], { dtype: "int8" }), 1.5).dtype.name; // => "float64"
```

## Things that differ

The main differences are listed below. Each one is caused by JavaScript, and the [compatibility page](/compatibility.html) has the full list.

- A JS number that is an integer but too large for an exact float (such as `2**60`) is inferred as `float64`, not `int64`. Use a `bigint` or `{ dtype: "int64" }`.
- `int64` and `uint64` values come back from `toArray()` as JS numbers, which are exact only up to 2^53. Use `toTypedArray()` (a `BigInt64Array`) or `tolist()` when you need full precision. See [Data types](/guides/dtypes.html).
- Out-of-range integers raise `ValueError`, where NumPy raises `OverflowError`.
- Floating-point warnings are Node `RuntimeWarning`s, emitted with `process.emitWarning`.
