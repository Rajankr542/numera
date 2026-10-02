# Coming from NumPy

numera keeps NumPy's names, semantics and results. Most of the differences come from JavaScript itself: it has no keyword arguments, no operator overloading and no slice syntax.

## Translation table

| NumPy (Python) | numera {{version}} |
| --- | --- |
| `import numpy as np` | `import np from "@cyfora/numera"` |
| `np.zeros((2, 3))` | `np.zeros([2, 3])` |
| `np.array([1, 2], dtype=np.float32)` | `np.array([1, 2], { dtype: "float32" })` or `{ dtype: np.float32 }` |
| `np.expand_dims(a, 0)` | `np.expandDims(a, 0)` |
| `np.random.default_rng(42)` | `np.random.defaultRng(42)` |
| `a.sum(axis=0, keepdims=True)` | `a.sum({ axis: 0, keepdims: true })` |
| `a + b`, `a * 2`, `a ** 2` | `np.add(a, b)`, `np.multiply(a, 2)`, `np.power(a, 2)` |
| `a @ b` | `np.matmul(a, b)` or `np.dot(a, b)` |
| `a[1, 2]` | `a.get(1, 2)`, or `a.item(1, 2)` for a JS number |
| `a[0:2, ::2]` | `a.slice([[0, 2], [null, null, 2]])` |
| `a[..., 0]`, `a[None]` | `a.get(np.ellipsis, 0)`, `a.get(np.newaxis)` |
| `a[mask]` | `a.get(mask)` |
| `a[0:2] = 5` | `a.set([[0, 2]], 5)` |
| `a.shape`, `a.dtype`, `a.T` | the same |
| `a.tolist()` | `a.toArray()` |
| `None` | `null` |

## Naming

Function names use camelCase: `expand_dims` becomes `expandDims`, and `swapaxes` becomes `swapAxes`. Names that are a single word are unchanged. The [NumPy name index](/reference/numpy-index.html) lists every NumPy name and shows which ones exist in {{version}}.

## Keyword arguments

Positional arguments stay positional. Keyword arguments go into an options object, which is always the last argument:

```js
const a = np.array([[1, 2], [3, 4]]);
np.sum(a, { axis: 0 });                       // => [4, 6]
np.sum(a, { axis: 1, keepdims: true });       // => [[3], [7]]
np.std([1, 2, 3, 4], { ddof: 1 }).item();     // => 1.2909944487358056
```

## Operators

Every operator is a function call:

```js
const x = np.array([1, 2, 3]);
np.add(x, 1);                   // => [2, 3, 4]
np.subtract(10, x);             // => [9, 8, 7]
np.multiply(x, x);              // => [1, 4, 9]
np.divide(x, 2);                // => [0.5, 1, 1.5]
np.floorDivide(x, 2);           // => [0, 1, 1]
np.mod(x, 2);                   // => [1, 0, 1]
np.power(x, 2);                 // => [1, 4, 9]
np.matmul([[1, 2], [3, 4]], [[1], [1]]); // => [[3], [7]]
```

> **Note**: In 1.0.1, comparisons (`np.greater`, `np.equal`, …), logical functions and ufunc options such as `out` and `where` are not available yet. They arrived in 1.0.2.

## Return values

- **Array results are `NDArray`s.** Use `toArray()` for nested JS arrays, `toTypedArray()` for a flat typed array, or `item()` for a single element.
- **Scalar results are 0-d arrays.** `np.sum(a)` returns a 0-d `NDArray`, so call `.item()` to get a JS number.
- **Several results become an object or an array.** NumPy's named tuples become objects with the same field names, for example `np.linalg.eigh(a).eigenvalues` and `np.linalg.svd(a).S`. Plain tuples become JS arrays, for example `np.nonzero(a)`.

```js
np.sum([1, 2, 3]).item();                                  // => 6
np.array([[1, 2], [3, 4]]).item(1, 0);                     // => 3
np.linalg.svd([[3, 0], [0, 4]]).S;                         // => [4, 3]
np.nonzero([0, 1, 0, 2]).map((i) => i.toArray());          // => [[1, 3]]
```

## Inputs

Element-wise functions, reductions and `np.linalg` accept an `NDArray`, a nested JS array, or a scalar. Shape functions (`reshape`, `transpose`, `squeeze`, `expandDims`, `swapAxes`, `moveAxis`, `ravel`) and the `*Like` creators need an `NDArray`: wrap plain arrays with `np.array(...)` first.

JS numbers are *weak* scalars, like Python numbers under NEP 50: they don't widen the dtype of an array.

```js
np.add(np.array([1, 2], { dtype: "int8" }), 1).dtype.name;   // => "int8"
np.add(np.array([1, 2], { dtype: "int8" }), 1.5).dtype.name; // => "float64"
```
