# Indexing and slicing

NumPy's `a[...]` syntax has no equivalent in JavaScript. numera uses three methods instead, and they cover the same cases:

| NumPy | numera | Result |
| --- | --- | --- |
| `a[i, j, ...]` | `a.get(i, j, ...)` | view (basic) or copy (advanced) |
| `a[0:2, ::2]` | `a.slice([[0, 2], [null, null, 2]])` | view |
| `a[index] = v` | `a.set(index, v)` | writes in place |

## Integers

`get` takes one index per axis. Negative indices count from the end. A full integer index gives a 0-d array, so call `.item()` (or `a.item(i, j)`) to get the JS value:

```js
const b = np.arange(12).reshape(3, 4);
b.get(1);               // => [4, 5, 6, 7]
b.get(1, -1).item();    // => 7
b.item(2, 0);           // => 8
```

Out-of-range indices raise `np.IndexError`:

```js
let name;
try { np.array([1, 2, 3]).get(5); } catch (e) { name = e.constructor.name; }
name; // => "IndexError"
```

## Slices

A slice is a tuple `[start, stop, step]`, and `null` stands for an omitted part, as in `a[start:stop:step]`. You can write a slice inside `get` (as one of the per-axis indices) or use `slice`, which takes a single list:

- a flat list of numbers or `null` of length 1–3 is **one** slice: `a.slice([1, null])` is `a[1:]`;
- otherwise the list holds one spec per axis: `a.slice([[0, 2], [null, null, 2]])` is `a[0:2, ::2]`.

```js
const x = np.arange(10);
x.slice([2, 8, 3]);                 // => [2, 5]
x.slice([null, null, -1]).get(0).item(); // => 9
x.slice([7, null]);                 // => [7, 8, 9]

const b = np.arange(12).reshape(3, 4);
b.slice([[0, 2], [null, null, 2]]); // => [[0, 2], [4, 6]]
b.slice([null, [1, 3]]);            // => [[1, 2], [5, 6], [9, 10]]
b.get([0, 2], 1);                   // => [1, 5]
```

Slices return **views**, so they share memory with the original array:

```js
const v = np.arange(4);
v.slice([1, 3]).set([], [9, 9]);
v;                                   // => [0, 9, 9, 3]
```

## Ellipsis and newaxis

`np.ellipsis` (NumPy's `...`) stands for "all remaining axes", and `np.newaxis` (NumPy's `None`) inserts a length-1 axis:

```js
const b = np.arange(12).reshape(3, 4);
b.get(np.ellipsis, -1);              // => [3, 7, 11]
b.get(np.newaxis).shape;             // => [1, 3, 4]
b.get(np.ellipsis, np.newaxis).shape; // => [3, 4, 1]
```

## Integer-array (fancy) indexing

An integer `NDArray` used as an index picks elements by position. The result is a **copy**:

```js
np.array([10, 20, 30, 40]).get(np.array([3, 1]));   // => [40, 20]
const b = np.arange(12).reshape(3, 4);
b.get(np.array([2, 0]));                             // => [[8, 9, 10, 11], [0, 1, 2, 3]]
np.take([10, 20, 30], [2, 0]);                       // => [30, 10]
```

> **Note**: In `get`, a plain JS array such as `[0, 2]` is a *slice tuple*, not a list of indices. Wrap the indices in `np.array([...])` for fancy indexing, or use `np.take`.

## Boolean masks

A boolean array of the same shape selects the elements where it is `true`, as a flat copy:

```js
const a = np.arange(12).reshape(3, 4);
a.get(np.greater(a, 8));                 // => [9, 10, 11]
const m = np.array([1, -2, 3, -4]);
m.get(np.array([true, false, true, false])); // => [1, 3]
```

## Assignment

`set(index, value)` takes the same index forms as `get`. The value broadcasts to the selection and is cast to the array's dtype:

```js
const c = np.zeros(4);
c.set([[0, 2]], [7, 8]);       // c[0:2] = [7, 8]
c.set(3, 1);                   // c[3] = 1
c;                             // => [7, 8, 0, 1]

const d = np.zeros([2, 3]);
d.set(0, 5);                   // d[0] = 5
d;                             // => [[5, 5, 5], [0, 0, 0]]

const m = np.array([1, -2, 3, -4]);
m.set(np.less(m, 0), 0);       // m[m < 0] = 0
m;                             // => [1, 0, 3, 0]
```

Writing to a read-only view, such as the result of `broadcastTo`, raises `ValueError`.

## Other indexing routines

| Function | Purpose |
| --- | --- |
| `np.where(cond, x, y)` | pick from `x` or `y` element-wise |
| `np.nonzero(a)`, `np.argwhere(a)` | positions of non-zero elements |
| `np.take`, `np.takeAlongAxis`, `np.put`, `np.putmask` | index by position |
| `np.choose`, `np.select`, `np.compress`, `np.extract` | select by condition |
| `np.diagonal`, `np.trace` | diagonals |
| `np.ravelMultiIndex`, `np.unravelIndex` | convert flat and multi-dimensional indices |

```js
np.where(np.greater([1, 5, 2], 1), [1, 5, 2], -1); // => [-1, 5, 2]
np.argwhere([[0, 1], [1, 0]]);                     // => [[0, 1], [1, 0]]
```

See the [indexing reference](/reference/indexing.html) for all of them.
