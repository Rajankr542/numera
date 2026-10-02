# Indexing and slicing

numera replaces NumPy's `a[...]` syntax with three methods:

| NumPy | numera | Result |
| --- | --- | --- |
| `a[i, j, ...]` | `a.get(i, j, ...)` | view (basic) or copy (advanced) |
| `a[0:2, ::2]` | `a.slice([[0, 2], [null, null, 2]])` | view |
| `a[index] = v` | `a.set(index, v)` | writes in place |

## Integers

```js
const b = np.arange(12).reshape(3, 4);
b.get(1);               // => [4, 5, 6, 7]
b.get(1, -1).item();    // => 7
b.item(2, 0);           // => 8
```

## Slices

A slice is a tuple `[start, stop, step]`, and `null` stands for an omitted part. A flat list of length 1–3 passed to `slice` is one slice. Otherwise the list holds one spec per axis.

```js
const x = np.arange(10);
x.slice([2, 8, 3]);                 // => [2, 5]
x.slice([7, null]);                 // => [7, 8, 9]

const b = np.arange(12).reshape(3, 4);
b.slice([[0, 2], [null, null, 2]]); // => [[0, 2], [4, 6]]
b.slice([null, [1, 3]]);            // => [[1, 2], [5, 6], [9, 10]]
b.get([0, 2], 1);                   // => [1, 5]
```

Slices are **views**:

```js
const v = np.arange(4);
v.slice([1, 3]).set([], [9, 9]);
v;                                   // => [0, 9, 9, 3]
```

## Ellipsis and newaxis

```js
const b = np.arange(12).reshape(3, 4);
b.get(np.ellipsis, -1);              // => [3, 7, 11]
b.get(np.newaxis).shape;             // => [1, 3, 4]
b.get(np.ellipsis, np.newaxis).shape; // => [3, 4, 1]
```

## Integer arrays and boolean masks

Wrap the indices in `np.array(...)`. A plain JS array inside `get` is a slice tuple.

```js
np.array([10, 20, 30, 40]).get(np.array([3, 1]));   // => [40, 20]
np.take([10, 20, 30], [2, 0]);                       // => [30, 10]
np.array([1, -2, 3, -4]).get(np.array([true, false, true, false])); // => [1, 3]
```

## Assignment

```js
const c = np.zeros(4);
c.set([[0, 2]], [7, 8]);       // c[0:2] = [7, 8]
c.set(3, 1);                   // c[3] = 1
c;                             // => [7, 8, 0, 1]

const d = np.zeros([2, 3]);
d.set(0, 5);                   // d[0] = 5
d;                             // => [[5, 5, 5], [0, 0, 0]]
```

## where, nonzero, take

```js
np.where([true, false], [1, 2], [3, 4]);           // => [1, 4]
np.nonzero([0, 1, 0, 2]).map((i) => i.toArray());  // => [[1, 3]]
```
