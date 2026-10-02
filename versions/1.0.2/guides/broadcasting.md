# Broadcasting

Broadcasting describes how element-wise operations treat arrays of different shapes. numera follows NumPy's rules exactly. The smaller array is "stretched" over the larger one without copying any data.

## The rule

Compare the two shapes starting from the **last** dimension. Two dimensions are compatible when they are equal or one of them is 1. Missing leading dimensions count as 1. The result takes the larger size in each dimension.

| A | B | Result |
| --- | --- | --- |
| `[3, 4]` | `[4]` | `[3, 4]` |
| `[3, 1]` | `[4]` | `[3, 4]` |
| `[8, 1, 6, 1]` | `[7, 1, 5]` | `[8, 7, 6, 5]` |
| `[2, 3]` | `[3, 2]` | error: `BroadcastError` |

```js
np.broadcastShapes([3, 1], [4]);           // => [3, 4]
np.broadcastShapes([8, 1, 6, 1], [7, 1, 5]); // => [8, 7, 6, 5]
```

## Scalars

A scalar broadcasts against any shape:

```js
np.multiply([[1, 2], [3, 4]], 10);        // => [[10, 20], [30, 40]]
```

## Row and column vectors

A shape `[n]` array acts as a row. Reshape it to `[n, 1]` to act as a column:

```js
const col = np.arange(3).reshape(3, 1);   // [[0], [1], [2]]
np.add(col, np.arange(2));                // => [[0, 1], [1, 2], [2, 3]]
np.add(np.arange(3).get(np.ellipsis, np.newaxis), [10, 20]); // => [[10, 20], [11, 21], [12, 22]]
```

## A practical example: centring data

Subtracting the column means from a matrix broadcasts a `[2]` vector over a `[2, 2]` matrix. Subtracting row means needs `keepdims: true`, so the means keep a length-1 axis:

```js
const m = np.array([[1, 2], [3, 4]]);
np.subtract(m, np.mean(m, { axis: 0 }));                   // => [[-1, -1], [1, 1]]
np.subtract(m, np.mean(m, { axis: 1, keepdims: true }));   // => [[-0.5, 0.5], [-0.5, 0.5]]
```

## Errors

Incompatible shapes raise `np.BroadcastError`, with NumPy's message:

```js
let msg;
try { np.add(np.ones([2, 3]), np.ones([3, 2])); } catch (e) { msg = e.message; }
msg; // => "operands could not be broadcast together with shapes (2, 3) (3, 2)"
```

## Explicit broadcasting

`np.broadcastTo` returns a read-only view with the target shape. `np.broadcastArrays` broadcasts several arrays against each other:

```js
np.broadcastTo([1, 2, 3], [2, 3]);        // => [[1, 2, 3], [1, 2, 3]]
np.broadcastTo([1, 2, 3], [2, 3]).flags.writeable; // => false
np.broadcastArrays([[1], [2]], [3, 4]).map((x) => x.shape); // => [[2, 2], [2, 2]]
```

Matrix functions such as `np.matmul` and every `np.linalg` routine broadcast their *batch* dimensions, that is, all axes except the last two:

```js
np.matmul(np.ones([5, 2, 3]), np.ones([3, 4])).shape; // => [5, 2, 4]
```
