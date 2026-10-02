# Broadcasting

Broadcasting describes how element-wise operations treat arrays of different shapes. numera follows NumPy's rules exactly.

## The rule

Compare the shapes starting from the **last** dimension. Two dimensions are compatible when they are equal or one of them is 1. Missing leading dimensions count as 1.

| A | B | Result |
| --- | --- | --- |
| `[3, 4]` | `[4]` | `[3, 4]` |
| `[3, 1]` | `[4]` | `[3, 4]` |
| `[8, 1, 6, 1]` | `[7, 1, 5]` | `[8, 7, 6, 5]` |
| `[2, 3]` | `[3, 2]` | error: `BroadcastError` |

```js
np.broadcastShapes([3, 1], [4]);             // => [3, 4]
np.broadcastShapes([8, 1, 6, 1], [7, 1, 5]); // => [8, 7, 6, 5]
np.multiply([[1, 2], [3, 4]], 10);           // => [[10, 20], [30, 40]]
```

## Row and column vectors

```js
const col = np.arange(3).reshape(3, 1);
np.add(col, np.arange(2));                // => [[0, 1], [1, 2], [2, 3]]
np.add(np.arange(3).get(np.ellipsis, np.newaxis), [10, 20]); // => [[10, 20], [11, 21], [12, 22]]
```

## Centring data

```js
const m = np.array([[1, 2], [3, 4]]);
np.subtract(m, np.mean(m, { axis: 0 }));                   // => [[-1, -1], [1, 1]]
np.subtract(m, np.mean(m, { axis: 1, keepdims: true }));   // => [[-0.5, 0.5], [-0.5, 0.5]]
```

## Errors

```js
let msg;
try { np.add(np.ones([2, 3]), np.ones([3, 2])); } catch (e) { msg = e.message; }
msg; // => "operands could not be broadcast together with shapes (2, 3) (3, 2)"
```

## Explicit broadcasting

```js
np.broadcastTo([1, 2, 3], [2, 3]);        // => [[1, 2, 3], [1, 2, 3]]
np.broadcastTo([1, 2, 3], [2, 3]).flags.writeable; // => false
np.matmul(np.ones([5, 2, 3]), np.ones([3, 4])).shape; // => [5, 2, 4]
```
