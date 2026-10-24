# Universal functions (ufunc)

A *universal function* (ufunc) operates on arrays element by element. It supports [broadcasting](/guides/broadcasting.html), type promotion and a set of standard options. Like NumPy's, numera's ufuncs are callable objects: `np.add(a, b)` computes, and `np.add.reduce(a)` reduces with the same operation.

```js
np.add([1, 2, 3], 10);              // => [11, 12, 13]
np.sqrt([1, 4, 9]);                 // => [1, 2, 3]
np.add.reduce([1, 2, 3]).item();    // => 6
```

## Broadcasting

Inputs of different shapes are broadcast against each other using NumPy's rules: dimensions are compared from the last one, and each pair must be equal or contain a 1. Incompatible shapes raise `BroadcastError`.

```js
np.add(np.ones([3, 1]), np.arange(4)).shape;  // => [3, 4]
np.multiply([[1], [2]], [10, 20]);            // => [[10, 20], [20, 40]]
```

## Output type determination

The result dtype comes from NumPy's promotion of the input dtypes. JS numbers are *weak* scalars (NEP 50), so they don't widen an array's dtype. See [Data types](/guides/dtypes.html).

```js
np.add(np.array([1], { dtype: "int8" }), 1).dtype.name;     // => "int8"
np.add(np.array([1], { dtype: "int8" }), 1.5).dtype.name;   // => "float64"
np.divide([1, 2], 2).dtype.name;                            // => "float64"
np.add([1, 2], [3, 4], { dtype: "float32" }).dtype.name;    // => "float32"
```

## Optional arguments

Every ufunc accepts an options object as its last argument:

| Option | NumPy | Meaning |
| --- | --- | --- |
| `out` | `out=` | write the result into this array and return it |
| `where` | `where=` | boolean mask: compute only where `true` |
| `dtype` | `dtype=` | loop dtype: inputs are cast to it and the result has it |
| `casting` | `casting=` | casting rule for inputs and `out` (default `"same_kind"`) |
| `order` | `order=` | memory layout of a new result: `"K"` (default), `"C"`, `"F"` or `"A"` |

```js
const o = np.zeros(3);
np.multiply([1, 2, 3], 2, { out: o }) === o;  // => true
o;                                            // => [2, 4, 6]

const w = np.full([3], -1.0, { dtype: "float64" });
np.sqrt([1, 4, 9], { out: w, where: [true, false, true] });
w;                                            // => [1, -1, 3]
```

Without `out`, positions where `where` is `false` are zero-filled. NumPy leaves them uninitialised. Writing a result into an `out` array of a lower kind is checked with `casting`:

```js
let msg;
try { np.add(np.array([1.5]), 1, { out: np.zeros(1, { dtype: "int32" }) }); } catch (e) { msg = e.message; }
msg; // => "Cannot cast ufunc 'add' output from float64 to int32 with casting rule 'same_kind'"
```

## Methods

Binary ufuncs (two inputs, one output) have NumPy's five methods:

| Method | NumPy | Meaning |
| --- | --- | --- |
| `reduce(a, { axis, dtype, out, keepdims, initial, where })` | `ufunc.reduce` | combine along an axis |
| `accumulate(a, { axis, dtype, out })` | `ufunc.accumulate` | running reduction |
| `reduceat(a, indices, { axis, dtype, out })` | `ufunc.reduceat` | reductions over slices |
| `outer(a, b)` | `ufunc.outer` | apply to every pair |
| `at(a, indices, b)` | `ufunc.at` | unbuffered in-place operation |

Unary ufuncs have `at(a, indices)`.

```js
np.maximum.reduce([[1, 5], [4, 2]], { axis: 1 }); // => [5, 4]
np.add.accumulate([1, 2, 3]);                     // => [1, 3, 6]
np.add.reduceat([1, 2, 3, 4], [0, 2]);            // => [3, 7]
np.multiply.outer([1, 2, 3], [1, 10]);            // => [[1, 10], [2, 20], [3, 30]]

const c = np.array([1, 2, 3, 4]);
np.add.at(c, [0, 0, 1], 10);                      // repeated indices accumulate
c;                                                // => [21, 12, 3, 4]
```

## Floating-point errors

Division by zero, overflow, underflow and invalid operations are handled according to `np.seterr` / `np.errstate`. By default they emit a `RuntimeWarning`, and they can be ignored or raised instead. See [Errors and floating-point](/guides/errors.html).

```js
np.errstate({ divide: "ignore" }, () => np.divide([1, -1], [0, 0]).toArray().map(String)); // => ["Infinity", "-Infinity"]
```
