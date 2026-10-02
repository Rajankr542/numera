# Array basics

The central object in numera is `NDArray`: an n-dimensional, fixed-size array of elements that all have the same dtype. It works like NumPy's `ndarray`: it has the same attributes, views share memory in the same way, and copies follow the same rules.

## Creating arrays

```js
np.array([[1, 2], [3, 4]]);                  // => [[1, 2], [3, 4]]
np.array([1, 2, 3], { dtype: "uint8" }).dtype.name; // => "uint8"
np.zeros([2, 2]);                            // => [[0, 0], [0, 0]]
np.empty([3]).shape;                         // => [3]
np.arange(5);                                // => [0, 1, 2, 3, 4]
np.linspace(0, 1, 3);                        // => [0, 0.5, 1]
np.identity(2);                              // => [[1, 0], [0, 1]]
np.zerosLike(np.array([[1, 2]]));            // => [[0, 0]]
np.fromTypedArray(new Float32Array([1, 2])).dtype.name; // => "float32"
```

## Attributes

```js
const a = np.zeros([2, 3], { dtype: "float32" });
a.shape;      // => [2, 3]
a.ndim;       // => 2
a.size;       // => 6
a.dtype.name; // => "float32"
a.itemSize;   // => 4
a.nbytes;     // => 24
a.strides;    // => [12, 4]
a.flags.cContiguous; // => true
```

`strides` are in bytes, as in NumPy.

## Converting to JavaScript

| Method | Returns |
| --- | --- |
| `a.toArray()` | nested JS arrays of numbers or booleans |
| `a.toTypedArray()` | a flat typed-array copy in C order (`Float64Array`, `Int32Array`, `BigInt64Array`, …) |
| `a.item(...index)` | one element as a JS number or boolean |
| `String(a)` | a compact text form, e.g. `array([1,2], dtype=int64)` |

```js
const b = np.array([[1, 2], [3, 4]], { dtype: "int16" });
b.toArray();                           // => [[1, 2], [3, 4]]
b.toTypedArray().constructor.name;     // => "Int16Array"
b.item(1, 0);                          // => 3
```

## Views and copies

Reshaping, transposing and basic slicing return **views**, which share memory with the original:

```js
const a = np.arange(6);
const v = a.reshape(2, 3);       // a view
v.set([0, 0], 100);
a;                               // => [100, 1, 2, 3, 4, 5]
np.mayShareMemory(a, v);         // => true
```

Integer-array and boolean-mask indexing return **copies**. Use `a.copy()` to get an independent array.

## Memory layout

```js
const t = np.arange(6).reshape(2, 3).T;
t.flags.cContiguous;                          // => false
t.flags.fContiguous;                          // => true
```

## Changing shape

```js
np.arange(6).reshape([3, -1]).shape;          // => [3, 2]
np.zeros([2, 3]).ravel().shape;               // => [6]
np.squeeze(np.zeros([1, 3, 1])).shape;        // => [3]
np.expandDims(np.array([1, 2]), 0).shape;     // => [1, 2]
np.transpose(np.zeros([2, 3])).shape;         // => [3, 2]
```

## Complex numbers

`complex64` and `complex128` arrays exist (for example as `np.fft` results), but in 1.0.1 their elements cannot be converted to JS values with `toArray()` or `item()`. Use `toTypedArray()`, which returns interleaved `[re, im, …]` values:

```js
Array.from(np.fft.fft([1, 2, 3, 4]).toTypedArray()); // => [10, 0, -2, 2, -2, 0, -2, -2]
```
