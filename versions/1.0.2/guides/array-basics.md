# Array basics

The central object in numera is `NDArray`: an n-dimensional, fixed-size array of elements that all have the same type (its dtype). It works like NumPy's `ndarray`: it has the same attributes, views share memory in the same way, and copies follow the same rules.

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

The [array creation reference](/reference/creation.html) has the full list, including `logspace`, `fromfunction`, `frombuffer`, `meshgrid`, `tri`, `diag` and `vander`.

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

`strides` are in bytes, as in NumPy. `flags` holds `cContiguous`, `fContiguous`, `writeable` and `ownData`.

## Converting to JavaScript

| Method | Returns |
| --- | --- |
| `a.toArray()` / `a.tolist()` | nested JS arrays of numbers, booleans or `Complex` values |
| `a.toTypedArray()` | a flat typed-array copy in C order (`Float64Array`, `Int32Array`, `BigInt64Array`, …) |
| `a.item(...index)` | one element as a JS number, boolean or `Complex` |
| `String(a)` / `a.toString()` | NumPy's `repr`, e.g. `array([1, 2])` |
| `a.tobytes()` | the raw bytes as a `Uint8Array` |

```js
const b = np.array([[1, 2], [3, 4]], { dtype: "int16" });
b.toArray();                           // => [[1, 2], [3, 4]]
b.toTypedArray().constructor.name;     // => "Int16Array"
b.item(1, 0);                          // => 3
String(np.array([1.5, 2]));            // => "array([1.5, 2. ])"
```

## Views and copies

Reshaping, transposing and basic slicing return **views**. A view shares memory with the original array, so writing to it changes the original. `a.base` is the array that owns the memory.

```js
const a = np.arange(6);
const v = a.reshape(2, 3);       // a view
v.set([0, 0], 100);
a;                               // => [100, 1, 2, 3, 4, 5]
v.base === a;                    // => true
np.sharesMemory(a, v);           // => true
a.copy().base;                   // => null
```

Integer-array and boolean-mask indexing return **copies**, as in NumPy. Use `a.copy()` or `np.copy(a)` to get an independent array.

## Memory layout

Arrays are stored in C (row-major) order unless a view makes them strided. The transpose of a C-contiguous array is F-contiguous. `np.ascontiguousarray` makes a C-ordered copy when one is needed.

```js
const t = np.arange(6).reshape(2, 3).T;
t.flags.cContiguous;                          // => false
t.flags.fContiguous;                          // => true
np.ascontiguousarray(t).flags.cContiguous;    // => true
```

## Changing shape

```js
np.arange(6).reshape([3, -1]).shape;          // => [3, 2]
np.zeros([2, 3]).ravel().shape;               // => [6]
np.squeeze(np.zeros([1, 3, 1])).shape;        // => [3]
np.expandDims(np.array([1, 2]), 0).shape;               // => [1, 2]
np.concatenate([[[1, 2]], [[3, 4]]], 0);      // => [[1, 2], [3, 4]]
np.stack([[1, 2], [3, 4]]);                   // => [[1, 2], [3, 4]]
np.split(np.arange(6), 3).map((x) => x.toArray()); // => [[0, 1], [2, 3], [4, 5]]
np.tile([1, 2], 2);                           // => [1, 2, 1, 2]
```

See [Shape manipulation](/reference/shape.html) for the full list.

## Complex numbers

`complex64` and `complex128` arrays hold `Complex` values. Create them with `np.complex(re, im)` or plain `{ re, im }` objects.

```js
const z = np.array([np.complex(3, 4)]);
z.dtype.name;                         // => "complex128"
np.abs(z);                            // => [5]
String(z.item(0));                    // => "(3+4j)"
```
