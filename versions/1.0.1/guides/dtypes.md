# Data types

Every array has a single dtype. numera supports the same numeric dtypes as NumPy and uses its promotion table and NEP 50 weak scalars.

## Numeric dtypes

| Kind | dtypes | `toTypedArray()` |
| --- | --- | --- |
| boolean | `bool` | `Uint8Array` |
| signed integer | `int8`, `int16`, `int32`, `int64` | `Int8Array` … `BigInt64Array` |
| unsigned integer | `uint8`, `uint16`, `uint32`, `uint64` | `Uint8Array` … `BigUint64Array` |
| floating | `float16`, `float32`, `float64` | `Uint16Array` (raw bits), `Float32Array`, `Float64Array` |
| complex | `complex64`, `complex128` | interleaved `Float32Array` / `Float64Array` |

You can pass a dtype as a name (`"float32"`) or a `DType` object (`np.float32`):

```js
const d = np.dtype("float32");
d.name;              // => "float32"
d.kind;              // => "f"
d.itemSize;          // => 4
d === np.float32;    // => true
np.zeros([2], { dtype: np.int32 }).dtype.name; // => "int32"
```

## Inference

```js
np.array([true, false]).dtype.name;  // => "bool"
np.array([1, 2]).dtype.name;         // => "int64"
np.array([1, 2.5]).dtype.name;       // => "float64"
```

## Type promotion

```js
np.promoteTypes("int8", "uint8").name;     // => "int16"
np.promoteTypes("int32", "float32").name;  // => "float64"
```

JS numbers are **weak** scalars: they can raise the *kind* (integer → float) but never the *size*:

```js
const i8 = np.array([1, 2], { dtype: "int8" });
np.add(i8, 1).dtype.name;        // => "int8"
np.add(i8, 1.5).dtype.name;      // => "float64"
np.multiply(np.array([1, 2], { dtype: "float32" }), 2.5).dtype.name; // => "float32"
```

Integer overflow inside array arithmetic wraps around, as in NumPy:

```js
np.add(np.array([1, 2], { dtype: "uint8" }), 255); // => [0, 1]
np.add(np.array([100], { dtype: "int8" }), np.array([100], { dtype: "int8" })); // => [-56]
```

## Casting

`a.astype(dtype)` converts and copies. Float → int truncates toward zero:

```js
np.array([-1.5, 2.7]).astype("int32");    // => [-1, 2]
np.array([true, false]).astype("int32");  // => [1, 0]
```

## 64-bit integers

JS numbers are exact only up to 2^53, while `int64` and `uint64` go further. Use `bigint` inputs and `toTypedArray()` for full precision:

```js
const big = np.array([2n ** 60n]);
big.dtype.name;                         // => "int64"
big.toTypedArray()[0];                  // => "1152921504606846976"
```

## float16

```js
np.array([0.1], { dtype: "float16" }).item(); // => 0.0999755859375
```
