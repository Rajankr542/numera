# Data types

Every array has a single dtype. numera supports the same numeric dtypes as NumPy, uses its promotion table and casting rules, and applies NEP 50's weak scalars. Non-numeric dtypes have their own modules: strings in `np.strings` and `np.char`, datetimes as `datetime64`/`timedelta64`, and records in `np.rec`.

## Numeric dtypes

| Kind | dtypes | JS element type in `toArray()` | `toTypedArray()` |
| --- | --- | --- | --- |
| boolean | `bool` | `boolean` | `Uint8Array` |
| signed integer | `int8`, `int16`, `int32`, `int64` | `number` | `Int8Array` … `BigInt64Array` |
| unsigned integer | `uint8`, `uint16`, `uint32`, `uint64` | `number` | `Uint8Array` … `BigUint64Array` |
| floating | `float16`, `float32`, `float64` | `number` | `Uint16Array` (raw bits), `Float32Array`, `Float64Array` |
| complex | `complex64`, `complex128` | `Complex` | interleaved `Float32Array` / `Float64Array` |

You can pass a dtype as a name (`"float32"`), a NumPy alias (`"single"`, `"double"`, `"int_"`) or a `DType` object (`np.float32`). `np.dtype(x)` resolves any of these to the canonical `DType`:

```js
const d = np.dtype("single");
d.name;              // => "float32"
d.kind;              // => "f"
d.itemSize;          // => 4
d === np.float32;    // => true
np.zeros([2], { dtype: np.int32 }).dtype.name; // => "int32"
```

## Inference

`np.array` infers the dtype from the data, like NumPy:

```js
np.array([true, false]).dtype.name;  // => "bool"
np.array([1, 2]).dtype.name;         // => "int64"
np.array([1, 2.5]).dtype.name;       // => "float64"
np.array([np.complex(1, 2)]).dtype.name; // => "complex128"
```

All booleans give `bool`, all integers give `int64`, anything else gives `float64`. One exception: a JS number that is an integer but larger than 2^53 is inferred as `float64`, because it is no longer exact. Use a `bigint` for large integers.

## Type promotion

When two arrays are combined, the result has the smallest dtype that can hold both, using NumPy's promotion table:

```js
np.promoteTypes("int8", "uint8").name;     // => "int16"
np.promoteTypes("int32", "float32").name;  // => "float64"
np.add(np.array([1], { dtype: "int32" }), np.array([1], { dtype: "float32" })).dtype.name; // => "float64"
```

JS numbers are **weak** scalars (NEP 50). They can raise the *kind* (integer → float → complex) but never the *size*:

```js
const i8 = np.array([1, 2], { dtype: "int8" });
np.add(i8, 1).dtype.name;        // => "int8"
np.add(i8, 1.5).dtype.name;      // => "float64"
np.resultType("int8", 300).name; // => "int8"
np.multiply(np.array([1, 2], { dtype: "float32" }), 2.5).dtype.name; // => "float32"
```

A weak scalar that doesn't fit the array's dtype raises an error. Integer overflow inside array arithmetic wraps around, as in NumPy:

```js
np.add(np.array([1, 2], { dtype: "uint8" }), 255); // => [0, 1]
np.add(np.array([100], { dtype: "int8" }), np.array([100], { dtype: "int8" })); // => [-56]
```

## Casting

`a.astype(dtype)` converts and copies. Float → int truncates toward zero. By default the cast is `"unsafe"`, which allows anything. Pass `casting` to restrict it:

```js
np.array([-1.5, 2.7]).astype("int32");                        // => [-1, 2]
np.array([true, false]).astype("int32");                      // => [1, 0]
np.array([1, 2], { dtype: "int8" }).astype("float32", { casting: "safe" }).dtype.name; // => "float32"
np.canCast("int8", "int16");                                  // => true
np.canCast("float64", "int32");                               // => false
np.canCast("float64", "int32", "unsafe");                     // => true
```

The five casting rules are `"no"`, `"equiv"`, `"safe"`, `"same_kind"` and `"unsafe"`. Ufuncs check their inputs and `out=` with `"same_kind"` by default:

```js
const out = np.zeros(1, { dtype: "int32" });
np.add(np.array([1.5]), 1, { out, casting: "unsafe" });
out;                                                          // => [2]
```

## Limits

```js
np.iinfo("int8").max;      // => 127
np.iinfo("int8").min;      // => -128
np.finfo("float64").eps;   // => 2.220446049250313e-16
np.finfo("float32").eps;   // => 1.1920928955078125e-7
```

## 64-bit integers

JS numbers are exact only up to 2^53, while `int64` and `uint64` go further. numera stores them exactly in native memory. Only the conversion back to JS can lose precision:

```js
const big = np.array([2n ** 60n]);
big.dtype.name;                         // => "int64"
big.toTypedArray()[0];                  // => "1152921504606846976"
np.array([1, 2], { dtype: "int64" }).toTypedArray().constructor.name; // => "BigInt64Array"
```

Use `toTypedArray()` (or `bigint` inputs) when values may exceed `Number.MAX_SAFE_INTEGER`. `toArray()` returns JS numbers.

## float16

`float16` is stored natively, and arithmetic follows NumPy's half-precision results. `toArray()` returns the values as JS numbers, while `toTypedArray()` returns the raw 16-bit patterns as a `Uint16Array`:

```js
np.array([0.1], { dtype: "float16" }).item(); // => 0.0999755859375
```
