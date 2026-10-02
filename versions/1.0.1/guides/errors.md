# Errors

## Error classes

Errors from the native core arrive as typed JavaScript errors. Every library error extends `np.NativpyError` and has a stable `code`:

| Class | Raised for |
| --- | --- |
| `ValueError` | invalid argument values, out-of-range integers |
| `ShapeError` | impossible reshapes and shape mismatches |
| `BroadcastError` | shapes that don't broadcast |
| `DTypeError` | unknown dtypes and unsupported dtype combinations |
| `IndexError` | out-of-bounds or invalid indices |
| `LinAlgError` | singular matrices, non-convergence |
| `MemoryError` | failed native allocation |
| `NotImplementedError` | features not yet implemented |

```js
let err;
try { np.zeros(2).reshape(3); } catch (e) { err = e; }
err instanceof np.ShapeError;    // => true
err instanceof np.NativpyError;  // => true
err.code;                        // => "NATIVPY_SHAPE_ERROR"
err.message;                     // => "cannot reshape array of size 2 into shape (3,)"
```

Out-of-range scalars raise `ValueError`:

```js
let msg;
try { np.add(np.array([1], { dtype: "uint8" }), 256); } catch (e) { msg = e.message; }
msg; // => "integer 256 out of bounds for target dtype"
```

## Floating-point behaviour

Division by zero gives `Infinity`, and invalid operations give `NaN`. In 1.0.1 this happens silently: `seterr`, `errstate` and floating-point warnings arrived in 1.0.2.

```js
np.divide([1, -1], [0, 0]).toArray().map(String); // => ["Infinity", "-Infinity"]
Number.isNaN(np.sqrt([-1]).item(0));              // => true
```
