# Errors and floating-point

## Error classes

Errors from the native core arrive as typed JavaScript errors. Raw C++ exceptions never reach your code. Every library error extends `np.NativpyError` and has a stable `code`:

| Class | Raised for | NumPy equivalent |
| --- | --- | --- |
| `ValueError` | invalid argument values, out-of-range integers, read-only writes | `ValueError`, `OverflowError` |
| `ShapeError` | impossible reshapes and shape mismatches | `ValueError` |
| `BroadcastError` | shapes that don't broadcast | `ValueError` |
| `DTypeError` | unknown dtypes, disallowed casts | `TypeError`, `UFuncTypeError` |
| `IndexError` | out-of-bounds or invalid indices | `IndexError` |
| `LinAlgError` | singular matrices, non-convergence | `numpy.linalg.LinAlgError` |
| `FloatingPointError` | floating-point errors in `"raise"` mode | `FloatingPointError` |
| `MemoryError` | failed native allocation | `MemoryError` |
| `NotImplementedError` | features not yet implemented | `NotImplementedError` |

```js
let err;
try { np.zeros(2).reshape(3); } catch (e) { err = e; }
err instanceof np.ShapeError;    // => true
err instanceof np.NativpyError;  // => true
err.code;                        // => "NATIVPY_SHAPE_ERROR"
err.message;                     // => "cannot reshape array of size 2 into shape (3,)"
```

Messages follow NumPy's wording, so they are familiar and easy to search for.

## Floating-point errors

Like NumPy, numera doesn't throw for floating-point problems by default. Division by zero gives `inf`, and invalid operations give `nan`. Each condition is handled according to a mode:

| Condition | Example | Default |
| --- | --- | --- |
| `divide` | `1 / 0` | `"warn"` |
| `over` | `exp(1000)` | `"warn"` |
| `under` | `exp(-1000)` | `"ignore"` |
| `invalid` | `0 / 0`, `sqrt(-1)` | `"warn"` |

The modes are `"ignore"`, `"warn"`, `"raise"` and `"print"`. `"warn"` emits a Node `RuntimeWarning` through `process.emitWarning`, which you can watch with `process.on("warning", ...)`. `"raise"` throws `np.FloatingPointError`.

```js
np.divide([1, -1], [0, 0]).toArray().map(String); // => ["Infinity", "-Infinity"]
np.geterr();  // => {"divide": "warn", "over": "warn", "under": "ignore", "invalid": "warn"}
```

## errstate

`np.errstate(settings, fn)` runs `fn` with temporary settings and restores the previous ones afterwards, even if `fn` throws. It is the equivalent of NumPy's `with np.errstate(...)` block:

```js
np.errstate({ divide: "ignore" }, () => np.divide([1], [0]).item()) === Infinity; // => true
let name;
np.errstate({ divide: "raise" }, () => {
  try { np.divide([1], [0]); } catch (e) { name = e.name; }
});
name; // => "FloatingPointError"
```

`fn` must be synchronous. The settings are not carried across `await`.

## seterr

`np.seterr` changes the global settings and returns the previous ones, so you can restore them:

```js
const old = np.seterr({ all: "ignore" });
np.divide([1], [0]).item() === Infinity; // => true
np.seterr(old).divide;        // => "ignore"
np.geterr().divide;           // => "warn"
```

## Testing helpers

`np.testing` has NumPy's assertion helpers. They throw `np.testing.AssertionError` with NumPy-style messages:

```js
np.testing.assertAllclose([1, 2], [1, 2.0000001]);   // passes
let failed = false;
try { np.testing.assertArrayEqual([1, 2], [1, 3]); } catch (e) { failed = e instanceof np.testing.AssertionError; }
failed; // => true
```
