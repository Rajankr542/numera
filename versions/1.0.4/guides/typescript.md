# TypeScript

numera is written in TypeScript and its declarations ship in the package (`dist/index.d.ts`), so you don't need an `@types` package. Every function in the [API reference](/reference/index.html) has a typed signature, and the [NumPy name index](/reference/numpy-index.html) shows the declaration of every implemented name.

## Setup

numera is an ES module. Use a module resolution that understands package `exports`:

```json
{
  "compilerOptions": {
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "target": "ES2022",
    "strict": true
  }
}
```

## Exported types

Import the types with `import type`:

```ts norun
import np, { NDArray } from "@cyfora/numera";
import type { ArrayLike, DTypeLike, Shape, UfuncOptions, ReduceOptions } from "@cyfora/numera";

function normalize(x: ArrayLike, opts: ReduceOptions = {}): NDArray {
  const a = np.asarray(x);
  return np.divide(np.subtract(a, np.mean(a, opts)), np.std(a, opts));
}
```

| Type | Meaning |
| --- | --- |
| `NDArray` | the array class (a value as well as a type, so `instanceof NDArray` works) |
| `ArrayLike` | `NDArray` or nested JS arrays of numbers, booleans, bigints or complex values |
| `Operand` | `ArrayLike` or a single complex value, as accepted by ufuncs |
| `NestedArray` | the nested JS array returned by `toArray()` |
| `Shape` | `readonly number[]` |
| `DType`, `DTypeName`, `DTypeLike` | a dtype object, a canonical name such as `"float32"`, or anything `np.dtype()` accepts |
| `Casting` | `"no" \| "equiv" \| "safe" \| "same_kind" \| "unsafe"` |
| `ComplexLike` | `Complex` or `{ re, im }` |
| `IndexSpec`, `SliceTuple` | arguments of `get`, `slice` and `set` |
| `UfuncOptions`, `UfuncReduceOptions`, `UfuncAccumulateOptions`, `UfuncReduceatOptions` | ufunc options (`out`, `where`, `dtype`, `casting`, `order`, …) |
| `ReduceOptions`, `VarOptions`, `ArgReduceOptions` | options for `sum`, `var`/`std`, `argmax`/`argmin` |
| `ArrayOptions`, `CreationOptions`, `LinspaceOptions`, `EyeOptions`, `LikeOptions` | creation options |
| `SvdResult`, `QrResult`, `EigResult`, `LstsqResult` | `np.linalg` results |
| `FftOptions`, `FftNOptions`, `FftNorm` | `np.fft` options |
| `Seed`, `Size` | `np.random` seeds and output sizes |
| `ErrMode`, `ErrSettings`, `ErrState` | `seterr` / `errstate` |

## Results are typed as `NDArray`

The element type of an array is a runtime property (`a.dtype`), not part of the TypeScript type. Functions return `NDArray`, and conversions return the general types:

```ts norun
const a: NDArray = np.arange(6).reshape(2, 3);
const nested = a.toArray();          // NestedArray
const flat = a.toTypedArray();       // a typed array, or BigInt64Array for int64
const x: number = np.sum(a).item() as number;
```

Check `a.dtype.name` (or `a.dtype.kind`) at runtime when a value's element type matters.

## Errors

The error classes are exported as values, so you can narrow with `instanceof`:

```ts norun
import np, { ShapeError, LinAlgError } from "@cyfora/numera";

try {
  np.linalg.inv(m);
} catch (e) {
  if (e instanceof LinAlgError) { /* singular */ }
  else throw e;
}
```

See [Errors and floating-point](/guides/errors.html).
