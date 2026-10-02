# Universal functions (ufunc)

A *universal function* operates on arrays element by element, with [broadcasting](/guides/broadcasting.html) and NumPy type promotion. numera {{version}} has {{ufuncs}} of them: the arithmetic functions plus `abs`, `negative`, `sqrt`, `exp`, `log` and `matmul`.

```js
np.add([1, 2, 3], 10);              // => [11, 12, 13]
np.sqrt([1, 4, 9]);                 // => [1, 2, 3]
np.multiply([[1], [2]], [10, 20]);  // => [[10, 20], [20, 40]]
```

## Output type determination

The result dtype comes from NumPy's promotion of the input dtypes. JS numbers are weak scalars (NEP 50). See [Data types](/guides/dtypes.html).

```js
np.add(np.array([1], { dtype: "int8" }), 1).dtype.name;     // => "int8"
np.add(np.array([1], { dtype: "int8" }), 1.5).dtype.name;   // => "float64"
np.divide([1, 2], 2).dtype.name;                            // => "float64"
```

## Not yet in {{version}}

The ufunc options (`out`, `where`, `dtype`, `casting`, `order`), the ufunc methods (`reduce`, `accumulate`, `reduceat`, `outer`, `at`) and `seterr`/`errstate` are not available in this release. They were added in 1.0.2, together with about 90 more ufuncs (comparisons, logic, trigonometry, rounding, bitwise, …).
