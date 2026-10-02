# Reductions

`sum`, `prod`, `min`, `max` (`amin`, `amax`), `mean`, `var`, `std`, `argmin` and `argmax` are available as functions and as methods.

## Axis and keepdims

```js
const a = np.array([[1, 2], [3, 4]]);
np.sum(a).item();                 // => 10
np.sum(a, { axis: 0 });           // => [4, 6]
np.sum(a, { axis: -1 });          // => [3, 7]
np.mean(a, { axis: [0, 1] }).item(); // => 2.5
a.argmax({ axis: 1 });            // => [1, 1]
np.sum(a, { axis: 1, keepdims: true }); // => [[3], [7]]
np.min([[1, 5], [3, 2]], { axis: 0 });  // => [1, 2]
np.prod([1, 2, 3, 4]).item();     // => 24
```

## dtype

```js
np.sum([1, 2, 3], { dtype: "float32" }).dtype.name;    // => "float32"
np.sum(np.array([100, 100], { dtype: "int8" })).item(); // => 200
```

`sum` and `prod` of small integer types accumulate in `int64`, as in NumPy. Reducing an empty array with no identity, such as `max([])`, raises `ValueError`.

## Variance and standard deviation

```js
np.var([1, 2, 3, 4]).item();             // => 1.25
np.std([1, 2, 3, 4]).item();             // => 1.118033988749895
np.std([1, 2, 3, 4], { ddof: 1 }).item(); // => 1.2909944487358056
```

## NaN

Reductions propagate NaN. The `nan*` variants arrived in 1.0.2.

```js
Number.isNaN(np.max([1, NaN, 3]).item()); // => true
```
