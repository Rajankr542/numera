# Reductions and statistics

A reduction combines the elements of an array along one or more axes: `sum`, `prod`, `min`, `max`, `mean`, `var`, `std`, `argmin`, `argmax`, `any`, `all` and the NaN-ignoring variants. They take the same options as NumPy, and each one is available as both a function and a method.

## Axis

With no `axis`, the array is reduced to a single value, returned as a 0-d array. `axis` can be a number, a negative number or a list of axes:

```js
const a = np.array([[1, 2], [3, 4]]);
np.sum(a).item();                 // => 10
np.sum(a, { axis: 0 });           // => [4, 6]
np.sum(a, { axis: -1 });          // => [3, 7]
np.mean(a, { axis: [0, 1] }).item(); // => 2.5
a.sum({ axis: 0 });               // => [4, 6]
a.argmax({ axis: 1 });            // => [1, 1]
```

## keepdims

`keepdims: true` keeps the reduced axes with length 1, so the result still broadcasts against the input:

```js
const a = np.array([[1, 2], [3, 4]]);
np.sum(a, { axis: 1, keepdims: true }); // => [[3], [7]]
np.divide(a, np.sum(a, { axis: 1, keepdims: true })); // => [[0.3333333333333333, 0.6666666666666666], [0.42857142857142855, 0.5714285714285714]]
```

## dtype, initial and where

```js
np.sum([1, 2, 3], { dtype: "float32" }).dtype.name;    // => "float32"
np.sum(np.array([100, 100], { dtype: "int8" })).item(); // => 200
np.sum([1, 2], { initial: 10 }).item();                // => 13
np.sum([[1, 2], [3, 4]], { where: [[true, false], [true, true]] }).item(); // => 8
np.max([], { initial: -Infinity }).item() === -Infinity; // => true
```

As in NumPy, `sum` and `prod` of small integer types accumulate in the platform integer (`int64`), so `int8` values don't overflow. Reducing an empty array with no identity (such as `max([])`) raises `ValueError` unless you pass `initial`.

## Variance and standard deviation

`ddof` sets the "delta degrees of freedom". Use `ddof: 1` for the sample estimate:

```js
np.var([1, 2, 3, 4]).item();             // => 1.25
np.std([1, 2, 3, 4]).item();             // => 1.118033988749895
np.std([1, 2, 3, 4], { ddof: 1 }).item(); // => 1.2909944487358056
```

## NaN handling

Ordinary reductions propagate NaN. The `nan*` versions skip it:

```js
Number.isNaN(np.max([1, NaN, 3]).item()); // => true
np.nanmax([1, NaN, 3]).item();   // => 3
np.nansum([1, NaN, 2]).item();   // => 3
np.nanmean([1, NaN, 3]).item();  // => 2
```

## Order statistics

```js
np.median([3, 1, 2]).item();                 // => 2
np.median([[1, 3], [2, 4]], { axis: 0 });    // => [1.5, 3.5]
np.percentile([1, 2, 3, 4], [25, 75]);       // => [1.75, 3.25]
np.quantile([1, 2, 3, 4], 0.25).item();      // => 1.75
np.ptp([1, 5, 2]).item();                    // => 4
np.average([1, 2, 3], { weights: [3, 1, 0] }).item(); // => 1.25
```

## Cumulative operations and differences

```js
np.cumsum([1, 2, 3]);        // => [1, 3, 6]
np.cumprod([1, 2, 3]);       // => [1, 2, 6]
np.diff([1, 4, 9]);          // => [3, 5]
```

## Counting and histograms

```js
np.countNonzero([0, 1, 2]).valueOf();  // => 2
np.bincount([0, 1, 1, 3]);             // => [1, 2, 0, 1]
const h = np.histogram(np.array([1, 2, 2, 3]), 3);
h.hist;                                // => [1, 2, 1]
h.edges;                               // => [1, 1.6666666666666665, 2.333333333333333, 3]
```

## Logical reductions

```js
np.any([0, 1]).item();  // => true
np.all([1, 0]).item();  // => false
```

## Ufunc reductions

Every binary ufunc has a `reduce` method, so you can reduce with any operation. There are also `accumulate`, `reduceat` and `outer`:

```js
np.maximum.reduce([[1, 5], [4, 2]], { axis: 1 }); // => [5, 4]
np.multiply.accumulate([1, 2, 3, 4]);           // => [1, 2, 6, 24]
np.add.reduceat([1, 2, 3, 4], [0, 2]);          // => [3, 7]
```

See the [reductions](/reference/reduce.html) and [statistics](/reference/statistics.html) references for all functions.
