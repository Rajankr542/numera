# Random numbers

For the same seed, numera {{version}} produces the **same numbers as NumPy**.

- **`Generator`**: created with `np.random.defaultRng(seed)`, backed by PCG64. It has `random`, `uniform`, `standardNormal`, `normal`, `integers`, `choice`, `shuffle` and `permutation`.
- **Legacy API**: `np.random.seed`, `rand`, `randn`, `random`, `randomSample`, `standardNormal`, `normal`, `uniform`, `randint`, `choice`, `shuffle`, `permutation` and `new np.random.RandomState(seed)`, backed by MT19937.

## Generator

```js
const rng = np.random.defaultRng(42);
rng.random(3);                // => [0.7739560485559633, 0.4388784397520523, 0.8585979199113825]
rng.integers(0, 10, [5]);     // => [0, 6, 2, 0, 5]
np.random.defaultRng(42).normal(5, 2, [3]); // => [5.609434159508862, 2.920031787519009, 6.5009023916129145]
typeof np.random.defaultRng(1).standardNormal(); // => "number"
np.random.defaultRng(1).random([2, 2]).shape;    // => [2, 2]
```

The Python equivalent produces the same values:

```python
import numpy as np
rng = np.random.default_rng(42)
rng.random(3)   # -> array([0.77395605, 0.43887844, 0.85859792])
```

## Shuffling

```js
const a = np.arange(5);
np.random.defaultRng(0).shuffle(a);
a;                                     // => [2, 4, 3, 0, 1]
np.random.defaultRng(42).permutation(4).shape; // => [4]
```

## Legacy API

```js
np.random.seed(0);
np.random.rand(2, 2);       // => [[0.5488135039273248, 0.7151893663724195], [0.6027633760716439, 0.5448831829968969]]
np.random.randn(2);         // => [1.8675579901499675, -0.977277879876411]
new np.random.RandomState(0).rand(3); // => [0.5488135039273248, 0.7151893663724195, 0.6027633760716439]
```
