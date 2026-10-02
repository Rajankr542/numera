# Random numbers

numera has both of NumPy's random APIs. For the same seed, each one produces the **same numbers as NumPy**, bit for bit. The differential tests check this against real NumPy output.

- **`Generator`** (recommended): created with `np.random.defaultRng(seed)`, backed by PCG64. This matches NumPy's `np.random.default_rng`.
- **`RandomState`** (legacy): the global `np.random.seed`/`np.random.rand` functions and `new np.random.RandomState(seed)`, backed by MT19937.

## Generator

```js
const rng = np.random.defaultRng(42);
rng.random(3);                // => [0.7739560485559633, 0.4388784397520523, 0.8585979199113825]
```

The Python equivalent produces the same values:

```python
import numpy as np
rng = np.random.default_rng(42)
rng.random(3)   # -> array([0.77395605, 0.43887844, 0.85859792])
```

Calls continue the same stream, just as in NumPy:

```js
const rng = np.random.defaultRng(42);
rng.normal(5, 2, [3]);         // => [5.609434159508862, 2.920031787519009, 6.5009023916129145]
rng.integers(1, 7, [4]);       // => [1, 5, 2, 1]
rng.choice(5, { size: 3, replace: false }); // => [1, 3, 4]
rng.binomial(10, 0.5, [3]);    // => [3, 5, 4]
rng.poisson(3, [3]);           // => [4, 1, 7]
```

`size` is a number or a shape. Omit it to get a single JS number:

```js
typeof np.random.defaultRng(1).standardNormal(); // => "number"
np.random.defaultRng(1).random([2, 2]).shape;    // => [2, 2]
```

## Shuffling and permutations

```js
const a = np.arange(5);
np.random.defaultRng(0).shuffle(a);   // in place
a;                                     // => [2, 4, 3, 0, 1]
np.random.defaultRng(42).permutation(4).shape; // => [4]
```

## Distributions

`Generator` has all of NumPy's distributions except `multivariate_normal`: `uniform`, `normal`, `standardNormal`, `integers`, `choice`, `binomial`, `negativeBinomial`, `poisson`, `geometric`, `hypergeometric`, `logseries`, `zipf`, `exponential`, `standardExponential`, `gamma`, `standardGamma`, `beta`, `chisquare`, `noncentralChisquare`, `f`, `noncentralF`, `standardT`, `standardCauchy`, `pareto`, `weibull`, `power`, `laplace`, `gumbel`, `logistic`, `lognormal`, `rayleigh`, `wald`, `vonmises`, `triangular`, `multinomial`, `dirichlet` and `multivariateHypergeometric`. See [discrete](/reference/random-discrete.html), [continuous](/reference/random-continuous.html) and [multivariate](/reference/random-multivariate.html) distributions.

## Float32 output

```js
np.random.defaultRng(42).random(2, "float32"); // => [0.08925092220306396, 0.7739560008049011]
```

## Independent streams

`spawn(n)` creates independent child generators for parallel work:

```js
const kids = np.random.defaultRng(42).spawn(2);
kids.length; // => 2
```

> **Note**: In this release, the child streams from `spawn` are independent but are **not** the same numbers that NumPy's `spawn` produces.

## Legacy RandomState API

The global functions use a shared `RandomState`. It is seeded with `np.random.seed`, and the results match NumPy's legacy functions:

```js
np.random.seed(0);
np.random.rand(2, 2);       // => [[0.5488135039273248, 0.7151893663724195], [0.6027633760716439, 0.5448831829968969]]
np.random.randn(2);         // => [1.8675579901499675, -0.977277879876411]

const rs = new np.random.RandomState(0);
rs.rand(3);                 // => [0.5488135039273248, 0.7151893663724195, 0.6027633760716439]
```

Use `Generator` for new code: it is faster, statistically better, and it is NumPy's recommendation too.

## Bit generators

`np.random.PCG64`, `PCG64DXSM`, `MT19937`, `Philox`, `SFC64` and `SeedSequence` are available and produce NumPy's raw streams. In this release, a `Generator` constructed from them supports only a few methods. Use `np.random.defaultRng(seed)` for drawing from distributions. See [Bit generators](/reference/random-bitgen.html).
