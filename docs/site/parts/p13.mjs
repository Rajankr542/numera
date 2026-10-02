// P13 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.
/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "random-bitgen",
    title: "Random — bit generators",
    intro: "Low-level bit generators that back `np.random.Generator`. All bit generators implement `random()`, `getState()` and `setState()`. Prefer `np.random.defaultRng()` for everyday use.",
    entries: [
      {
        name: "SeedSequence",
        sig: "new np.random.SeedSequence(entropy?, options?)",
        desc: "Hashes seed entropy into a reproducible pool for initialising bit generators. `entropy` can be an integer or array of integers; omit for OS-level randomness.",
        args: [
          { name: "[entropy]", type: "number | bigint | readonly (number|bigint)[]", desc: "Seed entropy; omit for non-reproducible randomness." },
          { name: "[options.spawnKey]", type: "readonly number[]", desc: "Spawn key for child sequences." },
          { name: "[options.poolSize]", type: "number", desc: "Size of the hashed entropy pool (>= 4). Default 4." },
        ],
        returns: "SeedSequence",
        example: `const ss = new np.random.SeedSequence(42);
ss.entropy;    // => 42
ss.poolSize;   // => 4
const [c] = ss.spawn(1);
c.spawnKey;    // => [0]`,
      },
      {
        name: "BitGenerator",
        sig: "new np.random.BitGenerator(bitGenerator?)",
        desc: "Namespace of available P13 bit-generator constructors. Use individual constructors (e.g. `PCG64`, `MT19937`) or `np.random.defaultRng()` instead.",
        args: [],
        returns: "BitGenerator",
        example: `const bg = new np.random.PCG64(new np.random.SeedSequence(0));
typeof bg.random(); // => "number"`,
      },
      {
        name: "bit_generator",
        sig: "rng.bit_generator",
        desc: "The name of the underlying bit generator type (read-only property). Always `\"PCG64\"` for generators created by `defaultRng`. Equivalent to `numpy.random.Generator.bit_generator`.",
        args: [],
        returns: "string",
        example: `const rng = np.random.defaultRng(42);
rng.bit_generator; // => "PCG64"`,
      },
      {
        name: "spawn",
        sig: "rng.spawn(n)",
        desc: "Return `n` independent Generator children seeded from OS entropy. Each child is an independent PCG64-backed Generator. Equivalent to `numpy.random.Generator.spawn`.",
        args: [{ name: "n", type: "number", desc: "Number of child generators to create." }],
        returns: "Generator[]",
        example: `const rng = np.random.defaultRng(42);
const children = rng.spawn(3);
children.length; // => 3`,
      },
      {
        name: "PCG64DXSM",
        sig: "new np.random.PCG64DXSM(seedSequence)",
        desc: "PCG64-DXSM bit generator — a variant of PCG64 with the DXSM output function. Preferred over PCG64 for new code in NumPy >= 1.17.",
        args: [
          { name: "seedSequence", type: "SeedSequence", desc: "Seed sequence to initialise the generator state." },
        ],
        returns: "PCG64DXSM",
        example: `const bg = new np.random.PCG64DXSM(new np.random.SeedSequence(1));
const v = bg.random();
v >= 0 && v < 1; // => true`,
      },
      {
        name: "Philox",
        sig: "new np.random.Philox(seedSequence)",
        desc: "Philox 4×64-10 counter-based bit generator. Suitable for parallel random-number generation.",
        args: [
          { name: "seedSequence", type: "SeedSequence", desc: "Seed sequence to initialise the generator state." },
        ],
        returns: "Philox",
        example: `const bg = new np.random.Philox(new np.random.SeedSequence(7));
const v = bg.random();
v >= 0 && v < 1; // => true`,
      },
      {
        name: "SFC64",
        sig: "new np.random.SFC64(seedSequence)",
        desc: "SFC64 (Small Fast Counting) bit generator. Fast, non-cryptographic, excellent statistical properties.",
        args: [
          { name: "seedSequence", type: "SeedSequence", desc: "Seed sequence to initialise the generator state." },
        ],
        returns: "SFC64",
        example: `const bg = new np.random.SFC64(new np.random.SeedSequence(3));
const v = bg.random();
v >= 0 && v < 1; // => true`,
      },
    ],
  },
  {
    id: "random-discrete",
    title: "Random — discrete distributions (Generator)",
    intro: "Discrete distribution samplers on `np.random.Generator`. Obtain a generator with `np.random.defaultRng(seed)`. All methods return a JS number when `size` is omitted and an `int64` `NDArray` otherwise.",
    entries: [
      {
        name: "binomial",
        sig: "rng.binomial(n, p, size?)",
        desc: "Draw samples from a binomial distribution: number of successes in `n` Bernoulli trials with success probability `p`. Equivalent to `numpy.random.Generator.binomial`.",
        args: [
          { name: "n", type: "number", desc: "Number of trials (integer >= 0)." },
          { name: "p", type: "number", desc: "Probability of success in [0, 1]." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<int64>",
        example: `const rng = np.random.defaultRng(42);
rng.binomial(10, 0.5);          // => 6
rng.binomial(10, 0);            // => 0
rng.binomial(10, 1);            // => 10`,
      },
      {
        name: "negativeBinomial",
        sig: "rng.negativeBinomial(n, p, size?)",
        desc: "Draw samples from a negative binomial distribution: number of failures before `n` successes, each with probability `p`. Equivalent to `numpy.random.Generator.negative_binomial`.",
        args: [
          { name: "n", type: "number", desc: "Number of successes (> 0)." },
          { name: "p", type: "number", desc: "Probability of success in (0, 1]." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<int64>",
        example: `const rng = np.random.defaultRng(42);
rng.negativeBinomial(5, 0.5);  // => 6`,
      },
      {
        name: "poisson",
        sig: "rng.poisson(lam?, size?)",
        desc: "Draw samples from a Poisson distribution with expected rate `lam`. Equivalent to `numpy.random.Generator.poisson`.",
        args: [
          { name: "[lam]", type: "number", desc: "Expected number of events (>= 0). Default 1.0." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<int64>",
        example: `const rng = np.random.defaultRng(42);
typeof rng.poisson(5);          // => "number"
rng.poisson(0);                 // => 0`,
      },
      {
        name: "zipf",
        sig: "rng.zipf(a, size?)",
        desc: "Draw samples from a Zipf distribution with parameter `a` (> 1). Equivalent to `numpy.random.Generator.zipf`.",
        args: [
          { name: "a", type: "number", desc: "Distribution parameter (> 1)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<int64>",
        example: `const rng = np.random.defaultRng(42);
rng.zipf(2) >= 1;  // => true`,
      },
      {
        name: "hypergeometric",
        sig: "rng.hypergeometric(ngood, nbad, nsample, size?)",
        desc: "Draw samples from a hypergeometric distribution: number of successes in `nsample` draws without replacement from a population of `ngood` good and `nbad` bad items. Equivalent to `numpy.random.Generator.hypergeometric`.",
        args: [
          { name: "ngood", type: "number", desc: "Number of good items in the population (>= 0)." },
          { name: "nbad", type: "number", desc: "Number of bad items in the population (>= 0)." },
          { name: "nsample", type: "number", desc: "Number of draws (>= 0, <= ngood + nbad)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<int64>",
        example: `const rng = np.random.defaultRng(42);
rng.hypergeometric(0, 0, 0);  // => 0`,
      },
      {
        name: "logseries",
        sig: "rng.logseries(p, size?)",
        desc: "Draw samples from a logarithmic series distribution with parameter `p` in [0, 1). Equivalent to `numpy.random.Generator.logseries`.",
        args: [
          { name: "p", type: "number", desc: "Distribution parameter in [0, 1)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<int64>",
        example: `const rng = np.random.defaultRng(42);
rng.logseries(0.9) >= 1;  // => true`,
      },
    ],
  },
  {
    id: "random-continuous",
    title: "Random — continuous distributions (Generator)",
    intro: "Continuous distribution samplers on `np.random.Generator`. Obtain a generator with `np.random.defaultRng(seed)`. All methods return a JS number when `size` is omitted and a `float64` `NDArray` otherwise.",
    entries: [
      {
        name: "standardExponential",
        sig: "rng.standardExponential(size?)",
        desc: "Draw samples from the standard exponential distribution (rate=1). Equivalent to `numpy.random.Generator.standard_exponential`.",
        args: [{ name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." }],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.standardExponential(); // => 2.4042086039659947`,
      },
      {
        name: "exponential",
        sig: "rng.exponential(scale?, size?)",
        desc: "Draw samples from an exponential distribution with given scale (inverse rate). Equivalent to `numpy.random.Generator.exponential`.",
        args: [
          { name: "[scale]", type: "number", desc: "Scale (1/rate). Default 1.0." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.exponential(2); // => 4.808417207931989`,
      },
      {
        name: "standardGamma",
        sig: "rng.standardGamma(shape, size?)",
        desc: "Draw samples from a standard Gamma distribution (scale=1). Equivalent to `numpy.random.Generator.standard_gamma`.",
        args: [
          { name: "shape", type: "number", desc: "Shape parameter (> 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.standardGamma(2); // => 2.0918172704999494`,
      },
      {
        name: "gamma",
        sig: "rng.gamma(shape, scale?, size?)",
        desc: "Draw samples from a Gamma distribution. Equivalent to `numpy.random.Generator.gamma`.",
        args: [
          { name: "shape", type: "number", desc: "Shape parameter (> 0)." },
          { name: "[scale]", type: "number", desc: "Scale parameter (> 0). Default 1.0." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.gamma(2, 3); // => 6.275451811499848`,
      },
      {
        name: "beta",
        sig: "rng.beta(a, b, size?)",
        desc: "Draw samples from a Beta distribution with shape parameters `a` and `b`. Equivalent to `numpy.random.Generator.beta`.",
        args: [
          { name: "a", type: "number", desc: "Alpha shape parameter (> 0)." },
          { name: "b", type: "number", desc: "Beta shape parameter (> 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.beta(2, 5); // => 0.24395464376443093`,
      },
      {
        name: "chisquare",
        sig: "rng.chisquare(df, size?)",
        desc: "Draw samples from a chi-square distribution with `df` degrees of freedom. Equivalent to `numpy.random.Generator.chisquare`.",
        args: [
          { name: "df", type: "number", desc: "Degrees of freedom (> 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.chisquare(2); // => 4.808417207931989`,
      },
      {
        name: "f",
        sig: "rng.f(dfnum, dfden, size?)",
        desc: "Draw samples from an F distribution. Equivalent to `numpy.random.Generator.f`.",
        args: [
          { name: "dfnum", type: "number", desc: "Numerator degrees of freedom (> 0)." },
          { name: "dfden", type: "number", desc: "Denominator degrees of freedom (> 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.f(2, 5); // => 6.208757071292293`,
      },
      {
        name: "standardCauchy",
        sig: "rng.standardCauchy(size?)",
        desc: "Draw samples from the standard Cauchy distribution (location=0, scale=1). Equivalent to `numpy.random.Generator.standard_cauchy`.",
        args: [{ name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." }],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
typeof rng.standardCauchy(); // => "number"`,
      },
      {
        name: "pareto",
        sig: "rng.pareto(a, size?)",
        desc: "Draw samples from a Pareto II (Lomax) distribution with shape `a`. Equivalent to `numpy.random.Generator.pareto`.",
        args: [
          { name: "a", type: "number", desc: "Shape parameter (> 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.pareto(3) >= 0; // => true`,
      },
      {
        name: "weibull",
        sig: "rng.weibull(a, size?)",
        desc: "Draw samples from a Weibull distribution with shape `a`. Equivalent to `numpy.random.Generator.weibull`.",
        args: [
          { name: "a", type: "number", desc: "Shape parameter (>= 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.weibull(1); // => 2.4042086039659947`,
      },
      {
        name: "Generator.power",
        sig: "rng.power(a, size?)",
        desc: "Draw samples from a power distribution with exponent `a` in (0, 1]. Equivalent to `numpy.random.Generator.power`.",
        args: [
          { name: "a", type: "number", desc: "Exponent in (0, 1]." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.power(0.5); // => 0.8274868469979773`,
      },
      {
        name: "laplace",
        sig: "rng.laplace(loc?, scale?, size?)",
        desc: "Draw samples from the Laplace (double exponential) distribution. Equivalent to `numpy.random.Generator.laplace`.",
        args: [
          { name: "[loc]", type: "number", desc: "Location (mean). Default 0.0." },
          { name: "[scale]", type: "number", desc: "Scale (>= 0). Default 1.0." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
typeof rng.laplace(); // => "number"`,
      },
      {
        name: "gumbel",
        sig: "rng.gumbel(loc?, scale?, size?)",
        desc: "Draw samples from a Gumbel distribution. Equivalent to `numpy.random.Generator.gumbel`.",
        args: [
          { name: "[loc]", type: "number", desc: "Location. Default 0.0." },
          { name: "[scale]", type: "number", desc: "Scale (>= 0). Default 1.0." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
typeof rng.gumbel(); // => "number"`,
      },
      {
        name: "logistic",
        sig: "rng.logistic(loc?, scale?, size?)",
        desc: "Draw samples from a logistic distribution. Equivalent to `numpy.random.Generator.logistic`.",
        args: [
          { name: "[loc]", type: "number", desc: "Location. Default 0.0." },
          { name: "[scale]", type: "number", desc: "Scale (>= 0). Default 1.0." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
typeof rng.logistic(); // => "number"`,
      },
      {
        name: "lognormal",
        sig: "rng.lognormal(mean?, sigma?, size?)",
        desc: "Draw samples from a log-normal distribution. Equivalent to `numpy.random.Generator.lognormal`.",
        args: [
          { name: "[mean]", type: "number", desc: "Mean of the underlying normal. Default 0.0." },
          { name: "[sigma]", type: "number", desc: "Std dev of underlying normal (>= 0). Default 1.0." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.lognormal() > 0; // => true`,
      },
      {
        name: "rayleigh",
        sig: "rng.rayleigh(scale?, size?)",
        desc: "Draw samples from a Rayleigh distribution. Equivalent to `numpy.random.Generator.rayleigh`.",
        args: [
          { name: "[scale]", type: "number", desc: "Scale parameter (>= 0). Default 1.0." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.rayleigh() > 0; // => true`,
      },
      {
        name: "standardT",
        sig: "rng.standardT(df, size?)",
        desc: "Draw samples from a standard Student's t-distribution. Equivalent to `numpy.random.Generator.standard_t`.",
        args: [
          { name: "df", type: "number", desc: "Degrees of freedom (> 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
typeof rng.standardT(2); // => "number"`,
      },
      {
        name: "noncentralChisquare",
        sig: "rng.noncentralChisquare(df, nonc, size?)",
        desc: "Draw samples from a noncentral chi-square distribution. Equivalent to `numpy.random.Generator.noncentral_chisquare`.",
        args: [
          { name: "df", type: "number", desc: "Degrees of freedom (> 0)." },
          { name: "nonc", type: "number", desc: "Noncentrality parameter (>= 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.noncentralChisquare(3, 1) > 0; // => true`,
      },
      {
        name: "noncentralF",
        sig: "rng.noncentralF(dfnum, dfden, nonc, size?)",
        desc: "Draw samples from a noncentral F distribution. Equivalent to `numpy.random.Generator.noncentral_f`.",
        args: [
          { name: "dfnum", type: "number", desc: "Numerator degrees of freedom (> 0)." },
          { name: "dfden", type: "number", desc: "Denominator degrees of freedom (> 0)." },
          { name: "nonc", type: "number", desc: "Noncentrality parameter (>= 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.noncentralF(3, 5, 1) > 0; // => true`,
      },
      {
        name: "wald",
        sig: "rng.wald(mean, scale, size?)",
        desc: "Draw samples from a Wald (inverse Gaussian) distribution. Equivalent to `numpy.random.Generator.wald`.",
        args: [
          { name: "mean", type: "number", desc: "Distribution mean (> 0)." },
          { name: "scale", type: "number", desc: "Scale parameter (> 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
rng.wald(1, 1) > 0; // => true`,
      },
      {
        name: "vonmises",
        sig: "rng.vonmises(mu, kappa, size?)",
        desc: "Draw samples from a von Mises distribution. Equivalent to `numpy.random.Generator.vonmises`.",
        args: [
          { name: "mu", type: "number", desc: "Mode angle in radians." },
          { name: "kappa", type: "number", desc: "Concentration parameter (>= 0)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
typeof rng.vonmises(0, 1); // => "number"`,
      },
      {
        name: "triangular",
        sig: "rng.triangular(left, mode, right, size?)",
        desc: "Draw samples from a triangular distribution over `[left, right]` with peak at `mode`. Equivalent to `numpy.random.Generator.triangular`.",
        args: [
          { name: "left", type: "number", desc: "Lower bound." },
          { name: "mode", type: "number", desc: "Peak value in [left, right]." },
          { name: "right", type: "number", desc: "Upper bound (> left)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape; omit for a scalar." },
        ],
        returns: "number | NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
const v = rng.triangular(0, 0.5, 1);
v >= 0 && v <= 1; // => true`,
      },
    ],
  },
  {
    id: "random-multivariate",
    title: "Random — multivariate distributions (Generator)",
    intro: "Multivariate samplers on `np.random.Generator`. All methods return an `NDArray` (never a scalar, since the output is always multi-dimensional).",
    entries: [
      {
        name: "multinomial",
        sig: "rng.multinomial(n, pvals, size?)",
        desc: "Draw samples from a multinomial distribution: `n` experiments, each with outcome probabilities `pvals`. Returns int64 NDArray of shape `(*size, d)`. Equivalent to `numpy.random.Generator.multinomial`.",
        args: [
          { name: "n", type: "number", desc: "Number of experiments (non-negative integer)." },
          { name: "pvals", type: "number[]", desc: "Outcome probabilities (sum <= 1)." },
          { name: "[size]", type: "number | number[]", desc: "Number of samples. Default: one sample." },
        ],
        returns: "NDArray<int64>",
        example: `const rng = np.random.defaultRng(42);
rng.multinomial(10, [0.5, 0.3, 0.2]).toArray(); // => [6, 3, 1]`,
      },
      {
        name: "dirichlet",
        sig: "rng.dirichlet(alpha, size?)",
        desc: "Draw samples from a Dirichlet distribution. Returns float64 NDArray of shape `(*size, d)`. Equivalent to `numpy.random.Generator.dirichlet`.",
        args: [
          { name: "alpha", type: "number[]", desc: "Concentration parameters (all > 0)." },
          { name: "[size]", type: "number | number[]", desc: "Number of samples. Default: one sample." },
        ],
        returns: "NDArray<float64>",
        example: `const rng = np.random.defaultRng(42);
const d = rng.dirichlet([1, 1, 1]);
d.shape; // => [3]`,
      },
      {
        name: "multivariateHypergeometric",
        sig: "rng.multivariateHypergeometric(colors, nsample, size?, method?)",
        desc: "Draw samples from a multivariate hypergeometric distribution: take `nsample` items without replacement from groups of sizes `colors`. Returns int64 NDArray of shape `(*size, len(colors))`. Equivalent to `numpy.random.Generator.multivariate_hypergeometric`.",
        args: [
          { name: "colors", type: "number[]", desc: "Number of items per color group." },
          { name: "nsample", type: "number", desc: "Number of items to sample." },
          { name: "[size]", type: "number | number[]", desc: "Number of samples." },
          { name: "[method]", type: "\"count\" | \"marginals\"", desc: "Sampling algorithm. Default `\"marginals\"`." },
        ],
        returns: "NDArray<int64>",
        example: `const rng = np.random.defaultRng(42);
rng.multivariateHypergeometric([3, 5, 2], 4).toArray(); // => [2, 1, 1]`,
      },
      {
        name: "permuted",
        sig: "rng.permuted(x, axis?)",
        desc: "Return a copy of `x` with the lanes along `axis` independently and randomly shuffled. Unlike `shuffle` (which shuffles axis-0 slices jointly), `permuted` shuffles each lane independently. Equivalent to `numpy.random.Generator.permuted`.",
        args: [
          { name: "x", type: "NDArray", desc: "Input array (at least 1-dimensional)." },
          { name: "[axis]", type: "number", desc: "Axis along which to permute. Default `0`." },
        ],
        returns: "NDArray",
        example: `const rng = np.random.defaultRng(0);
const a = np.arange(6).reshape(2, 3);
const p = rng.permuted(a, 1);
p.shape; // => [2, 3]`,
      },
    ],
  },
  {
    id: "random-state",
    title: "Random state",
    entries: [
      {
        name: "get_state",
        sig: "np.random.get_state()",
        desc: "Return the internal state of the global RandomState as a bigint array (MT19937 words). Can be restored with `set_state`. Equivalent to `numpy.random.get_state`.",
        args: [],
        returns: "bigint[]",
        example: `const state = np.random.get_state();
typeof state; // => "object"`,
      },
      {
        name: "set_state",
        sig: "np.random.set_state(words)",
        desc: "Restore the global RandomState from a bigint array returned by `get_state`. Equivalent to `numpy.random.set_state`.",
        args: [{ name: "words", type: "bigint[]", desc: "State words from get_state()." }],
        returns: "void",
        example: `const state = np.random.get_state();
np.random.set_state(state);
typeof state; // => "object"`,
      },
      {
        name: "ranf",
        sig: "np.random.ranf(size?)",
        desc: "Return random floats in [0.0, 1.0). Alias for `np.random.random`. Equivalent to `numpy.random.ranf`.",
        args: [{ name: "[size]", type: "number | number[]", desc: "Output shape." }],
        returns: "NDArray | number",
        example: `np.random.ranf([3]).shape; // => [3]`,
      },
      {
        name: "sample",
        sig: "np.random.sample(size?)",
        desc: "Alias for `np.random.random`. Return random floats in [0.0, 1.0). Equivalent to `numpy.random.sample`.",
        args: [{ name: "[size]", type: "number | number[]", desc: "Output shape." }],
        returns: "NDArray | number",
        example: `np.random.sample([2, 3]).shape; // => [2, 3]`,
      },
      {
        name: "random_integers",
        sig: "np.random.random_integers(low, high?, size?)",
        desc: "Return random integers from `low` (inclusive) to `high` (inclusive). If `high` is null, returns integers in [1, low]. Deprecated in NumPy — use `randint` instead. Equivalent to `numpy.random.random_integers`.",
        args: [
          { name: "low", type: "number", desc: "Lowest integer (or highest if high omitted)." },
          { name: "[high]", type: "number | null", desc: "Highest integer (inclusive)." },
          { name: "[size]", type: "number | number[]", desc: "Output shape." },
        ],
        returns: "NDArray | number",
        example: `np.random.random_integers(5).valueOf() <= 5; // => true`,
      },
    ],
  },
];
