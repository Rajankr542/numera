// P13-3: discrete Generator distributions (D-161).
// Expected values from NumPy 2.5.3 with np.random.default_rng(seed).

import { describe, expect, it } from "vitest";
import np, { NDArray } from "../src/index.js";

// Ensure the p13_discrete augmentation is loaded.
import "../src/p13_discrete.js";

describe("P13-3 Generator.binomial", () => {
  it("scalar matches NumPy (seed 42)", () => {
    expect(np.random.defaultRng(42).binomial(100, 0.3)).toBe(33);
  });

  it("array matches NumPy (seed 42)", () => {
    expect(
      (np.random.defaultRng(42).binomial(10, 0.5, 5) as NDArray).toArray(),
    ).toEqual([6, 5, 7, 6, 3]);
  });

  it("boundary: n=0 returns 0", () => {
    expect(np.random.defaultRng(0).binomial(0, 0.5)).toBe(0);
  });

  it("boundary: p=0 returns 0", () => {
    expect(np.random.defaultRng(0).binomial(10, 0)).toBe(0);
  });

  it("boundary: p=1 returns n", () => {
    expect(np.random.defaultRng(0).binomial(10, 1)).toBe(10);
  });

  it("raises: n < 0", () => {
    expect(() => np.random.defaultRng(0).binomial(-1, 0.5)).toThrow(np.ValueError);
  });

  it("raises: p < 0", () => {
    expect(() => np.random.defaultRng(0).binomial(10, -0.1)).toThrow(np.ValueError);
  });

  it("raises: p > 1", () => {
    expect(() => np.random.defaultRng(0).binomial(10, 1.1)).toThrow(np.ValueError);
  });
});

describe("P13-3 Generator.negativeBinomial", () => {
  it("scalar matches NumPy (seed 42)", () => {
    // After binomial draw from default_rng(42), negativeBinomial is next.
    // Use fresh seed directly on negativeBinomial for isolation.
    const rng = np.random.defaultRng(42);
    rng.binomial(100, 0.3); // advance stream to match sequential test
    expect(rng.negativeBinomial(5, 0.5)).toBe(3);
  });

  it("array matches NumPy (seed 42)", () => {
    expect(
      (np.random.defaultRng(42).negativeBinomial(3, 0.5, 4) as NDArray).toArray(),
    ).toEqual([5, 5, 7, 4]);
  });

  it("raises: n <= 0", () => {
    expect(() => np.random.defaultRng(0).negativeBinomial(0, 0.5)).toThrow(np.ValueError);
  });

  it("raises: p <= 0", () => {
    expect(() => np.random.defaultRng(0).negativeBinomial(5, 0)).toThrow(np.ValueError);
  });

  it("raises: p > 1", () => {
    expect(() => np.random.defaultRng(0).negativeBinomial(5, 1.1)).toThrow(np.ValueError);
  });
});

describe("P13-3 Generator.poisson", () => {
  it("scalar matches NumPy (seed 42)", () => {
    const rng = np.random.defaultRng(42);
    rng.binomial(100, 0.3);
    rng.negativeBinomial(5, 0.5);
    expect(rng.poisson(20)).toBe(24);
  });

  it("array matches NumPy (seed 42)", () => {
    expect(
      (np.random.defaultRng(42).poisson(5, 4) as NDArray).toArray(),
    ).toEqual([8, 7, 7, 2]);
  });

  it("default lam=1", () => {
    expect(typeof np.random.defaultRng(42).poisson()).toBe("number");
  });

  it("lam=0 always returns 0", () => {
    expect(
      (np.random.defaultRng(1).poisson(0, 3) as NDArray).toArray(),
    ).toEqual([0, 0, 0]);
  });

  it("raises: lam < 0", () => {
    expect(() => np.random.defaultRng(0).poisson(-1)).toThrow(np.ValueError);
  });
});

describe("P13-3 Generator.zipf", () => {
  it("scalar matches NumPy (seed 42)", () => {
    const rng = np.random.defaultRng(42);
    rng.binomial(100, 0.3);
    rng.negativeBinomial(5, 0.5);
    rng.poisson(20);
    expect(rng.zipf(2)).toBe(1);
  });

  it("array matches NumPy (seed 42)", () => {
    expect(
      (np.random.defaultRng(42).zipf(2, 4) as NDArray).toArray(),
    ).toEqual([4, 1, 1, 1]);
  });

  it("raises: a <= 1", () => {
    expect(() => np.random.defaultRng(0).zipf(1.0)).toThrow(np.ValueError);
  });
});

describe("P13-3 Generator.geometric", () => {
  it("scalar matches NumPy (seed 42)", () => {
    const rng = np.random.defaultRng(42);
    rng.binomial(100, 0.3);
    rng.negativeBinomial(5, 0.5);
    rng.poisson(20);
    rng.zipf(2);
    expect(rng.geometric(0.3)).toBe(4);
  });

  it("array matches NumPy (seed 42)", () => {
    expect(
      (np.random.defaultRng(42).geometric(0.5, 4) as NDArray).toArray(),
    ).toEqual([3, 1, 3, 2]);
  });

  it("raises: p <= 0", () => {
    expect(() => np.random.defaultRng(0).geometric(0)).toThrow(np.ValueError);
  });

  it("raises: p > 1", () => {
    expect(() => np.random.defaultRng(0).geometric(1.1)).toThrow(np.ValueError);
  });
});

describe("P13-3 Generator.hypergeometric", () => {
  it("scalar matches NumPy (seed 42)", () => {
    const rng = np.random.defaultRng(42);
    rng.binomial(100, 0.3);
    rng.negativeBinomial(5, 0.5);
    rng.poisson(20);
    rng.zipf(2);
    rng.geometric(0.3);
    expect(rng.hypergeometric(10, 20, 5)).toBe(2);
  });

  it("array matches NumPy (seed 42)", () => {
    expect(
      (np.random.defaultRng(42).hypergeometric(10, 20, 8, 3) as NDArray).toArray(),
    ).toEqual([5, 2, 2]);
  });

  it("boundary: all zeros", () => {
    expect(np.random.defaultRng(0).hypergeometric(0, 0, 0)).toBe(0);
  });

  it("raises: ngood < 0", () => {
    expect(() => np.random.defaultRng(0).hypergeometric(-1, 20, 5)).toThrow(np.ValueError);
  });

  it("raises: nbad < 0", () => {
    expect(() => np.random.defaultRng(0).hypergeometric(10, -1, 5)).toThrow(np.ValueError);
  });

  it("raises: nsample < 0", () => {
    expect(() => np.random.defaultRng(0).hypergeometric(10, 20, -1)).toThrow(np.ValueError);
  });

  it("raises: nsample > ngood + nbad", () => {
    expect(() => np.random.defaultRng(0).hypergeometric(10, 20, 31)).toThrow(np.ValueError);
  });
});

describe("P13-3 Generator.logseries", () => {
  it("scalar matches NumPy (seed 42)", () => {
    const rng = np.random.defaultRng(42);
    rng.binomial(100, 0.3);
    rng.negativeBinomial(5, 0.5);
    rng.poisson(20);
    rng.zipf(2);
    rng.geometric(0.3);
    rng.hypergeometric(10, 20, 5);
    expect(rng.logseries(0.9)).toBe(5);
  });

  it("array matches NumPy (seed 42)", () => {
    expect(
      (np.random.defaultRng(42).logseries(0.9, 4) as NDArray).toArray(),
    ).toEqual([1, 1, 22, 2]);
  });

  it("raises: p < 0", () => {
    expect(() => np.random.defaultRng(0).logseries(-0.1)).toThrow(np.ValueError);
  });

  it("raises: p >= 1", () => {
    expect(() => np.random.defaultRng(0).logseries(1.0)).toThrow(np.ValueError);
  });
});

describe("P13-3 Generator discrete: reproducibility", () => {
  it("same seed produces same sequence for all discrete distributions", () => {
    const rng1 = np.random.defaultRng(123);
    const rng2 = np.random.defaultRng(123);
    expect(rng1.binomial(50, 0.4)).toBe(rng2.binomial(50, 0.4));
    expect(rng1.poisson(7)).toBe(rng2.poisson(7));
    expect(rng1.geometric(0.2)).toBe(rng2.geometric(0.2));
    expect(rng1.zipf(3)).toBe(rng2.zipf(3));
    expect(rng1.negativeBinomial(4, 0.6)).toBe(rng2.negativeBinomial(4, 0.6));
    expect(rng1.hypergeometric(5, 15, 4)).toBe(rng2.hypergeometric(5, 15, 4));
    expect(rng1.logseries(0.7)).toBe(rng2.logseries(0.7));
  });

  it("returns integer scalars without size and int64 NDArray with size", () => {
    const g = np.random.defaultRng(1);
    expect(typeof g.binomial(10, 0.5)).toBe("number");
    const arr = g.binomial(10, 0.5, [2, 3]) as NDArray;
    expect(arr.shape).toEqual([2, 3]);
    expect(arr.dtype.name).toBe("int64");
  });
});
