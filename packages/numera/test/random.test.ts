import { describe, expect, it } from "vitest";
import np, { NDArray } from "../src/index.js";

// Reference values from NumPy 2.x (bit-exact; D-019).
describe("random: Generator (defaultRng / PCG64)", () => {
  it("reproduces NumPy streams", () => {
    expect(np.random.defaultRng(42).random()).toBe(0.7739560485559633);
    expect((np.random.defaultRng(42).standardNormal(3) as NDArray).toArray()).toEqual([
      0.30471707975443135, -1.0399841062404955, 0.7504511958064572,
    ]);
    expect((np.random.defaultRng(42).integers(0, 10, 8) as NDArray).toArray()).toEqual([
      0, 7, 6, 4, 4, 8, 0, 6,
    ]);
    expect(np.random.defaultRng([1, 2, 3]).random()).toBe(0.6704722626516632);
    expect(np.random.defaultRng(2n ** 100n + 7n).random()).toBe(0.0653230385916338);
  });

  it("same seed → same stream; state advances", () => {
    const a = np.random.defaultRng(7);
    const b = np.random.defaultRng(7);
    expect((a.random(4) as NDArray).toArray()).toEqual((b.random(4) as NDArray).toArray());
    expect(a.random()).not.toBe(np.random.defaultRng(7).random());
  });

  it("returns scalars without size and dtype/shape with size", () => {
    const g = np.random.defaultRng(1);
    expect(typeof g.random()).toBe("number");
    const r = g.random([2, 3], "float32") as NDArray;
    expect(r.shape).toEqual([2, 3]);
    expect(r.dtype.name).toBe("float32");
    const b = g.integers({ low: 0, high: 2, size: 5, dtype: "bool" }) as NDArray;
    expect(b.dtype.name).toBe("bool");
  });

  it("choice / permutation / shuffle", () => {
    expect((np.random.defaultRng(42).choice(10, 4, false) as NDArray).toArray()).toEqual([4, 0, 5, 6]);
    expect(np.random.defaultRng(42).permutation(6).toArray()).toEqual([3, 2, 5, 4, 1, 0]);
    expect((np.random.defaultRng(42).choice([10, 20, 30, 40], [2]) as NDArray).toArray()).toEqual([10, 40]);
    const x = np.arange(10);
    np.random.defaultRng(3).shuffle(x);
    expect([...(x.toArray() as number[])].sort((p, q) => p - q)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  });

  it("raises NumPy errors", () => {
    const g = np.random.defaultRng(0);
    expect(() => g.normal(0, -1)).toThrow(np.ValueError);
    expect(() => g.integers(5, 5)).toThrow(np.ValueError);
    expect(() => g.integers(0, 300, 1, "uint8")).toThrow(np.ValueError);
    expect(() => g.choice(3, 5, false)).toThrow(np.ValueError);
    expect(() => g.choice(3, 2, true, [0.5, 0.25, 0.25])).not.toThrow();
    expect(() => g.shuffle(np.broadcastTo(np.arange(3), [2, 3]))).toThrow(np.ValueError);
  });
});

describe("random: legacy RandomState / global functions (MT19937)", () => {
  it("reproduces np.random.seed streams", () => {
    np.random.seed(42);
    expect((np.random.randn(3) as NDArray).toArray()).toEqual([
      0.4967141530112327, -0.13826430117118466, 0.6476885381006925,
    ]);
    np.random.seed(42);
    expect((np.random.randint(0, 10, 8) as NDArray).toArray()).toEqual([6, 3, 7, 4, 6, 9, 2, 6]);
    np.random.seed(42);
    expect(np.random.permutation(6).toArray()).toEqual([0, 1, 5, 2, 4, 3]);
    np.random.seed(42);
    expect((np.random.choice(10, 4, false) as NDArray).toArray()).toEqual([8, 1, 5, 0]);
  });

  it("RandomState instances are independent of global state", () => {
    const rs = new np.random.RandomState(42);
    np.random.seed(0);
    expect((rs.randn(1) as NDArray).toArray()).toEqual([0.4967141530112327]);
  });

  it("validates seeds", () => {
    expect(() => np.random.seed(-1)).toThrow(np.ValueError);
    expect(() => np.random.seed(2 ** 32)).toThrow(np.ValueError);
  });
});
