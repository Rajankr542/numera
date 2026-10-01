import { describe, expect, it } from "vitest";
import np, { IndexError, NotImplementedError, ValueError } from "../src/index.js";

describe("reductions (M7, D-017)", () => {
  const a = np.arange(6).reshape([2, 3]);

  it("sum/prod/min/max over all axes return 0-d arrays", () => {
    const s = np.sum(a);
    expect(s.shape).toEqual([]);
    expect(s.item()).toBe(15);
    expect(s.dtype.name).toBe("int64");
    expect(np.prod(np.arange(1, 5)).item()).toBe(24);
    expect(np.max(a).item()).toBe(5);
    expect(a.min().item()).toBe(0);
  });

  it("axis, negative axis, axis tuples and keepdims", () => {
    expect(np.sum(a, { axis: 0 }).toArray()).toEqual([3, 5, 7]);
    expect(a.sum({ axis: -1 }).toArray()).toEqual([3, 12]);
    expect(a.sum({ axis: 1, keepdims: true }).shape).toEqual([2, 1]);
    expect(a.sum({ axis: [0, 1], keepdims: true }).toArray()).toEqual([[15]]);
    expect(a.T.max({ axis: 0 }).toArray()).toEqual([2, 5]);
    expect(() => a.sum({ axis: 2 })).toThrow(IndexError);
    expect(() => a.sum({ axis: [0, -2] })).toThrow(ValueError);
  });

  it("result dtypes follow NumPy", () => {
    expect(np.sum(np.array([1, 2], { dtype: "uint8" })).dtype.name).toBe("uint64");
    expect(np.sum(np.array([true, true, false])).item()).toBe(2);
    expect(np.mean(np.array([1, 2], { dtype: "int32" })).dtype.name).toBe("float64");
    expect(np.mean(np.array([1, 2], { dtype: "float32" })).dtype.name).toBe("float32");
    expect(np.max(np.array([1, 2], { dtype: "int16" })).dtype.name).toBe("int16");
    expect(np.sum(np.array([100, 100], { dtype: "int8" }), { dtype: "int8" }).item()).toBe(-56);
    // Complex mean is P1-3b; complex sum/prod are covered in complex vitest cases (P1-3e).
    expect(() => np.mean(np.zeros([2], { dtype: "complex128" }))).toThrow(NotImplementedError);
  });

  it("initial, empty inputs and NaN", () => {
    expect(np.sum(a, { initial: 10 }).item()).toBe(25);
    expect(np.max(np.zeros([0]), { initial: -1 }).item()).toBe(-1);
    expect(() => np.max(np.zeros([0]))).toThrow(ValueError);
    expect(np.sum(np.zeros([0])).item()).toBe(0);
    expect(Number.isNaN(np.mean(np.zeros([0])).item())).toBe(true);
    const n = np.array([1, NaN, 3]);
    expect(Number.isNaN(np.min(n).item())).toBe(true);
    expect(np.argmax(n).item()).toBe(1);
  });

  it("mean/var/std with ddof", () => {
    expect(np.mean(a, { axis: 0 }).toArray()).toEqual([1.5, 2.5, 3.5]);
    expect(np.var(a).item()).toBeCloseTo(35 / 12, 14);
    expect(a.var({ ddof: 1 }).item()).toBe(3.5);
    expect(np.std(np.array([1, 3])).item()).toBe(1);
  });

  it("argmin/argmax", () => {
    const b = np.array([[3, 1, 3], [0, 5, 5]]);
    expect(np.argmax(b).item()).toBe(4);
    expect(np.argmin(b, { axis: 1 }).toArray()).toEqual([1, 0]);
    expect(b.argmax({ axis: 0, keepdims: true }).toArray()).toEqual([[0, 1, 1]]);
    expect(b.argmax().dtype.name).toBe("int64");
    expect(() => np.argmin(np.zeros([0]))).toThrow(ValueError);
  });
});
