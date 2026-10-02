import { describe, expect, it } from "vitest";
import np, { DTypeError, IndexError, ValueError } from "../src/index.js";

describe("P6 tile / repeat (D-091)", () => {
  it("tile", () => {
    expect(np.tile([1, 2], 2).toArray()).toEqual([1, 2, 1, 2]);
    expect(np.tile([1, 2], [2, 1]).toArray()).toEqual([[1, 2], [1, 2]]);
    expect(np.tile([[1, 2], [3, 4]], 2).toArray()).toEqual([[1, 2, 1, 2], [3, 4, 3, 4]]);
    expect(np.tile(np.array(7), [2, 2]).toArray()).toEqual([[7, 7], [7, 7]]);
    expect(np.tile([1, 2], 0).shape).toEqual([0]);
    expect(np.tile(np.zeros([0, 2]), [2, 2]).shape).toEqual([0, 4]);
    const a = np.array([1, 2], { dtype: "float32" });
    const t = np.tile(a, 1);
    expect(np.mayShareMemory(t, a)).toBe(false);
    expect(t.dtype.name).toBe("float32");
    expect(np.tile([NaN, Infinity], 2).toArray()).toEqual([NaN, Infinity, NaN, Infinity]);
    expect(np.tile(np.array([{ re: 1, im: 2 }]), 2).dtype.name).toBe("complex128");
    expect(() => np.tile([1], -1)).toThrow(ValueError);
    expect(() => np.tile([1], 1.5)).toThrow(DTypeError);
  });
  it("repeat", () => {
    expect(np.repeat(3, 4).toArray()).toEqual([3, 3, 3, 3]);
    expect(np.repeat([[1, 2], [3, 4]], 2).toArray()).toEqual([1, 1, 2, 2, 3, 3, 4, 4]);
    expect(np.repeat([[1, 2], [3, 4]], 3, 1).toArray()).toEqual([[1, 1, 1, 2, 2, 2], [3, 3, 3, 4, 4, 4]]);
    expect(np.repeat([[1, 2], [3, 4]], [1, 2], 0).toArray()).toEqual([[1, 2], [3, 4], [3, 4]]);
    expect(np.repeat([[1, 2], [3, 4]], [1, 2], -1).toArray()).toEqual([[1, 2, 2], [3, 4, 4]]);
    expect(np.repeat([1, 2], np.array([0, 2])).toArray()).toEqual([2, 2]);
    expect(np.repeat([true, false], 2).dtype.name).toBe("bool");
    expect(np.repeat([], 2).shape).toEqual([0]);
    expect(np.repeat(np.arange(4).reshape(2, 2).T, 2, 0).toArray()).toEqual([[0, 2], [0, 2], [1, 3], [1, 3]]);
    expect(() => np.repeat([1, 2, 3], [1, 2])).toThrow(/could not be broadcast/);
    expect(() => np.repeat([1], -1)).toThrow(/negative/);
    expect(() => np.repeat([1], 2, 1)).toThrow(IndexError);
    expect(() => np.repeat([1], np.array([1.5]))).toThrow(DTypeError);
  });
  it("NDArray.repeat", () => {
    const a = np.array([[1, 2], [3, 4]]);
    expect(a.repeat(2).shape).toEqual([8]);
    expect(a.repeat([2, 1], 0).toArray()).toEqual([[1, 2], [1, 2], [3, 4]]);
  });
});
