import { describe, expect, it } from "vitest";
import np, { BroadcastError, DTypeError, NotImplementedError, ValueError } from "../src/index.js";

describe("ufuncs (M4) + broadcasting (M5)", () => {
  it("broadcasts per PLAN §14/§54", () => {
    const r = np.add(np.ones([1000, 3]), np.array([1, 2, 3]));
    expect(r.shape).toEqual([1000, 3]);
    expect(r.toArray()[999]).toEqual([2, 3, 4]);
    expect(np.broadcastShapes([3, 1], [4], [])).toEqual([3, 4]);
    expect(() => np.broadcastShapes([3], [4])).toThrow(BroadcastError);
    expect(() => np.add(np.ones([2, 3]), np.ones([3, 2]))).toThrow(BroadcastError);
  });

  it("broadcastTo returns a zero-stride view", () => {
    const a = np.array([1, 2, 3]);
    const b = np.broadcastTo(a, [2, 3]);
    expect(b.strides).toEqual([0, 8]);
    expect(np.mayShareMemory(a, b)).toBe(true);
    expect(b.toArray()).toEqual([[1, 2, 3], [1, 2, 3]]);
  });

  it("accepts nested lists and NEP 50 weak scalars", () => {
    expect(np.multiply([1, 2], [3, 4]).toArray()).toEqual([3, 8]);
    expect(np.add(np.ones(2, { dtype: "int8" }), 3).dtype).toBe(np.int8);
    expect(np.add(2, np.ones(2, { dtype: "float32" })).dtype).toBe(np.float32);
    expect(np.add(np.ones(2, { dtype: "int32" }), 0.5).dtype).toBe(np.float64);
    expect(np.add(np.array([true]), 1).dtype).toBe(np.int64);
    expect(() => np.add(np.ones(2, { dtype: "uint8" }), -1)).toThrow(ValueError);
    expect(np.subtract(5, 2).toArray()).toBe(3);
  });

  it("reports unsupported loops with typed errors", () => {
    expect(() => np.subtract([true], [false])).toThrow(DTypeError);
    expect(() => np.negative([true])).toThrow(DTypeError);
    expect(() => np.power([2], [-1])).toThrow(ValueError);
    expect(() => np.add(np.zeros(2, { dtype: "complex128" }), 1)).toThrow(NotImplementedError);
  });

  it("works on non-contiguous inputs and returns C-contiguous results", () => {
    const t = np.arange(6).reshape([2, 3]).T;
    const r = np.multiply(t, 2);
    expect(r.flags.cContiguous).toBe(true);
    expect(r.toArray()).toEqual([[0, 6], [2, 8], [4, 10]]);
    expect(np.sqrt(np.array([4, 9], { dtype: "uint8" })).dtype).toBe(np.float16);
    expect(np.abs([-3, 3]).toArray()).toEqual([3, 3]);
    expect(np.floorDivide([-7], [2]).toArray()).toEqual([-4]);
    expect(np.mod([-7], [2]).toArray()).toEqual([1]);
    expect(np.divide([1], [0]).toArray()).toEqual([Infinity]);
  });
});
