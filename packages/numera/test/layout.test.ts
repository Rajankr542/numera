import { describe, expect, it } from "vitest";
import np, { ValueError } from "../src/index.js";

describe("memory order (P3-1, D-055)", () => {
  it("creation order C/F", () => {
    expect(np.zeros([2, 3], { order: "F" }).strides).toEqual([8, 16]);
    expect(np.empty([2, 3], { order: "F" }).flags.fContiguous).toBe(true);
    const o = np.ones([2, 3], { order: "F" });
    expect(o.strides).toEqual([8, 16]);
    expect(o.toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    const f = np.full([2, 2], 7, { order: "F", dtype: "int32" });
    expect(f.strides).toEqual([4, 8]);
    expect(f.toArray()).toEqual([[7, 7], [7, 7]]);
    expect(() => np.zeros([2], { order: "K" as never })).toThrow(ValueError);
    expect(() => np.ones([2], { order: "A" as never })).toThrow(ValueError);
  });

  it("array/copy order", () => {
    const a = np.array([[1, 2, 3], [4, 5, 6]], { order: "F" });
    expect(a.strides).toEqual([8, 16]);
    expect(a.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
    expect(np.copy(a).strides).toEqual([8, 16]); // K
    expect(a.copy().strides).toEqual([24, 8]); // NDArray.copy default C
    expect(a.copy({ order: "A" }).strides).toEqual([8, 16]);
    expect(np.array(a).strides).toEqual([8, 16]);
    expect(np.copy([1, 2]).toArray()).toEqual([1, 2]);
  });

  it("*Like keeps layout (K)", () => {
    const a = np.ones([2, 3], { order: "F" });
    expect(np.zerosLike(a).strides).toEqual([8, 16]);
    expect(np.onesLike(a).toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    expect(np.fullLike(a, 3).strides).toEqual([8, 16]);
    expect(np.fullLike(a, 3).toArray()).toEqual([[3, 3, 3], [3, 3, 3]]);
    expect(np.emptyLike(a).strides).toEqual([8, 16]);
    expect(np.zerosLike(a, { order: "C" }).strides).toEqual([24, 8]);
  });

  it("reshape/ravel/flatten order", () => {
    const a = np.arange(6).reshape(2, 3);
    expect(a.ravel({ order: "F" }).toArray()).toEqual([0, 3, 1, 4, 2, 5]);
    expect(a.flatten({ order: "F" }).toArray()).toEqual([0, 3, 1, 4, 2, 5]);
    expect(np.ravel(a, { order: "F" }).toArray()).toEqual([0, 3, 1, 4, 2, 5]);
    expect(a.reshape([3, 2], { order: "F" }).toArray()).toEqual([[0, 4], [3, 2], [1, 5]]);
    expect(a.reshape(3, 2, { order: "F" }).toArray()).toEqual([[0, 4], [3, 2], [1, 5]]);
    expect(np.reshape(a, [3, 2], { order: "F" }).toArray()).toEqual([[0, 4], [3, 2], [1, 5]]);
    const t = a.T;
    expect(t.ravel({ order: "K" }).toArray()).toEqual([0, 1, 2, 3, 4, 5]);
    expect(np.mayShareMemory(t.ravel({ order: "K" }), a)).toBe(true);
    expect(() => a.reshape([6], { order: "K" })).toThrow(ValueError);
    expect(() => a.ravel({ order: "Z" as never })).toThrow(ValueError);
  });

  it("astype order/copy", () => {
    const a = np.ones([2, 3], { order: "F" });
    expect(a.astype("float32").strides).toEqual([4, 8]);
    expect(a.astype("float32", { order: "C" }).strides).toEqual([12, 4]);
    expect(a.astype("float64", { copy: false })).toBe(a);
    expect(a.astype("float64", { copy: false, order: "C" })).not.toBe(a);
  });

  it("ascontiguousarray / asfortranarray", () => {
    const a = np.arange(6).reshape(2, 3);
    expect(np.ascontiguousarray(a)).toBe(a);
    expect(np.ascontiguousarray(a.T).flags.cContiguous).toBe(true);
    expect(np.asfortranarray(a).strides).toEqual([8, 16]);
    expect(np.ascontiguousarray(np.array(5)).shape).toEqual([1]);
    expect(np.asfortranarray(np.array(5)).shape).toEqual([1]);
  });
});
