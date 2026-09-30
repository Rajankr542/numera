import { describe, expect, it } from "vitest";
import np, { IndexError, ValueError } from "../src/index.js";

describe("np.array bulk number path (PLAN §80)", () => {
  it("matches the per-element path for inference, ranges and errors", () => {
    expect(np.array([[1, 2], [3, 4]]).dtype).toBe(np.int64);
    expect(np.array([1, -0]).dtype).toBe(np.float64);
    expect(np.array([1, 2 ** 53]).dtype).toBe(np.float64);
    expect(np.array([[], []]).shape).toEqual([2, 0]);
    expect(np.array([]).dtype).toBe(np.float64);
    expect(np.array([1.9, -1.9], { dtype: "int8" }).toArray()).toEqual([1, -1]);
    expect(np.array([NaN, 1], { dtype: "float32" }).toArray()).toEqual([NaN, 1]);
    expect(() => np.array([1, 300], { dtype: "int8" })).toThrow(ValueError);
    expect(() => np.array([1, NaN], { dtype: "int32" })).toThrow(/NaN/);
    // Mixed / ragged / sparse inputs take the fallback path.
    expect(np.array([1, true]).dtype).toBe(np.int64);
    expect(np.array([1, 2n]).dtype).toBe(np.int64);
    expect(() => np.array([[1, 2], [3]] as never)).toThrow(ValueError);
    // eslint-disable-next-line no-sparse-arrays
    expect(() => np.array([1, , 3] as never)).toThrow(ValueError);
  });
});

describe("M2 creation routines", () => {
  it("ones / full / *Like", () => {
    expect(np.ones([2, 2], { dtype: "int8" }).toArray()).toEqual([
      [1, 1],
      [1, 1],
    ]);
    expect(np.full(3, 5).dtype).toBe(np.int64);
    expect(np.full(2, true).toArray()).toEqual([true, true]);
    expect(np.full(2, 2.5, { dtype: "float32" }).toArray()).toEqual([2.5, 2.5]);
    expect(() => np.full(3, 300, { dtype: "int8" })).toThrow(ValueError);
    const a = np.arange(6).reshape([2, 3]);
    expect(np.zerosLike(a).dtype).toBe(np.int64);
    expect(np.onesLike(a, { dtype: "float32" }).shape).toEqual([2, 3]);
    expect(np.fullLike(a, 7).toArray()).toEqual([
      [7, 7, 7],
      [7, 7, 7],
    ]);
    expect(np.emptyLike(a).shape).toEqual([2, 3]);
  });

  it("asarray avoids copies for matching NDArrays", () => {
    const a = np.arange(3);
    expect(np.asarray(a)).toBe(a);
    expect(np.asarray(a, { dtype: "int64" })).toBe(a);
    expect(np.asarray(a, { dtype: "float64" }).dtype).toBe(np.float64);
    expect(np.asarray([1, 2]).toArray()).toEqual([1, 2]);
  });

  it("arange", () => {
    expect(np.arange(5).toArray()).toEqual([0, 1, 2, 3, 4]);
    expect(np.arange(10, 0, -3).toArray()).toEqual([10, 7, 4, 1]);
    expect(np.arange(0, 1, 0.25).dtype).toBe(np.float64);
    expect(np.arange(0, 1, 0.3, { dtype: "int32" }).toArray()).toEqual([0, 0, 0, 0]);
    expect(np.arange(5, 0).size).toBe(0);
    expect(() => np.arange(0, 5, 0)).toThrow(ValueError);
    expect(() => np.arange(NaN)).toThrow(ValueError);
    expect(() => np.arange(3, undefined, undefined, { dtype: "bool" })).toThrow(ValueError);
  });

  it("linspace / eye / identity", () => {
    expect(np.linspace(0, 1, 5).toArray()).toEqual([0, 0.25, 0.5, 0.75, 1]);
    expect(np.linspace(0, 1, 4, { endpoint: false }).toArray()).toEqual([0, 0.25, 0.5, 0.75]);
    expect(np.linspace(-1, 1, 5, { dtype: "int32" }).toArray()).toEqual([-1, -1, 0, 0, 1]);
    expect(np.linspace(0, 1).size).toBe(50);
    expect(() => np.linspace(0, 1, -1)).toThrow(ValueError);
    expect(np.eye(2, 3, { k: 1, dtype: "int32" }).toArray()).toEqual([
      [0, 1, 0],
      [0, 0, 1],
    ]);
    expect(np.identity(2).toArray()).toEqual([
      [1, 0],
      [0, 1],
    ]);
  });
});

describe("M3 shape manipulation", () => {
  it("transpose / T are views with permuted strides", () => {
    const a = np.arange(24).reshape([2, 3, 4]);
    const t = a.T;
    expect(t.shape).toEqual([4, 3, 2]);
    expect(t.strides).toEqual([8, 32, 96]);
    expect(np.mayShareMemory(a, t)).toBe(true);
    expect(np.transpose(a, [1, 0, 2]).shape).toEqual([3, 2, 4]);
    expect(a.transpose(2, 0, 1).shape).toEqual([4, 2, 3]);
    expect(() => a.transpose(0, 0, 1)).toThrow(ValueError);
    expect(() => a.transpose(0, 1, 5)).toThrow(IndexError);
  });

  it("squeeze / expandDims / swapAxes / moveAxis", () => {
    const a = np.zeros([1, 3, 1]);
    expect(a.squeeze().shape).toEqual([3]);
    expect(np.squeeze(a, -1).shape).toEqual([1, 3]);
    expect(() => a.squeeze(1)).toThrow(ValueError);
    expect(np.expandDims(np.zeros([2, 3]), -1).strides).toEqual([24, 8, 8]);
    expect(np.expandDims(np.zeros([2, 3]), [0, 2]).shape).toEqual([1, 2, 1, 3]);
    expect(np.swapAxes(np.zeros([2, 3, 4]), 0, 2).shape).toEqual([4, 3, 2]);
    expect(np.moveAxis(np.zeros([2, 3, 4]), 0, -1).shape).toEqual([3, 4, 2]);
  });

  it("ravel is a view when contiguous; flatten always copies", () => {
    const a = np.arange(6).reshape([2, 3]);
    expect(np.mayShareMemory(a, a.ravel())).toBe(true);
    expect(np.mayShareMemory(a, a.flatten())).toBe(false);
    expect(a.T.ravel().toArray()).toEqual([0, 3, 1, 4, 2, 5]);
    expect(np.mayShareMemory(a, a.T.ravel())).toBe(false);
  });
});
