import { describe, expect, it } from "vitest";
import np, { BroadcastError, IndexError, ValueError } from "../src/index.js";

describe("indexing (M6)", () => {
  it("PLAN §9: a.slice([null, [1, 3]]) is a view", () => {
    const a = np.arange(12).reshape([3, 4]);
    const b = a.slice([null, [1, 3]]);
    expect(b.toArray()).toEqual([[1, 2], [5, 6], [9, 10]]);
    expect(np.mayShareMemory(a, b)).toBe(true);
    expect(b.strides).toEqual([32, 8]);
    b.set([0, 0], 100);
    expect(a.item(0, 1)).toBe(100);
  });

  it("PLAN §13 examples", () => {
    const a = np.arange(10);
    expect(a.slice([0, 5]).toArray()).toEqual([0, 1, 2, 3, 4]);
    expect(a.slice([[null, null, -1]]).toArray()).toEqual([9, 8, 7, 6, 5, 4, 3, 2, 1, 0]);
    expect(a.slice([null, null, -1]).strides).toEqual([-8]);
    const m = np.arange(64).reshape([8, 8]);
    expect(m.slice([[1, 4], [2, 6]]).shape).toEqual([3, 4]);
    expect(m.get(1, 2).shape).toEqual([]);
    expect(m.get(1, 2).item()).toBe(10);
    expect(m.get(-1).toArray()).toEqual([56, 57, 58, 59, 60, 61, 62, 63]);
    const image = np.zeros([4, 5, 3], { dtype: np.uint8 });
    const red = image.slice([[null, null], [null, null], 0]);
    expect(red.shape).toEqual([4, 5]);
    expect(red.strides).toEqual([15, 3]);
  });

  it("ellipsis and newaxis", () => {
    const a = np.arange(24).reshape([2, 3, 4]);
    expect(a.get(np.ellipsis, 0).toArray()).toEqual([[0, 4, 8], [12, 16, 20]]);
    expect(a.get(np.newaxis, np.ellipsis, np.newaxis).shape).toEqual([1, 2, 3, 4, 1]);
    expect(a.get(null, np.newaxis, 1).shape).toEqual([2, 1, 4]);
    expect(() => a.get(np.ellipsis, np.ellipsis)).toThrow(IndexError);
    expect(() => a.get(0, 0, 0, 0)).toThrow(IndexError);
    expect(() => a.get(2)).toThrow(IndexError);
    expect(() => a.get(1.5)).toThrow(ValueError);
  });

  it("integer and boolean array indexing copies", () => {
    const a = np.arange(12).reshape([3, 4]);
    const r = a.get(np.array([2, 0]));
    expect(r.toArray()).toEqual([[8, 9, 10, 11], [0, 1, 2, 3]]);
    expect(np.mayShareMemory(a, r)).toBe(false);
    expect(a.get(np.array([0, 2]), np.array([1, 3])).toArray()).toEqual([1, 11]);
    expect(a.get(null, np.array([[0], [3]])).shape).toEqual([3, 2, 1]);
    const mask = np.array([[true, false, false, true], [false, false, false, false], [true, true, false, false]]);
    expect(a.get(mask).toArray()).toEqual([0, 3, 8, 9]);
    expect(a.get(np.array([true, false, true])).shape).toEqual([2, 4]);
    expect(a.get(true).shape).toEqual([1, 3, 4]);
    expect(a.get(false).shape).toEqual([0, 3, 4]);
    expect(() => a.get(np.array([3]))).toThrow(IndexError);
    expect(() => a.get(np.array([true, false]))).toThrow(IndexError);
    expect(() => a.get(np.array([0.5]))).toThrow(IndexError);
  });

  it("set: broadcast, cast, boolean mask, fancy", () => {
    const a = np.zeros([3, 4], { dtype: np.int32 });
    a.set([null, 1], 7.9);
    expect(a.get(null, 1).toArray()).toEqual([7, 7, 7]);
    a.set([2], [1, 2, 3, 4]);
    expect(a.get(2).toArray()).toEqual([1, 2, 3, 4]);
    a.set(np.array([[false, false, false, false], [true, true, true, true], [false, false, false, false]]), -1);
    expect(a.get(1).toArray()).toEqual([-1, -1, -1, -1]);
    a.set([np.array([0, 0]), np.array([0, 0])], np.array([5, 6]));
    expect(a.item(0, 0)).toBe(6);
    expect(() => a.set([0], [1, 2])).toThrow(BroadcastError);
  });

  it("D-016: broadcastTo views are read-only; views inherit, copies are writeable", () => {
    const a = np.arange(3);
    const b = np.broadcastTo(a, [2, 3]);
    expect(a.flags.writeable).toBe(true);
    expect(b.flags.writeable).toBe(false);
    expect(() => b.set([0, 0], 99)).toThrow(ValueError);
    expect(() => b.set([0, 0], 99)).toThrow("assignment destination is read-only");
    expect(() => b.set(np.array([0]), 1)).toThrow(ValueError);
    expect(a.toArray()).toEqual([0, 1, 2]);
    expect(b.get(0).flags.writeable).toBe(false);
    expect(b.transpose().flags.writeable).toBe(false);
    expect(np.expandDims(b, 0).flags.writeable).toBe(false);
    expect(b.copy().flags.writeable).toBe(true);
    expect(b.get(np.array([0])).flags.writeable).toBe(true);
    const c = b.copy();
    c.set([0, 0], 5);
    expect(c.item(0, 0)).toBe(5);
    expect(a.slice([[1, null]]).flags.writeable).toBe(true);
  });

  it("nonzero, take, where", () => {
    const [r, c] = np.nonzero([[0, 1], [2, 0]]);
    expect(r!.toArray()).toEqual([0, 1]);
    expect(c!.toArray()).toEqual([1, 0]);
    expect(r!.dtype.name).toBe("int64");
    expect(() => np.nonzero(np.array(3))).toThrow(ValueError);
    const a = np.arange(12).reshape([3, 4]);
    expect(np.take(a, [0, -1]).toArray()).toEqual([0, 11]);
    expect(np.take(a, [0, 3], 1).toArray()).toEqual([[0, 3], [4, 7], [8, 11]]);
    expect(() => np.take(a, [0], 2)).toThrow(IndexError);
    const w = np.where([true, false, true], [1, 2, 3], 0.5);
    expect(w.toArray()).toEqual([1, 0.5, 3]);
    expect(w.dtype.name).toBe("float64");
    expect(np.where([0, 3]).length).toBe(1);
  });
});
