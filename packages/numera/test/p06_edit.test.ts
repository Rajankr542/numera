import { describe, expect, it } from "vitest";
import np, { IndexError, ValueError } from "../src/index.js";

describe("P6 insert / delete / append (D-093)", () => {
  const a = np.array([[1, 1], [2, 2], [3, 3]]);
  it("insert", () => {
    expect(np.insert(a, 1, 5).toArray()).toEqual([1, 5, 1, 2, 2, 3, 3]);
    expect(np.insert(a, 1, 5, 1).toArray()).toEqual([[1, 5, 1], [2, 5, 2], [3, 5, 3]]);
    expect(np.insert(a, [1], [[1], [2], [3]], 1).toArray()).toEqual([[1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    expect(np.insert(a, 1, [1, 2, 3], 1).toArray()).toEqual([[1, 1, 1], [2, 2, 2], [3, 3, 3]]);
    expect(np.insert(np.arange(5), [1, 1, 3], [7, 8, 9]).toArray()).toEqual([0, 7, 8, 1, 2, 9, 3, 4]);
    expect(np.insert(np.arange(5), { start: 1, stop: 4 }, 0).toArray()).toEqual([0, 0, 1, 0, 2, 0, 3, 4]);
    expect(np.insert([1, 2], [-1, 2], [8.7, 9]).toArray()).toEqual([1, 8, 2, 9]);
    expect(np.insert([1, 2], 0, [5, 6]).toArray()).toEqual([5, 6, 1, 2]);
    expect(np.insert([1, 2], [true, false], 9).toArray()).toEqual([9, 1, 2]);
    expect(np.insert([1.5], 1, NaN).toArray()).toEqual([1.5, NaN]);
    expect(np.insert([], 0, 1).toArray()).toEqual([1]);
    expect(np.insert(np.zeros([2, 3], { order: "F" }), 1, 1, 0).flags.fContiguous).toBe(true);
    expect(() => np.insert([1, 2], 3, 0)).toThrow(IndexError);
    expect(() => np.insert([1, 2], [[0]], 0)).toThrow(/one dimensional/);
    expect(() => np.insert(a, 0, 0, 2)).toThrow(IndexError);
  });
  it("delete", () => {
    expect(np.delete(np.arange(6), [0, -1]).toArray()).toEqual([1, 2, 3, 4]);
    expect(np.delete(a, 1, 0).toArray()).toEqual([[1, 1], [3, 3]]);
    expect(np.delete(a, 0, 1).toArray()).toEqual([[1], [2], [3]]);
    expect(np.delete(a, 1).toArray()).toEqual([1, 2, 2, 3, 3]);
    expect(np.delete(np.arange(10), { start: 1, stop: 8, step: 3 }).toArray()).toEqual([0, 2, 3, 5, 6, 8, 9]);
    expect(np.delete(np.arange(5), { start: -1, step: -2 }).toArray()).toEqual([1, 3]);
    expect(np.delete(np.arange(3), [true, false, true]).toArray()).toEqual([1]);
    expect(np.delete(np.arange(3), []).toArray()).toEqual([0, 1, 2]);
    expect(np.delete(np.arange(3), [1, 1]).toArray()).toEqual([0, 2]);
    expect(() => np.delete(np.arange(3), 3)).toThrow(/out of bounds for axis 0 with size 3/);
    expect(() => np.delete(np.arange(3), [true])).toThrow(/match the axis length of 3/);
    expect(() => np.delete(np.arange(3), [0.5])).toThrow(IndexError);
  });
  it("append", () => {
    expect(np.append([1, 2], [[3, 4]]).toArray()).toEqual([1, 2, 3, 4]);
    expect(np.append([[1, 2]], [[3, 4]], 0).toArray()).toEqual([[1, 2], [3, 4]]);
    expect(np.append([1], [2.5]).dtype.name).toBe("float64");
    expect(() => np.append([[1, 2]], [3], 0)).toThrow(ValueError);
  });
});

describe("P6 resize / trimZeros (D-093)", () => {
  it("np.resize", () => {
    expect(np.resize([1, 2, 3], [2, 4]).toArray()).toEqual([[1, 2, 3, 1], [2, 3, 1, 2]]);
    expect(np.resize([1, 2, 3], 2).toArray()).toEqual([1, 2]);
    expect(np.resize(np.zeros([0], { dtype: "int8" }), 3).toArray()).toEqual([0, 0, 0]);
    expect(np.resize([1], []).toArray()).toEqual(1);
    expect(() => np.resize([1], [-1])).toThrow(/non-negative/);
  });
  it("NDArray.resize in place", () => {
    const x = np.array([[1, 2], [3, 4]]);
    expect(x.resize(3, 3)).toBeUndefined();
    expect(x.toArray()).toEqual([[1, 2, 3], [4, 0, 0], [0, 0, 0]]);
    x.resize([2]);
    expect(x.toArray()).toEqual([1, 2]);
    const f = np.asfortranarray(np.array([[1, 2], [3, 4]]));
    f.resize([2, 3]);
    expect(f.flags.fContiguous).toBe(true);
    expect(f.toArray()).toEqual([[1, 2, 0], [3, 4, 0]]);
    const y = np.arange(4);
    const v = y.get([1, 3]);
    expect(() => y.resize(8)).toThrow(/refcheck/);
    y.resize(2, { refcheck: false });
    expect(y.toArray()).toEqual([0, 1]);
    expect(v.toArray()).toEqual([1, 2]);
    expect(() => v.resize(1)).toThrow(/does not own/);
    expect(() => np.zeros([2]).resize(-1)).toThrow(ValueError);
  });
  it("trimZeros", () => {
    const z = np.array([0, 0, 1, 0, 2, 0]);
    expect(np.trimZeros(z).toArray()).toEqual([1, 0, 2]);
    expect(np.trimZeros(z, "b").toArray()).toEqual([0, 0, 1, 0, 2]);
    expect(np.trimZeros(z, "F").toArray()).toEqual([1, 0, 2, 0]);
    expect(np.mayShareMemory(np.trimZeros(z), z)).toBe(true);
    expect(np.trimZeros([0, 0]).shape).toEqual([0]);
    expect(np.trimZeros([0, NaN, 0]).toArray()).toEqual([NaN]);
    const m = np.array([[0, 0, 0, 0], [0, 1, 0, 2], [0, 0, 0, 0]]);
    expect(np.trimZeros(m).toArray()).toEqual([[1, 0, 2]]);
    expect(np.trimZeros(m, "fb", 0).shape).toEqual([1, 4]);
    expect(np.trimZeros(m, "fb", []).shape).toEqual([3, 4]);
    expect(() => np.trimZeros(z, "x")).toThrow(/unexpected character/);
    expect(() => np.trimZeros(z, "fb", 1)).toThrow(IndexError);
  });
});
