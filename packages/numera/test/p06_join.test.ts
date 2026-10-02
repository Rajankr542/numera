import { describe, expect, it } from "vitest";
import np, { DTypeError, IndexError, ValueError } from "../src/index.js";

describe("P6 join (D-090)", () => {
  const a = np.array([[1, 2], [3, 4]]);
  it("concatenate axis / null / negative / options", () => {
    expect(np.concatenate([a, np.array([[5, 6]])]).toArray()).toEqual([[1, 2], [3, 4], [5, 6]]);
    expect(np.concatenate([a, np.array([[5], [6]])], 1).toArray()).toEqual([[1, 2, 5], [3, 4, 6]]);
    expect(np.concatenate([a, [[5], [6]]], -1).shape).toEqual([2, 3]);
    expect(np.concatenate([a, [7, 8, 9]], null).toArray()).toEqual([1, 2, 3, 4, 7, 8, 9]);
    expect(np.concat([[1], [2.5]]).dtype.name).toBe("float64");
    expect(np.concatenate([np.zeros([0, 2]), a]).shape).toEqual([2, 2]);
  });
  it("concatenate dtype / casting / out", () => {
    const f = np.array([1.5, 2.5], { dtype: "float32" });
    expect(() => np.concatenate([f], { dtype: "int8" })).toThrow(DTypeError);
    const r = np.concatenate([f, f], { dtype: "int8", casting: "unsafe" });
    expect(r.dtype.name).toBe("int8");
    expect(r.toArray()).toEqual([1, 2, 1, 2]);
    const out = np.zeros([4], { dtype: "int64" });
    expect(np.concatenate([[1, 2], [3, 4]], { out })).toBe(out);
    expect(out.toArray()).toEqual([1, 2, 3, 4]);
    expect(() => np.concatenate([[1], [2]], { out: np.zeros([3]) })).toThrow(ValueError);
    expect(() => np.concatenate([[1.5]], { out: np.zeros([1], { dtype: "int8" }) })).toThrow(DTypeError);
    expect(() => np.concatenate([[1]], { out: np.zeros([1]), dtype: "float64" })).toThrow(DTypeError);
  });
  it("concatenate errors and layout", () => {
    expect(() => np.concatenate([])).toThrow(ValueError);
    expect(() => np.concatenate([np.array(1), np.array(2)])).toThrow(ValueError);
    expect(() => np.concatenate([a, [1, 2, 3]])).toThrow(ValueError);
    expect(() => np.concatenate([a], 2)).toThrow(IndexError);
    expect(() => np.concatenate([a], 1.5)).toThrow(ValueError);
    const F = np.zeros([2, 3], { order: "F" });
    expect(np.concatenate([F, F]).strides).toEqual([8, 32]);
    expect(np.concatenate([F, F], 1).strides).toEqual([8, 16]);
    expect(np.concatenate([F, np.zeros([2, 3])]).strides).toEqual([24, 8]);
  });
  it("concatenate on every dtype and complex", () => {
    for (const dt of ["bool", "int8", "uint16", "int64", "float16", "float32", "complex64"] as const) {
      const x = np.ones([2], { dtype: dt });
      const c = np.concatenate([x, x]);
      expect(c.dtype.name).toBe(dt);
      expect(c.shape).toEqual([4]);
    }
    expect(np.concatenate([np.zeros([1], { dtype: "float32" }), [{ re: 0, im: 1 }]]).dtype.name).toBe("complex128");
    expect(np.concatenate([[NaN], [Infinity]]).toArray()).toEqual([NaN, Infinity]);
  });
  it("stack / vstack / hstack / dstack / columnStack", () => {
    expect(np.stack([[1, 2], [3, 4]]).toArray()).toEqual([[1, 2], [3, 4]]);
    expect(np.stack([[1, 2], [3, 4]], -1).toArray()).toEqual([[1, 3], [2, 4]]);
    expect(np.stack([[1, 2], [3, 4]], { axis: 1, dtype: "float32" }).dtype.name).toBe("float32");
    expect(() => np.stack([[1, 2], [3]])).toThrow(ValueError);
    expect(() => np.stack([])).toThrow(ValueError);
    expect(() => np.stack([[1]], 3)).toThrow(IndexError);
    expect(np.vstack([[1, 2], [3, 4]]).toArray()).toEqual([[1, 2], [3, 4]]);
    expect(np.vstack([np.array(1), np.array(2)]).shape).toEqual([2, 1]);
    expect(np.hstack([[1, 2], [3]]).toArray()).toEqual([1, 2, 3]);
    expect(np.hstack([np.array(1), np.array(2)]).toArray()).toEqual([1, 2]);
    expect(np.hstack([a, a]).shape).toEqual([2, 4]);
    expect(np.dstack([[1, 2], [3, 4]]).shape).toEqual([1, 2, 2]);
    expect(np.columnStack([[1, 2], [3, 4]]).toArray()).toEqual([[1, 3], [2, 4]]);
    expect(np.columnStack([a, [5, 6]]).toArray()).toEqual([[1, 2, 5], [3, 4, 6]]);
    expect(np.stack(a).toArray()).toEqual([[1, 2], [3, 4]]);
  });
  it("block", () => {
    const r = np.block([[np.ones([2, 2]), np.zeros([2, 1])], [np.zeros([1, 3])]]);
    expect(r.toArray()).toEqual([[1, 1, 0], [1, 1, 0], [0, 0, 0]]);
    expect(np.block([1, 2]).toArray()).toEqual([1, 2]);
    expect(np.block(5).toArray()).toEqual(5);
    expect(np.block([[1], [2]]).toArray()).toEqual([[1], [2]]);
    expect(() => np.block([[1, 2], 3])).toThrow(/List depths are mismatched/);
    expect(() => np.block([])).toThrow(/cannot be empty/);
    expect(() => np.block([[1], []])).toThrow(ValueError);
    const x = np.array([1, 2]);
    const b = np.block(x);
    expect(np.mayShareMemory(b, x)).toBe(false);
  });
  it("unstack / atleast*d", () => {
    const u = np.unstack(np.arange(6).reshape(2, 3), 1);
    expect(u.map((v) => v.toArray())).toEqual([[0, 3], [1, 4], [2, 5]]);
    expect(np.unstack(np.array([1, 2]), { axis: 0 })[0]!.shape).toEqual([]);
    expect(() => np.unstack(np.array(1))).toThrow(ValueError);
    expect(np.atleast1d(5).shape).toEqual([1]);
    expect((np.atleast1d(1, [2, 3]) as unknown[]).length).toBe(2);
    expect(np.atleast2d([1, 2]).shape).toEqual([1, 2]);
    expect(np.atleast3d([1, 2, 3]).shape).toEqual([1, 3, 1]);
    expect(np.atleast3d(np.zeros([2, 3])).shape).toEqual([2, 3, 1]);
    expect(np.atleast3d(np.array(1)).shape).toEqual([1, 1, 1]);
    const v = np.zeros([3]);
    expect(np.mayShareMemory(np.atleast2d(v), v)).toBe(true);
  });
});

describe("P6 split (D-090)", () => {
  it("split / arraySplit", () => {
    const s = np.split(np.arange(6), 3);
    expect(s.map((x) => x.toArray())).toEqual([[0, 1], [2, 3], [4, 5]]);
    expect(() => np.split(np.arange(7), 3)).toThrow(/equal division/);
    expect(np.split(np.arange(6), [2, 10]).map((x) => x.shape)).toEqual([[2], [4], [0]]);
    expect(np.arraySplit(np.arange(7), 3).map((x) => x.toArray())).toEqual([[0, 1, 2], [3, 4], [5, 6]]);
    expect(np.arraySplit(np.arange(3), 5).map((x) => x.size)).toEqual([1, 1, 1, 0, 0]);
    expect(np.arraySplit(np.arange(5), [3, 1]).map((x) => x.toArray())).toEqual([[0, 1, 2], [], [1, 2, 3, 4]]);
    expect(() => np.arraySplit(np.arange(3), 0)).toThrow(ValueError);
    expect(np.split(np.arange(6).reshape(2, 3), [1], 1)[1]!.shape).toEqual([2, 2]);
    expect(() => np.split(np.arange(3), 1, 1)).toThrow(IndexError);
    const a = np.arange(4);
    expect(np.mayShareMemory(np.split(a, 2)[1]!, a)).toBe(true);
  });
  it("hsplit / vsplit / dsplit", () => {
    expect(np.hsplit(np.arange(6).reshape(2, 3), 3)[0]!.shape).toEqual([2, 1]);
    expect(np.hsplit(np.arange(4), 2)[0]!.toArray()).toEqual([0, 1]);
    expect(() => np.hsplit(np.array(1), 1)).toThrow(ValueError);
    expect(np.vsplit(np.arange(6).reshape(2, 3), 2)[1]!.toArray()).toEqual([[3, 4, 5]]);
    expect(() => np.vsplit(np.arange(3), 1)).toThrow(/2 or more/);
    expect(np.dsplit(np.zeros([1, 1, 4]), [1])[1]!.shape).toEqual([1, 1, 3]);
    expect(() => np.dsplit(np.zeros([2, 2]), 1)).toThrow(/3 or more/);
  });
});
