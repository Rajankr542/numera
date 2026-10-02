import { describe, expect, it } from "vitest";
import np, { BroadcastError, DTypeError, FloatingPointError, ValueError } from "../src/index.js";

const DTYPES = [
  "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
  "float16", "float32", "float64", "complex64", "complex128",
] as const;

describe("P5 comparisons (D-080)", () => {
  it("compare every dtype and return bool", () => {
    for (const dt of DTYPES) {
      const a = np.array([0, 1, 1], { dtype: dt });
      const b = np.array([1, 1, 0], { dtype: dt });
      const r = np.less(a, b);
      expect(r.dtype).toBe(np.bool);
      expect(r.toArray()).toEqual([true, false, false]);
      expect(np.equal(a, b).toArray()).toEqual([false, true, false]);
      expect(np.notEqual(a, b).toArray()).toEqual([true, false, true]);
      expect(np.lessEqual(a, b).toArray()).toEqual([true, true, false]);
      expect(np.greater(a, b).toArray()).toEqual([false, false, true]);
      expect(np.greaterEqual(a, b).toArray()).toEqual([false, true, true]);
    }
  });

  it("broadcasts, promotes and handles scalars, 0-d and empty", () => {
    expect(np.less([[1], [3]], [2, 4]).toArray()).toEqual([[true, true], [false, true]]);
    expect(np.equal(np.array([1, 2], { dtype: "int8" }), 1.0).toArray()).toEqual([true, false]);
    expect(np.equal(np.array([1], { dtype: "int8" }), 1.5).toArray()).toEqual([false]);
    expect(np.greater(3, 2).toArray()).toBe(true);
    expect(np.less(np.array(1), np.array(2)).shape).toEqual([]);
    expect(np.less(np.zeros([0, 3]), 1).shape).toEqual([0, 3]);
    expect(() => np.less([1, 2], [1, 2, 3])).toThrow(BroadcastError);
  });

  it("compares out-of-range JS integers exactly", () => {
    const i8 = np.array([1, -1], { dtype: "int8" });
    expect(np.less(i8, 1000).toArray()).toEqual([true, true]);
    expect(np.greater(1000, i8).toArray()).toEqual([true, true]);
    expect(np.equal(i8, 1000).toArray()).toEqual([false, false]);
    const u8 = np.array([0, 5], { dtype: "uint8" });
    expect(np.less(u8, -1).toArray()).toEqual([false, false]);
    expect(np.notEqual(u8, -1).toArray()).toEqual([true, true]);
    expect(np.equal(np.array([1n], { dtype: "uint64" }), -1).toArray()).toEqual([false]);
    expect(np.less(np.array([1], { dtype: "int64" }), 2n ** 64n * 4n).toArray()).toEqual([true]);
  });

  it("NaN compares false; complex compares lexicographically", () => {
    const n = np.array([NaN, 1]);
    const m = np.array([1, NaN]);
    expect(np.less(n, m).toArray()).toEqual([false, false]);
    expect(np.equal(n, n).toArray()).toEqual([false, true]);
    expect(np.notEqual(n, n).toArray()).toEqual([true, false]);
    const a = np.array([np.complex(1, 2), np.complex(1, 1), np.complex(0, 5)]);
    const b = np.array([np.complex(1, 1), np.complex(1, 1), np.complex(1, -5)]);
    expect(np.less(a, b).toArray()).toEqual([false, false, true]);
    expect(np.lessEqual(a, b).toArray()).toEqual([false, true, true]);
    expect(np.greater(a, b).toArray()).toEqual([true, false, false]);
    expect(np.equal(a, b).toArray()).toEqual([false, true, false]);
    np.errstate({ invalid: "raise" }, () => {
      expect(() => np.less(np.array([np.complex(NaN, 0)]), np.array([np.complex(1, 0)]))).toThrow(FloatingPointError);
      expect(np.less(n, m).toArray()).toEqual([false, false]);
    });
  });

  it("supports out, where, dtype=bool only, casting", () => {
    const out = np.zeros(2);
    expect(np.less([0, 2], [1, 1], { out })).toBe(out);
    expect(out.toArray()).toEqual([1, 0]);
    expect(np.equal([1], [1.5], { dtype: "bool" }).toArray()).toEqual([false]);
    expect(() => np.equal([1], [1], { dtype: "float64" })).toThrow(DTypeError);
    expect(() => np.equal(np.array([1]), np.array([1], { dtype: "float64" }), { casting: "no" })).toThrow(DTypeError);
    expect(() => np.less([1], [2], { out: np.zeros(1, { dtype: "int8" }), casting: "no" })).toThrow(DTypeError);
    const o2 = np.array([true, true]);
    np.less([1, 3], [2, 2], { out: o2, where: [false, true] });
    expect(o2.toArray()).toEqual([true, false]);
  });

  it("reduce/accumulate/outer follow NumPy for bool-output ufuncs", () => {
    expect(np.equal.reduce([true, false, false]).toArray()).toBe(true);
    expect(() => np.equal.reduce([1, 1, 1])).toThrow(DTypeError);
    expect(() => np.less.reduce([1.5, 2.5])).toThrow(DTypeError);
    expect(np.less.reduce([1, 2], { dtype: "bool" }).toArray()).toBe(false);
    expect(np.equal.accumulate([true, false, true]).toArray()).toEqual([true, false, false]);
    expect(() => np.equal.reduce([[true]], { axis: [0, 1] })).toThrow(ValueError);
    expect(np.equal.outer([1, 2], [1]).toArray()).toEqual([[true], [false]]);
    expect(np.notEqual.reduce([true, false, true]).toArray()).toBe(false);
  });
});

describe("P5 logical ops and all/any (D-080)", () => {
  it("logical ufuncs on every dtype", () => {
    for (const dt of DTYPES) {
      const a = np.array([0, 0, 1, 1], { dtype: dt });
      const b = np.array([0, 1, 0, 1], { dtype: dt });
      expect(np.logicalAnd(a, b).toArray()).toEqual([false, false, false, true]);
      expect(np.logicalOr(a, b).toArray()).toEqual([false, true, true, true]);
      expect(np.logicalXor(a, b).toArray()).toEqual([false, true, true, false]);
      expect(np.logicalNot(a).toArray()).toEqual([true, true, false, false]);
      expect(np.logicalNot(a).dtype).toBe(np.bool);
    }
    expect(np.logicalNot([0.5, NaN, 0]).toArray()).toEqual([false, false, true]);
    expect(np.logicalNot(np.array([np.complex(0, 1)])).toArray()).toEqual([false]);
    expect(np.logicalAnd(np.array([1], { dtype: "int8" }), 1000).toArray()).toEqual([true]);
    expect(np.logicalNot(np.array(0)).shape).toEqual([]);
    expect(np.logicalNot([1], { dtype: "bool" }).toArray()).toEqual([false]);
    expect(() => np.logicalNot([1], { dtype: "float64" })).toThrow(DTypeError);
    expect(() => np.logicalAnd([1], [1], { dtype: "int64" })).toThrow(DTypeError);
  });

  it("logical reduce/accumulate use NumPy identities", () => {
    expect(np.logicalAnd.reduce([1, 2, 0]).toArray()).toBe(false);
    expect(np.logicalAnd.reduce(np.array([], { dtype: "int64" })).toArray()).toBe(true);
    expect(np.logicalOr.reduce(np.array([], { dtype: "float64" })).toArray()).toBe(false);
    expect(np.logicalXor.reduce([1, 1, 1]).toArray()).toBe(true);
    expect(np.logicalXor.reduce([[1, 1], [1, 0]], { axis: null }).toArray()).toBe(true);
    expect(np.logicalAnd.accumulate([1, 2, 0, 3]).toArray()).toEqual([true, true, false, false]);
    const out = np.zeros([], { dtype: "int64" });
    np.logicalAnd.reduce([1, 2], { out });
    expect(out.toArray()).toBe(1);
  });

  it("all/any with axis, keepdims, where, out", () => {
    const m = [[1, 0], [1, 1]];
    expect(np.all(m).toArray()).toBe(false);
    expect(np.any(m).toArray()).toBe(true);
    expect(np.all(m, { axis: 0 }).toArray()).toEqual([true, false]);
    expect(np.any([[0, 0], [0, 1]], { axis: 1 }).toArray()).toEqual([false, true]);
    expect(np.all(m, { axis: [0, 1], keepdims: true }).toArray()).toEqual([[false]]);
    expect(np.all([]).toArray()).toBe(true);
    expect(np.any([]).toArray()).toBe(false);
    expect(np.any(np.zeros([2, 0]), { axis: 1 }).toArray()).toEqual([false, false]);
    expect(np.all([1, 0], { where: [true, false] }).toArray()).toBe(true);
    expect(np.all(5).toArray()).toBe(true);
    expect(np.any([NaN]).toArray()).toBe(true);
    expect(np.all([np.complex(0, 0)]).toArray()).toBe(false);
    const out = np.zeros([], { dtype: "int64" });
    expect(np.all([1, 2], { out })).toBe(out);
    expect(out.toArray()).toBe(1);
    expect(() => np.all(m, { axis: 2 })).toThrow();
    const a = np.array(m);
    expect(a.all().toArray()).toBe(false);
    expect(a.any({ axis: 0 }).toArray()).toEqual([true, true]);
  });
});
