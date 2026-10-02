import { describe, expect, it } from "vitest";
import np, { BroadcastError, DTypeError, FloatingPointError, IndexError, ValueError } from "../src/index.js";

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

describe("P5 classification and isscalar (D-081)", () => {
  it("isnan/isinf/isfinite on every dtype", () => {
    for (const dt of DTYPES) {
      const a = np.array([0, 1], { dtype: dt });
      expect(np.isnan(a).toArray()).toEqual([false, false]);
      expect(np.isinf(a).toArray()).toEqual([false, false]);
      expect(np.isfinite(a).toArray()).toEqual([true, true]);
      expect(np.isfinite(a).dtype).toBe(np.bool);
    }
    for (const dt of ["float16", "float32", "float64"] as const) {
      const a = np.array([NaN, Infinity, -Infinity, 1], { dtype: dt });
      expect(np.isnan(a).toArray()).toEqual([true, false, false, false]);
      expect(np.isinf(a).toArray()).toEqual([false, true, true, false]);
      expect(np.isfinite(a).toArray()).toEqual([false, false, false, true]);
      expect(np.isposinf(a).toArray()).toEqual([false, true, false, false]);
      expect(np.isneginf(a).toArray()).toEqual([false, false, true, false]);
    }
    const c = np.array([np.complex(1, NaN), np.complex(Infinity, 0), np.complex(1, 2)]);
    expect(np.isnan(c).toArray()).toEqual([true, false, false]);
    expect(np.isinf(c).toArray()).toEqual([false, true, false]);
    expect(np.isfinite(c).toArray()).toEqual([false, false, true]);
    expect(np.isnan(NaN).toArray()).toBe(true);
    expect(np.isnan(np.zeros([0, 2])).shape).toEqual([0, 2]);
  });

  it("options and errors", () => {
    const out = np.zeros(2, { dtype: "int8" });
    expect(np.isnan([NaN, 1], { out })).toBe(out);
    expect(out.toArray()).toEqual([1, 0]);
    expect(np.isnan([NaN], { dtype: "bool" }).toArray()).toEqual([true]);
    expect(() => np.isnan([1], { dtype: "float64" })).toThrow(DTypeError);
    expect(np.isinf([Infinity, 1], { where: [false, true] }).toArray()).toEqual([false, false]);
    const o2 = np.zeros(1);
    expect(np.isposinf([Infinity], { out: o2 })).toBe(o2);
    expect(o2.toArray()).toEqual([1]);
    expect(() => np.isposinf(np.array([np.complex(1, 1)]))).toThrow(DTypeError);
    expect(() => np.isneginf(np.array([np.complex(1, 1)]))).toThrow(DTypeError);
    expect(np.isneginf([1, 2]).toArray()).toEqual([false, false]);
    expect(() => np.isnat([1])).toThrow(DTypeError);
    expect(() => np.isnat(np.array([], { dtype: "float64" }))).toThrow(DTypeError);
  });

  it("isscalar", () => {
    for (const v of [1, 1.5, true, 1n, "a", np.complex(1, 2), { re: 1, im: 0 }]) expect(np.isscalar(v)).toBe(true);
    for (const v of [np.array(1), [1], null, undefined, {}, np.zeros(2)]) expect(np.isscalar(v)).toBe(false);
  });
});

describe("P5 bitwise (D-082)", () => {
  const INTS = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"] as const;
  it("and/or/xor/invert on every integer dtype and bool", () => {
    for (const dt of INTS) {
      const a = np.array([12, 10], { dtype: dt });
      const b = np.array([10, 6], { dtype: dt });
      expect(np.bitwiseAnd(a, b).dtype.name).toBe(dt);
      expect(np.bitwiseAnd(a, b).toArray()).toEqual([8, 2]);
      expect(np.bitwiseOr(a, 1).toArray()).toEqual([13, 11]);
      expect(np.bitwiseXor(a, b).toArray()).toEqual([6, 12]);
    }
    expect(np.invert(np.array([5], { dtype: "int8" })).toArray()).toEqual([-6]);
    expect(np.invert(np.array([1], { dtype: "uint8" })).toArray()).toEqual([254]);
    expect(np.invert([true, false]).toArray()).toEqual([false, true]);
    expect(np.bitwiseNot).toBe(np.invert);
    expect(np.bitwiseInvert).toBe(np.invert);
    expect(np.bitwiseAnd(true, false).toArray()).toBe(false);
    expect(np.bitwiseAnd(np.array([true]), 3).toArray()).toEqual([1]);
    expect(np.bitwiseAnd(np.array(6), 3).shape).toEqual([]);
    expect(np.bitwiseAnd(np.zeros([0], { dtype: "int32" }), 1).shape).toEqual([0]);
  });

  it("rejects floats, complex and uint64/int64 mixes", () => {
    expect(() => np.bitwiseAnd([1.5], 1)).toThrow(DTypeError);
    expect(() => np.bitwiseOr(np.array([1], { dtype: "float32" }), 1)).toThrow(DTypeError);
    expect(() => np.invert([1.5])).toThrow(DTypeError);
    expect(() => np.bitwiseXor(np.array([np.complex(1, 0)]), 1)).toThrow(DTypeError);
    expect(() => np.leftShift(np.array([1], { dtype: "uint64" }), np.array([1]))).toThrow(DTypeError);
    expect(() => np.bitwiseCount([1.5])).toThrow(DTypeError);
    expect(() => np.bitwiseAnd(np.array([1], { dtype: "uint8" }), -1)).toThrow(ValueError);
    expect(() => np.leftShift([1], [2], { dtype: "bool" })).toThrow(DTypeError);
  });

  it("reduce identities", () => {
    expect(np.bitwiseAnd.reduce(np.array([], { dtype: "uint8" })).toArray()).toBe(255);
    expect([...(np.bitwiseAnd.reduce(np.array([], { dtype: "uint64" })).toTypedArray() as BigUint64Array)]).toEqual([2n ** 64n - 1n]);
    expect(np.bitwiseAnd.reduce(np.array([], { dtype: "int8" })).toArray()).toBe(-1);
    expect(np.bitwiseAnd.reduce(np.array([], { dtype: "bool" })).toArray()).toBe(true);
    expect(np.bitwiseAnd.reduce([[1, 3], [3, 7]], { axis: [0, 1] }).toArray()).toBe(1);
    expect(np.bitwiseXor.reduce([[1, 2], [3, 4]], { axis: null }).toArray()).toBe(4);
    expect(np.bitwiseOr.accumulate([1, 2, 4]).toArray()).toEqual([1, 3, 7]);
    expect(np.leftShift.reduce([1, 2, 3]).toArray()).toBe(32);
  });

  it("shifts", () => {
    const i8 = (v: number[]) => np.array(v, { dtype: "int8" });
    expect(np.leftShift(i8([1, 1, 1, 1]), i8([3, 7, 9, -1])).toArray()).toEqual([8, -128, 0, 0]);
    expect(np.rightShift(i8([-128, -5, 5, -8]), i8([10, -1, -1, 1])).toArray()).toEqual([-1, -1, 0, -4]);
    expect(np.leftShift(np.array([1n], { dtype: "int64" }), 64).toArray()).toEqual([0]);
    expect(np.leftShift(true, true).dtype).toBe(np.int8);
    expect(np.leftShift(np.array([1], { dtype: "uint8" }), np.array([1], { dtype: "int8" })).dtype).toBe(np.int16);
    expect(np.leftShift(np.array([1], { dtype: "uint64" }), 1).dtype).toBe(np.uint64);
    expect(np.leftShift([1], [2], { dtype: "int8" }).toArray()).toEqual([4]);
    expect(np.bitwiseLeftShift).toBe(np.leftShift);
    expect(np.bitwiseRightShift).toBe(np.rightShift);
  });

  it("bitwiseCount", () => {
    const r = np.bitwiseCount(np.array([-1, -128, 7, 0], { dtype: "int8" }));
    expect(r.dtype).toBe(np.uint8);
    expect(r.toArray()).toEqual([1, 1, 3, 0]);
    expect(np.bitwiseCount([true, false]).toArray()).toEqual([1, 0]);
    expect(np.bitwiseCount(np.array([2n ** 64n - 1n], { dtype: "uint64" })).toArray()).toEqual([64]);
    expect(np.bitwiseCount([3], { dtype: "uint8" }).toArray()).toEqual([2]);
    expect(() => np.bitwiseCount([3], { dtype: "int64" })).toThrow(DTypeError);
  });
});

describe("P5 isclose family (D-083)", () => {
  it("isclose matches NumPy", () => {
    expect(np.isclose([1e10, 1e-7], [1.00001e10, 1e-8]).toArray()).toEqual([true, false]);
    expect(np.isclose([1e-8, 1e-7], [0, 0]).toArray()).toEqual([true, false]);
    expect(np.isclose([1, NaN], [1, NaN]).toArray()).toEqual([true, false]);
    expect(np.isclose([1, NaN], [1, NaN], { equalNan: true }).toArray()).toEqual([true, true]);
    expect(np.isclose([Infinity, -Infinity, 1], [Infinity, Infinity, Infinity]).toArray()).toEqual([true, false, false]);
    expect(np.isclose(1, 2, { rtol: 1 }).toArray()).toBe(true);
    expect(np.isclose(2, [1, 2, 3]).shape).toEqual([3]);
    expect(np.isclose(1, 1).shape).toEqual([]);
    expect(np.isclose(np.array([1.1], { dtype: "float32" }), 1.1, { rtol: 0, atol: 0 }).toArray()).toEqual([true]);
    expect(np.isclose(np.array([1], { dtype: "int8" }), 1).toArray()).toEqual([true]);
    expect(np.isclose(np.array([np.complex(NaN, 0)]), np.array([np.complex(0, NaN)]), { equalNan: true }).toArray()).toEqual([true]);
    expect(np.isclose(np.array([1], { dtype: "float16" }), np.array([1.001], { dtype: "float16" })).toArray()).toEqual([false]);
    expect(np.isclose(np.zeros([0]), 1).shape).toEqual([0]);
  });

  it("invalid tolerances follow seterr invalid", () => {
    expect(() => np.errstate({ invalid: "raise" }, () => np.isclose(1, 1, { atol: NaN }))).toThrow(FloatingPointError);
    expect(np.errstate({ invalid: "ignore" }, () => np.isclose(1, 1, { atol: NaN }).toArray())).toBe(true);
  });

  it("allclose / arrayEqual / arrayEquiv", () => {
    expect(np.allclose([1e10, 1e-8], [1.00001e10, 1e-9])).toBe(true);
    expect(np.allclose([1e10, 1e-7], [1.00001e10, 1e-8])).toBe(false);
    expect(np.allclose([], [])).toBe(true);
    expect(np.arrayEqual([1, 2], [1, 2])).toBe(true);
    expect(np.arrayEqual([1, 2], [1, 2, 3])).toBe(false);
    expect(np.arrayEqual([1, 2], [[1, 2]])).toBe(false);
    expect(np.arrayEqual([1, NaN], [1, NaN])).toBe(false);
    expect(np.arrayEqual([1, NaN], [1, NaN], { equalNan: true })).toBe(true);
    expect(np.arrayEqual([NaN, 1], [1, NaN], { equalNan: true })).toBe(false);
    expect(np.arrayEqual([1, 2], [1, 3], { equalNan: true })).toBe(false);
    expect(np.arrayEqual([[1, 2], [3]] as never, [1])).toBe(false);
    expect(np.arrayEquiv([1, 2], [[1, 2], [1, 2]])).toBe(true);
    expect(np.arrayEquiv([1, 2], [[1, 2], [1, 3]])).toBe(false);
    expect(np.arrayEquiv([1, 2], [1, 2, 3])).toBe(false);
  });
});

describe("P5 packbits/unpackbits (D-084)", () => {
  const a = np.array([[2, 7, 23], [4, 5, 0]], { dtype: "uint8" });
  it("packbits", () => {
    expect(np.packbits(a).toArray()).toEqual([248]);
    expect(np.packbits(a, { axis: 1 }).toArray()).toEqual([[224], [192]]);
    expect(np.packbits(a, { axis: 0 }).toArray()).toEqual([[192, 192, 128]]);
    expect(np.packbits(a, { axis: -1, bitorder: "little" }).toArray()).toEqual([[7], [3]]);
    expect(np.packbits(np.array(5)).toArray()).toEqual([128]);
    expect(np.packbits([-1, 0, 2]).toArray()).toEqual([160]);
    expect(np.packbits(Array(9).fill(true)).toArray()).toEqual([255, 128]);
    expect(np.packbits(np.array([], { dtype: "bool" })).shape).toEqual([0]);
    expect(np.packbits(np.zeros([2, 0], { dtype: "bool" }), { axis: 0 }).shape).toEqual([1, 0]);
    expect(np.packbits(a).dtype).toBe(np.uint8);
    const t = np.array([[true, false], [false, true]]).T;
    expect(np.packbits(t, { axis: 1 }).toArray()).toEqual([[128], [64]]);
    expect(() => np.packbits(np.array([1.5]))).toThrow(DTypeError);
    expect(() => np.packbits([1], { axis: 1 })).toThrow(IndexError);
    expect(() => np.packbits([1], { bitorder: "x" as never })).toThrow(ValueError);
  });

  it("unpackbits", () => {
    const u = np.array([[2], [7], [23]], { dtype: "uint8" });
    expect(np.unpackbits(u, { axis: 1 }).toArray()).toEqual([
      [0, 0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 1, 1, 1],
      [0, 0, 0, 1, 0, 1, 1, 1],
    ]);
    expect(np.unpackbits(u).shape).toEqual([24]);
    expect(np.unpackbits(u, { axis: 1, count: 3 }).toArray()).toEqual([[0, 0, 0], [0, 0, 0], [0, 0, 0]]);
    expect(np.unpackbits(u, { axis: 1, count: -3, bitorder: "little" }).toArray()).toEqual([
      [0, 1, 0, 0, 0],
      [1, 1, 1, 0, 0],
      [1, 1, 1, 0, 1],
    ]);
    expect(np.unpackbits(u, { axis: 1, count: 10 }).shape).toEqual([3, 10]);
    expect(np.unpackbits(u, { count: 0 }).shape).toEqual([0]);
    expect(np.unpackbits(np.array(5, { dtype: "uint8" })).toArray()).toEqual([0, 0, 0, 0, 0, 1, 0, 1]);
    expect(np.unpackbits(u, { axis: 0 }).shape).toEqual([24, 1]);
    expect(() => np.unpackbits(u, { axis: 1, count: -9 })).toThrow(ValueError);
    expect(() => np.unpackbits(np.array([1], { dtype: "int8" }))).toThrow(DTypeError);
    expect(() => np.unpackbits(u, { axis: 2 })).toThrow(IndexError);
    expect(() => np.unpackbits(u, { bitorder: "x" as never })).toThrow(ValueError);
    const bits = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1];
    expect(np.unpackbits(np.packbits(bits), { count: 10 }).toArray()).toEqual(bits);
  });
});
