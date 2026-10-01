import { describe, expect, it } from "vitest";
import np, { DTypeError, IndexError, NDArray, ValueError } from "../src/index.js";

const ar = (n: number, dtype = "int64") => np.arange(n, undefined, undefined, { dtype });
const list = (xs: NDArray[]) => xs.map((x) => x.toArray());

describe("P8 take mode= / takeAlongAxis / putAlongAxis", () => {
  const a = ar(6).reshape(2, 3);
  it("take modes", () => {
    expect(np.take(a, [-1, 7], { mode: "wrap" }).toArray()).toEqual([5, 1]);
    expect(np.take(a, [-9, 7], null, { mode: "clip" }).toArray()).toEqual([0, 5]);
    expect(np.take(a, [2, 0], { axis: 1, mode: "raise" }).toArray()).toEqual([[2, 0], [5, 3]]);
    expect(np.take(a, [[1]], 0).shape).toEqual([1, 1, 3]);
    expect(() => np.take(a, [6])).toThrow(IndexError);
    expect(() => np.take(a, [0], { mode: "bad" as never })).toThrow(ValueError);
    expect(() => np.take(np.zeros(0), [0], { mode: "wrap" })).toThrow(IndexError);
    expect(np.take(np.zeros(0), [], { mode: "wrap" }).shape).toEqual([0]);
    expect(() => np.take(a, np.array([0.5]))).toThrow(DTypeError);
    expect(np.take([1.5, NaN], [1]).toArray()).toEqual([NaN]);
  });
  it("NDArray.take", () => {
    expect(a.take([1], 1).toArray()).toEqual([[1], [4]]);
    expect(a.take([7], { mode: "clip" }).toArray()).toEqual([5]);
    expect(a.T.take([1]).toArray()).toEqual([3]);
  });
  it("takeAlongAxis", () => {
    expect(np.takeAlongAxis(a, np.array([[2], [0]]), 1).toArray()).toEqual([[2], [3]]);
    expect(np.takeAlongAxis(a, [[1, 0, 0]], 0).toArray()).toEqual([[3, 1, 2]]);
    expect(np.takeAlongAxis(a, [[0, 2]]).toArray()).toEqual([[0, 2], [3, 5]]);
    expect(np.takeAlongAxis(a, [5, 0], null).toArray()).toEqual([5, 0]);
    expect(() => np.takeAlongAxis(a, [1], 1)).toThrow(ValueError);
    expect(() => np.takeAlongAxis(a, [[1, 2]], null)).toThrow(ValueError);
    expect(() => np.takeAlongAxis(a, [[3]], 1)).toThrow(IndexError);
    expect(() => np.takeAlongAxis(a, [[0.5]], 1)).toThrow(IndexError);
  });
  it("putAlongAxis", () => {
    const b = a.copy();
    np.putAlongAxis(b, [[2], [0]], 99, 1);
    expect(b.toArray()).toEqual([[0, 1, 99], [99, 4, 5]]);
    const c = a.copy();
    np.putAlongAxis(c, [5], 7.9, null);
    expect(c.item(1, 2)).toBe(7);
    expect(() => np.putAlongAxis(np.broadcastTo(1, [2, 2]), [[0]], 1, 0)).toThrow(ValueError);
  });
});

describe("P8 put / putmask / place", () => {
  it("put", () => {
    const x = np.arange(5, undefined, undefined, { dtype: "float64" });
    expect(np.put(x, [0, 7], [9, 8], { mode: "wrap" })).toBeUndefined();
    expect(x.toArray()).toEqual([9, 1, 8, 3, 4]);
    np.put(x, [-9], [5], { mode: "clip" });
    expect(x.item(0)).toBe(5);
    expect(() => np.put(x, [5], [1])).toThrow(IndexError);
    const i = ar(5, "int8");
    np.put(i, [0, 1, 2], [1.9, -3.2]); // values repeat, unsafe cast
    expect(i.toArray()).toEqual([1, -3, 1, 3, 4]);
    const m = ar(6).reshape(2, 3);
    m.T.put([1], 9); // flat order of the transposed view
    expect(m.item(1, 0)).toBe(9);
    expect(() => np.put(np.zeros(0), [0], [1])).toThrow(IndexError);
    expect(() => np.put(np.broadcastTo(1, [3]), [0], [2])).toThrow(ValueError);
  });
  it("putmask", () => {
    const x = np.arange(5, undefined, undefined, { dtype: "float64" });
    np.putmask(x, [false, false, true, true, true], [10, 20]);
    expect(x.toArray()).toEqual([0, 1, 10, 20, 10]);
    const y = ar(6).reshape(2, 3).T;
    np.putmask(y, [[false, false], [false, true], [true, true]], [10, 20, 30]);
    expect(y.toArray()).toEqual([[0, 3], [1, 10], [20, 30]]);
    const z = ar(3, "int32");
    np.putmask(z, [true, false, true], 1.7); // JS values convert to the target dtype
    expect(z.toArray()).toEqual([1, 1, 1]);
    expect(() => np.putmask(z, [true, false, true], np.array([1.5]))).toThrow(DTypeError);
    expect(() => np.putmask(z, [true], 1)).toThrow(ValueError);
    np.putmask(z, [true, true, true], []);
    expect(z.toArray()).toEqual([1, 1, 1]);
  });
  it("place", () => {
    const x = np.arange(5, undefined, undefined, { dtype: "float64" });
    np.place(x, [0, 0, 1, 1, 1], [10, 20]);
    expect(x.toArray()).toEqual([0, 1, 10, 20, 10]);
    const y = ar(6).reshape(2, 3).T;
    np.place(y, [[false, false], [false, true], [true, true]], [10, 20]);
    expect(y.toArray()).toEqual([[0, 3], [1, 10], [20, 10]]);
    expect(() => np.place(ar(3), [1, 0, 1], [])).toThrow(ValueError);
    np.place(ar(3), [0, 0, 0], []);
    expect(() => np.place(ar(3, "int32"), [1, 0, 1], np.array([1], { dtype: "int64" }))).toThrow(DTypeError);
  });
});

describe("P8 choose / compress / extract / select", () => {
  it("choose", () => {
    const ch = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    expect(np.choose([0, 1, 2], ch).toArray()).toEqual([1, 5, 9]);
    expect(np.choose([0, 5], [[1, 2], [3, 4]], { mode: "clip" }).toArray()).toEqual([1, 4]);
    expect(np.choose([-1, 2], [[1, 2], [3, 4]], { mode: "wrap" }).toArray()).toEqual([3, 2]);
    expect(() => np.choose([0, 5], [[1, 2], [3, 4]])).toThrow(ValueError);
    expect(() => np.choose([0], [])).toThrow(ValueError);
    expect(() => np.choose([0.5], [[1], [2]])).toThrow(DTypeError);
    expect(np.choose([[0], [1]], [[1, 2, 3], [4, 5, 6]]).toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
    const r = np.choose([0, 1], [np.array([1, 2], { dtype: "int8" }), 3]);
    expect(r.dtype.name).toBe("int8");
    expect(r.toArray()).toEqual([1, 3]);
    expect(np.choose(1, [3, 4]).toArray()).toBe(4);
    expect(np.choose([1, 0], np.array([[1, 2], [3, 4]])).toArray()).toEqual([3, 2]);
    expect(np.array([1, 0]).choose([[1, 2], [3, 4]]).toArray()).toEqual([3, 2]);
  });
  it("compress / extract", () => {
    const a = ar(6).reshape(2, 3);
    expect(np.compress([0, 1], a, 0).toArray()).toEqual([[3, 4, 5]]);
    expect(np.compress([1, 0, 1, 1], a).toArray()).toEqual([0, 2, 3]);
    expect(np.compress([true, false], a, { axis: 1 }).toArray()).toEqual([[0], [3]]);
    expect(() => np.compress([1, 0, 1, 1], a, 1)).toThrow(IndexError);
    expect(() => np.compress([[1]], [1])).toThrow(ValueError);
    expect(np.compress([true], 5).toArray()).toEqual([5]);
    expect(a.compress([1, 0]).toArray()).toEqual([0]);
    expect(np.extract([[0, 0, 0], [1, 1, 1]], a).toArray()).toEqual([3, 4, 5]);
    expect(np.extract([1, 0, 1], [1, 2, 3, 4]).toArray()).toEqual([1, 3]);
    expect(np.extract(true, 5).toArray()).toEqual([5]);
  });
  it("select", () => {
    const r = np.select([[true, false], [true, true]], [[1, 2], [3.5, 4]], { default: 9 });
    expect(r.toArray()).toEqual([1, 4]);
    expect(r.dtype.name).toBe("float64");
    const i8 = np.array([1, 2], { dtype: "int8" });
    expect(np.select([[true, false]], [i8]).dtype.name).toBe("int8");
    expect(np.select([[true, false]], [i8], { default: 1.5 }).dtype.name).toBe("float64");
    expect(np.select([[false, false]], [[1, 2]]).toArray()).toEqual([0, 0]);
    expect(() => np.select([[1, 0]], [[1, 2]])).toThrow(DTypeError);
    expect(() => np.select([], [])).toThrow(ValueError);
    expect(() => np.select([[true]], [])).toThrow(ValueError);
  });
});

describe("P8 piecewise / argwhere / flatnonzero / countNonzero", () => {
  it("piecewise", () => {
    const x = np.array([-1, 0, 1.5]);
    const r = np.piecewise(x, [[true, false, false], [false, false, true]], [(v) => np.negative(v), 5, 7]);
    expect(r.toArray()).toEqual([1, 7, 5]);
    const n = np.arange(4);
    expect(np.piecewise(n, np.array([false, false, true, true]), [(v) => np.multiply(v, 10), -1]).toArray())
      .toEqual([-1, -1, 20, 30]);
    expect(np.piecewise([1, 2, 3], [true, false, true], [7]).toArray()).toEqual([7, 0, 7]);
    expect(np.piecewise(np.array(2.0), [true], [(v) => np.multiply(v, 3)]).toArray()).toBe(6);
    expect(np.piecewise([0, 1, 2], [[false, true, true]], [(v) => np.divide(v, 2)]).toArray()).toEqual([0, 0, 1]);
    let calls = 0;
    np.piecewise([1, 2], [[false, false]], [() => { calls++; return 0; }]);
    expect(calls).toBe(0);
    expect(() => np.piecewise([1, 2], [[true, false]], [1, 2, 3])).toThrow(ValueError);
  });
  it("argwhere / flatnonzero", () => {
    expect(np.argwhere([[0, 3], [4, 0]]).toArray()).toEqual([[0, 1], [1, 0]]);
    expect(np.argwhere(5).shape).toEqual([1, 0]);
    expect(np.argwhere(0).shape).toEqual([0, 0]);
    expect(np.argwhere(np.zeros([2, 0])).shape).toEqual([0, 2]);
    expect(np.flatnonzero([[0, 1], [2, 0]]).toArray()).toEqual([1, 2]);
    expect(np.flatnonzero(3).toArray()).toEqual([0]);
    expect(np.flatnonzero([NaN, 0, -0]).toArray()).toEqual([0]);
  });
  it("countNonzero", () => {
    const a = ar(6).reshape(2, 3);
    expect(np.countNonzero(a).toArray()).toBe(5);
    expect(np.countNonzero(a).dtype.name).toBe("int64");
    expect(np.countNonzero(a, { axis: 0, keepdims: true }).toArray()).toEqual([[1, 2, 2]]);
    expect(np.countNonzero(a, { axis: [0, 1] }).toArray()).toBe(5);
    expect(np.countNonzero([[NaN, 0], [0, 1]]).toArray()).toBe(2);
    expect(np.countNonzero([np.complex(0, 1), np.complex(0, 0)]).toArray()).toBe(1);
    expect(np.countNonzero(np.zeros(0)).toArray()).toBe(0);
    expect(np.countNonzero(5).toArray()).toBe(1);
    expect(() => np.countNonzero(a, { axis: 2 })).toThrow(IndexError);
  });
});

describe("P8 ravelMultiIndex / unravelIndex", () => {
  it("ravelMultiIndex", () => {
    expect(np.ravelMultiIndex([[1, 2], [3, 9]], [3, 4], { mode: ["raise", "clip"] }).toArray()).toEqual([7, 11]);
    expect(np.ravelMultiIndex([1, 2], [3, 4], { order: "F" }).toArray()).toBe(7);
    expect(np.ravelMultiIndex([-1, 5], [3, 4], { mode: "wrap" }).toArray()).toBe(9);
    expect(np.ravelMultiIndex([[[0, 1]], [[2], [3]]], [2, 4]).toArray()).toEqual([[2, 6], [3, 7]]);
    expect(np.ravelMultiIndex([[], []], [3, 4]).shape).toEqual([0]);
    expect(() => np.ravelMultiIndex([3, 1], [3, 4])).toThrow(ValueError);
    expect(() => np.ravelMultiIndex([1], [3, 4])).toThrow(ValueError);
    expect(() => np.ravelMultiIndex([1, 2], [3, 4], { order: "K" })).toThrow(ValueError);
    expect(() => np.ravelMultiIndex([1, 2], [3, 4], { mode: ["clip"] })).toThrow(ValueError);
    expect(() => np.ravelMultiIndex([1.5, 2], [3, 4])).toThrow(DTypeError);
    expect(() => np.ravelMultiIndex([1, 2], [3, -4])).toThrow(ValueError);
  });
  it("unravelIndex", () => {
    expect(list(np.unravelIndex([5, 7], [3, 4]))).toEqual([[1, 1], [1, 3]]);
    expect(list(np.unravelIndex(5, [3, 4], { order: "F" }))).toEqual([2, 1]);
    expect(list(np.unravelIndex([[1, 2]], [3, 4]))).toEqual([[[0, 0]], [[1, 2]]]);
    expect(list(np.unravelIndex([], [3, 4]))).toEqual([[], []]);
    expect(np.unravelIndex(0, [])).toEqual([]);
    expect(() => np.unravelIndex(12, [3, 4])).toThrow(ValueError);
    expect(() => np.unravelIndex(-1, [3, 4])).toThrow(ValueError);
    expect(() => np.unravelIndex(1.5, [3])).toThrow(DTypeError);
  });
});

describe("P8 diagonal / trace / NDArray methods", () => {
  it("diagonal is a read-only view", () => {
    const a = ar(24).reshape(2, 3, 4);
    const d = np.diagonal(a, { offset: 1, axis1: 0, axis2: 2 });
    expect(d.toArray()).toEqual([[1, 14], [5, 18], [9, 22]]);
    expect(d.flags.writeable).toBe(false);
    expect(d.strides).toEqual([32, 104]);
    expect(np.mayShareMemory(a, d)).toBe(true);
    expect(() => d.set(0, 1)).toThrow(ValueError);
    const m = ar(9, "float32").reshape(3, 3);
    expect(m.diagonal().toArray()).toEqual([0, 4, 8]);
    expect(m.diagonal(-1).toArray()).toEqual([3, 7]);
    expect(np.diagonal(m, 5).shape).toEqual([0]);
    expect(m.diagonal().dtype.name).toBe("float32");
    expect(() => np.diagonal(ar(3))).toThrow(ValueError);
    expect(() => np.diagonal(m, { axis1: 1, axis2: -1 })).toThrow(ValueError);
    expect(() => np.diagonal(m, { axis2: 2 })).toThrow(IndexError);
  });
  it("trace", () => {
    expect(np.trace(ar(24).reshape(2, 3, 4), { offset: 1, axis1: 0, axis2: 2 }).toArray()).toEqual([15, 23, 31]);
    expect(np.trace(np.eye(3, 3, { dtype: "int8" })).dtype.name).toBe("int64");
    expect(np.trace(np.ones([2, 2], { dtype: "uint8" })).dtype.name).toBe("uint64");
    expect(np.trace(np.ones([2, 2], { dtype: "uint8" }), { dtype: "float32" }).dtype.name).toBe("float32");
    expect(np.trace(np.eye(2, 2, { dtype: "bool" })).toArray()).toBe(2);
    expect(np.trace(np.zeros([0, 0])).toArray()).toBe(0);
    expect(np.trace([[1, NaN], [3, 4]]).toArray()).toBe(5);
    expect(np.trace([[1, 2], [3, 4]], 1).toArray()).toBe(2);
    expect(ar(4).reshape(2, 2).trace().toArray()).toBe(3);
    expect(() => np.trace([1, 2])).toThrow(ValueError);
  });
  it("NDArray.nonzero / put", () => {
    expect(list(np.array([[1, 0], [0, 1]]).nonzero())).toEqual([[0, 1], [0, 1]]);
    const y = np.zeros(3);
    y.put([1], 5);
    expect(y.toArray()).toEqual([0, 5, 0]);
  });
});
