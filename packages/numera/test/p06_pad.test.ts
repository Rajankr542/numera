import { describe, expect, it } from "vitest";
import np, { DTypeError } from "../src/index.js";

describe("P6 pad (D-092)", () => {
  const a = np.arange(4);
  it("width forms and constant", () => {
    expect(np.pad(a, 1).toArray()).toEqual([0, 0, 1, 2, 3, 0]);
    expect(np.pad(a, [1, 2]).toArray()).toEqual([0, 0, 1, 2, 3, 0, 0]);
    expect(np.pad([[1]], [[1, 0], [0, 2]]).toArray()).toEqual([[0, 0, 0], [1, 0, 0]]);
    expect(np.pad([[1]], { 1: [0, 1] }).toArray()).toEqual([[1, 0]]);
    expect(np.pad([[1]], { [-1]: 1 }).shape).toEqual([1, 3]);
    expect(np.pad(np.array(5), 3).toArray()).toEqual(5);
    expect(np.pad(a, 1, "constant", { constantValues: [7, 8] }).toArray()).toEqual([7, 0, 1, 2, 3, 8]);
    expect(np.pad([[1, 2], [3, 4]], 1, "constant", { constantValues: [[7, 8], [9, 10]] }).toArray()).toEqual([
      [9, 7, 7, 10], [9, 1, 2, 10], [9, 3, 4, 10], [9, 8, 8, 10],
    ]);
    expect(np.pad(np.array([1], { dtype: "int8" }), 1, "constant", { constantValues: 1.9 }).toArray()).toEqual([1, 1, 1]);
    expect(np.pad(np.zeros([0]), 2).toArray()).toEqual([0, 0, 0, 0]);
    expect(np.pad(a, 1, "empty").shape).toEqual([6]);
  });
  it("edge / reflect / symmetric / wrap", () => {
    expect(np.pad(a, [2, 1], "edge").toArray()).toEqual([0, 0, 0, 1, 2, 3, 3]);
    expect(np.pad(a, 5, "reflect").toArray()).toEqual([1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2]);
    expect(np.pad(a, 5, "symmetric").toArray()).toEqual([3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0]);
    expect(np.pad(a, 5, "wrap").toArray()).toEqual([3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0]);
    expect(np.pad(a, 2, "reflect", { reflectType: "odd" }).toArray()).toEqual([-2, -1, 0, 1, 2, 3, 4, 5]);
    expect(np.pad(np.array([1, 2, 3], { dtype: "int8" }), 3, "symmetric", { reflectType: "odd" }).toArray()).toEqual([
      -1, 0, 1, 1, 2, 3, 3, 4, 5,
    ]);
    expect(np.pad(np.array([1, 2], { dtype: "uint8" }), 2, "reflect", { reflectType: "odd" }).toArray()).toEqual([
      255, 0, 1, 2, 3, 4,
    ]);
    expect(np.pad([7], 2, "reflect").toArray()).toEqual([7, 7, 7, 7, 7]);
    expect(np.pad([[1, 2], [3, 4]], [[1, 2], [2, 1]], "wrap").toArray()).toEqual([
      [3, 4, 3, 4, 3], [1, 2, 1, 2, 1], [3, 4, 3, 4, 3], [1, 2, 1, 2, 1], [3, 4, 3, 4, 3],
    ]);
  });
  it("statistics and linear_ramp", () => {
    expect(np.pad(a, 1, "maximum").toArray()).toEqual([3, 0, 1, 2, 3, 3]);
    expect(np.pad(a, 1, "minimum", { statLength: [2, 1] }).toArray()).toEqual([0, 0, 1, 2, 3, 3]);
    expect(np.pad(a, 1, "mean").toArray()).toEqual([2, 0, 1, 2, 3, 2]);
    expect(np.pad(np.array([1, 2, 3, 4], { dtype: "float16" }), 1, "mean").toArray()).toEqual([2.5, 1, 2, 3, 4, 2.5]);
    expect(np.pad([1, 2, 3, 10], [1, 0], "median").toArray()).toEqual([2, 1, 2, 3, 10]);
    expect(np.pad([1.0, 2, 3, 10.5], [1, 0], "median").toArray()).toEqual([2.5, 1, 2, 3, 10.5]);
    expect(np.pad([1, NaN, 3], 1, "maximum").toArray()).toEqual([NaN, 1, NaN, 3, NaN]);
    expect(np.pad([1, NaN, 3], 1, "median").toArray()).toEqual([NaN, 1, NaN, 3, NaN]);
    expect(np.pad([1.5, 2], 1, "mean", { statLength: 0 }).toArray()).toEqual([NaN, 1.5, 2, NaN]);
    expect(() => np.pad(a, 1, "maximum", { statLength: 0 })).toThrow(/stat_length of 0/);
    expect(np.pad(np.array([{ re: 1, im: 2 }, { re: 3, im: 0 }]), 1, "mean").get(0).item()).toEqual({ re: 2, im: 1 });
    expect(np.pad(a, 3, "linear_ramp", { endValues: [5, -1] }).toArray()).toEqual([5, 3, 1, 0, 1, 2, 3, 1, 0, -1]);
    expect(np.pad(np.array([0, 3], { dtype: "float32" }), [3, 0], "linear_ramp").toArray()).toEqual([0, 0, 0, 0, 3]);
    expect(np.pad([true, false], 2, "linear_ramp", { endValues: 1 }).toArray()).toEqual([true, true, true, false, true, true]);
  });
  it("layout, callable mode and errors", () => {
    expect(np.pad(np.zeros([2, 3], { order: "F" }), 1, "edge").flags.fContiguous).toBe(true);
    const r = np.pad(a, 2, (v, w) => {
      v.set([[0, w[0]]], 9);
    });
    expect(r.toArray()).toEqual([9, 9, 0, 1, 2, 3, 0, 0]);
    expect(() => np.pad(a, 1, "bogus" as never)).toThrow(/not supported/);
    expect(() => np.pad(a, 1, "edge", { constantValues: 1 })).toThrow(/unsupported keyword/);
    expect(() => np.pad(a, 1.5)).toThrow(DTypeError);
    expect(() => np.pad(a, -1)).toThrow(/negative/);
    expect(() => np.pad(np.zeros([0, 2]), 1, "edge")).toThrow(/can't extend empty axis 0/);
    expect(np.pad(np.zeros([0, 2]), [[0, 0], [1, 1]], "wrap").shape).toEqual([0, 4]);
    expect(() => np.pad(a, [1, 2, 3])).toThrow(/broadcast/);
  });
});
