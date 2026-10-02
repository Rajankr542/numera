import { describe, expect, it } from "vitest";
import np, { DTypeError } from "../src/index.js";

describe("P4-3 rounding", () => {
  it("floor / ceil / trunc / rint / fix", () => {
    const x = [-1.5, -0.5, 0.5, 1.5, 2.5];
    expect(np.floor(x).toArray()).toEqual([-2, -1, 0, 1, 2]);
    expect(np.ceil(x).toArray()).toEqual([-1, -0, 1, 2, 3]);
    expect(np.trunc(x).toArray()).toEqual([-1, -0, 0, 1, 2]);
    expect(np.rint(x).toArray()).toEqual([-2, -0, 0, 2, 2]);
    expect(np.fix(x).toArray()).toEqual([-1, -0, 0, 1, 2]);
    expect(Object.is((np.trunc([-0.5]).toArray() as number[])[0], -0)).toBe(true);
    expect(np.floor([NaN, Infinity]).toArray()).toEqual([NaN, Infinity]);
    expect(np.floor([]).shape).toEqual([0]);
  });

  it("dtypes follow NumPy", () => {
    expect(np.floor(np.array([3], { dtype: "int16" })).dtype.name).toBe("int16");
    expect(np.ceil([true]).dtype.name).toBe("bool");
    expect(np.fix([1, -2]).dtype.name).toBe("int64");
    expect(np.rint(np.array([1], { dtype: "int16" })).dtype.name).toBe("float32");
    expect(np.trunc(np.array([1.5], { dtype: "float32" })).dtype.name).toBe("float32");
    expect(() => np.floor(np.zeros(1, { dtype: "complex128" }))).toThrow(DTypeError);
    expect(np.rint(np.array([{ re: 1.5, im: 2.5 }])).toArray()).toEqual([{ re: 2, im: 2 }]);
  });

  it("round / around with decimals", () => {
    expect(np.round([0.5, 1.5, 2.5, -0.5]).toArray()).toEqual([0, 2, 2, -0]);
    expect(np.round([1.25, 2.5], 1).toArray()).toEqual([1.2, 2.5]);
    expect(np.around([2.675], 2).toArray()).toEqual([2.68]);
    expect(np.round([15, 25, -35], -1).toArray()).toEqual([20, 20, -40]);
    const i8 = np.round(np.array([15, 25, 127], { dtype: "int8" }), -1);
    expect(i8.dtype.name).toBe("int8");
    expect(i8.toArray()).toEqual([20, 20, -126]); // wraps like NumPy
    expect(np.round(np.array([1234], { dtype: "uint16" }), -2).toArray()).toEqual([1200]);
    expect(np.round([1, 2], 2).dtype.name).toBe("int64");
    expect(np.round([true]).dtype.name).toBe("float16");
    expect(np.round(np.array([1.25], { dtype: "float16" }), 1).dtype.name).toBe("float16");
    expect(np.round(2.5).toArray()).toBe(2);
    expect(np.round([1.5], -400).toArray()).toEqual([NaN]);
    expect(np.round([{ re: 1.5, im: 2.5 }]).toArray()).toEqual([{ re: 2, im: 2 }]);
    expect(() => np.round([1], 0.5)).toThrow(DTypeError);
    expect(() => np.round([true], 1)).toThrow(DTypeError);
  });

  it("round out= and NDArray.round", () => {
    const out = np.zeros(2, { dtype: "float32" });
    expect(np.round([1.25, 2.5], 1, { out })).toBe(out);
    expect(out.toArray()).toEqual([Math.fround(1.2), 2.5]);
    expect(() => np.round([1.5], 0, { out: np.zeros(1, { dtype: "int64" }) })).toThrow(DTypeError);
    const o2 = np.zeros(1, { dtype: "float32" });
    np.round(np.array([12345], { dtype: "int16" }), -2, { out: o2 });
    expect(o2.toArray()).toEqual([12300]);
    expect(np.array([[1.26, 2.5]]).round(1).toArray()).toEqual([[1.3, 2.5]]);
    expect(np.array([3.5]).round().toArray()).toEqual([4]);
  });

  it("positive", () => {
    expect(np.positive([-3, 2]).toArray()).toEqual([-3, 2]);
    expect(np.positive(np.array([1], { dtype: "uint8" })).dtype.name).toBe("uint8");
    expect(np.positive([{ re: 1, im: 1 }]).toArray()).toEqual([{ re: 1, im: 1 }]);
    expect(() => np.positive([true])).toThrow(DTypeError);
  });
});
