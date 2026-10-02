import { describe, expect, it } from "vitest";
import np, { DTypeError, IndexError, NotImplementedError, ValueError } from "../src/index.js";

describe("P10 quantile / percentile / median", () => {
  const a = np.arange(10).astype("float64");
  const qs = [0, 0.1, 0.5, 0.9, 1];
  it("methods", () => {
    expect(np.quantile(a, qs).toArray()).toEqual([0, 0.9, 4.5, 8.1, 9]);
    expect(np.quantile(a, qs, { method: "lower" }).toArray()).toEqual([0, 0, 4, 8, 9]);
    expect(np.quantile(a, qs, { method: "nearest" }).toArray()).toEqual([0, 1, 4, 8, 9]);
    expect(np.quantile(a, qs, { method: "midpoint" }).toArray()).toEqual([0, 0.5, 4.5, 8.5, 9]);
    expect(np.quantile(a, qs, { method: "hazen" }).toArray()).toEqual([0, 0.5, 4.5, 8.5, 9]);
    expect(np.percentile(a, [10, 50]).toArray()).toEqual([0.9, 4.5]);
    expect(() => np.quantile(a, 0.5, { method: "bad" as never })).toThrow(ValueError);
  });
  it("dtypes", () => {
    const i8 = np.array([1, 2, 3, 4], { dtype: "int8" });
    expect(np.quantile(i8, 0.5).dtype.name).toBe("float64");
    expect(np.quantile(i8, 1).dtype.name).toBe("int8");
    expect(np.quantile(i8, 0.5, { method: "higher" }).dtype.name).toBe("int8");
    expect(np.quantile(i8, [0, 1], { method: "weibull" }).dtype.name).toBe("int64");
    expect(np.quantile(np.array([1, 2, 3], { dtype: "float32" }), 0.3).dtype.name).toBe("float32");
    expect(np.quantile(np.array([1, 2, 3], { dtype: "float32" }), [0.3]).dtype.name).toBe("float64");
    expect(np.quantile(np.array([-100, 100], { dtype: "int8" }), 0.75).item()).toBe(114);
  });
  it("axis, keepdims, q shape", () => {
    const m = np.arange(6).reshape(2, 3).astype("float64");
    const r = np.quantile(m, [[0.1, 0.5]], { axis: 1, keepdims: true });
    expect(r.shape).toEqual([1, 2, 2, 1]);
    expect(np.median(np.arange(24).reshape(2, 3, 4), { axis: [0, 2] }).toArray()).toEqual([7.5, 11.5, 15.5]);
  });
  it("weights", () => {
    const m = np.arange(6).reshape(2, 3).astype("float64");
    expect(np.quantile(m, 0.5, { axis: 1, weights: [1, 2, 3], method: "inverted_cdf" }).toArray()).toEqual([1, 4]);
    expect(() => np.quantile(m, 0.5, { axis: 1, weights: [1, 2, 3] })).toThrow(ValueError);
    expect(() => np.quantile(m, 0.5, { weights: [1, 2, 3], method: "inverted_cdf" })).toThrow(DTypeError);
  });
  it("NaN and errors", () => {
    expect(np.median([1, NaN, 3]).item()).toBeNaN();
    expect(np.nanmedian([1, NaN, 3]).item()).toBe(2);
    expect(np.nanquantile([[1, NaN], [NaN, NaN]], [0.5], { axis: 1 }).toArray()).toEqual([[1, NaN]]);
    expect(np.nanpercentile([1, NaN, 3, 5], 50).item()).toBe(3);
    expect(np.median([]).item()).toBeNaN();
    expect(() => np.quantile([], 0.5)).toThrow(IndexError);
    expect(() => np.quantile([1, 2], 1.5)).toThrow(ValueError);
    expect(() => np.percentile([1, 2], 101)).toThrow(ValueError);
    expect(() => np.quantile([true, false], 0.5)).toThrow(DTypeError);
    expect(() => np.quantile(np.array([1, 2], { dtype: "complex128" }), 0.5)).toThrow(DTypeError);
    expect(() => np.median(np.array([1, 2], { dtype: "complex128" }))).toThrow(NotImplementedError);
  });
});
