import { describe, expect, it } from "vitest";
import np, { DTypeError, IndexError, NDArray, NotImplementedError, ValueError } from "../src/index.js";

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

describe("P10 cumulative ops, diff, ptp", () => {
  it("cumsum / cumprod", () => {
    const m = np.array([[1, 2], [3, 4]], { dtype: "int8" });
    expect(np.cumsum(m).dtype.name).toBe("int64");
    expect(np.cumsum(m).toArray()).toEqual([1, 3, 6, 10]);
    expect(np.cumprod(m, { axis: 0 }).toArray()).toEqual([[1, 2], [3, 8]]);
    expect(m.cumsum({ axis: 1 }).toArray()).toEqual([[1, 3], [3, 7]]);
    expect(m.cumprod().toArray()).toEqual([1, 2, 6, 24]);
    expect(np.cumsum([1, 2], { dtype: "float32" }).dtype.name).toBe("float32");
    expect(np.cumsum(5).shape).toEqual([1]);
    const out = np.zeros(3, { dtype: "int64" });
    expect(np.cumsum([1.5, 1, 1], { out }).toArray()).toEqual([1, 2, 3]);
    expect(out.toArray()).toEqual([1, 2, 3]);
  });
  it("cumulativeSum / cumulativeProd / nancum*", () => {
    expect(np.cumulativeSum([1, 2, 3], { includeInitial: true }).toArray()).toEqual([0, 1, 3, 6]);
    expect(np.cumulativeProd(np.ones([2, 2]), { axis: 1, includeInitial: true }).toArray()).toEqual([[1, 1, 1], [1, 1, 1]]);
    expect(() => np.cumulativeSum(np.ones([2, 2]))).toThrow(ValueError);
    expect(np.nancumsum([1, NaN, 2]).toArray()).toEqual([1, 1, 3]);
    expect(np.nancumprod([2, NaN, 2]).toArray()).toEqual([2, 2, 4]);
  });
  it("diff", () => {
    expect(np.diff([1, 4, 9, 16]).toArray()).toEqual([3, 5, 7]);
    expect(np.diff([1, 4, 9, 16], 2).toArray()).toEqual([2, 2]);
    expect(np.diff([[1, 2], [4, 8]], { axis: 0 }).toArray()).toEqual([[3, 6]]);
    expect(np.diff([1, 2], { prepend: 0, append: [5] }).toArray()).toEqual([1, 1, 3]);
    expect(np.diff([true, false, false]).toArray()).toEqual([true, false]);
    expect(np.diff(np.array([5, 1], { dtype: "uint8" })).toArray()).toEqual([252]);
    expect(() => np.diff(5)).toThrow(ValueError);
    expect(() => np.diff([1, 2], -1)).toThrow(ValueError);
  });
  it("ptp", () => {
    expect(np.ptp([[1, 5], [2, 9]], { axis: 0 }).toArray()).toEqual([1, 4]);
    expect(np.array([[1, 5], [2, 9]]).ptp({ axis: 1, keepdims: true }).toArray()).toEqual([[4], [7]]);
    expect(np.ptp(np.array([-100, 100], { dtype: "int8" })).item()).toBe(-56);
    expect(() => np.ptp([true])).toThrow(DTypeError);
    expect(() => np.ptp([])).toThrow(ValueError);
  });
});

describe("P10 NaN reductions", () => {
  const x = np.array([[1, NaN, 3], [NaN, NaN, NaN]]);
  it("sum / prod / mean / var / std", () => {
    expect(np.nansum(x, { axis: 1 }).toArray()).toEqual([4, 0]);
    expect(np.nanprod(x, { axis: 1 }).toArray()).toEqual([3, 1]);
    expect(np.nanmean(x, { axis: 1 }).toArray()).toEqual([2, NaN]);
    expect(np.nanvar(x, { axis: 1 }).toArray()).toEqual([1, NaN]);
    expect(np.nanstd(x, { axis: 1, keepdims: true }).toArray()).toEqual([[1], [NaN]]);
    expect(np.nanvar([1, NaN, 2], { ddof: 1 }).item()).toBe(0.5);
    expect(np.nanvar(np.array([1, NaN, 2], { dtype: "float32" })).dtype.name).toBe("float32");
    expect(np.nansum(np.array([100, 100], { dtype: "int8" })).dtype.name).toBe("int64");
    expect(np.nanmean(np.array([1, 2], { dtype: "int8" })).item()).toBe(1.5);
    expect(np.nanmean([]).item()).toBeNaN();
    expect(() => np.nanmean([1, NaN], { dtype: "int64" })).toThrow(DTypeError);
  });
  it("min / max / argmin / argmax", () => {
    expect(np.nanmin(x, { axis: 1 }).toArray()).toEqual([1, NaN]);
    expect(np.nanmax(x, { axis: 1 }).toArray()).toEqual([3, NaN]);
    expect(np.nanmin(x, { axis: 1, initial: 0 }).toArray()).toEqual([0, 0]);
    expect(Object.is(np.nanmin([0, -0]).item(), -0)).toBe(true);
    expect(np.nanargmax([NaN, 2, 5, NaN]).item()).toBe(2);
    expect(np.nanargmin([[NaN, 1], [2, 3]], { axis: 0 }).toArray()).toEqual([1, 0]);
    expect(() => np.nanargmin(x, { axis: 1 })).toThrow(ValueError);
    expect(() => np.nanmin([])).toThrow(ValueError);
  });
});

describe("P10 average / cov / corrcoef / gradient / trapezoid", () => {
  it("average", () => {
    const [avg, scl] = np.average(np.ones([2, 3]), { axis: 1, weights: [1, 2, 3], returned: true });
    expect(avg.toArray()).toEqual([1, 1]);
    expect(scl.toArray()).toEqual([6, 6]);
    expect(np.average(np.arange(6).reshape(2, 3)).item()).toBe(2.5);
    expect(np.average([[1, 2], [3, 4]], { axis: 0, keepdims: true }).toArray()).toEqual([[2, 3]]);
    expect(() => np.average([1, 2], { weights: [1, -1] })).toThrow(ValueError);
    expect(() => np.average([[1, 2]], { weights: [1, 2] })).toThrow(DTypeError);
  });
  it("cov / corrcoef", () => {
    expect(np.cov([1, 2, 3]).item()).toBe(1);
    expect(np.cov([1, 2, 3], { y: [1, 5, 2] }).toArray()).toEqual([[1, 0.5], [0.5, 4.333333333333334]]);
    expect(np.cov([[1, 2], [3, 4]], { rowvar: false, aweights: [1, 2], fweights: [2, 1] }).toArray()).toEqual([
      [1.6, 1.6],
      [1.6, 1.6],
    ]);
    expect(np.cov([[1, 2, 3]], { bias: true }).item()).toBeCloseTo(2 / 3, 15);
    expect(() => np.cov([1, 2], { fweights: [1.5, 1] })).toThrow(DTypeError);
    expect(() => np.cov([1, 2], { ddof: 1.5 })).toThrow(ValueError);
    expect(np.corrcoef([[1, 2, 3], [1, 5, 2]]).toArray()).toEqual([
      [1, 0.24019223070763066],
      [0.24019223070763066, 0.9999999999999998],
    ]);
  });
  it("gradient", () => {
    expect((np.gradient([1, 2, 4, 7, 11]) as NDArray).toArray()).toEqual([1, 1.5, 2.5, 3.5, 4]);
    expect((np.gradient([1, 2, 4, 7, 11], 2) as NDArray).toArray()).toEqual([0.5, 0.75, 1.25, 1.75, 2]);
    expect((np.gradient([1, 2, 4, 7, 11], { edgeOrder: 2 }) as NDArray).toArray()).toEqual([0.5, 1.5, 2.5, 3.5, 4.5]);
    expect((np.gradient([1, 2, 4, 7, 11], [0, 1, 3, 4, 7]) as NDArray).toArray()).toEqual([
      1, 1, 2.3333333333333326, 2.5833333333333326, 1.3333333333333333,
    ]);
    const g = np.gradient([[1, 2, 6], [3, 4, 5]]) as NDArray[];
    expect(g.map((a) => a.toArray())).toEqual([
      [[2, 2, -1], [2, 2, -1]],
      [[1, 2.5, 4], [1, 1, 1]],
    ]);
    expect((np.gradient([[1, 2, 6], [3, 4, 5]], { axis: 1 }) as NDArray).toArray()).toEqual([[1, 2.5, 4], [1, 1, 1]]);
    expect(() => np.gradient([1, 2], { edgeOrder: 2 })).toThrow(ValueError);
  });
  it("trapezoid", () => {
    expect(np.trapezoid([1, 2, 3]).item()).toBe(4);
    expect(np.trapezoid([1, 2, 3], { x: [0, 1, 3] }).item()).toBe(6.5);
    expect(np.trapezoid([1, 2, 3], { dx: 0.5 }).item()).toBe(2);
    expect(np.trapezoid([[1, 2], [3, 4]], { axis: 0 }).toArray()).toEqual([2, 3]);
  });
});
