import { describe, expect, it } from "vitest";
import np, { DTypeError, IndexError, ValueError } from "../src/index.js";

const errorLike = (cls: new (m: string) => Error, message: string) =>
  expect.objectContaining({ name: new cls(message).name, message });
const L = (a: { toArray(): unknown }): unknown => a.toArray();
const P = [5, 1, 4, 1, 3, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3, 2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5];

describe("P9-1 sort / argsort (D-120)", () => {
  it("sorts every dtype with NaN last", () => {
    expect(L(np.sort([3, NaN, 1, 2]))).toEqual([1, 2, 3, NaN]);
    expect(L(np.sort([3, NaN, 1], { descending: true }))).toEqual([3, 1, NaN]);
    for (const dt of ["bool", "int8", "uint16", "int64", "uint64", "float16", "float32", "complex64"]) {
      const s = np.sort(np.array([2, 0, 1], { dtype: dt }));
      expect(s.dtype.name).toBe(dt);
    }
    expect(L(np.sort(np.array([1, 0, 1], { dtype: "bool" })))).toEqual([false, true, true]);
  });
  it("orders complex lexicographically", () => {
    const s = np.sort([{ re: 1, im: 2 }, { re: 1, im: 1 }, { re: 0, im: 5 }]);
    expect(L(s)).toEqual([{ re: 0, im: 5 }, { re: 1, im: 1 }, { re: 1, im: 2 }]);
  });
  it("axis, axis null and 0-d", () => {
    expect(L(np.sort([[3, 1], [0, 2]], { axis: 0 }))).toEqual([[0, 1], [3, 2]]);
    expect(L(np.sort([[3, 1], [0, 2]], { axis: null }))).toEqual([0, 1, 2, 3]);
    expect(L(np.sort(np.array(4), { axis: null }))).toEqual([4]);
    expect(() => np.sort(np.array(4))).toThrow(IndexError);
  });
  it("argsort: stable kinds keep ties in order, descending keeps NaN last", () => {
    expect(L(np.argsort([1, 0, 1, 0], { stable: true }))).toEqual([1, 3, 0, 2]);
    expect(L(np.argsort([1, 0, 1, 0], { kind: "mergesort" }))).toEqual([1, 3, 0, 2]);
    expect(L(np.argsort([3, 1, 2, 1, 3, NaN, 2], { descending: true }))).toEqual([0, 4, 2, 6, 1, 3, 5]);
    expect(np.argsort([2, 1]).dtype.name).toBe("int64");
  });
  it("kind validation", () => {
    expect(L(np.argsort([3, 1, 2], { kind: "H" }))).toEqual([1, 2, 0]);
    expect(() => np.sort([1], { kind: "xyz" })).toThrow(errorLike(ValueError, "sort kind must be one of 'quick', 'heap', or 'stable' (got 'xyz')"));
    expect(() => np.sort([1], { kind: "quicksort", stable: false })).toThrow(ValueError);
  });
  it("NDArray.sort is in place; argsort method", () => {
    const a = np.array([[3, 1, 2], [9, 7, 8]]);
    const view = a.get(0);
    a.sort();
    expect(L(a)).toEqual([[1, 2, 3], [7, 8, 9]]);
    expect(L(view)).toEqual([1, 2, 3]);
    expect(L(a.argsort({ axis: 0, descending: true }))).toEqual([[1, 1, 1], [0, 0, 0]]);
    expect(() => a.sort({ axis: null as unknown as number })).toThrow(DTypeError);
    expect(() => np.broadcastTo([1, 2], [2, 2]).sort()).toThrow(ValueError);
  });
});

describe("P9-2 partition / argpartition (D-121)", () => {
  it("reproduces NumPy introselect", () => {
    expect(L(np.partition(P, 10))).toEqual(
      [2, 1, 2, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 5, 9, 8, 7, 6, 9, 6, 8, 5, 5, 8, 6, 9, 7, 9, 5],
    );
    expect(L(np.argpartition(P, 10))).toEqual(
      [6, 1, 21, 3, 16, 28, 15, 24, 17, 4, 25, 27, 9, 19, 23, 2, 0, 14, 18, 13, 20, 12, 22, 11, 10, 8, 26, 7, 5, 29, 30, 31],
    );
  });
  it("kth lists, negative kth and the in-place method", () => {
    const sorted = [...P].sort((x, y) => x - y);
    const r = np.partition(P, [3, -1, 20]).toArray() as number[];
    for (const k of [3, 20, 31]) expect(r[k]).toBe(sorted[k]);
    const a = np.array([[3, 1, 2], [0, 5, 4]]);
    a.partition(0, { axis: 0 });
    expect(L(a)).toEqual([[0, 1, 2], [3, 5, 4]]);
    expect(L(np.array([4, 3, 9]).argpartition(np.array([0, 2])))).toEqual([1, 0, 2]);
  });
  it("validates kth and kind", () => {
    expect(() => np.partition([1, 2, 3], 3)).toThrow(errorLike(ValueError, "kth(=3) out of bounds (3)"));
    expect(() => np.partition([1, 2], true as unknown as number)).toThrow(errorLike(ValueError, "Booleans unacceptable as partition index"));
    expect(() => np.partition([1, 2], 0.5)).toThrow(DTypeError);
    expect(() => np.partition([1, 2], [])).toThrow(DTypeError);
    expect(() => np.partition([1, 2], 0, { kind: "x" })).toThrow(errorLike(ValueError, "select kind must be 'introselect' (got 'x')"));
    expect(L(np.partition(np.zeros([2, 0]), 5))).toEqual([[], []]);
    expect(L(np.argpartition(np.array(5), 0, { axis: null }))).toEqual([0]);
  });
});

describe("P9-3 lexsort / searchsorted / sortComplex (D-122)", () => {
  it("lexsort uses the last key as primary", () => {
    const surnames = [3, 1, 2, 1];
    const first = [0, 2, 1, 1];
    expect(L(np.lexsort([first, surnames]))).toEqual([3, 1, 2, 0]);
    expect(L(np.lexsort(np.array([[1, 2], [3, 1]])))).toEqual([1, 0]);
    expect(() => np.lexsort([])).toThrow(DTypeError);
    expect(() => np.lexsort([[1, 2], [1, 2, 3]])).toThrow(errorLike(ValueError, "all keys need to be the same shape"));
  });
  it("searchsorted sides, sorter, scalars and dtype promotion", () => {
    expect(L(np.searchsorted([1, 2, 2, 3], [2, 0, 4]))).toEqual([1, 0, 4]);
    expect(L(np.searchsorted([1, 2, 2, 3], [2, 0, 4], { side: "right" }))).toEqual([3, 0, 4]);
    expect(np.searchsorted([1, 2, 3], 2).ndim).toBe(0);
    expect(np.searchsorted(np.array([1, 2, 3], { dtype: "int8" }), 2.5).item()).toBe(2);
    expect(np.searchsorted(np.array([1, 2, 250], { dtype: "uint8" }), 300).item()).toBe(3);
    expect(np.searchsorted(np.array([0.1], { dtype: "float32" }), 0.1, { side: "right" }).item()).toBe(1);
    expect(L(np.searchsorted([3, 1, 2], [2.5], { sorter: [1, 2, 0] }))).toEqual([2]);
    expect(L(np.array([1, 3]).searchsorted([2]))).toEqual([1]);
    expect(() => np.searchsorted([1, 2], 1, { side: "x" as "left" })).toThrow(ValueError);
    expect(() => np.searchsorted([[1, 2]], 1)).toThrow(errorLike(ValueError, "object too deep for desired array"));
    expect(() => np.searchsorted([1, 2], 1, { sorter: [0, 5] })).toThrow(errorLike(ValueError, "Sorter index out of range."));
    expect(() => np.searchsorted([1, 2], 1, { sorter: [0] })).toThrow(errorLike(ValueError, "sorter.size must equal a.size"));
    expect(() => np.searchsorted([1, 2], 1, { sorter: [0.5, 1] })).toThrow(DTypeError);
  });
  it("sortComplex result dtypes", () => {
    expect(np.sortComplex(np.array([2, 1], { dtype: "int8" })).dtype.name).toBe("complex64");
    expect(np.sortComplex([2, 1]).dtype.name).toBe("complex128");
    expect(L(np.sortComplex([{ re: 1, im: 2 }, { re: 1, im: -1 }]))).toEqual([{ re: 1, im: -1 }, { re: 1, im: 2 }]);
  });
});

describe("P9-4 unique (D-123)", () => {
  it("values and optional outputs", () => {
    expect(L(np.unique([1, 1, 2, 2, 3, 3]))).toEqual([1, 2, 3]);
    const r = np.unique([1, 3, 4, 3], { returnIndex: true, returnInverse: true, returnCounts: true });
    expect([r.values, r.indices, r.inverse, r.counts].map((x) => x?.toArray())).toEqual([
      [1, 3, 4], [0, 1, 2], [0, 1, 2, 1], [1, 2, 1],
    ]);
    expect(L(np.unique([[1, 2], [2, 1]], { returnInverse: true }).inverse!)).toEqual([[0, 1], [1, 0]]);
  });
  it("NaN handling and signed zero", () => {
    expect(L(np.unique([NaN, 1, NaN]))).toEqual([1, NaN]);
    expect(L(np.unique([NaN, 1, NaN], { equalNan: false }))).toEqual([1, NaN, NaN]);
    expect(L(np.unique([0, -0]))).toEqual([0]); // which zero survives is unspecified (COMPATIBILITY)
  });
  it("axis", () => {
    const r = np.unique([[1, 0], [0, 1], [1, 0]], { axis: 0, returnIndex: true, returnInverse: true, returnCounts: true });
    expect([r.values, r.indices, r.inverse, r.counts].map((x) => x?.toArray())).toEqual([
      [[0, 1], [1, 0]], [1, 0], [1, 0, 1], [1, 2],
    ]);
    expect(L(np.unique([[1, 0, 1], [1, 0, 1]], { axis: 1 }))).toEqual([[0, 1], [0, 1]]);
    expect(np.unique(np.zeros([3, 0]), { axis: 0, returnCounts: true }).counts!.toArray()).toEqual([3]);
    expect(() => np.unique([1, 2], { axis: 1 })).toThrow(IndexError);
  });
  it("Array API helpers", () => {
    const a = np.uniqueAll([2, 1, 2]);
    expect([a.values, a.indices, a.inverseIndices, a.counts].map((x) => x.toArray())).toEqual([
      [1, 2], [1, 0], [1, 0, 1], [1, 2],
    ]);
    expect(L(np.uniqueCounts([2, 1, 2]).counts)).toEqual([1, 2]);
    expect(L(np.uniqueInverse([2, 1, 2]).inverseIndices)).toEqual([1, 0, 1]);
    expect(L(np.uniqueValues([NaN, NaN]))).toEqual([NaN, NaN]);
  });
});

describe("P9-5 set functions, isin, ediff1d (D-124)", () => {
  it("intersect / union / setdiff / setxor", () => {
    expect(L(np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1]))).toEqual([1, 3]);
    const r = np.intersect1d([1, 3, 4, 3], [3, 1, 2, 1], { returnIndices: true });
    expect([r.values, r.indices1, r.indices2].map((x) => x.toArray())).toEqual([[1, 3], [0, 1], [1, 0]]);
    expect(L(np.union1d([-1, 0, 1], [-2, 0, 2]))).toEqual([-2, -1, 0, 1, 2]);
    expect(L(np.setdiff1d([1, 2, 3, 2, 4, 1], [3, 4, 5, 6]))).toEqual([1, 2]);
    expect(L(np.setxor1d([1, 2, 3, 2, 4], [2, 3, 5, 7, 5]))).toEqual([1, 4, 5, 7]);
    expect(L(np.setxor1d([NaN, 1], [NaN, 2]))).toEqual([1, 2, NaN, NaN]);
    expect(np.union1d(np.array([1], { dtype: "int8" }), [0.5]).dtype.name).toBe("float64");
  });
  it("isin keeps the element shape", () => {
    expect(L(np.isin([[0, 2], [4, 6]], [1, 2, 4, 8]))).toEqual([[false, true], [true, false]]);
    expect(L(np.isin([1, 5], [1], { invert: true, kind: "table" }))).toEqual([false, true]);
    expect(np.isin(np.array(3), [3]).ndim).toBe(0);
    expect(() => np.isin([1], [1], { kind: "x" as "sort" })).toThrow(ValueError);
    expect(() => np.isin([1.5], [1], { kind: "table" })).toThrow(ValueError);
  });
  it("ediff1d", () => {
    expect(L(np.ediff1d([[1, 2], [4, 7]], { toBegin: [-9], toEnd: [100, 200] }))).toEqual([-9, 1, 2, 3, 100, 200]);
    expect(L(np.ediff1d(np.array(5)))).toEqual([]);
    expect(L(np.ediff1d(np.array([1, 0], { dtype: "uint8" })))).toEqual([255]);
    expect(() => np.ediff1d([true, false])).toThrow(DTypeError);
    expect(() => np.ediff1d(np.array([1, 2], { dtype: "uint8" }), { toBegin: 1 })).toThrow(DTypeError);
  });
});
