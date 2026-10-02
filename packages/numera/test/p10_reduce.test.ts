import { describe, expect, it } from "vitest";
import np, { ValueError, DTypeError, NDArray } from "../src/index.js";

// P10-7: where= and out= for np.sum/prod/min/max/mean/var/std (D-136).

describe("P10-7 where= / out= for reduction functions", () => {
  // ---- sum ----
  describe("np.sum with where=", () => {
    it("basic boolean mask", () => {
      const a = np.array([1, 2, 3, 4, 5], { dtype: "float64" });
      const mask = np.array([true, false, true, false, true]);
      // sum of elements 0,2,4 = 1+3+5 = 9
      const r = np.sum(a, { where: mask });
      expect(r.item()).toBeCloseTo(9);
    });

    it("2-D with axis", () => {
      const a = np.array([[1, 2, 3], [4, 5, 6]], { dtype: "float64" });
      const mask = np.array([[true, false, true], [true, true, false]]);
      // axis=0: col0: 1+4=5, col1: 0+5=5, col2: 3+0=3
      const r = np.sum(a, { axis: 0, where: mask });
      expect(r.toArray()).toEqual([5, 5, 3]);
    });

    it("keepdims", () => {
      const a = np.array([1.0, 2.0, 3.0, 4.0]);
      const mask = np.array([true, false, true, false]);
      const r = np.sum(a, { keepdims: true, where: mask });
      expect(r.shape).toEqual([1]);
      expect(r.item()).toBeCloseTo(4);
    });

    it("broadcasts scalar-true mask (no-op)", () => {
      const a = np.array([1.0, 2.0, 3.0]);
      // A [3] boolean mask of all true = same as plain sum.
      const mask = np.array([true, true, true]);
      expect(np.sum(a, { where: mask }).item()).toBeCloseTo(np.sum(a).item());
    });
  });

  // ---- prod ----
  describe("np.prod with where=", () => {
    it("basic mask — masked-out treated as 1", () => {
      const a = np.array([2.0, 3.0, 4.0, 5.0]);
      const mask = np.array([true, false, true, false]);
      // 2 * 4 = 8
      expect(np.prod(a, { where: mask }).item()).toBeCloseTo(8);
    });
  });

  // ---- min / max ----
  describe("np.min / np.max with where= (require initial=)", () => {
    it("min with initial=", () => {
      const a = np.array([1.0, 2.0, 3.0, 4.0]);
      const mask = np.array([false, true, false, true]);
      // only 2 and 4 count; identity/initial=1e10
      const r = np.min(a, { where: mask, initial: 1e10 });
      expect(r.item()).toBeCloseTo(2);
    });

    it("max with initial=", () => {
      const a = np.array([1.0, 2.0, 3.0, 4.0]);
      const mask = np.array([true, false, true, false]);
      // only 1 and 3 count; initial=-1e10
      const r = np.max(a, { where: mask, initial: -1e10 });
      expect(r.item()).toBeCloseTo(3);
    });

    it("min without initial= throws", () => {
      const a = np.array([1.0, 2.0, 3.0]);
      const mask = np.array([true, false, true]);
      expect(() => np.min(a, { where: mask })).toThrow(ValueError);
    });

    it("max without initial= throws", () => {
      const a = np.array([1.0, 2.0, 3.0]);
      const mask = np.array([true, false, true]);
      expect(() => np.max(a, { where: mask })).toThrow(ValueError);
    });
  });

  // ---- mean ----
  describe("np.mean with where=", () => {
    it("basic mask", () => {
      const a = np.array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const mask = np.array([true, false, true, false, true]);
      // mean of [1,3,5] = 3
      expect(np.mean(a, { where: mask }).item()).toBeCloseTo(3);
    });

    it("2-D axis=1", () => {
      const a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
      const mask = np.array([[true, true, false], [false, true, true]]);
      // row0: (1+2)/2=1.5; row1: (5+6)/2=5.5
      const r = np.mean(a, { axis: 1, where: mask });
      expect(r.toArray()[0]).toBeCloseTo(1.5);
      expect(r.toArray()[1]).toBeCloseTo(5.5);
    });
  });

  // ---- var / std ----
  describe("np.var / np.std with where=", () => {
    it("var of subset", () => {
      // var([1,3,5]) = var of [1,3,5] = 8/3 ≈ 2.667
      const a = np.array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const mask = np.array([true, false, true, false, true]);
      const v = np.var(a, { where: mask });
      expect(v.item()).toBeCloseTo(8 / 3, 5);
    });

    it("std of subset", () => {
      const a = np.array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const mask = np.array([true, false, true, false, true]);
      const s = np.std(a, { where: mask });
      expect(s.item()).toBeCloseTo(Math.sqrt(8 / 3), 5);
    });

    it("var ddof=1", () => {
      // sample var([1,3,5]) = ((0-0)^2+(2-0)^2+(-2-0)^2... wait:
      // values [1,3,5], mean=3; (1-3)^2+(3-3)^2+(5-3)^2 = 4+0+4=8; /ddof1=2 → 4
      const a = np.array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const mask = np.array([true, false, true, false, true]);
      const v = np.var(a, { where: mask, ddof: 1 });
      expect(v.item()).toBeCloseTo(4, 5);
    });
  });

  // ---- out= ----
  describe("out= parameter", () => {
    it("sum writes into out", () => {
      const a = np.array([1.0, 2.0, 3.0]);
      const out = np.zeros([]);
      const r = np.sum(a, { out });
      // r should be the same object / same buffer as out
      expect(r.item()).toBeCloseTo(6);
      expect(out.item()).toBeCloseTo(6);
    });

    it("mean writes into out", () => {
      const a = np.array([2.0, 4.0, 6.0]);
      const out = np.zeros([]);
      np.mean(a, { out });
      expect(out.item()).toBeCloseTo(4);
    });

    it("out shape mismatch throws", () => {
      const a = np.array([[1.0, 2.0], [3.0, 4.0]]);
      const out = np.zeros([3]);  // wrong shape for axis=0 result
      expect(() => np.sum(a, { axis: 0, out })).toThrow(ValueError);
    });

    it("sum with where= and out=", () => {
      const a = np.array([1.0, 2.0, 3.0, 4.0]);
      const mask = np.array([true, true, false, false]);
      const out = np.zeros([]);
      np.sum(a, { where: mask, out });
      // only 1+2=3
      expect(out.item()).toBeCloseTo(3);
    });
  });

  // ---- error cases ----
  describe("error cases", () => {
    it("non-boolean where= raises DTypeError", () => {
      const a = np.array([1.0, 2.0, 3.0]);
      const mask = np.array([1.0, 0.0, 1.0]);  // float64 mask, not bool
      expect(() => np.sum(a, { where: mask })).toThrow(DTypeError);
    });
  });
});
