import { describe, expect, it } from "vitest";
import np from "../src/index.js";

const c = (re: number, im: number) => ({ re, im });

describe("np.emath (P15-1, D-180)", () => {
  it("keeps in-domain real input real, with the ufunc float loop", () => {
    expect(np.emath.sqrt([4, 0.25]).toArray()).toEqual([2, 0.5]);
    expect(np.emath.sqrt(np.array([4], { dtype: "float32" })).dtype.name).toBe("float32");
    expect(np.emath.sqrt(np.array([4], { dtype: "int8" })).dtype.name).toBe("float16");
    expect(np.emath.log2([8]).toArray()).toEqual([3]);
    expect(np.emath.log10([100]).toArray()).toEqual([2]);
    expect(np.emath.arcsin([0]).toArray()).toEqual([0]);
    expect(np.emath.arccos([1]).toArray()).toEqual([0]);
  });

  it("returns complex results for out-of-domain real input", () => {
    const r = np.emath.sqrt([-4, 4]);
    expect(r.dtype.name).toBe("complex128");
    expect(r.toArray()).toEqual([c(0, 2), c(2, 0)]);
    expect(np.emath.sqrt(np.array([-4], { dtype: "float32" })).dtype.name).toBe("complex64");
    expect(np.emath.sqrt(np.array([-4], { dtype: "int16" })).dtype.name).toBe("complex64");
    expect(np.emath.sqrt(np.array([-4], { dtype: "float16" })).dtype.name).toBe("complex128");
    const l = np.emath.log(-1).item() as { re: number; im: number };
    expect(l.re).toBe(0);
    expect(l.im).toBeCloseTo(Math.PI, 15);
    const a = np.emath.arcsin([2]).item() as { re: number; im: number };
    expect(a.re).toBeCloseTo(Math.PI / 2, 15);
    expect(a.im).toBeCloseTo(1.3169578969248166, 14);
    const t = np.emath.arctanh([-4]).item() as { re: number; im: number };
    expect(t.re).toBeCloseTo(-0.2554128118829953, 14);
    expect(t.im).toBeCloseTo(Math.PI / 2, 15);
    expect((np.emath.arccos([2]).item() as { im: number }).im).toBeCloseTo(-1.3169578969248166, 14);
  });

  it("keeps complex input complex", () => {
    expect(np.emath.sqrt(np.array([c(-4, 0)])).toArray()).toEqual([c(0, 2)]);
  });

  it("logn and power", () => {
    expect(np.emath.logn(2, [4, 8]).toArray()).toEqual([2, 3]);
    const l = np.emath.logn(2, [-8]).item() as { re: number; im: number };
    expect(l.re).toBeCloseTo(3, 14);
    expect(l.im).toBeCloseTo(4.532360141827194, 13);
    expect(np.emath.power([2, 4], -1).toArray()).toEqual([0.5, 0.25]);
    expect(np.emath.power([2, 4], 2).dtype.name).toBe("int64");
    const p = np.emath.power([-4], 0.5).item() as { re: number; im: number };
    expect(p.re).toBeCloseTo(0, 15);
    expect(p.im).toBeCloseTo(2, 15);
  });

  it("edge cases: empty, 0-d, NaN, Inf, bool", () => {
    expect(np.emath.sqrt([]).shape).toEqual([0]);
    expect(np.emath.sqrt(4).shape).toEqual([]);
    expect(np.emath.sqrt([NaN]).toArray()).toEqual([NaN]);
    expect(np.errstate({ divide: "ignore" }, () => np.emath.log([0]).toArray())).toEqual([-Infinity]);
    expect(np.emath.sqrt([true]).dtype.name).toBe("float16");
    expect(np.errstate({ divide: "raise" }, () => () => np.emath.log([0]))).toBeTypeOf("function");
    expect(() => np.errstate({ divide: "raise" }, () => np.emath.log([0]))).toThrow(np.FloatingPointError);
  });
});
