import { describe, expect, it } from "vitest";
import np, { DTypeError, ValueError } from "../src/index.js";

const close = (a: unknown, b: number[], tol = 1e-12) => {
  const v = a as number[];
  expect(v.length).toBe(b.length);
  v.forEach((x, i) => {
    if (!Number.isFinite(b[i]!)) expect(x).toBe(b[i]);
    else expect(Math.abs(x - b[i]!)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i]!)));
  });
};

describe("P4-6 special", () => {
  it("i0", () => {
    // Values from numpy 2.x np.i0.
    close(np.i0([0, 1, -2.5, 30]).toArray(), [1, 1.2660658777520082, 3.2898391440501231, 781672297823.9775]);
    expect(np.i0([-800]).toArray()).toEqual([Infinity]);
    expect(np.i0(np.array([1], { dtype: "int8" })).dtype.name).toBe("float64");
    expect(np.i0(np.array([1], { dtype: "float32" })).dtype.name).toBe("float32");
    expect(np.i0(np.array([1], { dtype: "float16" })).dtype.name).toBe("float16");
    expect(np.i0(np.array(2)).toArray()).toBeCloseTo(2.2795853023360673, 14);
    expect(np.i0([]).shape).toEqual([0]);
    expect(() => np.i0([{ re: 1, im: 1 }])).toThrow(DTypeError);
  });

  it("sinc", () => {
    close(np.sinc([0, 0.5, 1]).toArray(), [1, 0.6366197723675814, 3.8981718325193755e-17], 1e-15);
    expect(np.sinc(np.array([0, 1], { dtype: "int8" })).dtype.name).toBe("float64");
    expect(np.sinc(np.array([0.5], { dtype: "float32" })).dtype.name).toBe("float32");
    expect(np.sinc(np.array([0], { dtype: "float16" })).toArray()).toEqual([1]);
    const z = np.sinc([{ re: 0.5, im: 1 }]).toArray() as { re: number; im: number }[];
    expect(z[0]!.re).toBeCloseTo(1.47593333, 7);
    expect(z[0]!.im).toBeCloseTo(-2.95186666, 7);
    expect(np.sinc([{ re: 0, im: 0 }]).toArray()).toEqual([{ re: 1, im: 0 }]);
  });

  it("nanToNum", () => {
    const r = np.nanToNum([NaN, Infinity, -Infinity, 1]).toArray() as number[];
    expect(r).toEqual([0, Number.MAX_VALUE, -Number.MAX_VALUE, 1]);
    expect(np.nanToNum(np.array([NaN, Infinity, -Infinity], { dtype: "float16" })).toArray()).toEqual([0, 65504, -65504]);
    expect(np.nanToNum([NaN, Infinity, -Infinity], { nan: 1, posinf: 2, neginf: 3 }).toArray()).toEqual([1, 2, 3]);
    expect(np.nanToNum([{ re: NaN, im: Infinity }], { posinf: 7 }).toArray()).toEqual([{ re: 0, im: 7 }]);
    expect(np.nanToNum([1, 2]).dtype.name).toBe("int64");
    const a = np.array([NaN, 1]);
    expect(np.nanToNum(a, { copy: false, nan: 5 })).not.toBe(undefined);
    expect(a.toArray()).toEqual([5, 1]);
    const b = np.array([NaN]);
    np.nanToNum(b);
    expect(b.toArray()).toEqual([NaN]);
  });

  it("realIfClose", () => {
    const r = np.realIfClose([{ re: 1, im: 1e-15 }]);
    expect(r.dtype.name).toBe("float64");
    expect(r.toArray()).toEqual([1]);
    expect(np.realIfClose([{ re: 1, im: 1e-10 }]).dtype.name).toBe("complex128");
    expect(np.realIfClose([{ re: 1, im: 1e-10 }], 1e-9).dtype.name).toBe("float64");
    expect(np.realIfClose([{ re: 1, im: 0.5 }], 0.6).toArray()).toEqual([1]);
    expect(np.realIfClose(np.array([{ re: 1, im: 1e-5 }], { dtype: "complex64" })).dtype.name).toBe("float32");
    expect(np.realIfClose([1, 2]).toArray()).toEqual([1, 2]);
  });

  it("unwrap", () => {
    close(np.unwrap([0, 3, 6.5, 0.2]).toArray(), [0, 3, 6.5 - 2 * Math.PI, 0.2]);
    close(np.unwrap([0, 3, 6.5], { discont: 1 }).toArray(), [0, 3, 6.5 - 2 * Math.PI]);
    expect(np.unwrap([0, 3, 6.5], { discont: 5 }).toArray()).toEqual([0, 3, 6.5]);
    expect(np.unwrap([0, 7, 14], { period: 10 }).toArray()).toEqual([0, -3, -6]);
    expect(np.unwrap([0, 5, 10], { period: 10 }).toArray()).toEqual([0, 5, 10]);
    expect(np.unwrap([0, -5, -10], { period: 10 }).toArray()).toEqual([0, -5, -10]);
    expect(np.unwrap([0, 5, 10.5], { period: 10 }).dtype.name).toBe("float64");
    expect(np.unwrap([0, 6, 12, 1], { period: 6 }).toArray()).toEqual([0, 0, 0, 1]);
    expect(np.unwrap([0, 3, -3, 0], { period: 6 }).toArray()).toEqual([0, 3, 3, 6]);
    const i8 = np.unwrap(np.array([0, 7, 14], { dtype: "int8" }), { period: 10 });
    expect(i8.dtype.name).toBe("int8");
    expect(i8.toArray()).toEqual([0, -3, -6]);
    expect(np.unwrap([0, 3, 7]).dtype.name).toBe("float64");
    expect(np.unwrap(np.array([0, 3, 7], { dtype: "float32" })).dtype.name).toBe("float32");
    expect(np.unwrap([[0, 7], [14, 1]], { period: 10, axis: 0 }).toArray()).toEqual([[0, 7], [4, 11]]);
    expect(np.unwrap([true, false, true], { period: 1, discont: 0 }).toArray()).toEqual([1, -1, -1]);
    expect(np.unwrap([1.5]).toArray()).toEqual([1.5]);
    expect(np.unwrap([[]]).shape).toEqual([1, 0]);
    expect(() => np.unwrap(np.array(1.5))).toThrow(ValueError);
    expect(() => np.unwrap([0, 3], { axis: 1 })).toThrow();
    expect(() => np.unwrap([{ re: 0, im: 1 }])).toThrow(DTypeError);
  });
});
