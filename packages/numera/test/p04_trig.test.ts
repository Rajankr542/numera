import { describe, expect, it } from "vitest";
import np, { DTypeError } from "../src/index.js";

const close = (a: unknown, b: number[], tol = 1e-12) => {
  const v = a as number[];
  expect(v.length).toBe(b.length);
  v.forEach((x, i) => {
    if (Number.isNaN(b[i]!)) expect(x).toBeNaN();
    else if (!Number.isFinite(b[i]!)) expect(x).toBe(b[i]);
    else expect(Math.abs(x - b[i]!)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b[i]!)));
  });
};

describe("P4-1 trig / hyperbolic", () => {
  it("computes values for float64", () => {
    const x = [0, 0.5, -1];
    close(np.sin(x).toArray(), x.map(Math.sin));
    close(np.cos(x).toArray(), x.map(Math.cos));
    close(np.tan(x).toArray(), x.map(Math.tan));
    close(np.arcsin(x).toArray(), x.map(Math.asin));
    close(np.arccos(x).toArray(), x.map(Math.acos));
    close(np.arctan(x).toArray(), x.map(Math.atan));
    close(np.sinh(x).toArray(), x.map(Math.sinh));
    close(np.cosh(x).toArray(), x.map(Math.cosh));
    close(np.tanh(x).toArray(), x.map(Math.tanh));
    close(np.arcsinh(x).toArray(), x.map(Math.asinh));
    close(np.arccosh([1, 2]).toArray(), [0, Math.acosh(2)]);
    close(np.arctanh(x).toArray(), [0, Math.atanh(0.5), -Infinity]);
    close(np.arctan2([1, -1], [1, -1]).toArray(), [Math.PI / 4, (-3 * Math.PI) / 4]);
    close(np.hypot([3, 5], 4).toArray(), [5, Math.hypot(5, 4)]);
    close(np.degrees([Math.PI]).toArray(), [180]);
    close(np.rad2deg([Math.PI]).toArray(), [180]);
    close(np.radians([180]).toArray(), [Math.PI]);
    close(np.deg2rad([90]).toArray(), [Math.PI / 2]);
  });

  it("handles NaN/Inf, empty and 0-d inputs", () => {
    expect(np.arcsin([2]).toArray()).toEqual([NaN]);
    expect(np.tanh([Infinity, -Infinity]).toArray()).toEqual([1, -1]);
    expect(np.sin([]).shape).toEqual([0]);
    expect(np.sin(np.array(0)).toArray()).toBe(0);
    expect(np.hypot(Infinity, NaN).toArray()).toBe(Infinity);
  });

  it("follows NumPy loop dtypes", () => {
    expect(np.sin(np.array([1], { dtype: "int8" })).dtype.name).toBe("float16");
    expect(np.sin(np.array([1], { dtype: "uint16" })).dtype.name).toBe("float32");
    expect(np.sin(np.array([1], { dtype: "int32" })).dtype.name).toBe("float64");
    expect(np.sin(np.array([true])).dtype.name).toBe("float16");
    expect(np.cos(np.array([1], { dtype: "float32" })).dtype.name).toBe("float32");
    expect(np.arctan2(np.array([1], { dtype: "int8" }), np.array([2], { dtype: "uint16" })).dtype.name).toBe("float32");
    expect(np.arctan2(np.array([1], { dtype: "int8" }), np.array([2], { dtype: "int8" })).dtype.name).toBe("float16");
    expect(np.sin([1], { dtype: "float32" }).dtype.name).toBe("float32");
    expect(() => np.sin([1], { dtype: "int32" })).toThrow(DTypeError);
  });

  it("has complex loops for the unary functions only", () => {
    const z = np.arcsin(np.array([{ re: 2, im: 0 }]));
    expect(z.dtype.name).toBe("complex128");
    const v = (z.toArray() as { re: number; im: number }[])[0]!;
    expect(v.re).toBeCloseTo(Math.PI / 2, 12);
    expect(Math.abs(v.im)).toBeCloseTo(1.3169578969248166, 12);
    expect(np.cosh(np.zeros(2, { dtype: "complex64" })).dtype.name).toBe("complex64");
    expect(() => np.arctan2(np.zeros(1, { dtype: "complex128" }), 1)).toThrow(DTypeError);
    expect(() => np.degrees(np.zeros(1, { dtype: "complex128" }))).toThrow(DTypeError);
  });

  it("aliases are the same objects and ufunc methods work", () => {
    expect(np.asin).toBe(np.arcsin);
    expect(np.acos).toBe(np.arccos);
    expect(np.atan).toBe(np.arctan);
    expect(np.asinh).toBe(np.arcsinh);
    expect(np.acosh).toBe(np.arccosh);
    expect(np.atanh).toBe(np.arctanh);
    expect(np.atan2).toBe(np.arctan2);
    expect(np.hypot.reduce([3, 4]).toArray()).toBe(5);
    expect(np.hypot.reduce([], {}).toArray()).toBe(0); // identity 0
    const out = np.zeros(2);
    expect(np.sin([0, 0], { out })).toBe(out);
    expect(np.sin([1, 2], { where: [true, false] }).toArray()).toEqual([Math.sin(1), 0]);
    const o = np.arctan2.outer([1], [1, -1]).toArray() as number[][];
    close(o[0], [Math.PI / 4, (3 * Math.PI) / 4]);
  });
});
