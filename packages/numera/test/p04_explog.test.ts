import { describe, expect, it } from "vitest";
import np, { DTypeError } from "../src/index.js";

const near = (a: unknown, b: number, tol = 1e-12) => {
  expect(Math.abs((a as number) - b)).toBeLessThanOrEqual(tol * Math.max(1, Math.abs(b)));
};

describe("P4-2 exp / log", () => {
  it("computes values", () => {
    expect(np.exp2([0, 3, -1]).toArray()).toEqual([1, 8, 0.5]);
    near(np.expm1(1e-10).toArray(), 1.00000000005e-10);
    expect(np.log2([8, 1]).toArray()).toEqual([3, 0]);
    expect(np.log10([1000]).toArray()).toEqual([3]);
    near(np.log1p(1e-10).toArray(), 9.9999999995e-11);
    // glibc's cbrt is not correctly rounded (aarch64 gives 3.0000000000000004, as NumPy does).
    near(np.cbrt(-8).toArray(), -2, 1e-15);
    near(np.cbrt(27).toArray(), 3, 1e-15);
    near(np.logaddexp(1, 2).toArray(), 2.313261687518223);
    expect(np.logaddexp2(1, 1).toArray()).toBe(2);
    expect(np.square([2, -3]).toArray()).toEqual([4, 9]);
    expect(np.reciprocal([4, 0.5]).toArray()).toEqual([0.25, 2]);
  });

  it("handles edge cases", () => {
    expect(np.log1p(-1).toArray()).toBe(-Infinity);
    expect(np.log2([-1]).toArray()).toEqual([NaN]);
    expect(np.logaddexp([Infinity, -Infinity, Infinity], [Infinity, -Infinity, -Infinity]).toArray()).toEqual([
      Infinity,
      -Infinity,
      Infinity,
    ]);
    expect(np.logaddexp(NaN, 1).toArray()).toBeNaN();
    expect(np.exp2([]).shape).toEqual([0]);
    expect(np.logaddexp.reduce([], {}).toArray()).toBe(-Infinity);
    expect(np.logaddexp2.reduce([1, 1, 2]).toArray()).toBe(3);
  });

  it("integer loops for square and reciprocal", () => {
    const s = np.square(np.array([200, 3], { dtype: "uint8" }));
    expect(s.dtype.name).toBe("uint8");
    expect(s.toArray()).toEqual([64, 9]);
    expect(np.square([true]).dtype.name).toBe("int8");
    expect(np.reciprocal([2, -1, 1]).toArray()).toEqual([0, -1, 1]);
    expect(np.reciprocal(np.array([0, 2], { dtype: "uint8" })).toArray()).toEqual([255, 0]);
    expect(np.reciprocal([true]).dtype.name).toBe("int8");
    expect(() => np.square([1], { dtype: "bool" })).toThrow(DTypeError);
  });

  it("dtypes and complex loops", () => {
    expect(np.log10(np.array([1], { dtype: "int16" })).dtype.name).toBe("float32");
    expect(np.cbrt(np.array([1], { dtype: "int8" })).dtype.name).toBe("float16");
    expect(np.logaddexp(np.array([1], { dtype: "int8" }), 2).dtype.name).toBe("float16");
    const z = np.log2(np.array([{ re: -1, im: 0 }])).toArray() as { re: number; im: number }[];
    near(z[0]!.re, 0);
    near(z[0]!.im, 4.532360141827194);
    const e = np.exp2(np.array([{ re: 1, im: 1 }])).toArray() as { re: number; im: number }[];
    near(e[0]!.re, 1.5384778027279442);
    near(e[0]!.im, 1.2779225526272695);
    const q = np.square(np.array([{ re: 2, im: 1 }], { dtype: "complex64" }));
    expect(q.dtype.name).toBe("complex64");
    expect(q.toArray()).toEqual([{ re: 3, im: 4 }]);
    expect(() => np.cbrt(np.zeros(1, { dtype: "complex128" }))).toThrow(DTypeError);
    expect(() => np.logaddexp(np.zeros(1, { dtype: "complex128" }), 1)).toThrow(DTypeError);
  });
});
