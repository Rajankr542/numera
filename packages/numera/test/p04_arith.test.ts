import { describe, expect, it } from "vitest";
import np, { DTypeError } from "../src/index.js";

describe("P4-4 arithmetic", () => {
  it("fmod follows the dividend's sign", () => {
    expect(np.fmod([-7, 7], [3, -3]).toArray()).toEqual([-1, 1]);
    expect(np.fmod([-7.5, 7.5], [2, -2]).toArray()).toEqual([-1.5, 1.5]);
    expect(np.fmod(np.array([-128], { dtype: "int8" }), -1).toArray()).toEqual([0]);
    expect(np.fmod([1.5], 0).toArray()).toEqual([NaN]);
    expect(np.fmod([true], [true]).dtype.name).toBe("int8");
    expect(() => np.fmod(np.zeros(1, { dtype: "complex128" }), 1)).toThrow(DTypeError);
  });

  it("aliases of existing ufuncs", () => {
    expect(np.remainder).toBe(np.mod);
    expect(np.trueDivide).toBe(np.divide);
    expect(np.pow).toBe(np.power);
    expect(np.absolute).toBe(np.abs);
    expect(np.remainder([-7], 3).toArray()).toEqual([2]);
  });

  it("floatPower uses float64/complex128 loops", () => {
    const r = np.floatPower(np.array([2], { dtype: "int8" }), 3);
    expect(r.dtype.name).toBe("float64");
    expect(r.toArray()).toEqual([8]);
    expect(np.floatPower(np.array([2], { dtype: "float32" }), 2).dtype.name).toBe("float64");
    expect(np.floatPower([2], [-1]).toArray()).toEqual([0.5]);
    expect(np.floatPower(np.array([{ re: 1, im: 1 }], { dtype: "complex64" }), 2).dtype.name).toBe("complex128");
    expect(() => np.floatPower([2], 2, { dtype: "float32" })).toThrow(DTypeError);
  });

  it("sign and heaviside", () => {
    expect(np.sign([-2, -0, 3, NaN]).toArray()).toEqual([-1, 0, 1, NaN]);
    expect(np.sign(np.array([-5, 0, 5], { dtype: "int16" })).toArray()).toEqual([-1, 0, 1]);
    expect(np.sign(np.array([5], { dtype: "uint8" })).dtype.name).toBe("uint8");
    expect(() => np.sign([true])).toThrow(DTypeError);
    const z = np.sign(np.array([{ re: 3, im: 4 }, { re: 0, im: 0 }])).toArray() as { re: number; im: number }[];
    expect(z[0]!.re).toBeCloseTo(0.6, 15);
    expect(z[0]!.im).toBeCloseTo(0.8, 15);
    expect(z[1]).toEqual({ re: 0, im: 0 });
    expect(np.heaviside([-1, 0, 2, NaN], 0.5).toArray()).toEqual([0, 0.5, 1, NaN]);
    expect(np.heaviside(np.array([1], { dtype: "int8" }), 1).dtype.name).toBe("float16");
  });

  it("maximum / minimum propagate NaN; fmax / fmin ignore it", () => {
    expect(np.maximum([NaN, 1, 3], [1, NaN, 2]).toArray()).toEqual([NaN, NaN, 3]);
    expect(np.minimum([NaN, 1, 3], [1, NaN, 2]).toArray()).toEqual([NaN, NaN, 2]);
    expect(np.fmax([NaN, 1, NaN], [1, NaN, NaN]).toArray()).toEqual([1, 1, NaN]);
    expect(np.fmin([NaN, 1], [1, NaN]).toArray()).toEqual([1, 1]);
    expect(np.maximum([true, false], [false, false]).toArray()).toEqual([true, false]);
    expect(np.minimum(np.array([1], { dtype: "int8" }), np.array([1], { dtype: "uint8" })).dtype.name).toBe("int16");
    expect(np.maximum.reduce([3, 9, 2]).toArray()).toBe(9);
    expect(np.minimum.accumulate([3, 1, 2]).toArray()).toEqual([3, 1, 1]);
    expect(() => np.maximum.reduce([])).toThrow();
    expect(np.maximum([{ re: 1, im: 2 }], [{ re: 1, im: 3 }]).toArray()).toEqual([{ re: 1, im: 3 }]);
  });

  it("fabs is float-only", () => {
    expect(np.fabs([-1.5, 2]).toArray()).toEqual([1.5, 2]);
    expect(np.fabs(np.array([-1], { dtype: "int8" })).dtype.name).toBe("float16");
    expect(() => np.fabs(np.zeros(1, { dtype: "complex128" }))).toThrow(DTypeError);
  });

  it("clip and NDArray.clip / conjugate", () => {
    expect(np.clip([1, 5, 9], 2, 6).toArray()).toEqual([2, 5, 6]);
    expect(np.clip([1, 5, 9], null, 4).toArray()).toEqual([1, 4, 4]);
    expect(np.clip([1, 5, 9], 4).toArray()).toEqual([4, 5, 9]);
    expect(np.clip([1, 2], 3, 1).toArray()).toEqual([1, 1]);
    expect(np.clip([1, NaN], 0, 0.5).toArray()).toEqual([0.5, NaN]);
    expect(np.clip([1, 2], NaN, 0.5).toArray()).toEqual([NaN, NaN]);
    expect(np.clip([1, 2]).toArray()).toEqual([1, 2]);
    expect(np.clip([1, 5, 9], 1.5, 4).dtype.name).toBe("float64");
    expect(np.clip(np.array([1, 5], { dtype: "int8" }), 2, 4).dtype.name).toBe("int8");
    expect(np.clip([[1, 5], [7, 9]], [2, 6], 8).toArray()).toEqual([[2, 6], [7, 8]]);
    const out = np.zeros(3);
    expect(np.clip([1, 5, 9], 2, 6, { out })).toBe(out);
    expect(out.toArray()).toEqual([2, 5, 6]);
    expect(np.array([1, 5, 9]).clip(2, 6).toArray()).toEqual([2, 5, 6]);
    expect(np.array([1, 5, 9]).clip(null, 2).toArray()).toEqual([1, 2, 2]);
    expect(np.array([{ re: 1, im: 2 }]).conjugate().toArray()).toEqual([{ re: 1, im: -2 }]);
    expect(np.array([1, 2]).conjugate().toArray()).toEqual([1, 2]);
  });
});
