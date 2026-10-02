import { describe, expect, it } from "vitest";
import np, { BroadcastError, DTypeError } from "../src/index.js";

describe("P4-5 float bits / integer / multi-output", () => {
  it("copysign, nextafter, spacing, ldexp, signbit", () => {
    expect(np.copysign([3, 2], [-0.0, 1]).toArray()).toEqual([-3, 2]);
    expect(np.copysign(np.array([1], { dtype: "int8" }), -1).dtype.name).toBe("float16");
    expect(np.nextafter(1, 2).toArray()).toBe(1 + Number.EPSILON);
    expect(np.nextafter([0], [-1]).toArray()).toEqual([-5e-324]);
    expect(np.nextafter(np.array([1], { dtype: "float16" }), 2).toArray()).toEqual([1 + 1 / 1024]);
    expect(np.spacing([1, -1, 0]).toArray()).toEqual([Number.EPSILON, -Number.EPSILON, 5e-324]);
    expect(np.spacing([Infinity, NaN]).toArray()).toEqual([NaN, NaN]);
    expect(np.spacing(np.array([1], { dtype: "int8" })).dtype.name).toBe("float16");
    expect(np.ldexp([1.5, 1.5], [3, -1]).toArray()).toEqual([12, 0.75]);
    expect(np.ldexp(np.array([1, 2], { dtype: "int8" }), 3).dtype.name).toBe("float16");
    expect(() => np.ldexp(1.5, 2.5)).toThrow(DTypeError);
    expect(() => np.ldexp([1.5], np.array([2], { dtype: "uint64" }))).toThrow(DTypeError);
    const sb = np.signbit([-0.0, 1, -3, NaN]);
    expect(sb.dtype.name).toBe("bool");
    expect(sb.toArray()).toEqual([true, false, true, false]);
    expect(np.signbit(np.array([-1], { dtype: "int8" })).toArray()).toEqual([true]);
    expect(() => np.signbit(np.zeros(1, { dtype: "complex128" }))).toThrow(DTypeError);
  });

  it("gcd and lcm", () => {
    expect(np.gcd([12, -12, 0, 7], [18, 18, 0, 0]).toArray()).toEqual([6, 6, 0, 7]);
    expect(np.lcm([4, 0, -3], [6, 5, 7]).toArray()).toEqual([12, 0, 21]);
    expect(np.gcd.reduce([12, 18, 27]).toArray()).toBe(3);
    expect(np.gcd.reduce(np.array([], { dtype: "int64" })).toArray()).toBe(0);
    expect(() => np.gcd.reduce([])).toThrow(DTypeError); // [] is float64, as in NumPy
    expect(() => np.gcd([1.5], 2)).toThrow(DTypeError);
    expect(() => np.gcd([true], [true])).toThrow(DTypeError);
    expect(np.lcm(np.array([100], { dtype: "int8" }), 3).toArray()).toEqual([44]); // wraps like NumPy
  });

  it("divmod", () => {
    const [q, r] = np.divmod([7, -7], 2);
    expect(q.toArray()).toEqual([3, -4]);
    expect(r.toArray()).toEqual([1, 1]);
    const [fq, fr] = np.divmod([7.5, -7.5, 1.5], [2, 2, 0]);
    expect(fq.toArray()).toEqual([3, -4, Infinity]);
    expect(fr.toArray()).toEqual([1.5, 0.5, NaN]);
    expect(np.divmod([true], [true])[0].dtype.name).toBe("int8");
    expect(np.divmod(np.array([5], { dtype: "uint8" }), 0)[0].toArray()).toEqual([0]);
    expect(() => np.divmod(np.zeros(1, { dtype: "complex128" }), 1)).toThrow(DTypeError);
    const o1 = np.zeros(2);
    const o2 = np.zeros(2, { dtype: "int64" });
    const res = np.divmod([7, 8], 3, { out: [o1, o2] });
    expect(res[0]).toBe(o1);
    expect(res[1]).toBe(o2);
    expect(o1.toArray()).toEqual([2, 2]);
    expect(o2.toArray()).toEqual([1, 2]);
    expect(() => np.divmod([7, 8], 3, { out: [np.zeros(3), null] })).toThrow(BroadcastError);
    expect(np.divmod([[1], [2]], [3, 4])[0].shape).toEqual([2, 2]);
  });

  it("modf and frexp", () => {
    const [f, i] = np.modf([-2.5, Infinity, 3.25]);
    expect(f.toArray()).toEqual([-0.5, 0, 0.25]);
    expect(i.toArray()).toEqual([-2, Infinity, 3]);
    expect(np.modf(np.array([8], { dtype: "int8" }))[0].dtype.name).toBe("float16");
    const [m, e] = np.frexp([8, 0, -3, Infinity, NaN]);
    expect(m.toArray()).toEqual([0.5, 0, -0.75, Infinity, NaN]);
    expect(e.dtype.name).toBe("int32");
    expect(e.toArray()).toEqual([4, 0, 2, 0, 0]);
    expect(np.frexp(np.array(8.5))[1].toArray()).toBe(4);
    expect(() => np.frexp(np.zeros(1, { dtype: "complex128" }))).toThrow(DTypeError);
    expect(() => np.modf([1.5], { out: [np.zeros(1, { dtype: "int64" }), null] })).toThrow(DTypeError);
    expect(np.modf([]).map((x) => x.shape)).toEqual([[0], [0]]);
  });
});
