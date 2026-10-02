import { describe, expect, it } from "vitest";
import np, { NDArray, ValueError } from "../src/index.js";

const L = (a: unknown): unknown => (a as NDArray).toArray();
const close = (a: unknown, b: number[], digits = 12) => {
  const v = (a as NDArray).toArray() as number[];
  expect(v.length).toBe(b.length);
  v.forEach((x, i) => expect(x).toBeCloseTo(b[i]!, digits));
};

describe("P14-7 polynomial functions (D-173)", () => {
  it("roots and poly", () => {
    expect(L(np.roots([1, -3, 2]))).toEqual([2, 1]);
    expect(np.roots([1, 0, 1]).dtype.name).toBe("complex128");
    expect(L(np.roots([0, 0, 1, 2, 0, 0]))).toEqual([-2, 0, 0]);
    expect(np.roots([5]).shape).toEqual([0]);
    expect(np.roots([0, 0]).shape).toEqual([0]);
    expect(L(np.poly([1, 2]))).toEqual([1, -3, 2]);
    const pc = np.poly([{ re: 0, im: 1 }, { re: 0, im: -1 }]) as NDArray;
    expect(pc.dtype.name).toBe("float64");
    expect(L(pc)).toEqual([1, 0, 1]);
    expect(np.poly([])).toBe(1);
    close(np.poly([[1, 2], [3, 4]]), [1, -5, -2]);
    expect((np.poly(np.array([1, 2]).astype("float32")) as NDArray).dtype.name).toBe("float32");
    expect(() => np.poly([[1, 2, 3]])).toThrow(ValueError);
  });
  it("polyval, polyder, polyint", () => {
    expect(np.polyval([1, 2], 3).item()).toBe(5);
    expect(L(np.polyval([1, 2], [1, 2]))).toEqual([3, 4]);
    expect(np.polyval(np.array([1, 2]).astype("int8"), np.array([3]).astype("int16")).dtype.name).toBe("int16");
    expect(L(np.polyder([1, 2, 3]))).toEqual([2, 2]);
    expect(np.polyder([1, 2, 3], 3).shape).toEqual([0]);
    expect(L(np.polyint([1, 2, 3]))).toEqual([1 / 3, 1, 3, 0]);
    expect(L(np.polyint([1, 2], 2, [1, 2]))).toEqual([1 / 6, 1, 1, 2]);
    expect(L(np.polyint([3], 0))).toEqual([3]);
    expect(() => np.polyder([1], -1)).toThrow(ValueError);
    expect(() => np.polyint([1], 3, [1, 2])).toThrow(ValueError);
  });
  it("polyadd, polysub, polymul, polydiv", () => {
    expect(L(np.polyadd([1, 2], [3, 4, 5]))).toEqual([3, 5, 7]);
    expect(L(np.polysub([1], [1.5, 2]))).toEqual([-1.5, -1]);
    expect(L(np.polymul([1, 2], [3, 4]))).toEqual([3, 10, 8]);
    const [q, r] = np.polydiv([1, -3, 2], [1, -1]);
    expect(L(q)).toEqual([1, -2]);
    expect(L(r)).toEqual([0]);
    const [q2, r2] = np.polydiv([1, 2], [1, 2, 3]);
    expect([L(q2), L(r2)]).toEqual([[0], [1, 2]]);
  });
  it("polyfit (plain, full, cov, weights, 2-D y)", () => {
    close(np.polyfit([0, 1, 2], [1, 3, 5], 1), [2, 1]);
    const full = np.polyfit([0, 1, 2, 3], [1, 3, 5, 8], 2, { full: true }) as (NDArray | number)[];
    close(full[0], [0.25, 1.55, 1.05]);
    close(full[1], [0.05]);
    expect(full[2]).toBe(3);
    expect(full[4]).toBeCloseTo(8.881784197001252e-16, 25);
    const [c, V] = np.polyfit([0, 1, 2, 3], [1, 3, 5, 8], 1, { cov: true }) as NDArray[];
    close(c, [2.3, 0.8]);
    close(V!.ravel(), [0.03, -0.045, -0.045, 0.105]);
    close(np.polyfit([0, 1, 2], [1, 3, 5], 1, { w: [1, 2, 3] }), [2, 1]);
    expect((np.polyfit([0, 1, 2], [[1, 0], [3, 1], [5, 2]], 1) as NDArray).shape).toEqual([2, 2]);
    expect(() => np.polyfit([0, 1], [1, 2], -1)).toThrow(ValueError);
    expect(() => np.polyfit([0, 1], [1, 2, 3], 1)).toThrow(TypeError);
  });
});

describe("P14-7 poly1d (D-173)", () => {
  const p = new np.poly1d([1, -2, 3]);
  it("properties, indexing, iteration", () => {
    expect(p.order).toBe(2);
    expect(p.length).toBe(2);
    expect([p.get(0), p.get(2), p.get(5)]).toEqual([3, 1, 0]);
    expect([...p]).toEqual([1, -2, 3]);
    expect(L(new np.poly1d([0, 0, 1, 2]).coeffs)).toEqual([1, 2]);
    expect(L(new np.poly1d([0, 0]).coeffs)).toEqual([0]);
    expect(L(new np.poly1d([1, 2], true).coeffs)).toEqual([1, -3, 2]);
    expect(p.roots.dtype.name).toBe("complex128");
    const s = new np.poly1d([1, 2]);
    s.set(3, 7);
    expect(L(s.coeffs)).toEqual([7, 0, 1, 2]);
    expect(() => new np.poly1d([[1]])).toThrow(ValueError);
  });
  it("arithmetic methods", () => {
    expect(p.call(2).item()).toBe(3);
    expect(L(p.add(1).coeffs)).toEqual([1, -2, 4]);
    expect(L(new np.poly1d(2).sub(p).coeffs)).toEqual([-1, 2, -1]);
    expect(L(p.mul(2).coeffs)).toEqual([2, -4, 6]);
    expect(L(p.div(2).coeffs)).toEqual([0.5, -1, 1.5]);
    const [q, r] = p.div(new np.poly1d([1, 1]));
    expect([L(q.coeffs), L(r.coeffs)]).toEqual([[1, -3], [6]]);
    expect(L(p.pow(2).coeffs)).toEqual([1, -4, 10, -12, 9]);
    expect(L(p.neg().coeffs)).toEqual([-1, 2, -3]);
    expect(p.equals(new np.poly1d([1, -2, 3]))).toBe(true);
    expect(p.notEquals(new np.poly1d([1]))).toBe(true);
    expect(L(p.deriv().coeffs)).toEqual([2, -2]);
    expect(np.polyadd(p, [1]) instanceof np.poly1d).toBe(true);
    expect(String(np.polyval([1, 1], new np.poly1d([1, 0])))).toBe(" \n1 x + 1");
    expect(() => p.pow(-1)).toThrow(ValueError);
  });
  it("toString is NumPy's str()", () => {
    expect(String(p)).toBe("   2\n1 x - 2 x + 3");
    expect(String(new np.poly1d([1.5, 0, -2.25, 1e-5], { variable: "z" }))).toBe("     3\n1.5 z - 2.25 z + 1e-05");
    expect(String(new np.poly1d([0]))).toBe(" \n0");
    expect(String(new np.poly1d([{ re: 0, im: 1 }, { re: 2, im: 3 }]))).toBe(" \n1j x + (2 + 3j)");
    const long = String(new np.poly1d(np.arange(1, 40)));
    expect(long.split("\n").length).toBe(10);
  });
});
