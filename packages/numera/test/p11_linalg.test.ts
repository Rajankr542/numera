import { afterAll, describe, expect, it } from "vitest";
import np, { NDArray } from "../src/index.js";

const flat = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v as number]);
const close = (a: NDArray, b: unknown, tol = 1e-10): void => {
  const g = flat(a.toArray());
  const e = flat(b);
  expect(g.length).toBe(e.length);
  g.forEach((x, i) => {
    const y = e[i]!;
    if (!Number.isFinite(y)) expect(x).toBe(y);
    else expect(Math.abs(x - y)).toBeLessThanOrEqual(tol * (1 + Math.abs(y)));
  });
};

for (const which of ["default", "fallback"] as const) {
  describe(`P11-1 (${which} backend)`, () => {
    np.linalg._setBackend(which);
    afterAll(() => np.linalg._setBackend("default"));
    it("cholesky", () => {
      np.linalg._setBackend(which);
      close(np.linalg.cholesky([[4, 2], [2, 3]]), [[2, 0], [1, Math.SQRT2]]);
      close(np.linalg.cholesky([[4, 2], [2, 3]], { upper: true }), [[2, 1], [0, Math.SQRT2]]);
      expect(np.linalg.cholesky(np.array([[4]], { dtype: "float32" })).dtype.name).toBe("float32");
      const c = np.linalg.cholesky(np.array([[4, np.complex(1, -1)], [np.complex(1, 1), 3]]));
      expect(c.dtype.name).toBe("complex128");
      const v = c.item(1, 0) as { re: number; im: number };
      expect(v.re).toBeCloseTo(0.5);
      expect(v.im).toBeCloseTo(0.5);
      expect(() => np.linalg.cholesky([[1, 2], [2, 1]])).toThrow(np.LinAlgError);
      expect(() => np.linalg.cholesky([1, 2])).toThrow(np.LinAlgError);
      expect(np.linalg.cholesky(np.zeros([0, 0])).shape).toEqual([0, 0]);
    });
    it("slogdet / svdvals / matrixPower", () => {
      np.linalg._setBackend(which);
      const r = np.linalg.slogdet([[1, 2], [3, 4]]);
      close(r.sign, -1);
      close(r.logabsdet, Math.log(2));
      const z = np.linalg.slogdet(np.zeros([2, 2]));
      close(z.sign, 0);
      close(z.logabsdet, -Infinity);
      close(np.linalg.svdvals([[3, 0], [0, 4]]), [4, 3]);
      expect(np.linalg.svdvals(np.eye(2, 2, { dtype: "complex64" })).dtype.name).toBe("float32");
      expect(np.linalg.matrixPower([[1, 1], [0, 1]], 3).toArray()).toEqual([[1, 3], [0, 1]]);
      close(np.linalg.matrixPower([[1, 1], [0, 1]], -2), [[1, -2], [0, 1]]);
      expect(np.linalg.matrixPower([[1, 1], [0, 1]], 0).dtype.name).toBe("int64");
      expect(() => np.linalg.matrixPower([[1]], 1.5)).toThrow(np.DTypeError);
      expect(() => np.linalg.matrixPower([[1, 2], [2, 4]], -1)).toThrow(np.LinAlgError);
    });
  });
}

for (const which of ["default", "fallback"] as const) {
  describe(`P11-2 (${which} backend)`, () => {
    afterAll(() => np.linalg._setBackend("default"));
    it("pinv", () => {
      np.linalg._setBackend(which);
      close(np.linalg.pinv([[1, 2], [3, 4]]), [[-2, 1], [1.5, -0.5]], 1e-9);
      close(np.linalg.pinv([[2, 1], [1, 2]], { hermitian: true }), [[2 / 3, -1 / 3], [-1 / 3, 2 / 3]], 1e-9);
      close(np.linalg.pinv([[1, 0], [0, 1e-20]], { rtol: null }), [[1, 0], [0, 0]]);
      expect(np.linalg.pinv(np.zeros([0, 3])).shape).toEqual([3, 0]);
      expect(() => np.linalg.pinv([[1]], { rcond: 1e-3, rtol: 1e-3 })).toThrow(np.ValueError);
    });
    it("matrixRank / cond", () => {
      np.linalg._setBackend(which);
      expect(np.linalg.matrixRank([[1, 2], [2, 4]]).item()).toBe(1);
      expect(np.linalg.matrixRank(np.eye(4)).item()).toBe(4);
      expect(np.linalg.matrixRank([[1, 0], [0, 1e-3]], { tol: 1e-2 }).item()).toBe(1);
      expect(np.linalg.matrixRank([0, 0]).item()).toBe(0);
      expect(np.linalg.matrixRank([[2, 1], [1, 2]], { hermitian: true }).item()).toBe(2);
      close(np.linalg.cond([[1, 0], [0, 2]]), 2);
      close(np.linalg.cond([[1, 2], [3, 4]], "fro"), 15);
      close(np.linalg.cond([[1, 2], [3, 4]], Infinity), 21);
      expect(np.linalg.cond([[1, 2], [2, 4]], 1).item()).toBe(Infinity);
      expect(() => np.linalg.cond(np.zeros([0, 0]))).toThrow(np.LinAlgError);
    });
  });
}

describe("P11-3", () => {
  const a = np.arange(6).reshape([2, 3]);
  it("vectorNorm / matrixNorm", () => {
    close(np.linalg.vectorNorm(a), Math.sqrt(55));
    close(np.linalg.vectorNorm(a, { axis: [1, 0], ord: 1 }), 15);
    close(np.linalg.vectorNorm(a, { axis: 1, ord: Infinity }), [2, 5]);
    expect(np.linalg.vectorNorm(a, { keepdims: true }).shape).toEqual([1, 1]);
    expect(np.linalg.vectorNorm(np.ones([2, 3, 4]), { axis: [0, 2], keepdims: true }).shape).toEqual([1, 3, 1]);
    close(np.linalg.vectorNorm(np.ones([2, 3, 4]), { axis: [0, 2] }), [Math.sqrt(8), Math.sqrt(8), Math.sqrt(8)]);
    close(np.linalg.matrixNorm(a), Math.sqrt(55));
    close(np.linalg.matrixNorm(a, { ord: 2, keepdims: true }), [[7.348469228349534]], 1e-9);
    expect(() => np.linalg.matrixNorm([1, 2])).toThrow(np.IndexError);
  });
  it("matrixTranspose / diagonal / trace / outer", () => {
    expect(np.linalg.matrixTranspose(a).shape).toEqual([3, 2]);
    expect(() => np.linalg.matrixTranspose([1, 2])).toThrow(np.ValueError);
    const s = np.arange(8).reshape([2, 2, 2]);
    expect(np.linalg.diagonal(s).toArray()).toEqual([[0, 3], [4, 7]]);
    expect(np.linalg.diagonal(a, { offset: 1 }).toArray()).toEqual([1, 5]);
    expect(np.linalg.trace(s).toArray()).toEqual([3, 11]);
    expect(np.linalg.outer([1, 2], [3, 4]).toArray()).toEqual([[3, 4], [6, 8]]);
    expect(() => np.linalg.outer([[1]], [1])).toThrow(np.ValueError);
  });
  it("tensorinv / tensorsolve", () => {
    const t = np.eye(24).reshape([4, 6, 8, 3]);
    const ti = np.linalg.tensorinv(t);
    expect(ti.shape).toEqual([8, 3, 4, 6]);
    const t1 = np.linalg.tensorinv(np.eye(6).reshape([6, 2, 3]), { ind: 1 });
    expect(t1.shape).toEqual([2, 3, 6]);
    expect(() => np.linalg.tensorinv(t, { ind: 0 })).toThrow(np.ValueError);
    const A = np.eye(6).reshape([6, 2, 3]);
    const x = np.linalg.tensorsolve(A, [1, 2, 3, 4, 5, 6]);
    expect(x.toArray()).toEqual([[1, 2, 3], [4, 5, 6]]);
    const B = np.eye(6).reshape([2, 3, 6]);
    expect(np.linalg.tensorsolve(B, [1, 2, 3, 4, 5, 6], { axes: [0, 1] }).shape).toEqual([2, 3]);
    expect(() => np.linalg.tensorsolve(np.ones([2, 3]), [1, 2])).toThrow(np.LinAlgError);
  });
});
