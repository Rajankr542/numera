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
