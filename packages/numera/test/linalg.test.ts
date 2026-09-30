import { describe, expect, it } from "vitest";
import np, { NDArray } from "../src/index.js";

const close = (a: NDArray, b: number[] | number, tol = 1e-10): void => {
  const got = a.toArray();
  const flat = (v: unknown): number[] => (Array.isArray(v) ? v.flatMap(flat) : [v as number]);
  const g = flat(got);
  const e = flat(b);
  expect(g.length).toBe(e.length);
  g.forEach((x, i) => expect(Math.abs(x - e[i]!)).toBeLessThanOrEqual(tol * (1 + Math.abs(e[i]!))));
};

describe("linalg: products", () => {
  it("matmul/dot/inner/outer accept nested arrays", () => {
    const a = [[1, 2], [3, 4]];
    expect(np.matmul(a, [[5], [6]]).toArray()).toEqual([[17], [39]]);
    expect(np.dot([1, 2, 3], [4, 5, 6]).item()).toBe(32);
    expect(np.inner([1, 2], [3, 4]).item()).toBe(11);
    expect(np.outer([1, 2], [3, 4]).toArray()).toEqual([[3, 4], [6, 8]]);
    const f32 = np.array(a, { dtype: "float32" });
    expect(np.matmul(f32, f32).dtype.name).toBe("float32");
    expect(np.matmul(f32, a).dtype.name).toBe("float64"); // float32 x int64 -> float64
  });
  it("maps shape errors to ShapeError", () => {
    expect(() => np.matmul([[1, 2]], [[1, 2]])).toThrow(np.ShapeError);
    expect(() => np.matmul(np.array(1), [1])).toThrow(np.ValueError);
  });
});

describe("linalg: decompositions", () => {
  const a = [[4, 1], [2, 3]];
  it("det/inv/solve", () => {
    close(np.linalg.det(a), 10);
    close(np.linalg.inv(a), [0.3, -0.1, -0.2, 0.4]);
    close(np.linalg.solve(a, [1, 2]), [0.1, 0.6]);
    expect(() => np.linalg.inv([[1, 2], [2, 4]])).toThrow(np.LinAlgError);
    expect(() => np.linalg.det(np.eye(2, 2, { dtype: "float16" }))).toThrow(np.DTypeError);
    expect(np.LinAlgError).toBe(np.linalg.LinAlgError);
  });
  it("eig returns complex, eigh ascending real", () => {
    const { eigenvalues } = np.linalg.eig(a);
    expect(eigenvalues.dtype.name).toBe("complex128");
    const re = Array.from(eigenvalues.toTypedArray() as Float64Array).filter((_, i) => i % 2 === 0);
    expect(re.sort()).toEqual([2, 5].map((x) => expect.closeTo(x, 10)) as unknown as number[]);
    close(np.linalg.eigvalsh([[2, 1], [1, 2]]), [1, 3]);
  });
  it("svd/qr/lstsq/norm result objects", () => {
    const s = np.linalg.svd(a, { computeUV: false });
    expect(s.U).toBeNull();
    expect(s.Vh).toBeNull();
    expect(s.S.shape).toEqual([2]);
    const f = np.linalg.svd([[1, 2, 3]]);
    expect(f.U!.shape).toEqual([1, 1]);
    expect(f.Vh!.shape).toEqual([3, 3]);
    const q = np.linalg.qr(a, "r");
    expect(q.Q).toBeNull();
    expect(q.R.shape).toEqual([2, 2]);
    expect(() => np.linalg.qr(a, "bogus" as "r")).toThrow(np.ValueError);
    const l = np.linalg.lstsq([[1, 0], [0, 1], [1, 1]], [1, 2, 3]);
    expect(l.rank).toBe(2);
    close(l.x, [1, 2]);
    close(np.linalg.norm([3, 4]), 5);
    close(np.linalg.norm(a, { ord: "fro" }), Math.sqrt(30));
    close(np.linalg.norm(a, { ord: Infinity }), 5);
    expect(np.linalg.norm(a, { axis: 1, keepdims: true }).shape).toEqual([2, 1]);
    expect(() => np.linalg.norm([1, 2], { ord: "nuc" })).toThrow(np.ValueError);
  });
  it("reports a backend and can force the fallback", () => {
    expect(["accelerate", "fallback"]).toContain(np.linalg.backend());
    np.linalg._setBackend("fallback");
    try {
      expect(np.linalg.backend()).toBe("fallback");
      close(np.linalg.det(a), 10);
    } finally {
      np.linalg._setBackend("default");
    }
  });
});
