import { describe, expect, it } from "vitest";
import np from "../src/index.js";

describe("P11-6 einsum", () => {
  const a = np.arange(6).reshape([2, 3]);
  const b = np.arange(12).reshape([3, 4]);
  it("explicit / implicit / ellipsis / repeated labels", () => {
    expect(np.einsum("ij,jk->ik", a, b).toArray()).toEqual(np.matmul(a, b).toArray());
    expect(np.einsum("ij,jk", a, b).toArray()).toEqual(np.matmul(a, b).toArray());
    expect(np.einsum("ij", a).toArray()).toEqual(a.toArray());
    expect(np.einsum("ji", a).toArray()).toEqual([[0, 3], [1, 4], [2, 5]]);
    expect(np.einsum("ii", np.arange(9).reshape([3, 3])).item()).toBe(12);
    expect(np.einsum("ii->i", np.arange(9).reshape([3, 3])).toArray()).toEqual([0, 4, 8]);
    expect(np.einsum("...ii->...i", np.arange(18).reshape([2, 3, 3])).toArray()).toEqual([[0, 4, 8], [9, 13, 17]]);
    expect(np.einsum("i,i", [1, 2, 3], [4, 5, 6]).item()).toBe(32);
    expect(np.einsum("i,j->ij", [1, 2], [3, 4]).toArray()).toEqual([[3, 4], [6, 8]]);
    expect(np.einsum("...j,j", a, [1, 1, 1]).toArray()).toEqual([3, 12]);
    expect(np.einsum(",i->i", 2, [1, 2]).toArray()).toEqual([2, 4]);
    expect(np.einsum("ij->", a).item()).toBe(15);
  });
  it("sublist form, dtypes, optimize", () => {
    expect(np.einsum(a, [0, 1], b, [1, 2], [0, 2]).toArray()).toEqual(np.matmul(a, b).toArray());
    expect(np.einsum(a, [np.ellipsis, 1], [1, 1, 1], [1]).toArray()).toEqual([3, 12]);
    expect(np.einsum("i->", np.array([100, 100], { dtype: "int8" })).item()).toBe(-56);
    expect(np.einsum("i,i", [true, false], [true, true]).item()).toBe(true);
    const z = np.einsum("i,i", np.array([np.complex(0, 1)]), np.array([np.complex(0, 1)])).item() as { re: number };
    expect(z.re).toBe(-1);
    for (const optimize of [true, "greedy", "optimal", ["optimal", 10]] as const) {
      expect(np.einsum("ij,jk,kl->il", a, b, np.ones([4, 2]), { optimize }).toArray()).toEqual([
        [98, 98],
        [296, 296],
      ]);
    }
    const [path] = np.einsumPath("ij,jk,kl->il", np.ones([2, 2]), np.ones([2, 5]), np.ones([5, 2]));
    expect(path).toEqual(["einsum_path", [1, 2], [0, 1]]);
    expect(np.einsum("ij,jk,kl->il", np.ones([2, 2]), np.ones([2, 5]), np.ones([5, 2]), { optimize: path }).toArray())
      .toEqual([[10, 10], [10, 10]]);
  });
  it("einsumPath report", () => {
    const [, rep] = np.einsumPath("ij,jk,kl->il", np.ones([2, 2]), np.ones([2, 5]), np.ones([5, 2]));
    expect(rep.split("\n")[0]).toBe("  Complete contraction:  ij,jk,kl->il");
    expect(rep).toContain("  Optimized FLOP count:  5.700e+01");
    expect(rep.split("\n").at(-1)).toBe("   3                   jl,ij->il                                   il->il");
  });
  it("errors", () => {
    expect(() => np.einsum("ijk", a)).toThrow(/too many subscripts for operand 0/);
    expect(() => np.einsum("i", a)).toThrow(/more dimensions than subscripts/);
    expect(() => np.einsum("ij,jk", a)).toThrow(/fewer operands/);
    expect(() => np.einsum("ij", a, a)).toThrow(/more operands/);
    expect(() => np.einsum("ij,jk", a, np.ones([2, 2]))).toThrow(np.ValueError);
    expect(() => np.einsum("ij->ik", a)).toThrow(/never appeared in an input/);
    expect(() => np.einsum("ij->ii", a)).toThrow(/multiple times/);
    expect(() => np.einsum("i$", a)).toThrow(/invalid subscript '\$'/);
    expect(() => np.einsum("ii", a)).toThrow(/collapsing index 'i' don't match \(2 != 3\)/);
    expect(() => np.einsum("ij,jk", a, b, { optimize: ["einsum_path", [0]] })).toThrow(/more operands has to be contracted/);
    expect(() => np.einsum()).toThrow(/No input operands/);
  });
  it("fallback backend", () => {
    np.linalg._setBackend("fallback");
    try {
      expect(np.einsum("ij,jk->ik", a, b).toArray()).toEqual(np.matmul(a, b).toArray());
    } finally {
      np.linalg._setBackend("default");
    }
  });
});
