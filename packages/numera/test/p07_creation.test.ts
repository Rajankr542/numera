import { describe, expect, it } from "vitest";
import np, { DTypeError, NotImplementedError, ValueError } from "../src/index.js";

const list = (xs: { toArray(): unknown }[]) => xs.map((x) => x.toArray());

describe("P7-1 logspace / geomspace", () => {
  it("matches NumPy values and dtypes", () => {
    expect(np.logspace(0, 2, 3).toArray()).toEqual([1, 10, 100]);
    expect(np.logspace(0, 2, 2, { endpoint: false }).toArray()).toEqual([1, 10]);
    expect(np.logspace(0, 2.5, 3, { dtype: "int64" }).toArray()).toEqual([1, 17, 316]);
    expect(np.logspace(0, 3, 4, { base: 2, dtype: "float32" }).dtype).toBe(np.float32);
    expect(np.logspace(0, 1, 0).shape).toEqual([0]);
    expect(np.geomspace(1, 1000, 4).toArray()).toEqual([1, 10, 100, 1000]);
    expect(np.geomspace(-1000, -1, 4).toArray()).toEqual([-1000, -100, -10, -1]);
    const g = np.geomspace(-1, 1, 3).toArray() as number[];
    expect(g[0]).toBe(-1);
    expect(g[1]).toBeNaN();
    expect(g[2]).toBe(1);
    expect(() => np.geomspace(0, 1, 3)).toThrow(ValueError);
    expect(() => np.logspace(0, 1, -1)).toThrow(ValueError);
  });

  it("complex bounds give complex128", () => {
    const c = np.geomspace(np.complex(0, 1), np.complex(0, 1000), 4);
    expect(c.dtype).toBe(np.complex128);
    const last = c.item(3) as { re: number; im: number };
    expect(last.re).toBeCloseTo(0, 9);
    expect(last.im).toBeCloseTo(1000, 9);
  });
});

describe("P7-2 tri / tril / triu / diag / diagflat / vander", () => {
  it("tri family", () => {
    expect(np.tri(2).toArray()).toEqual([[1, 0], [1, 1]]);
    expect(np.tri(2, 3, { k: -1, dtype: "int32" }).dtype).toBe(np.int32);
    const m = [[1, 2, 3], [4, 5, 6], [7, 8, 9]];
    expect(np.tril(m).toArray()).toEqual([[1, 0, 0], [4, 5, 0], [7, 8, 9]]);
    expect(np.triu(m, -1).toArray()).toEqual([[1, 2, 3], [4, 5, 6], [0, 8, 9]]);
    // 1-D input is broadcast to a square matrix like NumPy.
    expect(np.tril([1, 2, 3]).toArray()).toEqual([[1, 0, 0], [1, 2, 0], [1, 2, 3]]);
    expect(np.triu(np.arange(8).reshape([2, 2, 2])).toArray()).toEqual([[[0, 1], [0, 3]], [[4, 5], [0, 7]]]);
    expect(() => np.tril(np.array(1))).toThrow(DTypeError);
  });

  it("diag is a read-only view of 2-D input and builds from 1-D input", () => {
    const a = np.arange(9).reshape([3, 3]);
    const d = np.diag(a, -1);
    expect(d.toArray()).toEqual([3, 7]);
    expect(d.flags.writeable).toBe(false);
    expect(np.mayShareMemory(a, d)).toBe(true);
    expect(np.diag(a, 5).shape).toEqual([0]);
    expect(np.diag([1, 2], -1).toArray()).toEqual([[0, 0, 0], [1, 0, 0], [0, 2, 0]]);
    expect(() => np.diag(np.zeros([2, 2, 2]))).toThrow(ValueError);
    expect(np.diagflat([[1, 2], [3, 4]], 1).shape).toEqual([5, 5]);
  });

  it("vander", () => {
    expect(np.vander([1, 2, 3]).toArray()).toEqual([[1, 1, 1], [4, 2, 1], [9, 3, 1]]);
    expect(np.vander([2, 3], 4, { increasing: true }).toArray()).toEqual([[1, 2, 4, 8], [1, 3, 9, 27]]);
    expect(np.vander(np.array([1, 2], { dtype: "int8" })).dtype).toBe(np.int64);
    expect(np.vander([1.5]).dtype).toBe(np.float64);
    expect(np.vander([1, 2], 0).shape).toEqual([2, 0]);
    expect(() => np.vander([[1]])).toThrow(ValueError);
  });
});

describe("P7-3 index helpers and fillDiagonal", () => {
  it("tri/diag/mask indices", () => {
    expect(list(np.trilIndices(3, 0, 2))).toEqual([[0, 1, 1, 2, 2], [0, 0, 1, 0, 1]]);
    expect(list(np.triuIndices(2))).toEqual([[0, 0, 1], [0, 1, 1]]);
    expect(list(np.triuIndicesFrom(np.zeros([3, 3]), 1))).toEqual([[0, 0, 1], [1, 2, 2]]);
    expect(() => np.trilIndicesFrom(np.zeros([3]))).toThrow(ValueError);
    expect(list(np.diagIndices(2, 3))).toEqual([[0, 1], [0, 1], [0, 1]]);
    expect(() => np.diagIndicesFrom(np.zeros([2, 3]))).toThrow(ValueError);
    expect(list(np.maskIndices(3, np.tril, -1))).toEqual([[1, 2, 2], [0, 0, 1]]);
    expect(np.trilIndices(2)[0]!.dtype).toBe(np.int64);
  });

  it("fillDiagonal (wrap, cyclic values, nd)", () => {
    const a = np.zeros([7, 3]);
    np.fillDiagonal(a, 5, { wrap: true });
    expect(a.toArray()).toEqual([[5, 0, 0], [0, 5, 0], [0, 0, 5], [0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5]]);
    const b = np.zeros([5, 3]);
    np.fillDiagonal(b, 1);
    expect(b.toArray()).toEqual([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0]]);
    const c = np.zeros([2, 2, 2], { dtype: "int32" });
    np.fillDiagonal(c, [4, 9]);
    expect(c.toArray()).toEqual([[[4, 0], [0, 0]], [[0, 0], [0, 9]]]);
    const w = np.zeros([2, 4]);
    np.fillDiagonal(w, [1, 2, 3]);
    expect(w.toArray()).toEqual([[1, 0, 0, 0], [0, 2, 0, 0]]);
    expect(() => np.fillDiagonal(np.zeros([3]), 1)).toThrow(ValueError);
    expect(() => np.fillDiagonal(np.zeros([2, 3, 2]), 1)).toThrow(ValueError);
    expect(() => np.fillDiagonal(np.broadcastTo(np.zeros([1]), [2, 2]), 1)).toThrow();
  });
});

describe("P7-4 from* constructors and astype", () => {
  it("fromfunction / fromiter", () => {
    expect(np.fromfunction((i: { toArray(): unknown }) => i, [3]).toArray()).toEqual([0, 1, 2]);
    expect(np.fromfunction((i, j) => np.multiply(i, j), [2, 3], { dtype: "int32" }).toArray()).toEqual([[0, 0, 0], [0, 1, 2]]);
    function* gen() { for (let i = 0; i < 4; i++) yield i * i; }
    expect(np.fromiter(gen(), "int32").toArray()).toEqual([0, 1, 4, 9]);
    expect(np.fromiter(gen(), "float64", 2).toArray()).toEqual([0, 1]);
    expect(np.fromiter([], "int64").shape).toEqual([0]);
    expect(() => np.fromiter([1, 2], "int64", 3)).toThrow("iterator too short: Expected 3 but iterator had only 2 items.");
  });

  it("frombuffer", () => {
    const f = new Float64Array([1.5, 2.5, 3.5]);
    expect(np.frombuffer(f).toArray()).toEqual([1.5, 2.5, 3.5]);
    expect(np.frombuffer(f.buffer, { count: 1, offset: 8 }).toArray()).toEqual([2.5]);
    expect(np.frombuffer(new DataView(f.buffer, 8), { dtype: "float64" }).toArray()).toEqual([2.5, 3.5]);
    const sub = new Uint8Array(f.buffer, 8, 8);
    expect(np.frombuffer(sub).toArray()).toEqual([2.5]);
    const copied = np.frombuffer(f);
    f[0] = 99;
    expect(copied.item(0)).toBe(1.5);
    expect(() => np.frombuffer(new Uint8Array(3), { dtype: "int16" })).toThrow("buffer size must be a multiple of element size");
    expect(() => np.frombuffer(new Uint8Array(4), { offset: 5 })).toThrow("offset must be non-negative and no greater than buffer length (4)");
    expect(() => np.frombuffer(new Uint8Array(4), { dtype: "uint8", count: 5 })).toThrow("buffer is smaller than requested size");
  });

  it("fromstring text mode", () => {
    expect(np.fromstring("1 2  3\n4", { sep: " " }).toArray()).toEqual([1, 2, 3, 4]);
    expect(np.fromstring("1,2,3", { sep: ",", dtype: "int32", count: 2 }).toArray()).toEqual([1, 2]);
    expect(np.fromstring("-1.5e3 , inf", { sep: "," }).toArray()).toEqual([-1500, Infinity]);
    expect(np.fromstring("", { sep: "," }).shape).toEqual([0]);
    expect(() => np.fromstring("1 2", {})).toThrow("The binary mode of fromstring is removed, use frombuffer instead");
    expect(() => np.fromstring("1,2", { sep: ",", count: 5 })).toThrow("string is smaller than requested size");
  });

  it("np.astype", () => {
    const x = np.array([1.7, -2.2]);
    expect(np.astype(x, "int32").toArray()).toEqual([1, -2]);
    expect(np.astype(x, "float64", { copy: false })).toBe(x);
    expect(np.astype(x, "float64")).not.toBe(x);
    expect(() => np.astype([1] as never, "int32")).toThrow(DTypeError);
  });
});

describe("P7-5 indices / meshgrid / mgrid / ogrid / ix_", () => {
  it("indices", () => {
    const g = np.indices([2, 2], { dtype: "int32" });
    expect(g.shape).toEqual([2, 2, 2]);
    expect(g.dtype).toBe(np.int32);
    expect(list(np.indices([1, 2], { sparse: true }))).toEqual([[[0]], [[0, 1]]]);
    expect(np.indices([]).shape).toEqual([0]);
  });

  it("meshgrid", () => {
    const [x, y, z] = np.meshgrid([1, 2], [3, 4, 5], [6]);
    expect(x!.shape).toEqual([3, 2, 1]);
    expect(y!.toArray()).toEqual([[[3], [3]], [[4], [4]], [[5], [5]]]);
    expect(z!.shape).toEqual([3, 2, 1]);
    const [a, b] = np.meshgrid([1, 2], [3, 4, 5], { indexing: "ij" });
    expect(a!.shape).toEqual([2, 3]);
    expect(b!.toArray()).toEqual([[3, 4, 5], [3, 4, 5]]);
    expect(list(np.meshgrid([1, 2]))).toEqual([[1, 2]]);
    expect(np.meshgrid()).toEqual([]);
    const v = np.meshgrid([1, 2], [3], { copy: false });
    expect(v[0]!.flags.writeable).toBe(false);
    expect(np.meshgrid([1, 2], [3])[0]!.flags.writeable).toBe(true);
    expect(() => np.meshgrid([1], { indexing: "ji" as never })).toThrow(ValueError);
  });

  it("mgrid / ogrid", () => {
    expect(np.mgrid([0, 4, 2], [1, 3]).toArray()).toEqual([[[0, 0], [2, 2]], [[1, 2], [1, 2]]]);
    expect(np.mgrid([0, 4]).dtype).toBe(np.int64);
    expect(np.mgrid([0, 1, 0.5], [0, 2]).dtype).toBe(np.float64);
    expect(np.mgrid([0, 1, np.complex(0, 3)], [0, 2]).toArray()).toEqual([[[0, 0], [0.5, 0.5], [1, 1]], [[0, 1], [0, 1], [0, 1]]]);
    expect(np.mgrid([null, 3]).toArray()).toEqual([0, 1, 2]);
    expect(np.mgrid([3, 0]).shape).toEqual([0]);
    expect(np.mgrid([0, 5, np.complex(0, 1)]).toArray()).toEqual([0]);
    expect(() => np.mgrid([0, 3, 0])).toThrow(ValueError);
    expect(() => np.mgrid()).toThrow(ValueError);
    const og = np.ogrid([0, 3], [0, 1, np.complex(0, 2)]);
    expect(og.map((a) => a.shape)).toEqual([[3, 1], [1, 2]]);
    expect(og[1]!.toArray()).toEqual([[0, 1]]);
    expect(np.ogrid([0, 3]).toArray()).toEqual([0, 1, 2]);
  });

  it("ix_", () => {
    const a = np.arange(12).reshape([3, 4]);
    const [r, c] = np.ix_([0, 2], [1, 3]);
    expect(a.get(r!, c!).toArray()).toEqual([[1, 3], [9, 11]]);
    expect(list(np.ix_([false, true], [1]))).toEqual([[[1]], [[1]]]);
    expect(np.ix_([])[0]!.dtype).toBe(np.int64);
    expect(() => np.ix_([[1]])).toThrow("Cross index must be 1 dimensional");
  });
});

describe("P7-6 r_ / c_ / s_ / indexExp", () => {
  it("r_ slices, scalars and directives", () => {
    expect(np.r_("1:4", 0, 4, [7, 8]).toArray()).toEqual([1, 2, 3, 0, 4, 7, 8]);
    expect(np.r_(np.array([1, 2]), np.array([3])).toArray()).toEqual([1, 2, 3]);
    expect(np.r_("-1", [[1], [2]], [[3], [4]]).toArray()).toEqual([[1, 3], [2, 4]]);
    expect(np.r_("0,2", [1, 2], [3, 4]).toArray()).toEqual([[1, 2], [3, 4]]);
    expect(np.r_("0,2,0", [1, 2], [3, 4]).shape).toEqual([4, 1]);
    expect(np.r_("1,2,0", [1, 2], [3, 4]).toArray()).toEqual([[1, 3], [2, 4]]);
    expect(np.r_("0,2", "1:3").shape).toEqual([1, 2]);
    expect(np.r_(":3").toArray()).toEqual([0, 1, 2]);
    expect(np.r_("0:1:3j", 5).toArray()).toEqual([0, 0.5, 1, 5]);
    expect(() => np.r_()).toThrow("need at least one array to concatenate");
    expect(() => np.r_("r", [1])).toThrow(NotImplementedError);
    expect(() => np.r_([1], "0,2")).toThrow(ValueError);
    expect(() => np.r_("1:")).toThrow(ValueError);
  });

  it("r_ dtype follows NEP 50 weak scalars", () => {
    const i8 = np.array([1, 2], { dtype: "int8" });
    expect(np.r_(i8, 3).dtype).toBe(np.int8);
    expect(np.r_(i8, 3.5).dtype).toBe(np.float64);
    expect(() => np.r_(i8, 300)).toThrow(ValueError);
    expect(np.r_(np.array([1], { dtype: "float32" }), 2.5).dtype).toBe(np.float32);
    expect(np.r_(1, 2.5).dtype).toBe(np.float64);
    expect(np.r_(true, false).dtype).toBe(np.bool);
    expect(np.r_(np.array([1], { dtype: "float32" }), np.complex(1, 1)).dtype).toBe(np.complex64);
  });

  it("c_", () => {
    expect(np.c_([1, 2, 3], [4, 5, 6]).toArray()).toEqual([[1, 4], [2, 5], [3, 6]]);
    expect(np.c_([[1, 2, 3]], 0, 0, [[4, 5, 6]]).toArray()).toEqual([[1, 2, 3, 0, 0, 4, 5, 6]]);
  });

  it("s_ / indexExp", () => {
    expect(np.s_(1)).toBe(1);
    expect(np.s_("1:3")).toEqual([1, 3, null]);
    expect(np.s_(":")).toEqual([null, null, null]);
    expect(np.s_("...", "::-1")).toEqual([np.ellipsis, [null, null, -1]]);
    expect(np.indexExp(1)).toEqual([1]);
    const a = np.arange(10);
    expect(a.get(np.s_("::3")).toArray()).toEqual([0, 3, 6, 9]);
    expect(a.get(...np.indexExp("2:4")).toArray()).toEqual([2, 3]);
    expect(() => np.s_("abc")).toThrow(ValueError);
    expect(() => np.s_("1:2:3j")).toThrow(ValueError);
  });
});
