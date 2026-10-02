import { describe, expect, it } from "vitest";
import np, { DTypeError, IndexError, NDArray, NotImplementedError, ValueError } from "../src/index.js";

const i32 = (v: number[]) => np.array(v, { dtype: "int32" });

describe("P3-2 NDArray methods (D-060)", () => {
  it("base follows NumPy identity semantics", () => {
    const x = np.array([0, 1, 2, 3]);
    expect(x.base).toBeNull();
    const v = x.get([1, 3]);
    expect(v.base).toBe(x);
    expect(v.get([0, 1]).base).toBe(x);
    expect(x.T.base).toBe(x);
    expect(x.copy().base).toBeNull();
    const r = np.zeros([6]).reshape(2, 3);
    expect(r.base?.shape).toEqual([6]);
    expect(x.view("uint8").base).toBe(x);
  });

  it("mT", () => {
    expect(np.ones([2, 3, 4]).mT.shape).toEqual([2, 4, 3]);
    expect(np.array([[1, 2], [3, 4]]).mT.toArray()).toEqual([[1, 3], [2, 4]]);
    expect(() => np.array([1, 2]).mT).toThrow(ValueError);
    expect(() => np.array(1).mT).toThrow(/ndim < 2/);
  });

  it("fill casts like np.array", () => {
    const z = np.zeros([3], { dtype: "int8" });
    z.fill(2.7);
    expect(z.toArray()).toEqual([2, 2, 2]);
    expect(() => z.fill(300)).toThrow(ValueError);
    expect(() => z.fill(NaN)).toThrow(ValueError);
    const f = np.zeros([2, 2]);
    f.fill(Infinity);
    expect(f.toArray()).toEqual([[Infinity, Infinity], [Infinity, Infinity]]);
    f.T.fill(np.array([5]));
    expect(f.toArray()).toEqual([[5, 5], [5, 5]]);
    const b = np.zeros([2], { dtype: "bool" });
    b.fill(true);
    expect(b.toArray()).toEqual([true, true]);
    const c = np.zeros([1], { dtype: "complex128" });
    c.fill(np.complex(1, 2));
    expect(c.item()).toEqual(np.complex(1, 2));
    const e = np.zeros([0]);
    e.fill(1);
    expect(e.size).toBe(0);
    const s = np.array(1.5);
    s.fill(3);
    expect(s.item()).toBe(3);
    expect(() => np.broadcastTo(np.array([1]), [3]).fill(1)).toThrow(/read-only/);
  });

  it("tolist", () => {
    expect(np.array([[1, 2], [3, 4]]).tolist()).toEqual([[1, 2], [3, 4]]);
    expect(np.array(5).tolist()).toBe(5);
    expect(np.zeros([2, 0]).tolist()).toEqual([[], []]);
  });

  it("tobytes orders", () => {
    const a = np.array([[0, 1, 2], [3, 4, 5]], { dtype: "int8" });
    expect(Array.from(a.tobytes())).toEqual([0, 1, 2, 3, 4, 5]);
    expect(Array.from(a.tobytes({ order: "F" }))).toEqual([0, 3, 1, 4, 2, 5]);
    expect(Array.from(a.T.tobytes({ order: "K" }))).toEqual([0, 3, 1, 4, 2, 5]);
    expect(Array.from(a.T.tobytes({ order: "A" }))).toEqual([0, 1, 2, 3, 4, 5]);
    expect(Array.from(i32([1]).tobytes())).toEqual([1, 0, 0, 0]);
    expect(np.zeros([0]).tobytes().length).toBe(0);
    expect(np.zeros([2, 3], { dtype: "float32" }).tobytes().length).toBe(24);
    expect(() => a.tobytes({ order: "X" as never })).toThrow(ValueError);
  });

  it("view(dtype)", () => {
    const a = i32([0, 1, 2]);
    const b = a.view("uint8");
    expect(b.shape).toEqual([12]);
    expect(b.toArray()).toEqual([0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0]);
    expect(np.mayShareMemory(a, b)).toBe(true);
    b.set(4, 9);
    expect(a.item(1)).toBe(9);
    expect(a.view().shape).toEqual([3]);
    expect(a.view().base).toBe(a);
    expect(np.array([1], { dtype: "float32" }).view("uint32").item()).toBe(0x3f800000);
    expect(b.view("int32").toArray()).toEqual([0, 9, 2]);
    expect(() => a.view("int64")).toThrow(/divisor/);
    expect(() => np.zeros([2, 3], { dtype: "int32" }).T.view("int16")).toThrow(/contiguous/);
    expect(() => np.array(1, { dtype: "int32" }).view("int16")).toThrow(/0d array/);
    expect(np.zeros([3, 0], { dtype: "int32" }).view("int16").shape).toEqual([3, 0]);
    expect(() => a.view("nope")).toThrow(DTypeError);
    const ro = np.broadcastTo(a, [2, 3]);
    expect(ro.view("float32").flags.writeable).toBe(false);
  });

  it("byteswap", () => {
    const a = np.array([1, 2], { dtype: "int16" });
    expect(a.byteswap().toArray()).toEqual([256, 512]);
    expect(a.toArray()).toEqual([1, 2]);
    expect(a.byteswap({ inplace: true })).toBe(a);
    expect(a.toArray()).toEqual([256, 512]);
    expect(np.array([true]).byteswap().toArray()).toEqual([true]);
    expect(np.array([1.5], { dtype: "float16" }).byteswap().view("uint16").item()).toBe(62);
    const z = np.array([np.complex(1, 2)], { dtype: "complex64" }).byteswap();
    expect(Array.from(z.view("uint8").toArray() as number[])).toEqual([63, 128, 0, 0, 64, 0, 0, 0]);
    const t = np.zeros([2, 3], { dtype: "int16" }).T.byteswap();
    expect(t.strides).toEqual([2, 6]);
    expect(() => np.broadcastTo(a, [2, 2]).byteswap({ inplace: true })).toThrow(/read-only/);
  });

  it("setflags", () => {
    const x = np.array([1, 2, 3]);
    const v = x.get([0, 2]);
    x.setflags({ write: false });
    expect(x.flags.writeable).toBe(false);
    expect(() => x.set(0, 1)).toThrow(/read-only/);
    expect(() => v.setflags({ write: true })).toThrow(/cannot set WRITEABLE flag/);
    v.setflags({ write: false });
    x.setflags({ write: true });
    v.setflags({ write: true });
    expect(v.flags.writeable).toBe(true);
    expect(() => x.setflags({ align: true })).toThrow(NotImplementedError);
    x.setflags({});
    expect(x.flags.writeable).toBe(true);
  });

  it("astype casting/copy", () => {
    const f = np.array([1.5, 2.5]);
    expect(f.astype("int32").toArray()).toEqual([1, 2]);
    expect(() => f.astype("int32", { casting: "safe" })).toThrow(DTypeError);
    expect(() => f.astype("float32", { casting: "same_kind" })).not.toThrow();
    expect(() => f.astype("float32", { casting: "safe" })).toThrow(/rule 'safe'/);
    expect(f.astype("float64", { copy: false })).toBe(f);
    expect(f.astype("float64")).not.toBe(f);
  });

  it("flat get", () => {
    const a = np.array([[0, 1, 2], [3, 4, 5]]).T; // [[0,3],[1,4],[2,5]]
    const fl = a.flat;
    expect(fl.base).toBe(a);
    expect(fl.length).toBe(6);
    expect(fl.get(1)).toBe(3);
    expect(fl.get(-1)).toBe(5);
    expect((fl.get([1, 4]) as NDArray).toArray()).toEqual([3, 1, 4]);
    expect(fl.get([null, null, -1]).toArray()).toEqual([5, 2, 4, 1, 3, 0]);
    expect(fl.get(np.array([[0, 1], [2, 5]])).toArray()).toEqual([[0, 3], [1, 5]]);
    expect(fl.get(np.array([true, false, false, false, false, true])).toArray()).toEqual([0, 5]);
    expect(fl.get([true, false, false, false, false, true]).toArray()).toEqual([0, 5]);
    expect(() => fl.get(6)).toThrow(IndexError);
    expect(() => fl.get(1.5)).toThrow(IndexError);
    expect(() => fl.get(np.array([true, false]))).toThrow(/boolean index did not match/);
    expect(() => fl.get(np.array([0.5]))).toThrow(IndexError);
    expect(fl.copy().toArray()).toEqual([0, 3, 1, 4, 2, 5]);
    expect(np.array(7).flat.get(0)).toBe(7);
  });

  it("flat set and iteration", () => {
    const g = np.array([[0, 1], [2, 3], [4, 5]]);
    g.flat.set([0, 4], [10, 20]);
    expect(g.toArray()).toEqual([[10, 20], [10, 20], [4, 5]]);
    g.flat = [7, 8];
    expect(g.toArray()).toEqual([[7, 8], [7, 8], [7, 8]]);
    g.flat.set([1, 3], 0);
    expect(g.flat.toArray()).toEqual([7, 0, 0, 8, 7, 8]);
    g.flat.set(np.array([true, false, false, false, false, true]), 9.9);
    expect(g.flat.toArray()).toEqual([9, 0, 0, 8, 7, 9]);
    g.flat.set(np.array([0, 0]), np.array([1, 2]));
    expect(g.item(0, 0)).toBe(2);
    expect(() => g.flat.set(1, NaN)).toThrow(ValueError);
    const it = g.flat;
    expect(it.index).toBe(0);
    it.next();
    it.next();
    expect(it.index).toBe(2);
    expect(it.coords).toEqual([1, 0]);
    expect([...np.array([[1, 2], [3, 4]]).T.flat]).toEqual([1, 3, 2, 4]);
    expect([...np.zeros([0, 2]).flat]).toEqual([]);
    const ro = np.broadcastTo(np.array([1]), [2]);
    expect(() => ro.flat.set(0, 1)).toThrow(/read-only/);
  });
});
