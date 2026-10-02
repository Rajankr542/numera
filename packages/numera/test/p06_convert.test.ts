import { describe, expect, it } from "vitest";
import np, { BroadcastError, DTypeError, ValueError } from "../src/index.js";

describe("P6 copyto (D-094)", () => {
  it("arrays, broadcasting, casting, where", () => {
    const d = np.zeros([2, 3]);
    np.copyto(d, [1, 2, 3]);
    expect(d.toArray()).toEqual([[1, 2, 3], [1, 2, 3]]);
    np.copyto(d, 9, { where: [true, false, true] });
    expect(d.toArray()).toEqual([[9, 2, 9], [9, 2, 9]]);
    np.copyto(d, np.array([[[7]]]));
    expect(d.get(0, 0).item()).toBe(7);
    const i8 = np.zeros([3], { dtype: "int8" });
    expect(() => np.copyto(i8, np.array([1.5]))).toThrow(DTypeError);
    np.copyto(i8, np.array([1.5, 2.5, -3.5]), { casting: "unsafe" });
    expect(i8.toArray()).toEqual([1, 2, -3]);
    expect(() => np.copyto(np.zeros([3]), np.zeros([2]))).toThrow(/could not broadcast input array from shape \(2,\) into shape \(3,\)/);
    expect(() => np.copyto(np.zeros([3]), 1, { where: np.array([1, 0, 1]) })).toThrow(DTypeError);
    expect(() => np.copyto(np.zeros([3]), 1, { casting: "bogus" as never })).toThrow(/casting must be one of/);
    const ro = np.broadcastTo(np.zeros([1]), [3]);
    expect(() => np.copyto(ro, 1)).toThrow(/read-only/);
  });
  it("NEP 50 JS scalars", () => {
    const i8 = np.zeros([2], { dtype: "int8" });
    np.copyto(i8, 5);
    np.copyto(i8, true);
    expect(i8.toArray()).toEqual([1, 1]);
    expect(() => np.copyto(i8, 1.5)).toThrow(/Cannot cast scalar from dtype\('float64'\) to dtype\('int8'\)/);
    expect(() => np.copyto(i8, 300)).toThrow(/out of bounds for int8/);
    expect(() => np.copyto(np.zeros([1], { dtype: "uint8" }), -1)).toThrow(ValueError);
    expect(() => np.copyto(np.zeros([1], { dtype: "bool" }), 2)).toThrow(DTypeError);
    np.copyto(np.zeros([1], { dtype: "bool" }), 2, { casting: "unsafe" });
    const f = np.zeros([1], { dtype: "float32" });
    np.copyto(f, 1e300);
    expect(f.toArray()).toEqual([Infinity]);
    expect(() => np.copyto(f, { re: 1, im: 2 })).toThrow(DTypeError);
    const c = np.zeros([1], { dtype: "complex64" });
    np.copyto(c, { re: 1, im: 2 });
    expect(c.get(0).item()).toEqual({ re: 1, im: 2 });
    expect(() => np.copyto(np.zeros([1]), 1, { casting: "equiv" })).toThrow(DTypeError);
  });
});

describe("P6 conversions and metadata (D-094)", () => {
  it("asanyarray / asarrayChkfinite", () => {
    const a = np.zeros([2]);
    expect(np.asanyarray(a)).toBe(a);
    expect(np.asanyarray([1, 2]).dtype.name).toBe("int64");
    expect(np.asarrayChkfinite([1, 2], { dtype: "float32" }).dtype.name).toBe("float32");
    expect(() => np.asarrayChkfinite([1, NaN])).toThrow("array must not contain infs or NaNs");
    expect(() => np.asarrayChkfinite([-Infinity])).toThrow(ValueError);
    expect(() => np.asarrayChkfinite(np.array([{ re: 1, im: NaN }]))).toThrow(ValueError);
    expect(np.asarrayChkfinite(np.zeros([0])).shape).toEqual([0]);
  });
  it("require", () => {
    const x = np.zeros([2, 2]);
    expect(np.require(x)).toBe(x);
    expect(np.require(x, null, "O")).toBe(x);
    expect(np.require(x, null, ["C_CONTIGUOUS", "W"])).toBe(x);
    const r = np.require([1, 2], "float32", ["F", "W", "O"]);
    expect(r.dtype.name).toBe("float32");
    expect(r.flags.fContiguous).toBe(true);
    const v = x.T.get([null, 1]);
    expect(np.require(v, null, "C").flags.cContiguous).toBe(true);
    expect(np.require(np.zeros([2, 3]), null, "F").flags.fContiguous).toBe(true);
    const ro = np.broadcastTo(np.zeros([1]), [3]);
    expect(np.require(ro, null, "W").flags.writeable).toBe(true);
    expect(np.require(x.get(0), null, "O").flags.ownData).toBe(true);
    expect(() => np.require(x, null, "CF")).toThrow('Cannot specify both "C" and "F" order');
    expect(() => np.require(x, null, "X")).toThrow(ValueError);
  });
  it("broadcastArrays", () => {
    const [p, q] = np.broadcastArrays(np.arange(3), np.zeros([2, 1]));
    expect(p!.shape).toEqual([2, 3]);
    expect(q!.shape).toEqual([2, 3]);
    expect(p!.strides).toEqual([0, 8]);
    expect(p!.flags.writeable).toBe(true);
    const same = np.zeros([2]);
    expect(np.broadcastArrays(same)[0]).toBe(same);
    expect(np.broadcastArrays()).toEqual([]);
    expect(np.broadcastArrays(1, [1, 2])[0]!.shape).toEqual([2]);
    expect(() => np.broadcastArrays(np.zeros([2]), np.zeros([3]))).toThrow(/shape mismatch/);
    expect(BroadcastError).toBeDefined();
  });
  it("shape / size / ndim / isfortran", () => {
    expect(np.shape([[1, 2]])).toEqual([1, 2]);
    expect(np.shape(5)).toEqual([]);
    expect(np.size([[1, 2], [3, 4]])).toBe(4);
    expect(np.size([[1, 2, 3]], 1)).toBe(3);
    expect(np.size(np.zeros([2, 3, 4]), [0, 2])).toBe(8);
    expect(() => np.size([1], 1)).toThrow(/out of bounds/);
    expect(np.ndim(5)).toBe(0);
    expect(np.ndim(np.zeros([1, 1, 1]))).toBe(3);
    expect(np.isfortran(np.zeros([2, 3], { order: "F" }))).toBe(true);
    expect(np.isfortran(np.zeros([3], { order: "F" }))).toBe(false);
    expect(np.isfortran(np.zeros([2, 3]))).toBe(false);
  });
});
