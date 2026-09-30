import { describe, expect, it } from "vitest";
import np, { DTypeError, NDArray, ShapeError, ValueError } from "../src/index.js";

describe("end-to-end native path (PLAN §92)", () => {
  it("creates a native array and reads it back", () => {
    const a = np.array([
      [1, 2],
      [3, 4],
    ]);
    expect(a).toBeInstanceOf(NDArray);
    expect(a.shape).toEqual([2, 2]);
    expect(a.ndim).toBe(2);
    expect(a.size).toBe(4);
    expect(a.dtype).toBe(np.int64);
    expect(a.itemSize).toBe(8);
    expect(a.strides).toEqual([16, 8]);
    expect(a.nbytes).toBe(32);
    expect(a.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
  });
});

describe("dtype inference (D-004)", () => {
  it.each([
    [[1, 2, 3], "int64"],
    [[1, 2.5], "float64"],
    [[true, false], "bool"],
    [[true, 2], "int64"],
    [[], "float64"],
    [[1n, 2n], "int64"],
    [5, "int64"],
    [[-0], "float64"],
    [[NaN], "float64"],
    [[2 ** 60], "float64"],
  ])("%s -> %s", (data, name) => {
    expect(np.array(data as never).dtype.name).toBe(name);
  });
});

describe("explicit dtypes", () => {
  it("float32", () => {
    const a = np.array([1, 2, 3], { dtype: np.float32 });
    expect(a.dtype).toBe(np.float32);
    expect(a.itemSize).toBe(4);
    expect(a.toTypedArray()).toEqual(new Float32Array([1, 2, 3]));
  });

  it("accepts names and aliases", () => {
    expect(np.array([1], { dtype: "uint16" }).dtype).toBe(np.uint16);
    expect(np.dtype("double")).toBe(np.float64);
    expect(() => np.dtype("float128")).toThrow(DTypeError);
  });

  it("rejects out-of-range integers and NaN for integer dtypes (D-009)", () => {
    expect(() => np.array([300], { dtype: "int8" })).toThrow(ValueError);
    expect(() => np.array([-1], { dtype: "uint8" })).toThrow(ValueError);
    expect(() => np.array([NaN], { dtype: "int64" })).toThrow(/NaN/);
    expect(() => np.array([Infinity], { dtype: "int32" })).toThrow(/infinity/);
    expect(np.array([1.7, -1.7], { dtype: "int8" }).toArray()).toEqual([1, -1]);
    expect(np.array([127, -128], { dtype: "int8" }).toArray()).toEqual([127, -128]);
  });

  it("int64/uint64 exact via bigint typed arrays (D-005)", () => {
    const a = np.array([2n ** 63n - 1n, -(2n ** 63n)], { dtype: "int64" });
    expect(a.toTypedArray()).toEqual(new BigInt64Array([2n ** 63n - 1n, -(2n ** 63n)]));
    const u = np.array([2n ** 64n - 1n], { dtype: "uint64" });
    expect(u.toTypedArray()).toEqual(new BigUint64Array([2n ** 64n - 1n]));
    expect(() => np.array([2n ** 64n], { dtype: "uint64" })).toThrow(ValueError);
  });

  it("float16 round-trip and raw bits", () => {
    const h = np.array([1.5, 65504, 65520, -0], { dtype: np.float16 });
    expect(h.toArray()).toEqual([1.5, 65504, Infinity, -0]);
    expect(h.toTypedArray()).toEqual(new Uint16Array([0x3e00, 0x7bff, 0x7c00, 0x8000]));
  });

  it("complex arrays allocate and convert elements to np.Complex (D-008, D-033)", () => {
    const z = np.zeros([2], { dtype: np.complex128 });
    expect(z.itemSize).toBe(16);
    expect(z.toTypedArray()).toEqual(new Float64Array(4));
    expect(z.toArray()).toEqual([np.complex(0, 0), np.complex(0, 0)]);
  });
});

describe("shape validation", () => {
  it("rejects ragged input", () => {
    expect(() => np.array([[1, 2], [3]])).toThrow(ValueError);
    expect(() => np.array([[1, 2], 3] as never)).toThrow(ValueError);
    expect(() => np.array([1, [2]] as never)).toThrow(ValueError);
  });
  it("rejects negative dimensions and non-integers", () => {
    expect(() => np.zeros([-1])).toThrow(ValueError);
    expect(() => np.zeros([1.5])).toThrow(ValueError);
    expect(() => np.zeros(Array(65).fill(1))).toThrow(ValueError);
  });
  it("supports empty and zero-d arrays", () => {
    const s = np.array(7);
    expect(s.shape).toEqual([]);
    expect(s.ndim).toBe(0);
    expect(s.size).toBe(1);
    expect(s.toArray()).toBe(7);
    expect(s.item()).toBe(7);
    const e = np.zeros([0, 3]);
    expect(e.size).toBe(0);
    expect(e.toArray()).toEqual([]);
    expect(np.zeros([2, 0]).toArray()).toEqual([[], []]);
  });
  it("reshape errors are ShapeErrors", () => {
    expect(() => np.zeros([12]).reshape([5, -1])).toThrow(ShapeError);
  });
});
