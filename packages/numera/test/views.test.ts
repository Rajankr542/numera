import { describe, expect, it } from "vitest";
import np, { IndexError, ValueError } from "../src/index.js";

const { asStrided } = np.lib.stride_tricks;

describe("memory model: views (PLAN §9, D-003)", () => {
  const a = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
  ]);

  it("column slice view shares memory with different offset/strides", () => {
    // b = a[:, 1:3]
    const b = asStrided(a, [3, 2], a.strides, 8);
    expect(b.toArray()).toEqual([
      [1, 2],
      [5, 6],
      [9, 10],
    ]);
    expect(np.mayShareMemory(a, b)).toBe(true);
    expect(b.strides).toEqual([32, 8]);
    expect(b.flags.cContiguous).toBe(false);
    expect(b.flags.ownData).toBe(false);
    expect(a.flags.ownData).toBe(true);
  });

  it("reshape of contiguous array is a view; copy is independent", () => {
    const r = a.reshape([2, 6]);
    expect(np.mayShareMemory(a, r)).toBe(true);
    expect(r.shape).toEqual([2, 6]);
    expect(a.reshape(-1).shape).toEqual([12]);
    expect(a.reshape(4, 3).shape).toEqual([4, 3]);
    const c = a.copy();
    expect(np.mayShareMemory(a, c)).toBe(false);
    expect(c.toArray()).toEqual(a.toArray());
  });

  it("transposed (F-order) view and its copy", () => {
    const t = asStrided(a, [4, 3], [8, 32], 0);
    expect(t.flags.fContiguous).toBe(true);
    expect(t.toArray()).toEqual([
      [0, 4, 8],
      [1, 5, 9],
      [2, 6, 10],
      [3, 7, 11],
    ]);
    const tc = t.copy();
    expect(tc.flags.cContiguous).toBe(true);
    expect(tc.strides).toEqual([24, 8]);
    // Reshape of a non-contiguous view copies (NumPy semantics for this layout).
    const flat = t.reshape([12]);
    expect(np.mayShareMemory(flat, a)).toBe(false);
    expect(flat.toArray()).toEqual([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]);
  });

  it("negative and zero strides", () => {
    const v = np.array([0, 1, 2, 3, 4]);
    expect(asStrided(v, [5], [-8], 32).toArray()).toEqual([4, 3, 2, 1, 0]);
    expect(asStrided(v, [2, 3], [0, 8], 8).toArray()).toEqual([
      [1, 2, 3],
      [1, 2, 3],
    ]);
  });

  it("rejects out-of-bounds views (safety over NumPy as_strided)", () => {
    const v = np.array([0, 1, 2, 3, 4]);
    expect(() => asStrided(v, [6], [8])).toThrow(ValueError);
    expect(() => asStrided(v, [5], [-8], 24)).toThrow(ValueError);
    expect(() => asStrided(v, [2], [8], -8)).toThrow(ValueError);
    expect(() => asStrided(v, [2], [8, 8])).toThrow(ValueError);
  });

  it("view keeps buffer alive after base goes out of scope", () => {
    const makeView = () => asStrided(np.array([10, 20, 30]), [2], [16], 0);
    const view = makeView();
    (globalThis as { gc?: () => void }).gc?.();
    expect(view.toArray()).toEqual([10, 30]);
  });
});

describe("item / indexing", () => {
  const a = np.array([
    [1, 2, 3],
    [4, 5, 6],
  ]);
  it("positive and negative indices", () => {
    expect(a.item(0, 0)).toBe(1);
    expect(a.item(1, 2)).toBe(6);
    expect(a.item(-1, -1)).toBe(6);
    expect(a.item(-2, 1)).toBe(2);
  });
  it("out of range and wrong arity raise IndexError", () => {
    expect(() => a.item(2, 0)).toThrow(IndexError);
    expect(() => a.item(0, -4)).toThrow(IndexError);
    expect(() => a.item(0)).toThrow(IndexError);
    expect(() => a.item()).toThrow(ValueError);
  });
});

describe("astype / typed arrays", () => {
  it("astype follows unsafe casting", () => {
    const f = np.array([1.5, -2.7, 300, 0]);
    expect(f.astype("int8").toArray()).toEqual([1, -2, 44, 0]);
    expect(f.astype("uint8").toArray()).toEqual([1, 254, 44, 0]);
    expect(f.astype("bool").toArray()).toEqual([true, true, true, false]);
    expect(np.array([200]).astype("int8").toArray()).toEqual([-56]);
  });
  it("fromTypedArray copies and infers dtype", () => {
    const src = new Float32Array([1, 2, 3, 4]);
    const a = np.fromTypedArray(src, [2, 2]);
    expect(a.dtype).toBe(np.float32);
    expect(a.toArray()).toEqual([
      [1, 2],
      [3, 4],
    ]);
    src[0] = 99;
    expect(a.item(0, 0)).toBe(1);
    expect(np.fromTypedArray(new Int16Array([5, 6])).shape).toEqual([2]);
    expect(np.fromTypedArray(Buffer.from([1, 2, 3])).dtype).toBe(np.uint8);
    expect(() => np.fromTypedArray(src, [3])).toThrow(np.ShapeError);
  });
  it("fromTypedArray honours byteOffset of subarrays", () => {
    const base = new Int32Array([1, 2, 3, 4, 5]);
    expect(np.fromTypedArray(base.subarray(2)).toArray()).toEqual([3, 4, 5]);
  });
  it("promoteTypes matches NumPy", () => {
    expect(np.promoteTypes("int8", "uint8")).toBe(np.int16);
    expect(np.promoteTypes(np.int64, np.uint64)).toBe(np.float64);
    expect(np.promoteTypes("float16", "int16")).toBe(np.float32);
  });
  it("canCast follows NumPy casting rules (D-045)", () => {
    expect(np.canCast("float64", "float32")).toBe(false); // default "safe"
    expect(np.canCast("float64", "float32", "same_kind")).toBe(true);
    expect(np.canCast(np.int8, np.int16)).toBe(true);
    expect(np.canCast("int64", "uint8", "same_kind")).toBe(false);
    expect(np.canCast("complex128", "float64", "unsafe")).toBe(true);
    expect(np.canCast("int32", "int32", "no")).toBe(true);
    expect(np.canCast("int32", "int64", "equiv")).toBe(false);
    expect(np.canCast("double", "complex")).toBe(true); // aliases
    // Arrays contribute their dtype; values are ignored (NumPy 2).
    expect(np.canCast(np.array([1000]), "int8")).toBe(false);
    expect(np.canCast(np.zeros([2], { dtype: "int8" }), np.int16)).toBe(true);
    expect(() => np.canCast(1 as never, "int8")).toThrow(np.DTypeError);
    expect(() => np.canCast(true as never, "int8")).toThrow(np.DTypeError);
    expect(() => np.canCast(1n as never, "int8")).toThrow(np.DTypeError);
    expect(() => np.canCast("int8", "int16", "bogus" as never)).toThrow(ValueError);
    expect(() => np.canCast("int8", "foo")).toThrow(np.DTypeError);
    expect(() => np.canCast("foo", "int8")).toThrow(np.DTypeError);
  });
});

describe("error translation (D-006)", () => {
  it("native errors become typed JS errors with codes", () => {
    try {
      np.array([300], { dtype: "int8" });
      expect.unreachable();
    } catch (e) {
      expect(e).toBeInstanceOf(ValueError);
      expect(e).toBeInstanceOf(np.NativpyError);
      expect((e as ValueError).code).toBe("NATIVPY_VALUE_ERROR");
      expect((e as Error).name).toBe("ValueError");
    }
  });
});
