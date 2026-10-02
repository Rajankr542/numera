import { describe, expect, it } from "vitest";
import np, { NDArray, ValueError } from "../src/index.js";

// ---- P16-A tests (D-191) ----

describe("P16-A constants (D-191)", () => {
  it("math constants", () => {
    expect(np.pi).toBe(Math.PI);
    expect(np.e).toBe(Math.E);
    expect(np.inf).toBe(Infinity);
    expect(np.nan).toBeNaN();
    expect(np.euler_gamma).toBeCloseTo(0.5772156649015329, 15);
    expect(np.PINF).toBe(Infinity);
    expect(np.NINF).toBe(-Infinity);
    expect(1 / np.PZERO).toBe(Infinity);
    expect(1 / np.NZERO).toBe(-Infinity);
  });

  it("bool sentinels", () => {
    expect(np.True_).toBe(true);
    expect(np.False_).toBe(false);
  });
});

describe("P16-A dtype aliases (D-191)", () => {
  it("float dtype aliases", () => {
    expect(np.double.name).toBe("float64");
    expect(np.single.name).toBe("float32");
    expect(np.half.name).toBe("float16");
    expect(np.longdouble.name).toBe("float64");
  });

  it("integer dtype aliases", () => {
    expect(np.int_.name).toBe("int64");
    expect(np.intc.name).toBe("int32");
    expect(np.intp.name).toBe("int64");
    expect(np.long.name).toBe("int64");
    expect(np.byte.name).toBe("int8");
    expect(np.short.name).toBe("int16");
  });

  it("unsigned integer dtype aliases", () => {
    expect(np.uint.name).toBe("uint64");
    expect(np.uintc.name).toBe("uint32");
    expect(np.uintp.name).toBe("uint64");
    expect(np.ulong.name).toBe("uint64");
    expect(np.ubyte.name).toBe("uint8");
    expect(np.ushort.name).toBe("uint16");
  });

  it("complex dtype aliases", () => {
    expect(np.cdouble.name).toBe("complex128");
    expect(np.csingle.name).toBe("complex64");
    expect(np.clongdouble.name).toBe("complex128");
  });

  it("dtype aliases are the same DType singletons", () => {
    // Aliases should be the same objects as the canonical names.
    expect(np.double).toBe(np.float64);
    expect(np.single).toBe(np.float32);
    expect(np.half).toBe(np.float16);
    expect(np.cdouble).toBe(np.complex128);
    expect(np.csingle).toBe(np.complex64);
  });
});

describe("P16-A np.vectorize (D-191)", () => {
  it("element-wise over 1-D arrays", () => {
    const add = np.vectorize((a: unknown, b: unknown) => (a as number) + (b as number));
    const result = add(np.array([1, 2, 3]), np.array([10, 20, 30]));
    expect(result instanceof NDArray).toBe(true);
    expect(result.toArray()).toEqual([11, 22, 33]);
  });

  it("broadcasting inputs", () => {
    const vf = np.vectorize((a: unknown, b: unknown) => (a as number) * (b as number));
    const result = vf(np.array([[1, 2], [3, 4]]), np.array([10, 20]));
    expect(result.toArray()).toEqual([[10, 40], [30, 80]]);
  });

  it("pyfunc property", () => {
    const fn = (x: unknown) => (x as number) * 2;
    const vf = np.vectorize(fn);
    expect(vf.pyfunc).toBe(fn);
    expect(vf.pyfunc(5)).toBe(10);
  });

  it("otypes forces output dtype", () => {
    const vf = np.vectorize((x: unknown) => (x as number) + 0.5, { otypes: ["float32"] });
    const result = vf(np.array([1, 2, 3]));
    expect(result.dtype.name).toBe("float32");
  });

  it("scalar input (0-d array)", () => {
    const vf = np.vectorize((x: unknown) => (x as number) ** 2);
    const result = vf(np.array(3));
    expect(result.item()).toBe(9);
  });

  it("nested array input", () => {
    const vf = np.vectorize((x: unknown) => (x as number) + 1);
    expect(vf([1, 2, 3]).toArray()).toEqual([2, 3, 4]);
  });

  it("inferred bool output dtype", () => {
    const vf = np.vectorize((x: unknown) => (x as number) > 0);
    const result = vf(np.array([-1, 0, 1]));
    expect(result.dtype.name).toBe("bool");
  });

  it("throws on non-null signature", () => {
    expect(() => np.vectorize(() => 0, { signature: "(n)->(n)" })).toThrow(ValueError);
  });
});

describe("P16-A np.shares_memory (D-191)", () => {
  it("shares memory between view and original", () => {
    const a = np.array([1, 2, 3, 4, 5]);
    const b = a.slice([[1, 3]]);
    expect(np.shares_memory(a, b)).toBe(true);
  });

  it("independent copies do not share memory", () => {
    const a = np.array([1, 2, 3]);
    const b = np.array([1, 2, 3]);
    expect(np.shares_memory(a, b)).toBe(false);
  });

  it("accepts maxWork option (ignored)", () => {
    const a = np.array([1, 2, 3]);
    const b = a.slice([[0, 2]]);
    expect(np.shares_memory(a, b, { maxWork: 100 })).toBe(true);
  });

  it("2-D arrays", () => {
    const a = np.zeros([4, 4]);
    const b = a.slice([[1, 3], [1, 3]]);
    expect(np.shares_memory(a, b)).toBe(true);
    expect(np.shares_memory(a, np.zeros([4, 4]))).toBe(false);
  });
});

describe("P16-A cumsum / cumprod (D-191)", () => {
  it("np.cumsum flat", () => {
    expect(np.cumsum([1, 2, 3, 4]).toArray()).toEqual([1, 3, 6, 10]);
  });

  it("np.cumprod flat", () => {
    expect(np.cumprod([1, 2, 3, 4]).toArray()).toEqual([1, 2, 6, 24]);
  });

  it("np.cumsum along axis 0", () => {
    expect(np.cumsum([[1, 2], [3, 4]], 0).toArray()).toEqual([[1, 2], [4, 6]]);
  });

  it("np.cumsum along axis 1", () => {
    expect(np.cumsum([[1, 2, 3], [4, 5, 6]], 1).toArray()).toEqual([[1, 3, 6], [4, 9, 15]]);
  });

  it("np.cumprod along axis 1", () => {
    expect(np.cumprod([[1, 2, 3], [4, 5, 6]], 1).toArray()).toEqual([[1, 2, 6], [4, 20, 120]]);
  });

  it("np.cumsum null axis = flatten", () => {
    expect(np.cumsum([[1, 2], [3, 4]], null).toArray()).toEqual([1, 3, 6, 10]);
  });

  it("NDArray.cumsum method flat", () => {
    const a = np.array([1, 2, 3, 4]);
    expect(a.cumsum().toArray()).toEqual([1, 3, 6, 10]);
  });

  it("NDArray.cumprod method flat", () => {
    const a = np.array([1, 2, 3, 4]);
    expect(a.cumprod().toArray()).toEqual([1, 2, 6, 24]);
  });

  it("NDArray.cumsum along axis", () => {
    const a = np.array([[1, 2], [3, 4]]);
    expect(a.cumsum(0).toArray()).toEqual([[1, 2], [4, 6]]);
    expect(a.cumsum(1).toArray()).toEqual([[1, 3], [3, 7]]);
  });

  it("NDArray.cumprod along axis", () => {
    const a = np.array([[1, 2], [3, 4]]);
    expect(a.cumprod(0).toArray()).toEqual([[1, 2], [3, 8]]);
    expect(a.cumprod(1).toArray()).toEqual([[1, 2], [3, 12]]);
  });

  it("cumsum with dtype option", () => {
    const a = np.array([1, 2, 3], { dtype: "int32" });
    const r = np.cumsum(a, null, { dtype: "float64" });
    expect(r.dtype.name).toBe("float64");
    expect(r.toArray()).toEqual([1, 3, 6]);
  });

  it("cumsum on empty array", () => {
    const r = np.cumsum(np.array([]));
    expect(r.size).toBe(0);
  });
});
