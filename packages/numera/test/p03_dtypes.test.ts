import { describe, expect, it } from "vitest";
import np, { DTypeError, ValueError } from "../src/index.js";

const n = (d: { name: string }) => d.name;

describe("P3-4 finfo / iinfo (D-062)", () => {
  it("finfo values match NumPy", () => {
    const h = np.finfo("float16");
    expect([h.bits, h.eps, h.epsneg, h.max, h.min, h.tiny, h.smallestSubnormal]).toEqual([
      16, 0.0009765625, 0.00048828125, 65504, -65504, 2 ** -14, 2 ** -24,
    ]);
    expect([h.precision, h.iexp, h.nexp, h.nmant, h.machep, h.negep, h.minexp, h.maxexp]).toEqual([
      3, 5, 5, 10, -10, -11, -14, 16,
    ]);
    const f = np.finfo(np.float32);
    expect([f.eps, f.max, f.tiny, f.resolution]).toEqual([
      Math.fround(1.1920929e-7), Math.fround(3.4028235e38), Math.fround(1.1754944e-38), Math.fround(1e-6),
    ]);
    const d = np.finfo("float64");
    expect([d.eps, d.max, d.tiny, d.smallestNormal, d.smallestSubnormal, d.resolution, d.precision]).toEqual([
      Number.EPSILON, Number.MAX_VALUE, 2.2250738585072014e-308, 2.2250738585072014e-308, 5e-324, 1e-15, 15,
    ]);
    expect(n(np.finfo("complex64").dtype)).toBe("float32");
    expect(n(np.finfo(np.zeros([2])).dtype)).toBe("float64");
    expect(Object.isFrozen(d)).toBe(true);
    expect(() => np.finfo("int8")).toThrow(/not compatible with finfo/);
  });

  it("iinfo", () => {
    const i = np.iinfo("int8");
    expect([i.bits, i.min, i.max, i.kind, n(i.dtype)]).toEqual([8, -128, 127, "i", "int8"]);
    const u = np.iinfo("uint64");
    expect([u.minExact, u.maxExact]).toEqual([0n, 18446744073709551615n]);
    expect(np.iinfo("int64").minExact).toBe(-(2n ** 63n));
    expect(np.iinfo(np.array([1], { dtype: "uint16" })).max).toBe(65535);
    expect(() => np.iinfo("float32")).toThrow(/Invalid integer data type 'f'/);
    expect(() => np.iinfo("bool")).toThrow(ValueError);
  });
});

describe("P3-4 resultType / minScalarType (D-062)", () => {
  it("resultType: strong dtypes, weak JS scalars", () => {
    const R = np.resultType;
    expect(
      [
        R("int8", 300), R(3.5), R(3), R(true), R(np.complex(0, 1), "float32"), R(1.5, "int8"), R(true, "int8"),
        R(np.array(5, { dtype: "int8" }), "int16"), R("int8", "uint8"), R(3, 1.5), R("float16", np.complex(0, 1)),
        R("uint8", -1), R("bool", 3), R("bool", 1.5), R("int32", np.complex(0, 1)), R("int8", np.zeros([2], { dtype: "float32" })),
        R("int8", 3, 1.5), R("uint64", "int64"), R([1, 2]),
      ].map(n),
    ).toEqual([
      "int8", "float64", "int64", "bool", "complex64", "float64", "int8", "int16", "int16", "float64", "complex64",
      "uint8", "int64", "float64", "complex128", "float32", "float64", "float64", "int64",
    ]);
    expect(() => R()).toThrow(/at least one array or dtype/);
    expect(() => R("nope")).toThrow(DTypeError);
  });

  it("minScalarType", () => {
    const vals = [10, -10, 300, -300, 2 ** 40, 3.1, 1e50, np.complex(1, 2), np.complex(0, 1e50), true, 0, -1.5, NaN, Infinity,
      -Infinity, 65504.5, 65520.5, 1e-50, 2 ** 32 - 1, 2 ** 32, -(2 ** 31), -(2 ** 31) - 1, 255, 256, -128, -129, 2n ** 64n - 1n, 3.39e38, -3.39e38];
    expect(vals.map((v) => n(np.minScalarType(v)))).toEqual([
      "uint8", "int8", "uint16", "int16", "uint64", "float16", "float64", "complex64", "complex128", "bool", "uint8", "float16",
      "float16", "float16", "float16", "float32", "float32", "float16", "uint32", "uint64", "int32", "int64", "uint8", "uint16",
      "int8", "int16", "uint64", "float32", "float32",
    ]);
    expect(n(np.minScalarType(np.array(300)))).toBe("uint16");
    expect(n(np.minScalarType(np.array(7e4, { dtype: "float32" })))).toBe("float32");
    expect(n(np.minScalarType(np.array(3.0, { dtype: "float32" })))).toBe("float16");
    expect(n(np.minScalarType(np.array(np.complex(1, 1), { dtype: "complex64" })))).toBe("complex64");
    expect(n(np.minScalarType(np.array([300])))).toBe("int64");
    expect(() => np.minScalarType(2n ** 64n)).toThrow(ValueError);
  });
});

describe("P3-4 type hierarchy (D-062)", () => {
  it("issubdtype", () => {
    expect(np.issubdtype("float32", np.floating)).toBe(true);
    expect(np.issubdtype(np.int8, np.signedinteger)).toBe(true);
    expect(np.issubdtype("bool", np.integer)).toBe(false);
    expect(np.issubdtype("bool", np.generic)).toBe(true);
    expect(np.issubdtype("complex64", np.inexact)).toBe(true);
    expect(np.issubdtype(np.number, np.generic)).toBe(true);
    expect(np.issubdtype(np.floating, np.float64)).toBe(false);
    expect(np.issubdtype("int8", "int8")).toBe(true);
    expect(np.issubdtype("int8", "int16")).toBe(false);
    expect(np.issubdtype("uint8", np.number)).toBe(true);
    expect(np.issubdtype("uint8", np.signedinteger)).toBe(false);
    expect(np.issubdtype("float", np.float64)).toBe(true);
    expect(np.floating.parent).toBe(np.inexact);
    expect(() => np.issubdtype("float64", "nope")).toThrow(DTypeError);
  });

  it("isdtype", () => {
    expect(np.isdtype(np.float64, "real floating")).toBe(true);
    expect(np.isdtype(np.int8, ["bool", "unsigned integer"])).toBe(false);
    expect(np.isdtype(np.int8, np.int8)).toBe(true);
    expect(np.isdtype(np.bool, "numeric")).toBe(false);
    expect(np.isdtype(np.complex64, "numeric")).toBe(true);
    expect(np.isdtype(np.uint8, "integral")).toBe(true);
    expect(np.isdtype(np.float64, ["signed integer", np.float64])).toBe(true);
    expect(() => np.isdtype("float64" as never, "numeric")).toThrow(/must be a NumPy dtype/);
    expect(() => np.isdtype(np.float64, "nope")).toThrow(/not a known kind name/);
    expect(() => np.isdtype(np.float64, np.floating as never)).toThrow(DTypeError);
  });

  it("commonType", () => {
    const C = np.commonType;
    expect(
      [
        C(np.arange(3)), C(np.zeros([2], { dtype: "float16" })), C(np.zeros([2], { dtype: "float16" }), np.arange(2)),
        C(np.zeros([2], { dtype: "complex64" }), np.zeros([1])), C(np.zeros([1], { dtype: "float32" }), np.zeros([1], { dtype: "complex64" })),
        C(), C([1.5]),
      ].map(n),
    ).toEqual(["float64", "float16", "float64", "complex128", "complex64", "float16", "float64"]);
    expect(() => C(np.array([true]))).toThrow(/non-numeric/);
  });

  it("mintypecode", () => {
    const M = np.mintypecode;
    expect([M(["d", "f"]), M("dF"), M(["i"]), M([np.zeros([2], { dtype: "float32" })]), M("dF", "fF"), M("i", "GDFgdf", "x")]).toEqual([
      "d", "D", "d", "f", "F", "x",
    ]);
    expect(M([np.float32, np.complex64])).toBe("F");
  });
});
