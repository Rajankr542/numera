import { describe, expect, it } from "vitest";
import np, { Complex, DTypeError } from "../src/index.js";

describe("complex conversion (D-033)", () => {
  it("round-trips Complex and {re, im} inputs", () => {
    const a = np.array([np.complex(1, 2), { re: -0, im: -0 }, 3]);
    expect(a.dtype).toBe(np.complex128);
    const v = a.toArray() as Complex[];
    expect(v.every((x) => x instanceof Complex)).toBe(true);
    expect(v.map((x) => [x.re, x.im])).toEqual([[1, 2], [-0, -0], [3, 0]]);
    expect(Object.isFrozen(v[0])).toBe(true);
  });

  it("widens complex64 to double like float32", () => {
    const z = np.array([np.complex(0.1, 0.2)], { dtype: "complex64" }).item() as Complex;
    expect(z.re).toBe(Math.fround(0.1));
    expect(z.im).toBe(Math.fround(0.2));
  });

  it("stores real, bool and bigint values with im = 0", () => {
    const a = np.array([true, 2n, 1.5], { dtype: "complex128" });
    expect((a.toArray() as Complex[]).map(String)).toEqual(["(1+0j)", "(2+0j)", "(1.5+0j)"]);
    expect(np.array([1, 2], { dtype: "complex64" }).item(1)).toEqual(np.complex(2, 0));
  });

  it("rejects complex values in real arrays instead of dropping im", () => {
    expect(() => np.array([np.complex(1, 1)], { dtype: "float64" })).toThrow(DTypeError);
    expect(() => np.array([np.complex(1, 0)], { dtype: "int32" })).toThrow(DTypeError);
    expect(() => np.array([{ re: "x" } as never])).toThrow(np.ValueError);
  });

  it("supports full, item and nested shapes", () => {
    const f = np.full([2, 1], np.complex(0, -1));
    expect(f.dtype).toBe(np.complex128);
    expect(f.item(1, 0)).toEqual(np.complex(0, -1));
    expect(np.array([[np.complex(1, 1)], [2]]).shape).toEqual([2, 1]);
  });

  it("formats like NumPy's repr", () => {
    expect(String(np.complex(1, -2))).toBe("(1-2j)");
    expect(String(np.complex(0, 3))).toBe("3j");
    expect(String(np.complex(NaN, Infinity))).toBe("(nan+infj)");
    expect(String(np.complex(-0, 1))).toBe("(-0+1j)");
    expect(String(np.complex(0, -0))).toBe("-0j");
    expect(String(np.complex(1, -0))).toBe("(1-0j)");
  });
});
