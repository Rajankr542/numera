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

const parts = (a: InstanceType<typeof np.NDArray>): number[][] =>
  (a.toArray() as Complex[]).map((z) => [z.re, z.im]);

describe("complex ufuncs (P1 step 2, D-033)", () => {
  const a = np.array([np.complex(1, 2), np.complex(-3, 0.5)]);
  const b = np.array([np.complex(2, -1), np.complex(0, 1)]);

  it("adds, subtracts, multiplies and divides", () => {
    expect(parts(np.add(a, b))).toEqual([[3, 1], [-3, 1.5]]);
    expect(parts(np.subtract(a, b))).toEqual([[-1, 3], [-3, -0.5]]);
    expect(parts(np.multiply(a, b))).toEqual([[4, 3], [-0.5, -3]]);
    expect(parts(np.divide(a, b))).toEqual([[0, 1], [0.5, 3]]);
  });

  it("divides by zero like NumPy (Smith's algorithm)", () => {
    const [[re, im]] = parts(np.divide([np.complex(-3, 0.5)], [np.complex(0, 0)]));
    expect(re).toBe(-Infinity);
    expect(im).toBe(Infinity);
  });

  it("raises to integer powers exactly and handles 0^b", () => {
    expect(parts(np.power(a, 2))).toEqual([[-3, 4], [8.75, -3]]);
    expect(parts(np.power([np.complex(0, 0)], [np.complex(2, 5)]))).toEqual([[0, 0]]);
    expect(parts(np.power([np.complex(0, 0)], [np.complex(0, 0)]))).toEqual([[1, 0]]);
    expect(parts(np.power([np.complex(0, 0)], [np.complex(-1, 0)])).flat().every(Number.isNaN)).toBe(true);
  });

  it("promotes real arrays and weak complex scalars (NEP 50)", () => {
    expect(np.add(np.array([1, 2], { dtype: "float32" }), np.complex(0, 1)).dtype).toBe(np.complex64);
    expect(np.add(np.array([1, 2], { dtype: "int8" }), np.complex(0, 1)).dtype).toBe(np.complex128);
    expect(np.add(np.zeros(1, { dtype: "complex64" }), np.complex(0, 1)).dtype).toBe(np.complex64);
    expect(np.add(np.zeros(1, { dtype: "complex64" }), 1.5).dtype).toBe(np.complex64);
    expect(np.add(np.zeros(1, { dtype: "complex64" }), np.zeros(1)).dtype).toBe(np.complex128);
    expect(parts(np.multiply([1, 2], { re: 0, im: 1 }))).toEqual([[0, 1], [0, 2]]);
  });

  it("abs and angle return the real dtype", () => {
    const m = np.abs(np.array([np.complex(3, 4), np.complex(Infinity, NaN)], { dtype: "complex64" }));
    expect(m.dtype).toBe(np.float32);
    expect(m.toArray()).toEqual([5, Infinity]);
    expect(np.angle([np.complex(0, 1)]).toArray()).toEqual([Math.PI / 2]);
    expect(np.angle([np.complex(-1, 0)], true).toArray()).toEqual([180]);
    expect(np.angle(np.zeros(1, { dtype: "complex64" })).dtype).toBe(np.float32);
  });

  it("sqrt, exp and log follow NumPy's special cases", () => {
    expect(parts(np.sqrt([np.complex(-1, 0)]))).toEqual([[0, 1]]);
    expect(parts(np.sqrt([np.complex(-4, -0)]))).toEqual([[0, -2]]);
    expect(parts(np.exp([np.complex(0, 0)]))).toEqual([[1, 0]]);
    expect(parts(np.log([np.complex(0, 0)]))).toEqual([[-Infinity, 0]]);
    expect(parts(np.log([np.complex(-1, 0)]))).toEqual([[0, Math.PI]]);
  });

  it("negates and conjugates", () => {
    expect(parts(np.negative(a))).toEqual([[-1, -2], [3, -0.5]]);
    expect(parts(np.conj(a))).toEqual([[1, -2], [-3, -0.5]]);
    expect(parts(a.conj())).toEqual(parts(np.conjugate(a)));
    expect(np.conj([1, 2]).toArray()).toEqual([1, 2]);
  });

  it("raises DTypeError for mod and floorDivide", () => {
    expect(() => np.mod(a, b)).toThrow(DTypeError);
    expect(() => np.floorDivide(a, 2)).toThrow(DTypeError);
  });
});

describe("real/imag views and predicates (D-033)", () => {
  it("real and imag are writable strided views", () => {
    const z = np.array([np.complex(1, 2), np.complex(3, 4)]);
    const re = z.real;
    const im = np.imag(z);
    expect(re.dtype).toBe(np.float64);
    expect(im.strides).toEqual([16]);
    expect(np.mayShareMemory(re, z)).toBe(true);
    im.set([1], -7);
    expect(parts(z)).toEqual([[1, 2], [3, -7]]);
    const t = np.array([[np.complex(1, 2), np.complex(3, 4)], [np.complex(5, 6), np.complex(7, 8)]]).T;
    expect(t.imag.toArray()).toEqual([[2, 6], [4, 8]]);
  });

  it("uses float32 components for complex64", () => {
    expect(np.real(np.zeros(2, { dtype: "complex64" })).dtype).toBe(np.float32);
  });

  it("real of real input is a view; imag is read-only zeros", () => {
    const x = np.array([1, 2, 3], { dtype: "int8" });
    const r = np.real(x);
    r.set([0], 9);
    expect(x.toArray()).toEqual([9, 2, 3]);
    const i = x.imag;
    expect(i.dtype).toBe(np.int8);
    expect(i.flags.writeable).toBe(false);
    expect(i.toArray()).toEqual([0, 0, 0]);
  });

  it("iscomplex/isreal test imag != 0 (NaN counts as nonzero)", () => {
    const z = np.array([np.complex(1, 0), np.complex(1, 1), np.complex(0, NaN)]);
    expect(np.iscomplex(z).toArray()).toEqual([false, true, true]);
    expect(np.isreal(z).toArray()).toEqual([true, false, false]);
    expect(np.iscomplex([1, 2]).toArray()).toEqual([false, false]);
    expect(np.isreal([1, 2]).toArray()).toEqual([true, true]);
  });

  it("iscomplexobj/isrealobj test the dtype", () => {
    expect(np.iscomplexobj([np.complex(1, 0)])).toBe(true);
    expect(np.iscomplexobj([1])).toBe(false);
    expect(np.isrealobj(np.zeros(1, { dtype: "complex64" }))).toBe(false);
    expect(np.isrealobj([true])).toBe(true);
  });
});
