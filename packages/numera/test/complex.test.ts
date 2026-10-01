import { afterAll, beforeAll, describe, expect, it } from "vitest";
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
    const [re, im] = parts(np.divide([np.complex(-3, 0.5)], [np.complex(0, 0)]))[0]!;
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

describe("complex reductions (P1 step 3, D-034)", () => {
  // [[1+2j, 3-1j], [0.5j, -2]]; expected values from NumPy 2.5.3.
  const m = np.array([
    [np.complex(1, 2), np.complex(3, -1)],
    [np.complex(0, 0.5), np.complex(-2, 0)],
  ]);
  const z = (a: InstanceType<typeof np.NDArray>): number[] => {
    const v = a.item() as Complex;
    return [v.re, v.im];
  };

  it("sum and prod, over all axes and per axis", () => {
    expect(z(np.sum(m))).toEqual([2, 1.5]);
    expect(parts(np.sum(m, { axis: 0 }))).toEqual([[1, 2.5], [1, -1]]);
    expect(z(np.prod(m))).toEqual([5, -5]);
    expect(parts(np.prod(m, { axis: 1 }))).toEqual([[5, 5], [-0, -1]]);
    expect(z(m.sum({ initial: 10 }))).toEqual([12, 1.5]);
    expect(np.sum(m).dtype).toBe(np.complex128);
  });

  it("empty input, inf products and real-to-complex dtype", () => {
    const e = np.zeros([0], { dtype: "complex128" });
    expect(z(np.sum(e))).toEqual([0, 0]);
    expect(z(np.prod(e))).toEqual([1, 0]);
    const p = z(np.prod(np.array([np.complex(Infinity, 0), np.complex(Infinity, 0)])));
    expect(p.every(Number.isNaN)).toBe(true); // NumPy: (inf+0j)**2 -> nan+nanj
    const s = np.sum(np.array([1, 2], { dtype: "float32" }), { dtype: "complex64" });
    expect(s.dtype).toBe(np.complex64);
    expect(z(s)).toEqual([3, 0]);
  });

  it("mean keeps the complex dtype", () => {
    expect(z(np.mean(m))).toEqual([0.5, 0.375]);
    const r = np.mean(m, { axis: 1, keepdims: true });
    expect(r.shape).toEqual([2, 1]);
    expect(parts(r.reshape([2]))).toEqual([[2, 0.5], [-1, 0.25]]);
    expect(np.mean(m.astype("complex64")).dtype).toBe(np.complex64);
    expect(z(np.mean(np.zeros([0], { dtype: "complex128" }))).every(Number.isNaN)).toBe(true);
  });

  it("min/max/argmin/argmax order by real part, then imag", () => {
    expect(z(np.max(m))).toEqual([3, -1]);
    expect(z(m.min())).toEqual([-2, 0]);
    expect(np.argmax(m).item()).toBe(1);
    expect(m.argmin().item()).toBe(3);
    expect(np.argmax(m, { axis: 0 }).toArray()).toEqual([0, 0]);
    expect(parts(np.max(m, { axis: 1 }))).toEqual([[3, -1], [0, 0.5]]);
    expect(z(np.max(np.array([np.complex(1, 1)]), { initial: 5 }))).toEqual([5, 0]);
  });

  it("min/max propagate the first NaN in either part", () => {
    const a = np.array([np.complex(1, 1), np.complex(0, NaN), np.complex(5, 0)]);
    const r = z(np.max(a));
    expect(r[0]).toBe(0);
    expect(Number.isNaN(r[1])).toBe(true);
    expect(np.argmax(a).item()).toBe(1);
    expect(np.argmin(a).item()).toBe(1);
  });

  it("var/std are real and support ddof and axis", () => {
    const v = np.var(m);
    expect(v.dtype).toBe(np.float64);
    expect(v.item()).toBe(4.421875);
    expect(m.std().item()).toBe(2.1028254801575903);
    expect(np.std(m, { ddof: 1 }).item()).toBe(2.4281337140555777);
    expect(np.var(m, { axis: 0 }).toArray()).toEqual([0.8125, 6.5]);
    expect(np.std(m.astype("complex64")).dtype).toBe(np.float32);
  });

  it("raises typed errors for unsupported or invalid cases", () => {
    const e = np.zeros([0], { dtype: "complex128" });
    expect(() => np.max(e)).toThrow(np.ValueError);
    expect(() => np.argmin(e)).toThrow(np.ValueError);
    expect(() => np.std(m, { dtype: "complex64" })).toThrow(np.NotImplementedError);
  });
});

describe("complex matmul/dot/inner/outer (P1 step 4, D-035/D-036)", () => {
  // Expected values from NumPy 2.5.3. No conjugation anywhere (that is vdot).
  const a = np.array([np.complex(1, 2), np.complex(3, -1)]);
  const b = np.array([np.complex(2, -1), np.complex(0, 1)]);
  const M = np.array([
    [np.complex(1, 1), np.complex(2, 0)],
    [np.complex(0, 0), np.complex(1, -1)],
  ]);
  const flat = (x: InstanceType<typeof np.NDArray>): number[][] => parts(x.flatten());
  const z = (x: InstanceType<typeof np.NDArray>): number[] => {
    const v = x.item() as Complex;
    return [v.re, v.im];
  };

  for (const backend of ["default", "fallback"] as const) {
    describe(`backend: ${backend}`, () => {
      beforeAll(() => np.linalg._setBackend(backend));
      afterAll(() => np.linalg._setBackend("default"));

      it("1-D dot, inner and matmul do not conjugate", () => {
        expect(z(np.dot(a, b))).toEqual([5, 6]);
        expect(z(np.inner(a, b))).toEqual([5, 6]);
        expect(z(np.matmul(a, b))).toEqual([5, 6]);
        expect(np.dot(a, b).shape).toEqual([]);
      });

      it("outer", () => {
        const r = np.outer(a, b);
        expect(r.shape).toEqual([2, 2]);
        expect(flat(r)).toEqual([[4, 3], [-2, 1], [5, -5], [1, 3]]);
        expect(np.outer(M, a).shape).toEqual([4, 2]); // inputs are flattened
      });

      it("2-D and N-D dot, 2-D inner, matmul", () => {
        expect(flat(np.dot(M, M))).toEqual([[0, 2], [4, 0], [0, 0], [0, -2]]);
        expect(flat(np.matmul(M, M))).toEqual([[0, 2], [4, 0], [0, 0], [0, -2]]);
        expect(flat(np.inner(M, M))).toEqual([[4, 2], [2, -2], [2, -2], [0, -2]]);
        // T = arange(8).reshape(2,2,2) * (1-1j); dot sums a's last axis with T's axis -2.
        const T = np.multiply(np.arange(8).reshape([2, 2, 2]), np.complex(1, -1));
        const d = np.dot(M, T);
        expect(d.shape).toEqual([2, 2, 2]);
        expect(flat(d)).toEqual([
          [4, -4], [8, -6], [20, -12], [24, -14], [0, -4], [0, -6], [0, -12], [0, -14],
        ]);
      });

      it("scalar operands broadcast-multiply", () => {
        expect(flat(np.dot(np.complex(0, 2), a))).toEqual([[-4, 2], [2, 6]]);
        expect(flat(np.dot(np.array(np.complex(0, 2)), M))).toEqual([[-2, 2], [0, 4], [0, 0], [2, 2]]);
      });

      it("promotes mixed real/complex inputs", () => {
        const b64 = b.astype("complex64");
        expect(np.dot(np.array([1, 2], { dtype: "float32" }), b64).dtype).toBe(np.complex64);
        const di = np.dot(np.array([1, 2], { dtype: "int32" }), b64);
        expect(di.dtype).toBe(np.complex128);
        expect(z(di)).toEqual([2, 1]);
        expect(np.outer(np.array([1, 2], { dtype: "float32" }), b64).dtype).toBe(np.complex64);
        expect(np.inner(np.array([1, 2], { dtype: "float32" }), b64).dtype).toBe(np.complex64);
      });

      it("empty contractions give +0+0j; inf follows NumPy (no Annex G recovery)", () => {
        const e = np.zeros([0], { dtype: "complex128" });
        const s = z(np.dot(e, e));
        expect(s).toEqual([0, 0]);
        expect(s.every((v) => !Object.is(v, -0))).toBe(true);
        const ie = np.inner(np.zeros([2, 0], { dtype: "complex128" }), np.zeros([3, 0], { dtype: "complex128" }));
        expect(ie.shape).toEqual([2, 3]);
        expect(flat(ie).every(([re, im]) => re === 0 && im === 0)).toBe(true);
        const o = z(np.outer([np.complex(Infinity, 0)], [np.complex(1, 0)]).reshape([]));
        expect(o[0]).toBe(Infinity);
        expect(Number.isNaN(o[1])).toBe(true);
        const o2 = z(np.outer([np.complex(Infinity, Infinity)], [np.complex(0, 1)]).reshape([]));
        expect(o2.every(Number.isNaN)).toBe(true);
        const mm = z(np.matmul([[np.complex(Infinity, 0)]], [[np.complex(1, 0)]]).reshape([]));
        expect(mm[0]).toBe(Infinity);
        expect(Number.isNaN(mm[1])).toBe(true);
      });

      it("shape errors stay typed", () => {
        expect(() => np.dot(a, M.reshape([4]))).toThrow(np.ShapeError);
        expect(() => np.inner(a, np.zeros([3], { dtype: "complex64" }))).toThrow(np.ShapeError);
      });
    });
  }
});

