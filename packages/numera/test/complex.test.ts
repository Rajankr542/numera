import { afterAll, beforeAll, describe, expect, it } from "vitest";
import np, { Complex, DTypeError, type NormOrder } from "../src/index.js";

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

describe("complex linalg.det (P1-5a, D-038)", () => {
  const z = (x: InstanceType<typeof np.NDArray>): number[] => {
    const v = x.item() as Complex;
    return [v.re, v.im];
  };
  const A = [
    [np.complex(1, 2), np.complex(3, -1)],
    [np.complex(0, 0.5), np.complex(2, 0)],
  ];
  for (const backend of ["default", "fallback"] as const) {
    describe(`backend: ${backend}`, () => {
      beforeAll(() => np.linalg._setBackend(backend));
      afterAll(() => np.linalg._setBackend("default"));

      it("keeps the complex dtype and matches NumPy", () => {
        const d = np.linalg.det(A);
        expect(d.dtype).toBe(np.complex128);
        expect(d.shape).toEqual([]);
        // NumPy: (1.5000000000000002+2.5j) for complex128, (1.5+2.5j) for complex64.
        const [re, im] = z(d);
        expect(re).toBeCloseTo(1.5, 14);
        expect(im).toBeCloseTo(2.5, 14);
        const d64 = np.linalg.det(np.array(A, { dtype: "complex64" }));
        expect(d64.dtype).toBe(np.complex64);
        expect(z(d64)).toEqual([1.5, 2.5]);
      });

      it("batched, empty, singular and NaN inputs", () => {
        const b = np.multiply(np.arange(8).reshape([2, 2, 2]), np.complex(1, 1));
        const bd = np.linalg.det(b);
        expect(bd.shape).toEqual([2]);
        for (const v of bd.toArray() as Complex[]) {
          expect(v.re).toBeCloseTo(0, 12);
          expect(v.im).toBeCloseTo(-4, 12);
        }
        expect(z(np.linalg.det(np.zeros([0, 0], { dtype: "complex64" })))).toEqual([1, 0]);
        const sing = z(np.linalg.det([[np.complex(0, 0), np.complex(1, 0)], [np.complex(0, 0), np.complex(0, 1)]]));
        expect(sing).toEqual([0, 0]);
        const nan = z(np.linalg.det([[np.complex(NaN, 0), 1], [1, 1]]));
        expect(nan.every(Number.isNaN)).toBe(true);
        expect(() => np.linalg.det(np.zeros([2, 3], { dtype: "complex128" }))).toThrow(np.LinAlgError);
      });
    });
  }
});

describe("complex linalg.inv / solve (P1-5b, D-039)", () => {
  const flat = (x: InstanceType<typeof np.NDArray>): number[][] =>
    (x.flatten().toArray() as Complex[]).map((v) => [v.re, v.im]);
  const close = (got: number[][], want: number[][], digits: number) => {
    expect(got.length).toBe(want.length);
    got.forEach(([re, im], i) => {
      expect(re).toBeCloseTo(want[i]![0]!, digits);
      expect(im).toBeCloseTo(want[i]![1]!, digits);
    });
  };
  // A = [[1+2j, 3-j], [0.5j, 2]]; NumPy inv(A):
  const A = [
    [np.complex(1, 2), np.complex(3, -1)],
    [np.complex(0, 0.5), np.complex(2, 0)],
  ];
  const invA = [[0.3529411764705883, -0.5882352941176472], [-0.235294117647059, 1.0588235294117652],
    [-0.1470588235294118, -0.08823529411764708], [0.7647058823529413, 0.05882352941176474]];
  for (const backend of ["default", "fallback"] as const) {
    describe(`backend: ${backend}`, () => {
      beforeAll(() => np.linalg._setBackend(backend));
      afterAll(() => np.linalg._setBackend("default"));

      it("inv matches NumPy for both widths", () => {
        const r = np.linalg.inv(A);
        expect(r.dtype).toBe(np.complex128);
        expect(r.shape).toEqual([2, 2]);
        close(flat(r), invA, 14);
        const r64 = np.linalg.inv(np.array(A, { dtype: "complex64" }));
        expect(r64.dtype).toBe(np.complex64);
        close(flat(r64), invA, 6);
        expect(np.linalg.inv(np.zeros([0, 0], { dtype: "complex64" })).shape).toEqual([0, 0]);
      });

      it("solve with vector, matrix and broadcast right-hand sides", () => {
        const x = np.linalg.solve(A, [1, np.complex(0, 1)]);
        expect(x.shape).toEqual([2]);
        // NumPy: [-0.7058823529411768-0.8235294117647061j, -0.20588235294117654+0.6764705882352944j]
        close(flat(x), [[-0.7058823529411768, -0.8235294117647061], [-0.20588235294117654, 0.6764705882352944]], 14);
        // Batch of two systems ([A, j*I]) against one broadcast (2,1) b.
        const stackA = np.array([A, [[np.complex(0, 1), 0], [0, np.complex(0, 1)]]]);
        const X = np.linalg.solve(stackA, [[1], [2]]);
        expect(X.shape).toEqual([2, 2, 1]);
        close(flat(X), [[-0.11764705882352955, 1.5294117647058831], [1.382352941176471, 0.0294117647058824], [0, -1], [0, -2]], 14);
      });

      it("singular input raises LinAlgError", () => {
        const S = [[np.complex(1, 1), np.complex(2, 2)], [np.complex(1, 1), np.complex(2, 2)]];
        expect(() => np.linalg.inv(S)).toThrow(np.LinAlgError);
        expect(() => np.linalg.solve(S, [1, 2])).toThrow(np.LinAlgError);
      });
    });
  }

  it("result dtype follows NumPy (_commonType)", () => {
    const a64 = np.array(A, { dtype: "complex64" });
    expect(np.linalg.solve(a64, np.array([1, 2], { dtype: "float32" })).dtype).toBe(np.complex64);
    expect(np.linalg.solve(a64, np.array([1, 2], { dtype: "float64" })).dtype).toBe(np.complex128);
    expect(np.linalg.solve(a64, np.array([1, 2], { dtype: "int8" })).dtype).toBe(np.complex128);
    expect(np.linalg.solve(np.eye(2, undefined, { dtype: "float32" }), a64).dtype).toBe(np.complex64);
    expect(() => np.linalg.solve(a64, np.array([1, 2], { dtype: "float16" }))).toThrow(np.DTypeError);
  });
});


describe("complex linalg.qr (P1-5c, D-040)", () => {
  const flat = (x: InstanceType<typeof np.NDArray>): number[][] =>
    (x.flatten().toArray() as Complex[]).map((v) => [v.re, v.im]);
  const close = (got: number[][], want: number[][], digits: number) => {
    expect(got.length).toBe(want.length);
    got.forEach(([re, im], i) => {
      expect(re).toBeCloseTo(want[i]![0]!, digits);
      expect(im).toBeCloseTo(want[i]![1]!, digits);
    });
  };
  // A (3x2) = [[1+2j, 3-j], [0.5j, 2], [1, j]]; NumPy qr(A) reference values.
  const A = [
    [np.complex(1, 2), np.complex(3, -1)],
    [np.complex(0, 0.5), 2],
    [1, np.complex(0, 1)],
  ];
  const Q = [[-0.4, -0.8], [-0.226778684, 0.0755928946], [0, -0.2], [-0.544268841, 0.0302371578],
    [-0.4, 0], [0.0604743157, -0.801284683]];
  const R = [[-2.5, 0], [-0.4, 2.8], [0, 0], [-2.6457513110645907, 0]];
  for (const backend of ["default", "fallback"] as const) {
    describe(`backend: ${backend}`, () => {
      beforeAll(() => np.linalg._setBackend(backend));
      afterAll(() => np.linalg._setBackend("default"));

      it("reduced qr matches NumPy for both widths", () => {
        for (const dtype of ["complex128", "complex64"] as const) {
          const { Q: q, R: r } = np.linalg.qr(np.array(A, { dtype }));
          expect(q!.dtype).toBe(np[dtype]);
          expect(r.dtype).toBe(np[dtype]);
          expect(q!.shape).toEqual([3, 2]);
          expect(r.shape).toEqual([2, 2]);
          close(flat(q!), Q, dtype === "complex128" ? 8 : 6);
          close(flat(r), R, dtype === "complex128" ? 14 : 5);
        }
      });

      it("complete and r modes, empty and NaN inputs", () => {
        const c = np.linalg.qr(A, "complete");
        expect(c.Q!.shape).toEqual([3, 3]);
        expect(c.R.shape).toEqual([3, 2]);
        // Q R reconstructs A.
        close(flat(np.matmul(c.Q!, c.R)), flat(np.array(A)), 12);
        const r = np.linalg.qr(np.transpose(np.array(A)), "r");
        expect(r.Q).toBeNull();
        expect(r.R.shape).toEqual([2, 3]);
        close(flat(r.R).slice(0, 1), [[-3.872983346207417, 0]], 12);
        const e = np.linalg.qr(np.zeros([3, 0], { dtype: "complex64" }), "complete");
        expect(e.Q!.dtype).toBe(np.complex64);
        expect(flat(e.Q!)).toEqual(flat(np.eye(3, undefined, { dtype: "complex64" })));
        const n = np.linalg.qr([[np.complex(NaN, 0), 1], [1, 1]]);
        expect(Number.isNaN(flat(n.R)[0]![0])).toBe(true);
        expect(() => np.linalg.qr([np.complex(1, 1)])).toThrow(np.LinAlgError);
      });
    });
  }
});


describe("complex linalg.svd (P1-5d, D-041)", () => {
  const flat = (x: InstanceType<typeof np.NDArray>): number[][] =>
    (x.flatten().toArray() as Complex[]).map((v) => [v.re, v.im]);
  const reals = (x: InstanceType<typeof np.NDArray>): number[] => x.toArray() as number[];
  // A (3x2) = [[1+2j, 3-j], [0.5j, 2], [1, j]]; NumPy S = [4.352020701327292, 1.5198407203449662].
  const A = [
    [np.complex(1, 2), np.complex(3, -1)],
    [np.complex(0, 0.5), 2],
    [1, np.complex(0, 1)],
  ];
  const S = [4.352020701327292, 1.5198407203449662];
  const maxAbs = (x: InstanceType<typeof np.NDArray>): number =>
    Math.max(0, ...flat(x).map(([re, im]) => Math.hypot(re, im)));
  for (const backend of ["default", "fallback"] as const) {
    describe(`backend: ${backend}`, () => {
      beforeAll(() => np.linalg._setBackend(backend));
      afterAll(() => np.linalg._setBackend("default"));

      it("dtypes, shapes, singular values and reconstruction", () => {
        for (const [dtype, sdt, digits, tol] of [
          ["complex128", "float64", 13, 1e-13],
          ["complex64", "float32", 5, 1e-5],
        ] as const) {
          const a = np.array(A, { dtype });
          for (const fullMatrices of [true, false]) {
            const { U, S: s, Vh } = np.linalg.svd(a, { fullMatrices });
            expect(U!.dtype).toBe(np[dtype]);
            expect(Vh!.dtype).toBe(np[dtype]);
            expect(s.dtype).toBe(np[sdt]);
            expect(U!.shape).toEqual([3, fullMatrices ? 3 : 2]);
            expect(Vh!.shape).toEqual([2, 2]);
            reals(s).forEach((v, i) => expect(v).toBeCloseTo(S[i]!, digits));
            // U[:, :2] * S @ Vh == A, and U^H U == I.
            const u2 = fullMatrices ? U!.slice([null, [0, 2]]) : U!;
            const rec = np.matmul(np.multiply(u2, s), Vh!);
            expect(maxAbs(np.subtract(rec, a))).toBeLessThan(10 * tol);
            const gram = np.matmul(np.conjugate(np.transpose(U!)), U!);
            expect(maxAbs(np.subtract(gram, np.eye(U!.shape[1]!)))).toBeLessThan(10 * tol);
          }
          const sv = np.linalg.svd(a, { computeUV: false });
          expect(sv.U).toBeNull();
          expect(sv.Vh).toBeNull();
          expect(sv.S.dtype).toBe(np[sdt]);
          reals(sv.S).forEach((v, i) => expect(v).toBeCloseTo(S[i]!, digits));
        }
      });

      it("wide, batched, empty and non-finite inputs", () => {
        const wide = np.conjugate(np.transpose(np.array(A)));
        const w = np.linalg.svd(wide);
        expect(w.U!.shape).toEqual([2, 2]);
        expect(w.Vh!.shape).toEqual([3, 3]);
        reals(w.S).forEach((v, i) => expect(v).toBeCloseTo(S[i]!, 13));
        const b = np.array([
          [[np.complex(0, 1), 0], [0, 2]],
          [[1, np.complex(0, 1)], [np.complex(0, 1), 1]],
        ]);
        const bs = np.linalg.svd(b, { computeUV: false }).S;
        expect(bs.shape).toEqual([2, 2]);
        (bs.toArray() as number[][]).flat().forEach((v, i) =>
          expect(v).toBeCloseTo([2, 1, Math.SQRT2, Math.SQRT2][i]!, 13));
        const e = np.linalg.svd(np.zeros([3, 0], { dtype: "complex128" }));
        expect(e.U!.shape).toEqual([3, 3]);
        expect(e.U!.dtype).toBe(np.complex128);
        expect(flat(e.U!)[4]).toEqual([1, 0]);
        expect(e.Vh!.shape).toEqual([0, 0]);
        expect(e.S.dtype).toBe(np.float64);
        expect(() => np.linalg.svd([[np.complex(NaN, 0), 1], [1, 1]])).toThrow(np.LinAlgError);
        expect(() => np.linalg.svd([[np.complex(1, Infinity)]], { computeUV: false })).toThrow(
          "SVD did not converge",
        );
      });
    });
  }
});

describe("complex linalg.eigh / eigvalsh (P1-5e.1, D-042)", () => {
  const flat = (x: InstanceType<typeof np.NDArray>): number[][] =>
    (x.flatten().toArray() as Complex[]).map((v) => [v.re, v.im]);
  const maxAbs = (x: InstanceType<typeof np.NDArray>): number =>
    Math.max(0, ...flat(x).map(([re, im]) => Math.hypot(re, im)));
  // H = [[2, 1-j], [1+j, 3]] -> NumPy eigenvalues [1, 4].
  const H = [
    [2, np.complex(1, -1)],
    [np.complex(1, 1), 3],
  ];
  // 3x3 Hermitian.
  const A3 = [
    [4, np.complex(1, -2), np.complex(0, 1)],
    [np.complex(1, 2), 3, np.complex(2, -1)],
    [np.complex(0, -1), np.complex(2, 1), 5],
  ];
  for (const backend of ["default", "fallback"] as const) {
    describe(`backend: ${backend}`, () => {
      beforeAll(() => np.linalg._setBackend(backend));
      afterAll(() => np.linalg._setBackend("default"));

      it("dtypes, values, reconstruction and unitarity", () => {
        for (const [dtype, wdt, digits, tol] of [
          ["complex128", "float64", 13, 1e-13],
          ["complex64", "float32", 5, 1e-5],
        ] as const) {
          const h = np.array(H, { dtype });
          const { eigenvalues, eigenvectors } = np.linalg.eigh(h);
          expect(eigenvalues.dtype).toBe(np[wdt]);
          expect(eigenvectors.dtype).toBe(np[dtype]);
          expect(eigenvectors.shape).toEqual([2, 2]);
          (eigenvalues.toArray() as number[]).forEach((v, i) => expect(v).toBeCloseTo([1, 4][i]!, digits));
          const vals = np.linalg.eigvalsh(h);
          expect(vals.dtype).toBe(np[wdt]);
          (vals.toArray() as number[]).forEach((v, i) => expect(v).toBeCloseTo([1, 4][i]!, digits));
          const a = np.array(A3, { dtype });
          const r = np.linalg.eigh(a);
          const V = r.eigenvectors;
          const av = np.matmul(a, V);
          const vl = np.multiply(V, r.eigenvalues);
          expect(maxAbs(np.subtract(av, vl))).toBeLessThan(50 * tol);
          const gram = np.matmul(np.conjugate(np.transpose(V)), V);
          expect(maxAbs(np.subtract(gram, np.eye(3)))).toBeLessThan(10 * tol);
          const w = r.eigenvalues.toArray() as number[];
          expect(w[0]! <= w[1]! && w[1]! <= w[2]!).toBe(true);
        }
      });

      it("lower triangle only, batched, empty and errors", () => {
        // Upper triangle and the diagonal's imaginary part are ignored (UPLO='L').
        const g = np.array([
          [np.complex(2, 5), np.complex(9, 9)],
          [np.complex(1, 1), np.complex(3, -7)],
        ]);
        (np.linalg.eigvalsh(g).toArray() as number[]).forEach((v, i) =>
          expect(v).toBeCloseTo([1, 4][i]!, 13));
        const b = np.array([
          [[1, 0], [0, -2]],
          [[0, np.complex(0, 1)], [np.complex(0, -1), 0]],
        ]);
        const bw = np.linalg.eigvalsh(b);
        expect(bw.shape).toEqual([2, 2]);
        (bw.toArray() as number[][]).flat().forEach((v, i) =>
          expect(v).toBeCloseTo([-2, 1, -1, 1][i]!, 13));
        const e = np.linalg.eigh(np.zeros([0, 0], { dtype: "complex128" }));
        expect(e.eigenvalues.shape).toEqual([0]);
        expect(e.eigenvalues.dtype).toBe(np.float64);
        expect(e.eigenvectors.dtype).toBe(np.complex128);
        expect(() => np.linalg.eigh(np.zeros([2, 3], { dtype: "complex64" }))).toThrow(np.LinAlgError);
        expect(() => np.linalg.eigvalsh(np.zeros([3], { dtype: "complex128" }))).toThrow(np.LinAlgError);
      });
    });
  }
});


describe("complex linalg.eig / eigvals (P1-5e.2, D-043)", () => {
  const flat = (x: InstanceType<typeof np.NDArray>): number[][] =>
    (x.flatten().toArray() as Complex[]).map((v) => [v.re, v.im]);
  const maxAbs = (x: InstanceType<typeof np.NDArray>): number =>
    Math.max(0, ...flat(x).map(([re, im]) => Math.hypot(re, im)));
  const sorted = (x: InstanceType<typeof np.NDArray>): number[][] =>
    flat(x).sort((p, q) => p[0]! - q[0]! || p[1]! - q[1]!);
  // Upper triangular -> eigenvalues are the diagonal: -1, 1+j, 2-j.
  const T = [
    [np.complex(1, 1), 2, 0],
    [0, np.complex(2, -1), np.complex(0, 1)],
    [0, 0, -1],
  ];
  const A = [
    [np.complex(1, 2), np.complex(3, -1), np.complex(0, 1)],
    [-2, np.complex(1, 1), np.complex(2, 3)],
    [0.5, np.complex(-1, 2), np.complex(4, -2)],
  ];
  for (const backend of ["default", "fallback"] as const) {
    describe(`backend: ${backend}`, () => {
      beforeAll(() => np.linalg._setBackend(backend));
      afterAll(() => np.linalg._setBackend("default"));

      it("dtypes, values, reconstruction and unit-norm vectors", () => {
        for (const [dtype, tol] of [
          ["complex128", 1e-12],
          ["complex64", 1e-5],
        ] as const) {
          const t = np.array(T, { dtype });
          const e = np.linalg.eig(t);
          expect(e.eigenvalues.dtype).toBe(np[dtype]);
          expect(e.eigenvectors.dtype).toBe(np[dtype]);
          const want = [[-1, 0], [1, 1], [2, -1]];
          for (const got of [sorted(e.eigenvalues), sorted(np.linalg.eigvals(t))]) {
            got.forEach(([re, im], i) => {
              expect(Math.abs(re! - want[i]![0]!)).toBeLessThan(10 * tol);
              expect(Math.abs(im! - want[i]![1]!)).toBeLessThan(10 * tol);
            });
          }
          expect(np.linalg.eigvals(t).dtype).toBe(np[dtype]);
          const a = np.array(A, { dtype });
          const r = np.linalg.eig(a);
          const V = r.eigenvectors;
          const res = np.subtract(np.matmul(a, V), np.multiply(V, r.eigenvalues));
          expect(maxAbs(res)).toBeLessThan(50 * tol);
          const norms = np.sum(np.multiply(np.conjugate(V), V), { axis: 0 });
          for (const [re, im] of flat(norms)) {
            expect(Math.abs(re! - 1)).toBeLessThan(10 * tol);
            expect(Math.abs(im!)).toBeLessThan(10 * tol);
          }
          const d = sorted(np.linalg.eigvals(a));
          sorted(r.eigenvalues).forEach(([re, im], i) => {
            expect(Math.abs(re! - d[i]![0]!)).toBeLessThan(50 * tol);
            expect(Math.abs(im! - d[i]![1]!)).toBeLessThan(50 * tol);
          });
        }
      });

      it("batched, empty and errors", () => {
        const b = np.array([
          [[1, 0], [0, np.complex(0, 1)]],
          [[0, 1], [-1, 0]],
        ]);
        const bw = np.linalg.eigvals(b);
        expect(bw.shape).toEqual([2, 2]);
        expect(np.linalg.eig(b).eigenvectors.shape).toEqual([2, 2, 2]);
        const s = flat(np.sum(bw, { axis: 1 }));
        expect(s[0]![0]).toBeCloseTo(1, 12);
        expect(s[0]![1]).toBeCloseTo(1, 12);
        expect(Math.hypot(s[1]![0]!, s[1]![1]!)).toBeLessThan(1e-12);
        const z = np.linalg.eig(np.zeros([0, 0], { dtype: "complex64" }));
        expect(z.eigenvalues.shape).toEqual([0]);
        expect(z.eigenvalues.dtype).toBe(np.complex64);
        expect(() => np.linalg.eig([[np.complex(NaN, 0), 0], [0, 1]])).toThrow(np.LinAlgError);
        expect(() => np.linalg.eigvals([[np.complex(0, Infinity)]])).toThrow(np.LinAlgError);
        expect(() => np.linalg.eig(np.zeros([2, 3], { dtype: "complex128" }))).toThrow(np.LinAlgError);
        expect(() => np.linalg.eigvals(np.zeros([3], { dtype: "complex64" }))).toThrow(np.LinAlgError);
      });
    });
  }
});

describe("complex linalg.lstsq / norm (P1-5e.3, D-044)", () => {
  // NumPy 2.5 reference values for a = [[1+2j, 3-j], [0.5j, 2], [1, 1j]].
  const A = [
    [np.complex(1, 2), np.complex(3, -1)],
    [np.complex(0, 0.5), 2],
    [1, np.complex(0, 1)],
  ];
  const B = [1, np.complex(0, 2), 3];
  for (const backend of ["default", "fallback"] as const) {
    describe(`backend: ${backend}`, () => {
      beforeAll(() => np.linalg._setBackend(backend));
      afterAll(() => np.linalg._setBackend("default"));

      it("lstsq dtypes, residuals and normal equations", () => {
        for (const [dtype, rdt, tol] of [
          ["complex128", "float64", 1e-12],
          ["complex64", "float32", 1e-5],
        ] as const) {
          const a = np.array(A, { dtype });
          const b = np.array(B, { dtype });
          const r = np.linalg.lstsq(a, b);
          expect(r.x.dtype).toBe(np[dtype]);
          expect(r.residuals.dtype).toBe(np[rdt]);
          expect(r.s.dtype).toBe(np[rdt]);
          expect(r.rank).toBe(2);
          expect(Math.abs((r.residuals.toArray() as number[])[0]! - 7.82285714285714)).toBeLessThan(10 * tol);
          // Aᴴ (b - A x) = 0.
          const res = np.subtract(b, np.matmul(a, r.x));
          const g = np.matmul(np.conjugate(np.transpose(a)), res);
          for (const v of g.toArray() as Complex[]) expect(Math.hypot(v.re, v.im)).toBeLessThan(10 * tol);
        }
        expect(np.linalg.lstsq(np.array(A), [1, 1, 1]).x.dtype).toBe(np.complex128);
        expect(np.linalg.lstsq(np.array(A, { dtype: "complex64" }), np.ones([3], { dtype: "float32" })).x.dtype).toBe(
          np.complex64,
        );
        const e = np.linalg.lstsq(np.zeros([0, 2], { dtype: "complex128" }), np.zeros([0], { dtype: "complex128" }));
        expect(e.x.shape).toEqual([2]);
        expect(e.rank).toBe(0);
      });

      it("norm orders, dtype and axis", () => {
        const want: [NormOrder, number][] = [
          [null, 4.6097722286464435],
          ["fro", 4.6097722286464435],
          ["nuc", 5.871861421672258],
          [1, 6.16227766016838],
          [-1, 3.73606797749979],
          [2, 4.352020701327292],
          [-2, 1.519840720344966],
          [Infinity, 5.39834563766817],
          [-Infinity, 2.0],
        ];
        for (const [dtype, rdt, tol] of [
          ["complex128", "float64", 1e-12],
          ["complex64", "float32", 1e-5],
        ] as const) {
          const a = np.array(A, { dtype });
          for (const [ord, v] of want) {
            const n = np.linalg.norm(a, { ord });
            expect(n.dtype).toBe(np[rdt]);
            expect(Math.abs((n.item() as number) - v)).toBeLessThan(10 * tol * (1 + v));
          }
          const b = np.array(B, { dtype });
          for (const [ord, v] of [[null, 3.7416573867739413], [0, 3], [3, 3.3019272488946263], [-1, 0.5454545454545455]] as const) {
            expect(Math.abs((np.linalg.norm(b, { ord }).item() as number) - v)).toBeLessThan(10 * tol);
          }
          const rows = np.linalg.norm(a, { axis: 1, keepdims: true });
          expect(rows.shape).toEqual([3, 1]);
          expect(rows.dtype).toBe(np[rdt]);
          expect(Math.abs((rows.toArray() as number[][])[0]![0]! - Math.sqrt(15))).toBeLessThan(10 * tol);
        }
      });
    });
  }
});

