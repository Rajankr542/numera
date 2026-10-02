import { describe, expect, it } from "vitest";
import np from "../src/index.js";
import { DTypeError, ValueError } from "../src/errors.js";

const P = np.polynomial;
const { Polynomial, Chebyshev, Legendre, Laguerre, Hermite, HermiteE } = P;
const arr = (a: { toArray(): unknown }) => a.toArray();
const close = (got: unknown, exp: number[], tol = 1e-12) => {
  const g = got as number[];
  expect(g.length).toBe(exp.length);
  g.forEach((v, i) => expect(Math.abs(v - (exp[i] as number))).toBeLessThanOrEqual(tol * (1 + Math.abs(exp[i] as number))));
};

// Expected values below were produced with NumPy 2.x numpy.polynomial.
describe("np.polynomial module functions (P15-3, D-182)", () => {
  it("evaluates every basis", () => {
    expect(P.chebyshev.chebval(0.5, [1, 2, 3]).item()).toBe(0.5);
    expect(arr(P.laguerre.lagval([0.5, 2], [1, 2, 3]))).toEqual([2.375, -4]);
    expect(P.hermite.hermval(0.5, [1, 2, 3]).item()).toBe(0);
    expect(P.hermite_e.hermeval(0.5, [1, 2, 3]).item()).toBe(-0.25);
    expect(P.legendre.legval(0.5, [1, 2, 3]).item()).toBe(1.625);
    expect(arr(P.polynomial.polyval([[1, 2], [3, 4]], [1, 2]))).toEqual([[3, 5], [7, 9]]);
  });

  it("divides series", () => {
    expect(P.chebyshev.chebdiv([1, 2, 3, 4], [1, 2]).map(arr)).toEqual([[-0.5, -1, 4], [2.5]]);
    expect(P.polynomial.polydiv([1, 2, 3, 4], [1, 2]).map(arr)).toEqual([[0.75, 0.5, 2], [0.25]]);
    expect(P.hermite.hermdiv([1, 2, 3, 4], [1, 2]).map(arr)).toEqual([[-7.25, 0.5, 2], [6.25]]);
    expect(() => P.polynomial.polydiv([1, 2], [0])).toThrow(ValueError);
  });

  it("differentiates, integrates, builds vander and companion matrices", () => {
    close(arr(P.legendre.legint([1, 2, 3], { m: 2, k: [1], lbnd: 0.5, scl: 2 })), [
      -0.66875, 2.05, 0.19047619047619047, 0.5333333333333333, 0.34285714285714286,
    ]);
    expect(arr(P.hermite_e.hermeder([1, 2, 3, 4], 2, 3))).toEqual([54, 216]);
    expect(arr(P.chebyshev.chebvander([0.5, 2], 3))).toEqual([[1, 0.5, -0.5, -1], [1, 2, 7, 26]]);
    close((arr(P.hermite.hermcompanion([1, 2, 3, 4])) as number[][]).flat(), [
      0, 0.7071067811865476, -0.044194173824159216, 0.7071067811865476, 0, 0.875, 0, 1, -0.375,
    ]);
    expect(() => P.polynomial.polyder([1, 2], -1)).toThrow(ValueError);
    expect(() => P.polynomial.polyint([1, 2], { m: 1, k: [1, 2] })).toThrow(ValueError);
    expect(() => P.polynomial.polycompanion([1])).toThrow(ValueError);
  });

  it("converts to and from the power basis and keeps module constants", () => {
    expect(arr(P.chebyshev.cheb2poly([1, 2, 3]))).toEqual([-2, 2, 6]);
    expect(arr(P.laguerre.poly2lag([1, 2, 3]))).toEqual([9, -14, 6]);
    close(arr(P.legendre.leg2poly(P.legendre.poly2leg([1, 2, 3, 4]))), [1, 2, 3, 4]);
    expect(arr(P.laguerre.lagx)).toEqual([1, -1]);
    expect(arr(P.hermite.hermx)).toEqual([0, 0.5]);
    expect(arr(P.laguerre.lagdomain)).toEqual([0, 1]);
    expect(arr(P.polynomial.polytrim([1, 2, 0, 1e-9], 1e-6))).toEqual([1, 2]);
  });

  it("finds roots (real when the result is real) and fits", () => {
    expect(arr(P.polynomial.polyroots([2, -3, 1]))).toEqual([1, 2]);
    expect(P.polynomial.polyroots([1, 2, 3]).dtype.name).toBe("complex128");
    expect(P.chebyshev.chebroots([1, 2, 3]).dtype.name).toBe("complex128");
    close(arr(P.laguerre.lagroots([1, 2, 3])), [0.9028324592902729, 4.4305008740430605]);
    close(arr(P.hermite.hermroots([1, 2, 3])), [-0.8333333333333334, 0.5]);
    close(arr(P.polynomial.polyfit([0, 1, 2, 3], [1, 3, 5, 7], 1)), [1, 2], 1e-12);
    close(arr(P.polynomial.polyfit([0, 1, 2, 3], [0, 1, 8, 27], [3])), [0, 0, 0, 1], 1e-12);
    expect(() => P.polynomial.polyfit([0, 1], [0], 1)).toThrow(DTypeError);
    expect(() => P.polynomial.polyfit([0, 1], [0, 1], -1)).toThrow(ValueError);
  });

  it("enforces maxpower", () => {
    expect(arr(P.chebyshev.chebpow([1, 2], 2))).toEqual([3, 4, 2]);
    expect(() => P.chebyshev.chebpow([1, 2], 17)).toThrow("Power is too large");
    expect(() => P.polynomial.polypow([1], -1)).toThrow(ValueError);
  });

  it("supports complex coefficients", () => {
    const r = P.polynomial.polymul([{ re: 0, im: 1 }], [1, 1]);
    expect(r.dtype.name).toBe("complex128");
    expect(arr(r)).toEqual([{ re: 0, im: 1 }, { re: 0, im: 1 }]);
  });
});

describe("np.polynomial classes (P15-3, D-182)", () => {
  it("prints like NumPy", () => {
    expect(String(new Polynomial([1, 2, 3]))).toBe("1.0 + 2.0·x + 3.0·x²");
    expect(String(new HermiteE([1, 2, 3]))).toBe("1.0 + 2.0·He₁(x) + 3.0·He₂(x)");
    expect(String(new Polynomial([1, -2.5, 0, 1e-5]))).toBe("1.0 - 2.5·x + 0.0·x² + (1.0e-05)·x³");
    expect(String(new Polynomial([1, 2], [0, 4]))).toBe("1.0 + 2.0·(-1.0 + 0.5x)");
    expect(String(new Chebyshev([1, 2], null, [0, 1]))).toBe("1.0 + 2.0·T₁((0.5 + 0.5x))");
    expect(String(new Polynomial([1, -1, NaN, Infinity]))).toBe("1.0 - 1.0·x - nan·x² + inf·x³");
    expect(String(new Polynomial([{ re: 1, im: 2 }, { re: 0, im: 3 }]))).toBe("(1+2j) + 3j·x");
    expect(new Polynomial([1, 2, 3]).repr()).toBe(
      "Polynomial([1., 2., 3.], domain=[-1.,  1.], window=[-1.,  1.], symbol='x')",
    );
    expect(new Chebyshev([1, 2, 3]).format("ascii")).toBe("1.0 + 2.0 T_1(x) + 3.0 T_2(x)");
    P.setDefaultPrintstyle("ascii");
    try {
      expect(String(new Polynomial([1, -2, 3]))).toBe("1.0 - 2.0 x + 3.0 x**2");
    } finally {
      P.setDefaultPrintstyle("unicode");
    }
    expect(() => P.setDefaultPrintstyle("x" as never)).toThrow(ValueError);
    const long = String(new Polynomial(np.arange(30)));
    expect(long.split("\n")[0]).toBe("0.0 + 1.0·x + 2.0·x² + 3.0·x³ + 4.0·x⁴ + 5.0·x⁵ + 6.0·x⁶ + 7.0·x⁷ +");
    expect(long.split("\n").length).toBe(5);
  });

  it("does arithmetic", () => {
    const p = new Polynomial([1, 2]);
    expect(arr(p.add(3).coef)).toEqual([4, 2]);
    expect(arr(p.rsub(3).coef)).toEqual([2, -2]);
    expect(arr(p.mul(p).coef)).toEqual([1, 4, 4]);
    expect(arr(new Legendre([1, 2, 3]).floordiv(new Legendre([1, 1])).coef)).toEqual([-2.5, 4.5]);
    expect(arr(new Laguerre([1, 2, 3]).mod(new Laguerre([1, 1])).coef)).toEqual([-4]);
    expect(arr(new Hermite([1, 2]).pow(3).coef)).toEqual([25, 54, 12, 8]);
    expect(arr(p.neg().coef)).toEqual([-1, -2]);
    expect(arr(p.truediv(2).coef)).toEqual([0.5, 1]);
    expect(() => p.add(new Chebyshev([1]))).toThrow("Polynomial types differ");
    expect(() => p.add(new Polynomial([1], [0, 1]))).toThrow("Domains differ");
    expect(() => p.pow(101)).toThrow("Power is too large");
    expect(p.equals(new Polynomial([1, 2]))).toBe(true);
    expect(p.equals(new Polynomial([1, 2, 0]))).toBe(false);
  });

  it("evaluates, composes, converts and maps domains", () => {
    expect(arr(new Polynomial([1, 2]).call(np.array([[1, 2], [3, 4]])))).toEqual([[3, 5], [7, 9]]);
    expect(String(new Polynomial([1, 2]).call(new Polynomial([0, 1])))).toBe("1.0 + 2.0·x");
    expect(arr(new Chebyshev([1, 2, 3]).convert(null, Polynomial).coef)).toEqual([-2, 2, 6]);
    expect(arr(new Polynomial([1, 2, 3]).convert(null, Laguerre).coef)).toEqual([9, -14, 6]);
    expect(arr(Chebyshev.cast(new Polynomial([1, 2, 3])).coef)).toEqual([2.5, 2, 1.5]);
    expect(new Polynomial([1, 2], [0, 2]).mapparms()).toEqual([-1, 1]);
    expect(new Polynomial([1, 2, 3]).linspace(3).map(arr)).toEqual([[-1, 0, 1], [2, 1, 6]]);
  });

  it("differentiates, integrates, trims", () => {
    close(arr(new Polynomial([1, 2, 3]).integ(2, [1, 2], 1).coef), [2.916666666666667, -2, 0.5, 1 / 3, 0.25]);
    expect(arr(new Chebyshev([1, 2, 3]).deriv().coef)).toEqual([2, 12]);
    close(arr(new Legendre([1, 2, 3], [0, 2]).integ().coef), [1 / 3, 0.4, 2 / 3, 0.6]);
    expect(arr(new Polynomial([1, 2, 3]).trim(2.5).coef)).toEqual([1, 2, 3]);
    expect(arr(new Polynomial([1, 2, 3]).cutdeg(1).coef)).toEqual([1, 2]);
    expect(new Polynomial([1, 2, 3]).degree()).toBe(2);
  });

  it("fits, builds from roots and finds roots", () => {
    const f = Polynomial.fit([0, 1, 2, 3], [1, 3, 5, 7], 1);
    expect(f.repr()).toBe("Polynomial([4., 3.], domain=[0., 3.], window=[-1.,  1.], symbol='x')");
    close(arr(f.convert().coef), [1, 2]);
    close(arr(Legendre.fit(np.linspace(0, 1, 5), np.power(np.linspace(0, 1, 5), 2), 2).coef), [1 / 3, 0.5, 1 / 6], 1e-12);
    const full = Polynomial.fitFull([0, 1, 2], [1, 2, 3], 1);
    expect(full.rank).toBe(2);
    expect(arr(Hermite.fromroots([1, 2]).coef)).toEqual([2.5, -1.5, 0.25]);
    close(arr(new Laguerre([1, 2, 3]).roots()), [0.9028324592902729, 4.4305008740430605]);
    expect(arr(Polynomial.identity().coef)).toEqual([0, 1]);
    expect(arr(Laguerre.basis(2).coef)).toEqual([0, 0, 1]);
    // the root 2 in the window [-1, 1] maps to 6 in the domain [0, 4]
    close(arr(new Polynomial([-2, 1], [0, 4]).roots()), [6]);
  });

  it("validates constructor arguments", () => {
    expect(() => new Polynomial([])).toThrow("Coefficient array is empty");
    expect(() => new Polynomial([[1]])).toThrow("Coefficient array is not 1-d");
    expect(() => new Polynomial([1], [0, 1, 2])).toThrow("Domain has wrong number of elements.");
    expect(() => new Polynomial([1], null, null, "1x")).toThrow(ValueError);
    expect(new Polynomial([1], null, null, "t").symbol).toBe("t");
  });
});
