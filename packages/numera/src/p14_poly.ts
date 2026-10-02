// P14 legacy polynomials (D-173): poly, poly1d, polyadd, polyder, polydiv,
// polyfit, polyint, polymul, polysub, polyval, roots. Each function follows
// numpy/lib/_polynomial_impl.py step by step on top of existing ops; the
// convolution and long-division loops are native (`addon.p14`).
import { nativeModule, type NativeNDArray } from "./addon.js";
import { Complex, isComplexLike } from "./complex.js";
import { arange, asarray, ones, zeros, zerosLike } from "./creation.js";
import { promoteTypes } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";
import { nonzero } from "./indexing.js";
import { dot, eigvals, inv, lstsq, outer } from "./linalg.js";
import { NDArray, type NestedArray } from "./ndarray.js";
import { all, equal } from "./p05_compare.js";
import { trimZeros } from "./p06_edit.js";
import { concatenate, hstack } from "./p06_join.js";
import { diag, vander } from "./p07_creation.js";
import { sort } from "./p09.js";
import { sum } from "./reduce.js";
import { add, conjugate, divide, imag, multiply, negative, real, sqrt, subtract } from "./ufunc.js";

interface P14PolyNative {
  convolveFull(a: NativeNDArray, b: NativeNDArray): NativeNDArray;
  polydiv(u: NativeNDArray, v: NativeNDArray): NativeNDArray[];
}
const native = nativeModule<P14PolyNative>("p14");

type ArrayLike = NDArray | NestedArray;
/** A polynomial: coefficients (highest power first) or a `poly1d`. */
export type PolyLike = ArrayLike | poly1d;
type Scalar = number | bigint | boolean | Complex | { readonly re: number; readonly im?: number };

const isScalar = (v: unknown): v is Scalar =>
  typeof v === "number" || typeof v === "bigint" || typeof v === "boolean" || isComplexLike(v);

const arr = (a: PolyLike): NDArray => (a instanceof poly1d ? a.coeffs : asarray(a as ArrayLike));
const atleast1 = (a: PolyLike): NDArray => {
  const x = arr(a);
  return x.ndim === 0 ? x.reshape([1]) : x;
};
const len = (a: NDArray): number => (a.ndim === 0 ? 0 : a.shape[0]!);
const sl = (a: NDArray, start: number | null, stop: number | null = null): NDArray => a.slice([[start, stop]]);
const isInexact = (a: NDArray): boolean => a.dtype.kind === "f" || a.dtype.kind === "c";
// NumPy `a + 0.0`: integers and bool become float64; inexact dtypes are kept.
const plusZero = (a: NDArray): NDArray => (isInexact(a) ? a : a.astype("float64"));
const anyPoly = (...xs: unknown[]): boolean => xs.some((x) => x instanceof poly1d);
const ret = (v: NDArray, truepoly: boolean): NDArray | poly1d => (truepoly ? new poly1d(v) : v);

function convolve(a: NDArray, b: NDArray): NDArray {
  return wrapNative(() => NDArray._wrap(native.convolveFull(a._native, b._native)));
}

/** NumPy mintypecode(dt.char) for 1-D `poly` input: float32/complex64 stay, the rest is float64/complex128. */
function minInexact(a: NDArray): NDArray {
  const n = a.dtype.name;
  if (n === "float32" || n === "float64" || n === "complex64" || n === "complex128") return a;
  return a.astype("float64");
}

/**
 * np.poly(seq_of_zeros): coefficients of the monic polynomial with the given
 * roots, or the characteristic polynomial of a square matrix. Returns 1.0
 * for no roots.
 */
export function poly(seqOfZeros: ArrayLike): NDArray | number {
  let z = atleast1(seqOfZeros);
  const sh = z.shape;
  if (sh.length === 2 && sh[0] === sh[1] && sh[0] !== 0) z = eigvals(z);
  else if (sh.length === 1) z = minInexact(z);
  else throw new ValueError("input must be 1d or non-empty square 2d array.");
  if (len(z) === 0) return 1.0;
  const dt = z.dtype;
  let a = ones([1], { dtype: dt });
  for (let k = 0; k < len(z); k++) {
    a = convolve(a, concatenate([ones([1], { dtype: dt }), negative(sl(z, k, k + 1))]));
  }
  if (a.dtype.kind === "c") {
    const r = z.astype("complex128");
    if (all(equal(sort(r), sort(conjugate(r)))).item() === true) a = real(a).copy();
  }
  return a;
}

/** np.roots(p): roots of the polynomial with coefficients `p` (eigenvalues of the companion matrix). */
export function roots(p: PolyLike): NDArray {
  let c = atleast1(p);
  if (c.ndim !== 1) throw new ValueError("Input must be a rank-1 array.");
  const nz = nonzero(c.ravel())[0]!.toArray() as number[];
  if (nz.length === 0) return asarray([]);
  const first = nz[0]!;
  const last = nz[nz.length - 1]!;
  const trailing = len(c) - last - 1;
  c = sl(c, first, last + 1);
  if (!isInexact(c)) c = c.astype("float64");
  const N = len(c);
  let r: NDArray;
  if (N > 1) {
    const A = diag(ones([N - 2], { dtype: c.dtype }), -1);
    A.set([0], negative(divide(sl(c, 1), c.get(0))));
    const w = eigvals(A);
    const realInput = A.dtype.kind !== "c";
    if (realInput && all(equal(imag(w), 0)).item() === true) {
      r = real(w).astype(A.dtype.name === "float32" ? "float32" : "float64");
    } else {
      r = w.astype(A.dtype.name === "float32" || A.dtype.name === "complex64" ? "complex64" : "complex128");
    }
  } else {
    r = asarray([]);
  }
  return hstack([r, zeros([trailing], { dtype: r.dtype })]);
}

/** np.polyint(p, m = 1, k = 0): `m`-th antiderivative; `k` gives the integration constants. */
export function polyint(p: poly1d, m?: number, k?: ArrayLike | null): poly1d;
export function polyint(p: ArrayLike, m?: number, k?: ArrayLike | null): NDArray;
export function polyint(p: PolyLike, m = 1, k: ArrayLike | null = null): NDArray | poly1d {
  if (!Number.isInteger(m)) throw new TypeError("m must be an integer");
  if (m < 0) throw new ValueError("Order of integral must be positive (see polyder)");
  let kk = k === null ? zeros([m]) : atleast1(k);
  if (len(kk) === 1 && m > 1) kk = multiply(kk.get(0), ones([m]));
  if (len(kk) < m) throw new ValueError("k must be a scalar or a rank-1 array of length 1 or >m.");
  const truepoly = p instanceof poly1d;
  const c = arr(p);
  if (m === 0) return ret(c, truepoly);
  const y = concatenate([divide(c, arange(len(c), 0, -1)), sl(kk, 0, 1)]);
  return ret(polyint(y, m - 1, sl(kk, 1)), truepoly);
}

/** np.polyder(p, m = 1): `m`-th derivative. */
export function polyder(p: poly1d, m?: number): poly1d;
export function polyder(p: ArrayLike, m?: number): NDArray;
export function polyder(p: PolyLike, m = 1): NDArray | poly1d {
  if (!Number.isInteger(m)) throw new TypeError("m must be an integer");
  if (m < 0) throw new ValueError("Order of derivative must be positive (see polyint)");
  const truepoly = p instanceof poly1d;
  const c = arr(p);
  const n = len(c) - 1;
  const val = m === 0 ? c : polyder(multiply(sl(c, null, -1), arange(n, 0, -1)), m - 1);
  return ret(val, truepoly);
}

export interface PolyfitOptions {
  /** Relative cutoff for small singular values (default `len(x) * eps`). */
  rcond?: number | null;
  /** Return `[c, residuals, rank, singularValues, rcond]`. */
  full?: boolean;
  /** Weights applied to the unsquared residuals. */
  w?: ArrayLike | null;
  /** Also return the covariance matrix: `[c, V]`. `"unscaled"` skips the residual scaling. */
  cov?: boolean | "unscaled";
}

const EPS: Record<string, number> = { float16: 0.0009765625, float32: 1.1920928955078125e-7, complex64: 1.1920928955078125e-7 };

/**
 * np.polyfit(x, y, deg, options): least-squares polynomial fit, coefficients
 * highest power first. `y` may be 2-D (one fit per column).
 */
export function polyfit(x: ArrayLike, y: ArrayLike, deg: number, options: PolyfitOptions = {}): NDArray | (NDArray | number)[] {
  if (!Number.isInteger(deg)) throw new TypeError("deg must be an integer");
  const order = deg + 1;
  const xa = plusZero(asarray(x));
  const ya = plusZero(asarray(y));
  if (deg < 0) throw new ValueError("expected deg >= 0");
  if (xa.ndim !== 1) throw new TypeError("expected 1D vector for x");
  if (xa.size === 0) throw new TypeError("expected non-empty vector for x");
  if (ya.ndim < 1 || ya.ndim > 2) throw new TypeError("expected 1D or 2D array for y");
  if (xa.shape[0] !== ya.shape[0]) throw new TypeError("expected x and y to have same length");
  const n = len(xa);
  const rcond = options.rcond ?? n * (EPS[xa.dtype.name] ?? 2.220446049250313e-16);
  let lhs = vander(xa, order);
  let rhs = ya;
  if (options.w != null) {
    const w = plusZero(asarray(options.w));
    if (w.ndim !== 1) throw new TypeError("expected a 1-d array for weights");
    if (w.shape[0] !== ya.shape[0]) throw new TypeError("expected w and y to have the same length");
    const col = w.reshape([n, 1]);
    lhs = multiply(lhs, col);
    rhs = multiply(rhs, rhs.ndim === 2 ? col : w);
  }
  const scale = sqrt(sum(multiply(lhs, lhs), { axis: 0 }));
  lhs = divide(lhs, scale);
  const r = lstsq(lhs, rhs, rcond);
  const c = divide(r.x.T, scale).T;
  if (r.rank !== order && !(options.full ?? false)) {
    process.emitWarning("Polyfit may be poorly conditioned", "RankWarning");
  }
  if (options.full ?? false) return [c, r.residuals, r.rank, r.s, rcond];
  if (options.cov) {
    const vbase = divide(inv(dot(lhs.T, lhs)), outer(scale, scale));
    let fac: NDArray | number = 1;
    if (options.cov !== "unscaled") {
      if (n <= order) {
        throw new ValueError("the number of data points must exceed order to scale the covariance matrix");
      }
      fac = divide(r.residuals, n - order);
    }
    return [c, ya.ndim === 1 ? multiply(vbase, fac) : multiply(vbase.reshape([order, order, 1]), fac)];
  }
  return c;
}

/** np.polyval(p, x): evaluates the polynomial at `x` (Horner's scheme; a `poly1d` x composes). */
export function polyval(p: PolyLike, x: poly1d): poly1d;
export function polyval(p: PolyLike, x: ArrayLike): NDArray;
export function polyval(p: PolyLike, x: PolyLike): NDArray | poly1d {
  const c = arr(p);
  const n = len(c);
  if (x instanceof poly1d) {
    let y = new poly1d([0]);
    for (let k = 0; k < n; k++) y = y.mul(x).add(c.get(k));
    return y;
  }
  const xa = asarray(x);
  let y = zerosLike(xa);
  for (let k = 0; k < n; k++) y = add(multiply(y, xa), c.get(k));
  return y;
}

function addSub(f: typeof add, a1: PolyLike, a2: PolyLike): NDArray | poly1d {
  const truepoly = anyPoly(a1, a2);
  const x = atleast1(a1);
  const y = atleast1(a2);
  const diff = len(y) - len(x);
  let val: NDArray;
  if (diff === 0) val = f(x, y);
  else if (diff > 0) val = f(concatenate([zeros([diff], { dtype: x.dtype }), x]), y);
  else val = f(x, concatenate([zeros([-diff], { dtype: y.dtype }), y]));
  return ret(val, truepoly);
}

/** np.polyadd(a1, a2): sum of two polynomials. */
export function polyadd(a1: poly1d, a2: PolyLike): poly1d;
export function polyadd(a1: PolyLike, a2: poly1d): poly1d;
export function polyadd(a1: ArrayLike, a2: ArrayLike): NDArray;
export function polyadd(a1: PolyLike, a2: PolyLike): NDArray | poly1d {
  return addSub(add, a1, a2);
}

/** np.polysub(a1, a2): difference `a1 - a2` of two polynomials. */
export function polysub(a1: poly1d, a2: PolyLike): poly1d;
export function polysub(a1: PolyLike, a2: poly1d): poly1d;
export function polysub(a1: ArrayLike, a2: ArrayLike): NDArray;
export function polysub(a1: PolyLike, a2: PolyLike): NDArray | poly1d {
  return addSub(subtract, a1, a2);
}

/** np.polymul(a1, a2): product of two polynomials (leading zeros are trimmed first). */
export function polymul(a1: poly1d, a2: PolyLike): poly1d;
export function polymul(a1: PolyLike, a2: poly1d): poly1d;
export function polymul(a1: ArrayLike, a2: ArrayLike): NDArray;
export function polymul(a1: PolyLike, a2: PolyLike): NDArray | poly1d {
  const truepoly = anyPoly(a1, a2);
  const val = convolve(new poly1d(a1).coeffs, new poly1d(a2).coeffs);
  return ret(val, truepoly);
}

/** np.polydiv(u, v): `[quotient, remainder]` of polynomial division. */
export function polydiv(u: poly1d, v: PolyLike): [poly1d, poly1d];
export function polydiv(u: PolyLike, v: poly1d): [poly1d, poly1d];
export function polydiv(u: ArrayLike, v: ArrayLike): [NDArray, NDArray];
export function polydiv(u: PolyLike, v: PolyLike): [NDArray, NDArray] | [poly1d, poly1d] {
  const truepoly = anyPoly(u, v);
  const a = plusZero(atleast1(u));
  const b = plusZero(atleast1(v));
  const dt = promoteTypes(a.dtype, b.dtype);
  const [q, r] = wrapNative(() => native.polydiv(a.astype(dt)._native, b.astype(dt)._native).map((h) => NDArray._wrap(h)));
  return truepoly ? [new poly1d(q!), new poly1d(r!)] : [q!, r!];
}

// ---------------- poly1d ----------------

/** Python `'{:.4g}'.format(q)`. */
function fmt4g(q: number): string {
  if (Number.isNaN(q)) return "nan";
  if (!Number.isFinite(q)) return q > 0 ? "inf" : "-inf";
  if (q === 0) return Object.is(q, -0) ? "-0" : "0";
  const [mant, expStr] = q.toExponential(3).split("e") as [string, string];
  const exp = Number(expStr);
  const strip = (s: string): string => (s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s);
  if (exp >= -4 && exp < 4) return strip(q.toFixed(3 - exp));
  return `${strip(mant)}e${exp < 0 ? "-" : "+"}${String(Math.abs(exp)).padStart(2, "0")}`;
}

/** NumPy's _raise_power: moves `**k` exponents to a line above. */
function raisePower(astr: string, wrap = 70): string {
  const re = /\*\*([0-9]*)/g;
  let n = 0;
  let line1 = "";
  let line2 = "";
  let output = " ";
  for (const mat of astr.matchAll(re)) {
    const power = mat[1]!;
    const partstr = astr.slice(n, mat.index);
    n = mat.index + mat[0].length;
    const toadd2 = partstr + " ".repeat(power.length - 1);
    const toadd1 = " ".repeat(partstr.length - 1) + power;
    if (line2.length + toadd2.length > wrap || line1.length + toadd1.length > wrap) {
      output += line1 + "\n" + line2 + "\n ";
      line1 = toadd1;
      line2 = toadd2;
    } else {
      line2 += toadd2;
      line1 += toadd1;
    }
  }
  output += line1 + "\n" + line2;
  return output + astr.slice(n);
}

export interface Poly1dOptions {
  /** The input values are roots, not coefficients. */
  r?: boolean;
  /** Variable name used by `toString()` (default "x"). */
  variable?: string | null;
}

/**
 * np.poly1d: a one-dimensional polynomial. Python operators are methods:
 * `p.call(x)` is `p(x)`, `add sub mul div pow neg equals`, `get(k)`/`set(k, v)`
 * are `p[k]`/`p[k] = v` (coefficient of x**k).
 */
// eslint-disable-next-line @typescript-eslint/naming-convention
export class poly1d implements Iterable<number | boolean | Complex> {
  #coeffs: NDArray;
  readonly variable: string;

  constructor(cOrR: PolyLike, options: Poly1dOptions | boolean = {}) {
    const o: Poly1dOptions = typeof options === "boolean" ? { r: options } : options;
    if (cOrR instanceof poly1d) {
      this.#coeffs = cOrR.#coeffs;
      this.variable = o.variable ?? cOrR.variable;
      return;
    }
    let c: NDArray = o.r ? asarray(poly(cOrR) as ArrayLike) : asarray(cOrR);
    if (c.ndim === 0) c = c.reshape([1]);
    if (c.ndim > 1) throw new ValueError("Polynomial must be 1d only.");
    c = trimZeros(c, "f");
    if (len(c) === 0) c = zeros([1], { dtype: c.dtype });
    this.#coeffs = c;
    this.variable = o.variable ?? "x";
  }

  /** Coefficients, highest power first. */
  get coeffs(): NDArray {
    return this.#coeffs;
  }
  get c(): NDArray {
    return this.#coeffs;
  }
  get coef(): NDArray {
    return this.#coeffs;
  }
  get coefficients(): NDArray {
    return this.#coeffs;
  }
  /** Degree of the polynomial. */
  get order(): number {
    return len(this.#coeffs) - 1;
  }
  get o(): number {
    return this.order;
  }
  /** Python `len(p)`: the order. */
  get length(): number {
    return this.order;
  }
  get roots(): NDArray {
    return roots(this.#coeffs);
  }
  get r(): NDArray {
    return this.roots;
  }

  /** `p(x)`: evaluates at `x` (a `poly1d` argument composes). */
  call(x: poly1d): poly1d;
  call(x: ArrayLike): NDArray;
  call(x: PolyLike): NDArray | poly1d {
    return x instanceof poly1d ? polyval(this.#coeffs, x) : polyval(this.#coeffs, x);
  }
  neg(): poly1d {
    return new poly1d(negative(this.#coeffs));
  }
  add(other: PolyLike): poly1d {
    return polyadd(this, new poly1d(other));
  }
  sub(other: PolyLike): poly1d {
    return polysub(this, new poly1d(other));
  }
  mul(other: PolyLike): poly1d {
    if (isScalar(other)) return new poly1d(multiply(this.#coeffs, other));
    return polymul(this, new poly1d(other));
  }
  /** `p / scalar` scales the coefficients; `p / q` returns `[quotient, remainder]`. */
  div(other: Scalar): poly1d;
  div(other: PolyLike): [poly1d, poly1d];
  div(other: PolyLike): poly1d | [poly1d, poly1d] {
    if (isScalar(other)) return new poly1d(divide(this.#coeffs, other));
    return polydiv(this, new poly1d(other));
  }
  pow(val: number): poly1d {
    if (!Number.isInteger(val) || val < 0) throw new ValueError("Power to non-negative integers only.");
    let res: NDArray = asarray([1]);
    for (let k = 0; k < val; k++) res = polymul(this.#coeffs, res);
    return new poly1d(res);
  }
  equals(other: poly1d): boolean {
    if (len(this.#coeffs) !== len(other.#coeffs)) return false;
    return all(equal(this.#coeffs, other.#coeffs)).item() === true;
  }
  notEquals(other: poly1d): boolean {
    return !this.equals(other);
  }
  /** `p[power]`: coefficient of x**power (0 outside the range). */
  get(power: number): number | boolean | Complex {
    if (power > this.order || power < 0) return zeros([], { dtype: this.#coeffs.dtype }).item();
    return this.#coeffs.item(this.order - power);
  }
  /** `p[power] = value`; extends the coefficients with zeros when needed. */
  set(power: number, value: Scalar): void {
    if (power < 0) throw new ValueError("Does not support negative powers.");
    let ind = this.order - power;
    if (power > this.order) {
      this.#coeffs = concatenate([zeros([power - this.order], { dtype: this.#coeffs.dtype }), this.#coeffs]);
      ind = 0;
    } else {
      this.#coeffs = this.#coeffs.copy();
    }
    this.#coeffs.set([ind], value as NestedArray);
  }
  integ(m = 1, k: ArrayLike = 0): poly1d {
    return new poly1d(polyint(this.#coeffs, m, k));
  }
  deriv(m = 1): poly1d {
    return new poly1d(polyder(this.#coeffs, m));
  }
  *[Symbol.iterator](): Iterator<number | boolean | Complex> {
    for (let k = 0; k < len(this.#coeffs); k++) yield this.#coeffs.item(k);
  }

  /** NumPy `str(p)`: the polynomial with exponents on a line above. */
  toString(): string {
    const all0 = this.#coeffs.toArray() as unknown[];
    const isZero = (v: unknown): boolean =>
      isComplexLike(v) ? v.re === 0 && (v.im ?? 0) === 0 : v === 0 || v === false || v === 0n;
    const firstNz = all0.findIndex((v) => !isZero(v));
    const coeffs = firstNz < 0 ? [] : all0.slice(firstNz);
    const N = coeffs.length - 1;
    const v = this.variable;
    let thestr = "0";
    coeffs.forEach((coeff, k) => {
      let coefstr: string;
      if (isComplexLike(coeff) && (coeff.im ?? 0) !== 0) {
        coefstr = coeff.re === 0 ? `${fmt4g(coeff.im!)}j` : `(${fmt4g(coeff.re)} + ${fmt4g(coeff.im!)}j)`;
      } else {
        coefstr = fmt4g(isComplexLike(coeff) ? coeff.re : Number(coeff));
      }
      const power = N - k;
      let newstr: string;
      if (power === 0) newstr = coefstr !== "0" ? coefstr : k === 0 ? "0" : "";
      else if (power === 1) newstr = coefstr === "0" ? "" : `${coefstr} ${v}`;
      else newstr = coefstr === "0" ? "" : `${coefstr} ${v}**${power}`;
      if (k > 0) {
        if (newstr !== "") thestr = newstr.startsWith("-") ? `${thestr} - ${newstr.slice(1)}` : `${thestr} + ${newstr}`;
      } else {
        thestr = newstr;
      }
    });
    return raisePower(thestr);
  }
}
