// P15 np.polynomial: series modules and the six convenience classes
// (numpy.polynomial), D-182. Series arithmetic is the native kernel
// `native/core/p15_polynomial.cpp`; this file validates arguments, handles
// domain/window mapping and printing, and does fitting (np.linalg.lstsq) and
// root finding (np.linalg.eigvals of the companion matrix).
import { nativeModule, type NativeNDArray } from "./addon.js";
import { Complex, isComplexLike } from "./complex.js";
import { asarray, linspace as npLinspace } from "./creation.js";
import { DTypeError, ValueError, wrapNative } from "./errors.js";
import { take } from "./indexing.js";
import { eigvals, lstsq } from "./linalg.js";
import { NDArray } from "./ndarray.js";
import { all, equal } from "./p05_compare.js";
import { flip } from "./p06_reorder.js";
import { sort } from "./p09.js";
import { arrayRepr, pyFloat, scalarStr } from "./p15_format.js";
import { sum } from "./reduce.js";
import { abs, add, divide, multiply, sqrt, subtract, type ArrayLike } from "./ufunc.js";

interface P15PolyNative {
  poly(op: string, basis: string, ...args: unknown[]): NativeNDArray & NativeNDArray[];
}
const native = nativeModule<P15PolyNative>("p15");

type BasisName = "polynomial" | "chebyshev" | "legendre" | "laguerre" | "hermite" | "hermite_e";
type Scalar = number | Complex;

const wrap = (r: NativeNDArray): NDArray => NDArray._wrap(r);
const call = (op: string, b: BasisName, ...args: unknown[]): NDArray =>
  wrapNative(() => wrap(native.poly(op, b, ...args)));

/** `np.array(c, ndmin=1)` as float64/complex128 (numera's series dtypes, D-182). */
function series(c: ArrayLike): NDArray {
  let a = c instanceof NDArray ? c : asarray(c as never);
  if (a.ndim === 0) a = a.reshape([1]);
  if (a.dtype.kind === "c") return a.dtype.name === "complex128" ? a : a.astype("complex128");
  return a.dtype.name === "float64" ? a : a.astype("float64");
}

const scalarArr = (v: ArrayLike): NDArray => series(v);

function asInt(v: number, what: string): number {
  if (!Number.isInteger(v)) throw new DTypeError(`${what} must be an integer`);
  return v;
}

// ---- module-level series functions (numpy.polynomial.<module>) ----

export interface IntegOptions {
  /** Order of integration (default 1). */
  m?: number;
  /** Integration constants (default all zero). */
  k?: ArrayLike;
  /** Lower bound (default 0). */
  lbnd?: number | Complex;
  /** Scale applied after each integration (default 1). */
  scl?: number;
}

export interface FitOptions {
  /** Cut-off ratio for small singular values (default `len(x) * eps`). */
  rcond?: number | null;
  /** Weights applied to the y values. */
  w?: ArrayLike | null;
}

export interface FitFullResult {
  coef: NDArray;
  residuals: NDArray;
  rank: number;
  singularValues: NDArray;
  rcond: number;
}

function vanderOf(b: BasisName, x: ArrayLike, deg: number): NDArray {
  const xa = asarray(x as never);
  const xs = xa.dtype.kind === "c" ? xa.astype("complex128") : xa.astype("float64");
  return call("vander", b, xs._native, asInt(deg, "deg"));
}

/** polyutils._fit: least-squares fit with column scaling. */
function fitSeries(
  b: BasisName,
  x: ArrayLike,
  y: ArrayLike,
  deg: number | readonly number[],
  opts: FitOptions,
  full: boolean,
): NDArray | FitFullResult {
  const xa = asarray(x as never).astype("float64");
  let ya = asarray(y as never);
  ya = ya.dtype.kind === "c" ? ya.astype("complex128") : ya.astype("float64");
  const degs = typeof deg === "number" ? null : [...deg].sort((p, q) => p - q);
  if (degs && degs.length === 0) throw new DTypeError("deg must be an int or non-empty 1-D array of int");
  for (const d of degs ?? [deg as number]) {
    if (!Number.isInteger(d)) throw new DTypeError("deg must be an int or non-empty 1-D array of int");
    if (d < 0) throw new ValueError("expected deg >= 0");
  }
  if (xa.ndim !== 1) throw new DTypeError("expected 1D vector for x");
  if (xa.size === 0) throw new DTypeError("expected non-empty vector for x");
  if (ya.ndim < 1 || ya.ndim > 2) throw new DTypeError("expected 1D or 2D array for y");
  if (xa.shape[0] !== ya.shape[0]) throw new DTypeError("expected x and y to have same length");
  const lmax = degs ? (degs[degs.length - 1] as number) : (deg as number);
  const order = degs ? degs.length : lmax + 1;
  let van = vanderOf(b, xa, lmax);
  if (degs) van = take(van, degs, { axis: 1 });
  let lhs = van.T;
  let rhs = ya.T;
  if (opts.w !== undefined && opts.w !== null) {
    const w = asarray(opts.w as never).astype("float64");
    if (w.ndim !== 1) throw new DTypeError("expected 1D vector for w");
    if (w.shape[0] !== xa.shape[0]) throw new DTypeError("expected x and w to have same length");
    lhs = multiply(lhs, w);
    rhs = multiply(rhs, w);
  }
  const rcond = opts.rcond ?? (xa.shape[0] as number) * 2 ** -52;
  const mag = abs(lhs);
  const scl = sqrt(sum(multiply(mag, mag), { axis: 1 }));
  const sv = (scl.toArray() as number[]).map((v) => (v === 0 ? 1 : v));
  const sclArr = asarray(sv);
  const r = lstsq(divide(lhs.T, sclArr), rhs.T, rcond);
  let c = divide(r.x.T, sclArr).T;
  if (degs) {
    const rows = c.toArray() as unknown[];
    const zero = c.dtype.kind === "c" ? new Complex(0, 0) : 0;
    const filler = (): unknown => (c.ndim === 2 ? (rows[0] as unknown[]).map(() => zero) : zero);
    const cc: unknown[] = Array.from({ length: lmax + 1 }, () => filler());
    degs.forEach((d, i) => (cc[d] = rows[i]));
    c = asarray(cc as never, { dtype: c.dtype.name });
  }
  if (r.rank !== order && !full) process.emitWarning("The fit may be poorly conditioned", "RankWarning");
  if (full) return { coef: c, residuals: r.residuals, rank: r.rank, singularValues: r.s, rcond };
  return c;
}

const ROOTS_REAL: Record<BasisName, boolean> = {
  polynomial: true,
  chebyshev: false,
  legendre: false,
  laguerre: true,
  hermite: true,
  hermite_e: false,
};

function rootsSeries(b: BasisName, c: ArrayLike): NDArray {
  const s = call("add", b, series(c)._native, series(0)._native); // as_series (trimmed copy)
  const n = s.size;
  const cv = s.toArray() as Scalar[];
  if (n < 2) return asarray([], { dtype: s.dtype.name });
  if (n === 2) {
    const [c0, c1] = cv as [Scalar, Scalar];
    const q = divide(asarray([c0] as never), asarray([c1] as never));
    if (b === "laguerre") return add(1, q);
    if (b === "hermite") return multiply(-0.5, q);
    return multiply(-1, q);
  }
  let m = call("companion", b, s._native);
  if (b !== "polynomial") m = flip(m);
  const r = sort(eigvals(m));
  if (ROOTS_REAL[b] && s.dtype.kind !== "c" && (r.imag.toArray() as number[]).every((v) => v === 0)) return r.real.copy();
  return r;
}

const LINE = (b: BasisName, off: Scalar, scl: Scalar): NDArray => {
  const isZero = (v: Scalar) => (v instanceof Complex ? v.re === 0 && v.im === 0 : v === 0);
  if (isZero(scl)) return asarray([off] as never);
  const o = asarray([off] as never), s = asarray([scl] as never);
  if (b === "hermite") return asarray([o.item(0), divide(s, 2).item(0)] as never);
  if (b === "laguerre") return asarray([add(o, s).item(0), multiply(-1, s).item(0)] as never);
  return asarray([off, scl] as never);
};

/** The `<prefix><op>` functions of one numpy.polynomial module. */
function makeModule(b: BasisName, prefix: string, maxpowerDefault: number | null) {
  const two = (op: string) => (c1: ArrayLike, c2: ArrayLike): NDArray => call(op, b, series(c1)._native, series(c2)._native);
  const one = (op: string) => (c: ArrayLike): NDArray => call(op, b, series(c)._native);
  const f = {
    add: two("add"),
    sub: two("sub"),
    mul: two("mul"),
    mulx: one("mulx"),
    div: (c1: ArrayLike, c2: ArrayLike): [NDArray, NDArray] => {
      const r = wrapNative(() => native.poly("div", b, series(c1)._native, series(c2)._native));
      return [wrap(r[0] as NativeNDArray), wrap(r[1] as NativeNDArray)];
    },
    pow: (c: ArrayLike, pow: number, maxpower: number | null = maxpowerDefault): NDArray => {
      if (!Number.isInteger(pow) || pow < 0) throw new ValueError("Power must be a non-negative integer.");
      if (maxpower !== null && pow > maxpower) throw new ValueError("Power is too large");
      return call("pow", b, series(c)._native, pow);
    },
    val: (x: ArrayLike, c: ArrayLike): NDArray => {
      const xa = asarray(x as never);
      const xs = xa.dtype.kind === "c" ? xa.astype("complex128") : xa.astype("float64");
      return call("val", b, xs._native, series(c)._native);
    },
    der: (c: ArrayLike, m = 1, scl = 1): NDArray => call("der", b, series(c)._native, asInt(m, "the order of derivation"), scl),
    int: (c: ArrayLike, opts: IntegOptions = {}): NDArray => {
      const { m = 1, k = [], lbnd = 0, scl = 1 } = opts;
      const kk = asarray((typeof k === "number" || isComplexLike(k) ? [k] : k) as never);
      const ks = kk.size === 0 ? asarray([] as number[]) : series(kk);
      return call("int", b, series(c)._native, asInt(m, "the order of integration"), ks._native, scalarArr(lbnd as ArrayLike)._native, scl);
    },
    vander: (x: ArrayLike, deg: number): NDArray => vanderOf(b, x, deg),
    companion: one("companion"),
    fromroots: (roots: ArrayLike): NDArray => {
      const r = asarray(roots as never);
      const rs = r.ndim === 0 ? r.reshape([1]) : r;
      return call("fromroots", b, (rs.dtype.kind === "c" ? rs.astype("complex128") : rs.astype("float64"))._native);
    },
    roots: (c: ArrayLike): NDArray => rootsSeries(b, c),
    fit: (x: ArrayLike, y: ArrayLike, deg: number | readonly number[], opts: FitOptions = {}): NDArray =>
      fitSeries(b, x, y, deg, opts, false) as NDArray,
    trim: (c: ArrayLike, tol = 0): NDArray => trimcoef(c, tol),
    line: (off: Scalar, scl: Scalar): NDArray => LINE(b, off, scl),
  };
  const domain = b === "laguerre" ? [0, 1] : [-1, 1];
  const xcoef: number[] = b === "laguerre" ? [1, -1] : b === "hermite" ? [0, 0.5] : [0, 1];
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(f)) out[prefix + k] = v;
  out[prefix + "domain"] = asarray(domain, { dtype: "float64" });
  out[prefix + "zero"] = asarray([0], { dtype: "float64" });
  out[prefix + "one"] = asarray([1], { dtype: "float64" });
  out[prefix + "x"] = asarray(xcoef, { dtype: "float64" });
  if (b !== "polynomial") {
    const short = prefix === "herme" ? "herme" : prefix;
    out[`${short}2poly`] = (c: ArrayLike) => call("topower", b, series(c)._native);
    out[`poly2${short}`] = (c: ArrayLike) => call("frompower", b, series(c)._native);
  } else {
    out.polyvalfromroots = (x: ArrayLike, r: ArrayLike): NDArray => {
      const xa = asarray(x as never);
      const rv = series(r).toArray() as Scalar[];
      let acc: NDArray | null = null;
      for (const root of rv) {
        const t = subtract(xa, root instanceof Complex ? asarray([root]).reshape([]) : root);
        acc = acc ? multiply(acc, t) : t;
      }
      return acc ?? asarray(1);
    };
  }
  return { fns: f, module: out };
}

/** polyutils.trimcoef: drop trailing coefficients with `|c| <= tol`. */
export function trimcoef(c: ArrayLike, tol = 0): NDArray {
  if (tol < 0) throw new ValueError("tol must be non-negative");
  const s = call("add", "polynomial", series(c)._native, series(0)._native);
  const mags = (abs(s).toArray() as number[]).map((v) => v > tol);
  const last = mags.lastIndexOf(true);
  if (last < 0) return multiply(s.slice([0, 1]), 0);
  return s.slice([0, last + 1]).copy();
}

// ---- printing (polyutils.format_float, ABCPolyBase._generate_string) ----

let useUnicode = true;

/** numpy.polynomial.set_default_printstyle("unicode" | "ascii"). */
export function setDefaultPrintstyle(style: "unicode" | "ascii"): void {
  if (style !== "unicode" && style !== "ascii") {
    throw new ValueError(`Unsupported format string '${String(style)}'. Valid options are 'ascii' and 'unicode'`);
  }
  useUnicode = style === "unicode";
}

/** Shortest round-trip digits of a float64, then at most `prec` fractional / significant digits. */
function formatFloat(x: number, parens = false): string {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x < 0 ? "-inf" : "inf";
  const a = Math.abs(x);
  const sign = x < 0 || Object.is(x, -0) ? "-" : "";
  if (x !== 0 && (a >= 1e8 || a < 1e-4)) {
    let [m = "0", e = "0"] = a.toExponential().split("e");
    if (m.replace(".", "").length - 1 > 8) [m = "0", e = "0"] = a.toExponential(8).split("e");
    if (!m.includes(".")) m += ".0";
    m = m.replace(/(\.\d*?[1-9])0+$/, "$1").replace(/\.0+$/, ".0");
    const ev = Number(e);
    const s = `${sign}${m}e${ev < 0 ? "-" : "+"}${String(Math.abs(ev)).padStart(2, "0")}`;
    return parens ? `(${s})` : s;
  }
  const p = pyFloat(a);
  const frac = (p.split(".")[1] ?? "").length;
  let s = frac <= 8 ? p : a.toFixed(8).replace(/0+$/, "");
  if (s.endsWith(".")) s += "0";
  return sign + s;
}

const formatScalar = (v: Scalar, parens = false): string =>
  v instanceof Complex ? scalarStr(v, "complex128") : formatFloat(v, parens);

const SUP: Record<string, string> = { 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
const SUB: Record<string, string> = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };
const tr = (s: string, m: Record<string, string>) => [...s].map((ch) => m[ch] ?? ch).join("");

// ---- classes (numpy.polynomial._polybase.ABCPolyBase) ----

type Other = ABCPolyBase | ArrayLike;

/** Options of the class constructors and factory methods. */
export interface SeriesOptions {
  domain?: ArrayLike | null;
  window?: ArrayLike | null;
  symbol?: string;
}

export interface ClassFitOptions extends FitOptions {
  /** Domain of the result; default the range of `x`; `[]` means the class default. */
  domain?: ArrayLike | null;
  window?: ArrayLike | null;
  symbol?: string;
}

type SeriesClass<T extends ABCPolyBase = ABCPolyBase> = {
  new (coef: ArrayLike, domain?: ArrayLike | null, window?: ArrayLike | null, symbol?: string): T;
  readonly basisName: string | null;
  readonly defaultDomain: NDArray;
  readonly defaultWindow: NDArray;
  readonly kernel: BasisName;
  readonly maxpower: number;
  identity(domain?: ArrayLike | null, window?: ArrayLike | null, symbol?: string): ABCPolyBase;
};

function pair(v: ArrayLike, what: string): NDArray {
  const a = series(v);
  if (a.ndim !== 1) throw new ValueError("Coefficient array is not 1-d");
  if (a.size !== 2) throw new ValueError(`${what} has wrong number of elements.`);
  return a;
}

function asScalars(a: NDArray): Scalar[] {
  return a.toArray() as Scalar[];
}

/** polyutils.mapparms(old, new) -> [off, scl]. */
function mapparms(oldD: NDArray, newD: NDArray): [Scalar, Scalar] {
  const [o0, o1] = [oldD.slice([0, 1]), oldD.slice([1, 2])];
  const [n0, n1] = [newD.slice([0, 1]), newD.slice([1, 2])];
  const oldlen = subtract(o1, o0);
  const newlen = subtract(n1, n0);
  const off = divide(subtract(multiply(o1, n0), multiply(o0, n1)), oldlen);
  const scl = divide(newlen, oldlen);
  return [off.item(0) as Scalar, scl.item(0) as Scalar];
}

const sc = (v: Scalar): NDArray => asarray([v] as never).reshape([]);

/** Base class of Polynomial, Chebyshev, Legendre, Laguerre, Hermite, HermiteE. */
export abstract class ABCPolyBase {
  static readonly maxpower: number = 100;
  /** Series coefficients, lowest degree first (float64 or complex128). */
  readonly coef: NDArray;
  readonly domain: NDArray;
  readonly window: NDArray;
  readonly symbol: string;

  constructor(coef: ArrayLike, domain: ArrayLike | null = null, window: ArrayLike | null = null, symbol = "x") {
    const c = series(coef);
    if (c.ndim !== 1) throw new ValueError("Coefficient array is not 1-d");
    if (c.size === 0) throw new ValueError("Coefficient array is empty");
    this.coef = c;
    const k = this.cls;
    this.domain = domain === null || domain === undefined ? k.defaultDomain.copy() : pair(domain, "Domain");
    this.window = window === null || window === undefined ? k.defaultWindow.copy() : pair(window, "Window");
    if (typeof symbol !== "string" || symbol.length === 0) throw new DTypeError("Symbol must be a non-empty string");
    if (!/^[\p{L}_][\p{L}\p{N}_]*$/u.test(symbol)) throw new ValueError("Symbol string must be a valid Python identifier");
    this.symbol = symbol;
  }

  protected get cls(): SeriesClass<this> {
    return this.constructor as unknown as SeriesClass<this>;
  }

  private make(coef: ArrayLike): this {
    return new this.cls(coef, this.domain, this.window, this.symbol);
  }

  private otherCoef(other: Other): NDArray {
    if (other instanceof ABCPolyBase) {
      if (!(other instanceof this.cls)) throw new DTypeError("Polynomial types differ");
      if (!isTrue(all(equal(this.domain, other.domain)))) throw new DTypeError("Domains differ");
      if (!isTrue(all(equal(this.window, other.window)))) throw new DTypeError("Windows differ");
      if (this.symbol !== other.symbol) throw new ValueError("Polynomial symbols differ");
      return other.coef;
    }
    return series(other);
  }

  private op(name: string, a: NDArray, b: NDArray): NDArray {
    return call(name, this.cls.kernel, a._native, b._native);
  }

  /** Domain of the class (class default). */
  static get domain(): NDArray {
    return (this as unknown as SeriesClass).defaultDomain.copy();
  }
  static get window(): NDArray {
    return (this as unknown as SeriesClass).defaultWindow.copy();
  }

  /** Evaluates the series at `x` (array-like, or another series to compose with). */
  call(arg: ArrayLike): NDArray;
  call<T extends ABCPolyBase>(arg: T): T;
  call(arg: ArrayLike | ABCPolyBase): NDArray | ABCPolyBase {
    const [off, scl] = this.mapparms();
    if (arg instanceof ABCPolyBase) {
      const mapped = arg.mul(sc(scl)).add(sc(off));
      return clenshawSeries(this.cls.kernel, mapped, asScalars(this.coef));
    }
    const x = asarray(arg as never);
    const xm = add(off instanceof Complex ? sc(off) : off, multiply(scl instanceof Complex ? sc(scl) : scl, x));
    const xs = xm.dtype.kind === "c" ? xm.astype("complex128") : xm.astype("float64");
    return call("val", this.cls.kernel, xs._native, this.coef._native);
  }

  get length(): number {
    return this.coef.size;
  }
  [Symbol.iterator](): Iterator<Scalar> {
    return (asScalars(this.coef) as Scalar[])[Symbol.iterator]();
  }

  neg(): this {
    return this.make(multiply(this.coef, -1));
  }
  pos(): this {
    return this;
  }
  add(other: Other): this {
    return this.make(this.op("add", this.coef, this.otherCoef(other)));
  }
  sub(other: Other): this {
    return this.make(this.op("sub", this.coef, this.otherCoef(other)));
  }
  /** `other - this` (Python `__rsub__`). */
  rsub(other: ArrayLike): this {
    return this.make(this.op("sub", series(other), this.coef));
  }
  mul(other: Other): this {
    return this.make(this.op("mul", this.coef, this.otherCoef(other)));
  }
  /** Division by a scalar (Python `/`; same as floordiv). */
  truediv(other: number | Complex): this {
    if (typeof other !== "number" && !(other instanceof Complex)) {
      throw new DTypeError("unsupported types for true division");
    }
    return this.floordiv(other);
  }
  floordiv(other: Other): this {
    return this.divmod(other)[0];
  }
  mod(other: Other): this {
    return this.divmod(other)[1];
  }
  divmod(other: Other): [this, this] {
    const r = wrapNative(() => native.poly("div", this.cls.kernel, this.coef._native, this.otherCoef(other)._native));
    return [this.make(wrap(r[0] as NativeNDArray)), this.make(wrap(r[1] as NativeNDArray))];
  }
  pow(n: number): this {
    if (!Number.isInteger(n) || n < 0) throw new ValueError("Power must be a non-negative integer.");
    if (n > this.cls.maxpower) throw new ValueError("Power is too large");
    return this.make(call("pow", this.cls.kernel, this.coef._native, n));
  }
  /** Python `==`: same class, domain, window, symbol and coefficients. */
  equals(other: unknown): boolean {
    return (
      other instanceof this.cls &&
      isTrue(all(equal(this.domain, other.domain))) &&
      isTrue(all(equal(this.window, other.window))) &&
      this.coef.size === other.coef.size &&
      isTrue(all(equal(this.coef, other.coef))) &&
      this.symbol === other.symbol
    );
  }
  hasSamecoef(other: ABCPolyBase): boolean {
    return this.coef.size === other.coef.size && isTrue(all(equal(this.coef, other.coef)));
  }
  hasSamedomain(other: ABCPolyBase): boolean {
    return isTrue(all(equal(this.domain, other.domain)));
  }
  hasSamewindow(other: ABCPolyBase): boolean {
    return isTrue(all(equal(this.window, other.window)));
  }
  hasSametype(other: unknown): boolean {
    return other instanceof this.cls;
  }

  copy(): this {
    return this.make(this.coef.copy());
  }
  degree(): number {
    return this.coef.size - 1;
  }
  cutdeg(deg: number): this {
    return this.truncate(deg + 1);
  }
  trim(tol = 0): this {
    return this.make(trimcoef(this.coef, tol));
  }
  truncate(size: number): this {
    if (!Number.isInteger(size) || size < 1) throw new ValueError("size must be a positive integer");
    return this.make(size >= this.coef.size ? this.coef : this.coef.slice([0, size]));
  }
  /** Converts to another class / domain / window: `this(kind.identity(domain, window))`. */
  convert<K extends ABCPolyBase = this>(
    domain: ArrayLike | null = null,
    kind: SeriesClass<K> | null = null,
    window: ArrayLike | null = null,
  ): K {
    const k = (kind ?? this.cls) as SeriesClass<K>;
    return this.call(k.identity(domain ?? k.defaultDomain, window ?? k.defaultWindow, this.symbol) as K);
  }
  /** [off, scl] of the linear map from domain to window. */
  mapparms(): [Scalar, Scalar] {
    return mapparms(this.domain, this.window);
  }
  /** Integral (`m` times, constants `k`, lower bound `lbnd` in the domain). */
  integ(m = 1, k: ArrayLike = [], lbnd: number | null = null): this {
    const [off, scl] = this.mapparms();
    const lb = lbnd === null ? 0 : add(off instanceof Complex ? sc(off) : off, multiply(scl instanceof Complex ? sc(scl) : scl, lbnd)).item();
    const inv = divide(1, scl instanceof Complex ? sc(scl) : scl).item();
    if (inv instanceof Complex) throw new DTypeError("complex domain scale is not supported by integ");
    const kk = asarray((typeof k === "number" || isComplexLike(k) ? [k] : k) as never);
    const ks = kk.size === 0 ? asarray([] as number[]) : series(kk);
    return this.make(
      call("int", this.cls.kernel, this.coef._native, asInt(m, "the order of integration"), ks._native, series(lb as ArrayLike)._native, inv as number),
    );
  }
  /** Derivative of order `m`. */
  deriv(m = 1): this {
    const [, scl] = this.mapparms();
    if (scl instanceof Complex) throw new DTypeError("complex domain scale is not supported by deriv");
    return this.make(call("der", this.cls.kernel, this.coef._native, asInt(m, "the order of derivation"), scl));
  }
  /** Roots, mapped back to the domain. */
  roots(): NDArray {
    const r = rootsSeries(this.cls.kernel, this.coef);
    const [off, scl] = mapparms(this.window, this.domain);
    return add(off instanceof Complex ? sc(off) : off, multiply(scl instanceof Complex ? sc(scl) : scl, r));
  }
  /** `[x, this(x)]` for `n` equally spaced points of `domain` (default the series' domain). */
  linspace(n = 100, domain: ArrayLike | null = null): [NDArray, NDArray] {
    const d = asScalars(domain === null ? this.domain : series(domain)) as number[];
    const x = npLinspace(d[0] as number, d[1] as number, n);
    return [x, this.call(x)];
  }

  /** Python `str()` (unicode, or ascii after `setDefaultPrintstyle("ascii")`). */
  toString(): string {
    return this.generate(useUnicode);
  }
  /** Python `format(p, "ascii" | "unicode")`. */
  format(style: "" | "ascii" | "unicode" = ""): string {
    if (style === "") return this.toString();
    if (style !== "ascii" && style !== "unicode") {
      throw new ValueError(`Unsupported format string '${String(style)}'. Valid options are 'ascii' and 'unicode'`);
    }
    return this.generate(style === "unicode");
  }
  /** Python `repr()`. */
  repr(): string {
    const inner = (a: NDArray) => arrayRepr(a).slice(6, -1);
    return `${this.constructor.name}(${inner(this.coef)}, domain=${inner(this.domain)}, window=${inner(this.window)}, symbol='${this.symbol}')`;
  }

  protected termUnicode(i: string, arg: string): string {
    return `·${this.cls.basisName}${tr(i, SUB)}(${arg})`;
  }
  protected termAscii(i: string, arg: string): string {
    return ` ${this.cls.basisName}_${i}(${arg})`;
  }

  private generate(unicode: boolean): string {
    const lineWidth = 75;
    const coefs = asScalars(this.coef);
    let out = formatScalar(coefs[0] as Scalar);
    const [off, scale] = this.mapparms();
    const isNum = (v: Scalar, k: number) => typeof v === "number" && v === k;
    let term: string, parens: boolean;
    if (isNum(off, 0) && isNum(scale, 1)) {
      term = this.symbol;
      parens = false;
    } else if (isNum(scale, 1)) {
      term = `${formatScalar(off)} + ${this.symbol}`;
      parens = true;
    } else if (isNum(off, 0)) {
      term = `${formatScalar(scale)}${this.symbol}`;
      parens = true;
    } else {
      term = `${formatScalar(off)} + ${formatScalar(scale)}${this.symbol}`;
      parens = true;
    }
    if (parens) term = `(${term})`;
    const rest = coefs.slice(1);
    rest.forEach((c, i) => {
      out += " ";
      const power = String(i + 1);
      let next: string;
      if (c instanceof Complex) next = `+ ${scalarStr(c, "complex128")}`;
      else if (c >= 0 || Number.isNaN(c)) next = Number.isNaN(c) ? `- ${formatFloat(c, true)}` : `+ ${formatFloat(c, true)}`;
      else next = `- ${formatFloat(-c, true)}`;
      next += unicode ? this.termUnicode(power, term) : this.termAscii(power, term);
      let lineLen = (out.split("\n").at(-1) ?? "").length + next.length;
      if (i < rest.length - 1) lineLen += 2;
      if (lineLen >= lineWidth) next = next.replace(" ", "\n");
      out += next;
    });
    return out;
  }

  /** Least-squares fit of degree `deg` to points `(x, y)`; domain defaults to the range of `x`. */
  static fit<T extends ABCPolyBase>(
    this: SeriesClass<T>,
    x: ArrayLike,
    y: ArrayLike,
    deg: number | readonly number[],
    opts: ClassFitOptions = {},
  ): T {
    return fitClass(this, x, y, deg, opts, false) as T;
  }
  /** As `fit`, also returning `[residuals, rank, singularValues, rcond]` diagnostics. */
  static fitFull<T extends ABCPolyBase>(
    this: SeriesClass<T>,
    x: ArrayLike,
    y: ArrayLike,
    deg: number | readonly number[],
    opts: ClassFitOptions = {},
  ): { series: T; residuals: NDArray; rank: number; singularValues: NDArray; rcond: number } {
    return fitClass(this, x, y, deg, opts, true) as never;
  }
  /** Series with the given roots (domain `[]` = class default, `null` = range of the roots). */
  static fromroots<T extends ABCPolyBase>(this: SeriesClass<T>, roots: ArrayLike, opts: SeriesOptions = {}): T {
    const r0 = asarray(roots as never);
    const r = r0.ndim === 0 ? r0.reshape([1]) : r0;
    const domain = resolveDomain(this, opts.domain === undefined ? [] : opts.domain, r);
    const window = opts.window ?? this.defaultWindow;
    const [off, scl] = mapparms(domain, series(window));
    const rnew = add(off instanceof Complex ? sc(off) : off, multiply(scl instanceof Complex ? sc(scl) : scl, r.astype(r.dtype.kind === "c" ? "complex128" : "float64")));
    const base = call("fromroots", this.kernel, rnew._native);
    const p = asarray(scl as never);
    let denom = asarray(1);
    for (let i = 0; i < r.size; i++) denom = multiply(denom, p);
    return new this(divide(base, denom), domain, window, opts.symbol ?? "x");
  }
  /** The identity `x` as a series of this class. */
  static identity<T extends ABCPolyBase>(
    this: SeriesClass<T>,
    domain: ArrayLike | null = null,
    window: ArrayLike | null = null,
    symbol = "x",
  ): T {
    const d = domain === null ? this.defaultDomain : pair(domain, "Domain");
    const w = window === null ? this.defaultWindow : pair(window, "Window");
    const [off, scl] = mapparms(w, d);
    return new this(LINE(this.kernel, off, scl), d, w, symbol);
  }
  /** The basis function of degree `deg`. */
  static basis<T extends ABCPolyBase>(this: SeriesClass<T>, deg: number, opts: SeriesOptions = {}): T {
    if (!Number.isInteger(deg) || deg < 0) throw new ValueError("deg must be non-negative integer");
    const c = new Array<number>(deg + 1).fill(0);
    c[deg] = 1;
    return new this(c, opts.domain ?? null, opts.window ?? null, opts.symbol ?? "x");
  }
  /** Converts `series` (any class) to this class. */
  static cast<T extends ABCPolyBase>(this: SeriesClass<T>, s: ABCPolyBase, domain: ArrayLike | null = null, window: ArrayLike | null = null): T {
    return s.convert(domain ?? this.defaultDomain, this, window ?? this.defaultWindow);
  }
}

const isTrue = (a: NDArray): boolean => Boolean(a.item());

function resolveDomain(k: SeriesClass, domain: ArrayLike | null, x: NDArray): NDArray {
  if (Array.isArray(domain) && domain.length === 0) return k.defaultDomain.copy();
  if (domain !== null) return pair(domain, "Domain");
  // polyutils.getdomain
  const xs = x.dtype.kind === "c" ? x : x.astype("float64");
  if (xs.dtype.kind === "c") {
    const re = xs.real.toArray() as number[], im = xs.imag.toArray() as number[];
    return asarray([new Complex(Math.min(...re), Math.min(...im)), new Complex(Math.max(...re), Math.max(...im))]);
  }
  const v = xs.ravel().toArray() as number[];
  return asarray([Math.min(...v), Math.max(...v)]);
}

function fitClass(
  k: SeriesClass,
  x: ArrayLike,
  y: ArrayLike,
  deg: number | readonly number[],
  opts: ClassFitOptions,
  full: boolean,
): ABCPolyBase | { series: ABCPolyBase; residuals: NDArray; rank: number; singularValues: NDArray; rcond: number } {
  const xa = asarray(x as never);
  let domain = resolveDomain(k, opts.domain === undefined ? null : opts.domain, xa);
  if (opts.domain === undefined || opts.domain === null) {
    const [d0, d1] = domain.toArray() as number[];
    if (d0 === d1) domain = asarray([(d0 as number) - 1, (d1 as number) + 1]);
  }
  const window = opts.window ? pair(opts.window, "Window") : k.defaultWindow;
  const [off, scl] = mapparms(domain, window);
  const xnew = add(off instanceof Complex ? sc(off) : off, multiply(scl instanceof Complex ? sc(scl) : scl, xa));
  const res = fitSeries(k.kernel, xnew, y, deg, opts, full);
  const symbol = opts.symbol ?? "x";
  if (res instanceof NDArray) return new k(res, domain, window, symbol);
  return { series: new k(res.coef, domain, window, symbol), residuals: res.residuals, rank: res.rank, singularValues: res.singularValues, rcond: res.rcond };
}

/** The Clenshaw / Horner recurrences of each basis evaluated on a series argument. */
function clenshawSeries<T extends ABCPolyBase>(b: BasisName, x: T, c: Scalar[]): T {
  const n = c.length;
  const s = (v: Scalar) => sc(v);
  const ofScalar = (v: Scalar): T => x.mul(0).add(s(v));
  if (b === "polynomial") {
    let c0 = x.mul(0).add(s(c[n - 1] as Scalar));
    for (let i = 2; i <= n; i++) c0 = c0.mul(x).add(s(c[n - i] as Scalar));
    return c0;
  }
  const x2 = x.mul(2);
  let c0: T, c1: T;
  if (n === 1) {
    c0 = ofScalar(c[0] as Scalar);
    c1 = ofScalar(0);
  } else if (n === 2) {
    c0 = ofScalar(c[0] as Scalar);
    c1 = ofScalar(c[1] as Scalar);
  } else {
    let nd = n;
    c0 = ofScalar(c[n - 2] as Scalar);
    c1 = ofScalar(c[n - 1] as Scalar);
    for (let i = 3; i <= n; i++) {
      const tmp = c0;
      nd -= 1;
      const ci = s(c[n - i] as Scalar);
      switch (b) {
        case "chebyshev":
          c0 = c1.rsub(ci);
          c1 = tmp.add(c1.mul(x2));
          break;
        case "legendre":
          c0 = c1.mul((nd - 1) / nd).rsub(ci);
          c1 = tmp.add(c1.mul(x).mul((2 * nd - 1) / nd));
          break;
        case "laguerre":
          c0 = c1.mul(nd - 1).truediv(nd).rsub(ci);
          c1 = tmp.add(x.rsub(2 * nd - 1).mul(c1).truediv(nd));
          break;
        case "hermite":
          c0 = c1.mul(2 * (nd - 1)).rsub(ci);
          c1 = tmp.add(c1.mul(x2));
          break;
        default:
          c0 = c1.mul(nd - 1).rsub(ci);
          c1 = tmp.add(c1.mul(x));
      }
    }
  }
  if (b === "laguerre") return c0.add(c1.mul(x.rsub(1)));
  if (b === "hermite") return c0.add(c1.mul(x2));
  return c0.add(c1.mul(x));
}


/** numpy.polynomial.Polynomial: power series `c0 + c1 x + c2 x² + ...`. */
export class Polynomial extends ABCPolyBase {
  static readonly kernel: BasisName = "polynomial";
  static readonly basisName: string | null = null;
  static readonly defaultDomain = asarray([-1, 1], { dtype: "float64" });
  static readonly defaultWindow = asarray([-1, 1], { dtype: "float64" });
  protected override termUnicode(i: string, arg: string): string {
    return i === "1" ? `·${arg}` : `·${arg}${tr(i, SUP)}`;
  }
  protected override termAscii(i: string, arg: string): string {
    return i === "1" ? ` ${arg}` : ` ${arg}**${i}`;
  }
}
/** numpy.polynomial.Chebyshev (first kind, T_n). */
export class Chebyshev extends ABCPolyBase {
  static readonly kernel: BasisName = "chebyshev";
  static readonly basisName: string | null = "T";
  static readonly defaultDomain = asarray([-1, 1], { dtype: "float64" });
  static readonly defaultWindow = asarray([-1, 1], { dtype: "float64" });
}
/** numpy.polynomial.Legendre (P_n). */
export class Legendre extends ABCPolyBase {
  static readonly kernel: BasisName = "legendre";
  static readonly basisName: string | null = "P";
  static readonly defaultDomain = asarray([-1, 1], { dtype: "float64" });
  static readonly defaultWindow = asarray([-1, 1], { dtype: "float64" });
}
/** numpy.polynomial.Laguerre (L_n), default domain/window [0, 1]. */
export class Laguerre extends ABCPolyBase {
  static readonly kernel: BasisName = "laguerre";
  static readonly basisName: string | null = "L";
  static readonly defaultDomain = asarray([0, 1], { dtype: "float64" });
  static readonly defaultWindow = asarray([0, 1], { dtype: "float64" });
}
/** numpy.polynomial.Hermite (physicists', H_n). */
export class Hermite extends ABCPolyBase {
  static readonly kernel: BasisName = "hermite";
  static readonly basisName: string | null = "H";
  static readonly defaultDomain = asarray([-1, 1], { dtype: "float64" });
  static readonly defaultWindow = asarray([-1, 1], { dtype: "float64" });
}
/** numpy.polynomial.HermiteE (probabilists', He_n). */
export class HermiteE extends ABCPolyBase {
  static readonly kernel: BasisName = "hermite_e";
  static readonly basisName: string | null = "He";
  static readonly defaultDomain = asarray([-1, 1], { dtype: "float64" });
  static readonly defaultWindow = asarray([-1, 1], { dtype: "float64" });
}

const mod = (b: BasisName, prefix: string, maxpower: number | null, cls: unknown, clsName: string) => {
  const m = makeModule(b, prefix, maxpower).module;
  m[clsName] = cls;
  return m;
};

export const polynomial = {
  Polynomial,
  Chebyshev,
  Legendre,
  Laguerre,
  Hermite,
  HermiteE,
  setDefaultPrintstyle,
  polynomial: mod("polynomial", "poly", null, Polynomial, "Polynomial"),
  chebyshev: mod("chebyshev", "cheb", 16, Chebyshev, "Chebyshev"),
  legendre: mod("legendre", "leg", 16, Legendre, "Legendre"),
  laguerre: mod("laguerre", "lag", 16, Laguerre, "Laguerre"),
  hermite: mod("hermite", "herm", 16, Hermite, "Hermite"),
  hermite_e: mod("hermite_e", "herme", 16, HermiteE, "HermiteE"),
  polyutils: { trimcoef },
} as const;
