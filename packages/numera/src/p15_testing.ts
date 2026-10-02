// P15 np.testing assertion functions (numpy.testing), D-181.
import { Complex, isComplexLike } from "./complex.js";
import { asarray } from "./creation.js";
import { NativpyError, NotImplementedError, ValueError } from "./errors.js";
import { errstate } from "./errstate.js";
import { NDArray } from "./ndarray.js";
import { all, any, equal, less, logicalAnd, logicalNot, logicalOr, notEqual } from "./p05_compare.js";
import { isinf, isnan } from "./p05_classify.js";
import { isclose } from "./p05_close.js";
import { argwhere, extract } from "./p08.js";
import { max } from "./reduce.js";
import { abs, broadcastShapes, broadcastTo, divide, subtract, type ArrayLike, type Operand } from "./ufunc.js";
import { where } from "./indexing.js";
import { array2string0d, arrayRepr, f16round, pyFloat, pyG, scalarStr } from "./p15_format.js";

/** Raised by every failing np.testing assertion (Python's AssertionError). */
export class AssertionError extends NativpyError {
  constructor(message: string) {
    super(message, "NATIVPY_ASSERTION");
  }
}

type Value = Operand | string | null | undefined | readonly unknown[] | Record<string, unknown>;

/** Python repr of a JS value (numbers that are safe integers print as ints). */
function pyRepr(v: unknown): string {
  if (v instanceof NDArray) return arrayRepr(v);
  if (v === null || v === undefined) return "None";
  if (typeof v === "boolean") return v ? "True" : "False";
  if (typeof v === "bigint") return String(v);
  if (typeof v === "number") return Number.isInteger(v) && !Object.is(v, -0) ? String(v) : pyFloat(v);
  if (typeof v === "string") return `'${v.replace(/\\/g, "\\\\").replace(/'/g, "\\'")}'`;
  if (v instanceof Complex || isComplexLike(v)) return scalarStr(new Complex(v.re, v.im ?? 0), "complex128");
  if (Array.isArray(v)) return `[${v.map(pyRepr).join(", ")}]`;
  if (typeof v === "function") return `<function ${v.name || "<lambda>"}>`;
  if (typeof v === "object") return `{${Object.entries(v).map(([k, x]) => `'${k}': ${pyRepr(x)}`).join(", ")}}`;
  return String(v);
}

export interface BuildErrMsgOptions {
  header?: string;
  verbose?: boolean;
  names?: readonly string[];
  precision?: number;
}

/** numpy.testing.build_err_msg: the message layout of every assertion. */
export function buildErrMsg(arrays: readonly unknown[], errMsg = "", opts: BuildErrMsgOptions = {}): string {
  const { header = "Items are not equal:", verbose = true, names = ["ACTUAL", "DESIRED"], precision = 8 } = opts;
  const msg = ["\n" + header];
  if (errMsg) {
    if (!errMsg.includes("\n") && errMsg.length < 79 - header.length) msg[0] += " " + errMsg;
    else msg.push(errMsg);
  }
  if (verbose) {
    arrays.forEach((a, i) => {
      let r = a instanceof NDArray ? arrayRepr(a, precision) : pyRepr(a);
      if ((r.match(/\n/g) ?? []).length > 3) r = r.split("\n").slice(0, 3).join("\n") + "...";
      msg.push(` ${names[i] ?? ""}: ${r}`);
    });
  }
  return msg.join("\n");
}

/** Common options of the array assertions. */
export interface AssertOptions {
  /** Extra text added to the failure message. */
  errMsg?: string;
  /** Include the array reprs in the message (default true). */
  verbose?: boolean;
  /** Also require equal shapes and dtypes; no 0-d broadcasting (default false). */
  strict?: boolean;
}

export interface AssertArrayCompareOptions extends AssertOptions {
  header?: string;
  precision?: number;
  equalNan?: boolean;
  equalInf?: boolean;
  names?: readonly [string, string];
}

type Flag = boolean | NDArray;
const isTrue = (a: NDArray): boolean => Boolean(a.item());
const itemOf = (a: NDArray, p: number[]): string => {
  const v = a.ndim === 0 ? a.item() : a.item(...p);
  return scalarStr(v as number | boolean | Complex, a.dtype.name);
};

/**
 * numpy.testing.assert_array_compare: checks `comparison(x, y)` element-wise
 * (after matching NaN / Inf positions) and raises AssertionError with
 * NumPy's mismatch report.
 */
export function assertArrayCompare(
  comparison: (x: NDArray, y: NDArray) => NDArray | boolean,
  actual: ArrayLike,
  desired: ArrayLike,
  opts: AssertArrayCompareOptions = {},
): void {
  const {
    errMsg = "",
    verbose = true,
    header = "",
    precision = 6,
    equalNan = true,
    equalInf = true,
    strict = false,
    names = ["ACTUAL", "DESIRED"],
  } = opts;
  let x = asarray(actual);
  let y = asarray(desired);
  const ox = x, oy = y;
  const fail = (m: string, arrays: unknown[] = [x, y]): never => {
    throw new AssertionError(buildErrMsg(arrays, m, { verbose, header, names, precision }));
  };
  const sameShape = x.shape.length === y.shape.length && x.shape.every((d, i) => d === y.shape[i]);
  const cond = strict ? sameShape && x.dtype.name === y.dtype.name : x.ndim === 0 || y.ndim === 0 || sameShape;
  if (!cond) {
    const reason = !sameShape
      ? `\n(shapes ${shapeTuple(x.shape)}, ${shapeTuple(y.shape)} mismatch)`
      : `\n(dtypes ${x.dtype.name}, ${y.dtype.name} mismatch)`;
    fail(errMsg + reason);
  }
  errstate({ all: "ignore" }, () => {
    const samePos = (fx: (a: NDArray) => NDArray, what: string): Flag => {
      const xid = fx(x), yid = fx(y);
      if (!isTrue(all(equal(xid, yid)))) fail(errMsg + `\n${what} location mismatch:`);
      if (xid.ndim === 0) return isTrue(xid);
      if (yid.ndim === 0) return isTrue(yid);
      return yid;
    };
    let flagged: Flag = false;
    if (equalNan) flagged = samePos(isnan, "nan");
    if (equalInf) {
      const f0 = flagged;
      const infs = samePos((a) => (typeof f0 === "boolean" ? (f0 ? logicalAnd(isinf(a), false) : isinf(a)) : logicalAnd(isinf(a), logicalNot(f0))), "inf");
      const mask = typeof infs === "boolean" ? asarray(infs) : infs;
      if (isTrue(any(mask))) {
        const [xi, yi] = x.ndim > 0 && y.ndim > 0 ? [extract(mask, x), extract(mask, y)] : [x, y];
        if (!isTrue(all(equal(xi, yi)))) fail(errMsg + "\ninf values mismatch:", [xi, yi]);
      }
      flagged = typeof flagged === "boolean" && typeof infs === "boolean" ? flagged || infs : logicalOr(asarray(flagged), asarray(infs));
      if (flagged instanceof NDArray && flagged.ndim === 0) flagged = isTrue(flagged);
    }
    if (flagged instanceof NDArray) {
      const keep = logicalNot(flagged);
      x = extract(keep, x);
      y = extract(keep, y);
      if (x.size === 0) return;
    } else if (flagged) {
      return;
    }
    const v = comparison(x, y);
    const val = typeof v === "boolean" ? asarray([v]) : v;
    const reduced = val.ravel().toArray() as boolean[];
    if (reduced.every(Boolean)) return;
    const nMismatch = reduced.filter((b) => !b).length;
    const nElements = flagged instanceof NDArray ? flagged.size : reduced.length;
    const remarks = [`Mismatched elements: ${nMismatch} / ${nElements} (${pyG((100 * nMismatch) / nElements, 3)}%)`];
    const invalids = logicalNot(val);
    if (typeof v !== "boolean" && invalids.ndim !== 0) {
      let positions = argwhere(invalids).toArray() as number[][];
      if (flagged instanceof NDArray) {
        const unflagged = argwhere(logicalNot(flagged)).toArray() as number[][];
        positions = unflagged.filter((_, i) => !reduced[i]);
      }
      const s = positions
        .slice(0, 5)
        .map((p) => ` [${p.join(", ")}]: ${itemOf(ox, p)} (${names[0]}), ${itemOf(oy, p)} (${names[1]})`)
        .join("\n");
      const head = positions.length === 1 ? "Mismatch at index:" : positions.length <= 5 ? "Mismatch at indices:" : "First 5 mismatches are at indices:";
      remarks.push(`${head}\n${s}`);
    }
    try {
      let error = abs(subtract(x, y));
      if (x.dtype.kind === "u") {
        const e2 = abs(subtract(y, x));
        error = where(less(e2, error), e2, error);
      }
      const inv = typeof v === "boolean" ? broadcastTo(invalids, error.shape) : invalids;
      remarks.push("Max absolute difference among violations: " + array2string0d(max(extract(inv, error))));
      const nzi = logicalAnd(inv, notEqual(y, 0));
      let rel = "inf";
      if (isTrue(any(nzi))) {
        const yb = broadcastTo(y, error.shape);
        rel = array2string0d(max(divide(extract(nzi, error), abs(extract(nzi, yb)))));
      }
      remarks.push("Max relative difference among violations: " + rel);
    } catch (e) {
      if (!(e instanceof NativpyError)) throw e; // e.g. boolean subtract (NumPy: TypeError)
    }
    throw new AssertionError(
      buildErrMsg([ox, oy], errMsg + "\n" + remarks.join("\n"), { verbose, header, names, precision }),
    );
  });
}

const shapeTuple = (s: readonly number[]) => (s.length === 1 ? `(${s[0]},)` : `(${s.join(", ")})`);

/** numpy.testing.assert_array_equal: shapes (0-d broadcasts) and elements equal; NaNs in the same places match. */
export function assertArrayEqual(actual: ArrayLike, desired: ArrayLike, opts: AssertOptions = {}): void {
  assertArrayCompare((x, y) => equal(x, y), actual, desired, { ...opts, header: "Arrays are not equal" });
}

export interface AlmostEqualOptions extends AssertOptions {
  /** Decimal places: `|actual - desired| < 1.5 * 10**-decimal` (default 6 for arrays, 7 for scalars). */
  decimal?: number;
}

/** numpy.testing.assert_array_almost_equal: `abs(desired - actual) < 1.5 * 10**-decimal` (default 6). */
export function assertArrayAlmostEqual(actual: ArrayLike, desired: ArrayLike, opts: AlmostEqualOptions = {}): void {
  const decimal = opts.decimal ?? 6;
  const tol = 1.5 * 10 ** -decimal;
  assertArrayCompare(
    (x, y) => {
      const yf = y.dtype.kind === "f" || y.dtype.kind === "c" ? y : y.astype("float64");
      return less(abs(subtract(x, yf)), tol);
    },
    actual,
    desired,
    { ...opts, header: `Arrays are not almost equal to ${decimal} decimals`, precision: decimal },
  );
}

export interface AllcloseOptions extends AssertOptions {
  /** Relative tolerance (default 1e-7). */
  rtol?: number;
  /** Absolute tolerance (default 0). */
  atol?: number;
  /** NaNs compare equal (default true). */
  equalNan?: boolean;
}

/** numpy.testing.assert_allclose: `|actual - desired| <= atol + rtol * |desired|` (rtol 1e-7, atol 0). */
export function assertAllclose(actual: ArrayLike, desired: ArrayLike, opts: AllcloseOptions = {}): void {
  const { rtol = 1e-7, atol = 0, equalNan = true } = opts;
  assertArrayCompare((x, y) => isclose(x, y, { rtol, atol, equalNan }), actual, desired, {
    ...opts,
    header: `Not equal to tolerance rtol=${pyG(rtol, 6)}, atol=${pyG(atol, 6)}`,
    equalNan,
  });
}

/** numpy.testing.assert_array_less: `x < y` element-wise (Infs are compared, not matched). */
export function assertArrayLess(x: ArrayLike, y: ArrayLike, opts: AssertOptions = {}): void {
  assertArrayCompare((a, b) => less(a, b), x, y, {
    ...opts,
    header: "Arrays are not strictly ordered `x < y`",
    equalInf: false,
    names: ["x", "y"],
  });
}

const isArrayValue = (v: unknown): boolean => v instanceof NDArray || Array.isArray(v);
const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v) && !(v instanceof NDArray) && !isComplexLike(v);
const toComplex = (v: unknown): Complex | undefined =>
  v instanceof Complex ? v : isComplexLike(v) ? new Complex(v.re, v.im ?? 0) : undefined;
const toNum = (v: unknown): number => (typeof v === "boolean" ? Number(v) : typeof v === "bigint" ? Number(v) : (v as number));

/**
 * numpy.testing.assert_equal: plain objects and JS arrays are compared
 * recursively (`key=` / `item=` in the message), NDArrays with
 * assertArrayEqual, scalars with NaN == NaN and the sign of zero checked.
 */
export function assertEqual(actual: unknown, desired: unknown, opts: AssertOptions = {}): void {
  const { errMsg = "", verbose = true } = opts;
  if (isPlainObject(desired)) {
    if (!isPlainObject(actual)) throw new AssertionError(actual === null ? "None" : typeof actual);
    assertEqual(Object.keys(actual).length, Object.keys(desired).length, opts);
    for (const k of Object.keys(desired)) {
      if (!(k in actual)) throw new AssertionError(pyRepr(k));
      assertEqual(actual[k], desired[k], { ...opts, errMsg: `key=${pyRepr(k)}\n${errMsg}` });
    }
    return;
  }
  if (Array.isArray(desired) && Array.isArray(actual)) {
    assertEqual(actual.length, desired.length, opts);
    for (let k = 0; k < desired.length; k++) {
      assertEqual(actual[k], desired[k], { ...opts, errMsg: `item=${k}\n${errMsg}` });
    }
    return;
  }
  if (actual instanceof NDArray || desired instanceof NDArray) {
    assertArrayEqual(actual as ArrayLike, desired as ArrayLike, opts);
    return;
  }
  const msg = buildErrMsg([actual, desired], errMsg, { verbose });
  const ca = toComplex(actual), cd = toComplex(desired);
  if (ca || cd) {
    const parts = (c: Complex | undefined, v: unknown): [unknown, unknown] => (c ? [c.re, c.im] : [v, 0]);
    const [ar, ai] = parts(ca, actual), [dr, di] = parts(cd, desired);
    try {
      assertEqual(ar, dr);
      assertEqual(ai, di);
    } catch (e) {
      if (e instanceof AssertionError) throw new AssertionError(msg);
      throw e;
    }
    return;
  }
  if (isArrayValue(actual) !== isArrayValue(desired)) throw new AssertionError(msg);
  const numeric = (v: unknown) => typeof v === "number" || typeof v === "bigint" || typeof v === "boolean";
  if (numeric(actual) && numeric(desired)) {
    const a = toNum(actual), d = toNum(desired);
    if (Number.isNaN(a) && Number.isNaN(d)) return;
    if (a === 0 && d === 0 && Object.is(a, -0) !== Object.is(d, -0)) throw new AssertionError(msg);
    const eq = typeof actual === "bigint" && typeof desired === "bigint" ? actual === desired : a === d;
    if (!eq) throw new AssertionError(msg);
    return;
  }
  if (actual !== desired) throw new AssertionError(msg);
}

/**
 * numpy.testing.assert_almost_equal: scalars must satisfy
 * `|desired - actual| < 1.5 * 10**-decimal` (default 7); arrays use
 * assertArrayAlmostEqual; complex values compare real and imaginary parts.
 */
export function assertAlmostEqual(actual: Operand, desired: Operand, opts: AlmostEqualOptions = {}): void {
  const { decimal = 7, errMsg = "", verbose = true } = opts;
  const build = () =>
    buildErrMsg([actual, desired], errMsg, { verbose, header: `Arrays are not almost equal to ${decimal} decimals` });
  const isCplx = (v: Operand) => (v instanceof NDArray ? v.dtype.kind === "c" : isArrayValue(v) ? asarray(v as ArrayLike).dtype.kind === "c" : isComplexLike(v));
  if (isCplx(actual) || isCplx(desired)) {
    const parts = (v: Operand): [Operand, Operand] => {
      if (!isCplx(v)) return [v, 0];
      if (!isArrayValue(v)) {
        const c = toComplex(v) as Complex;
        return [c.re, c.im];
      }
      const a = asarray(v as ArrayLike);
      return [a.real, a.imag];
    };
    const [ar, ai] = parts(actual), [dr, di] = parts(desired);
    try {
      assertAlmostEqual(ar, dr, { decimal });
      assertAlmostEqual(ai, di, { decimal });
    } catch (e) {
      if (e instanceof AssertionError) throw new AssertionError(build());
      throw e;
    }
    return;
  }
  if (isArrayValue(actual) || isArrayValue(desired)) {
    assertArrayAlmostEqual(actual as ArrayLike, desired as ArrayLike, { decimal, errMsg });
    return;
  }
  const a = toNum(actual), d = toNum(desired);
  if (!(Number.isFinite(a) && Number.isFinite(d))) {
    if (Number.isNaN(a) || Number.isNaN(d)) {
      if (!(Number.isNaN(a) && Number.isNaN(d))) throw new AssertionError(build());
    } else if (a !== d) {
      throw new AssertionError(build());
    }
    return;
  }
  if (Math.abs(d - a) >= 1.5 * 10 ** -decimal) throw new AssertionError(build());
}

export interface ApproxEqualOptions extends AssertOptions {
  /** Significant digits to compare (default 7). */
  significant?: number;
}

/** numpy.testing.assert_approx_equal: scalars agree to `significant` significant digits (default 7). */
export function assertApproxEqual(actual: number, desired: number, opts: ApproxEqualOptions = {}): void {
  const { significant = 7, errMsg = "", verbose = true } = opts;
  const a = Number(actual), d = Number(desired);
  if (a === d) return;
  const scale = 10 ** Math.floor(Math.log10(0.5 * (Math.abs(d) + Math.abs(a))));
  const scD = scale === 0 ? 0 : d / scale, scA = scale === 0 ? 0 : a / scale;
  const header = `Items are not equal to ${significant} significant digits:`;
  const message = buildErrMsg([], errMsg, { header }) + (verbose ? `\n ACTUAL: ${pyFloat(a)}\n DESIRED: ${pyFloat(d)}` : "");
  if (!(Number.isFinite(d) && Number.isFinite(a))) {
    if (Number.isNaN(d) || Number.isNaN(a)) {
      if (!(Number.isNaN(d) && Number.isNaN(a))) throw new AssertionError(message);
    } else if (d !== a) {
      throw new AssertionError(message);
    }
    return;
  }
  if (Math.abs(scD - scA) >= 10 ** -(significant - 1)) throw new AssertionError(message);
}

// ---- ULP helpers ----

type FloatName = "float16" | "float32" | "float64";

function commonFloat(x: NDArray, y: NDArray): FloatName {
  if (x.dtype.kind === "c" || y.dtype.kind === "c") throw new NotImplementedError("_nulp not implemented for complex array");
  const f = (a: NDArray): number => (a.dtype.kind === "f" ? a.itemSize : 8);
  const s = Math.max(f(x), f(y));
  return s === 2 ? "float16" : s === 4 ? "float32" : "float64";
}

/** Two's-complement-ordered integer representation of a float (NumPy `integer_repr`). */
function intRepr(v: number, t: FloatName): bigint {
  const buf = new DataView(new ArrayBuffer(8));
  let r: bigint;
  let bits: number;
  if (t === "float64") {
    buf.setFloat64(0, v);
    r = buf.getBigInt64(0);
    bits = 64;
  } else if (t === "float32") {
    buf.setFloat32(0, v);
    r = BigInt(buf.getInt32(0));
    bits = 32;
  } else {
    r = BigInt(halfBits(v));
    bits = 16;
  }
  const comp = -(1n << BigInt(bits - 1));
  return r < 0n ? BigInt.asIntN(bits, comp - r) : r;
}

/** IEEE binary16 bit pattern (as int16) of `v` rounded to float16. */
function halfBits(v: number): number {
  const h = f16round(v);
  let bits: number;
  if (Number.isNaN(h)) bits = 0x7e00;
  else {
    const sign = h < 0 || Object.is(h, -0) ? 0x8000 : 0;
    const a = Math.abs(h);
    if (a === Infinity) bits = sign | 0x7c00;
    else if (a < 2 ** -14) bits = sign | Math.round(a / 2 ** -24);
    else {
      let e = Math.floor(Math.log2(a));
      if (2 ** e > a) e -= 1;
      if (2 ** (e + 1) <= a) e += 1;
      bits = sign | ((e + 15) << 10) | Math.round((a / 2 ** e - 1) * 1024);
    }
  }
  return (bits << 16) >> 16;
}

const roundTo = (t: FloatName) => (t === "float16" ? f16round : t === "float32" ? Math.fround : (v: number) => v);

function flatNumbers(a: NDArray): number[] {
  return (a.ravel().toArray() as (number | boolean | bigint)[]).map(Number);
}

/** numpy.testing.nulp_diff-style helper: ULP distance per element (leading 1-length axis, as NumPy). */
function nulpDiff(x: NDArray, y: NDArray, t: FloatName): NDArray {
  const sx = x.shape, sy = y.shape;
  if (sx.length !== sy.length || sx.some((d, i) => d !== sy[i])) {
    throw new ValueError(`Arrays do not have the same shape: ${shapeTuple([1, ...sx])} - ${shapeTuple([1, ...sy])}`);
  }
  const round = roundTo(t);
  const bits = t === "float64" ? 64 : t === "float32" ? 32 : 16;
  const xs = flatNumbers(x), ys = flatNumbers(y);
  const out = xs.map((v, i) => {
    const d = BigInt.asIntN(bits, intRepr(round(v), t) - intRepr(round(ys[i] ?? 0), t));
    return Math.abs(round(Number(d)));
  });
  return asarray(out).astype(t).reshape([1, ...sx]);
}

const spacing = (v: number, t: FloatName): number => {
  if (!Number.isFinite(v)) return NaN;
  const a = Math.abs(v);
  const r = roundTo(t);
  const mant = t === "float64" ? 52 : t === "float32" ? 23 : 10;
  const minExp = t === "float64" ? -1022 : t === "float32" ? -126 : -14;
  const e = a === 0 ? minExp : Math.max(Math.floor(Math.log2(a)), minExp);
  let s = 2 ** (e - mant);
  if (r(a + s) === a) s *= 2; // guard against log2 rounding at powers of two
  return s;
};

/**
 * numpy.testing.assert_array_almost_equal_nulp: `|x - y| <= nulp * spacing(max(|x|, |y|))`
 * element-wise (default nulp 1).
 */
export function assertArrayAlmostEqualNulp(x: ArrayLike, y: ArrayLike, nulp = 1): void {
  const a = asarray(x), b = asarray(y);
  const shape = broadcastShapes(a.shape, b.shape);
  const ab = broadcastTo(a, shape), bb = broadcastTo(b, shape);
  const cplx = a.dtype.kind === "c" || b.dtype.kind === "c";
  const t = cplx ? "float64" : commonFloat(a, b);
  const absVals = (arr: NDArray) =>
    cplx ? flatNumbers(abs(arr)) : flatNumbers(arr).map(Math.abs);
  const ax = absVals(ab), ay = absVals(bb);
  const diff = flatNumbers(abs(subtract(ab, bb)));
  const ok = diff.every((d, i) => d <= nulp * spacing(Math.max(ax[i] ?? 0, ay[i] ?? 0), t));
  if (ok) return;
  if (cplx) throw new AssertionError(`Arrays are not equal to ${nulp} ULP`);
  const nd = flatNumbers(nulpDiff(ab, bb, t));
  throw new AssertionError(`Arrays are not equal to ${nulp} ULP (max is ${pyG(Math.max(...nd), 6)})`);
}

export interface MaxUlpOptions {
  /** Maximum allowed ULP distance (default 1). */
  maxulp?: number;
  /** Convert both inputs to this dtype first. */
  dtype?: FloatName | null;
}

/**
 * numpy.testing.assert_array_max_ulp: every element of `a` and `b` is at most
 * `maxulp` units in the last place apart; returns the ULP distances (shape
 * `[1, ...shape]`, the common float dtype, as NumPy).
 */
export function assertArrayMaxUlp(a: ArrayLike, b: ArrayLike, opts: MaxUlpOptions = {}): NDArray {
  const { maxulp = 1, dtype = null } = opts;
  let x = asarray(a), y = asarray(b);
  if (dtype) {
    x = x.astype(dtype);
    y = y.astype(dtype);
  }
  const ret = nulpDiff(x, y, commonFloat(x, y));
  const vals = flatNumbers(ret);
  if (!vals.every((v) => v <= maxulp)) {
    throw new AssertionError(
      `Arrays are not almost equal up to ${pyG(maxulp, 6)} ULP (max difference is ${pyG(Math.max(...vals), 6)} ULP)`,
    );
  }
  return ret;
}

/** numpy.testing.assert_: raises AssertionError(msg) when `val` is falsy (`msg` may be a function). */
export function assert_(val: unknown, msg: string | (() => string) = ""): void {
  if (!val) throw new AssertionError(typeof msg === "function" ? msg() : msg);
}

type ErrorCtor = abstract new (...args: never[]) => Error;
const fnName = (f: (...a: never[]) => unknown) => f.name || "<lambda>";

/**
 * numpy.testing.assert_raises(ErrorClass, fn, ...args): `fn(...args)` must
 * throw an instance of `ErrorClass`; returns the caught error. Other errors
 * propagate. Async functions are not awaited.
 */
export function assertRaises<A extends unknown[]>(errorClass: ErrorCtor, fn: (...args: A) => unknown, ...args: A): Error {
  try {
    fn(...args);
  } catch (e) {
    if (e instanceof errorClass) return e;
    throw e;
  }
  throw new AssertionError(`${errorClass.name} not raised by ${fnName(fn)}`);
}

/** numpy.testing.assert_raises_regex: as assertRaises, and the error message must match `pattern` (regex search). */
export function assertRaisesRegex<A extends unknown[]>(
  errorClass: ErrorCtor,
  pattern: string | RegExp,
  fn: (...args: A) => unknown,
  ...args: A
): Error {
  const e = assertRaises(errorClass, fn, ...args);
  const re = typeof pattern === "string" ? new RegExp(pattern) : pattern;
  if (!re.test(e.message)) throw new AssertionError(`"${re.source}" does not match "${e.message}"`);
  return e;
}

interface CapturedWarning {
  type: string;
  message: string;
}

function captureWarnings<T>(run: () => T): { result: T; warnings: CapturedWarning[] } {
  const warnings: CapturedWarning[] = [];
  const original = process.emitWarning;
  process.emitWarning = ((warning: string | Error, typeOrOpts?: unknown) => {
    const type =
      warning instanceof Error
        ? warning.name
        : typeof typeOrOpts === "string"
          ? typeOrOpts
          : ((typeOrOpts as { type?: string } | undefined)?.type ?? "Warning");
    warnings.push({ type, message: warning instanceof Error ? warning.message : String(warning) });
  }) as typeof process.emitWarning;
  try {
    return { result: run(), warnings };
  } finally {
    process.emitWarning = original;
  }
}

/**
 * numpy.testing.assert_warns(warningType, fn, ...args): `fn(...args)` must emit
 * a Node process warning (`process.emitWarning`, e.g. numera's
 * `RuntimeWarning` from np.seterr "warn") of type `warningType` (`null`: any
 * type). Matching warnings are recorded instead of emitted. Returns fn's result.
 */
export function assertWarns<A extends unknown[], R>(warningType: string | null, fn: (...args: A) => R, ...args: A): R {
  const { result, warnings } = captureWarnings(() => fn(...args));
  if (!warnings.some((w) => warningType === null || w.type === warningType)) {
    throw new AssertionError(`No warning raised when calling ${fnName(fn)}`);
  }
  return result;
}

/** numpy.testing.assert_no_warnings(fn, ...args): `fn(...args)` must not emit any process warning. */
export function assertNoWarnings<A extends unknown[], R>(fn: (...args: A) => R, ...args: A): R {
  const { result, warnings } = captureWarnings(() => fn(...args));
  if (warnings.length) {
    const list = warnings.map((w) => `${w.type}(${pyRepr(w.message)})`).join(", ");
    throw new AssertionError(`Got warnings when calling ${fnName(fn)}: [${list}]`);
  }
  return result;
}

/**
 * numpy.testing.assert_string_equal: equal strings, else a line diff
 * ("- " actual / "+ " desired lines; NumPy's difflib "? " hint lines are not produced).
 */
export function assertStringEqual(actual: string, desired: string): void {
  if (typeof actual !== "string") throw new AssertionError(`<class '${typeof actual}'>`);
  if (typeof desired !== "string") throw new AssertionError(`<class '${typeof desired}'>`);
  if (actual === desired) return;
  const a = actual.match(/[^\n]*\n|[^\n]+$/g) ?? [], d = desired.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  const n = a.length, m = d.length;
  const L = Array.from({ length: n + 1 }, () => new Array<number>(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      L[i]![j] = a[i] === d[j] ? L[i + 1]![j + 1]! + 1 : Math.max(L[i + 1]![j]!, L[i]![j + 1]!);
  const out: string[] = [];
  let i = 0, j = 0;
  while (i < n || j < m) {
    if (i < n && j < m && a[i] === d[j]) {
      i++;
      j++;
    } else if (j >= m || (i < n && L[i + 1]![j]! >= L[i]![j + 1]!)) {
      out.push("- " + a[i++]);
    } else {
      out.push("+ " + d[j++]);
    }
  }
  throw new AssertionError(`Differences in strings:\n${out.join("").trimEnd()}`);
}

const deepEqual = (a: unknown, b: unknown): boolean => {
  if (Object.is(a, b) || a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((x, i) => deepEqual(x, b[i]));
  if (isPlainObject(a) && isPlainObject(b)) {
    const ka = Object.keys(a), kb = Object.keys(b);
    return ka.length === kb.length && ka.every((k) => k in b && deepEqual(a[k], b[k]));
  }
  return false;
};

/** numpy.testing.print_assert_equal: deep `==` of JS values, with an ACTUAL/DESIRED message. */
export function printAssertEqual(testString: string, actual: Value, desired: Value): void {
  if (deepEqual(actual, desired)) return;
  throw new AssertionError(`${testString} failed\nACTUAL: \n${pyRepr(actual)}\nDESIRED: \n${pyRepr(desired)}\n`);
}

export const testing = {
  AssertionError,
  assert_,
  assertAllclose,
  assertAlmostEqual,
  assertApproxEqual,
  assertArrayAlmostEqual,
  assertArrayAlmostEqualNulp,
  assertArrayCompare,
  assertArrayEqual,
  assertArrayLess,
  assertArrayMaxUlp,
  assertEqual,
  assertNoWarnings,
  assertRaises,
  assertRaisesRegex,
  assertStringEqual,
  assertWarns,
  buildErrMsg,
  printAssertEqual,
} as const;
