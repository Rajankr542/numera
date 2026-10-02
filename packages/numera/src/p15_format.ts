// P15 private formatting helpers modelled on NumPy's arrayprint (D-181):
// `arrayRepr` (array_repr with `precision`, floatmode "maxprec", 75-column
// wrapping, summarization above 1000 elements), `array2string` of 0-d arrays,
// and `str()` of NumPy scalars. Used for np.testing messages only.
import { Complex } from "./complex.js";
import { NDArray } from "./ndarray.js";

type Elem = number | boolean | bigint | Complex;

const LINE_WIDTH = 75;
const THRESHOLD = 1000;
const EDGE = 3;

export function f16round(x: number): number {
  if (!Number.isFinite(x) || x === 0) return x;
  const a = Math.abs(x);
  if (a >= 65520) return Math.sign(x) * Infinity;
  const e = Math.max(Math.floor(Math.log2(a)), -14);
  let q = 2 ** (e - 10);
  if (a / q >= 2048) q *= 2; // log2 rounding guard
  const v = a / q;
  let r = Math.round(v);
  if (r - v === 0.5 && r % 2 === 1) r -= 1; // ties to even
  return Math.sign(x) * r * q;
}

/** Shortest decimal digits `d` and exponent `e` (value = d[0].d[1:] × 10^e) for the dtype. */
function shortest(x: number, kind: string): { d: string; e: number } {
  let y = x;
  if (kind === "float32" || kind === "float16") {
    const round = kind === "float32" ? Math.fround : f16round;
    for (let p = 1; p <= 17; p++) {
      const c = Number(x.toPrecision(p));
      if (round(c) === x) {
        y = c;
        break;
      }
    }
  }
  const [m, ex] = Math.abs(y).toExponential().split("e");
  return { d: (m ?? "0").replace(".", ""), e: Number(ex) };
}

const isNeg = (x: number) => x < 0 || Object.is(x, -0);

/** dragon4_positional(x, precision, unique, trim='.') */
function positional(x: number, precision: number, kind: string, sign: boolean): string {
  const { d, e } = shortest(x, kind);
  const frac = Math.max(0, d.length - 1 - e);
  let body: string;
  if (frac <= precision) {
    if (e >= 0) {
      const int = d.length > e + 1 ? d.slice(0, e + 1) : d + "0".repeat(e + 1 - d.length);
      body = `${int}.${d.slice(e + 1)}`;
    } else {
      body = `0.${"0".repeat(-e - 1)}${d}`;
    }
  } else {
    body = Math.abs(x).toFixed(precision).replace(/0+$/, "");
    if (!body.includes(".")) body += ".";
  }
  return (isNeg(x) ? "-" : sign ? "+" : "") + body;
}

/** dragon4_scientific(x, precision, unique, trim='.') split into parts. */
function scientific(x: number, precision: number, kind: string, sign: boolean): [string, string, string] {
  let { d, e } = shortest(x, kind);
  if (d.length - 1 > precision) {
    const [m, ex] = Math.abs(x).toExponential(precision).split("e");
    d = (m ?? "0").replace(".", "").replace(/0+$/, "") || "0";
    e = Number(ex);
  }
  const int = (isNeg(x) ? "-" : sign ? "+" : "") + d[0];
  return [int, d.slice(1), (e < 0 ? "-" : "+") + String(Math.abs(e))];
}

const CUTOFF: Record<string, number> = { float16: 1e3, float32: 1e6, float64: 1e8 };

/** NumPy FloatingFormat (floatmode "maxprec", suppress_small false, legacy off). */
function floatFormatter(data: number[], precision: number, kind: string, sign: boolean): (x: number) => string {
  const finite = data.filter(Number.isFinite);
  const nz = finite.filter((v) => v !== 0).map(Math.abs);
  let expFormat = false;
  if (nz.length) {
    const max = Math.max(...nz), min = Math.min(...nz);
    expFormat = max >= (CUTOFF[kind] ?? 1e8) || min < 0.0001 || max / min > 1000;
  }
  let padLeft = 0, padRight = 0, prec = precision, expSize = -1;
  if (finite.length && expFormat) {
    const parts = finite.map((v) => scientific(v, precision, kind, sign));
    expSize = Math.max(...parts.map((p) => p[2].length)) - 1;
    prec = Math.max(...parts.map((p) => p[1].length));
    padLeft = Math.max(...parts.map((p) => p[0].length));
    padRight = Math.max(expSize, 2) + 2 + prec;
    expSize = Math.max(expSize, 2);
  } else if (finite.length) {
    const parts = finite.map((v) => positional(v, precision, kind, sign).split("."));
    padLeft = Math.max(...parts.map((p) => (p[0] ?? "").length));
    padRight = Math.max(...parts.map((p) => (p[1] ?? "").length));
  }
  if (finite.length !== data.length) {
    const neginf = sign || data.some((v) => v === -Infinity);
    const offset = padRight + 1;
    padLeft = Math.max(padLeft, 3 - offset, 3 + (neginf ? 1 : 0) - offset);
  }
  return (x: number) => {
    if (!Number.isFinite(x)) {
      const ret = Number.isNaN(x) ? (sign ? "+nan" : "nan") : (x < 0 ? "-" : sign ? "+" : "") + "inf";
      return " ".repeat(Math.max(0, padLeft + padRight + 1 - ret.length)) + ret;
    }
    if (expFormat) {
      const [i, f, ex] = scientific(x, prec, kind, sign);
      const digits = ex.slice(1).padStart(expSize, "0");
      return `${i.padStart(padLeft)}.${f.padEnd(prec, "0")}e${ex[0]}${digits}`;
    }
    const [i = "", f = ""] = positional(x, precision, kind, sign).split(".");
    return `${i.padStart(padLeft)}.${f.padEnd(padRight)}`;
  };
}

const realKind = (name: string) => (name === "complex64" ? "float32" : name === "complex128" ? "float64" : name);

function elemFormatter(values: Elem[], dtype: string, precision: number, zeroD: boolean): (v: Elem) => string {
  const kind = dtype[0] === "b" && dtype === "bool" ? "b" : dtype.startsWith("complex") ? "c" : dtype.startsWith("float") ? "f" : "i";
  if (kind === "b") return (v) => (v ? (zeroD ? "True" : " True") : "False");
  if (kind === "i") {
    const w = Math.max(0, ...values.map((v) => String(v).length));
    return (v) => String(v).padStart(w);
  }
  if (kind === "f") {
    const f = floatFormatter(values.map(Number), precision, dtype, false);
    return (v) => f(Number(v));
  }
  const cs = values as Complex[];
  const rk = realKind(dtype);
  const fr = floatFormatter(cs.map((c) => c.re), precision, rk, false);
  const fi = floatFormatter(cs.map((c) => c.im), precision, rk, true);
  return (v) => {
    const c = v as Complex;
    const i = fi(c.im);
    const sp = i.trimEnd().length;
    return fr(c.re) + i.slice(0, sp) + "j" + i.slice(sp);
  };
}

const IMPLIED = new Set(["float64", "int64", "bool", "complex128"]);
export const shapeStr = (s: readonly number[]) => (s.length === 1 ? `(${s[0]},)` : `(${s.join(", ")})`);

function nested(a: NDArray): unknown {
  return a.toArray();
}

function leading(v: unknown, shape: number[], summarize: boolean): unknown {
  if (!Array.isArray(v) || !summarize) return v;
  const inner = (x: unknown) => leading(x, shape.slice(1), summarize);
  const n = shape[0] ?? 0;
  if (n > 2 * EDGE) return [...v.slice(0, EDGE).map(inner), ...v.slice(n - EDGE).map(inner)];
  return v.map(inner);
}

const flatten = (v: unknown): Elem[] => (Array.isArray(v) ? v.flatMap(flatten) : [v as Elem]);

/** NumPy array2string body (separator ", ", prefix `prefix`, suffix length `suffixLen`). */
function array2stringBody(a: NDArray, precision: number, prefix: string, suffixLen: number): string {
  if (a.size === 0) return "[]";
  const summarize = a.size > THRESHOLD;
  const data = nested(a);
  const fmt = elemFormatter(flatten(leading(data, a.shape, summarize)), a.dtype.name, precision, a.ndim === 0);
  if (a.ndim === 0) return fmt(data as Elem);
  const lineWidth = LINE_WIDTH - suffixLen;
  const nextPrefix = " ".repeat(prefix.length + 1);
  const sep = ", ";
  const rec = (v: unknown, axis: number, hanging: string, width: number): string => {
    if (axis === a.ndim) return fmt(v as Elem);
    const arr = v as unknown[];
    const n = arr.length;
    const show = summarize && 2 * EDGE < n;
    const idx = show ? [...Array(EDGE).keys(), -1, ...Array.from({ length: EDGE }, (_, k) => n - EDGE + k)] : [...arr.keys()];
    const nextHang = hanging + " ";
    const nextWidth = width - 1;
    let s = "";
    if (a.ndim - axis === 1) {
      const elemWidth = width - 1;
      let line = hanging;
      idx.forEach((i, k) => {
        const word = i < 0 ? "..." : rec(arr[i], axis + 1, nextHang, nextWidth);
        if (line.length + word.length > elemWidth && line.length > hanging.length) {
          s += line.trimEnd() + "\n";
          line = hanging;
        }
        line += word;
        if (k < idx.length - 1) line += sep;
      });
      s += line;
    } else {
      const lineSep = "," + "\n".repeat(a.ndim - axis - 1);
      idx.forEach((i, k) => {
        const nestedStr = i < 0 ? "..." : rec(arr[i], axis + 1, nextHang, nextWidth);
        s += hanging + nestedStr + (k < idx.length - 1 ? lineSep : "");
      });
    }
    return "[" + s.slice(hanging.length) + "]";
  };
  return rec(data, 0, nextPrefix, lineWidth);
}

/** numpy.array_repr(a, precision=precision). */
export function arrayRepr(a: NDArray, precision = 8): string {
  const prefix = "array(";
  const lst = array2stringBody(a, precision, prefix, 1);
  const extras: string[] = [];
  if ((a.size === 0 && !(a.ndim === 1)) || a.size > THRESHOLD) extras.push(`shape=${shapeStr(a.shape)}`);
  if (!IMPLIED.has(a.dtype.name) || a.size === 0) extras.push(`dtype=${a.dtype.name}`);
  if (!extras.length) return `${prefix}${lst})`;
  const arrStr = `${prefix}${lst},`;
  const extra = extras.join(", ");
  const lastLen = arrStr.length - (arrStr.lastIndexOf("\n") + 1);
  const spacer = lastLen + extra.length + 1 > LINE_WIDTH ? "\n" + " ".repeat(prefix.length) : " ";
  return arrStr + spacer + extra + ")";
}

/** numpy.array2string of a 0-d array (default precision 8). */
export const array2string0d = (a: NDArray, precision = 8): string => array2stringBody(a, precision, "", 0);

/** Python repr of a float (`1.0`, `1e-05`, `1.5e+20`) using the dtype's shortest digits. */
export function pyFloat(x: number, kind = "float64"): string {
  if (Number.isNaN(x)) return "nan";
  if (!Number.isFinite(x)) return x < 0 ? "-inf" : "inf";
  const { d, e } = shortest(x, kind);
  const s = isNeg(x) ? "-" : "";
  if (e < -4 || e >= 16) {
    const m = d.length > 1 ? `${d[0]}.${d.slice(1)}` : d;
    return `${s}${m}e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`;
  }
  const p = positional(Math.abs(x), 17, kind, false);
  return s + (p.endsWith(".") ? p + "0" : p);
}

/** str() of a NumPy scalar of the given dtype. */
export function scalarStr(v: Elem, dtype: string): string {
  if (typeof v === "boolean") return v ? "True" : "False";
  if (v instanceof Complex) {
    const k = realKind(dtype);
    const part = (x: number) => pyFloat(x, k).replace(/\.0$/, "");
    if (v.re === 0 && !Object.is(v.re, -0)) return part(v.im) + "j";
    const im = part(v.im);
    return `(${part(v.re)}${im.startsWith("-") ? "" : "+"}${im}j)`;
  }
  if (typeof v === "bigint" || dtype === "int64" || dtype[0] === "i" || dtype[0] === "u") return String(v);
  return pyFloat(v, dtype);
}

/** Python `%.{p}g`. */
export function pyG(x: number, p: number): string {
  if (!Number.isFinite(x)) return Number.isNaN(x) ? "nan" : x < 0 ? "-inf" : "inf";
  if (x === 0) return "0";
  const [m, ex] = x.toExponential(p - 1).split("e");
  const e = Number(ex);
  if (e < -4 || e >= p) {
    const mm = (m ?? "").includes(".") ? (m ?? "").replace(/0+$/, "").replace(/\.$/, "") : (m ?? "");
    return `${mm}e${e < 0 ? "-" : "+"}${String(Math.abs(e)).padStart(2, "0")}`;
  }
  const f = x.toFixed(Math.max(0, p - 1 - e));
  return f.includes(".") ? f.replace(/0+$/, "").replace(/\.$/, "") : f;
}
