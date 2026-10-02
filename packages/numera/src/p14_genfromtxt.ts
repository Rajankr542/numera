// P14 genfromtxt and fromregex (D-171). Implemented in TS on top of the
// rules of NumPy's LineSplitter and StringConverter (numpy/lib/_iotools.py)
// for an explicit numeric dtype; values are converted with Python
// int()/float()/complex() syntax.
import { array } from "./creation.js";
import { complex, type Complex } from "./complex.js";
import { dtype as toDType, type DType, type DTypeLike } from "./dtype.js";
import { IndexError, ValueError } from "./errors.js";
import type { NDArray, NestedArray } from "./ndarray.js";
import { finishTable, readText, type TextSource } from "./p14_text.js";

type Cell = number | bigint | boolean | Complex;

// ---------------- Python literal parsers ----------------

const DIGITS = String.raw`\d(?:_?\d)*`;
const NUM = String.raw`(?:(?:${DIGITS})(?:\.(?:${DIGITS})?)?|\.${DIGITS})(?:[eE][+-]?${DIGITS})?`;
const FLOAT_RE = new RegExp(`^[+-]?${NUM}$`);
const SPECIAL_RE = /^([+-]?)(inf|infinity|nan)$/i;
const INT_RE = new RegExp(`^[+-]?${DIGITS}$`);
const CNUM = `(?:${NUM}|inf(?:inity)?|nan)`;
const C_REAL = new RegExp(`^([+-]?${CNUM})$`, "i");
const C_IMAG = new RegExp(`^([+-]?(?:${CNUM})?)j$`, "i");
const C_BOTH = new RegExp(`^([+-]?${CNUM})([+-](?:${CNUM})?)j$`, "i");

/** Python float(s); `undefined` when `s` is not a float literal. @internal */
export function pyFloat(s: string): number | undefined {
  const t = s.trim();
  const sp = SPECIAL_RE.exec(t);
  if (sp !== null) {
    const v = sp[2]!.toLowerCase() === "nan" ? NaN : Infinity;
    return sp[1] === "-" ? -v : v;
  }
  return FLOAT_RE.test(t) ? Number(t.replaceAll("_", "")) : undefined;
}

/** Python int(s); `undefined` when `s` is not a decimal integer literal. @internal */
export function pyInt(s: string): bigint | undefined {
  const t = s.trim();
  return INT_RE.test(t) ? BigInt(t.replaceAll("_", "")) : undefined;
}

/** Python complex(s); `undefined` when malformed. @internal */
export function pyComplex(s: string): Complex | undefined {
  let t = s.trim();
  if (t.startsWith("(") && t.endsWith(")")) t = t.slice(1, -1).trim();
  const part = (p: string): number => (p === "" || p === "+" ? 1 : p === "-" ? -1 : pyFloat(p)!);
  let m = C_REAL.exec(t);
  if (m !== null) return complex(pyFloat(m[1]!)!, 0);
  m = C_IMAG.exec(t);
  if (m !== null) return complex(0, part(m[1]!));
  m = C_BOTH.exec(t);
  if (m !== null) return complex(pyFloat(m[1]!)!, part(m[2]!));
  return undefined;
}

// ---------------- converters (StringConverter with a dtype) ----------------

/** Converts one string, or returns `undefined` for NumPy's ValueError. */
type Convert = (s: string) => Cell | undefined;

const isWide = (dt: DType): boolean => dt.name === "int64" || dt.name === "uint64";

function converterFor(dt: DType): Convert {
  switch (dt.kind) {
    case "b":
      return (s) => {
        const u = s.toUpperCase();
        return u === "TRUE" ? true : u === "FALSE" ? false : undefined;
      };
    case "f":
      return pyFloat;
    case "c":
      return pyComplex;
    default:
      if (isWide(dt)) {
        // np.int64(s) / np.uint64(s): int() syntax, out of range raises.
        return (s) => {
          const v = pyInt(s);
          if (v === undefined) return undefined;
          const ok = dt.name === "int64" ? v >= -(2n ** 63n) && v < 2n ** 63n : v >= 0n && v < 2n ** 64n;
          if (!ok) throw new ValueError(`Python integer ${v} out of bounds for ${dt.name}`);
          return v;
        };
      }
      // int(float(s)) for the narrower integer types.
      return (s) => {
        const f = pyFloat(s);
        if (f === undefined || Number.isNaN(f)) return undefined;
        if (!Number.isFinite(f)) throw new ValueError("cannot convert float infinity to integer");
        return Math.trunc(f);
      };
  }
}

function defaultFor(dt: DType): Cell {
  switch (dt.kind) {
    case "b":
      return false;
    case "f":
      return NaN;
    case "c":
      return complex(NaN, 0);
    default:
      return isWide(dt) ? -1n : -1;
  }
}

/** Normalizes a user value (fill value) to the cell type the dtype expects. */
function asCell(v: unknown, dt: DType): Cell {
  if (isWide(dt) && typeof v === "number" && Number.isInteger(v)) return BigInt(v);
  if (typeof v === "string") {
    const c = converterFor(dt)(v);
    if (c === undefined) throw new ValueError(`could not convert string '${v}' to ${dt.name}`);
    return c;
  }
  return v as Cell;
}

// ---------------- line splitting (LineSplitter) ----------------

const stripChars = (s: string, chars: string): string => {
  let a = 0;
  let b = s.length;
  while (a < b && chars.includes(s[a]!)) a++;
  while (b > a && chars.includes(s[b - 1]!)) b--;
  return s.slice(a, b);
};

type Splitter = (line: string) => string[];

function makeSplitter(delimiter: GenfromtxtOptions["delimiter"], comments: string | null, autostrip: boolean): Splitter {
  const uncomment = (line: string): string => (comments === null ? line : line.split(comments)[0]!);
  let split: Splitter;
  if (delimiter == null || delimiter === "" || delimiter === 0) {
    split = (line) => {
      const s = stripChars(uncomment(line), " \r\n");
      return s === "" ? [] : s.split(/\s+/).filter((f) => f !== "");
    };
  } else if (typeof delimiter === "string") {
    split = (line) => {
      const s = stripChars(uncomment(line), " \r\n");
      return s === "" ? [] : s.split(delimiter);
    };
  } else if (typeof delimiter === "number") {
    if (!Number.isInteger(delimiter) || delimiter < 0) throw new ValueError("delimiter width must be a positive integer");
    split = (line) => {
      const s = [...stripChars(uncomment(line), "\r\n")];
      const out: string[] = [];
      for (let i = 0; i < s.length; i += delimiter) out.push(s.slice(i, i + delimiter).join(""));
      return out;
    };
  } else {
    const widths = [...delimiter];
    split = (line) => {
      const s = [...uncomment(line)];
      if (s.length === 0) return [];
      const out: string[] = [];
      let start = 0;
      for (const w of widths) {
        out.push(s.slice(start, start + w).join(""));
        start += w;
      }
      return out;
    };
  }
  return autostrip ? (line) => split(line).map((f) => f.trim()) : split;
}

const toLines = (text: string): string[] => {
  const lines = text.split(/\r\n|\r|\n/);
  if (lines.length > 0 && lines[lines.length - 1] === "") lines.pop();
  return lines;
};

// ---------------- genfromtxt ----------------

/** Per-column settings: one value for every column, a list (by position) or a `{column: value}` map. */
export type PerColumn<T> = T | readonly T[] | Readonly<Record<number, T>> | ReadonlyMap<number | null, T>;

export interface GenfromtxtOptions {
  /** Default float64. Numeric and bool dtypes only (no type inference). */
  dtype?: DTypeLike;
  /** Comment marker (default "#"); `null` disables comments. */
  comments?: string | null;
  /** A separator string, a field width, or a list of field widths. Default: whitespace. */
  delimiter?: string | number | readonly number[] | null;
  /** Lines to skip at the start. */
  skipHeader?: number;
  /** Lines to drop at the end. */
  skipFooter?: number;
  /** Strings that mark a missing value. A string is split at ",". A Map key `null` applies to all columns. */
  missingValues?: string | readonly (string | number)[] | Readonly<Record<number, string | number | readonly (string | number)[]>> | ReadonlyMap<number | null, string | number | readonly (string | number)[]>;
  /** Values used for missing or invalid cells. */
  fillingValues?: PerColumn<number | bigint | boolean | string | Complex>;
  /** Column index or indices to read. */
  usecols?: number | readonly number[] | null;
  /** Strip spaces around fields (default false). */
  autostrip?: boolean;
  /** Raise on lines with a wrong number of columns (default true); otherwise drop them with a warning. */
  invalidRaise?: boolean;
  /** Read at most this many rows. */
  maxRows?: number | null;
  /** Use filling values for unconvertible cells (default true); `false` raises unless the cell is a missing value. */
  loose?: boolean;
  /** Transpose the result. */
  unpack?: boolean;
  /** Minimum number of dimensions: 0, 1 or 2. */
  ndmin?: number;
}

const isMap = (v: unknown): v is ReadonlyMap<unknown, unknown> => v instanceof Map;
const isPlainObject = (v: unknown): v is Record<string, unknown> =>
  typeof v === "object" && v !== null && !Array.isArray(v) && !isMap(v) && Object.getPrototypeOf(v) === Object.prototype;

const mapEntries = (v: unknown): [number | null, unknown][] =>
  isMap(v)
    ? [...v.entries()].map(([k, x]) => [k === null ? null : Number(k), x])
    : Object.entries(v as Record<string, unknown>).map(([k, x]) => [Number(k), x]);

const pyStr = (v: unknown): string => String(v);

/**
 * np.genfromtxt(fname, options): reads a table, filling missing or invalid
 * cells. Unlike `loadtxt`, unconvertible cells become filling values (NaN for
 * floats, -1 for integers, false for bool) unless `loose` is false.
 */
export function genfromtxt(fname: TextSource, options: GenfromtxtOptions = {}): NDArray {
  if ((options.dtype as unknown) === null) {
    throw new ValueError("genfromtxt: dtype=null (type inference) is not supported; pass a numeric dtype");
  }
  const dt = toDType(options.dtype ?? "float64");
  const ndmin = options.ndmin ?? 0;
  if (ndmin !== 0 && ndmin !== 1 && ndmin !== 2) throw new ValueError(`Illegal value of ndmin keyword: ${ndmin}`);
  const skipHeader = options.skipHeader ?? 0;
  let skipFooter = options.skipFooter ?? 0;
  const maxRows = options.maxRows ?? null;
  if (maxRows !== null) {
    if (skipFooter) throw new ValueError("The keywords 'skip_footer' and 'max_rows' can not be specified at the same time.");
    if (maxRows < 1) throw new ValueError("'max_rows' must be at least 1.");
  }
  const comments = options.comments === undefined ? "#" : options.comments;
  const split = makeSplitter(options.delimiter, comments, options.autostrip ?? false);
  const lines = toLines(readText(fname));

  let pos = Math.min(skipHeader, lines.length);
  let firstValues: string[] = [];
  while (pos < lines.length && firstValues.length === 0) firstValues = split(lines[pos++]!);
  if (firstValues.length === 0) {
    process.emitWarning("genfromtxt: Empty input file", "UserWarning");
    return finishTable(array([], { dtype: dt }), ndmin, options.unpack ?? false);
  }
  const firstIndex = pos - 1;

  const u = options.usecols;
  let usecols: number[] | null = u == null ? null : typeof u === "number" ? [u] : [...u];
  if (usecols !== null && usecols.length === 0) usecols = null;
  if (usecols !== null) usecols = usecols.map((c) => (c < 0 ? c + firstValues.length : c));
  const nbcols = usecols?.length ?? firstValues.length;
  const remap = (k: number): number => {
    const j = usecols === null ? -1 : usecols.indexOf(k);
    return j >= 0 ? j : k;
  };
  const checkKey = (k: number): number => {
    if (!Number.isInteger(k) || k >= nbcols || k < -nbcols) throw new IndexError("list index out of range");
    return k < 0 ? k + nbcols : k;
  };

  // missing values per output column
  const missing: string[][] = Array.from({ length: nbcols }, () => [""]);
  const mv = options.missingValues;
  if (mv != null) {
    if (isMap(mv) || isPlainObject(mv)) {
      for (const [key, val] of mapEntries(mv)) {
        const vals = Array.isArray(val) ? val.map(pyStr) : [pyStr(val)];
        if (key === null) missing.forEach((m) => m.push(...vals));
        else missing[checkKey(remap(key))]!.push(...vals);
      }
    } else if (Array.isArray(mv)) {
      mv.forEach((v, j) => {
        if (j < nbcols && !missing[j]!.includes(pyStr(v))) missing[j]!.push(pyStr(v));
      });
    } else if (typeof mv === "string") {
      missing.forEach((m) => m.push(...mv.split(",")));
    } else {
      missing.forEach((m) => m.push(pyStr(mv)));
    }
  }

  // filling values per output column
  const fills: unknown[] = new Array<unknown>(nbcols).fill(undefined);
  const fv = options.fillingValues;
  if (fv != null) {
    if (isMap(fv) || isPlainObject(fv)) {
      for (const [key, val] of mapEntries(fv)) {
        if (key === null) continue;
        fills[checkKey(remap(key))] = val;
      }
    } else if (Array.isArray(fv)) {
      fv.slice(0, nbcols).forEach((v, j) => (fills[j] = v));
    } else {
      fills.fill(fv);
    }
  }
  const defaults = fills.map((f) => (f == null ? defaultFor(dt) : asCell(f, dt)));

  // split rows
  const rows: string[][] = [];
  const invalid: [number, number][] = [];
  for (let i = 0, p = firstIndex; p < lines.length; i++, p++) {
    const values = p === firstIndex ? firstValues : split(lines[p]!);
    const nb = values.length;
    if (nb === 0) continue;
    if (usecols !== null) {
      if (usecols.some((c) => c >= nb || c < -nb)) {
        invalid.push([i + skipHeader + 1, nb]);
        continue;
      }
      rows.push(usecols.map((c) => values[c < 0 ? c + nb : c]!));
    } else if (nb !== nbcols) {
      invalid.push([i + skipHeader + 1, nb]);
      continue;
    } else {
      rows.push(values);
    }
    if (rows.length === maxRows) break;
  }

  let bad = invalid;
  if (bad.length > 0) {
    const nbrows = rows.length + bad.length - skipFooter;
    if (skipFooter > 0) {
      const skipped = bad.filter(([ln]) => ln > nbrows + skipHeader).length;
      bad = bad.slice(0, bad.length - skipped);
      skipFooter -= skipped;
    }
    if (bad.length > 0) {
      const msg = ["Some errors were detected !",
        ...bad.map(([ln, nb]) => `    Line #${ln} (got ${nb} columns instead of ${nbcols})`)].join("\n");
      if (options.invalidRaise ?? true) throw new ValueError(msg);
      process.emitWarning(msg, "ConversionWarning");
    }
  }
  const kept = skipFooter > 0 ? rows.slice(0, Math.max(0, rows.length - skipFooter)) : rows;

  const conv = converterFor(dt);
  const loose = options.loose ?? true;
  const data: Cell[][] = kept.map((row) =>
    row.map((s, j) => {
      const v = conv(s);
      if (v !== undefined) return v;
      if (loose || missing[j]!.includes(s.trim())) return defaults[j]!;
      throw new ValueError(`Cannot convert string '${s}'`);
    }),
  );
  const out = array((data.length === 0 ? [] : data) as NestedArray, { dtype: dt });
  return finishTable(out, ndmin, options.unpack ?? false);
}

// ---------------- fromregex ----------------

/** A structured dtype description: `[fieldName, dtype]` pairs. */
export type FieldList = readonly (readonly [string, DTypeLike])[];

function fromPyString(s: string, dt: DType): Cell {
  switch (dt.kind) {
    case "b":
      return s.length > 0;
    case "f": {
      const v = pyFloat(s);
      if (v === undefined) throw new ValueError(`could not convert string to float: '${s}'`);
      return v;
    }
    case "c": {
      const v = pyComplex(s);
      if (v === undefined) throw new ValueError("complex() arg is a malformed string");
      return v;
    }
    default: {
      const v = pyInt(s);
      if (v === undefined) throw new ValueError(`invalid literal for int() with base 10: '${s}'`);
      return isWide(dt) ? v : Number(v);
    }
  }
}

/**
 * np.fromregex(file, regexp, dtype): every match of `regexp` in the text is
 * one record; capture groups map to the fields of `dtype`. Returns one 1-D
 * array per field (structured arrays are not available).
 */
export function fromregex(file: TextSource, regexp: string | RegExp, dtype: FieldList): Record<string, NDArray> {
  if (!Array.isArray(dtype) || dtype.length === 0 || !dtype.every((f) => Array.isArray(f) && f.length === 2)) {
    throw new TypeError("dtype must be a structured datatype.");
  }
  const fields = dtype.map(([name, d]) => [name, toDType(d)] as const);
  const re = typeof regexp === "string"
    ? new RegExp(regexp, "g")
    : new RegExp(regexp.source, regexp.flags.includes("g") ? regexp.flags : regexp.flags + "g");
  const text = readText(file);
  const records: string[][] = [];
  for (const m of text.matchAll(re)) {
    records.push(m.length === 1 ? [m[0]] : m.slice(1).map((g) => g ?? ""));
  }
  for (const r of records) {
    if (r.length !== fields.length) {
      throw new ValueError(`could not assign tuple of length ${r.length} to structure with ${fields.length} fields.`);
    }
  }
  const out: Record<string, NDArray> = {};
  fields.forEach(([name, dt], j) => {
    out[name] = array(records.map((r) => fromPyString(r[j]!, dt)) as NestedArray, { dtype: dt });
  });
  return out;
}
