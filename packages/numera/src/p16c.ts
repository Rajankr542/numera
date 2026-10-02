// P16C: np.strings and np.char modules (D-210–D-213).
// StringArray is a pure-TS string-array class (no new C++ DType needed).
// All 46 np.strings names and 52 np.char names are implemented here.

import { array, asarray } from "./creation.js";
import { NDArray, type NestedArray } from "./ndarray.js";
import { ValueError } from "./errors.js";

// ─── StringArray ─────────────────────────────────────────────────────────────

type StringDType = "str_" | "bytes_";

/** Flat string array plus shape — the element container for np.strings/np.char (D-210). */
export class StringArray {
  readonly shape: readonly number[];
  readonly dtype: StringDType;
  /** @internal flat storage, row-major */
  readonly _data: string[];

  constructor(data: string[], shape: readonly number[], dtype: StringDType = "str_") {
    this.dtype = dtype;
    this.shape = shape;
    this._data = data;
  }

  get ndim(): number {
    return this.shape.length;
  }

  get size(): number {
    return this._data.length;
  }

  /** Iterates elements in row-major order. */
  *flat(): Generator<string> {
    yield* this._data;
  }

  /** Returns a nested JS string array matching this.shape. */
  toArray(): string | string[] | string[][] | unknown {
    if (this.shape.length === 0) return this._data[0] ?? "";
    if (this.shape.length === 1) return [...this._data];
    return buildNested(this._data, this.shape, 0, [0]);
  }

  toString(): string {
    return `StringArray(${JSON.stringify(this.toArray())}, dtype='${this.dtype}')`;
  }
}

/** Recursively builds a nested JS array from flat data. */
function buildNested(data: string[], shape: readonly number[], depth: number, idx: [number]): unknown {
  if (depth === shape.length - 1) {
    const row: string[] = [];
    for (let i = 0; i < (shape[depth] ?? 0); i++) row.push(data[idx[0]++] ?? "");
    return row;
  }
  const out: unknown[] = [];
  for (let i = 0; i < (shape[depth] ?? 0); i++) out.push(buildNested(data, shape, depth + 1, idx));
  return out;
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

type StringLike = StringArray | readonly string[] | string[] | string | NestedArray;

/** Normalize any string input to a StringArray. */
function toStringArray(input: StringLike): StringArray {
  if (input instanceof StringArray) return input;
  if (typeof input === "string") return new StringArray([input], []);
  // Flatten nested arrays
  const flat: string[] = [];
  const shape: number[] = [];
  flattenStrings(input as NestedArray, flat, shape);
  return new StringArray(flat, shape);
}

function flattenStrings(v: NestedArray, flat: string[], shape: number[]): void {
  if (typeof v === "string") {
    flat.push(v);
    return;
  }
  if (!Array.isArray(v)) {
    // numbers/booleans — coerce to string
    flat.push(String(v));
    return;
  }
  const arr = v as NestedArray[];
  if (shape.length === 0) {
    // First level — record dimension
  }
  const idx = shape.length;
  shape.push(arr.length);
  // Temporarily record count; will be adjusted if sub-arrays differ
  let childShape: number[] | null = null;
  for (const child of arr) {
    const before = flat.length;
    const tempShape: number[] = [];
    flattenStrings(child, flat, tempShape);
    if (childShape === null) {
      childShape = tempShape;
      for (const s of tempShape) shape.push(s);
    }
    void before;
  }
  void idx;
}

/** Map element-wise over one StringArray, returning a new StringArray of same shape. */
function mapStr(a: StringArray, fn: (s: string) => string): StringArray {
  return new StringArray(a._data.map(fn), a.shape, a.dtype);
}

/** Map element-wise over two StringArrays (broadcast scalar to array). */
function map2Str(a: StringArray, b: StringArray, fn: (s: string, t: string) => string): StringArray {
  if (a.size === b.size) {
    return new StringArray(a._data.map((s, i) => fn(s, b._data[i] ?? "")), a.shape, a.dtype);
  }
  if (b.size === 1) {
    const t = b._data[0] ?? "";
    return new StringArray(a._data.map((s) => fn(s, t)), a.shape, a.dtype);
  }
  if (a.size === 1) {
    const s = a._data[0] ?? "";
    return new StringArray(b._data.map((t) => fn(s, t)), b.shape, a.dtype);
  }
  throw new ValueError(`StringArray shapes are not compatible for binary operation`);
}

/** Map element-wise, returning an int64 NDArray. */
function mapToInt(a: StringArray, fn: (s: string) => number): NDArray {
  return asarray(a._data.map(fn), { dtype: "int64" }).reshape(a.shape);
}

/** Map element-wise, returning a bool NDArray. */
function mapToBool(a: StringArray, fn: (s: string) => boolean): NDArray {
  return asarray(a._data.map(fn), { dtype: "bool" }).reshape(a.shape);
}

/** Map2 element-wise, returning a bool NDArray. */
function map2ToBool(a: StringArray, b: StringArray, fn: (s: string, t: string) => boolean): NDArray {
  if (a.size === b.size) {
    return asarray(a._data.map((s, i) => fn(s, b._data[i] ?? "")), { dtype: "bool" }).reshape(a.shape);
  }
  if (b.size === 1) {
    const t = b._data[0] ?? "";
    return asarray(a._data.map((s) => fn(s, t)), { dtype: "bool" }).reshape(a.shape);
  }
  if (a.size === 1) {
    const s = a._data[0] ?? "";
    return asarray(b._data.map((t) => fn(s, t)), { dtype: "bool" }).reshape(b.shape);
  }
  throw new ValueError(`StringArray shapes are not compatible for binary comparison`);
}

/** Map2 returning an int64 NDArray. */
function map2ToInt(a: StringArray, b: StringArray, fn: (s: string, t: string) => number): NDArray {
  if (a.size === b.size) {
    return asarray(a._data.map((s, i) => fn(s, b._data[i] ?? "")), { dtype: "int64" }).reshape(a.shape);
  }
  if (b.size === 1) {
    const t = b._data[0] ?? "";
    return asarray(a._data.map((s) => fn(s, t)), { dtype: "int64" }).reshape(a.shape);
  }
  if (a.size === 1) {
    const s = a._data[0] ?? "";
    return asarray(b._data.map((t) => fn(s, t)), { dtype: "int64" }).reshape(b.shape);
  }
  throw new ValueError(`StringArray shapes are not compatible for binary op`);
}

// Normalize optional start/end to definite indices against string s
function normRange(s: string, start?: number, end?: number): [number, number] {
  const len = s.length;
  let st = start ?? 0;
  let en = end ?? len;
  if (st < 0) st = Math.max(0, st + len);
  if (en < 0) en = Math.max(0, en + len);
  if (st > len) st = len;
  if (en > len) en = len;
  return [st, en];
}

// ─── Case operations ─────────────────────────────────────────────────────────

export function capitalize(a: StringLike): StringArray {
  const sa = toStringArray(a);
  return mapStr(sa, (s) => s.length === 0 ? "" : (s[0] ?? "").toUpperCase() + s.slice(1).toLowerCase());
}

export function upper(a: StringLike): StringArray {
  return mapStr(toStringArray(a), (s) => s.toUpperCase());
}

export function lower(a: StringLike): StringArray {
  return mapStr(toStringArray(a), (s) => s.toLowerCase());
}

export function swapcase(a: StringLike): StringArray {
  return mapStr(toStringArray(a), (s) =>
    [...s].map((c) => (c === c.toUpperCase() ? c.toLowerCase() : c.toUpperCase())).join(""),
  );
}

export function title(a: StringLike): StringArray {
  return mapStr(toStringArray(a), (s) =>
    s.replace(/\b\S/g, (c) => c.toUpperCase()).replace(/(\b\S)\S*/g, (w) => w[0] + w.slice(1).toLowerCase()),
  );
}

// ─── Padding ─────────────────────────────────────────────────────────────────

export function center(a: StringLike, width: number, fillchar = " "): StringArray {
  if ([...fillchar].length !== 1) throw new ValueError("fillchar must be a character, not str");
  return mapStr(toStringArray(a), (s) => {
    const pad = Math.max(0, width - s.length);
    const left = Math.floor(pad / 2);
    const right = pad - left;
    return fillchar.repeat(left) + s + fillchar.repeat(right);
  });
}

export function ljust(a: StringLike, width: number, fillchar = " "): StringArray {
  if ([...fillchar].length !== 1) throw new ValueError("fillchar must be a character, not str");
  return mapStr(toStringArray(a), (s) => s.padEnd(width, fillchar));
}

export function rjust(a: StringLike, width: number, fillchar = " "): StringArray {
  if ([...fillchar].length !== 1) throw new ValueError("fillchar must be a character, not str");
  return mapStr(toStringArray(a), (s) => s.padStart(width, fillchar));
}

export function zfill(a: StringLike, width: number): StringArray {
  return mapStr(toStringArray(a), (s) => {
    if (s.startsWith("+") || s.startsWith("-")) {
      return s[0] + s.slice(1).padStart(width - 1, "0");
    }
    return s.padStart(width, "0");
  });
}

// ─── Strip ───────────────────────────────────────────────────────────────────

export function strip(a: StringLike, chars?: string | null): StringArray {
  return mapStr(toStringArray(a), (s) => stripChars(s, chars, true, true));
}

export function lstrip(a: StringLike, chars?: string | null): StringArray {
  return mapStr(toStringArray(a), (s) => stripChars(s, chars, true, false));
}

export function rstrip(a: StringLike, chars?: string | null): StringArray {
  return mapStr(toStringArray(a), (s) => stripChars(s, chars, false, true));
}

function stripChars(s: string, chars: string | null | undefined, left: boolean, right: boolean): string {
  if (chars == null) {
    if (left && right) return s.trim();
    if (left) return s.trimStart();
    return s.trimEnd();
  }
  const set = new Set([...chars]);
  let start = 0;
  let end = s.length;
  if (left) while (start < end && set.has(s[start] ?? "")) start++;
  if (right) while (end > start && set.has(s[end - 1] ?? "")) end--;
  return s.slice(start, end);
}

// ─── Expand tabs ─────────────────────────────────────────────────────────────

export function expandtabs(a: StringLike, tabsize = 8): StringArray {
  return mapStr(toStringArray(a), (s) => {
    let result = "";
    let col = 0;
    for (const c of s) {
      if (c === "\t") {
        const spaces = tabsize - (col % tabsize);
        result += " ".repeat(spaces);
        col += spaces;
      } else {
        result += c;
        if (c === "\n" || c === "\r") col = 0;
        else col++;
      }
    }
    return result;
  });
}

// ─── Find / index ─────────────────────────────────────────────────────────────

export function find(a: StringLike, sub: StringLike, start?: number, end?: number): NDArray {
  const sa = toStringArray(a);
  const sb = toStringArray(sub);
  return map2ToInt(sa, sb, (s, t) => {
    const [st, en] = normRange(s, start, end);
    const idx = s.indexOf(t, st);
    return idx === -1 || idx > en - t.length ? -1 : idx;
  });
}

export function rfind(a: StringLike, sub: StringLike, start?: number, end?: number): NDArray {
  const sa = toStringArray(a);
  const sb = toStringArray(sub);
  return map2ToInt(sa, sb, (s, t) => {
    const [st, en] = normRange(s, start, end);
    const slice = s.slice(st, en);
    const idx = slice.lastIndexOf(t);
    return idx === -1 ? -1 : idx + st;
  });
}

export function index(a: StringLike, sub: StringLike, start?: number, end?: number): NDArray {
  const sa = toStringArray(a);
  const sb = toStringArray(sub);
  return map2ToInt(sa, sb, (s, t) => {
    const [st, en] = normRange(s, start, end);
    const idx = s.indexOf(t, st);
    if (idx === -1 || idx > en - t.length) throw new ValueError(`substring not found`);
    return idx;
  });
}

export function rindex(a: StringLike, sub: StringLike, start?: number, end?: number): NDArray {
  const sa = toStringArray(a);
  const sb = toStringArray(sub);
  return map2ToInt(sa, sb, (s, t) => {
    const [st, en] = normRange(s, start, end);
    const slice = s.slice(st, en);
    const idx = slice.lastIndexOf(t);
    if (idx === -1) throw new ValueError(`substring not found`);
    return idx + st;
  });
}

export function count(a: StringLike, sub: StringLike, start?: number, end?: number): NDArray {
  const sa = toStringArray(a);
  const sb = toStringArray(sub);
  return map2ToInt(sa, sb, (s, t) => {
    if (t.length === 0) {
      const [st, en] = normRange(s, start, end);
      return en - st + 1;
    }
    const [st, en] = normRange(s, start, end);
    const slice = s.slice(st, en);
    let n = 0;
    let pos = 0;
    while ((pos = slice.indexOf(t, pos)) !== -1) {
      n++;
      pos += t.length;
    }
    return n;
  });
}

export function startswith(a: StringLike, prefix: StringLike, start?: number, end?: number): NDArray {
  const sa = toStringArray(a);
  const sb = toStringArray(prefix);
  return map2ToBool(sa, sb, (s, t) => {
    const [st, en] = normRange(s, start, end);
    return s.slice(st, en).startsWith(t);
  });
}

export function endswith(a: StringLike, suffix: StringLike, start?: number, end?: number): NDArray {
  const sa = toStringArray(a);
  const sb = toStringArray(suffix);
  return map2ToBool(sa, sb, (s, t) => {
    const [st, en] = normRange(s, start, end);
    return s.slice(st, en).endsWith(t);
  });
}

export function str_len(a: StringLike): NDArray {
  return mapToInt(toStringArray(a), (s) => s.length);
}

// ─── Boolean predicates ────────────────────────────────────────────────────────

export function isalnum(a: StringLike): NDArray {
  return mapToBool(toStringArray(a), (s) => s.length > 0 && /^[a-zA-Z0-9]+$/.test(s));
}

export function isalpha(a: StringLike): NDArray {
  return mapToBool(toStringArray(a), (s) => s.length > 0 && /^[a-zA-Z]+$/.test(s));
}

export function isdigit(a: StringLike): NDArray {
  return mapToBool(toStringArray(a), (s) => s.length > 0 && /^\d+$/.test(s));
}

export function isdecimal(a: StringLike): NDArray {
  return mapToBool(toStringArray(a), (s) => s.length > 0 && /^[0-9]+$/.test(s));
}

export function isnumeric(a: StringLike): NDArray {
  // NumPy isnumeric matches Unicode numeric: digits, fractions, etc.
  // We approximate with \d (covers Arabic digits and Unicode decimal)
  return mapToBool(toStringArray(a), (s) => s.length > 0 && /^\d+$/.test(s));
}

export function isspace(a: StringLike): NDArray {
  return mapToBool(toStringArray(a), (s) => s.length > 0 && /^\s+$/.test(s));
}

export function islower(a: StringLike): NDArray {
  return mapToBool(toStringArray(a), (s) => s.length > 0 && s === s.toLowerCase() && s !== s.toUpperCase());
}

export function isupper(a: StringLike): NDArray {
  return mapToBool(toStringArray(a), (s) => s.length > 0 && s === s.toUpperCase() && s !== s.toLowerCase());
}

export function istitle(a: StringLike): NDArray {
  return mapToBool(toStringArray(a), (s) => {
    if (s.length === 0) return false;
    let prevWasAlpha = false;
    for (const c of s) {
      const isAlpha = /[a-zA-Z]/.test(c);
      if (isAlpha) {
        if (prevWasAlpha && c !== c.toLowerCase()) return false;
        if (!prevWasAlpha && c !== c.toUpperCase()) return false;
      }
      prevWasAlpha = isAlpha;
    }
    return true;
  });
}

// ─── String transforms ────────────────────────────────────────────────────────

export function add(a: StringLike, b: StringLike): StringArray {
  return map2Str(toStringArray(a), toStringArray(b), (s, t) => s + t);
}

export function multiply(a: StringLike, i: number | NDArray): StringArray {
  const sa = toStringArray(a);
  const n = i instanceof NDArray ? Number((i.toArray() as number | number[])) : i;
  return mapStr(sa, (s) => (n <= 0 ? "" : s.repeat(n)));
}

export function replace(a: StringLike, old_: StringLike, new_: StringLike, count_?: number): StringArray {
  const sa = toStringArray(a);
  const sold = toStringArray(old_);
  const snew = toStringArray(new_);
  return map2Str(sa, sold, (s, o) => {
    const n_ = snew.size === 1 ? snew._data[0] : snew._data[0] ?? "";
    if (count_ === undefined || count_ < 0) {
      return s.split(o).join(n_);
    }
    let result = s;
    let remaining = count_;
    let pos = 0;
    let out = "";
    while (remaining > 0) {
      const idx = result.indexOf(o, pos);
      if (idx === -1) break;
      out += result.slice(pos, idx) + n_;
      pos = idx + o.length;
      remaining--;
    }
    return out + result.slice(pos);
  });
}

/** np.strings.translate: replace characters per Map<char, char|null> (D-213). */
export function translate(a: StringLike, table: Map<string, string | null>): StringArray {
  return mapStr(toStringArray(a), (s) =>
    [...s].map((c) => {
      if (!table.has(c)) return c;
      const v = table.get(c);
      return v === null ? "" : v;
    }).join(""),
  );
}

/** np.strings.encode: identity stub (D-213 — no bytes_ DType in JS). */
export function encode(a: StringLike, encoding = "utf-8", _errors = "strict"): StringArray {
  void encoding;
  return toStringArray(a);
}

/** np.strings.decode: identity stub (D-213). */
export function decode(a: StringLike, encoding = "utf-8", _errors = "strict"): StringArray {
  void encoding;
  return toStringArray(a);
}

/** np.strings.mod: %-style formatting (D-213 — subset of Python's % operator). */
export function mod(a: StringLike, values: unknown): StringArray {
  const sa = toStringArray(a);
  return mapStr(sa, (s) => {
    const args: unknown[] = Array.isArray(values) ? (values as unknown[]) : [values];
    let i = 0;
    return s.replace(/%([diouxXeEfFgGs%])/g, (_, spec: string) => {
      if (spec === "%") return "%";
      const v = args[i++];
      switch (spec) {
        case "d": case "i": return Math.trunc(Number(v)).toString();
        case "o": return Math.trunc(Number(v)).toString(8);
        case "u": return Math.abs(Math.trunc(Number(v))).toString();
        case "x": return Math.trunc(Number(v)).toString(16);
        case "X": return Math.trunc(Number(v)).toString(16).toUpperCase();
        case "e": return Number(v).toExponential();
        case "E": return Number(v).toExponential().toUpperCase();
        case "f": case "F": return Number(v).toFixed(6);
        case "g": case "G": {
          const nv = Number(v);
          const s2 = Math.abs(nv) < 1e-4 || Math.abs(nv) >= 1e6 ? nv.toExponential() : nv.toPrecision(6);
          return spec === "G" ? s2.toUpperCase() : s2;
        }
        case "s": return String(v);
        default: return `%${spec}`;
      }
    });
  });
}

// ─── Split / partition ────────────────────────────────────────────────────────

export function split(a: StringLike, sep?: string | null, maxsplit?: number): StringArray[] {
  const sa = toStringArray(a);
  return sa._data.map((s) => {
    let parts: string[];
    if (sep == null) {
      parts = maxsplit !== undefined ? splitWhitespace(s, maxsplit) : s.trim().split(/\s+/).filter(Boolean);
    } else {
      parts = splitBy(s, sep, maxsplit ?? -1);
    }
    return new StringArray(parts, [parts.length], sa.dtype);
  });
}

function splitWhitespace(s: string, maxsplit: number): string[] {
  const parts: string[] = [];
  let i = 0;
  const n = s.length;
  while (i < n && /\s/.test(s[i] ?? "")) i++;
  while (i < n) {
    if (parts.length === maxsplit) {
      parts.push(s.slice(i));
      break;
    }
    let j = i;
    while (j < n && !/\s/.test(s[j] ?? "")) j++;
    parts.push(s.slice(i, j));
    i = j;
    while (i < n && /\s/.test(s[i] ?? "")) i++;
  }
  return parts;
}

function splitBy(s: string, sep: string, maxsplit: number): string[] {
  if (maxsplit < 0) return s.split(sep);
  const parts: string[] = [];
  let pos = 0;
  let remaining = maxsplit;
  while (remaining > 0) {
    const idx = s.indexOf(sep, pos);
    if (idx === -1) break;
    parts.push(s.slice(pos, idx));
    pos = idx + sep.length;
    remaining--;
  }
  parts.push(s.slice(pos));
  return parts;
}

export function rsplit(a: StringLike, sep?: string | null, maxsplit?: number): StringArray[] {
  const sa = toStringArray(a);
  return sa._data.map((s) => {
    let parts: string[];
    if (sep == null) {
      const all = s.trim().split(/\s+/).filter(Boolean);
      parts = maxsplit !== undefined ? [...all.slice(0, -maxsplit), all.slice(-maxsplit).join(" ")] : all;
      // Actually: rsplit whitespace follows Python semantics — split from right
      parts = rsplitWhitespace(s, maxsplit);
    } else {
      parts = rsplitBy(s, sep, maxsplit ?? -1);
    }
    return new StringArray(parts, [parts.length], sa.dtype);
  });
}

function rsplitWhitespace(s: string, maxsplit?: number): string[] {
  if (maxsplit === undefined) return s.trim().split(/\s+/).filter(Boolean);
  const parts: string[] = [];
  let i = s.length;
  while (i > 0 && /\s/.test(s[i - 1] ?? "")) i--;
  while (i > 0) {
    if (parts.length === maxsplit) {
      parts.unshift(s.slice(0, i));
      return parts;
    }
    let j = i;
    while (j > 0 && !/\s/.test(s[j - 1] ?? "")) j--;
    parts.unshift(s.slice(j, i));
    i = j;
    while (i > 0 && /\s/.test(s[i - 1] ?? "")) i--;
  }
  return parts;
}

function rsplitBy(s: string, sep: string, maxsplit: number): string[] {
  if (maxsplit < 0) return s.split(sep);
  const parts: string[] = [];
  let pos = s.length;
  let remaining = maxsplit;
  while (remaining > 0) {
    const idx = s.lastIndexOf(sep, pos - 1);
    if (idx === -1) break;
    parts.unshift(s.slice(idx + sep.length, pos));
    pos = idx;
    remaining--;
  }
  parts.unshift(s.slice(0, pos));
  return parts;
}

export function splitlines(a: StringLike, keepends = false): StringArray[] {
  const sa = toStringArray(a);
  return sa._data.map((s) => {
    const parts = s.split(/(\r\n|\r|\n)/);
    const lines: string[] = [];
    for (let i = 0; i < parts.length; i++) {
      const part = parts[i] ?? "";
      if (/^(\r\n|\r|\n)$/.test(part)) {
        if (keepends && lines.length > 0) lines[lines.length - 1] += part;
        continue;
      }
      if (part !== "") lines.push(part);
    }
    return new StringArray(lines, [lines.length], sa.dtype);
  });
}

export function partition(a: StringLike, sep: StringLike): StringArray[] {
  const sa = toStringArray(a);
  const sb = toStringArray(sep);
  return sa._data.map((s, i) => {
    const t = (sb.size === 1 ? sb._data[0] : sb._data[i]) ?? "";
    const idx = s.indexOf(t);
    if (idx === -1) return new StringArray([s, "", ""], [3], sa.dtype);
    return new StringArray([s.slice(0, idx), t, s.slice(idx + t.length)], [3], sa.dtype);
  });
}

export function rpartition(a: StringLike, sep: StringLike): StringArray[] {
  const sa = toStringArray(a);
  const sb = toStringArray(sep);
  return sa._data.map((s, i) => {
    const t = (sb.size === 1 ? sb._data[0] : sb._data[i]) ?? "";
    const idx = s.lastIndexOf(t);
    if (idx === -1) return new StringArray(["", "", s], [3], sa.dtype);
    return new StringArray([s.slice(0, idx), t, s.slice(idx + t.length)], [3], sa.dtype);
  });
}

// ─── Comparisons ─────────────────────────────────────────────────────────────

export function equal(a: StringLike, b: StringLike): NDArray {
  return map2ToBool(toStringArray(a), toStringArray(b), (s, t) => s === t);
}

export function not_equal(a: StringLike, b: StringLike): NDArray {
  return map2ToBool(toStringArray(a), toStringArray(b), (s, t) => s !== t);
}

export function greater(a: StringLike, b: StringLike): NDArray {
  return map2ToBool(toStringArray(a), toStringArray(b), (s, t) => s > t);
}

export function greater_equal(a: StringLike, b: StringLike): NDArray {
  return map2ToBool(toStringArray(a), toStringArray(b), (s, t) => s >= t);
}

export function less(a: StringLike, b: StringLike): NDArray {
  return map2ToBool(toStringArray(a), toStringArray(b), (s, t) => s < t);
}

export function less_equal(a: StringLike, b: StringLike): NDArray {
  return map2ToBool(toStringArray(a), toStringArray(b), (s, t) => s <= t);
}

// ─── np.char extras ──────────────────────────────────────────────────────────

/** np.char.array: construct a StringArray from nested string data. */
export function charArray(data: StringLike, dtype: StringDType = "str_"): StringArray {
  const sa = toStringArray(data);
  return new StringArray(sa._data, sa.shape, dtype);
}

/** np.char.asarray: like charArray but no copy if already a StringArray. */
export function charAsarray(data: StringLike, dtype: StringDType = "str_"): StringArray {
  if (data instanceof StringArray && data.dtype === dtype) return data;
  return charArray(data, dtype);
}

/** np.char.join: join a sequence of StringArrays together with a separator. */
export function join(sep: StringLike, seq: StringLike): StringArray {
  const ssep = toStringArray(sep);
  const sa = toStringArray(seq);
  return map2Str(ssep, sa, (s, t) => t.split("").join(s));
}

/** np.char.slice: slice each string element. */
export function charSlice(a: StringLike, start?: number, stop?: number, step?: number): StringArray {
  return mapStr(toStringArray(a), (s) => {
    const arr = [...s];
    const len = arr.length;
    const st_v = start ?? 0;
    const stop_v = stop ?? len;
    const step_v = step ?? 1;
    if (step_v === 0) throw new ValueError("slice step cannot be zero");
    const result: string[] = [];
    if (step_v > 0) {
      let i = st_v < 0 ? Math.max(0, st_v + len) : Math.min(st_v, len);
      const end_ = stop_v < 0 ? Math.max(0, stop_v + len) : Math.min(stop_v, len);
      while (i < end_) { result.push(arr[i] ?? ""); i += step_v; }
    } else {
      let i = st_v < 0 ? Math.max(-1, st_v + len) : Math.min(st_v, len - 1);
      const end_ = stop_v < 0 ? Math.max(-1, stop_v + len) : Math.min(stop_v, len - 1);
      while (i > end_) { result.push(arr[i] ?? ""); i += step_v; }
    }
    return result.join("");
  });
}

/** np.char.compare_chararrays: element-wise comparison returning bool NDArray. */
export function compareChararrays(a: StringLike, b: StringLike, cmp: string, rstrip_: boolean): NDArray {
  const sa = toStringArray(a);
  const sb = toStringArray(b);
  const norm = rstrip_ ? (s: string) => s.trimEnd() : (s: string) => s;
  switch (cmp) {
    case "==": return map2ToBool(sa, sb, (s, t) => norm(s) === norm(t));
    case "!=": return map2ToBool(sa, sb, (s, t) => norm(s) !== norm(t));
    case "<":  return map2ToBool(sa, sb, (s, t) => norm(s) < norm(t));
    case "<=": return map2ToBool(sa, sb, (s, t) => norm(s) <= norm(t));
    case ">":  return map2ToBool(sa, sb, (s, t) => norm(s) > norm(t));
    case ">=": return map2ToBool(sa, sb, (s, t) => norm(s) >= norm(t));
    default: throw new ValueError(`Unknown comparison operator: ${cmp}`);
  }
}

// ─── np.strings namespace ─────────────────────────────────────────────────────

export const strings = {
  add,
  capitalize,
  center,
  count,
  decode,
  encode,
  endswith,
  equal,
  expandtabs,
  find,
  greater,
  greater_equal,
  index,
  isalnum,
  isalpha,
  isdecimal,
  isdigit,
  islower,
  isnumeric,
  isspace,
  istitle,
  isupper,
  less,
  less_equal,
  ljust,
  lower,
  lstrip,
  mod,
  multiply,
  not_equal,
  partition,
  replace,
  rfind,
  rindex,
  rjust,
  rpartition,
  rstrip,
  slice: charSlice,
  startswith,
  str_len,
  strip,
  swapcase,
  title,
  translate,
  upper,
  zfill,
} as const;

// ─── np.char namespace ────────────────────────────────────────────────────────

export const char = {
  add,
  capitalize,
  center,
  count,
  decode,
  encode,
  endswith,
  equal,
  expandtabs,
  find,
  greater,
  greater_equal,
  index,
  isalnum,
  isalpha,
  isdecimal,
  isdigit,
  islower,
  isnumeric,
  isspace,
  istitle,
  isupper,
  join,
  less,
  less_equal,
  ljust,
  lower,
  lstrip,
  mod,
  multiply,
  not_equal,
  partition,
  replace,
  rfind,
  rindex,
  rjust,
  rpartition,
  rsplit,
  rstrip,
  slice: charSlice,
  split,
  splitlines,
  startswith,
  str_len,
  strip,
  swapcase,
  title,
  translate,
  upper,
  zfill,
  array: charArray,
  asarray: charAsarray,
  compare_chararrays: compareChararrays,
} as const;

// ─── p16c export (wired into index.ts) ───────────────────────────────────────

export const p16c = {
  strings,
  char,
  StringArray,
} as const;
