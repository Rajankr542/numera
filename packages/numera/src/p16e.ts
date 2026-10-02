// Parity milestone P16E: np.rec record arrays and shares_memory (D-230).
// Pure TypeScript — no new C++ DType.
// The `rec` module is exported as `np.rec`; `recarray` and `sharesMemory` are
// added to the `np` root object so that coverage counts them.

import { array as npArray, zeros } from "./creation.js";
import { NotImplementedError, ValueError } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { toArray, type ArrayLike } from "./ufunc.js";

// ── format_parser ────────────────────────────────────────────────────────────

/** Parsed field descriptor produced by `format_parser`. */
export interface FieldDesc {
  name: string;
  dtype: string;
  /** Repeat count (>1 means the field is an array column of that length). */
  count: number;
}

/**
 * Simple NumPy format-string parser (np.rec.format_parser, D-230).
 * Parses a list of format strings (e.g. `"f4"`, `"3i4"`, `"<u2"`) or a
 * comma-separated dtype descriptor into an array of FieldDesc objects.
 */
export class format_parser {
  readonly dtype: readonly FieldDesc[];

  constructor(
    formats: string | string[],
    names?: string[] | string | null,
    _titles?: unknown,
    _aligned?: boolean,
    _byteorder?: string | null,
  ) {
    const fmtList = parseFormats(formats);
    const nameList = parseNames(names, fmtList.length);
    this.dtype = fmtList.map((f, i) => ({ name: nameList[i] as string, dtype: f.dtype, count: f.count }));
  }
}

// ── dtype string → string ────────────────────────────────────────────────────

interface ParsedFmt { dtype: string; count: number }

/** Maps NumPy format-char letters to numera dtype names. */
const FMT_CHAR: Record<string, string> = {
  "?": "bool", b: "int8", B: "uint8",
  h: "int16", H: "uint16",
  i: "int32", I: "uint32",
  l: "int64", L: "uint64",
  q: "int64", Q: "uint64",
  e: "float16", f: "float32", d: "float64",
  F: "complex64", D: "complex128",
  g: "float64", G: "complex128",
};

/** Parse a single format string like `"3f4"`, `"<u2"`, `"float64"`. */
function parseSingleFmt(s: string): ParsedFmt {
  s = s.trim();
  if ("<>|=!".includes(s[0] ?? "")) s = s.slice(1);

  const namedDtype = tryNamedDtype(s);
  if (namedDtype !== null) return { dtype: namedDtype, count: 1 };

  const m = s.match(/^(\d*)([?bBhHiIlLqQefdgFDG])(\d*)$/);
  if (!m) throw new ValueError(`unrecognised format string '${s}'`);

  const count = m[1] ? parseInt(m[1], 10) : 1;
  const char = m[2] as string;
  const size = m[3] ? parseInt(m[3], 10) : 0;

  let dtypeName: string = FMT_CHAR[char] ?? "float64";

  if (size > 0) {
    const sizedMap: Record<string, string> = {
      i1: "int8", i2: "int16", i4: "int32", i8: "int64",
      u1: "uint8", u2: "uint16", u4: "uint32", u8: "uint64",
      f2: "float16", f4: "float32", f8: "float64",
      c8: "complex64", c16: "complex128",
    };
    const key = `${char}${size}`;
    dtypeName = sizedMap[key] ?? dtypeName;
  }

  return { dtype: dtypeName, count };
}

const NAMED_DTYPE_MAP: Record<string, string> = {
  bool: "bool", int8: "int8", uint8: "uint8", int16: "int16", uint16: "uint16",
  int32: "int32", uint32: "uint32", int64: "int64", uint64: "uint64",
  float16: "float16", float32: "float32", float64: "float64",
  complex64: "complex64", complex128: "complex128",
};

function tryNamedDtype(s: string): string | null {
  return NAMED_DTYPE_MAP[s.toLowerCase()] ?? null;
}

function parseFormats(formats: string | string[]): ParsedFmt[] {
  if (typeof formats === "string") {
    if (formats.includes(",")) return formats.split(",").map(parseSingleFmt);
    return [parseSingleFmt(formats)];
  }
  return formats.map(parseSingleFmt);
}

function parseNames(names: string[] | string | null | undefined, count: number): Array<string | undefined> {
  if (!names) return Array.from({ length: count }, (_, i) => `f${i}`);
  if (typeof names === "string") return names.split(",").map((n) => n.trim());
  return names;
}

// ── RecArrayOptions ──────────────────────────────────────────────────────────

export interface RecArrayOptions {
  /** Field names. Required unless embedded in `formats` as `"name:fmt"`. */
  names?: string[] | string | null;
  /** Format string per field, or a single comma-separated dtype descriptor. */
  formats?: string | string[] | null;
  titles?: unknown;
  byteorder?: string | null;
  aligned?: boolean;
}

// ── recarray ─────────────────────────────────────────────────────────────────

/**
 * Lightweight structured record array (np.rec.recarray, D-230).
 *
 * Wraps a `Record<string, NDArray>` of named equal-length columns.
 * Fields are accessible as attributes: `rec.x`, `rec.y`, etc.
 */
export class recarray {
  /** Ordered field names. */
  readonly names: readonly string[];
  /** Shape of each record column (first dimension is the number of records). */
  readonly shape: readonly number[];

  private readonly _fields: Readonly<Record<string, NDArray>>;

  constructor(shape: number | number[], opts: RecArrayOptions = {}) {
    const shapeArr: number[] = typeof shape === "number" ? [shape] : [...shape];
    const formats = opts.formats ?? "float64";
    const fp = new format_parser(formats, opts.names ?? null);
    this.names = fp.dtype.map((f) => f.name);
    this.shape = shapeArr;
    const fields: Record<string, NDArray> = {};
    for (const fd of fp.dtype) {
      const colShape = fd.count > 1 ? [...shapeArr, fd.count] : shapeArr;
      const sz = colShape.length === 0 ? [1] : colShape;
      fields[fd.name] = zeros(sz, { dtype: fd.dtype });
    }
    this._fields = fields;
    this._installAccessors();
  }

  /** @internal Builds a recarray directly from pre-built column arrays. */
  static _fromColumns(names: string[], columns: NDArray[]): recarray {
    const first = columns[0];
    const shp: number[] = (columns.length > 0 && first !== undefined) ? [...first.shape] : [];
    const inst = Object.create(recarray.prototype) as recarray;
    // Use Object.defineProperty to write readonly props in the factory
    Object.defineProperty(inst, "names", { value: names, writable: false, enumerable: true, configurable: true });
    Object.defineProperty(inst, "shape", { value: shp, writable: false, enumerable: true, configurable: true });
    const fields: Record<string, NDArray> = {};
    for (let i = 0; i < names.length; i++) fields[names[i] as string] = columns[i] as NDArray;
    Object.defineProperty(inst, "_fields", { value: fields, writable: false, enumerable: false, configurable: true });
    inst._installAccessors();
    return inst;
  }

  /** Returns the NDArray column for the named field. */
  field(name: string): NDArray {
    const col = this._fields[name];
    if (col === undefined) throw new ValueError(`recarray has no field '${name}'`);
    return col;
  }

  /** Returns all columns as a plain record object. */
  toRecord(): Readonly<Record<string, NDArray>> {
    return this._fields;
  }

  _installAccessors(): void {
    for (const name of this.names) {
      if (!(name in this)) {
        Object.defineProperty(this, name, {
          get: () => this._fields[name],
          enumerable: true,
          configurable: true,
        });
      }
    }
  }

  [key: string]: unknown;
}

// ── fromarrays ───────────────────────────────────────────────────────────────

/**
 * Build a recarray from a list of column arrays (np.rec.fromarrays, D-230).
 */
export function fromarrays(
  arrayList: ArrayLike[],
  opts: RecArrayOptions = {},
): recarray {
  if (arrayList.length === 0) throw new ValueError("fromarrays: arrayList must be non-empty");
  const columns = arrayList.map((a) => toArray(a));

  let names: string[];
  if (opts.names) {
    names = typeof opts.names === "string"
      ? opts.names.split(",").map((n) => n.trim())
      : [...opts.names];
  } else if (opts.formats) {
    names = extractNamesFromFormats(opts.formats, columns.length);
  } else {
    names = columns.map((_, i) => `f${i}`);
  }

  if (names.length !== columns.length) {
    throw new ValueError(`fromarrays: got ${columns.length} arrays but ${names.length} names`);
  }
  return recarray._fromColumns(names, columns);
}

function extractNamesFromFormats(formats: string | string[], count: number): string[] {
  const fmtList = typeof formats === "string" ? formats.split(",") : formats;
  const names = fmtList.map((f) => {
    const colon = f.indexOf(":");
    return colon >= 0 ? f.slice(0, colon).trim() : null;
  });
  if (names.every((n) => n !== null)) return names as string[];
  return Array.from({ length: count }, (_, i) => `f${i}`);
}

// ── fromrecords ──────────────────────────────────────────────────────────────

/**
 * Build a recarray from a list of row tuples (np.rec.fromrecords, D-230).
 */
export function fromrecords(
  recList: unknown[][] | ArrayLike,
  opts: RecArrayOptions = {},
): recarray {
  let rows: unknown[][];
  if (recList instanceof NDArray) {
    rows = recList.toArray() as unknown[][];
  } else {
    rows = recList as unknown[][];
  }
  if (rows.length === 0) throw new ValueError("fromrecords: recList must be non-empty");
  const firstRow = rows[0] as unknown[];
  const nFields = firstRow.length;

  let names: string[];
  if (opts.names) {
    names = typeof opts.names === "string"
      ? opts.names.split(",").map((n) => n.trim())
      : [...opts.names];
  } else {
    names = Array.from({ length: nFields }, (_, i) => `f${i}`);
  }

  const columns: unknown[][] = Array.from({ length: nFields }, () => []);
  for (const row of rows) {
    const r = row as unknown[];
    for (let j = 0; j < nFields; j++) columns[j]!.push(r[j]);
  }

  const fmts = opts.formats ? parseFormats(opts.formats) : null;
  const arrays: NDArray[] = columns.map((col, i) => {
    const fmtDtype = fmts ? fmts[i]?.dtype : undefined;
    return fmtDtype !== undefined
      ? npArray(col as number[], { dtype: fmtDtype })
      : npArray(col as number[]);
  });

  return recarray._fromColumns(names, arrays);
}

// ── array ────────────────────────────────────────────────────────────────────

/**
 * Flexible recarray constructor (np.rec.array, D-230).
 */
export function array(
  obj: recarray | ArrayLike[] | unknown[][],
  opts: RecArrayOptions = {},
): recarray {
  if (obj instanceof recarray) {
    return fromarrays(Object.values(obj.toRecord()) as ArrayLike[], { names: [...obj.names], ...opts });
  }
  if (Array.isArray(obj) && obj.length > 0 && obj[0] instanceof NDArray) {
    return fromarrays(obj as NDArray[], opts);
  }
  return fromrecords(obj as unknown[][], opts);
}

// ── find_duplicate ───────────────────────────────────────────────────────────

/**
 * Return values that appear more than once in `list` (np.rec.find_duplicate, D-230).
 */
export function find_duplicate(list: unknown[]): unknown[] {
  const seen = new Set<unknown>();
  const dupes = new Set<unknown>();
  for (const v of list) {
    if (seen.has(v)) dupes.add(v);
    else seen.add(v);
  }
  return [...dupes];
}

// ── fromfile / fromstring ────────────────────────────────────────────────────

/**
 * Not implemented: deferred to a future milestone (D-230).
 * Only accessible via `np.rec.fromfile`.
 */
function fromfile(_file: unknown, _opts?: unknown): never {
  throw new NotImplementedError("np.rec.fromfile is not implemented in this release");
}

/**
 * Not implemented: deferred to a future milestone (D-230).
 * Only accessible via `np.rec.fromstring`.
 */
function fromstring(_string: unknown, _opts?: unknown): never {
  throw new NotImplementedError("np.rec.fromstring is not implemented in this release");
}

// ── rec module object ─────────────────────────────────────────────────────────

/** `np.rec` module — lightweight record arrays (D-230). */
export const rec = {
  recarray,
  record: recarray,
  format_parser,
  fromarrays,
  fromrecords,
  array,
  find_duplicate,
  fromfile,
  fromstring,
} as const;

// ── shares_memory ─────────────────────────────────────────────────────────────

/** Options for `shares_memory`. */
export interface SharesMemoryOptions {
  /** Maximum work items (NumPy default 1024). numera is always exact. */
  maxWork?: number;
}

/**
 * Return `true` if two arrays share the same memory buffer (np.shares_memory, D-230).
 * Delegates to the native `sharesMemory` check, which is always exact.
 */
export function sharesMemory(a: NDArray, b: NDArray, _opts?: SharesMemoryOptions): boolean {
  return a._native.sharesMemory(b._native);
}

// ── p16e public exports ───────────────────────────────────────────────────────

export const p16e = {
  rec,
  recarray,
  sharesMemory,
} as const;
