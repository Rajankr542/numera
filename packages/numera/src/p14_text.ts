// P14 text I/O (D-171): loadtxt, savetxt. Parsing and %-formatting are
// native (`addon.p14`, native/core/p14_text.cpp); this file resolves sources,
// validates options and applies NumPy's ndmin/unpack/header rules.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { gunzipSync, gzipSync } from "node:zlib";
import { nativeModule, type NativeNDArray } from "./addon.js";
import { asarray } from "./creation.js";
import { dtype as toDType, type DTypeLike } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray } from "./ndarray.js";

type ArrayLike = NDArray | NestedArray;

/** Native text kernels of `addon.p14`. Internal. */
export interface P14TextNative {
  loadtxt(
    text: string,
    opts: {
      delimiter: string | null;
      comments: string[];
      quote: string | null;
      skiprows: number;
      usecols: number[] | null;
      maxRows: number;
      dtype: string;
    },
  ): NativeNDArray;
  formatRows(a: NativeNDArray, fmt: string, newline: string): string;
  tofileText(a: NativeNDArray, sep: string, format: string): string;
}
const native = nativeModule<P14TextNative>("p14");

/** A text source: a path (string or file URL), the contents as bytes, or a list of lines. */
export type TextSource = string | URL | Uint8Array | readonly string[];

const toPath = (f: string | URL): string => (typeof f === "string" ? f : fileURLToPath(f));

/** @internal Reads a text source into one string (lines joined by "\n"). */
export function readText(src: TextSource): string {
  if (Array.isArray(src)) {
    return (src as readonly string[]).map((l) => (l.endsWith("\n") ? l : l + "\n")).join("");
  }
  if (typeof src === "string" || src instanceof URL) {
    const p = toPath(src);
    const raw = readFileSync(p);
    return (p.endsWith(".gz") ? gunzipSync(raw) : raw).toString("utf8");
  }
  if (ArrayBuffer.isView(src)) {
    return Buffer.from(src.buffer, src.byteOffset, src.byteLength).toString("utf8");
  }
  throw new TypeError("expected a path, a URL, a Buffer/Uint8Array or an array of lines");
}

const controlChar = (v: string): string => {
  if ([...v].length !== 1) {
    throw new TypeError(
      `Text reading control character must be a single unicode character or None; but got: '${v}'`,
    );
  }
  return v;
};

export interface LoadtxtOptions {
  /** Default float64. Numeric dtypes only. */
  dtype?: DTypeLike;
  /** Comment prefix(es); `null` disables comments. Default "#". */
  comments?: string | readonly string[] | null;
  /** Single-character delimiter; `null` (default) splits on whitespace. */
  delimiter?: string | null;
  /** Number of leading raw lines to skip (comments included). */
  skiprows?: number;
  /** Column index or indices to read (negative counts from the end). */
  usecols?: number | readonly number[] | null;
  /** Transpose the result (`x, y = loadtxt(..., unpack=True)`). */
  unpack?: boolean;
  /** Minimum number of dimensions: 0, 1 or 2. */
  ndmin?: number;
  /** Read at most this many data rows. */
  maxRows?: number | null;
  /** Quote character; delimiters inside quotes are kept, a doubled quote is a literal one. */
  quotechar?: string | null;
}

/** Applies NumPy's squeeze/ndmin/unpack rules to a 2-D table. @internal */
export function finishTable(arr: NDArray, ndmin: number, unpack: boolean): NDArray {
  if (ndmin !== 0 && ndmin !== 1 && ndmin !== 2) throw new ValueError(`Illegal value of ndmin keyword: ${ndmin}`);
  let a = arr;
  if (a.ndim > ndmin) a = a.squeeze();
  if (a.ndim < ndmin) {
    if (ndmin === 1) a = a.reshape([a.size]);
    else a = (a.ndim === 0 ? a.reshape([1, 1]) : a.reshape([1, a.size])).T;
  }
  return unpack ? a.T : a;
}

/**
 * np.loadtxt(fname, options): reads a numeric table. Blank lines and
 * comments are skipped; every row must have the same number of columns.
 */
export function loadtxt(fname: TextSource, options: LoadtxtOptions = {}): NDArray {
  const dt = toDType(options.dtype ?? "float64");
  const delimiter = options.delimiter == null ? null : controlChar(options.delimiter);
  const quote = options.quotechar == null ? null : controlChar(options.quotechar);
  const c = options.comments === undefined ? ["#"] : options.comments === null ? [] : options.comments;
  const comments = typeof c === "string" ? [c] : [...c];
  if (comments.some((s) => s === "")) {
    throw new ValueError("comments cannot be an empty string. Use comments=None to disable comments.");
  }
  if (delimiter !== null && comments.includes(delimiter)) {
    throw new TypeError("The values for control characters 'comment' and 'delimiter' are incompatible");
  }
  if (quote !== null && (quote === delimiter || comments.includes(quote))) {
    throw new TypeError("The values for control characters 'quotechar' and 'delimiter'/'comment' are incompatible");
  }
  const u = options.usecols;
  const usecols = u == null ? null : typeof u === "number" ? [u] : [...u];
  if (usecols !== null && usecols.some((k) => !Number.isInteger(k))) {
    throw new TypeError("usecols must be an int or a sequence of ints");
  }
  const skiprows = options.skiprows ?? 0;
  const maxRows = options.maxRows ?? -1;
  if (options.maxRows != null && maxRows < 0) throw new ValueError("argument must be nonnegative");
  const text = readText(fname);
  const raw = wrapNative(() =>
    NDArray._wrap(
      native.loadtxt(text, { delimiter, comments, quote, skiprows, usecols, maxRows, dtype: dt.name }),
    ),
  );
  return finishTable(raw, options.ndmin ?? 0, options.unpack ?? false);
}

export interface SavetxtOptions {
  /** One format for all columns, a format per column, or a full row format. Default "%.18e". */
  fmt?: string | readonly string[];
  /** Column separator (default " "). */
  delimiter?: string;
  /** Line terminator (default "\n"). */
  newline?: string;
  /** Text written (commented) before the data. */
  header?: string;
  /** Text written (commented) after the data. */
  footer?: string;
  /** Prefix for header and footer lines (default "# "). */
  comments?: string;
}

const countPercent = (s: string): number => s.split("%").length - 1;

/**
 * np.savetxt(fname, X, options): writes a 1-D or 2-D array as text.
 * `fname = null` returns the text; a path ending in `.gz` is gzip-compressed.
 */
export function savetxt(fname: string | URL | null, X: ArrayLike, options: SavetxtOptions = {}): string | undefined {
  const fmt = options.fmt ?? "%.18e";
  const delimiter = options.delimiter ?? " ";
  const newline = options.newline ?? "\n";
  const header = options.header ?? "";
  const footer = options.footer ?? "";
  const comments = options.comments ?? "# ";
  let x = asarray(X);
  if (x.ndim === 0 || x.ndim > 2) throw new ValueError(`Expected 1D or 2D array, got ${x.ndim}D array instead`);
  if (x.ndim === 1) x = x.reshape([x.shape[0]!, 1]);
  const ncol = x.shape[1]!;
  const cplx = x.dtype.kind === "c";
  let rowFormat: string;
  if (typeof fmt !== "string") {
    if (fmt.length !== ncol) throw new ValueError(`fmt has wrong shape.  [${fmt.map((f) => `'${f}'`).join(", ")}]`);
    rowFormat = fmt.join(delimiter);
  } else {
    const n = countPercent(fmt);
    const err = new ValueError(`fmt has wrong number of % formats:  ${fmt}`);
    if (n === 1) {
      rowFormat = new Array<string>(ncol).fill(cplx ? ` (${fmt}+${fmt}j)` : fmt).join(delimiter);
    } else if (cplx ? n !== 2 * ncol : n !== ncol) {
      throw err;
    } else {
      rowFormat = fmt;
    }
  }
  let out = "";
  if (header.length > 0) out += comments + header.replaceAll("\n", "\n" + comments) + newline;
  out += wrapNative(() => native.formatRows(x._native, rowFormat, newline));
  if (footer.length > 0) out += comments + footer.replaceAll("\n", "\n" + comments) + newline;
  if (fname === null) return out;
  const p = toPath(fname);
  writeFileSync(p, p.endsWith(".gz") ? gzipSync(Buffer.from(out, "utf8")) : out);
  return undefined;
}

/** @internal Python-scalar text of the items of `a` joined by `sep` (tofile). */
export function tofileText(a: NDArray, sep: string, format: string): string {
  return wrapNative(() => native.tofileText(a._native, sep, format));
}
