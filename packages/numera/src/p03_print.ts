// P3-5: array printing (D-063). Digits come from the native Dragon4 port
// (native/core/p03_print.cpp); this file does option handling and line layout
// (NumPy arrayprint `_formatArray`, `array_repr`, `array_str`).
import { asarray } from "./creation.js";
import { DTypeError, NotImplementedError, ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray } from "./ndarray.js";
import { p03native } from "./p03_native.js";

export type FloatMode = "fixed" | "unique" | "maxprec" | "maxprec_equal";
export type SignOption = "-" | "+" | " ";
export type TrimOption = "k" | "." | "0" | "-";
/** JS callbacks keyed by NumPy formatter names; each gets the element (`item()`). */
export type Formatter = Partial<
  Record<
    | "all"
    | "bool"
    | "int"
    | "float"
    | "complexfloat"
    | "int_kind"
    | "float_kind"
    | "complex_kind"
    | "intKind"
    | "floatKind"
    | "complexKind",
    (x: never) => string
  >
>;

/** NumPy `np.get_printoptions()` (D-063). */
export interface PrintOptions {
  precision: number;
  threshold: number;
  edgeitems: number;
  linewidth: number;
  suppress: boolean;
  nanstr: string;
  infstr: string;
  sign: SignOption;
  floatmode: FloatMode;
  formatter: Formatter | null;
  legacy: false;
  overrideRepr: ((a: NDArray) => string) | null;
}

export type PrintOptionsInput = {
  [K in keyof PrintOptions]?: PrintOptions[K] | null;
};

const DEFAULTS: PrintOptions = {
  precision: 8,
  threshold: 1000,
  edgeitems: 3,
  linewidth: 75,
  suppress: false,
  nanstr: "nan",
  infstr: "inf",
  sign: "-",
  floatmode: "maxprec",
  formatter: null,
  legacy: false,
  overrideRepr: null,
};

let current: PrintOptions = { ...DEFAULTS };

const FLOATMODES: readonly string[] = ["fixed", "unique", "maxprec", "maxprec_equal"];

function nonNegInt(v: unknown, name: string): number {
  if (typeof v !== "number" || !Number.isInteger(v)) throw new DTypeError(`${name} must be an integer`);
  if (v < 0) throw new ValueError(`${name} must be >= 0`);
  return v;
}

// NumPy _make_options_dict: the given (non-null) options, validated.
function makeOptions(opts: PrintOptionsInput): Partial<PrintOptions> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(opts)) if (v !== undefined && v !== null) out[k] = v;
  if (out.floatmode !== undefined && !FLOATMODES.includes(out.floatmode as string)) {
    throw new ValueError(
      `floatmode option must be one of ${FLOATMODES.map((m) => `"${m}"`).join(", ")}`,
    );
  }
  if (out.sign !== undefined && !["-", "+", " "].includes(out.sign as string)) {
    throw new ValueError("sign option must be one of ' ', '+', or '-'");
  }
  if (out.legacy !== undefined && out.legacy !== false) {
    throw new NotImplementedError("legacy printing modes are not supported; only legacy=false");
  }
  if (out.threshold !== undefined) {
    if (typeof out.threshold !== "number") throw new DTypeError("threshold must be numeric");
    if (Number.isNaN(out.threshold)) {
      throw new ValueError("threshold must be non-NAN, try Infinity for untruncated representation");
    }
  }
  if (out.precision !== undefined) out.precision = nonNegInt(out.precision, "precision");
  if (out.edgeitems !== undefined) out.edgeitems = nonNegInt(out.edgeitems, "edgeitems");
  if (out.linewidth !== undefined && typeof out.linewidth !== "number") {
    throw new DTypeError("linewidth must be numeric");
  }
  if (out.suppress !== undefined) out.suppress = Boolean(out.suppress);
  return out as Partial<PrintOptions>;
}

/**
 * NumPy `np.set_printoptions` (D-063). `undefined`/`null` options are left
 * unchanged, except `formatter` and `overrideRepr`, which reset on every call.
 */
export function setPrintoptions(opts: PrintOptionsInput = {}): void {
  const next = makeOptions(opts);
  current = {
    ...current,
    ...next,
    formatter: opts.formatter ?? null,
    overrideRepr: opts.overrideRepr ?? null,
  };
}

/** NumPy `np.get_printoptions()`: a copy of the current print options. */
export function getPrintoptions(): PrintOptions {
  return { ...current };
}

/**
 * NumPy `np.printoptions` context manager as a callback (D-063): the options
 * apply while `fn()` runs and are restored afterwards; returns `fn`'s result.
 */
export function printoptions<T>(opts: PrintOptionsInput, fn: () => T): T {
  const saved = current;
  setPrintoptions(opts);
  try {
    return fn();
  } finally {
    current = saved;
  }
}

// ---------------------------------------------------------------------------
// format_float_positional / format_float_scientific

export interface FormatFloatPositionalOptions {
  precision?: number | null;
  unique?: boolean;
  fractional?: boolean;
  trim?: TrimOption;
  sign?: boolean;
  padLeft?: number | null;
  padRight?: number | null;
  minDigits?: number | null;
}

export interface FormatFloatScientificOptions {
  precision?: number | null;
  unique?: boolean;
  trim?: TrimOption;
  sign?: boolean;
  padLeft?: number | null;
  expDigits?: number | null;
  minDigits?: number | null;
}

function noneOrPositive(x: number | null | undefined, name: string): number {
  if (x === undefined || x === null) return -1;
  if (typeof x !== "number" || !Number.isInteger(x)) throw new DTypeError(`${name} must be an integer`);
  if (x < 0) throw new ValueError(`${name} must be >= 0`);
  return x;
}

function floatScalar(x: number | boolean | NDArray): NDArray {
  if (typeof x === "number" || typeof x === "boolean") return asarray(Number(x), { dtype: "float64" });
  if (!(x instanceof NDArray)) throw new DTypeError(`must be real number, not ${typeof x}`);
  if (x.size !== 1) throw new ValueError("only size-1 arrays can be formatted as a float scalar");
  if (x.dtype.kind === "c") throw new DTypeError("must be real number, not complex");
  return x.dtype.kind === "f" ? x : x.astype("float64");
}

function checkTrim(trim: string, unique: boolean, precision: number): void {
  if (!["k", ".", "0", "-"].includes(trim)) {
    throw new DTypeError(`if supplied, trim must be 'k', '.', '0' or '-' found \`${trim}\``);
  }
  if (!unique && precision < 0) throw new DTypeError("in non-unique mode `precision` must be supplied");
}

/** NumPy `np.format_float_positional` via Dragon4 (D-063). */
export function formatFloatPositional(x: number | boolean | NDArray, opts: FormatFloatPositionalOptions = {}): string {
  const precision = noneOrPositive(opts.precision, "precision");
  const padLeft = noneOrPositive(opts.padLeft, "pad_left");
  const padRight = noneOrPositive(opts.padRight, "pad_right");
  const minDigits = noneOrPositive(opts.minDigits, "min_digits");
  const fractional = opts.fractional ?? true;
  const unique = opts.unique ?? true;
  if (!fractional && precision === 0) throw new ValueError("precision must be greater than 0 if fractional=False");
  if (minDigits > 0 && precision > 0 && minDigits > precision) {
    throw new ValueError("min_digits must be less than or equal to precision");
  }
  const trim = opts.trim ?? "k";
  checkTrim(trim, unique, precision);
  const a = floatScalar(x);
  return wrapNative(() =>
    p03native.formatFloat(a._native, {
      scientific: false,
      unique,
      fractional,
      sign: opts.sign ?? false,
      precision,
      minDigits,
      padLeft,
      padRight,
      trim,
    }),
  );
}

/** NumPy `np.format_float_scientific` via Dragon4 (D-063). */
export function formatFloatScientific(x: number | boolean | NDArray, opts: FormatFloatScientificOptions = {}): string {
  const precision = noneOrPositive(opts.precision, "precision");
  const padLeft = noneOrPositive(opts.padLeft, "pad_left");
  const expDigits = noneOrPositive(opts.expDigits, "exp_digits");
  const minDigits = noneOrPositive(opts.minDigits, "min_digits");
  const unique = opts.unique ?? true;
  if (minDigits > 0 && precision > 0 && minDigits > precision) {
    throw new ValueError("min_digits must be less than or equal to precision");
  }
  const trim = opts.trim ?? "k";
  checkTrim(trim, unique, precision);
  const a = floatScalar(x);
  return wrapNative(() =>
    p03native.formatFloat(a._native, {
      scientific: true,
      unique,
      fractional: true,
      sign: opts.sign ?? false,
      precision,
      minDigits,
      padLeft,
      expDigits,
      trim,
    }),
  );
}

// ---------------------------------------------------------------------------
// array2string / arrayRepr / arrayStr

export interface Array2StringOptions {
  maxLineWidth?: number | null;
  precision?: number | null;
  suppressSmall?: boolean | null;
  separator?: string;
  prefix?: string;
  suffix?: string;
  formatter?: Formatter | null;
  threshold?: number | null;
  edgeitems?: number | null;
  sign?: SignOption | null;
  floatmode?: FloatMode | null;
  legacy?: false | null;
}

export interface ArrayReprOptions {
  maxLineWidth?: number | null;
  precision?: number | null;
  suppressSmall?: boolean | null;
}

type AnyArray = NDArray | NestedArray | number | boolean;

function toArray(a: AnyArray): NDArray {
  return a instanceof NDArray ? a : asarray(a as NestedArray);
}

function formatterFor(f: Formatter | null, kind: string): ((x: never) => string) | undefined {
  if (f === null) return undefined;
  const pick = (...keys: (keyof Formatter)[]) => {
    for (const k of keys) if (f[k] !== undefined && f[k] !== null) return f[k];
    return undefined;
  };
  switch (kind) {
    case "b":
      return pick("bool", "all");
    case "i":
    case "u":
      return pick("int", "int_kind", "intKind", "all");
    case "f":
      return pick("float", "float_kind", "floatKind", "all");
    default:
      return pick("complexfloat", "complex_kind", "complexKind", "all");
  }
}

function elementStrings(data: NDArray, o: PrintOptions): string[] {
  const fn = formatterFor(o.formatter, data.dtype.kind);
  if (fn !== undefined) {
    const flat = data.reshape(-1);
    const out: string[] = [];
    for (let k = 0; k < flat.size; ++k) {
      const s = (fn as (x: unknown) => unknown)(flat.item(k));
      if (typeof s !== "string") throw new DTypeError("formatter callbacks must return a string");
      out.push(s);
    }
    return out;
  }
  return wrapNative(() =>
    p03native.formatElements(data._native, {
      precision: o.precision,
      floatmode: o.floatmode,
      suppress: o.suppress,
      sign: o.sign,
      nanstr: o.nanstr,
      infstr: o.infstr,
    }),
  );
}

function extendLine(s: string, line: string, word: string, width: number, nextPrefix: string): [string, string] {
  let wrap = line.length + word.length > width;
  if (line.length <= nextPrefix.length) wrap = false;
  if (wrap) {
    s += line.trimEnd() + "\n";
    line = nextPrefix;
  }
  return [s, line + word];
}

function extendLinePretty(
  s: string,
  line: string,
  word: string,
  width: number,
  nextPrefix: string,
): [string, string] {
  const words = word.split("\n");
  if (words.length === 1) return extendLine(s, line, word, width, nextPrefix);
  const maxLen = Math.max(...words.map((w) => w.length));
  let indent: string;
  if (line.length + maxLen > width && line.length > nextPrefix.length) {
    s += line.trimEnd() + "\n";
    line = nextPrefix + words[0];
    indent = nextPrefix;
  } else {
    indent = " ".repeat(line.length);
    line += words[0];
  }
  for (const w of words.slice(1)) {
    s += line.trimEnd() + "\n";
    line = indent + w;
  }
  line += " ".repeat(maxLen - words[words.length - 1]!.length);
  return [s, line];
}

// NumPy _formatArray over the (possibly summarized) element strings `strs`
// of shape `dshape`; `fullShape` is the shape of the original array.
function formatArray(
  strs: string[],
  dshape: number[],
  fullShape: number[],
  lineWidth: number,
  nextLinePrefix: string,
  separator: string,
  edgeItems: number,
  summaryInsert: string,
): string {
  const nd = dshape.length;
  const strides = new Array<number>(nd);
  let acc = 1;
  for (let d = nd - 1; d >= 0; --d) {
    strides[d] = acc;
    acc *= dshape[d]!;
  }
  const recurse = (axis: number, offset: number, hanging: string, width: number): string => {
    if (axis === nd) return strs[offset]!;
    const nextHanging = hanging + " ";
    const nextWidth = width - 1;
    const dlen = dshape[axis]!;
    const showSummary = summaryInsert !== "" && 2 * edgeItems < fullShape[axis]!;
    const leading = showSummary ? edgeItems : 0;
    const trailing = showSummary ? edgeItems : dlen;
    const at = (i: number) => offset + i * strides[axis]!;
    let s = "";
    if (nd - axis === 1) {
      const elemWidth = width - Math.max(separator.trimEnd().length, 1);
      let line = hanging;
      for (let i = 0; i < leading; ++i) {
        [s, line] = extendLinePretty(s, line, recurse(axis + 1, at(i), nextHanging, nextWidth), elemWidth, hanging);
        line += separator;
      }
      if (showSummary) {
        [s, line] = extendLine(s, line, summaryInsert, elemWidth, hanging);
        line += separator;
      }
      for (let i = trailing; i > 1; --i) {
        [s, line] = extendLinePretty(
          s,
          line,
          recurse(axis + 1, at(dlen - i), nextHanging, nextWidth),
          elemWidth,
          hanging,
        );
        line += separator;
      }
      [s, line] = extendLinePretty(s, line, recurse(axis + 1, at(dlen - 1), nextHanging, nextWidth), elemWidth, hanging);
      s += line;
    } else {
      const lineSep = separator.trimEnd() + "\n".repeat(nd - axis - 1);
      for (let i = 0; i < leading; ++i) s += hanging + recurse(axis + 1, at(i), nextHanging, nextWidth) + lineSep;
      if (showSummary) s += hanging + summaryInsert + lineSep;
      for (let i = trailing; i > 1; --i) {
        s += hanging + recurse(axis + 1, at(dlen - i), nextHanging, nextWidth) + lineSep;
      }
      s += hanging + recurse(axis + 1, at(dlen - 1), nextHanging, nextWidth);
    }
    return "[" + s.slice(hanging.length) + "]";
  };
  return recurse(0, 0, nextLinePrefix, lineWidth);
}

/** NumPy `np.array2string` (D-063). */
export function array2string(a: AnyArray, opts: Array2StringOptions = {}): string {
  const arr = toArray(a);
  const overrides = makeOptions({
    precision: opts.precision,
    threshold: opts.threshold,
    edgeitems: opts.edgeitems,
    linewidth: opts.maxLineWidth,
    suppress: opts.suppressSmall,
    sign: opts.sign,
    formatter: opts.formatter,
    floatmode: opts.floatmode,
    legacy: opts.legacy,
  });
  const o: PrintOptions = { ...current, ...overrides };
  const separator = opts.separator ?? " ";
  const prefix = opts.prefix ?? "";
  o.linewidth -= (opts.suffix ?? "").length;
  if (arr.size === 0) return "[]";
  let data = arr;
  let summaryInsert = "";
  if (arr.size > o.threshold) {
    summaryInsert = "...";
    data = wrapNative(() => NDArray._wrap(p03native.leadingTrailing(arr._native, o.edgeitems)));
  }
  const strs = elementStrings(data, o);
  return formatArray(strs, data.shape, arr.shape, o.linewidth, " " + " ".repeat(prefix.length), separator, o.edgeitems, summaryInsert);
}

const IMPLIED = new Set(["float64", "int64", "complex128", "bool"]);

function shapeRepr(shape: number[]): string {
  return shape.length === 1 ? `(${shape[0]},)` : `(${shape.join(", ")})`;
}

/** NumPy `np.array_repr` (D-063); also `NDArray.toString()`. */
export function arrayRepr(a: AnyArray, opts: ArrayReprOptions = {}): string {
  const arr = toArray(a);
  if (current.overrideRepr !== null) return current.overrideRepr(arr);
  const maxLineWidth = opts.maxLineWidth ?? current.linewidth;
  const prefix = "array(";
  const lst = array2string(arr, {
    maxLineWidth,
    precision: opts.precision,
    suppressSmall: opts.suppressSmall,
    separator: ", ",
    prefix,
    suffix: ")",
  });
  const extras: string[] = [];
  const shape = arr.shape;
  if ((arr.size === 0 && !(shape.length === 1 && shape[0] === 0)) || arr.size > current.threshold) {
    extras.push(`shape=${shapeRepr(shape)}`);
  }
  if (!IMPLIED.has(arr.dtype.name) || arr.size === 0) extras.push(`dtype=${arr.dtype.name}`);
  if (extras.length === 0) return prefix + lst + ")";
  const arrStr = prefix + lst + ",";
  const extraStr = extras.join(", ") + ")";
  const lastLineLen = arrStr.length - (arrStr.lastIndexOf("\n") + 1);
  const spacer = lastLineLen + extraStr.length + 1 > maxLineWidth ? "\n" + " ".repeat(prefix.length) : " ";
  return arrStr + spacer + extraStr;
}

/** NumPy `np.array_str` (D-063); a 0-d array prints like a NumPy scalar's `str`. */
export function arrayStr(a: AnyArray, opts: ArrayReprOptions = {}): string {
  const arr = toArray(a);
  if (arr.ndim === 0) return wrapNative(() => p03native.scalarStr(arr._native));
  return array2string(arr, {
    maxLineWidth: opts.maxLineWidth,
    precision: opts.precision,
    suppressSmall: opts.suppressSmall,
    separator: " ",
    prefix: "",
  });
}

NDArray.prototype.toString = function (this: NDArray): string {
  return arrayRepr(this);
};
