/**
 * P16-D: datetime64, timedelta64, busday functions (D-220).
 *
 * Design: DatetimeArray / TimedeltaArray are TS subclasses of NDArray that
 * store int64 data + a unit string. No new C++ DType enum entries are needed.
 * Arithmetic and formatting are handled in TypeScript. NaT = int64 min sentinel.
 */

import { addon } from "./addon.js";
import { ValueError, DTypeError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";

// ---------------------------------------------------------------------------
// Unit helpers
// ---------------------------------------------------------------------------

/** Ordered datetime units (coarsest to finest). */
const UNIT_ORDER = ["Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns", "ps", "fs", "as"] as const;
export type DatetimeUnit = (typeof UNIT_ORDER)[number];

const UNIT_SET = new Set<string>(UNIT_ORDER);

function isValidUnit(u: string): u is DatetimeUnit {
  return UNIT_SET.has(u);
}

function unitIndex(u: string): number {
  const i = UNIT_ORDER.indexOf(u as DatetimeUnit);
  if (i === -1) throw new ValueError(`unknown datetime unit: '${u}'`);
  return i;
}

/** Returns true if unit b is finer (smaller) than or equal to unit a. */
function unitLE(a: DatetimeUnit, b: DatetimeUnit): boolean {
  return unitIndex(a) <= unitIndex(b);
}

/**
 * Conversion factor from `from` to `to` in multiples (exact BigInt when
 * possible). Returns null when the units are abstract (Y/M relative to D) and
 * conversion is not defined.
 */
function unitFactor(from: DatetimeUnit, to: DatetimeUnit): bigint | null {
  const factors: Record<string, bigint> = {
    "W->D": 7n,
    "D->h": 24n,
    "h->m": 60n,
    "m->s": 60n,
    "s->ms": 1000n,
    "ms->us": 1000n,
    "us->ns": 1000n,
    "ns->ps": 1000n,
    "ps->fs": 1000n,
    "fs->as": 1000n,
  };
  const fi = unitIndex(from);
  const ti = unitIndex(to);
  if (fi === ti) return 1n;
  if (fi < ti) {
    // Converting from coarser to finer (multiplication)
    // Y/M cannot convert to finer units
    if (from === "Y" || from === "M") return null;
    if (to === "Y" || to === "M") return null;
    let f = 1n;
    for (let i = fi; i < ti; i++) {
      const key = `${UNIT_ORDER[i]}->${UNIT_ORDER[i + 1]}`;
      const fac = factors[key];
      if (fac === undefined) return null;
      f *= fac;
    }
    return f;
  }
  // Converting finer to coarser (division) – not supported for storage
  return null;
}

/** Convert a value in `from` units to `to` units. Positive = multiply. */
function convertUnits(value: bigint, from: DatetimeUnit, to: DatetimeUnit): bigint {
  if (from === to) return value;
  if (isNaT(value)) return value;
  const fi = unitIndex(from);
  const ti = unitIndex(to);
  if (fi < ti) {
    // from is coarser, multiply
    const fac = unitFactor(from, to);
    if (fac === null) throw new ValueError(`Cannot convert datetime unit '${from}' to '${to}'`);
    return value * fac;
  } else {
    // from is finer, divide
    const fac = unitFactor(to, from);
    if (fac === null) throw new ValueError(`Cannot convert datetime unit '${from}' to '${to}'`);
    return value / fac;
  }
}

// ---------------------------------------------------------------------------
// NaT sentinel
// ---------------------------------------------------------------------------

const NAT_VALUE = -9223372036854775808n; // int64 min

function isNaT(v: bigint): boolean {
  return v === NAT_VALUE;
}

// ---------------------------------------------------------------------------
// Parsing ISO-8601 date/time strings to epoch offsets
// ---------------------------------------------------------------------------

/** Parse an ISO-8601 date or datetime string to (epoch_value, unit). */
function parseIsoString(s: string): { value: bigint; unit: DatetimeUnit } {
  if (s === "NaT" || s === "nat") return { value: NAT_VALUE, unit: "s" };

  // Full datetime: 2023-01-15T10:30:00.000000
  // Date only: 2023-01-15
  // Year only: 2023
  // Year-month: 2023-01
  const EPOCH = new Date("1970-01-01T00:00:00Z");

  // Determine unit from format
  const hasTime = s.includes("T") || s.includes(" ");
  const parts = s.replace(" ", "T");

  if (/^\d{4}$/.test(s)) {
    // Year-only
    const year = parseInt(s, 10);
    const epochYears = year - 1970;
    return { value: BigInt(epochYears), unit: "Y" };
  }
  if (/^\d{4}-\d{2}$/.test(s)) {
    // Year-month
    const parts2 = s.split("-").map(Number);
    const [y, m] = [parts2[0]!, parts2[1]!];
    const epochMonths = BigInt((y - 1970) * 12 + (m - 1));
    return { value: epochMonths, unit: "M" };
  }
  if (!hasTime) {
    // Date only: 2023-01-15
    const d = new Date(s + "T00:00:00Z");
    if (isNaN(d.getTime())) throw new ValueError(`Invalid datetime string: '${s}'`);
    const epochDays = BigInt(Math.round((d.getTime() - EPOCH.getTime()) / 86400000));
    return { value: epochDays, unit: "D" };
  }
  // Has time component
  const timePart = parts.split("T")[1] ?? "";
  let unit: DatetimeUnit = "s";
  if (timePart.includes(".")) {
    const dec = timePart.split(".")[1] ?? "";
    const sigDigits = dec.replace(/0+$/, "").length;
    if (sigDigits <= 3) unit = "ms";
    else if (sigDigits <= 6) unit = "us";
    else unit = "ns";
  }
  const d = new Date(parts + "Z");
  if (isNaN(d.getTime())) throw new ValueError(`Invalid datetime string: '${s}'`);
  const ms = BigInt(d.getTime());
  const epochValue = convertUnits(ms, "ms", unit);
  return { value: epochValue, unit };
}

// ---------------------------------------------------------------------------
// DatetimeDType — a pseudo-dtype object for datetime64
// ---------------------------------------------------------------------------

/** Pseudo-dtype for datetime64, mirrors NumPy's DType interface minimally. */
export class DatetimeDType {
  readonly kind = "M" as const;
  readonly unit: DatetimeUnit;
  readonly itemSize = 8;

  constructor(unit: DatetimeUnit) {
    this.unit = unit;
  }

  get name(): string {
    return `datetime64[${this.unit}]`;
  }

  toString(): string {
    return this.name;
  }
}

/** Pseudo-dtype for timedelta64. */
export class TimedeltaDType {
  readonly kind = "m" as const;
  readonly unit: DatetimeUnit;
  readonly itemSize = 8;

  constructor(unit: DatetimeUnit) {
    this.unit = unit;
  }

  get name(): string {
    return `timedelta64[${this.unit}]`;
  }

  toString(): string {
    return this.name;
  }
}

// ---------------------------------------------------------------------------
// DatetimeArray
// ---------------------------------------------------------------------------

/**
 * An array of datetime64 values, stored as int64 (epoch offsets) + unit.
 * This is a thin TS wrapper: the underlying NDArray has dtype int64.
 */
export class DatetimeArray {
  /** The underlying int64 NDArray. */
  readonly data: NDArray;
  /** The datetime unit (e.g. "D", "s", "ms"). */
  readonly unit: DatetimeUnit;
  /** Shape of the array. */
  readonly shape: readonly number[];
  /** Number of elements. */
  readonly size: number;
  /** Number of dimensions. */
  readonly ndim: number;

  constructor(data: NDArray, unit: DatetimeUnit) {
    if (data.dtype.name !== "int64") {
      throw new DTypeError("DatetimeArray requires int64 backing storage");
    }
    this.data = data;
    this.unit = unit;
    this.shape = data.shape;
    this.size = data.size;
    this.ndim = data.ndim;
  }

  get dtype(): DatetimeDType {
    return new DatetimeDType(this.unit);
  }

  /** Returns the int64 backing array as BigInt64Array. */
  toTypedArray(): BigInt64Array {
    return this.data.toTypedArray() as BigInt64Array;
  }

  /** Returns the values as BigInt array (nested for multi-dim). */
  toArray(): bigint | bigint[] {
    const raw = this.data.toTypedArray() as BigInt64Array;
    if (this.ndim === 0) return raw[0]!;
    return Array.from(raw) as bigint[];
  }

  toString(): string {
    const raw = this.data.toTypedArray() as BigInt64Array;
    if (this.ndim === 0) {
      const v = raw[0]!;
      return isNaT(v) ? "NaT" : formatDatetime(v, this.unit);
    }
    const strs = Array.from(raw).map((v) =>
      isNaT(v) ? "NaT" : formatDatetime(v, this.unit),
    );
    return `[${strs.join(", ")}]`;
  }

  /** Create a DatetimeArray from an array of epoch-offset bigints. */
  static fromBigInts(values: bigint[], unit: DatetimeUnit): DatetimeArray {
    const buf = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) buf[i] = values[i]!;
    const nd = wrapNative(() => NDArray._wrap(addon.fromTypedArray(buf, [values.length], "int64")));
    return new DatetimeArray(nd, unit);
  }

  /** Create a 0-d DatetimeArray. */
  static scalar(value: bigint, unit: DatetimeUnit): DatetimeArray {
    const buf = new BigInt64Array([value]);
    const nd = wrapNative(() => NDArray._wrap(addon.fromTypedArray(buf, [], "int64")));
    return new DatetimeArray(nd, unit);
  }
}

// ---------------------------------------------------------------------------
// TimedeltaArray
// ---------------------------------------------------------------------------

export class TimedeltaArray {
  readonly data: NDArray;
  readonly unit: DatetimeUnit;
  readonly shape: readonly number[];
  readonly size: number;
  readonly ndim: number;

  constructor(data: NDArray, unit: DatetimeUnit) {
    if (data.dtype.name !== "int64") {
      throw new DTypeError("TimedeltaArray requires int64 backing storage");
    }
    this.data = data;
    this.unit = unit;
    this.shape = data.shape;
    this.size = data.size;
    this.ndim = data.ndim;
  }

  get dtype(): TimedeltaDType {
    return new TimedeltaDType(this.unit);
  }

  toTypedArray(): BigInt64Array {
    return this.data.toTypedArray() as BigInt64Array;
  }

  toArray(): bigint | bigint[] {
    const raw = this.data.toTypedArray() as BigInt64Array;
    if (this.ndim === 0) return raw[0]!;
    return Array.from(raw) as bigint[];
  }

  toString(): string {
    const raw = this.data.toTypedArray() as BigInt64Array;
    if (this.ndim === 0) {
      const v = raw[0]!;
      return isNaT(v) ? "NaT" : `${v} ${this.unit}`;
    }
    const strs = Array.from(raw).map((v) =>
      isNaT(v) ? "NaT" : `${v} ${this.unit}`,
    );
    return `[${strs.join(", ")}]`;
  }

  static fromBigInts(values: bigint[], unit: DatetimeUnit): TimedeltaArray {
    const buf = new BigInt64Array(values.length);
    for (let i = 0; i < values.length; i++) buf[i] = values[i]!;
    const nd = wrapNative(() => NDArray._wrap(addon.fromTypedArray(buf, [values.length], "int64")));
    return new TimedeltaArray(nd, unit);
  }

  static scalar(value: bigint, unit: DatetimeUnit): TimedeltaArray {
    const buf = new BigInt64Array([value]);
    const nd = wrapNative(() => NDArray._wrap(addon.fromTypedArray(buf, [], "int64")));
    return new TimedeltaArray(nd, unit);
  }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function padLeft(n: number, width: number): string {
  const s = String(Math.abs(n));
  return s.padStart(width, "0");
}

/** Format an epoch-offset int64 value in the given unit as ISO-8601 string. */
export function formatDatetime(value: bigint, unit: DatetimeUnit): string {
  if (isNaT(value)) return "NaT";

  // Convert to milliseconds for Date object (except for Y/M which stay abstract)
  if (unit === "Y") {
    const year = 1970 + Number(value);
    return String(year).padStart(4, "0");
  }
  if (unit === "M") {
    const totalMonths = Number(value);
    const year = Math.floor(totalMonths / 12) + 1970;
    const month = ((totalMonths % 12) + 12) % 12 + 1;
    return `${String(year).padStart(4, "0")}-${padLeft(month, 2)}`;
  }

  // Convert to milliseconds
  const msValue = convertUnits(value, unit, "ms");
  const ms = Number(msValue);
  const date = new Date(ms);

  const Y = date.getUTCFullYear();
  const Mo = date.getUTCMonth() + 1;
  const D = date.getUTCDate();
  const h = date.getUTCHours();
  const m = date.getUTCMinutes();
  const s = date.getUTCSeconds();
  const msRem = date.getUTCMilliseconds();

  const dateStr = `${String(Y).padStart(4, "0")}-${padLeft(Mo, 2)}-${padLeft(D, 2)}`;

  if (unit === "D" || unit === "W") return dateStr;

  const timeStr = `${padLeft(h, 2)}:${padLeft(m, 2)}:${padLeft(s, 2)}`;
  if (unit === "h") return `${dateStr}T${padLeft(h, 2)}`;
  if (unit === "m") return `${dateStr}T${padLeft(h, 2)}:${padLeft(m, 2)}`;
  if (unit === "s") return `${dateStr}T${timeStr}`;

  // sub-second: need to show fractional seconds
  // Compute the sub-second part in the native unit
  const epochInUnit = value;
  // seconds portion
  const secFactor = unitFactor("s", unit)!;
  const fullSeconds = epochInUnit / secFactor;
  const subSecond = epochInUnit - fullSeconds * secFactor;

  if (unit === "ms") {
    return `${dateStr}T${timeStr}.${padLeft(Number(subSecond), 3)}`;
  }
  if (unit === "us") {
    // Get ms portion and us remainder
    const msF = unitFactor("ms", unit)!; // 1000
    const msPart = subSecond / msF;
    const usPart = subSecond % msF;
    return `${dateStr}T${timeStr}.${padLeft(Number(msPart), 3)}${padLeft(Number(usPart), 3)}`;
  }
  if (unit === "ns") {
    const nsF = unitFactor("ms", unit)!; // 1_000_000
    const usF = 1000n;
    const msPart = subSecond / nsF;
    const rem = subSecond % nsF;
    const usPart = rem / usF;
    const nsPart = rem % usF;
    return `${dateStr}T${timeStr}.${padLeft(Number(msPart), 3)}${padLeft(Number(usPart), 3)}${padLeft(Number(nsPart), 3)}`;
  }
  // ps, fs, as: similar pattern
  return `${dateStr}T${timeStr}`;
}

// ---------------------------------------------------------------------------
// np.datetime64 constructor
// ---------------------------------------------------------------------------

export type DatetimeInput = string | number | bigint | DatetimeArray;

/**
 * np.datetime64(value, unit?) — construct a datetime64 scalar or array.
 *
 * @param value - ISO-8601 string, integer (epoch offset), or DatetimeArray.
 * @param unit  - Optional unit string (e.g. "D", "s", "ms"). If omitted,
 *               inferred from the string format.
 */
export function datetime64(value: DatetimeInput, unit?: string): DatetimeArray {
  if (unit !== undefined && !isValidUnit(unit)) {
    throw new ValueError(`unknown datetime unit: '${unit}'`);
  }
  const u = (unit as DatetimeUnit | undefined);

  if (typeof value === "string") {
    if (value === "NaT") {
      return DatetimeArray.scalar(NAT_VALUE, u ?? "s");
    }
    const parsed = parseIsoString(value);
    const targetUnit = u ?? parsed.unit;
    const convertedValue =
      parsed.unit === targetUnit
        ? parsed.value
        : convertUnits(parsed.value, parsed.unit, targetUnit);
    return DatetimeArray.scalar(convertedValue, targetUnit);
  }

  if (typeof value === "number" || typeof value === "bigint") {
    const bv = BigInt(value);
    return DatetimeArray.scalar(bv, u ?? "D");
  }

  if (value instanceof DatetimeArray) {
    if (u === undefined || u === value.unit) return value;
    const raw = value.toTypedArray();
    const converted = Array.from(raw).map((v) =>
      isNaT(v) ? v : convertUnits(v, value.unit, u),
    );
    if (value.ndim === 0) return DatetimeArray.scalar(converted[0]!, u);
    return DatetimeArray.fromBigInts(converted, u);
  }

  throw new ValueError(`unsupported datetime64 input type: ${typeof value}`);
}

// ---------------------------------------------------------------------------
// np.timedelta64 constructor
// ---------------------------------------------------------------------------

export type TimedeltaInput = number | bigint | TimedeltaArray;

/**
 * np.timedelta64(value, unit) — construct a timedelta64 scalar or array.
 */
export function timedelta64(value: TimedeltaInput, unit?: string): TimedeltaArray {
  if (unit !== undefined && !isValidUnit(unit)) {
    throw new ValueError(`unknown timedelta unit: '${unit}'`);
  }
  const u = (unit as DatetimeUnit | undefined) ?? "s";

  if (typeof value === "number" || typeof value === "bigint") {
    return TimedeltaArray.scalar(BigInt(value), u);
  }

  if (value instanceof TimedeltaArray) {
    if (unit === undefined || unit === value.unit) return value;
    const raw = value.toTypedArray();
    const converted = Array.from(raw).map((v) =>
      isNaT(v) ? v : convertUnits(v, value.unit, u),
    );
    if (value.ndim === 0) return TimedeltaArray.scalar(converted[0]!, u);
    return TimedeltaArray.fromBigInts(converted, u);
  }

  throw new ValueError(`unsupported timedelta64 input type: ${typeof value}`);
}

// ---------------------------------------------------------------------------
// np.datetime_data
// ---------------------------------------------------------------------------

/**
 * np.datetime_data(dtype) — returns [unit, count] tuple for a datetime/timedelta dtype.
 * count is always 1 in NumPy for the fundamental types.
 */
export function datetime_data(dtype: DatetimeDType | TimedeltaDType | DatetimeArray | TimedeltaArray): [DatetimeUnit, 1] {
  if (dtype instanceof DatetimeDType || dtype instanceof TimedeltaDType) {
    return [dtype.unit, 1];
  }
  if (dtype instanceof DatetimeArray || dtype instanceof TimedeltaArray) {
    return [dtype.unit, 1];
  }
  throw new DTypeError("datetime_data requires a datetime64 or timedelta64 dtype");
}

// ---------------------------------------------------------------------------
// np.datetime_as_string
// ---------------------------------------------------------------------------

export interface DatetimeAsStringOptions {
  /** Target unit for output (truncates if coarser). Default: array's own unit. */
  unit?: string;
  /** Timezone ('UTC', 'local', or offset like '+05:30'). Default: 'UTC'. */
  timezone?: string;
  /** Casting rule (ignored for now; kept for API compatibility). */
  casting?: string;
}

/**
 * np.datetime_as_string(arr, {unit, timezone, casting}) — format datetime64
 * array elements as ISO-8601 strings.
 */
export function datetime_as_string(
  arr: DatetimeArray,
  options: DatetimeAsStringOptions | string = {},
): string | string[] {
  if (typeof options === "string") {
    options = { unit: options };
  }
  if (!(arr instanceof DatetimeArray)) {
    throw new DTypeError("datetime_as_string requires a DatetimeArray");
  }
  const targetUnit = (options.unit as DatetimeUnit | undefined) ?? arr.unit;
  if (options.unit !== undefined && !isValidUnit(options.unit)) {
    throw new ValueError(`unknown datetime unit: '${options.unit}'`);
  }

  const raw = arr.toTypedArray();
  const format = (v: bigint): string => {
    if (isNaT(v)) return "NaT";
    const converted = convertUnits(v, arr.unit, targetUnit);
    return formatDatetime(converted, targetUnit);
  };

  if (arr.ndim === 0) return format(raw[0]!);
  return Array.from(raw).map(format);
}

// ---------------------------------------------------------------------------
// busdaycalendar
// ---------------------------------------------------------------------------

export interface BusdaycalendarOptions {
  /** Which weekdays are business days. "1111100" = Mon-Fri (default). */
  weekmask?: string | boolean[];
  /** List of holiday dates (ISO-8601 strings or epoch-day integers). */
  holidays?: (string | number | bigint)[];
}

/**
 * np.busdaycalendar — a business-day calendar with weekmask and holidays.
 * weekmask is a 7-element boolean array (Mon=0, Sun=6) or string.
 */
export class busdaycalendar {
  /** 7-element boolean array: Mon (0) through Sun (6). */
  readonly weekmask: readonly boolean[];
  /** Sorted array of holiday epoch-day integers (int64). */
  readonly holidays: readonly bigint[];

  constructor(options: BusdaycalendarOptions = {}) {
    this.weekmask = parseMask(options.weekmask ?? "Mon Tue Wed Thu Fri");
    const rawHolidays = (options.holidays ?? []).map((h) => {
      if (typeof h === "string") return parseIsoString(h).value;
      if (typeof h === "number") return BigInt(h);
      return h as bigint;
    });
    // Convert all holidays to epoch-days and sort
    const holidays = rawHolidays.map((v) => convertUnits(v, "D", "D")).sort((a, b) =>
      a < b ? -1 : a > b ? 1 : 0,
    );
    this.holidays = holidays;
  }
}

function parseMask(mask: string | boolean[]): boolean[] {
  if (Array.isArray(mask)) {
    if (mask.length !== 7) throw new ValueError("weekmask must have 7 elements");
    return mask.slice();
  }
  if (typeof mask === "string") {
    // Could be "1111100" or "Mon Tue Wed Thu Fri"
    if (/^[01]{7}$/.test(mask)) {
      return [...mask].map((c) => c === "1");
    }
    const dayNames: Record<string, number> = {
      Mon: 0, Tue: 1, Wed: 2, Thu: 3, Fri: 4, Sat: 5, Sun: 6,
      Monday: 0, Tuesday: 1, Wednesday: 2, Thursday: 3, Friday: 4, Saturday: 5, Sunday: 6,
    };
    const result = [false, false, false, false, false, false, false];
    const parts = mask.trim().split(/\s+/);
    for (const p of parts) {
      const idx = dayNames[p];
      if (idx === undefined) throw new ValueError(`Invalid weekday name: '${p}'`);
      result[idx] = true;
    }
    return result;
  }
  throw new ValueError("weekmask must be a string or boolean array");
}

// ---------------------------------------------------------------------------
// Business day helpers
// ---------------------------------------------------------------------------

/**
 * Get epoch-day integer for a date input.
 * Accepts ISO-8601 strings, integers (epoch days), or BigInt.
 */
function toEpochDay(date: string | number | bigint | DatetimeArray): bigint {
  if (typeof date === "string") {
    const parsed = parseIsoString(date);
    return convertUnits(parsed.value, parsed.unit, "D");
  }
  if (typeof date === "number") return BigInt(date);
  if (typeof date === "bigint") return date;
  if (date instanceof DatetimeArray) {
    const raw = date.toTypedArray();
    const v = raw[0]!;
    return convertUnits(v, date.unit, "D");
  }
  throw new ValueError(`Invalid date input: ${typeof date}`);
}

function toEpochDayArray(dates: (string | number | bigint | DatetimeArray)[] | string | number | bigint | DatetimeArray): bigint[] {
  if (Array.isArray(dates)) {
    return dates.map(toEpochDay);
  }
  if (dates instanceof DatetimeArray) {
    const raw = dates.toTypedArray();
    return Array.from(raw).map((v) => convertUnits(v, dates.unit, "D"));
  }
  return [toEpochDay(dates)];
}

/** Get the day-of-week (0=Mon, 6=Sun) for an epoch-day. */
function epochDayToWeekday(epochDay: bigint): number {
  // Epoch day 0 = 1970-01-01 = Thursday (3)
  const d = Number(((epochDay % 7n) + 7n) % 7n);
  // 0=Thu, 1=Fri, 2=Sat, 3=Sun, 4=Mon, 5=Tue, 6=Wed
  // We want Mon=0: (d - 4 + 7) % 7
  return ((d - 4) + 7) % 7;
}

function isHoliday(day: bigint, sortedHolidays: readonly bigint[]): boolean {
  // Binary search
  let lo = 0, hi = sortedHolidays.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const h = sortedHolidays[mid]!;
    if (h === day) return true;
    if (h < day) lo = mid + 1;
    else hi = mid - 1;
  }
  return false;
}

function defaultCalendar(): busdaycalendar {
  return new busdaycalendar();
}

function resolveCalendar(
  weekmask?: string | boolean[],
  holidays?: (string | number | bigint)[],
  cal?: busdaycalendar,
): busdaycalendar {
  if (cal !== undefined) return cal;
  return new busdaycalendar({ weekmask, holidays });
}

function isBusday(day: bigint, cal: busdaycalendar): boolean {
  if (isNaT(day)) return false;
  const wd = epochDayToWeekday(day);
  if (!cal.weekmask[wd]) return false;
  return !isHoliday(day, cal.holidays);
}

// ---------------------------------------------------------------------------
// np.is_busday
// ---------------------------------------------------------------------------

export interface IsBusdayOptions {
  weekmask?: string | boolean[];
  holidays?: (string | number | bigint)[];
  busdaycal?: busdaycalendar;
}

/**
 * np.is_busday(dates, {weekmask, holidays, busdaycal}) — test if dates are
 * business days.
 */
export function is_busday(
  dates: string | number | bigint | DatetimeArray | (string | number | bigint | DatetimeArray)[],
  options: IsBusdayOptions = {},
): boolean | boolean[] {
  const cal = resolveCalendar(options.weekmask, options.holidays, options.busdaycal);
  const days = toEpochDayArray(dates as Parameters<typeof toEpochDayArray>[0]);
  const results = days.map((d) => isBusday(d, cal));
  const isScalar =
    !Array.isArray(dates) &&
    !(dates instanceof DatetimeArray) ||
    (dates instanceof DatetimeArray && dates.ndim === 0);
  if (isScalar && results.length === 1) return results[0]!;
  return results;
}

// ---------------------------------------------------------------------------
// np.busday_count
// ---------------------------------------------------------------------------

export interface BusdayCountOptions {
  weekmask?: string | boolean[];
  holidays?: (string | number | bigint)[];
  busdaycal?: busdaycalendar;
}

/**
 * np.busday_count(begindates, enddates, options) — count business days in
 * [begindates, enddates). Negative if end < begin.
 */
export function busday_count(
  begindates: string | number | bigint | DatetimeArray,
  enddates: string | number | bigint | DatetimeArray,
  options: BusdayCountOptions = {},
): number | number[] {
  const cal = resolveCalendar(options.weekmask, options.holidays, options.busdaycal);
  const begins = toEpochDayArray(begindates);
  const ends = toEpochDayArray(enddates);

  // Broadcast: scalar vs array
  const len = Math.max(begins.length, ends.length);
  const results: number[] = [];
  for (let i = 0; i < len; i++) {
    const b = begins[i % begins.length]!;
    const e = ends[i % ends.length]!;
    results.push(countBusDays(b, e, cal));
  }
  const isScalar =
    !Array.isArray(begindates) && !Array.isArray(enddates) &&
    !(begindates instanceof DatetimeArray) && !(enddates instanceof DatetimeArray);
  if (isScalar) return results[0]!;
  return results;
}

function countBusDays(begin: bigint, end: bigint, cal: busdaycalendar): number {
  if (isNaT(begin) || isNaT(end)) return 0;
  let count = 0;
  const step = begin <= end ? 1n : -1n;
  const sign = begin <= end ? 1 : -1;
  let d = begin;
  while (d !== end) {
    if (isBusday(d, cal)) count++;
    d += step;
  }
  return count * sign;
}

// ---------------------------------------------------------------------------
// np.busday_offset
// ---------------------------------------------------------------------------

export type BusdayRoll = "raise" | "nat" | "forward" | "following" | "backward" | "preceding" | "modifiedfollowing" | "modifiedpreceding";

export interface BusdayOffsetOptions {
  roll?: BusdayRoll;
  weekmask?: string | boolean[];
  holidays?: (string | number | bigint)[];
  busdaycal?: busdaycalendar;
}

/**
 * np.busday_offset(dates, offsets, options) — shift dates by a number of
 * business days.
 */
export function busday_offset(
  dates: string | number | bigint | DatetimeArray | (string | number | bigint | DatetimeArray)[],
  offsets: number | number[],
  options: BusdayOffsetOptions = {},
): DatetimeArray | DatetimeArray[] | null {
  const roll: BusdayRoll = options.roll ?? "raise";
  const cal = resolveCalendar(options.weekmask, options.holidays, options.busdaycal);
  const days = toEpochDayArray(dates as Parameters<typeof toEpochDayArray>[0]);
  const offs = Array.isArray(offsets) ? offsets : [offsets];

  const len = Math.max(days.length, offs.length);
  const results: (bigint | null)[] = [];
  for (let i = 0; i < len; i++) {
    const d = days[i % days.length]!;
    const o = offs[i % offs.length]!;
    results.push(applyBusdayOffset(d, o, roll, cal));
  }

  const isScalar =
    !Array.isArray(dates) &&
    !(dates instanceof DatetimeArray) &&
    !Array.isArray(offsets);

  if (isScalar) {
    const v = results[0]!;
    if (v === null) return null; // NaT from roll='nat'
    return DatetimeArray.scalar(v, "D");
  }

  // Array result
  return results.map((v) =>
    v === null ? DatetimeArray.scalar(NAT_VALUE, "D") : DatetimeArray.scalar(v, "D"),
  );
}

function applyBusdayOffset(
  day: bigint,
  offset: number,
  roll: BusdayRoll,
  cal: busdaycalendar,
): bigint | null {
  if (isNaT(day)) return NAT_VALUE;

  // First apply roll to get to a business day
  let current = rollToBusinessDay(day, roll, cal);
  if (current === null) return null; // nat roll

  // Then step offset business days
  const step = offset >= 0 ? 1n : -1n;
  let remaining = Math.abs(offset);
  while (remaining > 0) {
    current += step;
    if (isBusday(current, cal)) remaining--;
  }
  return current;
}

function rollToBusinessDay(
  day: bigint,
  roll: BusdayRoll,
  cal: busdaycalendar,
): bigint | null {
  if (isBusday(day, cal)) return day;

  switch (roll) {
    case "raise":
      throw new ValueError(`Non-business day date in busday_offset`);
    case "nat":
      return null;
    case "forward":
    case "following": {
      let d = day + 1n;
      while (!isBusday(d, cal)) d += 1n;
      return d;
    }
    case "backward":
    case "preceding": {
      let d = day - 1n;
      while (!isBusday(d, cal)) d -= 1n;
      return d;
    }
    case "modifiedfollowing": {
      let d = day + 1n;
      while (!isBusday(d, cal)) d += 1n;
      // If moved to next month, go backward instead
      if (epochDayToMonth(d) !== epochDayToMonth(day)) {
        d = day - 1n;
        while (!isBusday(d, cal)) d -= 1n;
      }
      return d;
    }
    case "modifiedpreceding": {
      let d = day - 1n;
      while (!isBusday(d, cal)) d -= 1n;
      if (epochDayToMonth(d) !== epochDayToMonth(day)) {
        d = day + 1n;
        while (!isBusday(d, cal)) d += 1n;
      }
      return d;
    }
    default:
      throw new ValueError(`unknown roll parameter: '${roll as string}'`);
  }
}

function epochDayToMonth(epochDay: bigint): number {
  const ms = Number(epochDay) * 86400000;
  const d = new Date(ms);
  return d.getUTCFullYear() * 12 + d.getUTCMonth();
}

// ---------------------------------------------------------------------------
// Export the p16d surface object (spread into np by index.ts)
// ---------------------------------------------------------------------------

export const p16d = {
  datetime64,
  timedelta64,
  datetime_data,
  datetime_as_string,
  busdaycalendar,
  is_busday,
  busday_count,
  busday_offset,
} as const;
