import { describe, it, expect } from "vitest";
import np from "../src/index.js";
import {
  DatetimeArray,
  TimedeltaArray,
  DatetimeDType,
  TimedeltaDType,
  datetime64,
  timedelta64,
  datetime_data,
  datetime_as_string,
  busdaycalendar,
  is_busday,
  busday_count,
  busday_offset,
  formatDatetime,
} from "../src/p16d.js";

describe("datetime64", () => {
  it("constructs from ISO date string", () => {
    const d = datetime64("2023-01-15");
    expect(d).toBeInstanceOf(DatetimeArray);
    expect(d.unit).toBe("D");
    expect(d.ndim).toBe(0);
    // epoch day for 2023-01-15
    const raw = d.toTypedArray();
    expect(typeof raw[0]).toBe("bigint");
    expect(raw[0]).toBe(19372n); // days since 1970-01-01
  });

  it("constructs from ISO date string with explicit unit", () => {
    const d = datetime64("2023-01-15", "D");
    expect(d.unit).toBe("D");
    expect(d.toTypedArray()[0]).toBe(19372n);
  });

  it("constructs from year-only string", () => {
    const d = datetime64("1970");
    expect(d.unit).toBe("Y");
    expect(d.toTypedArray()[0]).toBe(0n);
  });

  it("constructs from year-month string", () => {
    const d = datetime64("1970-01");
    expect(d.unit).toBe("M");
    expect(d.toTypedArray()[0]).toBe(0n);
  });

  it("constructs from datetime string with seconds", () => {
    const d = datetime64("2023-01-15T10:30:00", "s");
    expect(d.unit).toBe("s");
    // epoch seconds = epoch days * 86400 + 10*3600 + 30*60
    const expected = 19372n * 86400n + 10n * 3600n + 30n * 60n;
    expect(d.toTypedArray()[0]).toBe(expected);
  });

  it("NaT from string", () => {
    const d = datetime64("NaT");
    expect(d.toTypedArray()[0]).toBe(-9223372036854775808n);
  });

  it("constructs from integer epoch offset", () => {
    const d = datetime64(0, "D");
    expect(d.toTypedArray()[0]).toBe(0n);
    expect(d.unit).toBe("D");
  });

  it("constructs from bigint", () => {
    const d = datetime64(19372n, "D");
    expect(d.toTypedArray()[0]).toBe(19372n);
  });

  it("dtype returns DatetimeDType", () => {
    const d = datetime64("2023-01-15");
    expect(d.dtype).toBeInstanceOf(DatetimeDType);
    expect(d.dtype.unit).toBe("D");
    expect(d.dtype.kind).toBe("M");
    expect(d.dtype.name).toBe("datetime64[D]");
  });

  it("toString formats as ISO string", () => {
    const d = datetime64("2023-01-15");
    expect(d.toString()).toBe("2023-01-15");
  });

  it("NaT toString is NaT", () => {
    const d = datetime64("NaT");
    expect(d.toString()).toBe("NaT");
  });

  it("is accessible as np.datetime64", () => {
    const d = np.datetime64("2023-01-15");
    expect(d).toBeInstanceOf(DatetimeArray);
  });
});

describe("timedelta64", () => {
  it("constructs from integer and unit", () => {
    const td = timedelta64(5, "D");
    expect(td).toBeInstanceOf(TimedeltaArray);
    expect(td.unit).toBe("D");
    expect(td.toTypedArray()[0]).toBe(5n);
  });

  it("constructs from bigint", () => {
    const td = timedelta64(100n, "s");
    expect(td.unit).toBe("s");
    expect(td.toTypedArray()[0]).toBe(100n);
  });

  it("dtype returns TimedeltaDType", () => {
    const td = timedelta64(5, "D");
    expect(td.dtype).toBeInstanceOf(TimedeltaDType);
    expect(td.dtype.kind).toBe("m");
    expect(td.dtype.name).toBe("timedelta64[D]");
  });

  it("toString includes value and unit", () => {
    const td = timedelta64(5, "D");
    expect(td.toString()).toBe("5 D");
  });

  it("NaT representation", () => {
    const td = timedelta64(-9223372036854775808n, "D");
    expect(td.toString()).toBe("NaT");
  });

  it("is accessible as np.timedelta64", () => {
    const td = np.timedelta64(3, "D");
    expect(td).toBeInstanceOf(TimedeltaArray);
  });
});

describe("datetime_data", () => {
  it("returns [unit, 1] for DatetimeDType", () => {
    const dt = new DatetimeDType("D");
    const [unit, count] = datetime_data(dt);
    expect(unit).toBe("D");
    expect(count).toBe(1);
  });

  it("returns [unit, 1] for TimedeltaDType", () => {
    const td = new TimedeltaDType("ms");
    const [unit, count] = datetime_data(td);
    expect(unit).toBe("ms");
    expect(count).toBe(1);
  });

  it("returns [unit, 1] for DatetimeArray", () => {
    const d = datetime64("2023-01-15");
    const [unit, count] = datetime_data(d);
    expect(unit).toBe("D");
    expect(count).toBe(1);
  });

  it("is accessible as np.datetime_data", () => {
    const d = np.datetime64("2023-01-15");
    const [unit] = np.datetime_data(d);
    expect(unit).toBe("D");
  });
});

describe("datetime_as_string", () => {
  it("formats date-unit datetime as ISO date", () => {
    const d = datetime64("2023-01-15");
    const s = datetime_as_string(d);
    expect(s).toBe("2023-01-15");
  });

  it("truncates to coarser unit", () => {
    const d = datetime64("2023-01-15T10:30:00", "s");
    const s = datetime_as_string(d, { unit: "D" });
    expect(s).toBe("2023-01-15");
  });

  it("truncates to hour", () => {
    const d = datetime64("2023-01-15T10:30:00", "s");
    const s = datetime_as_string(d, { unit: "h" });
    expect(s).toBe("2023-01-15T10");
  });

  it("formats NaT as NaT", () => {
    const d = datetime64("NaT");
    const s = datetime_as_string(d);
    expect(s).toBe("NaT");
  });

  it("accepts string options shorthand", () => {
    const d = datetime64("2023-01-15");
    const s = datetime_as_string(d, "D");
    expect(s).toBe("2023-01-15");
  });

  it("is accessible as np.datetime_as_string", () => {
    const d = np.datetime64("2023-01-15");
    expect(np.datetime_as_string(d)).toBe("2023-01-15");
  });

  it("formats an array of dates", () => {
    const a = DatetimeArray.fromBigInts([19372n, 19373n], "D");
    const strs = datetime_as_string(a);
    expect(strs).toEqual(["2023-01-15", "2023-01-16"]);
  });
});

describe("busdaycalendar", () => {
  it("default weekmask is Mon-Fri", () => {
    const cal = new busdaycalendar();
    expect(cal.weekmask).toEqual([true, true, true, true, true, false, false]);
  });

  it("accepts string weekmask Mon-Fri", () => {
    const cal = new busdaycalendar({ weekmask: "Mon Tue Wed Thu Fri" });
    expect(cal.weekmask[0]).toBe(true);
    expect(cal.weekmask[5]).toBe(false); // Saturday
    expect(cal.weekmask[6]).toBe(false); // Sunday
  });

  it("accepts bit-string weekmask", () => {
    const cal = new busdaycalendar({ weekmask: "1111110" }); // Mon-Sat
    expect(cal.weekmask[5]).toBe(true);  // Saturday
    expect(cal.weekmask[6]).toBe(false); // Sunday
  });

  it("accepts boolean array weekmask", () => {
    const wm = [true, true, true, true, true, false, false];
    const cal = new busdaycalendar({ weekmask: wm });
    expect(cal.weekmask).toEqual(wm);
  });

  it("stores sorted holidays", () => {
    const cal = new busdaycalendar({ holidays: ["2023-01-16", "2023-01-02"] });
    const h = cal.holidays as bigint[];
    expect(h.length).toBe(2);
    expect(h[0]! < h[1]!).toBe(true);
  });

  it("is accessible as np.busdaycalendar", () => {
    const cal = new np.busdaycalendar();
    expect(cal.weekmask[0]).toBe(true);
  });
});

describe("is_busday", () => {
  it("Monday is a business day", () => {
    expect(is_busday("2023-01-16")).toBe(true); // Monday
  });

  it("Sunday is not a business day", () => {
    expect(is_busday("2023-01-15")).toBe(false); // Sunday
  });

  it("Saturday is not a business day", () => {
    expect(is_busday("2023-01-14")).toBe(false); // Saturday
  });

  it("holiday is not a business day", () => {
    expect(is_busday("2023-01-16", { holidays: ["2023-01-16"] })).toBe(false);
  });

  it("accepts array input", () => {
    const result = is_busday(["2023-01-16", "2023-01-15", "2023-01-17"]);
    expect(result).toEqual([true, false, true]);
  });

  it("accepts busdaycal option", () => {
    const cal = new busdaycalendar({ weekmask: "Mon Tue Wed Thu Fri", holidays: ["2023-01-16"] });
    expect(is_busday("2023-01-16", { busdaycal: cal })).toBe(false);
  });

  it("4-day week: Friday is not busday with Mon-Thu weekmask", () => {
    const result = is_busday("2023-01-20", { weekmask: "Mon Tue Wed Thu" }); // Friday
    expect(result).toBe(false);
  });

  it("is accessible as np.is_busday", () => {
    expect(np.is_busday("2023-01-16")).toBe(true);
  });
});

describe("busday_count", () => {
  it("counts business days in a week (Mon-Fri: 5)", () => {
    // 2023-01-02 is Monday, 2023-01-07 is Saturday
    const count = busday_count("2023-01-02", "2023-01-07");
    expect(count).toBe(5);
  });

  it("counts 21 business days in January 2023", () => {
    expect(busday_count("2023-01-01", "2023-01-31")).toBe(21);
  });

  it("returns 0 for same begin/end", () => {
    expect(busday_count("2023-01-16", "2023-01-16")).toBe(0);
  });

  it("returns negative when end < begin", () => {
    // NumPy semantics: busday_count(begin, end) counts [begin, end)
    // From Sat Jan 7 to Mon Jan 2: goes backward through Fri,Thu,Wed,Tue = -4
    expect(busday_count("2023-01-07", "2023-01-02")).toBe(-4);
  });

  it("excludes holidays", () => {
    const count = busday_count("2023-01-02", "2023-01-07", { holidays: ["2023-01-04"] });
    expect(count).toBe(4);
  });

  it("works with 4-day week weekmask", () => {
    // Mon-Thu only: 4 business days per week
    const count = busday_count("2023-01-02", "2023-01-09", { weekmask: "Mon Tue Wed Thu" });
    expect(count).toBe(4);
  });

  it("is accessible as np.busday_count", () => {
    expect(np.busday_count("2023-01-01", "2023-01-08")).toBe(5);
  });
});

describe("busday_offset", () => {
  it("offsets from a Monday by 5 business days", () => {
    const result = busday_offset("2023-01-16", 5) as DatetimeArray;
    expect(result.toString()).toBe("2023-01-23");
  });

  it("offset 0 from a Monday returns Monday", () => {
    const result = busday_offset("2023-01-16", 0) as DatetimeArray;
    expect(result.toString()).toBe("2023-01-16");
  });

  it("roll='forward' on Sunday moves to Monday", () => {
    const result = busday_offset("2023-01-15", 0, { roll: "forward" }) as DatetimeArray;
    expect(result.toString()).toBe("2023-01-16");
  });

  it("roll='backward' on Sunday moves to Friday", () => {
    const result = busday_offset("2023-01-15", 0, { roll: "backward" }) as DatetimeArray;
    expect(result.toString()).toBe("2023-01-13");
  });

  it("roll='nat' on Sunday returns null", () => {
    const result = busday_offset("2023-01-15", 0, { roll: "nat" });
    expect(result).toBeNull();
  });

  it("roll='raise' on Sunday throws", () => {
    expect(() => busday_offset("2023-01-15", 0, { roll: "raise" })).toThrow();
  });

  it("negative offset goes backward in business days", () => {
    const result = busday_offset("2023-01-23", -5) as DatetimeArray;
    expect(result.toString()).toBe("2023-01-16");
  });

  it("is accessible as np.busday_offset", () => {
    const result = np.busday_offset("2023-01-16", 5) as DatetimeArray;
    expect(result.toString()).toBe("2023-01-23");
  });

  it("excludes holidays in offset", () => {
    // Monday + 5 days, but Wednesday is a holiday — skip to Thursday
    const result = busday_offset("2023-01-16", 5, { holidays: ["2023-01-18"] }) as DatetimeArray;
    expect(result.toString()).toBe("2023-01-24");
  });
});

describe("formatDatetime", () => {
  it("formats epoch day 0 as 1970-01-01", () => {
    expect(formatDatetime(0n, "D")).toBe("1970-01-01");
  });

  it("formats seconds to ISO", () => {
    // 2023-01-15T10:30:00 = 19372 days * 86400 + 37800 s
    const v = 19372n * 86400n + 37800n;
    expect(formatDatetime(v, "s")).toBe("2023-01-15T10:30:00");
  });

  it("formats year", () => {
    expect(formatDatetime(53n, "Y")).toBe("2023");
  });

  it("formats year-month", () => {
    expect(formatDatetime(636n, "M")).toBe("2023-01");
  });

  it("NaT returns NaT", () => {
    expect(formatDatetime(-9223372036854775808n, "D")).toBe("NaT");
  });
});

describe("DatetimeArray helpers", () => {
  it("fromBigInts creates 1-D array", () => {
    const a = DatetimeArray.fromBigInts([0n, 1n, 2n], "D");
    expect(a.ndim).toBe(1);
    expect(a.shape).toEqual([3]);
    expect(a.unit).toBe("D");
  });

  it("scalar creates 0-D array", () => {
    const a = DatetimeArray.scalar(19372n, "D");
    expect(a.ndim).toBe(0);
    expect(a.shape).toEqual([]);
  });
});

describe("TimedeltaArray helpers", () => {
  it("fromBigInts creates 1-D array", () => {
    const a = TimedeltaArray.fromBigInts([1n, 2n, 3n], "D");
    expect(a.ndim).toBe(1);
    expect(a.unit).toBe("D");
  });

  it("scalar creates 0-D array", () => {
    const a = TimedeltaArray.scalar(5n, "D");
    expect(a.ndim).toBe(0);
  });
});
