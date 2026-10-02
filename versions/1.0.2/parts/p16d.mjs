// P16D API reference entries (D-190, D-220).
// Datetime64/timedelta64/busday functions.
/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "dtype",
    title: "Data types",
    entries: [
      {
        name: "datetime64",
        sig: "np.datetime64(value, [unit])",
        desc: "Construct a `datetime64` scalar from an ISO-8601 string, integer epoch offset, or another `DatetimeArray`. The returned `DatetimeArray` holds int64 epoch offsets and a unit string.",
        args: [
          { name: "value", type: "string | number | bigint | DatetimeArray", desc: "ISO-8601 string (e.g. `\"2023-01-15\"`) or integer epoch offset." },
          { name: "[unit]", type: "string", desc: "Datetime unit: `\"Y\"` `\"M\"` `\"W\"` `\"D\"` `\"h\"` `\"m\"` `\"s\"` `\"ms\"` `\"us\"` `\"ns\"`. Inferred from the string when omitted." },
        ],
        returns: "DatetimeArray — a 0-d datetime64 array.",
        example: `np.datetime64("2023-01-15").unit; // => "D"`,
      },
      {
        name: "timedelta64",
        sig: "np.timedelta64(value, [unit])",
        desc: "Construct a `timedelta64` scalar from an integer duration and unit.",
        args: [
          { name: "value", type: "number | bigint | TimedeltaArray", desc: "Duration value." },
          { name: "[unit]", type: "string", desc: "Time unit (same set as `datetime64`). Defaults to `\"s\"`." },
        ],
        returns: "TimedeltaArray — a 0-d timedelta64 array.",
        example: `np.timedelta64(5, "D").unit; // => "D"`,
      },
      {
        name: "datetime_data",
        sig: "np.datetime_data(dtype)",
        desc: "Returns the unit and count for a `datetime64` or `timedelta64` dtype. The count is always `1`.",
        args: [
          { name: "dtype", type: "DatetimeDType | TimedeltaDType | DatetimeArray | TimedeltaArray", desc: "A datetime or timedelta dtype or array." },
        ],
        returns: "[unit, count] — a tuple `[string, 1]`.",
        example: `np.datetime_data(np.datetime64("2023-01-15"))[0]; // => "D"`,
      },
      {
        name: "datetime_as_string",
        sig: "np.datetime_as_string(arr, [options])",
        desc: "Convert a `DatetimeArray` to ISO-8601 string(s). Optionally truncate to a coarser unit.",
        args: [
          { name: "arr", type: "DatetimeArray", desc: "The datetime array to format." },
          { name: "[options.unit]", type: "string", desc: "Output unit (truncates to this resolution). Defaults to the array's own unit." },
          { name: "[options.timezone]", type: "string", desc: "Timezone hint (accepted, not applied; all output is UTC)." },
        ],
        returns: "string | string[] — one string per element.",
        example: `np.datetime_as_string(np.datetime64("2023-01-15")); // => "2023-01-15"`,
      },
    ],
  },
  {
    id: "utilities",
    title: "Utilities",
    entries: [
      {
        name: "busdaycalendar",
        sig: "new np.busdaycalendar([options])",
        desc: "A business-day calendar combining a weekmask (which weekdays are business days) and optional holidays. Used with `is_busday`, `busday_count`, and `busday_offset`.",
        args: [
          { name: "[options.weekmask]", type: "string | boolean[]", desc: "7-character bit string `\"1111100\"`, a space-separated list of day names like `\"Mon Tue Wed Thu Fri\"`, or a boolean array. Defaults to Mon–Fri." },
          { name: "[options.holidays]", type: "(string | number | bigint)[]", desc: "List of holiday dates as ISO-8601 strings or epoch-day integers." },
        ],
        returns: "busdaycalendar",
        example: `new np.busdaycalendar({ weekmask: "Mon Tue Wed Thu Fri" }).weekmask[0]; // => true`,
      },
      {
        name: "is_busday",
        sig: "np.is_busday(dates, [options])",
        desc: "Return `true` for each date that is a business day (not a weekend and not a holiday).",
        args: [
          { name: "dates", type: "string | number | DatetimeArray | Array", desc: "One or more dates." },
          { name: "[options.weekmask]", type: "string | boolean[]", desc: "Which weekdays count as business days. Defaults to Mon–Fri." },
          { name: "[options.holidays]", type: "(string | number | bigint)[]", desc: "Holidays to exclude." },
          { name: "[options.busdaycal]", type: "busdaycalendar", desc: "Pre-built calendar; overrides weekmask/holidays." },
        ],
        returns: "boolean | boolean[]",
        example: `np.is_busday("2023-01-16"); // => true`,
      },
      {
        name: "busday_count",
        sig: "np.busday_count(begindates, enddates, [options])",
        desc: "Count the number of business days in `[begindates, enddates)`. Negative when `end < begin`.",
        args: [
          { name: "begindates", type: "string | number | DatetimeArray", desc: "Start date(s), inclusive." },
          { name: "enddates", type: "string | number | DatetimeArray", desc: "End date(s), exclusive." },
          { name: "[options.weekmask]", type: "string | boolean[]", desc: "Which weekdays count. Defaults to Mon–Fri." },
          { name: "[options.holidays]", type: "(string | number | bigint)[]", desc: "Holidays to exclude." },
          { name: "[options.busdaycal]", type: "busdaycalendar", desc: "Pre-built calendar." },
        ],
        returns: "number | number[]",
        example: `np.busday_count("2023-01-01", "2023-01-08"); // => 5`,
      },
      {
        name: "busday_offset",
        sig: "np.busday_offset(dates, offsets, [options])",
        desc: "Shift `dates` by `offsets` business days. The `roll` option controls what to do when a date falls on a non-business day before stepping.",
        args: [
          { name: "dates", type: "string | number | DatetimeArray | Array", desc: "Starting date(s)." },
          { name: "offsets", type: "number | number[]", desc: "Number of business days to advance (positive) or retreat (negative)." },
          { name: "[options.roll]", type: "string", desc: "`\"raise\"` (default), `\"nat\"`, `\"forward\"`, `\"following\"`, `\"backward\"`, `\"preceding\"`, `\"modifiedfollowing\"`, `\"modifiedpreceding\"`." },
          { name: "[options.weekmask]", type: "string | boolean[]", desc: "Which weekdays count. Defaults to Mon–Fri." },
          { name: "[options.holidays]", type: "(string | number | bigint)[]", desc: "Holidays to exclude." },
          { name: "[options.busdaycal]", type: "busdaycalendar", desc: "Pre-built calendar." },
        ],
        returns: "DatetimeArray | DatetimeArray[]",
        example: `np.busday_offset("2023-01-16", 5).toString(); // => "2023-01-23"`,
      },
    ],
  },
];
