// Parity milestone P5 public functions (D-056): comparison, logic, bitwise.
// Functions exported from `p05` are spread into the default `np` object by
// index.ts, which also does `export * from "./p05.js"`.
import { equal, greater, greaterEqual, less, lessEqual, notEqual } from "./p05_compare.js";

export { equal, greater, greaterEqual, less, lessEqual, notEqual } from "./p05_compare.js";

export const p05 = {
  equal,
  notEqual,
  less,
  lessEqual,
  greater,
  greaterEqual,
} as const;
