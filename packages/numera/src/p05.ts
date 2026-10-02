// Parity milestone P5 public functions (D-056): comparison, logic, bitwise.
// Functions exported from `p05` are spread into the default `np` object by
// index.ts, which also does `export * from "./p05.js"`.
import {
  all,
  any,
  equal,
  greater,
  greaterEqual,
  less,
  lessEqual,
  logicalAnd,
  logicalNot,
  logicalOr,
  logicalXor,
  notEqual,
} from "./p05_compare.js";

export {
  all,
  any,
  equal,
  greater,
  greaterEqual,
  less,
  lessEqual,
  logicalAnd,
  logicalNot,
  logicalOr,
  logicalXor,
  notEqual,
} from "./p05_compare.js";
import { isfinite, isinf, isnan, isnat, isneginf, isposinf, isscalar } from "./p05_classify.js";

export { isfinite, isinf, isnan, isnat, isneginf, isposinf, isscalar } from "./p05_classify.js";
export type { AllAnyOptions } from "./p05_compare.js";

export const p05 = {
  equal,
  notEqual,
  less,
  lessEqual,
  greater,
  greaterEqual,
  logicalAnd,
  logicalOr,
  logicalXor,
  logicalNot,
  all,
  any,
  isnan,
  isinf,
  isfinite,
  isnat,
  isposinf,
  isneginf,
  isscalar,
} as const;
