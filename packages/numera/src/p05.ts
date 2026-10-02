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
import {
  bitwiseAnd,
  bitwiseCount,
  bitwiseInvert,
  bitwiseLeftShift,
  bitwiseNot,
  bitwiseOr,
  bitwiseRightShift,
  bitwiseXor,
  invert,
  leftShift,
  rightShift,
} from "./p05_bitwise.js";

export {
  bitwiseAnd,
  bitwiseCount,
  bitwiseInvert,
  bitwiseLeftShift,
  bitwiseNot,
  bitwiseOr,
  bitwiseRightShift,
  bitwiseXor,
  invert,
  leftShift,
  rightShift,
} from "./p05_bitwise.js";
import { allclose, arrayEqual, arrayEquiv, isclose } from "./p05_close.js";

export { allclose, arrayEqual, arrayEquiv, isclose } from "./p05_close.js";
export type { IscloseOptions } from "./p05_close.js";
import { packbits, unpackbits } from "./p05_bits.js";

export { packbits, unpackbits } from "./p05_bits.js";
export type { BitOrder, PackbitsOptions, UnpackbitsOptions } from "./p05_bits.js";
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
  bitwiseAnd,
  bitwiseOr,
  bitwiseXor,
  invert,
  bitwiseNot,
  bitwiseInvert,
  leftShift,
  rightShift,
  bitwiseLeftShift,
  bitwiseRightShift,
  bitwiseCount,
  isclose,
  allclose,
  arrayEqual,
  arrayEquiv,
  packbits,
  unpackbits,
} as const;
