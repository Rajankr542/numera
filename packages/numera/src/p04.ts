// Parity milestone P4 public functions (D-056): math ufuncs.
// Functions exported from `p04` are spread into the default `np` object by
// index.ts. Native kernels: rows in native/core/ufunc_math.cpp, multi-output
// helpers in native/bindings/p04_binding.cpp (`addon.p04`).
import * as trig from "./p04_trig.js";
import * as explog from "./p04_explog.js";
import * as rounding from "./p04_rounding.js";
import * as arith from "./p04_arith.js";
import * as bits from "./p04_bits.js";

export * from "./p04_trig.js";
export * from "./p04_explog.js";
export * from "./p04_rounding.js";
export * from "./p04_arith.js";
export * from "./p04_bits.js";

export const p04 = {
  ...trig,
  ...explog,
  ...rounding,
  ...arith,
  ...bits,
} as const;
