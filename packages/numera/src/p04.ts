// Parity milestone P4 public functions (D-056): math ufuncs.
// Functions exported from `p04` are spread into the default `np` object by
// index.ts. Native kernels: rows in native/core/ufunc_math.cpp, multi-output
// helpers in native/bindings/p04_binding.cpp (`addon.p04`).
import * as trig from "./p04_trig.js";

export * from "./p04_trig.js";

export const p04 = {
  ...trig,
} as const;
