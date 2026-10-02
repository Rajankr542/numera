// Parity milestone P13 public functions (D-056).
// Functions exported from `p13` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P13).
// Native kernels live in native/bindings/p13_binding.cpp (`addon.p13`).

export { BitGenerator, MT19937, PCG64, PCG64DXSM, Philox, SFC64, SeedSequence } from "./p13_bitgen.js";
export type { SeedLike } from "./p13_bitgen.js";

import { BitGenerator, MT19937, PCG64, PCG64DXSM, Philox, SFC64, SeedSequence } from "./p13_bitgen.js";

// Side-effect imports: register Generator and RandomState prototype augmentations.
import "./p13_discrete.js";
import "./p13_continuous.js";
import "./p13_multivariate.js";
import "./p13_randomstate.js";

export const p13 = {
  BitGenerator,
  MT19937,
  PCG64,
  PCG64DXSM,
  Philox,
  SFC64,
  SeedSequence,
} as const;
