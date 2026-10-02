// Parity milestone P11 public functions (D-056, D-140..D-142).
// `p11` is spread into the default `np` object by index.ts; the new
// `np.linalg` names are added to the `linalg` object in linalg.ts.
export type { CholeskyOptions, SlogdetResult } from "./p11_linalg.js";

import { cross, kron, tensordot, vdot } from "./p11_products.js";

export { cross, kron, tensordot, vdot };

export type { CrossOptions, TensordotAxes, TensordotOptions } from "./p11_products.js";

export const p11 = { vdot, kron, cross, tensordot } as const;
