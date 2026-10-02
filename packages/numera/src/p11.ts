// Parity milestone P11 public functions (D-056, D-140..D-142).
// `p11` is spread into the default `np` object by index.ts; the new
// `np.linalg` names are added to the `linalg` object in linalg.ts.
export type { CholeskyOptions, SlogdetResult } from "./p11_linalg.js";

import { cross, kron, matvec, tensordot, vdot, vecdot, vecmat } from "./p11_products.js";

import { einsum, einsumPath } from "./p11_einsum.js";

export { cross, einsum, einsumPath, kron, matvec, tensordot, vdot, vecdot, vecmat };
export type { EinsumOptimize, EinsumOptions, EinsumPath, Sublist } from "./p11_einsum.js";

export type { CrossOptions, TensordotAxes, TensordotOptions, VecdotOptions } from "./p11_products.js";

export const p11 = { vdot, kron, cross, tensordot, vecdot, matvec, vecmat, einsum, einsumPath } as const;
