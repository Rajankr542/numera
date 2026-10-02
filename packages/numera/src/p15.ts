// Parity milestone P15 public submodules (D-056): np.emath, np.testing,
// np.polynomial, np.ma. index.ts spreads `p15` into the default `np` object
// and does `export * from "./p15.js"`; only the submodule objects are named
// exports here (their members would clash with top-level names such as sqrt).
import { emath } from "./p15_emath.js";

export { emath } from "./p15_emath.js";

export const p15 = { emath } as const;
