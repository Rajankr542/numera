// Parity milestone P15 public submodules (D-056): np.emath, np.testing,
// np.polynomial, np.ma. index.ts spreads `p15` into the default `np` object
// and does `export * from "./p15.js"`; only the submodule objects (and
// classes/types) are named exports here, since their members would clash
// with top-level names such as sqrt.
import { emath } from "./p15_emath.js";
import { testing } from "./p15_testing.js";
import { polynomial } from "./p15_polynomial.js";

export { emath } from "./p15_emath.js";
export { testing, AssertionError } from "./p15_testing.js";
export type {
  AllcloseOptions,
  AlmostEqualOptions,
  ApproxEqualOptions,
  AssertArrayCompareOptions,
  AssertOptions,
  BuildErrMsgOptions,
  MaxUlpOptions,
} from "./p15_testing.js";

export { polynomial } from "./p15_polynomial.js";
export type { ClassFitOptions, FitOptions, IntegOptions, SeriesOptions } from "./p15_polynomial.js";

export const p15 = { emath, testing, polynomial } as const;
