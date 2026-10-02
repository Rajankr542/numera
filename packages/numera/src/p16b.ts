/**
 * P16B public functions (D-190).
 * Exports the np.ma masked array module.
 */
export { ma } from "./ma.js";
export type { MaskedArrayOptions } from "./ma.js";
export {
  MaskedArray, MaskedIterator, MaskError, MAError, MaskType,
  masked, masked_singleton, nomask, mvoid, bool_,
} from "./ma.js";
import { ma } from "./ma.js";

export const p16b = { ma } as const;
