// Parity milestone P14: extra dtypes and I/O (D-170–D-179).
// Native kernels: native/core/p14_*.cpp (`addon.p14`).
import { load, NpzFile, save, savez, savezCompressed } from "./p14_npy.js";

export { load, NpzFile, save, savez, savezCompressed } from "./p14_npy.js";
export type { BytesLike, FileLike, LoadOptions, NamedArrays } from "./p14_npy.js";

export const p14 = {
  save,
  load,
  savez,
  savezCompressed,
  NpzFile,
} as const;
