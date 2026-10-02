// Parity milestone P14: extra dtypes and I/O (D-170–D-179).
// Native kernels: native/core/p14_*.cpp (`addon.p14`).
import { load, NpzFile, save, savez, savezCompressed } from "./p14_npy.js";

export { load, NpzFile, save, savez, savezCompressed } from "./p14_npy.js";
import { loadtxt, savetxt } from "./p14_text.js";
import { fromregex, genfromtxt } from "./p14_genfromtxt.js";
import { fromfile } from "./p14_file.js";
import { bartlett, baseRepr, binaryRepr, blackman, hamming, hanning, kaiser } from "./p14_misc.js";

export { loadtxt, savetxt } from "./p14_text.js";
export { fromfile } from "./p14_file.js";
export { bartlett, baseRepr, binaryRepr, blackman, hamming, hanning, kaiser } from "./p14_misc.js";
export type { BinaryReprOptions, IntegerLike } from "./p14_misc.js";
export type { FromfileOptions, TofileOptions } from "./p14_file.js";
export { fromregex, genfromtxt } from "./p14_genfromtxt.js";
export type { FieldList, GenfromtxtOptions, PerColumn } from "./p14_genfromtxt.js";
export type { LoadtxtOptions, SavetxtOptions, TextSource } from "./p14_text.js";
export type { BytesLike, FileLike, LoadOptions, NamedArrays } from "./p14_npy.js";

export const p14 = {
  save,
  load,
  savez,
  savezCompressed,
  NpzFile,
  loadtxt,
  savetxt,
  genfromtxt,
  fromregex,
  fromfile,
  baseRepr,
  binaryRepr,
  bartlett,
  blackman,
  hamming,
  hanning,
  kaiser,
} as const;
