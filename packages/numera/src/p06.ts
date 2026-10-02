// Parity milestone P6: array manipulation (D-056, D-090..D-095).
// Functions exported from `p06` are spread into the default `np` object by
// index.ts. Native kernels: native/core/p06_manip.cpp via addon.p06.
import {
  arraySplit,
  atleast1d,
  atleast2d,
  atleast3d,
  block,
  columnStack,
  concat,
  concatenate,
  dsplit,
  dstack,
  hsplit,
  hstack,
  split,
  stack,
  unstack,
  vsplit,
  vstack,
} from "./p06_join.js";

import { append, del, insert, resize, trimZeros } from "./p06_edit.js";
import { pad } from "./p06_pad.js";
import { flip, fliplr, flipud, matrixTranspose, permuteDims, roll, rollaxis, rot90 } from "./p06_reorder.js";
import { repeat, tile } from "./p06_tile.js";

export * from "./p06_join.js";
export * from "./p06_tile.js";
export * from "./p06_pad.js";
export * from "./p06_edit.js";
export * from "./p06_reorder.js";

export const p06 = {
  concatenate,
  concat,
  stack,
  vstack,
  hstack,
  dstack,
  columnStack,
  block,
  unstack,
  split,
  arraySplit,
  hsplit,
  vsplit,
  dsplit,
  atleast1d,
  atleast2d,
  atleast3d,
  tile,
  repeat,
  pad,
  append,
  insert,
  delete: del,
  resize,
  trimZeros,
  flip,
  fliplr,
  flipud,
  roll,
  rollaxis,
  rot90,
  permuteDims,
  matrixTranspose,
} as const;
