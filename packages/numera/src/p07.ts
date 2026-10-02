// Parity milestone P7 public functions (D-056): creation routines and grids
// (D-100, D-101). Kernels: native/bindings/p07_binding.cpp (`addon.p07`).
import * as creation from "./p07_creation.js";
import * as grids from "./p07_grids.js";

export * from "./p07_creation.js";
export * from "./p07_grids.js";

export const p07 = {
  logspace: creation.logspace,
  geomspace: creation.geomspace,
  tri: creation.tri,
  tril: creation.tril,
  triu: creation.triu,
  diag: creation.diag,
  diagflat: creation.diagflat,
  vander: creation.vander,
  trilIndices: creation.trilIndices,
  triuIndices: creation.triuIndices,
  trilIndicesFrom: creation.trilIndicesFrom,
  triuIndicesFrom: creation.triuIndicesFrom,
  diagIndices: creation.diagIndices,
  diagIndicesFrom: creation.diagIndicesFrom,
  maskIndices: creation.maskIndices,
  fillDiagonal: creation.fillDiagonal,
  fromfunction: creation.fromfunction,
  fromiter: creation.fromiter,
  frombuffer: creation.frombuffer,
  fromstring: creation.fromstring,
  astype: creation.astype,
  indices: grids.indices,
  meshgrid: grids.meshgrid,
  mgrid: grids.mgrid,
  ogrid: grids.ogrid,
  ix_: grids.ix_,
  r_: grids.r_,
  c_: grids.c_,
  s_: grids.s_,
  indexExp: grids.indexExp,
} as const;
