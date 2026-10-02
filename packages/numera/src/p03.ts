// Parity milestone P3 public functions (D-056).
// Functions exported from `p03` are spread into the default `np` object by
// index.ts; also add named exports there (only the index.ts block for P3).
// Native kernels live in native/bindings/p03_binding.cpp (`addon.p03`).
// NDArray methods: add them with declaration merging, e.g.
//   declare module "./ndarray.js" { interface NDArray { foo(): NDArray } }
//   NDArray.prototype.foo = function () { ... };
// (see AGENTS.md "Parallel milestones").
import {
  commonType,
  complexfloating,
  finfo,
  floating,
  generic,
  iinfo,
  inexact,
  integer,
  isdtype,
  issubdtype,
  minScalarType,
  mintypecode,
  number,
  resultType,
  signedinteger,
  unsignedinteger,
} from "./p03_dtypes.js";
import { ndenumerate, nditer, ndindex } from "./p03_iter.js";

export { FlatIter } from "./ndarray.js";
export type { FlatIndex, ScalarValue } from "./ndarray.js";
export { NDIter, ndenumerate, nditer, ndindex } from "./p03_iter.js";
export type { NDIterOptions } from "./p03_iter.js";
export {
  AbstractDType,
  commonType,
  complexfloating,
  finfo,
  floating,
  generic,
  iinfo,
  inexact,
  integer,
  isdtype,
  issubdtype,
  minScalarType,
  mintypecode,
  number,
  resultType,
  signedinteger,
  unsignedinteger,
} from "./p03_dtypes.js";
export type { FInfo, IInfo } from "./p03_dtypes.js";

export const p03 = {
  ndindex,
  ndenumerate,
  nditer,
  finfo,
  iinfo,
  resultType,
  minScalarType,
  issubdtype,
  isdtype,
  commonType,
  mintypecode,
  generic,
  number,
  integer,
  signedinteger,
  unsignedinteger,
  inexact,
  floating,
  complexfloating,
} as const;
