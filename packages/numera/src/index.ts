import { addon } from "./addon.js";
import * as dtypes from "./dtype.js";
import * as errors from "./errors.js";
import {
  arange,
  array,
  asarray,
  asStrided,
  empty,
  emptyLike,
  eye,
  fromTypedArray,
  full,
  fullLike,
  identity,
  linspace,
  mayShareMemory,
  ones,
  onesLike,
  zeros,
  zerosLike,
} from "./creation.js";
import { NDArray } from "./ndarray.js";
import { Complex, complex } from "./complex.js";
import { ellipsis, newaxis, nonzero, take, where } from "./indexing.js";
import { argmax, argmin, max, mean, min, prod, std, sum, variance } from "./reduce.js";
import { dot, inner, linalg, matmul, outer } from "./linalg.js";
import { random } from "./random.js";
import { fftModule } from "./fft.js";
import { expandDims, moveAxis, ravel, reshape, squeeze, swapAxes, transpose } from "./shape.js";
import {
  abs,
  add,
  broadcastShapes,
  broadcastTo,
  divide,
  exp,
  floorDivide,
  log,
  mod,
  multiply,
  negative,
  power,
  sqrt,
  subtract,
  angle,
  conj,
  conjugate,
  imag,
  iscomplex,
  iscomplexobj,
  isreal,
  isrealobj,
  real,
} from "./ufunc.js";
export { canCast, DType, dtype, promoteTypes } from "./dtype.js";
export type { Casting, DTypeLike, DTypeName } from "./dtype.js";
export {
  bool,
  int8,
  uint8,
  int16,
  uint16,
  int32,
  uint32,
  int64,
  uint64,
  float16,
  float32,
  float64,
  complex64,
  complex128,
} from "./dtype.js";
export * from "./errors.js";
export { NDArray } from "./ndarray.js";
export { Complex, complex } from "./complex.js";
export type { ComplexLike } from "./complex.js";
export type { ArrayFlags, NestedArray, Shape } from "./ndarray.js";
export {
  arange,
  array,
  asarray,
  asStrided,
  empty,
  emptyLike,
  eye,
  fromTypedArray,
  full,
  fullLike,
  identity,
  inferDType,
  linspace,
  mayShareMemory,
  ones,
  onesLike,
  zeros,
  zerosLike,
} from "./creation.js";
export type { ArrayOptions, EyeOptions, LinspaceOptions } from "./creation.js";
export { expandDims, moveAxis, ravel, reshape, squeeze, swapAxes, transpose } from "./shape.js";
export {
  abs,
  add,
  broadcastShapes,
  broadcastTo,
  divide,
  exp,
  floorDivide,
  log,
  mod,
  multiply,
  negative,
  power,
  sqrt,
  subtract,
  angle,
  conj,
  conjugate,
  imag,
  iscomplex,
  iscomplexobj,
  isreal,
  isrealobj,
  real,
} from "./ufunc.js";
export type { ArrayLike, Operand, UfuncOptions } from "./ufunc.js";
export { ellipsis, newaxis, nonzero, take, where } from "./indexing.js";
export type { IndexSpec, SliceTuple } from "./indexing.js";
export { argmax, argmin, max, mean, min, prod, std, sum, variance } from "./reduce.js";
export type { ArgReduceOptions, ReduceOptions, VarOptions } from "./reduce.js";
export { dot, inner, linalg, matmul, outer } from "./linalg.js";
export type {
  EigResult,
  LstsqResult,
  NormOptions,
  NormOrder,
  QrMode,
  QrResult,
  SvdOptions,
  SvdResult,
} from "./linalg.js";
export { defaultRng, Generator, random, RandomState } from "./random.js";
export type { Seed, Size } from "./random.js";
export { fftModule as fft } from "./fft.js";
export type { FftNOptions, FftNorm, FftOptions } from "./fft.js";
export type { MethodReduceOptions } from "./ndarray.js";
/** Development instrumentation: live native buffers / bytes (PLAN §33). */
export function memoryStats(): { buffers: number; bytes: number } {
  return addon.memoryStats();
}

const lib = { stride_tricks: { asStrided } } as const;

const np = {
  NDArray,
  Complex,
  complex,
  DType: dtypes.DType,
  dtype: dtypes.dtype,
  promoteTypes: dtypes.promoteTypes,
  canCast: dtypes.canCast,
  array,
  asarray,
  empty,
  emptyLike,
  zeros,
  zerosLike,
  ones,
  onesLike,
  full,
  fullLike,
  arange,
  linspace,
  eye,
  identity,
  reshape,
  transpose,
  squeeze,
  expandDims,
  swapAxes,
  moveAxis,
  ravel,
  add,
  subtract,
  multiply,
  divide,
  power,
  mod,
  floorDivide,
  abs,
  negative,
  sqrt,
  exp,
  log,
  angle,
  conj,
  conjugate,
  imag,
  iscomplex,
  iscomplexobj,
  isreal,
  isrealobj,
  real,
  broadcastShapes,
  broadcastTo,
  nonzero,
  take,
  where,
  sum,
  prod,
  min,
  max,
  amin: min,
  amax: max,
  mean,
  var: variance,
  std,
  argmin,
  argmax,
  matmul,
  dot,
  inner,
  outer,
  linalg,
  random,
  fft: fftModule,
  newaxis,
  ellipsis,
  fromTypedArray,
  mayShareMemory,
  lib,
  memoryStats,
  bool: dtypes.bool,
  int8: dtypes.int8,
  uint8: dtypes.uint8,
  int16: dtypes.int16,
  uint16: dtypes.uint16,
  int32: dtypes.int32,
  uint32: dtypes.uint32,
  int64: dtypes.int64,
  uint64: dtypes.uint64,
  float16: dtypes.float16,
  float32: dtypes.float32,
  float64: dtypes.float64,
  complex64: dtypes.complex64,
  complex128: dtypes.complex128,
  ...errors,
} as const;

export { lib };
export default np;
