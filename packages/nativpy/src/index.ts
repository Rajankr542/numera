import { addon } from "./addon.js";
import * as dtypes from "./dtype.js";
import * as errors from "./errors.js";
import { array, asStrided, empty, fromTypedArray, mayShareMemory, zeros } from "./creation.js";
import { NDArray } from "./ndarray.js";

export { DType, dtype, promoteTypes } from "./dtype.js";
export type { DTypeLike, DTypeName } from "./dtype.js";
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
export type { ArrayFlags, NestedArray, Shape } from "./ndarray.js";
export { array, asStrided, empty, fromTypedArray, inferDType, mayShareMemory, zeros } from "./creation.js";
export type { ArrayOptions } from "./creation.js";

/** Development instrumentation: live native buffers / bytes (PLAN §33). */
export function memoryStats(): { buffers: number; bytes: number } {
  return addon.memoryStats();
}

const lib = { stride_tricks: { asStrided } } as const;

const np = {
  NDArray,
  DType: dtypes.DType,
  dtype: dtypes.dtype,
  promoteTypes: dtypes.promoteTypes,
  array,
  empty,
  zeros,
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
