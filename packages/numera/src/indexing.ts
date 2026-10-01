import { addon, nativeModule, type NativeNDArray } from "./addon.js";
import { array } from "./creation.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray } from "./ndarray.js";

/** Indexing free functions (PLAN §13, M6). Methods live on NDArray; see D-015. */
export { ellipsis, newaxis } from "./ndarray.js";
export type { IndexSpec, SliceTuple } from "./ndarray.js";

const toArray = (a: NDArray | NestedArray): NDArray => (a instanceof NDArray ? a : array(a));

/** NumPy nonzero: one int64 index array per dimension. */
export function nonzero(a: NDArray | NestedArray): NDArray[] {
  const x = toArray(a);
  return wrapNative(() => addon.nonzero(x._native).map((h) => NDArray._wrap(h)));
}

/** NumPy out-of-bounds index handling (`mode=`, D-110). */
export type ClipMode = "raise" | "wrap" | "clip";

/** Options for `take` (NumPy `axis=`, `mode=`). */
export interface TakeOptions {
  /** Axis to take along; omitted/`null` indexes the flattened array. */
  axis?: number | null;
  /** "raise" (default), "wrap" (modulo) or "clip" (to the valid range). */
  mode?: ClipMode;
}

interface P08TakeNative {
  take(a: NativeNDArray, indices: NativeNDArray, axis: number | null, mode?: string): NativeNDArray;
}

/**
 * NumPy take. Without `axis`, indexes the flattened array. `axis` may be given
 * positionally or in the options object together with `mode` (D-110).
 */
export function take(
  a: NDArray | NestedArray,
  indices: NDArray | NestedArray,
  axis?: number | null | TakeOptions,
  opts: TakeOptions = {},
): NDArray {
  const o: TakeOptions = typeof axis === "object" && axis !== null ? axis : { ...opts, axis };
  const x = toArray(a);
  const i = indices instanceof NDArray ? indices : array(indices, { dtype: "int64" });
  const native = nativeModule<P08TakeNative>("p08");
  return wrapNative(() =>
    NDArray._wrap(native.take(x._native, i._native, o.axis ?? null, o.mode ?? undefined)),
  );
}

/**
 * NumPy where. With only `condition`, equivalent to `nonzero(condition)`.
 * `x` and `y` must be given together.
 */
export function where(condition: NDArray | NestedArray): NDArray[];
export function where(
  condition: NDArray | NestedArray,
  x: NDArray | NestedArray,
  y: NDArray | NestedArray,
): NDArray;
export function where(
  condition: NDArray | NestedArray,
  x?: NDArray | NestedArray,
  y?: NDArray | NestedArray,
): NDArray | NDArray[] {
  if (x === undefined && y === undefined) return nonzero(condition);
  if (x === undefined || y === undefined) {
    throw new ValueError("either both or neither of x and y should be given");
  }
  const c = toArray(condition);
  const xs = toArray(x);
  const ys = toArray(y);
  return wrapNative(() => NDArray._wrap(addon.where(c._native, xs._native, ys._native)));
}
