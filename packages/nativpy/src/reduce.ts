import { addon } from "./addon.js";
import { dtype as toDType, type DTypeLike } from "./dtype.js";
import { wrapNative } from "./errors.js";
import { array } from "./creation.js";
import { NDArray } from "./ndarray.js";
import type { ArrayLike } from "./ufunc.js";

/**
 * Reductions (PLAN §17, M7). Semantics: DECISIONS D-017. Results are always
 * NDArrays (0-d when every axis is reduced); use `.item()` for a scalar.
 */
export interface ReduceOptions {
  /** Axis or axes to reduce; `null`/omitted reduces all axes. */
  axis?: number | readonly number[] | null;
  keepdims?: boolean;
  /** Accumulator/result dtype (sum, prod, mean, var, std). */
  dtype?: DTypeLike | null;
  /** Starting value (sum, prod, min, max). */
  initial?: number | null;
}

export interface VarOptions extends Omit<ReduceOptions, "initial"> {
  /** Delta degrees of freedom: divisor is N - ddof. */
  ddof?: number;
}

export interface ArgReduceOptions {
  axis?: number | null;
  keepdims?: boolean;
}

const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));

function run(op: string, a: ArrayLike, opts: ReduceOptions & { ddof?: number }): NDArray {
  const x = toArray(a);
  const axis = opts.axis ?? null;
  return wrapNative(() =>
    NDArray._wrap(
      addon.reduce(op, x._native, {
        axis: axis === null ? null : typeof axis === "number" ? [axis] : [...axis],
        keepdims: opts.keepdims ?? false,
        dtype: opts.dtype == null ? null : toDType(opts.dtype).name,
        initial: opts.initial ?? null,
        ddof: opts.ddof ?? 0,
      }),
    ),
  );
}

export const sum = (a: ArrayLike, opts: ReduceOptions = {}): NDArray => run("sum", a, opts);
export const prod = (a: ArrayLike, opts: ReduceOptions = {}): NDArray => run("prod", a, opts);
export const min = (a: ArrayLike, opts: Omit<ReduceOptions, "dtype"> = {}): NDArray =>
  run("min", a, opts);
export const max = (a: ArrayLike, opts: Omit<ReduceOptions, "dtype"> = {}): NDArray =>
  run("max", a, opts);
export const mean = (a: ArrayLike, opts: Omit<ReduceOptions, "initial"> = {}): NDArray =>
  run("mean", a, opts);
export const variance = (a: ArrayLike, opts: VarOptions = {}): NDArray => run("var", a, opts);
export const std = (a: ArrayLike, opts: VarOptions = {}): NDArray => run("std", a, opts);

function argReduce(isMax: boolean, a: ArrayLike, opts: ArgReduceOptions): NDArray {
  const x = toArray(a);
  return wrapNative(() =>
    NDArray._wrap(addon.argReduce(isMax, x._native, opts.axis ?? null, opts.keepdims ?? false)),
  );
}

export const argmin = (a: ArrayLike, opts: ArgReduceOptions = {}): NDArray =>
  argReduce(false, a, opts);
export const argmax = (a: ArrayLike, opts: ArgReduceOptions = {}): NDArray =>
  argReduce(true, a, opts);
