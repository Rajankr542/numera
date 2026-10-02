import { DTypeError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { native } from "./p06_native.js";
import { asArr, checkInt, wrap, type ArrayLike } from "./p06_util.js";

/** tile / repeat (P6, D-091). */

type IntList = number | readonly number[] | NDArray;

function intList(v: IntList, what: string): number[] {
  const flat: unknown[] =
    v instanceof NDArray ? (v.ndim === 0 ? [v.item()] : (v.ravel().toArray() as unknown[])) : typeof v === "number" ? [v] : [...v];
  if (v instanceof NDArray && v.dtype.kind !== "i" && v.dtype.kind !== "u" && v.dtype.kind !== "b") {
    throw new DTypeError(`Cannot cast ${what} from dtype('${v.dtype.name}') to dtype('int64') according to the rule 'safe'`);
  }
  return flat.map((x) => {
    const n = typeof x === "boolean" ? Number(x) : x;
    if (typeof n !== "number" || !Number.isSafeInteger(n)) {
      throw new DTypeError(`${what} must be integers`);
    }
    return n;
  });
}

/** NumPy tile: repeats the whole array `reps` times along each axis. */
export function tile(a: ArrayLike, reps: IntList): NDArray {
  const r = intList(reps, "reps");
  return wrapNative(() => wrap(native.tile(asArr(a)._native, r)));
}

/** NumPy repeat: repeats each element; flattens first when `axis` is omitted/null. */
export function repeat(a: ArrayLike, repeats: IntList, axis: number | null = null): NDArray {
  const r = intList(repeats, "repeats");
  if (axis !== null) checkInt(axis, "axis");
  return wrapNative(() => wrap(native.repeat(asArr(a)._native, r, axis)));
}

declare module "./ndarray.js" {
  interface NDArray {
    /** NumPy ndarray.repeat (same as `np.repeat(this, repeats, axis)`). */
    repeat(repeats: number | readonly number[] | NDArray, axis?: number | null): NDArray;
  }
}

NDArray.prototype.repeat = function (this: NDArray, repeats: IntList, axis: number | null = null): NDArray {
  return repeat(this, repeats, axis);
};
