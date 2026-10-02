import { array } from "./creation.js";
import { dtype as toDType, type Casting, type DTypeLike } from "./dtype.js";
import { IndexError, ValueError } from "./errors.js";
import { NDArray, type NestedArray } from "./ndarray.js";

/** P6 shared helpers (D-090). Internal. */
export type ArrayLike = NDArray | NestedArray;

export const asArr = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));

export const wrap = (h: Parameters<typeof NDArray._wrap>[0]): NDArray => NDArray._wrap(h);

export const wrapList = (hs: Parameters<typeof NDArray._wrap>[0][]): NDArray[] => hs.map(wrap);

export const isOpts = (x: unknown): x is Record<string, unknown> =>
  typeof x === "object" && x !== null && !Array.isArray(x) && !(x instanceof NDArray);

export const dtypeName = (dt: DTypeLike | null | undefined): string | null =>
  dt === undefined || dt === null ? null : toDType(dt).name;

export function checkInt(v: unknown, what: string): number {
  if (typeof v !== "number" || !Number.isSafeInteger(v)) {
    throw new ValueError(`${what} must be an integer`);
  }
  return v;
}

export const axesList = (axis: number | readonly number[]): number[] =>
  typeof axis === "number" ? [axis] : [...axis];

export function normAxis(axis: number, ndim: number): number {
  checkInt(axis, "axis");
  if (axis < -ndim || axis >= ndim) {
    // NumPy AxisError (IndexError + ValueError); numera raises IndexError (D-012).
    throw new IndexError(`axis ${axis} is out of bounds for array of dimension ${ndim}`);
  }
  return axis < 0 ? axis + ndim : axis;
}

export interface JoinOptions {
  /** Result dtype (NumPy `dtype=`); mutually exclusive with `out`. */
  dtype?: DTypeLike | null;
  /** Casting rule for every input (default "same_kind"). */
  casting?: Casting | null;
  /** Destination array of the right shape (NumPy `out=`). */
  out?: NDArray | null;
}

export const joinArgs = (o: JoinOptions): [string | null, string | null, NativeHandle | null] => [
  dtypeName(o.dtype),
  o.casting ?? null,
  o.out ? o.out._native : null,
];

type NativeHandle = NDArray["_native"];
