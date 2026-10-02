import { array } from "./creation.js";
import { DTypeError, ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray } from "./ndarray.js";
import { native } from "./p06_native.js";
import { asArr, wrap, type ArrayLike } from "./p06_util.js";
import { moveAxis } from "./shape.js";
import { broadcastTo } from "./ufunc.js";

/** pad (P6, D-092). */

export type PadMode =
  | "constant"
  | "edge"
  | "linear_ramp"
  | "maximum"
  | "mean"
  | "median"
  | "minimum"
  | "reflect"
  | "symmetric"
  | "wrap"
  | "empty";

/** Pad widths: `n`, `[n]`, `[before, after]`, `[[b0, a0], [b1, a1], ...]`, an array, or `{axis: n | [b, a]}`. */
export type PadWidth = number | NestedArray | NDArray | Record<number, number | readonly [number, number]>;

type PairValues = number | NestedArray | NDArray;

export interface PadOptions {
  /** mode "constant": value(s) for the padded areas (default 0), in NumPy pair forms. */
  constantValues?: PairValues;
  /** mode "linear_ramp": ramp end values (default 0). */
  endValues?: PairValues;
  /** maximum/mean/median/minimum: number of edge values used (default whole axis). */
  statLength?: PairValues | null;
  /** reflect/symmetric: "even" (default) or "odd" (2 * edge - reflected). */
  reflectType?: "even" | "odd";
}

/** Callable mode, NumPy-style: fills `vector[:before]` / `vector[-after:]` in place. */
export type PadFunction = (
  vector: NDArray,
  iaxisPadWidth: [number, number],
  iaxis: number,
  kwargs: Record<string, unknown>,
) => void;

const ALLOWED: Record<PadMode, readonly string[]> = {
  empty: [],
  edge: [],
  wrap: [],
  constant: ["constantValues"],
  linear_ramp: ["endValues"],
  maximum: ["statLength"],
  mean: ["statLength"],
  median: ["statLength"],
  minimum: ["statLength"],
  reflect: ["reflectType"],
  symmetric: ["reflectType"],
};

const roundHalfEven = (x: number): number => {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
};

// NumPy _as_pairs: broadcasts x to (ndim, 2), returned as an NDArray.
function asPairs(x: PairValues, ndim: number): NDArray {
  const a = x instanceof NDArray ? x : array(x as NestedArray);
  if (a.ndim < 3) {
    if (a.size === 1) return broadcastTo(a.reshape(1), [ndim, 2]);
    if (a.size === 2 && !(a.ndim === 2 && a.shape[0] === 2 && a.shape[1] === 1)) {
      return broadcastTo(a.reshape(1, 2), [ndim, 2]);
    }
  }
  return broadcastTo(a, [ndim, 2]);
}

function indexPairs(x: PairValues, ndim: number, what: string, integral: boolean): number[] {
  const a = x instanceof NDArray ? x : array(x as NestedArray);
  if (integral && a.dtype.kind !== "i" && a.dtype.kind !== "u") {
    if (!(a.dtype.kind === "f" && (a.toArray() as unknown as number[]).length === 0)) {
      throw new DTypeError(`\`${what}\` must be of integral type.`);
    }
  }
  const flat = (asPairs(a, ndim).toArray() as number[][]).flat().map((v) => roundHalfEven(Number(v)));
  if (flat.some((v) => v < 0)) throw new ValueError("index can't contain negative values");
  return flat;
}

function widthList(padWidth: PadWidth, ndim: number): number[] {
  if (typeof padWidth === "object" && padWidth !== null && !Array.isArray(padWidth) && !(padWidth instanceof NDArray)) {
    const seq: number[] = new Array<number>(2 * ndim).fill(0);
    for (const [k, w] of Object.entries(padWidth as Record<string, unknown>)) {
      let axis = Number(k);
      if (!Number.isInteger(axis) || axis < -ndim || axis >= ndim) {
        throw new ValueError(`pad_width key ${k} is not a valid axis for an array of dimension ${ndim}`);
      }
      if (axis < 0) axis += ndim;
      const pair = typeof w === "number" ? [w, w] : Array.isArray(w) && w.length === 2 ? w : null;
      if (pair === null || !pair.every((v) => Number.isSafeInteger(v))) {
        throw new DTypeError("`pad_width` must be of integral type.");
      }
      if (pair.some((v: number) => v < 0)) throw new ValueError("index can't contain negative values");
      seq[2 * axis] = pair[0] as number;
      seq[2 * axis + 1] = pair[1] as number;
    }
    return seq;
  }
  if (typeof padWidth === "number" && !Number.isSafeInteger(padWidth)) {
    throw new DTypeError("`pad_width` must be of integral type.");
  }
  return indexPairs(padWidth as PairValues, ndim, "pad_width", true);
}

/**
 * NumPy pad. `mode` is a mode name (default "constant") or a JS function
 * called NumPy-style on each 1-d lane of a zero-padded result.
 */
export function pad(
  a: ArrayLike,
  padWidth: PadWidth,
  mode: PadMode | PadFunction = "constant",
  options: PadOptions & Record<string, unknown> = {},
): NDArray {
  const x = asArr(a);
  const nd = x.ndim;
  const width = widthList(padWidth, nd);
  if (typeof mode === "function") {
    const padded = wrapNative(() => wrap(native.pad(x._native, "constant", width, null, null, false)));
    for (let axis = 0; axis < nd; axis++) {
      const view = moveAxis(padded, axis, -1);
      const lead = view.shape.slice(0, -1);
      const count = lead.reduce((p, q) => p * q, 1);
      const idx = new Array<number>(lead.length).fill(0);
      for (let k = 0; k < count; k++) {
        mode(view.get(...idx), [width[2 * axis]!, width[2 * axis + 1]!], axis, options);
        for (let d = lead.length - 1; d >= 0; d--) {
          if (++idx[d]! < lead[d]!) break;
          idx[d] = 0;
        }
      }
    }
    return padded;
  }
  const allowed = ALLOWED[mode as PadMode];
  if (allowed === undefined) throw new ValueError(`mode '${String(mode)}' is not supported`);
  const bad = Object.keys(options).filter((k) => options[k] !== undefined && !allowed.includes(k));
  if (bad.length > 0) {
    throw new ValueError(`unsupported keyword arguments for mode '${mode}': {${bad.map((k) => `'${k}'`).join(", ")}}`);
  }
  let values: NDArray | null = null;
  const v = mode === "constant" ? options.constantValues : mode === "linear_ramp" ? options.endValues : undefined;
  if (v !== undefined && nd > 0) values = asPairs(v, nd).copy();
  let stat: number[] | null = null;
  if (ALLOWED[mode as PadMode].includes("statLength")) {
    stat =
      options.statLength === undefined || options.statLength === null
        ? new Array<number>(2 * nd).fill(-1)
        : indexPairs(options.statLength, nd, "stat_length", false);
  }
  const odd = options.reflectType === "odd";
  return wrapNative(() =>
    wrap(native.pad(x._native, mode, width, values ? values._native : null, stat, odd)),
  );
}
