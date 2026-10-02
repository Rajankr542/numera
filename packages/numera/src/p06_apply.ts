import { asarray, zeros } from "./creation.js";
import { ValueError } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { asArr, normAxis, type ArrayLike } from "./p06_util.js";
import { expandDims } from "./shape.js";

/** applyAlongAxis / applyOverAxes with JS callbacks (P6, D-090). */

type Result = ArrayLike;

/**
 * NumPy apply_along_axis: calls `func1d(lane, ...args)` on every 1-d lane along
 * `axis`; the results (all of the first result's shape, cast to its dtype)
 * replace that axis.
 */
export function applyAlongAxis<A extends unknown[]>(
  func1d: (lane: NDArray, ...args: A) => Result,
  axis: number,
  arr: ArrayLike,
  ...args: A
): NDArray {
  const a = asArr(arr);
  const nd = a.ndim;
  const ax = normAxis(axis, nd);
  const dims = [...Array(nd).keys()];
  const view = a.transpose([...dims.slice(0, ax), ...dims.slice(ax + 1), ax]);
  const lead = view.shape.slice(0, -1);
  const count = lead.reduce((p, q) => p * q, 1);
  if (count === 0) throw new ValueError("Cannot apply_along_axis when any iteration dimensions are 0");
  const idx = new Array<number>(lead.length).fill(0);
  const res0 = asarray(func1d(view.get(...idx), ...args) as ArrayLike);
  const buff = zeros([...lead, ...res0.shape], { dtype: res0.dtype });
  const put = (r: NDArray) => {
    if (lead.length === 0) buff.set([], r);
    else buff.set([...idx], r);
  };
  put(res0);
  for (let k = 1; k < count; k++) {
    for (let d = lead.length - 1; d >= 0; d--) {
      if (++idx[d]! < lead[d]!) break;
      idx[d] = 0;
    }
    put(asarray(func1d(view.get(...idx), ...args) as ArrayLike));
  }
  const bd = [...Array(buff.ndim).keys()];
  const rn = res0.ndim;
  return buff.transpose([...bd.slice(0, ax), ...bd.slice(buff.ndim - rn), ...bd.slice(ax, buff.ndim - rn)]);
}

/**
 * NumPy apply_over_axes: repeatedly `val = func(val, axis)` for each axis; a
 * result with one dimension fewer gets the axis re-inserted (keepdims).
 */
export function applyOverAxes(
  func: (a: NDArray, axis: number) => Result,
  a: ArrayLike,
  axes: number | readonly number[],
): NDArray {
  let val = asArr(a);
  const n = val.ndim;
  for (let axis of typeof axes === "number" ? [axes] : axes) {
    if (axis < 0) axis += n;
    let res = asarray(func(val, axis) as ArrayLike);
    if (res.ndim !== val.ndim) {
      res = expandDims(res, axis);
      if (res.ndim !== val.ndim) throw new ValueError("function is not returning an array of the correct shape");
    }
    val = res;
  }
  return val;
}
