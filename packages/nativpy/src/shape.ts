import { addon } from "./addon.js";
import { NDArray, type Shape } from "./ndarray.js";
import { wrapNative } from "./errors.js";

/** Shape functions (PLAN §12, M3). Views unless noted. */

const axesOf = (axis: number | readonly number[]): number[] =>
  typeof axis === "number" ? [axis] : [...axis];

export function reshape(a: NDArray, shape: Shape | number): NDArray {
  return a.reshape(typeof shape === "number" ? [shape] : shape);
}

export function transpose(a: NDArray, axes?: readonly number[]): NDArray {
  return a.transpose(axes ?? []);
}

export function squeeze(a: NDArray, axis?: number | readonly number[]): NDArray {
  return a.squeeze(axis);
}

export function expandDims(a: NDArray, axis: number | readonly number[]): NDArray {
  return wrapNative(() => NDArray._wrap(addon.expandDims(a._native, axesOf(axis))));
}

export function swapAxes(a: NDArray, axis1: number, axis2: number): NDArray {
  return a.swapAxes(axis1, axis2);
}

export function moveAxis(
  a: NDArray,
  source: number | readonly number[],
  destination: number | readonly number[],
): NDArray {
  return wrapNative(() =>
    NDArray._wrap(addon.moveaxis(a._native, axesOf(source), axesOf(destination))),
  );
}

export function ravel(a: NDArray): NDArray {
  return a.ravel();
}
