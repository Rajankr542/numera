import { ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { native } from "./p06_native.js";
import { asArr, axesList, checkInt, normAxis, wrap, type ArrayLike } from "./p06_util.js";

/** flip / roll / rot90 / axis permutations (P6, D-090). All but roll return views. */

type Axes = number | readonly number[];

/** NumPy flip: reverses the order of elements along `axis` (all axes when omitted). A view. */
export function flip(m: ArrayLike, axis: Axes | null = null): NDArray {
  const a = asArr(m);
  const axes = axis === null ? null : axesList(axis).map((ax) => checkInt(ax, "axis"));
  return wrapNative(() => wrap(native.flip(a._native, axes)));
}

/** NumPy fliplr: reverses axis 1 (left/right). */
export function fliplr(m: ArrayLike): NDArray {
  const a = asArr(m);
  if (a.ndim < 2) throw new ValueError("Input must be >= 2-d.");
  return flip(a, 1);
}

/** NumPy flipud: reverses axis 0 (up/down). */
export function flipud(m: ArrayLike): NDArray {
  const a = asArr(m);
  if (a.ndim < 1) throw new ValueError("Input must be >= 1-d.");
  return flip(a, 0);
}

/**
 * NumPy roll: shifts elements cyclically along `axis` (the flattened array when
 * omitted). `shift` and `axis` broadcast against each other. Returns a copy.
 */
export function roll(a: ArrayLike, shift: Axes, axis: Axes | null = null): NDArray {
  const x = asArr(a);
  if (axis === null) {
    return roll(x.ravel(), shift, 0).reshape(x.shape);
  }
  const sh = axesList(shift).map((s) => checkInt(s, "shift"));
  const ax = axesList(axis).map((d) => normAxis(d, x.ndim));
  if (sh.length !== ax.length && sh.length !== 1 && ax.length !== 1) {
    throw new ValueError(
      "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between " +
        `arg 0 with shape (${sh.length},) and arg 1 with shape (${ax.length},).`,
    );
  }
  const n = Math.max(sh.length, ax.length);
  const shifts = new Array<number>(x.ndim).fill(0);
  for (let k = 0; k < n; k++) {
    const d = ax[ax.length === 1 ? 0 : k]!;
    shifts[d] = shifts[d]! + sh[sh.length === 1 ? 0 : k]!;
  }
  return wrapNative(() => wrap(native.roll(x._native, shifts)));
}

/** NumPy rollaxis: moves `axis` to position `start` (prefer moveAxis). A view. */
export function rollaxis(a: ArrayLike, axis: number, start = 0): NDArray {
  const x = asArr(a);
  const n = x.ndim;
  const ax = normAxis(axis, n);
  checkInt(start, "start");
  let st = start < 0 ? start + n : start;
  if (!(st >= 0 && st < n + 1)) {
    throw new ValueError(`'start' arg requires ${-n} <= start < ${n + 1}, but ${start} was passed in`);
  }
  if (ax < st) st -= 1;
  if (ax === st) return x.get("...");
  const axes = [...Array(n).keys()].filter((d) => d !== ax);
  axes.splice(st, 0, ax);
  return x.transpose(axes);
}

/** NumPy rot90: rotates by 90° `k` times in the plane of `axes` (from the first towards the second). A view. */
export function rot90(m: ArrayLike, k = 1, axes: readonly number[] = [0, 1]): NDArray {
  if (axes.length !== 2) throw new ValueError("len(axes) must be 2.");
  const a = asArr(m);
  const [a0, a1] = axes as [number, number];
  checkInt(a0, "axes");
  checkInt(a1, "axes");
  if (a0 === a1 || Math.abs(a0 - a1) === a.ndim) throw new ValueError("Axes must be different.");
  if (a0 >= a.ndim || a0 < -a.ndim || a1 >= a.ndim || a1 < -a.ndim) {
    throw new ValueError(`Axes=(${a0}, ${a1}) out of range for array of ndim=${a.ndim}.`);
  }
  checkInt(k, "k");
  const kk = ((k % 4) + 4) % 4;
  if (kk === 0) return a.get("...");
  if (kk === 2) return flip(flip(a, a0), a1);
  const perm = [...Array(a.ndim).keys()];
  const p0 = a0 < 0 ? a0 + a.ndim : a0;
  const p1 = a1 < 0 ? a1 + a.ndim : a1;
  [perm[p0], perm[p1]] = [perm[p1]!, perm[p0]!];
  return kk === 1 ? flip(a, a1).transpose(perm) : flip(a.transpose(perm), a1);
}

/** NumPy permute_dims (array API transpose with required axes). A view. */
export function permuteDims(a: ArrayLike, axes?: readonly number[]): NDArray {
  const x = asArr(a);
  return axes === undefined ? x.transpose() : x.transpose([...axes]);
}

/** NumPy matrix_transpose: swaps the last two axes. A view. */
export function matrixTranspose(x: ArrayLike): NDArray {
  const a = asArr(x);
  if (a.ndim < 2) {
    throw new ValueError(`Input array must be at least 2-dimensional, but it is ${a.ndim}`);
  }
  return a.swapAxes(-1, -2);
}
