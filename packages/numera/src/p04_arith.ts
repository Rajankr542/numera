import { NDArray } from "./ndarray.js";
import {
  abs,
  binaryUfunc,
  conjugate,
  divide,
  mod,
  power,
  toArray,
  unaryUfunc,
  type ArrayLike,
  type BinaryUfunc,
  type Operand,
  type UnaryUfunc,
} from "./ufunc.js";
import { positive } from "./p04_rounding.js";

// P4-4: arithmetic ufuncs (native rows in ufunc_math.cpp), clip (D-072).

export const fmod: BinaryUfunc = binaryUfunc("fmod");
export const floatPower: BinaryUfunc = binaryUfunc("float_power");
export const sign: UnaryUfunc = unaryUfunc("sign");
export const heaviside: BinaryUfunc = binaryUfunc("heaviside");
export const maximum: BinaryUfunc = binaryUfunc("maximum");
export const minimum: BinaryUfunc = binaryUfunc("minimum");
export const fmax: BinaryUfunc = binaryUfunc("fmax");
export const fmin: BinaryUfunc = binaryUfunc("fmin");
export const fabs: UnaryUfunc = unaryUfunc("fabs");

// NumPy aliases of existing ufuncs (same objects).
export const remainder: BinaryUfunc = mod;
export const trueDivide: BinaryUfunc = divide;
export const pow: BinaryUfunc = power;
export const absolute: UnaryUfunc = abs;

/** `np.clip` / `a.clip` options. */
export interface ClipOptions {
  /** Write the result into this array and return it. */
  out?: NDArray | null;
}

type Bound = Operand | null | undefined;

/**
 * NumPy clip: `minimum(maximum(a, min), max)`, so NaN bounds or elements
 * propagate and `min > max` gives `max` everywhere. Either bound may be
 * `null`/`undefined` (no clipping on that side); bounds broadcast and promote
 * like ufunc operands (JS scalars are weak).
 */
export function clip(a: ArrayLike, min?: Bound, max?: Bound, opts: ClipOptions = {}): NDArray {
  const x = toArray(a);
  const out = opts.out ?? undefined;
  const hasMin = min !== null && min !== undefined;
  const hasMax = max !== null && max !== undefined;
  if (!hasMin && !hasMax) return positive(x, { out });
  if (!hasMax) return maximum(x, min as Operand, { out });
  if (!hasMin) return minimum(x, max as Operand, { out });
  return minimum(maximum(x, min as Operand), max as Operand, { out });
}

declare module "./ndarray.js" {
  interface NDArray {
    /** NumPy `a.clip(min, max)`: see `np.clip`. */
    clip(min?: Operand | null, max?: Operand | null, opts?: ClipOptions): NDArray;
    /** NumPy `a.conjugate()`: complex conjugate (a copy for real dtypes). */
    conjugate(): NDArray;
  }
}

NDArray.prototype.clip = function (this: NDArray, min?: Bound, max?: Bound, opts: ClipOptions = {}): NDArray {
  return clip(this, min, max, opts);
};
NDArray.prototype.conjugate = function (this: NDArray): NDArray {
  return conjugate(this);
};
