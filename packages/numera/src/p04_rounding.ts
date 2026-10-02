import { array } from "./creation.js";
import { DTypeError } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { divide, multiply, toArray, unaryUfunc, type ArrayLike, type UnaryUfunc } from "./ufunc.js";

// P4-3: rounding ufuncs (native rows in ufunc_math.cpp) and np.round (D-071).

export const floor: UnaryUfunc = unaryUfunc("floor");
export const ceil: UnaryUfunc = unaryUfunc("ceil");
export const trunc: UnaryUfunc = unaryUfunc("trunc");
export const rint: UnaryUfunc = unaryUfunc("rint");
export const positive: UnaryUfunc = unaryUfunc("positive");

/** `np.round` / `np.around` / `a.round` options. */
export interface RoundOptions {
  /** Write the result into this array and return it. */
  out?: NDArray | null;
}

/**
 * NumPy round/around: round half to even to `decimals` places (negative
 * values round to tens, hundreds, ...). Floats compute `rint(x * 10**d) / 10**d`
 * like NumPy, so results carry the same floating-point error. Integer inputs
 * keep their dtype (negative `decimals` round via float64 and cast back).
 */
export function round(a: ArrayLike | number, decimals = 0, opts: RoundOptions = {}): NDArray {
  if (!Number.isInteger(decimals)) throw new DTypeError("decimals must be an integer");
  const x = toArray(a as ArrayLike);
  const out = opts.out ?? undefined;
  const kind = x.dtype.kind;
  if (kind === "b" || kind === "i" || kind === "u") {
    if (decimals >= 0) {
      if (kind === "b") {
        if (decimals !== 0) throw new DTypeError("Cannot cast ufunc 'multiply' output from float64 to bool");
        return rint(x, { out });
      }
      return positive(x, { out });
    }
    const f = array(10 ** -decimals, { dtype: "float64" });
    const y = multiply(rint(divide(x, f)), f);
    if (out) return positive(y, { out, casting: "unsafe" });
    return y.astype(x.dtype);
  }
  if (decimals === 0) return rint(x, { out });
  const f = 10 ** Math.abs(decimals);
  const scaled = decimals > 0 ? multiply(x, f) : divide(x, f);
  const r = rint(scaled);
  return decimals > 0 ? divide(r, f, { out }) : multiply(r, f, { out });
}

/** NumPy around (alias of round). */
export const around = round;

/** NumPy fix: round toward zero (same as trunc; integer dtypes are kept). */
export function fix(x: ArrayLike | number, opts: RoundOptions = {}): NDArray {
  return trunc(x as ArrayLike, { out: opts.out ?? undefined });
}

declare module "./ndarray.js" {
  interface NDArray {
    /** NumPy `a.round(decimals)`: see `np.round`. */
    round(decimals?: number, opts?: RoundOptions): NDArray;
  }
}

NDArray.prototype.round = function (this: NDArray, decimals = 0, opts: RoundOptions = {}): NDArray {
  return round(this, decimals, opts);
};
