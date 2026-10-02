import { nativeModule, type NativeNDArray } from "./addon.js";
import { dtype, type Casting, type DTypeLike } from "./dtype.js";
import { DTypeError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import {
  binaryUfunc,
  operands,
  toArray,
  unaryUfunc,
  type ArrayLike,
  type BinaryUfunc,
  type Operand,
  type UnaryUfunc,
} from "./ufunc.js";

// P4-5: float-bit and integer ufuncs (rows in ufunc_math.cpp) and the
// multi-output ufuncs divmod / modf / frexp (native/core/p04_multi.cpp, D-073).

export const copysign: BinaryUfunc = binaryUfunc("copysign");
export const nextafter: BinaryUfunc = binaryUfunc("nextafter");
export const spacing: UnaryUfunc = unaryUfunc("spacing");
export const ldexp: BinaryUfunc = binaryUfunc("ldexp");
export const signbit: UnaryUfunc = unaryUfunc("signbit");
export const gcd: BinaryUfunc = binaryUfunc("gcd");
export const lcm: BinaryUfunc = binaryUfunc("lcm");

interface P04Native {
  multi(
    op: string,
    a: NativeNDArray,
    b: NativeNDArray | null,
    params: { dtype?: string; casting?: Casting },
    out0: NativeNDArray | null,
    out1: NativeNDArray | null,
  ): [NativeNDArray, NativeNDArray];
}
const native = nativeModule<P04Native>("p04");

/** Options of the two-output ufuncs `divmod`, `modf`, `frexp`. */
export interface MultiUfuncOptions {
  /** Two output arrays (either may be `null`), like NumPy's `out=(o1, o2)`. */
  out?: readonly [NDArray | null, NDArray | null] | null;
  /** Loop dtype of the first output. */
  dtype?: DTypeLike | null;
  casting?: Casting;
}

function multi(op: string, a: NDArray, b: NDArray | null, opts: MultiUfuncOptions): [NDArray, NDArray] {
  const out = opts.out ?? [null, null];
  if (!Array.isArray(out) || out.length !== 2) {
    throw new DTypeError(`${op}: out must be a pair of arrays (or null entries)`);
  }
  for (const o of out) {
    if (o !== null && o !== undefined && !(o instanceof NDArray)) {
      throw new DTypeError("return arrays must be of ArrayType");
    }
  }
  const params: { dtype?: string; casting?: Casting } = {};
  if (opts.dtype !== undefined && opts.dtype !== null) params.dtype = dtype(opts.dtype).name;
  if (opts.casting !== undefined) params.casting = opts.casting;
  const [o0, o1] = out as [NDArray | null, NDArray | null];
  const r = wrapNative(() => native.multi(op, a._native, b?._native ?? null, params, o0?._native ?? null, o1?._native ?? null));
  return [o0 ?? NDArray._wrap(r[0]), o1 ?? NDArray._wrap(r[1])];
}

/** NumPy divmod: `[floorDivide(a, b), mod(a, b)]` in one pass. */
export function divmod(a: Operand, b: Operand, opts: MultiUfuncOptions = {}): [NDArray, NDArray] {
  const loop = opts.dtype === undefined || opts.dtype === null ? undefined : dtype(opts.dtype);
  const [x, y] = operands(a, b, loop);
  return multi("divmod", x, y, opts);
}

/** NumPy modf: `[fractional, integral]` parts, both with the sign of `x`. */
export function modf(x: ArrayLike | number, opts: MultiUfuncOptions = {}): [NDArray, NDArray] {
  return multi("modf", toArray(x as ArrayLike), null, opts);
}

/** NumPy frexp: `[mantissa, exponent]` with `x = m * 2**e`, `0.5 <= |m| < 1`; exponent is int32. */
export function frexp(x: ArrayLike | number, opts: MultiUfuncOptions = {}): [NDArray, NDArray] {
  return multi("frexp", toArray(x as ArrayLike), null, opts);
}
