// P5 isclose / allclose / arrayEqual / arrayEquiv (D-083).
import { nativeModule, type NativeNDArray } from "./addon.js";
import { isComplexLike, type ComplexLike } from "./complex.js";
import { array, asarray } from "./creation.js";
import { FloatingPointError, wrapNative } from "./errors.js";
import { geterr } from "./errstate.js";
import { NDArray } from "./ndarray.js";
import { all, equal, logicalAnd, logicalOr } from "./p05_compare.js";
import { isnan } from "./p05_classify.js";
import { broadcastShapes, type ArrayLike } from "./ufunc.js";

interface P05Native {
  isclose(a: NativeNDArray, b: NativeNDArray, rtol: number, atol: number, equalNan: boolean): NativeNDArray;
}
const native = nativeModule<P05Native>("p05");

export interface IscloseOptions {
  /** Relative tolerance (default 1e-5). */
  rtol?: number;
  /** Absolute tolerance (default 1e-8). */
  atol?: number;
  /** Treat NaNs in the same position as equal (default false). */
  equalNan?: boolean;
}

type CloseOperand = ArrayLike | ComplexLike;

// JS scalars are weak (NEP 50): they take an inexact partner array's dtype.
function operands(a: CloseOperand, b: CloseOperand): [NDArray, NDArray] {
  const conv = (v: CloseOperand, other: CloseOperand): NDArray => {
    if (v instanceof NDArray) return v;
    const scalar = typeof v === "number" || typeof v === "bigint" || typeof v === "boolean" || isComplexLike(v);
    if (scalar && other instanceof NDArray) {
      const k = other.dtype.kind;
      if (k === "c" || (k === "f" && !isComplexLike(v))) return array(v, { dtype: other.dtype });
    }
    return asarray(v);
  };
  return [conv(a, b), conv(b, a)];
}

function checkTolerances(rtol: number, atol: number): void {
  if (Number.isFinite(rtol) && Number.isFinite(atol)) return;
  const msg = `One of rtol or atol is not valid, atol: ${atol}, rtol: ${rtol}`;
  const mode = geterr().invalid;
  if (mode === "raise") throw new FloatingPointError(msg);
  if (mode === "warn") process.emitWarning(msg, "RuntimeWarning");
  else if (mode === "print") process.stdout.write(`Warning: ${msg}\n`);
}

/**
 * NumPy isclose: element-wise `|a - b| <= atol + rtol * |b|` (b finite) or
 * `a == b`; `equalNan` also matches NaN with NaN. Bool result (0-d for scalars).
 */
export function isclose(a: CloseOperand, b: CloseOperand, opts: IscloseOptions = {}): NDArray {
  const { rtol = 1e-5, atol = 1e-8, equalNan = false } = opts;
  checkTolerances(rtol, atol);
  const [x, y] = operands(a, b);
  return wrapNative(() => NDArray._wrap(native.isclose(x._native, y._native, rtol, atol, equalNan)));
}

/** NumPy allclose: true if `isclose(a, b, opts)` holds everywhere. */
export function allclose(a: CloseOperand, b: CloseOperand, opts: IscloseOptions = {}): boolean {
  return all(isclose(a, b, opts)).item() as boolean;
}

function tryArray(v: ArrayLike): NDArray | null {
  try {
    return asarray(v);
  } catch {
    return null;
  }
}

/**
 * NumPy array_equal: true if the arrays have the same shape and elements.
 * `equalNan` treats NaNs in the same position as equal.
 */
export function arrayEqual(a1: ArrayLike, a2: ArrayLike, opts: { equalNan?: boolean } = {}): boolean {
  const x = tryArray(a1);
  const y = tryArray(a2);
  if (x === null || y === null) return false;
  if (x.ndim !== y.ndim || x.shape.some((d, i) => d !== y.shape[i])) return false;
  if (!opts.equalNan) return all(equal(x, y)).item() as boolean;
  const nx = isnan(x);
  const ny = isnan(y);
  if (!(all(equal(nx, ny)).item() as boolean)) return false;
  // NaNs line up; every other element must compare equal.
  return all(logicalOr(equal(x, y), logicalAnd(nx, ny))).item() as boolean;
}

/**
 * NumPy array_equiv: true if the arrays are shape-consistent (broadcastable)
 * and all elements are equal.
 */
export function arrayEquiv(a1: ArrayLike, a2: ArrayLike): boolean {
  const x = tryArray(a1);
  const y = tryArray(a2);
  if (x === null || y === null) return false;
  try {
    broadcastShapes(x.shape, y.shape);
  } catch {
    return false;
  }
  return all(equal(x, y)).item() as boolean;
}
