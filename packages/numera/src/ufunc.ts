import { addon } from "./addon.js";
import { wrapNative } from "./errors.js";
import { array } from "./creation.js";
import { NDArray, type NestedArray, type Shape } from "./ndarray.js";

/**
 * Element-wise arithmetic (PLAN §15, M4) with NumPy broadcasting (M5).
 * Semantics and divergences: DECISIONS D-014.
 */
export type ArrayLike = NDArray | NestedArray;
type Scalar = number | boolean | bigint;

const isScalar = (x: ArrayLike): x is Scalar =>
  typeof x === "number" || typeof x === "boolean" || typeof x === "bigint";

// NumPy 2 weak scalar promotion (NEP 50), adapted to JS numbers (D-014):
// integer-valued numbers / bigints behave like Python int, other numbers like
// Python float, booleans like Python bool.
function scalarFor(value: Scalar, like: NDArray): NDArray {
  const kind = like.dtype.kind;
  if (typeof value === "boolean") return array(value, { dtype: like.dtype });
  const isInt = typeof value === "bigint" || Number.isInteger(value);
  if (isInt) {
    if (kind === "b") return array(value, { dtype: "int64" });
    return array(value, { dtype: like.dtype });
  }
  if (kind === "f" || kind === "c") return array(value, { dtype: like.dtype });
  return array(value, { dtype: "float64" });
}

function operands(a: ArrayLike, b: ArrayLike): [NDArray, NDArray] {
  const aArr = a instanceof NDArray ? a : isScalar(a) ? undefined : array(a);
  const bArr = b instanceof NDArray ? b : isScalar(b) ? undefined : array(b);
  if (aArr && bArr) return [aArr, bArr];
  if (aArr) return [aArr, scalarFor(b as Scalar, aArr)];
  if (bArr) return [scalarFor(a as Scalar, bArr), bArr];
  return [array(a), array(b)];
}

const toArray = (a: ArrayLike): NDArray => (a instanceof NDArray ? a : array(a));

function binary(op: string, a: ArrayLike, b: ArrayLike): NDArray {
  const [x, y] = operands(a, b);
  return wrapNative(() => NDArray._wrap(addon.binary(op, x._native, y._native)));
}

function unary(op: string, a: ArrayLike): NDArray {
  const x = toArray(a);
  return wrapNative(() => NDArray._wrap(addon.unary(op, x._native)));
}

export const add = (a: ArrayLike, b: ArrayLike): NDArray => binary("add", a, b);
export const subtract = (a: ArrayLike, b: ArrayLike): NDArray => binary("subtract", a, b);
export const multiply = (a: ArrayLike, b: ArrayLike): NDArray => binary("multiply", a, b);
export const divide = (a: ArrayLike, b: ArrayLike): NDArray => binary("divide", a, b);
export const power = (a: ArrayLike, b: ArrayLike): NDArray => binary("power", a, b);
export const mod = (a: ArrayLike, b: ArrayLike): NDArray => binary("mod", a, b);
export const floorDivide = (a: ArrayLike, b: ArrayLike): NDArray => binary("floorDivide", a, b);

export const abs = (a: ArrayLike): NDArray => unary("abs", a);
export const negative = (a: ArrayLike): NDArray => unary("negative", a);
export const sqrt = (a: ArrayLike): NDArray => unary("sqrt", a);
export const exp = (a: ArrayLike): NDArray => unary("exp", a);
export const log = (a: ArrayLike): NDArray => unary("log", a);

/** NumPy broadcast_shapes. Throws BroadcastError on mismatch. */
export function broadcastShapes(...shapes: (Shape | number)[]): number[] {
  const list = shapes.map((s) => (typeof s === "number" ? [s] : [...s]));
  return wrapNative(() => addon.broadcastShapes(list));
}

/** NumPy broadcast_to: a zero-stride view (treat as read-only). */
export function broadcastTo(a: ArrayLike, shape: Shape | number): NDArray {
  const x = toArray(a);
  const s = typeof shape === "number" ? [shape] : [...shape];
  return wrapNative(() => NDArray._wrap(addon.broadcastTo(x._native, s)));
}
