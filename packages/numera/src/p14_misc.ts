// P14 misc (D-172): baseRepr, binaryRepr (TS on bigint, NumPy's algorithms)
// and the window functions (native `addon.p14`).
import { nativeModule, type NativeNDArray } from "./addon.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";

interface P14WindowNative {
  window(kind: string, m: number): NativeNDArray;
  kaiser(m: number, beta: number): NativeNDArray;
  toBytes(a: NativeNDArray): Buffer;
}
const native = nativeModule<P14WindowNative>("p14");

/** An integer: a JS integer `number`, a `bigint`, a boolean or a 0-d integer/bool NDArray. */
export type IntegerLike = number | bigint | boolean | NDArray;

/** Python operator.index(v) as a bigint. @internal */
function index(v: IntegerLike): bigint {
  if (typeof v === "bigint") return v;
  if (typeof v === "boolean") return v ? 1n : 0n;
  if (typeof v === "number") {
    if (!Number.isInteger(v)) throw new TypeError("'float' object cannot be interpreted as an integer");
    return BigInt(v);
  }
  if (v instanceof NDArray && v.ndim === 0 && "biu".includes(v.dtype.kind)) {
    const b = wrapNative(() => native.toBytes(v._native));
    if (v.dtype.name === "int64") return b.readBigInt64LE(0);
    if (v.dtype.name === "uint64") return b.readBigUInt64LE(0);
    return BigInt(v.item() as number | boolean);
  }
  throw new TypeError("only integer scalar arrays can be converted to a scalar index");
}

const DIGITS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ";

/**
 * np.base_repr(number, base = 2, padding = 0): string of `number` in `base`
 * (2..36), with `padding` zeros prepended.
 */
export function baseRepr(number: IntegerLike, base = 2, padding = 0): string {
  if (!Number.isInteger(base) || !Number.isInteger(padding)) {
    throw new TypeError("base and padding must be integers");
  }
  if (base > DIGITS.length) throw new ValueError("Bases greater than 36 not handled in base_repr.");
  if (base < 2) throw new ValueError("Bases less than 2 not handled in base_repr.");
  const v = index(number);
  let num = v < 0n ? -v : v;
  const b = BigInt(base);
  let res = "";
  while (num > 0n) {
    res = DIGITS[Number(num % b)]! + res;
    num /= b;
  }
  if (padding > 0) res = "0".repeat(padding) + res;
  if (v < 0n) res = "-" + res;
  return res === "" ? "0" : res;
}

export interface BinaryReprOptions {
  /** Output length; negative numbers are then written in two's complement. */
  width?: number | null;
}

/**
 * np.binary_repr(num, {width}): binary string of `num`. Without `width`,
 * negative numbers get a minus sign; with `width`, two's complement is used.
 */
export function binaryRepr(num: IntegerLike, options: BinaryReprOptions = {}): string {
  const width = options.width ?? null;
  if (width !== null && !Number.isInteger(width)) throw new TypeError("width must be an integer");
  const insufficient = (binwidth: number): void => {
    if (width !== null && width < binwidth) {
      throw new ValueError(`Insufficient bit width=${width} provided for binwidth=${binwidth}`);
    }
  };
  const n = index(num);
  if (n === 0n) return "0".repeat(width || 1);
  if (n > 0n) {
    const binary = n.toString(2);
    insufficient(binary.length);
    return binary.padStart(Math.max(binary.length, width ?? 0), "0");
  }
  if (width === null) return "-" + (-n).toString(2);
  let poswidth = (-n).toString(2).length;
  // gh-8679: one digit fewer at the two's-complement boundary.
  if (2n ** BigInt(poswidth - 1) === -n) poswidth -= 1;
  const binary = (2n ** BigInt(poswidth + 1) + n).toString(2);
  insufficient(binary.length);
  return "1".repeat(Math.max(binary.length, width) - binary.length) + binary;
}

const windowSize = (M: number | bigint | NDArray): number => Number(M instanceof NDArray ? M.item() : M);

const makeWindow = (kind: string) => (M: number | bigint | NDArray): NDArray =>
  wrapNative(() => NDArray._wrap(native.window(kind, windowSize(M))));

/** np.bartlett(M): triangular window of `M` points (float64; empty when M < 1). */
export const bartlett = makeWindow("bartlett");
/** np.blackman(M): Blackman window of `M` points. */
export const blackman = makeWindow("blackman");
/** np.hamming(M): Hamming window of `M` points. */
export const hamming = makeWindow("hamming");
/** np.hanning(M): Hann window of `M` points. */
export const hanning = makeWindow("hanning");

/** np.kaiser(M, beta): Kaiser window of `M` points with shape parameter `beta`. */
export function kaiser(M: number | bigint | NDArray, beta: number | NDArray): NDArray {
  const b = beta instanceof NDArray ? Number(beta.item()) : beta;
  return wrapNative(() => NDArray._wrap(native.kaiser(windowSize(M), b)));
}
