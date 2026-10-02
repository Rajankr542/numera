import { isComplexLike, type ComplexLike } from "./complex.js";
import { array, asarray } from "./creation.js";
import { dtype as toDType, type Casting, type DTypeLike } from "./dtype.js";
import { DTypeError, ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray } from "./ndarray.js";
import { native } from "./p06_native.js";
import { asArr, normAxis, type ArrayLike } from "./p06_util.js";
import { broadcastTo } from "./ufunc.js";

/** copyto / require / asarray variants / metadata helpers (P6, D-094). */

const KIND_ORDER: Record<string, number> = { b: 0, u: 1, i: 1, f: 2, c: 3 };
const INT_RANGE: Record<string, [bigint, bigint]> = {
  int8: [-128n, 127n],
  uint8: [0n, 255n],
  int16: [-32768n, 32767n],
  uint16: [0n, 65535n],
  int32: [-2147483648n, 2147483647n],
  uint32: [0n, 4294967295n],
  int64: [-(2n ** 63n), 2n ** 63n - 1n],
  uint64: [0n, 2n ** 64n - 1n],
};

export interface CopytoOptions {
  /** Casting rule (default "same_kind"). */
  casting?: Casting;
  /** Boolean mask broadcast to `dst`: only those elements are written. */
  where?: NDArray | NestedArray | boolean;
}

type Scalar = number | boolean | bigint | ComplexLike;

const isScalar = (v: unknown): v is Scalar =>
  typeof v === "number" || typeof v === "boolean" || typeof v === "bigint" || isComplexLike(v);

// NEP 50 weak-scalar check for a JS scalar written into `dst` (D-094).
function scalarSource(dst: NDArray, v: Scalar, casting: Casting): [NDArray, Casting] {
  if (typeof v === "boolean") return [array(v), casting];
  const kind =
    typeof v === "bigint" ? "i" : typeof v === "number" ? (Number.isSafeInteger(v) ? "i" : "f") : "c";
  const from = kind === "i" ? "int64" : kind === "f" ? "float64" : "complex128";
  const dk = dst.dtype.kind;
  const ok =
    casting === "unsafe" ||
    (casting !== "equiv" && KIND_ORDER[kind]! <= KIND_ORDER[dk]! && dk !== "b");
  if (!ok) {
    throw new DTypeError(
      `Cannot cast scalar from dtype('${from}') to dtype('${dst.dtype.name}') according to the rule '${casting}'`,
    );
  }
  const range = INT_RANGE[dst.dtype.name];
  if (kind === "i" && range !== undefined) {
    const b = BigInt(v as number | bigint);
    if (b < range[0] || b > range[1]) {
      throw new ValueError(`Python integer ${b} out of bounds for ${dst.dtype.name}`);
    }
  }
  return [array(v as NestedArray, { dtype: dst.dtype }), "unsafe"];
}

const CASTINGS = ["no", "equiv", "safe", "same_kind", "unsafe"];

/** NumPy copyto: copies `src` (broadcast) into `dst` in place, under `casting`. */
export function copyto(dst: NDArray, src: ArrayLike | Scalar, options: CopytoOptions = {}): void {
  if (!(dst instanceof NDArray)) throw new DTypeError("copyto() argument 1 must be an NDArray");
  const casting = options.casting ?? "same_kind";
  if (!CASTINGS.includes(casting)) {
    throw new ValueError(
      `casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '${String(casting)}')`,
    );
  }
  const [s, c] =
    src instanceof NDArray
      ? [src, casting]
      : isScalar(src)
        ? scalarSource(dst, src, casting)
        : [array(src as NestedArray), casting];
  let where: NDArray | null = null;
  if (options.where !== undefined) {
    const w = options.where;
    where = w instanceof NDArray ? w : array(w as NestedArray, { dtype: "bool" });
  }
  wrapNative(() => native.copyto(dst._native, s._native, c, where ? where._native : null));
}

/** NumPy asanyarray (numera has no subclasses: same as asarray). */
export function asanyarray(a: ArrayLike, options: { dtype?: DTypeLike } = {}): NDArray {
  return asarray(a, options);
}

/** NumPy asarray_chkfinite: asarray that raises ValueError on NaN or infinity. */
export function asarrayChkfinite(a: ArrayLike, options: { dtype?: DTypeLike } = {}): NDArray {
  const x = asarray(a, options);
  if (!wrapNative(() => native.allFinite(x._native))) {
    throw new ValueError("array must not contain infs or NaNs");
  }
  return x;
}

const FLAG: Record<string, string> = {
  C: "C",
  C_CONTIGUOUS: "C",
  CONTIGUOUS: "C",
  F: "F",
  F_CONTIGUOUS: "F",
  FORTRAN: "F",
  A: "A",
  ALIGNED: "A",
  W: "W",
  WRITEABLE: "W",
  O: "O",
  OWNDATA: "O",
  E: "E",
  ENSUREARRAY: "E",
};

/** Requirement flags for `require`: letters (`"CW"`) or a list of letters / flag names. */
export type Requirements = string | readonly string[];

/** NumPy require: an array of `dtype` satisfying the requirement flags (copying only if needed). */
export function require(a: ArrayLike, dtype?: DTypeLike | null, requirements?: Requirements | null): NDArray {
  const list = requirements === undefined || requirements === null ? [] : typeof requirements === "string" ? [...requirements] : [...requirements];
  const reqs = new Set<string>();
  for (const r of list) {
    const f = FLAG[r.toUpperCase()];
    if (f === undefined) throw new ValueError(`unknown requirement flag '${r}'`);
    reqs.add(f);
  }
  const dt = dtype === undefined || dtype === null ? undefined : toDType(dtype);
  if (reqs.size === 0) return asanyarray(a, dt === undefined ? {} : { dtype: dt });
  let order: "C" | "F" | "A" = "A";
  if (reqs.has("C") && reqs.has("F")) throw new ValueError('Cannot specify both "C" and "F" order');
  if (reqs.has("C")) order = "C";
  if (reqs.has("F")) order = "F";
  let arr = asarray(a, dt === undefined ? {} : { dtype: dt });
  if (order === "C" && !arr.flags.cContiguous) arr = arr.copy({ order: "C" });
  if (order === "F" && !arr.flags.fContiguous) arr = arr.copy({ order: "F" });
  const f = arr.flags;
  if ((reqs.has("W") && !f.writeable) || (reqs.has("O") && !f.ownData)) {
    return arr.copy({ order });
  }
  return arr;
}

/** NumPy broadcast_arrays: views of every input broadcast to the common shape. */
export function broadcastArrays(...args: ArrayLike[]): NDArray[] {
  const xs = args.map(asArr);
  if (xs.length === 0) return [];
  const shape = wrapNative(() => broadcastShapeOf(xs));
  return xs.map((x) => (sameShape(x.shape, shape) ? x : broadcastView(x, shape)));
}

function broadcastShapeOf(xs: NDArray[]): number[] {
  const nd = Math.max(...xs.map((x) => x.ndim));
  const out = new Array<number>(nd).fill(1);
  for (const x of xs) {
    for (let i = 1; i <= x.ndim; i++) {
      const d = x.shape[x.ndim - i]!;
      const o = out[nd - i]!;
      if (d !== o && d !== 1 && o !== 1) {
        throw new ValueError(
          "shape mismatch: objects cannot be broadcast to a single shape.  Mismatch is between " +
            `arg 0 with shape (${xs[0]!.shape.join(", ")}${xs[0]!.ndim === 1 ? "," : ""}) and ` +
            `arg ${xs.indexOf(x)} with shape (${x.shape.join(", ")}${x.ndim === 1 ? "," : ""}).`,
        );
      }
      if (o === 1) out[nd - i] = d;
    }
  }
  return out;
}

const sameShape = (a: readonly number[], b: readonly number[]): boolean =>
  a.length === b.length && a.every((v, i) => v === b[i]);

// broadcast_to views are read-only (D-016); broadcast_arrays results keep the
// input's writeable flag (NumPy behaviour, with a FutureWarning there).
function broadcastView(x: NDArray, shape: number[]): NDArray {
  const b = broadcastTo(x, shape);
  return wrapNative(() => NDArray._wrap(x._native.view(shape, [...b.strides], b._native.offset())));
}

/** NumPy shape: the shape of any array-like. */
export function shape(a: ArrayLike): number[] {
  return [...asArr(a).shape];
}

/** NumPy size: number of elements, or the length of `axis`. */
export function size(a: ArrayLike, axis?: number | readonly number[] | null): number {
  const x = asArr(a);
  if (axis === undefined || axis === null) return x.size;
  const axes = typeof axis === "number" ? [axis] : [...axis];
  return axes.reduce((p, ax) => p * x.shape[normAxis(ax, x.ndim)]!, 1);
}

/** NumPy ndim: number of dimensions of any array-like. */
export function ndim(a: ArrayLike): number {
  return asArr(a).ndim;
}

/** NumPy isfortran: true if F-contiguous and not C-contiguous. */
export function isfortran(a: NDArray): boolean {
  if (!(a instanceof NDArray)) throw new DTypeError("isfortran() argument must be an NDArray");
  const f = a.flags;
  return f.fContiguous && !f.cContiguous;
}
