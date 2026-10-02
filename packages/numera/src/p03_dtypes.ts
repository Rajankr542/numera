// P3-4: dtype introspection (D-062).
import { isComplexLike, type ComplexLike } from "./complex.js";
import { array, asarray } from "./creation.js";
import { DType, dtype as toDType, promoteTypes, type DTypeLike } from "./dtype.js";
import { DTypeError, ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray } from "./ndarray.js";
import { p03native } from "./p03_native.js";

type JSScalar = number | boolean | bigint | ComplexLike;

/** NumPy abstract scalar type (`np.generic`, `np.floating`, ...), for `issubdtype` (D-062). */
export class AbstractDType {
  /** @internal */
  constructor(
    readonly name: string,
    readonly parent: AbstractDType | null,
  ) {
    Object.freeze(this);
  }

  toString(): string {
    return this.name;
  }
}

export const generic = new AbstractDType("generic", null);
export const number = new AbstractDType("number", generic);
export const integer = new AbstractDType("integer", number);
export const signedinteger = new AbstractDType("signedinteger", integer);
export const unsignedinteger = new AbstractDType("unsignedinteger", integer);
export const inexact = new AbstractDType("inexact", number);
export const floating = new AbstractDType("floating", inexact);
export const complexfloating = new AbstractDType("complexfloating", inexact);

const KIND_PARENT: Record<DType["kind"], AbstractDType> = {
  b: generic,
  i: signedinteger,
  u: unsignedinteger,
  f: floating,
  c: complexfloating,
};

/** Machine limits of a float dtype (NumPy `finfo`, D-062). */
export interface FInfo {
  readonly bits: number;
  readonly eps: number;
  readonly epsneg: number;
  readonly max: number;
  readonly min: number;
  readonly tiny: number;
  readonly smallestNormal: number;
  readonly smallestSubnormal: number;
  readonly resolution: number;
  readonly precision: number;
  readonly iexp: number;
  readonly nexp: number;
  readonly nmant: number;
  readonly machep: number;
  readonly negep: number;
  readonly minexp: number;
  readonly maxexp: number;
  /** The float dtype (the component dtype for complex). */
  readonly dtype: DType;
}

/** Machine limits of an integer dtype (NumPy `iinfo`, D-062). */
export interface IInfo {
  readonly bits: number;
  /** As a JS number (lossy beyond 2^53, D-005); see `minExact`. */
  readonly min: number;
  readonly max: number;
  readonly minExact: bigint;
  readonly maxExact: bigint;
  readonly dtype: DType;
  readonly kind: "i" | "u";
}

/** NumPy `finfo`: limits of a float or complex dtype. */
export function finfo(dt: DTypeLike | NDArray): FInfo {
  const d = dt instanceof NDArray ? dt.dtype : toDType(dt);
  const r = wrapNative(() => p03native.finfo(d.name));
  return Object.freeze({
    bits: r.bits!,
    eps: r.eps!,
    epsneg: r.epsneg!,
    max: r.max!,
    min: r.min!,
    tiny: r.tiny!,
    smallestNormal: r.tiny!,
    smallestSubnormal: r.smallestSubnormal!,
    resolution: r.resolution!,
    precision: r.precision!,
    iexp: r.iexp!,
    nexp: r.nexp!,
    nmant: r.nmant!,
    machep: r.machep!,
    negep: r.negep!,
    minexp: r.minexp!,
    maxexp: r.maxexp!,
    dtype: toDType(r.dtype),
  });
}

/** NumPy `iinfo`: limits of an integer dtype. */
export function iinfo(dt: DTypeLike | NDArray): IInfo {
  const d = dt instanceof NDArray ? dt.dtype : toDType(dt);
  const r = wrapNative(() => p03native.iinfo(d.name));
  return Object.freeze({
    bits: r.bits,
    min: Number(r.min),
    max: Number(r.max),
    minExact: r.min,
    maxExact: r.max,
    dtype: d,
    kind: d.kind as "i" | "u",
  });
}

const isJSScalar = (x: unknown): x is JSScalar =>
  typeof x === "number" || typeof x === "boolean" || typeof x === "bigint" || isComplexLike(x);

/**
 * NumPy `result_type`: DTypes, names and arrays are promoted with
 * `promoteTypes`; JS scalars are weak (NEP 50, D-062).
 */
export function resultType(...args: (DTypeLike | NDArray | NestedArray)[]): DType {
  if (args.length === 0) throw new ValueError("at least one array or dtype is required");
  let strong: DType | undefined;
  let weak = -1; // 0 bool, 1 int, 2 float, 3 complex
  for (const a of args) {
    if (isJSScalar(a)) {
      const k =
        typeof a === "boolean"
          ? 0
          : typeof a === "bigint" || (typeof a === "number" && Number.isInteger(a))
            ? 1
            : typeof a === "number"
              ? 2
              : 3;
      weak = Math.max(weak, k);
      continue;
    }
    const d =
      a instanceof DType || typeof a === "string" ? toDType(a) : a instanceof NDArray ? a.dtype : asarray(a).dtype;
    strong = strong === undefined ? d : promoteTypes(strong, d);
  }
  if (strong === undefined) return toDType(["bool", "int64", "float64", "complex128"][weak]!);
  const k = strong.kind;
  if (weak === 1 && k === "b") return toDType("int64");
  if (weak === 2 && (k === "b" || k === "i" || k === "u")) return toDType("float64");
  if (weak === 3 && k !== "c") {
    return toDType(strong.name === "float16" || strong.name === "float32" ? "complex64" : "complex128");
  }
  return strong;
}

/** NumPy `min_scalar_type`: smallest dtype holding the value (arrays with ndim > 0: their dtype). */
export function minScalarType(a: NDArray | JSScalar): DType {
  let x: NDArray;
  if (a instanceof NDArray) x = a;
  else if (typeof a === "bigint" || (typeof a === "number" && Number.isSafeInteger(a))) {
    const v = BigInt(a);
    if (v < -(2n ** 63n) || v >= 2n ** 64n) {
      throw new ValueError(`integer ${v} does not fit any integer dtype (NumPy returns object)`);
    }
    x = array(v, { dtype: v < 2n ** 63n ? "int64" : "uint64" });
  } else x = typeof a === "number" ? array(a, { dtype: "float64" }) : array(a);
  return toDType(wrapNative(() => p03native.minScalarType(x._native)));
}

/** NumPy `issubdtype`: whether `a` is `b` or below it in the type hierarchy. */
export function issubdtype(a: DTypeLike | AbstractDType, b: DTypeLike | AbstractDType): boolean {
  if (!(b instanceof AbstractDType)) {
    return a instanceof AbstractDType ? false : toDType(a) === toDType(b);
  }
  let t: AbstractDType | null = a instanceof AbstractDType ? a : KIND_PARENT[toDType(a).kind];
  for (; t !== null; t = t.parent) if (t === b) return true;
  return false;
}

const KIND_NAMES: Record<string, readonly DType["kind"][]> = {
  bool: ["b"],
  "signed integer": ["i"],
  "unsigned integer": ["u"],
  integral: ["i", "u"],
  "real floating": ["f"],
  "complex floating": ["c"],
  numeric: ["i", "u", "f", "c"],
};

/** NumPy `isdtype` (array API): whether `dt` is of `kind` (a DType, a kind name, or a list of them). */
export function isdtype(dt: DType, kind: DType | string | readonly (DType | string)[]): boolean {
  if (!(dt instanceof DType)) {
    throw new DTypeError(`dtype argument must be a NumPy dtype, but it is a ${typeof dt}.`);
  }
  const kinds = Array.isArray(kind) ? kind : [kind];
  return kinds.some((k: unknown) => {
    if (k instanceof DType) return k === dt;
    if (typeof k === "string") {
      const ks = KIND_NAMES[k];
      if (ks === undefined) throw new ValueError(`kind argument is a string, but '${k}' is not a known kind name.`);
      return ks.includes(dt.kind);
    }
    throw new DTypeError(
      `kind argument must be comprised of NumPy dtypes or strings only, but is a ${k instanceof AbstractDType ? "AbstractDType" : typeof k}.`,
    );
  });
}

/** NumPy `common_type`: the float/complex dtype every input converts to safely (integers count as float64). */
export function commonType(...arrays: (NDArray | NestedArray)[]): DType {
  let complex = false;
  let precision = 0;
  for (const a of arrays) {
    const d = a instanceof NDArray ? a.dtype : asarray(a).dtype;
    if (d.kind === "b") throw new DTypeError("can't get common type for non-numeric array");
    if (d.kind === "c") complex = true;
    const p = d.kind === "i" || d.kind === "u" ? 2 : { float16: 0, float32: 1, float64: 2, complex64: 1, complex128: 2 }[d.name as string]!;
    precision = Math.max(precision, p);
  }
  return toDType(complex ? ["complex64", "complex64", "complex128"][precision]! : ["float16", "float32", "float64"][precision]!);
}

const TYPE_CHARS: Record<string, string> = {
  bool: "?",
  int8: "b",
  uint8: "B",
  int16: "h",
  uint16: "H",
  int32: "i",
  uint32: "I",
  int64: "l",
  uint64: "L",
  float16: "e",
  float32: "f",
  float64: "d",
  complex64: "F",
  complex128: "D",
};
const BY_ELSIZE = "GDFgdfQqLlIiHhBb?";

/** NumPy `mintypecode`: the smallest-size type character of `typeset` that `typechars` can be cast to. */
export function mintypecode(
  typechars: string | readonly (string | DType | NDArray | NestedArray)[],
  typeset = "GDFgdf",
  defaultCode = "d",
): string {
  const list = typeof typechars === "string" ? [...typechars] : typechars;
  const codes = list.map((t) =>
    typeof t === "string" ? t : TYPE_CHARS[(t instanceof DType ? t : t instanceof NDArray ? t.dtype : asarray(t).dtype).name]!,
  );
  const hit = new Set(codes.filter((c) => typeset.includes(c)));
  if (hit.size === 0) return defaultCode;
  if (hit.has("F") && hit.has("d")) return "D";
  return [...hit].sort((x, y) => BY_ELSIZE.indexOf(x) - BY_ELSIZE.indexOf(y))[0]!;
}
