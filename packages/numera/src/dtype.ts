import { addon } from "./addon.js";
import { DTypeError, wrapNative } from "./errors.js";

export type DTypeName =
  | "bool"
  | "int8"
  | "uint8"
  | "int16"
  | "uint16"
  | "int32"
  | "uint32"
  | "int64"
  | "uint64"
  | "float16"
  | "float32"
  | "float64"
  | "complex64"
  | "complex128";

/** A nativpy data type (PLAN §10). Instances are singletons per name. */
export class DType {
  readonly name: DTypeName;
  readonly itemSize: number;
  readonly alignment: number;
  /** NumPy kind character: b, i, u, f, c. */
  readonly kind: "b" | "i" | "u" | "f" | "c";

  /** @internal */
  constructor(name: DTypeName) {
    const info = addon.dtypes[name];
    if (info === undefined) throw new DTypeError(`data type '${name}' not understood`);
    this.name = name;
    this.itemSize = info.itemsize;
    this.alignment = info.alignment;
    this.kind = info.kind as DType["kind"];
  }

  toString(): string {
    return this.name;
  }
}

const registry = new Map<string, DType>();
for (const name of Object.keys(addon.dtypes)) {
  registry.set(name, new DType(name as DTypeName));
}

/** Aliases accepted in addition to canonical names (NumPy spellings). */
const aliases: Record<string, DTypeName> = {
  bool_: "bool",
  float: "float64",
  double: "float64",
  single: "float32",
  half: "float16",
  int: "int64",
  complex: "complex128",
};

export type DTypeLike = DType | DTypeName | string;

/** Resolves a dtype name or object to the canonical DType. */
export function dtype(like: DTypeLike): DType {
  if (like instanceof DType) return like;
  const canonical = aliases[like] ?? like;
  const dt = registry.get(canonical);
  if (dt === undefined) throw new DTypeError(`data type '${like}' not understood`);
  return dt;
}

function get(name: DTypeName): DType {
  return dtype(name);
}

export const bool = get("bool");
export const int8 = get("int8");
export const uint8 = get("uint8");
export const int16 = get("int16");
export const uint16 = get("uint16");
export const int32 = get("int32");
export const uint32 = get("uint32");
export const int64 = get("int64");
export const uint64 = get("uint64");
export const float16 = get("float16");
export const float32 = get("float32");
export const float64 = get("float64");
export const complex64 = get("complex64");
export const complex128 = get("complex128");

/** Equivalent of numpy.promote_types. */
export function promoteTypes(a: DTypeLike, b: DTypeLike): DType {
  return dtype(addon.promoteTypes(dtype(a).name, dtype(b).name));
}

/** NumPy casting rule names (D-045). */
export type Casting = "no" | "equiv" | "safe" | "same_kind" | "unsafe";

/**
 * Equivalent of numpy.can_cast: whether `from` can be cast to `to` under the
 * `casting` rule. `from` may be a dtype or an array (its dtype is used).
 * NumPy 2 has no value-based casting, so JS scalars are rejected (D-045).
 */
export function canCast(
  from: DTypeLike | { readonly dtype: DType },
  to: DTypeLike,
  casting: Casting = "safe",
): boolean {
  const kind = typeof from;
  if (kind === "number" || kind === "boolean" || kind === "bigint") {
    throw new DTypeError(
      "canCast() does not support JS numbers, booleans or bigints because the result used to depend on the value (NEP 50)",
    );
  }
  const src = typeof from === "object" && !(from instanceof DType) ? from.dtype : from;
  return wrapNative(() => addon.canCast(dtype(src).name, dtype(to).name, casting));
}
