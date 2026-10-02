// P14 fromfile and NDArray.tofile (D-171). Binary mode reuses P7's
// `frombuffer` rules and the native C-order byte copy; text mode reuses P7's
// `fromstring` parser and the native Python-scalar formatter.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { nativeModule, type NativeNDArray } from "./addon.js";
import { dtype as toDType, type DTypeLike } from "./dtype.js";
import { ValueError, wrapNative } from "./errors.js";
import { NDArray } from "./ndarray.js";
import { frombuffer, fromstring } from "./p07_creation.js";
import { tofileText } from "./p14_text.js";

const native = nativeModule<{ toBytes(a: NativeNDArray): Buffer }>("p14");

const toPath = (f: string | URL): string => (typeof f === "string" ? f : fileURLToPath(f));

export interface FromfileOptions {
  /** Default float64. */
  dtype?: DTypeLike;
  /** Number of items to read; -1 (default) reads everything. */
  count?: number;
  /** Item separator; empty (default) reads binary data. Whitespace in `sep` matches any whitespace. */
  sep?: string;
  /** Bytes to skip first (binary mode only). */
  offset?: number;
}

/**
 * np.fromfile(file, {dtype, count, sep, offset}): reads a raw binary file (as
 * written by `tofile()`), or a text file of numbers separated by `sep`.
 * `file` is a path or the file contents as bytes.
 */
export function fromfile(file: string | URL | Uint8Array, options: FromfileOptions = {}): NDArray {
  const bytes = typeof file === "string" || file instanceof URL ? readFileSync(toPath(file)) : file;
  const sep = options.sep ?? "";
  const count = options.count ?? -1;
  const dt = toDType(options.dtype ?? "float64");
  if (sep !== "") {
    if ((options.offset ?? 0) !== 0) throw new TypeError("'offset' argument only permitted for binary files");
    return fromstring(Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength).toString("utf8"), {
      dtype: dt,
      count,
      sep,
    });
  }
  const offset = options.offset ?? 0;
  if (!Number.isSafeInteger(offset) || offset < 0) throw new ValueError("offset must be a non-negative integer");
  const avail = Math.max(0, bytes.byteLength - offset);
  const items = Math.floor(avail / dt.itemSize);
  // NumPy returns the items that are available when `count` is too large.
  const n = count >= 0 ? Math.min(count, items) : items;
  if (n === 0) return frombuffer(new Uint8Array(0), { dtype: dt });
  return frombuffer(bytes.subarray(offset, offset + n * dt.itemSize), { dtype: dt });
}

export interface TofileOptions {
  /** Item separator; empty (default) writes raw C-order bytes. */
  sep?: string;
  /** `%`-format applied to each item in text mode (default: Python `str` of the item). */
  format?: string;
}

/** NDArray.tofile implementation. Returns the bytes when `file` is null. */
export function tofile(a: NDArray, file: string | URL | null, options: TofileOptions = {}): Buffer | undefined {
  const sep = options.sep ?? "";
  const bytes =
    sep === ""
      ? wrapNative(() => native.toBytes(a._native))
      : Buffer.from(tofileText(a, sep, options.format ?? ""), "utf8");
  if (file === null) return bytes;
  writeFileSync(toPath(file), bytes);
  return undefined;
}

declare module "./ndarray.js" {
  interface NDArray {
    /**
     * NumPy ndarray.tofile(file, {sep, format}): writes the items in C order as
     * raw bytes (`sep` empty) or as text joined by `sep`. `file = null` returns the bytes.
     */
    tofile(file: string | URL | null, options?: TofileOptions): Buffer | undefined;
  }
}
NDArray.prototype.tofile = function (this: NDArray, file: string | URL | null, options: TofileOptions = {}) {
  return tofile(this, file, options);
};
