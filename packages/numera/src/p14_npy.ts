// P14 NPY / NPZ binary files (D-170): save, load, savez, savezCompressed.
// The .npy codec and CRC-32 are native (`addon.p14`); the zip container is
// written and read here, with deflate from node:zlib.
import { readFileSync, writeFileSync } from "node:fs";
import { deflateRawSync, inflateRawSync } from "node:zlib";
import { fileURLToPath } from "node:url";
import { nativeModule, type NativeNDArray } from "./addon.js";
import { asarray } from "./creation.js";
import { ValueError, wrapNative } from "./errors.js";
import { isComplexLike } from "./complex.js";
import { NDArray, type NestedArray } from "./ndarray.js";

type ArrayLike = NDArray | NestedArray;

/** Native table `addon.p14` (native/bindings/p14_binding.cpp). Internal. */
export interface P14NpyNative {
  npyEncode(a: NativeNDArray): Buffer;
  npyDecode(data: Uint8Array): NativeNDArray;
  crc32(data: Uint8Array, crc?: number): number;
}
const native = nativeModule<P14NpyNative>("p14");

/** A file argument: a path (string or file URL), or `null` to work on bytes. */
export type FileLike = string | URL | null;
/** Raw file contents accepted by `load`. */
export type BytesLike = Uint8Array | ArrayBuffer;

const toPath = (f: string | URL): string => (typeof f === "string" ? f : fileURLToPath(f));
const withExt = (f: string | URL, ext: string): string => {
  const p = toPath(f);
  return p.endsWith(ext) ? p : p + ext;
};

const encodeNpy = (a: ArrayLike): Buffer => wrapNative(() => native.npyEncode(asarray(a)._native));
const decodeNpy = (b: Uint8Array): NDArray => wrapNative(() => NDArray._wrap(native.npyDecode(b)));

/**
 * np.save(file, arr): writes `arr` in .npy format (`.npy` is appended to a
 * path without it). With `file = null` the encoded bytes are returned instead.
 */
export function save(file: FileLike, arr: ArrayLike): Buffer | undefined {
  const bytes = encodeNpy(arr);
  if (file === null) return bytes;
  writeFileSync(withExt(file, ".npy"), bytes);
  return undefined;
}

export interface LoadOptions {
  /** Only `null` is supported (no memory mapping). */
  mmapMode?: null;
  /** Ignored: object arrays are never loaded. */
  allowPickle?: boolean;
}

const MAGIC_NPY = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
const startsWith = (b: Uint8Array, sig: readonly number[]): boolean =>
  b.length >= sig.length && sig.every((v, i) => b[i] === v);

/**
 * np.load(file): reads a .npy file (returns an NDArray) or .npz archive
 * (returns an `NpzFile`). `file` is a path or the file contents as bytes.
 */
export function load(file: string | URL | BytesLike, options: LoadOptions = {}): NDArray | NpzFile {
  if (options.mmapMode != null) throw new ValueError("mmapMode is not supported; use mmapMode: null");
  let bytes: Uint8Array;
  if (typeof file === "string" || file instanceof URL) bytes = readFileSync(toPath(file));
  else if (file instanceof ArrayBuffer) bytes = new Uint8Array(file);
  else if (ArrayBuffer.isView(file)) bytes = new Uint8Array(file.buffer, file.byteOffset, file.byteLength);
  else throw new ValueError("load expects a path, a URL or a Buffer/Uint8Array/ArrayBuffer");
  if (bytes.length === 0) throw new ValueError("No data left in file");
  if (startsWith(bytes, [0x50, 0x4b, 0x03, 0x04]) || startsWith(bytes, [0x50, 0x4b, 0x05, 0x06])) {
    return new NpzFile(bytes);
  }
  if (startsWith(bytes, MAGIC_NPY)) return decodeNpy(bytes);
  throw new ValueError(
    "This file is neither .npy nor .npz data (pickled data is not supported by numera)",
  );
}

// ---------------- zip container ----------------

interface ZipMember {
  name: string;
  method: number;
  crc: number;
  compSize: number;
  size: number;
  offset: number;
}

const ZIP64_LIMIT = 2 ** 31 - 1;

class Writer {
  private chunks: Buffer[] = [];
  length = 0;
  push(b: Buffer): void {
    this.chunks.push(b);
    this.length += b.length;
  }
  result(): Buffer {
    return Buffer.concat(this.chunks, this.length);
  }
}

function le(size: number, value: number | bigint): Buffer {
  const b = Buffer.alloc(size);
  if (size === 2) b.writeUInt16LE(Number(value));
  else if (size === 4) b.writeUInt32LE(Number(value));
  else b.writeBigUInt64LE(BigInt(value));
  return b;
}

/** Builds a zip archive like Python's zipfile with `force_zip64=True` (D-170). */
function writeZip(entries: [string, Buffer][], compress: boolean): Buffer {
  const w = new Writer();
  const members: (ZipMember & { flags: number; nameBytes: Buffer })[] = [];
  for (const [name, data] of entries) {
    const ascii = /^[\x00-\x7f]*$/.test(name);
    const nameBytes = Buffer.from(name, ascii ? "latin1" : "utf8");
    const flags = ascii ? 0 : 0x800;
    const method = compress ? 8 : 0;
    const body = compress ? deflateRawSync(data, { level: 6 }) : data;
    const crc = native.crc32(data);
    const offset = w.length;
    w.push(Buffer.concat([
      le(4, 0x04034b50), le(2, 45), le(2, flags), le(2, method), le(2, 0), le(2, 0x21), le(4, crc),
      le(4, 0xffffffff), le(4, 0xffffffff), le(2, nameBytes.length), le(2, 20), nameBytes,
      le(2, 1), le(2, 16), le(8, data.length), le(8, body.length),
    ]));
    w.push(body);
    members.push({ name, method, crc, compSize: body.length, size: data.length, offset, flags, nameBytes });
  }
  const cdOffset = w.length;
  for (const m of members) {
    const extra: number[] = [];
    let size = m.size;
    let compSize = m.compSize;
    let offset = m.offset;
    if (m.size > ZIP64_LIMIT || m.compSize > ZIP64_LIMIT) {
      extra.push(m.size, m.compSize);
      size = compSize = 0xffffffff;
    }
    if (m.offset > ZIP64_LIMIT) {
      extra.push(m.offset);
      offset = 0xffffffff;
    }
    const extraBuf = extra.length === 0
      ? Buffer.alloc(0)
      : Buffer.concat([le(2, 1), le(2, 8 * extra.length), ...extra.map((v) => le(8, v))]);
    w.push(Buffer.concat([
      le(4, 0x02014b50), le(2, 0x032d), le(2, 45), le(2, m.flags), le(2, m.method), le(2, 0), le(2, 0x21),
      le(4, m.crc), le(4, compSize), le(4, size), le(2, m.nameBytes.length), le(2, extraBuf.length), le(2, 0),
      le(2, 0), le(2, 0), le(4, 0o600 << 16), le(4, offset), m.nameBytes, extraBuf,
    ]));
  }
  let count = members.length;
  let cdSize = w.length - cdOffset;
  let cdOff = cdOffset;
  if (count > 0xffff || cdOff > ZIP64_LIMIT || cdSize > ZIP64_LIMIT) {
    const pos = w.length;
    w.push(Buffer.concat([
      le(4, 0x06064b50), le(8, 44), le(2, 45), le(2, 45), le(4, 0), le(4, 0), le(8, count), le(8, count),
      le(8, cdSize), le(8, cdOff), le(4, 0x07064b50), le(4, 0), le(8, pos), le(4, 1),
    ]));
    count = Math.min(count, 0xffff);
    cdSize = Math.min(cdSize, 0xffffffff);
    cdOff = Math.min(cdOff, 0xffffffff);
  }
  w.push(Buffer.concat([
    le(4, 0x06054b50), le(2, 0), le(2, 0), le(2, count), le(2, count), le(4, cdSize), le(4, cdOff), le(2, 0),
  ]));
  return w.result();
}

const badZip = (msg: string): never => {
  throw new ValueError(`invalid .npz archive: ${msg}`);
};

function readZip(bytes: Uint8Array): Map<string, ZipMember> {
  const b = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let eocd = -1;
  for (let i = b.length - 22; i >= Math.max(0, b.length - 22 - 0xffff); i--) {
    if (b.readUInt32LE(i) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  if (eocd < 0) badZip("end of central directory not found");
  let count = b.readUInt16LE(eocd + 10);
  let cdOffset = b.readUInt32LE(eocd + 16);
  if (eocd >= 20 && b.readUInt32LE(eocd - 20) === 0x07064b50) {
    const rec = Number(b.readBigUInt64LE(eocd - 12));
    if (rec + 56 > b.length || b.readUInt32LE(rec) !== 0x06064b50) badZip("bad zip64 end record");
    count = Number(b.readBigUInt64LE(rec + 32));
    cdOffset = Number(b.readBigUInt64LE(rec + 48));
  }
  const out = new Map<string, ZipMember>();
  let p = cdOffset;
  for (let k = 0; k < count; k++) {
    if (p + 46 > b.length || b.readUInt32LE(p) !== 0x02014b50) badZip("bad central directory entry");
    const flags = b.readUInt16LE(p + 8);
    const method = b.readUInt16LE(p + 10);
    const crc = b.readUInt32LE(p + 16);
    let compSize = b.readUInt32LE(p + 20);
    let size = b.readUInt32LE(p + 24);
    const nlen = b.readUInt16LE(p + 28);
    const xlen = b.readUInt16LE(p + 30);
    const clen = b.readUInt16LE(p + 32);
    let offset = b.readUInt32LE(p + 42);
    const name = b.toString(flags & 0x800 ? "utf8" : "latin1", p + 46, p + 46 + nlen);
    let x = p + 46 + nlen;
    const xend = x + xlen;
    while (x + 4 <= xend) {
      const tag = b.readUInt16LE(x);
      const len = b.readUInt16LE(x + 2);
      if (tag === 1) {
        let q = x + 4;
        const next = (): number => {
          const v = Number(b.readBigUInt64LE(q));
          q += 8;
          return v;
        };
        if (size === 0xffffffff) size = next();
        if (compSize === 0xffffffff) compSize = next();
        if (offset === 0xffffffff) offset = next();
      }
      x += 4 + len;
    }
    out.set(name, { name, method, crc, compSize, size, offset });
    p = xend + clen;
  }
  return out;
}

function memberData(bytes: Uint8Array, m: ZipMember): Buffer {
  const b = Buffer.from(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (m.offset + 30 > b.length || b.readUInt32LE(m.offset) !== 0x04034b50) badZip(`bad local header for ${m.name}`);
  const start = m.offset + 30 + b.readUInt16LE(m.offset + 26) + b.readUInt16LE(m.offset + 28);
  if (start + m.compSize > b.length) badZip(`truncated member ${m.name}`);
  const raw = b.subarray(start, start + m.compSize);
  let data: Buffer;
  if (m.method === 0) data = raw;
  else if (m.method === 8) data = inflateRawSync(raw);
  else return badZip(`unsupported compression method ${m.method} for ${m.name}`);
  if (data.length !== m.size || native.crc32(data) !== m.crc) badZip(`bad CRC-32 for ${m.name}`);
  return data;
}

/**
 * Lazy view of a .npz archive returned by `np.load` (NumPy `NpzFile`).
 * `get(name)` decodes a member each time it is called.
 */
export class NpzFile implements Iterable<string> {
  /** Member names with the `.npy` suffix removed. */
  readonly files: string[];
  private readonly bytes: Uint8Array;
  private readonly members: Map<string, ZipMember>;

  /** @internal */
  constructor(bytes: Uint8Array) {
    this.bytes = bytes;
    this.members = readZip(bytes);
    this.files = [...this.members.keys()].map((n) => (n.endsWith(".npy") ? n.slice(0, -4) : n));
  }

  /** The array stored under `key` (with or without `.npy`); raw bytes for non-.npy members. */
  get(key: string): NDArray | Buffer {
    let m = this.members.get(key);
    if (m === undefined && this.files.includes(key)) m = this.members.get(key + ".npy");
    if (m === undefined) throw new ValueError(`${key} is not a file in the archive`);
    const data = memberData(this.bytes, m);
    return startsWith(data, MAGIC_NPY) ? decodeNpy(data) : data;
  }

  has(key: string): boolean {
    return this.members.has(key) || this.files.includes(key);
  }

  keys(): string[] {
    return [...this.files];
  }

  entries(): [string, NDArray | Buffer][] {
    return this.files.map((f) => [f, this.get(f)]);
  }

  [Symbol.iterator](): Iterator<string> {
    return this.files[Symbol.iterator]();
  }

  /** No-op (the archive is held in memory); kept for NumPy parity. */
  close(): void {}
}

/** Named arrays for savez: a plain object (NumPy keyword arguments). */
export type NamedArrays = Record<string, ArrayLike>;

const isNamed = (x: unknown): x is NamedArrays =>
  typeof x === "object" && x !== null && !Array.isArray(x) && !(x instanceof NDArray) && !isComplexLike(x);

function savezImpl(file: FileLike, args: (ArrayLike | NamedArrays)[], compress: boolean): Buffer | undefined {
  let named: NamedArrays = {};
  let positional = args;
  const last = args[args.length - 1];
  if (args.length > 0 && isNamed(last)) {
    named = last;
    positional = args.slice(0, -1);
  }
  const all = new Map<string, ArrayLike>(Object.entries(named));
  positional.forEach((v, i) => {
    if (isNamed(v)) throw new ValueError("savez: named arrays must be passed as the last argument");
    const key = `arr_${i}`;
    if (all.has(key)) throw new ValueError(`Cannot use un-named variables and keyword ${key}`);
    all.set(key, v);
  });
  const entries: [string, Buffer][] = [...all].map(([k, v]) => [k + ".npy", encodeNpy(v)]);
  const zip = writeZip(entries, compress);
  if (file === null) return zip;
  writeFileSync(withExt(file, ".npz"), zip);
  return undefined;
}

/**
 * np.savez(file, ...arrays, {name: array, ...}): uncompressed .npz archive.
 * Positional arrays are named `arr_0`, `arr_1`, ...; a trailing plain object
 * gives named arrays. `file = null` returns the archive bytes.
 */
export function savez(file: FileLike, ...arrays: (ArrayLike | NamedArrays)[]): Buffer | undefined {
  return savezImpl(file, arrays, false);
}

/** np.savez_compressed: like `savez` with deflate compression. */
export function savezCompressed(file: FileLike, ...arrays: (ArrayLike | NamedArrays)[]): Buffer | undefined {
  return savezImpl(file, arrays, true);
}
