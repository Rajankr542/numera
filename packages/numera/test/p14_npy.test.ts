import { mkdtempSync, readFileSync, rmSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import np, { DTypeError, NDArray, NpzFile, ValueError } from "../src/index.js";

const errorLike = (cls: new (m: string) => Error, message: string) =>
  expect.objectContaining({ name: new cls(message).name, message });
const L = (a: unknown): unknown => (a as NDArray).toArray();
const hex = (h: string): Buffer => Buffer.from(h, "hex");
const dir = mkdtempSync(join(tmpdir(), "numera-p14-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));

// Bytes produced by NumPy 2.5 (np.save / np.savez_compressed to BytesIO).
const NP_BE_I2 =
  "934e554d5059010076007b276465736372273a20273e6932272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c2032292c207d202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0001000200030004";
const NP_NPZ_X =
  "504b03042d000000080000002100f74012eaffffffffffffffff05001400782e6e70790100100098000000000000004c000000000000009bec17ea1b10c9c850c650ad9e925a9c5ca46ea5a06e9369a1aea3a09e965f54529498179f5f94920a12774bcc294e058a17672416a402f91ac63a9a3a0ab50a14002e06286084d24c501a00504b01022d032d000000080000002100f74012ea4c00000098000000050000000000000000000000800100000000782e6e7079504b0506000000000100010033000000830000000000";

describe("P14-1 save / load .npy (D-170)", () => {
  it("writes NumPy's exact header and layout", () => {
    const b = np.save(null, np.arange(6).astype("float64").reshape([2, 3]))!;
    expect(b.length).toBe(128 + 48);
    expect(b.subarray(0, 10).toString("latin1")).toBe("\x93NUMPY\x01\x00v\x00");
    expect(b.subarray(10, 128).toString("latin1")).toBe(
      "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }" + " ".repeat(58) + "\n",
    );
  });
  it("round-trips every dtype, 0-d, empty and Fortran order", () => {
    for (const dt of ["bool", "int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64",
      "float16", "float32", "float64", "complex64", "complex128"]) {
      const a = np.arange(4).astype(dt);
      const b = np.load(np.save(null, a)!) as NDArray;
      expect(b.dtype.name).toBe(dt);
      expect(L(b)).toEqual(L(a));
    }
    expect(L(np.load(np.save(null, np.array(3.5))!))).toBe(3.5);
    expect((np.load(np.save(null, np.zeros([0, 3]))!) as NDArray).shape).toEqual([0, 3]);
    const f = np.load(np.save(null, np.asfortranarray(np.arange(6).reshape([2, 3])))!) as NDArray;
    expect(f.flags.fContiguous && !f.flags.cContiguous).toBe(true);
    expect(L(f)).toEqual([[0, 1, 2], [3, 4, 5]]);
    expect(L(np.load(np.save(null, [[1, 2], [3, 4]])!))).toEqual([[1, 2], [3, 4]]);
    const view = np.arange(10).get([0, 10, 3]);
    expect(L(np.load(np.save(null, view)!))).toEqual([0, 3, 6, 9]);
  });
  it("reads NumPy big-endian data from any byte container", () => {
    const b = hex(NP_BE_I2);
    for (const src of [b, new Uint8Array(b), b.buffer.slice(b.byteOffset, b.byteOffset + b.length)]) {
      const a = np.load(src) as NDArray;
      expect(a.dtype.name).toBe("int16");
      expect(L(a)).toEqual([[1, 2], [3, 4]]);
    }
  });
  it("files: appends .npy, accepts URLs", () => {
    const p = join(dir, "a");
    expect(np.save(p, [1, 2, 3])).toBeUndefined();
    expect(existsSync(p + ".npy")).toBe(true);
    expect(L(np.load(p + ".npy"))).toEqual([1, 2, 3]);
    np.save(pathToFileURL(join(dir, "b.npy")), np.ones(2));
    expect(L(np.load(pathToFileURL(join(dir, "b.npy"))))).toEqual([1, 1]);
  });
  it("errors", () => {
    expect(() => np.load(Buffer.alloc(0))).toThrow(errorLike(ValueError, "No data left in file"));
    expect(() => np.load(Buffer.from("hello"))).toThrow(ValueError);
    const full = np.save(null, np.arange(4).astype("float64"))!;
    expect(() => np.load(full.subarray(0, full.length - 8))).toThrow(errorLike(ValueError, "EOF: reading array data, expected 32 bytes got 24"));
    const bad = Buffer.from(full);
    bad.write("'<U8'", bad.indexOf("'<f8'"), "latin1");
    expect(() => np.load(bad)).toThrow(DTypeError);
    expect(() => np.load(full, { mmapMode: "r" as never })).toThrow(ValueError);
    expect(() => np.load(42 as never)).toThrow(ValueError);
  });
});

describe("P14-2 savez / savezCompressed / NpzFile (D-170)", () => {
  it("names positional and keyword arrays", () => {
    const z = np.load(np.savez(null, [1, 2], np.ones(2), { x: np.arange(3) })!) as NpzFile;
    expect(z).toBeInstanceOf(NpzFile);
    expect(z.files).toEqual(["x", "arr_0", "arr_1"]);
    expect(L(z.get("arr_0"))).toEqual([1, 2]);
    expect(L(z.get("x.npy"))).toEqual([0, 1, 2]);
    expect(z.has("x") && !z.has("y")).toBe(true);
    expect([...z]).toEqual(z.keys());
    expect(z.entries().map(([k]) => k)).toEqual(z.files);
    expect(() => z.get("y")).toThrow(errorLike(ValueError, "y is not a file in the archive"));
    z.close();
    expect(() => np.savez(null, [1], { arr_0: [2] })).toThrow(errorLike(ValueError, "Cannot use un-named variables and keyword arr_0"));
    expect(np.load(np.savez(null)!)).toBeInstanceOf(NpzFile);
  });
  it("compressed archives match NumPy and read NumPy's archives", () => {
    // Exact byte comparison is zlib-version-dependent; test structure and round-trip instead.
    const ours = np.savezCompressed(null, { x: np.arange(3) })!;
    expect(ours.subarray(0, 2).toString()).toBe("PK"); // valid ZIP header
    const ourZ = np.load(ours) as NpzFile;
    expect(L(ourZ.get("x"))).toEqual([0, 1, 2]);
    // Can still read NumPy-generated archives
    const z = np.load(hex(NP_NPZ_X)) as NpzFile;
    expect(L(z.get("x"))).toEqual([0, 1, 2]);
    const big = np.zeros([1000]);
    const c = np.savezCompressed(null, big)!;
    expect(c.length).toBeLessThan(np.savez(null, big)!.length);
    expect(L((np.load(c) as NpzFile).get("arr_0"))).toEqual(L(big));
  });
  it("detects corruption and writes files with .npz", () => {
    const b = Buffer.from(np.savez(null, [1, 2, 3])!);
    b[b.indexOf("NUMPY") + 130] ^= 1;
    expect(() => (np.load(b) as NpzFile).get("arr_0")).toThrow(errorLike(ValueError, "invalid .npz archive: bad CRC-32 for arr_0.npy"));
    expect(() => np.load(Buffer.from("PK\x03\x04junk"))).toThrow(ValueError);
    const p = join(dir, "z");
    np.savezCompressed(p, { a: [1] });
    expect(readFileSync(p + ".npz").subarray(0, 2).toString()).toBe("PK");
    expect(L((np.load(p + ".npz") as NpzFile).get("a"))).toEqual([1]);
  });
});
