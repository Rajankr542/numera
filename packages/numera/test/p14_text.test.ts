import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gunzipSync } from "node:zlib";
import { afterAll, describe, expect, it } from "vitest";
import np, { NDArray, ValueError } from "../src/index.js";

const errorLike = (cls: new (m: string) => Error, message: string) =>
  expect.objectContaining({ name: new cls(message).name, message });
const L = (a: unknown): unknown => (a as NDArray).toArray();
const dir = mkdtempSync(join(tmpdir(), "numera-p14t-"));
afterAll(() => rmSync(dir, { recursive: true, force: true }));
const fails = (f: () => unknown, cls: new (m: string) => Error, message: string) => {
  let err: unknown;
  try {
    f();
  } catch (e) {
    err = e;
  }
  expect(err).toEqual(errorLike(cls, message));
};

describe("P14-3 loadtxt (D-171)", () => {
  it("reads whitespace tables from paths, bytes and line arrays", () => {
    const p = join(dir, "t.txt");
    writeFileSync(p, "# x y\n1 2\n\n3 4 # c\r\n");
    expect(L(np.loadtxt(p))).toEqual([[1, 2], [3, 4]]);
    expect(L(np.loadtxt(Buffer.from("1 2\n3 4")))).toEqual([[1, 2], [3, 4]]);
    expect(L(np.loadtxt(["1 2", "3 4"], { unpack: true }))).toEqual([[1, 3], [2, 4]]);
    expect(np.loadtxt(["5"]).shape).toEqual([]);
  });
  it("delimiter, comments, skiprows, maxRows, usecols, quotechar", () => {
    const lines = ["h", "1;2;3", "// skip", "4;5;6", "7;8;9"];
    const a = np.loadtxt(lines, { delimiter: ";", comments: ["//", "#"], skiprows: 1, maxRows: 2, usecols: [0, -1] });
    expect(L(a)).toEqual([[1, 3], [4, 6]]);
    expect(L(np.loadtxt(['"1,5",2'], { delimiter: ",", quotechar: '"', dtype: "int64", usecols: [1], ndmin: 1 }))).toEqual([2]);
    expect(L(np.loadtxt(["1 2", "3 4"], { usecols: 1, dtype: "int8" }))).toEqual([2, 4]);
  });
  it("dtypes: ints, bools, complex, inf/nan", () => {
    expect(L(np.loadtxt(["0 7 -1"], { dtype: "bool" }))).toEqual([false, true, true]);
    const c = np.loadtxt(["1+2j (1j) 3"], { dtype: "complex128" });
    expect(c.dtype.name).toBe("complex128");
    expect(np.loadtxt(["inf -nan 1e400 .5"]).toArray()).toEqual([Infinity, NaN, Infinity, 0.5]);
    expect(np.loadtxt(["18446744073709551615"], { dtype: "uint64" }).toArray()).toBe(2 ** 64);
  });
  it("ndmin and empty input", () => {
    expect(np.loadtxt(["1 2 3"], { ndmin: 2 }).shape).toEqual([1, 3]);
    expect(np.loadtxt(["1", "2"], { ndmin: 2 }).shape).toEqual([2, 1]);
    expect(np.loadtxt(["5"], { ndmin: 1 }).shape).toEqual([1]);
    expect(np.loadtxt([]).shape).toEqual([0]);
    expect(np.loadtxt([], { usecols: [0, 1] }).shape).toEqual([0, 2]);
    expect(np.loadtxt([], { ndmin: 2 }).shape).toEqual([0, 1]);
  });
  it("NumPy error messages", () => {
    fails(() => np.loadtxt(["1,2", "3,"], { delimiter: "," }), ValueError,
      "could not convert string '' to float64 at row 1, column 2.");
    fails(() => np.loadtxt(["1 2", "3"]), ValueError,
      "the number of columns changed from 2 to 1 at row 2; use `usecols` to select a subset and avoid this error");
    fails(() => np.loadtxt(["300"], { dtype: "int8" }), ValueError, "could not convert string '300' to int8 at row 0, column 1.");
    fails(() => np.loadtxt(["1.5"], { dtype: "int64" }), ValueError, "could not convert string '1.5' to int64 at row 0, column 1.");
    fails(() => np.loadtxt(["1 2"], { usecols: [5] }), ValueError, "invalid column index 5 at row 1 with 2 columns");
    fails(() => np.loadtxt(["1"], { ndmin: 3 }), ValueError, "Illegal value of ndmin keyword: 3");
    fails(() => np.loadtxt(["1"], { comments: "" }), ValueError,
      "comments cannot be an empty string. Use comments=None to disable comments.");
    fails(() => np.loadtxt(["1"], { skiprows: -1 }), ValueError, "argument must be nonnegative");
    expect(() => np.loadtxt(["1"], { delimiter: ",," })).toThrow(TypeError);
  });
});

describe("P14-3 savetxt (D-171)", () => {
  it("default format and options", () => {
    expect(np.savetxt(null, [[1.5, 2]])).toBe("1.500000000000000000e+00 2.000000000000000000e+00\n");
    expect(np.savetxt(null, [1, 2], { fmt: "%d" })).toBe("1\n2\n");
    expect(np.savetxt(null, [[1, 2]], { fmt: ["%d", "%.1f"], delimiter: ",", newline: ";" })).toBe("1,2.0;");
    expect(np.savetxt(null, [[1, 2]], { fmt: "%d|%d" })).toBe("1|2\n");
    expect(np.savetxt(null, [[1.5, 2]], { fmt: "%s", header: "a\nb", footer: "f", comments: "% " })).toBe(
      "% a\n% b\n1.5 2.0\n% f\n",
    );
  });
  it("%s uses NumPy scalar str; %d truncates; complex pairs", () => {
    expect(np.savetxt(null, np.array([1e16, 1e7, 0.1]).astype("float32"), { fmt: "%s" })).toBe("1e+16\n1e+07\n0.1\n");
    expect(np.savetxt(null, [1e16, 1e7, 0.5], { fmt: "%s" })).toBe("1e+16\n10000000.0\n0.5\n");
    expect(np.savetxt(null, [[true, false]], { fmt: "%s" })).toBe("True False\n");
    expect(np.savetxt(null, [-2.7], { fmt: "%d" })).toBe("-2\n");
    expect(np.savetxt(null, [[255, -3]], { fmt: "%#x|%+05d" })).toBe("0xff|-0003\n");
    expect(np.savetxt(null, np.array([[1, -2]]).astype("complex128"), { fmt: "%.1f" })).toBe(" (1.0+0.0j)  (-2.0+0.0j)\n");
  });
  it("empty arrays, files and gzip", () => {
    expect(np.savetxt(null, np.zeros([0, 2]))).toBe("");
    expect(np.savetxt(null, np.zeros([2, 0]))).toBe("\n\n");
    const p = join(dir, "s.txt");
    np.savetxt(p, [[1, 2]], { fmt: "%d" });
    expect(readFileSync(p, "utf8")).toBe("1 2\n");
    np.savetxt(p + ".gz", [[1, 2]], { fmt: "%d" });
    expect(gunzipSync(readFileSync(p + ".gz")).toString()).toBe("1 2\n");
    expect(L(np.loadtxt(p + ".gz"))).toEqual([1, 2]);
  });
  it("errors", () => {
    fails(() => np.savetxt(null, 1), ValueError, "Expected 1D or 2D array, got 0D array instead");
    fails(() => np.savetxt(null, [[1, 2]], { fmt: ["%d"] }), ValueError, "fmt has wrong shape.  ['%d']");
    fails(() => np.savetxt(null, [[1, 2]], { fmt: "%d %d %d" }), ValueError, "fmt has wrong number of % formats:  %d %d %d");
    expect(() => np.savetxt(null, [[1, 2]], { fmt: "%r" })).toThrow(ValueError);
  });
});

describe("P14-4 genfromtxt (D-171)", () => {
  const g = np.genfromtxt;
  it("fills invalid cells with dtype defaults (loose)", () => {
    expect(L(g(["1.5 2", "3 x"], { dtype: "int32" }))).toEqual([[1, 2], [3, -1]]);
    expect(L(g(["1.5 2", "3 x"], { dtype: "int64" }))).toEqual([[-1, 2], [3, -1]]);
    expect(L(g(["1 2", "3 "], { delimiter: "," }))).toEqual([NaN, 3]);
    expect(L(g(["true FALSE 1 yes"], { dtype: "bool" }))).toEqual([true, false, false, false]);
    expect(L(g(["1 1_0 inf 0x10 -nan"]))).toEqual([1, 10, Infinity, NaN, NaN]);
    const c = g(["1+2j (3j) x 1"], { dtype: "complex128" });
    expect(c.get(1).item()).toEqual(expect.objectContaining({ re: 0, im: 3 }));
    expect(Number.isNaN((c.get(2).item() as { re: number }).re)).toBe(true);
    expect(np.save(null, g(["123456789012345678"], { dtype: "int64", ndmin: 1 }))!.readBigInt64LE(128)).toBe(
      123456789012345678n,
    );
  });
  it("missingValues / fillingValues (scalar, list, map, usecols remap)", () => {
    expect(L(g(["N/A,2", ",4", "5,-"], { delimiter: ",", missingValues: "N/A,-", fillingValues: 0, loose: false })))
      .toEqual([[0, 2], [0, 4], [5, 0]]);
    expect(L(g(["N/A,2", ",4"], { delimiter: ",", missingValues: { 0: "N/A" }, fillingValues: { 1: 9, 0: 8 } })))
      .toEqual([[8, 2], [8, 4]]);
    expect(L(g(["N/A,2", ",4"], { delimiter: ",", fillingValues: [7, 8] }))).toEqual([[7, 2], [7, 4]]);
    expect(L(g(["1,", "2,3"], { delimiter: ",", fillingValues: { 1: 5 }, usecols: [1] }))).toEqual([5, 3]);
    expect(L(g(["1 2", "3 4"], { missingValues: new Map([[null, "3"]]), loose: false, fillingValues: -5 })))
      .toEqual([[1, 2], [3, 4]]);
    fails(() => g(["1 x"], { dtype: "int16", loose: false }), ValueError, "Cannot convert string 'x'");
  });
  it("skipHeader, skipFooter, usecols, maxRows, widths, ndmin, unpack", () => {
    expect(L(g(["1 2", "3 4", "5", "f"], { skipFooter: 2 }))).toEqual([[1, 2], [3, 4]]);
    expect(L(g(["h", "# c", "1 2", "3 4"], { skipHeader: 1, unpack: true }))).toEqual([[1, 3], [2, 4]]);
    expect(L(g(["1 2", "3 4"], { usecols: -1 }))).toEqual([2, 4]);
    expect(L(g(["1 2 # x", " 3 4"], { maxRows: 1 }))).toEqual([1, 2]);
    expect(L(g(["12345678"], { delimiter: [2, 3, 3], dtype: "int64" }))).toEqual([12, 345, 678]);
    expect(L(g(["12345678"], { delimiter: 3, dtype: "int64" }))).toEqual([123, 456, 78]);
    expect(g(["1,2"], { delimiter: ",", dtype: "float32", ndmin: 2 }).shape).toEqual([1, 2]);
    expect(L(g(["1 2 3", "1"], { usecols: [1], invalidRaise: false }))).toEqual(2);
  });
  it("errors", () => {
    fails(() => g(["# c", "", "1 2", "3", "4 5 6", "7 8"]), ValueError,
      "Some errors were detected !\n    Line #2 (got 1 columns instead of 2)\n    Line #3 (got 3 columns instead of 2)");
    fails(() => g(["1"], { maxRows: 0 }), ValueError, "'max_rows' must be at least 1.");
    fails(() => g(["1"], { maxRows: 1, skipFooter: 1 }), ValueError,
      "The keywords 'skip_footer' and 'max_rows' can not be specified at the same time.");
    expect(() => g(["-1 300"], { dtype: "uint8" })).toThrow(ValueError);
  });
});

describe("P14-4 fromregex (D-171)", () => {
  it("one array per field", () => {
    const r = np.fromregex(["a1 b22 c3"], /([a-z])(\d+)/, [["c", "bool"], ["n", "int64"]]);
    expect(Object.keys(r)).toEqual(["c", "n"]);
    expect(L(r.n)).toEqual([1, 22, 3]);
    expect(L(r.c)).toEqual([true, true, true]);
    expect(L(np.fromregex(Buffer.from("x=1.5;x=2"), String.raw`x=([\d.]+)`, [["x", "float32"]]).x)).toEqual([1.5, 2]);
    expect(L(np.fromregex(["a1 b22"], /\d+/, [["n", "int8"]]).n)).toEqual([1, 22]);
    expect(np.fromregex(["xx"], /(\d)/, [["n", "int64"]]).n.shape).toEqual([0]);
  });
  it("errors", () => {
    fails(() => np.fromregex(["a1"], /([a-z])(\d)/, [["n", "int64"]]), ValueError,
      "could not assign tuple of length 2 to structure with 1 fields.");
    expect(() => np.fromregex(["a1"], /(\d)/, "int64" as never)).toThrow(TypeError);
  });
});

describe("P14-5 fromfile / NDArray.tofile (D-171)", () => {
  it("binary round trip with dtype, count, offset; C order", () => {
    const b = np.arange(5).astype("int16").tofile(null)!;
    expect(b.length).toBe(10);
    expect(L(np.fromfile(b, { dtype: "int16", offset: 2, count: 2 }))).toEqual([1, 2]);
    expect(L(np.fromfile(b, { dtype: "int32" }))).toEqual([65536, 196610]);
    expect(L(np.fromfile(b, { dtype: "int16", count: 9 }))).toEqual([0, 1, 2, 3, 4]);
    const f = np.asfortranarray(np.arange(4).reshape([2, 2]));
    expect(L(np.fromfile(f.tofile(null)!, { dtype: "int64" }))).toEqual([0, 1, 2, 3]);
    const p = join(dir, "raw.bin");
    expect(np.array([1.5, -2]).tofile(p)).toBeUndefined();
    expect(L(np.fromfile(p))).toEqual([1.5, -2]);
  });
  it("text mode", () => {
    const s = (a: NDArray, o: { sep: string; format?: string }) => a.tofile(null, o)!.toString();
    expect(s(np.array([[1.5, 2], [3, 4]]), { sep: ", " })).toBe("1.5, 2.0, 3.0, 4.0");
    expect(s(np.array([1.5, 2]), { sep: ",", format: "%.2f" })).toBe("1.50,2.00");
    expect(s(np.array([true, false]), { sep: " " })).toBe("True False");
    expect(s(np.array([{ re: 1, im: 2 }, { re: -0, im: -1 }]), { sep: " " })).toBe("(1+2j) (-0-1j)");
    expect(s(np.array([0.1]).astype("float32"), { sep: " " })).toBe("0.10000000149011612");
    const p = join(dir, "t.csv");
    np.array([1, 2, 3]).tofile(p, { sep: "\n", format: "%d" });
    expect(readFileSync(p, "utf8")).toBe("1\n2\n3");
    expect(L(np.fromfile(p, { sep: "\n", dtype: "uint8" }))).toEqual([1, 2, 3]);
    expect(L(np.fromfile(Buffer.from("1, 2, 3,4"), { sep: ",", count: 2 }))).toEqual([1, 2]);
    expect(() => np.fromfile(Buffer.from("1 2"), { sep: " ", offset: 1 })).toThrow(TypeError);
  });
});
