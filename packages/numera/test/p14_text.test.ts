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
