import { afterEach, describe, expect, it } from "vitest";
import np, { DTypeError, NotImplementedError, ValueError } from "../src/index.js";

const P = np.formatFloatPositional;
const S = np.formatFloatScientific;

afterEach(() => np.setPrintoptions({ precision: 8, threshold: 1000, edgeitems: 3, linewidth: 75, suppress: false, nanstr: "nan", infstr: "inf", sign: "-", floatmode: "maxprec" }));

describe("P3-5 formatFloatPositional / formatFloatScientific (D-063)", () => {
  it("matches NumPy dragon4 positional output", () => {
    expect(P(0.5, { padLeft: 2 })).toBe(" 0.5");
    expect(P(0.0001, { precision: 2 })).toBe("0.00");
    expect(P(1)).toBe("1.");
    expect(P(1, { trim: "-" })).toBe("1");
    expect(P(1, { trim: "0" })).toBe("1.0");
    expect(P(1e20)).toBe("100000000000000000000.");
    expect(P(1.5, { padRight: 4 })).toBe("1.5   ");
    expect(P(1, { trim: "-", padRight: 3 })).toBe("1    ");
    expect(P(-0)).toBe("-0.");
    expect(P(0, { sign: true })).toBe("+0.");
    expect(P(2.5, { precision: 0 })).toBe("2.");
    expect(P(3.5, { precision: 0 })).toBe("4.");
    expect(P(np.array(0.1, { dtype: "float32" }), { unique: false, precision: 10 })).toBe("0.1000000015");
    expect(P(np.array(0.1, { dtype: "float32" }))).toBe("0.1");
    expect(P(np.array(0.1, { dtype: "float16" }))).toBe("0.1");
    expect(P(1234.5678, { precision: 3, fractional: false })).toBe("1230.");
    expect(P(1.5, { minDigits: 4 })).toBe("1.5000");
    expect(P(0.3, { minDigits: 20 })).toBe("0.29999999999999998890");
    expect(P(Infinity)).toBe("inf");
    expect(P(-Infinity)).toBe("-inf");
    expect(P(NaN)).toBe("nan");
    expect(P(true)).toBe("1.");
    expect(P(np.array(3, { dtype: "int8" }))).toBe("3.");
  });

  it("matches NumPy dragon4 scientific output", () => {
    expect(S(1)).toBe("1.e+00");
    expect(S(1, { trim: "-" })).toBe("1e+00");
    expect(S(1, { trim: "0" })).toBe("1.0e+00");
    expect(S(123.456, { precision: 2 })).toBe("1.23e+02");
    expect(S(1e100, { expDigits: 4 })).toBe("1.e+0100");
    expect(S(1e-5, { expDigits: 1 })).toBe("1.e-5");
    expect(S(-1.5, { padLeft: 3 })).toBe(" -1.5e+00");
    expect(S(9.9999, { precision: 2 })).toBe("1.e+01");
    expect(S(1.5, { expDigits: 0 })).toBe("1.5e+");
    expect(S(Infinity)).toBe("inf");
    expect(S(np.array(0.1, { dtype: "float16" }))).toBe("1.e-01");
    expect(S(5e-324)).toBe("5.e-324");
  });

  it("validates arguments like NumPy", () => {
    expect(() => P(1.5, { trim: "x" as never })).toThrow(DTypeError);
    expect(() => P(1.5, { unique: false })).toThrow(/precision. must be supplied/);
    expect(() => P(1.5, { precision: -1 })).toThrow(ValueError);
    expect(() => P(1.5, { precision: 0, fractional: false })).toThrow(/greater than 0/);
    expect(() => P(1.5, { precision: 2, minDigits: 3 })).toThrow(/min_digits/);
    expect(() => S(1.5, { padLeft: -1 })).toThrow(/pad_left must be >= 0/);
    expect(() => P(np.array([1, 2]))).toThrow(ValueError);
    expect(() => P(np.array({ re: 1, im: 1 }))).toThrow(DTypeError);
  });
});

describe("P3-5 arrayRepr / arrayStr / toString (D-063)", () => {
  it("repr matches NumPy for common arrays", () => {
    expect(String(np.array([1.5, -2, NaN, Infinity]))).toBe("array([ 1.5, -2. ,  nan,  inf])");
    expect(String(np.array([1e-5, 1]))).toBe("array([1.e-05, 1.e+00])");
    expect(String(np.array([{ re: 1, im: 2 }, { re: -0, im: -3.5 }]))).toBe("array([ 1.+2.j , -0.-3.5j])");
    expect(String(np.zeros([2, 0]))).toBe("array([], shape=(2, 0), dtype=float64)");
    expect(String(np.array([], { dtype: "bool" }))).toBe("array([], dtype=bool)");
    expect(String(np.array([true, false]))).toBe("array([ True, False])");
    expect(String(np.array([-1, 10], { dtype: "int8" }))).toBe("array([-1, 10], dtype=int8)");
    expect(String(np.array([[1, 2], [3, 4]]))).toBe("array([[1, 2],\n       [3, 4]])");
    expect(String(np.array(1.5))).toBe("array(1.5)");
    expect(String(np.array(1, { dtype: "float32" }))).toBe("array(1., dtype=float32)");
    expect(String(np.array([1000], { dtype: "float16" }))).toBe("array([1.e+03], dtype=float16)");
    expect(String(np.array([1e6, 1], { dtype: "float32" }))).toBe("array([1.e+06, 1.e+00], dtype=float32)");
    expect(String(np.array([1, 1001], { dtype: "float64" }))).toBe("array([1.000e+00, 1.001e+03])");
    expect(np.arrayRepr(np.array([1, 2]))).toBe("array([1, 2])");
  });

  it("summarizes large arrays and wraps lines", () => {
    const a = np.array(Array.from({ length: 2000 }, (_, i) => i));
    expect(String(a)).toBe("array([   0,    1,    2, ..., 1997, 1998, 1999], shape=(2000,))");
    expect(np.arrayStr(a.reshape(2, 1000))).toBe(
      "[[   0    1    2 ...  997  998  999]\n [1000 1001 1002 ... 1997 1998 1999]]",
    );
    expect(String(np.linspace(0, 1, 6))).toBe("array([0. , 0.2, 0.4, 0.6, 0.8, 1. ])");
    expect(np.arrayRepr(np.zeros([2, 2, 2]))).toBe(
      "array([[[0., 0.],\n        [0., 0.]],\n\n       [[0., 0.],\n        [0., 0.]]])",
    );
  });

  it("str of arrays and 0-d arrays", () => {
    expect(np.arrayStr(np.array([1, 2, 3]))).toBe("[1 2 3]");
    expect(np.arrayStr(np.array({ re: 1, im: 2 }))).toBe("(1+2j)");
    expect(np.arrayStr(np.array(2))).toBe("2");
    expect(np.arrayStr(np.array(1e16))).toBe("1e+16");
    expect(np.arrayStr(np.array(1, { dtype: "float32" }))).toBe("1.0");
    expect(np.arrayStr(np.array(NaN))).toBe("nan");
    expect(np.arrayStr(np.array(true))).toBe("True");
  });
});

describe("P3-5 array2string and print options (D-063)", () => {
  it("array2string options", () => {
    const x = np.array([1, 2, 3]);
    expect(np.array2string(x, { separator: "," })).toBe("[1,2,3]");
    expect(np.array2string(np.array(Array.from({ length: 10 }, (_, i) => i)), { threshold: 5, edgeitems: 2 })).toBe(
      "[0 1 ... 8 9]",
    );
    expect(np.array2string(np.array([1, 2], { dtype: "float64" }), { sign: "+" })).toBe("[+1. +2.]");
    expect(np.array2string(np.array([1, 2], { dtype: "float64" }), { sign: " " })).toBe("[ 1.  2.]");
    expect(np.array2string(np.array([1.123, 2]), { floatmode: "fixed", precision: 2 })).toBe("[1.12 2.00]");
    expect(np.array2string(np.array([1.5, 2]), { floatmode: "maxprec_equal" })).toBe("[1.5 2.0]");
    expect(np.array2string(np.array([1e-10, 1]), { suppressSmall: true })).toBe("[0. 1.]");
    expect(np.array2string(np.array([0, 1, 2], { dtype: "float64" }), { formatter: { float_kind: (v: number) => v.toFixed(2) } })).toBe(
      "[0.00 1.00 2.00]",
    );
    expect(np.array2string(np.array([1, 2]), { formatter: { all: () => "x" } })).toBe("[x x]");
    expect(np.array2string(np.array([]))).toBe("[]");
    expect(np.array2string([1, 2])).toBe("[1 2]");
    expect(() => np.array2string(x, { formatter: { int: () => 1 as never } })).toThrow(DTypeError);
  });

  it("setPrintoptions / getPrintoptions / printoptions", () => {
    const o = np.getPrintoptions();
    expect(o).toMatchObject({ precision: 8, threshold: 1000, edgeitems: 3, linewidth: 75, floatmode: "maxprec", sign: "-", legacy: false });
    np.setPrintoptions({ precision: 2 });
    expect(String(np.array([1 / 3]))).toBe("array([0.33])");
    expect(np.getPrintoptions().precision).toBe(2);
    np.setPrintoptions({ precision: 8 });
    const r = np.printoptions({ precision: 2, suppress: true, threshold: 5 }, () => String(np.linspace(0, 10, 10)));
    expect(r).toBe("array([ 0.  ,  1.11,  2.22, ...,  7.78,  8.89, 10.  ], shape=(10,))");
    expect(np.getPrintoptions().precision).toBe(8);
    expect(() =>
      np.printoptions({ precision: 1 }, () => {
        throw new Error("boom");
      }),
    ).toThrow("boom");
    expect(np.getPrintoptions().precision).toBe(8);
    np.setPrintoptions({ nanstr: "NaN", infstr: "Inf" });
    expect(String(np.array([NaN, -Infinity, 1]))).toBe("array([ NaN, -Inf,   1.])");
    np.setPrintoptions({ overrideRepr: () => "custom" });
    expect(String(np.array([1]))).toBe("custom");
    np.setPrintoptions({});
    expect(String(np.array([1]))).toBe("array([1])");
  });

  it("validates print options", () => {
    expect(() => np.setPrintoptions({ floatmode: "x" as never })).toThrow(/floatmode option must be one of/);
    expect(() => np.setPrintoptions({ sign: "x" as never })).toThrow(/sign option must be one of/);
    expect(() => np.setPrintoptions({ threshold: NaN })).toThrow(ValueError);
    expect(() => np.setPrintoptions({ precision: 1.5 })).toThrow(DTypeError);
    expect(() => np.setPrintoptions({ legacy: "1.13" as never })).toThrow(NotImplementedError);
  });
});
