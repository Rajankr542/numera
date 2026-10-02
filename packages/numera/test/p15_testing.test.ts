import { describe, expect, it } from "vitest";
import np, { AssertionError } from "../src/index.js";

const t = np.testing;
const msgOf = (f: () => unknown): string => {
  try {
    f();
  } catch (e) {
    expect(e).toBeInstanceOf(AssertionError);
    return (e as Error).message;
  }
  throw new Error("expected an AssertionError");
};

describe("np.testing (P15-2, D-181)", () => {
  it("assertArrayEqual passes and reports mismatches as NumPy", () => {
    t.assertArrayEqual([1, 2, NaN], [1, 2, NaN]);
    t.assertArrayEqual([1, 1], 1);
    t.assertArrayEqual(np.zeros([0, 2]), np.zeros([0, 2]));
    expect(msgOf(() => t.assertArrayEqual([1, 2, 3], [1, 2, 4]))).toBe(
      "\nArrays are not equal\n\nMismatched elements: 1 / 3 (33.3%)\nMismatch at index:\n [2]: 3 (ACTUAL), 4 (DESIRED)\n" +
        "Max absolute difference among violations: 1\nMax relative difference among violations: 0.25\n" +
        " ACTUAL: array([1, 2, 3])\n DESIRED: array([1, 2, 4])",
    );
    expect(msgOf(() => t.assertArrayEqual([1, 2], [1, 2, 3]))).toBe(
      "\nArrays are not equal\n\n(shapes (2,), (3,) mismatch)\n ACTUAL: array([1, 2])\n DESIRED: array([1, 2, 3])",
    );
    expect(msgOf(() => t.assertArrayEqual([1, NaN], [1, 2]))).toContain("nan location mismatch:");
    expect(msgOf(() => t.assertArrayEqual([Infinity], [-Infinity]))).toContain("inf values mismatch:");
    expect(msgOf(() => t.assertArrayEqual([1, 2], [1.0, 2.5], { errMsg: "hello" }))).toContain(
      "Arrays are not equal\nhello\nMismatched",
    );
    expect(msgOf(() => t.assertArrayEqual(np.array([1, 2]), np.array([1, 2], { dtype: "float32" }), { strict: true }))).toContain(
      "(dtypes int64, float32 mismatch)",
    );
    expect(msgOf(() => t.assertArrayEqual([1, 2], 1, { strict: true }))).toContain("(shapes (2,), () mismatch)");
    expect(msgOf(() => t.assertArrayEqual([true, false], [true, true]))).toBe(
      "\nArrays are not equal\n\nMismatched elements: 1 / 2 (50%)\nMismatch at index:\n [1]: False (ACTUAL), True (DESIRED)\n" +
        " ACTUAL: array([ True, False])\n DESIRED: array([ True,  True])",
    );
    expect(msgOf(() => t.assertArrayEqual([1, 2], [1, 3], { verbose: false }))).not.toContain("ACTUAL: array");
  });

  it("formats floats, complex, many mismatches and summarised arrays", () => {
    expect(msgOf(() => t.assertArrayEqual(np.zeros(2000), np.ones(2000)))).toContain(
      "First 5 mismatches are at indices:\n [0]: 0.0 (ACTUAL), 1.0 (DESIRED)",
    );
    expect(msgOf(() => t.assertArrayEqual(np.zeros(2000), np.ones(2000)))).toContain(
      " ACTUAL: array([0., 0., 0., ..., 0., 0., 0.], shape=(2000,))",
    );
    expect(msgOf(() => t.assertArrayEqual([{ re: 1, im: 2 }], [{ re: 1, im: 3 }]))).toContain(
      " [0]: (1+2j) (ACTUAL), (1+3j) (DESIRED)\nMax absolute difference among violations: 1.\nMax relative difference among violations: 0.31622777",
    );
    expect(msgOf(() => t.assertArrayEqual(np.array([1, 2], { dtype: "uint8" }), np.array([2, 1], { dtype: "uint8" })))).toContain(
      "Max absolute difference among violations: 1\nMax relative difference among violations: 1.",
    );
  });

  it("assertAllclose / assertArrayAlmostEqual / assertArrayLess", () => {
    t.assertAllclose([1, 2], [1, 2 + 1e-8]);
    t.assertAllclose([1, NaN], [1, NaN]);
    t.assertAllclose([1, 1.1], [1, 1], { rtol: 0.2 });
    t.assertAllclose([], []);
    expect(msgOf(() => t.assertAllclose([1, 2], [1, 2.1]))).toContain(
      "Not equal to tolerance rtol=1e-07, atol=0\n\nMismatched elements: 1 / 2 (50%)",
    );
    expect(msgOf(() => t.assertAllclose([1, NaN], [1, NaN], { equalNan: false }))).toContain("Mismatched elements: 1 / 2");
    t.assertArrayAlmostEqual([1, 2], [1, 2.0000001]);
    expect(msgOf(() => t.assertArrayAlmostEqual([1, 2], [1, 2.001]))).toContain(
      "Arrays are not almost equal to 6 decimals",
    );
    t.assertArrayAlmostEqual([1, 2], [1, 2.01], { decimal: 1 });
    t.assertArrayLess([1, 1], [2, 2]);
    expect(msgOf(() => t.assertArrayLess([1, 3], [2, 2]))).toBe(
      "\nArrays are not strictly ordered `x < y`\n\nMismatched elements: 1 / 2 (50%)\nMismatch at index:\n [1]: 3 (x), 2 (y)\n" +
        "Max absolute difference among violations: 1\nMax relative difference among violations: 0.5\n x: array([1, 3])\n y: array([2, 2])",
    );
    expect(() => t.assertArrayLess([1, Infinity], [2, Infinity])).toThrow(AssertionError);
  });

  it("assertEqual on scalars, arrays, objects", () => {
    t.assertEqual(1, 1);
    t.assertEqual(NaN, NaN);
    t.assertEqual("a", "a");
    t.assertEqual({ a: [1, 2] }, { a: [1, 2] });
    t.assertEqual(np.array([1, 2]), [1, 2]);
    t.assertEqual({ re: 1, im: 2 }, { re: 1, im: 2 });
    expect(msgOf(() => t.assertEqual(1, 2))).toBe("\nItems are not equal:\n ACTUAL: 1\n DESIRED: 2");
    expect(msgOf(() => t.assertEqual([1, 2], [1, 3]))).toBe("\nItems are not equal:\nitem=1\n\n ACTUAL: 2\n DESIRED: 3");
    expect(msgOf(() => t.assertEqual({ a: 1 }, { a: 2 }))).toBe("\nItems are not equal:\nkey='a'\n\n ACTUAL: 1\n DESIRED: 2");
    expect(msgOf(() => t.assertEqual(0, -0))).toBe("\nItems are not equal:\n ACTUAL: 0\n DESIRED: -0.0");
    expect(msgOf(() => t.assertEqual("a", "b"))).toBe("\nItems are not equal:\n ACTUAL: 'a'\n DESIRED: 'b'");
    expect(() => t.assertEqual({ re: 1, im: 2 }, { re: 1, im: 3 })).toThrow(AssertionError);
  });

  it("assertAlmostEqual / assertApproxEqual", () => {
    t.assertAlmostEqual(1, 1 + 1e-8);
    t.assertAlmostEqual(NaN, NaN);
    t.assertAlmostEqual(Infinity, Infinity);
    t.assertAlmostEqual([1, 2], [1, 2 + 1e-7]);
    t.assertAlmostEqual({ re: 1, im: 2 }, { re: 1, im: 2 + 1e-9 });
    expect(msgOf(() => t.assertAlmostEqual(1, 1.1))).toBe(
      "\nArrays are not almost equal to 7 decimals\n ACTUAL: 1\n DESIRED: 1.1",
    );
    expect(msgOf(() => t.assertAlmostEqual({ re: 1, im: 1 }, { re: 1, im: 2 }))).toBe(
      "\nArrays are not almost equal to 7 decimals\n ACTUAL: (1+1j)\n DESIRED: (1+2j)",
    );
    expect(() => t.assertAlmostEqual(1, 1.01, { decimal: 1 })).not.toThrow();
    expect(() => t.assertAlmostEqual(NaN, 1)).toThrow(AssertionError);
    t.assertApproxEqual(1.0, 1.00000001);
    t.assertApproxEqual(1234.5, 1234.6, { significant: 4 });
    expect(msgOf(() => t.assertApproxEqual(1, 1.1))).toBe(
      "\nItems are not equal to 7 significant digits:\n ACTUAL: 1.0\n DESIRED: 1.1",
    );
    expect(() => t.assertApproxEqual(NaN, 1)).toThrow(AssertionError);
    t.assertApproxEqual(NaN, NaN);
  });

  it("ULP assertions", () => {
    const one = 1 + 2 ** -52;
    expect(t.assertArrayMaxUlp([1.0], [one]).toArray()).toEqual([[1]]);
    expect(msgOf(() => t.assertArrayMaxUlp([1.0], [1 + 1e-15]))).toBe(
      "Arrays are not almost equal up to 1 ULP (max difference is 5 ULP)",
    );
    expect(t.assertArrayMaxUlp(np.array([1], { dtype: "float32" }), np.array([1.0001], { dtype: "float32" }), { maxulp: 1000 }).dtype.name).toBe(
      "float32",
    );
    expect(t.assertArrayMaxUlp([0], [-0]).toArray()).toEqual([[0]]);
    expect(() => t.assertArrayMaxUlp([1, 2], [1])).toThrow(np.ValueError);
    expect(() => t.assertArrayMaxUlp([{ re: 1, im: 0 }], [1])).toThrow(np.NotImplementedError);
    t.assertArrayAlmostEqualNulp([1.0], [one]);
    expect(msgOf(() => t.assertArrayAlmostEqualNulp([1.0], [1 + 1e-15]))).toBe("Arrays are not equal to 1 ULP (max is 5)");
    t.assertArrayAlmostEqualNulp([1.0], [1 + 1e-15], 5);
    expect(msgOf(() => t.assertArrayAlmostEqualNulp([{ re: 1, im: 0 }], [{ re: 2, im: 0 }]))).toBe("Arrays are not equal to 1 ULP");
  });

  it("assertRaises / assertRaisesRegex / assert_", () => {
    const boom = () => {
      throw new np.ValueError("boom");
    };
    expect(t.assertRaises(np.ValueError, boom)).toBeInstanceOf(np.ValueError);
    expect(t.assertRaises(np.ShapeError, (x: number) => np.reshape(np.zeros(x), [5]), 3)).toBeInstanceOf(np.ShapeError);
    function noop() {}
    expect(msgOf(() => t.assertRaises(np.ValueError, noop))).toBe("ValueError not raised by noop");
    expect(() => t.assertRaises(np.DTypeError, boom)).toThrow(np.ValueError);
    t.assertRaisesRegex(np.ValueError, "bo+m", boom);
    expect(msgOf(() => t.assertRaisesRegex(np.ValueError, "xx", boom))).toBe('"xx" does not match "boom"');
    t.assert_(true);
    expect(msgOf(() => t.assert_(0, "msg"))).toBe("msg");
    expect(msgOf(() => t.assert_(false, () => "lazy"))).toBe("lazy");
  });

  it("assertWarns / assertNoWarnings capture process warnings", () => {
    const r = t.assertWarns("RuntimeWarning", () => np.divide([1], [0]));
    expect(r.toArray()).toEqual([Infinity]);
    t.assertWarns(null, () => process.emitWarning("x", "UserWarning"));
    function quiet() {
      return 1;
    }
    expect(msgOf(() => t.assertWarns("RuntimeWarning", quiet))).toBe("No warning raised when calling quiet");
    expect(t.assertNoWarnings(quiet)).toBe(1);
    expect(msgOf(() => t.assertNoWarnings(() => process.emitWarning("w", "UserWarning")))).toContain("Got warnings when calling");
  });

  it("assertStringEqual, buildErrMsg, printAssertEqual", () => {
    t.assertStringEqual("abc", "abc");
    expect(msgOf(() => t.assertStringEqual("abc", "abd"))).toBe("Differences in strings:\n- abc+ abd");
    expect(t.buildErrMsg([np.array([1, 2]), 3], "msg")).toBe("\nItems are not equal: msg\n ACTUAL: array([1, 2])\n DESIRED: 3");
    t.printAssertEqual("t", [1, { a: 2 }], [1, { a: 2 }]);
    expect(msgOf(() => t.printAssertEqual("t", [1], [2]))).toBe("t failed\nACTUAL: \n[1]\nDESIRED: \n[2]\n");
  });

  it("assertArrayCompare with a custom comparison", () => {
    t.assertArrayCompare((x, y) => np.lessEqual(x, y), [1, 2], [1, 3]);
    expect(msgOf(() => t.assertArrayCompare((x, y) => np.lessEqual(x, y), [2], [1], { header: "custom" }))).toContain("\ncustom\n");
  });
});
