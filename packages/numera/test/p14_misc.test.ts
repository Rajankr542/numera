import { describe, expect, it } from "vitest";
import np, { NDArray, ValueError } from "../src/index.js";

const L = (a: unknown): unknown => (a as NDArray).toArray();

describe("P14-6 baseRepr / binaryRepr (D-172)", () => {
  it("binaryRepr matches NumPy, including two's complement", () => {
    expect(np.binaryRepr(5)).toBe("101");
    expect(np.binaryRepr(-5)).toBe("-101");
    expect(np.binaryRepr(-5, { width: 8 })).toBe("11111011");
    expect(np.binaryRepr(-128, { width: 8 })).toBe("10000000");
    expect(np.binaryRepr(0)).toBe("0");
    expect(np.binaryRepr(0, { width: 4 })).toBe("0000");
    expect(np.binaryRepr(-1, { width: 1 })).toBe("1");
    expect(np.binaryRepr(true)).toBe("1");
    expect(np.binaryRepr(np.array(-3).astype("int8"))).toBe("-11");
    expect(np.binaryRepr(2n ** 64n - 1n)).toBe("1".repeat(64));
    expect(() => np.binaryRepr(5, { width: 2 })).toThrow(new ValueError("Insufficient bit width=2 provided for binwidth=3"));
    expect(() => np.binaryRepr(-128, { width: 7 })).toThrow(ValueError);
    expect(() => np.binaryRepr(2.5)).toThrow(TypeError);
  });
  it("baseRepr", () => {
    expect(np.baseRepr(255, 16)).toBe("FF");
    expect(np.baseRepr(-255, 16, 3)).toBe("-000FF");
    expect(np.baseRepr(0, 2, 2)).toBe("00");
    expect(np.baseRepr(35, 36)).toBe("Z");
    expect(np.baseRepr(2n ** 70n, 36)).toBe("6X5KXTVUWILUKG");
    expect(np.baseRepr(5)).toBe("101");
    expect(np.baseRepr(np.array(7), 2)).toBe("111");
    expect(() => np.baseRepr(10, 37)).toThrow(ValueError);
    expect(() => np.baseRepr(10, 1)).toThrow(ValueError);
  });
});

describe("P14-6 window functions (D-172)", () => {
  it("bartlett blackman hamming hanning (NumPy values)", () => {
    expect(L(np.bartlett(4))).toEqual([0, 0.6666666666666667, 0.6666666666666667, 0]);
    expect(L(np.blackman(5))).toEqual([-1.3877787807814457e-17, 0.34, 0.9999999999999999, 0.34, -1.3877787807814457e-17]);
    expect(L(np.hamming(5))).toEqual([0.08000000000000002, 0.54, 1, 0.54, 0.08000000000000002]);
    expect(L(np.hanning(4))).toEqual([0, 0.75, 0.75, 0]);
    expect(L(np.bartlett(5.5))).toEqual([0, 0.4444444444444444, 0.8888888888888888, 0.6666666666666667, 0.2222222222222222]);
    for (const f of [np.bartlett, np.blackman, np.hamming, np.hanning]) {
      expect(f(0).shape).toEqual([0]);
      expect(f(-1).shape).toEqual([0]);
      expect(L(f(1))).toEqual([1]);
      expect(f(3).dtype.name).toBe("float64");
    }
  });
  it("kaiser", () => {
    expect(L(np.kaiser(4, 5))).toEqual([0.036710892271286676, 0.7753221044454067, 0.7753221044454067, 0.036710892271286676]);
    expect(L(np.kaiser(3, 0))).toEqual([1, 1, 1]);
    expect(L(np.kaiser(1, 5))).toEqual([1]);
    expect(np.kaiser(0, 5).shape).toEqual([0]);
    const k = np.kaiser(20, 30).toArray() as number[];
    expect(k[0]).toBeCloseTo(1.279308481039694e-12, 25);
    expect(k[9]).toBeCloseTo(0.9599437470130808, 14);
  });
});
