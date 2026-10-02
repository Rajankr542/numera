import { describe, expect, it } from "vitest";
import np, { ValueError } from "../src/index.js";

describe("P10 correlate", () => {
  it("full mode", () => {
    // np.correlate([1,2,3], [0,1,0.5], 'full') = [0.5, 2.0, 3.5, 3.0, 0.0]
    const r = np.correlate([1, 2, 3], [0, 1, 0.5], "full");
    expect(r.toArray()).toEqual([0.5, 2.0, 3.5, 3.0, 0.0]);
  });

  it("same mode", () => {
    // np.correlate([1,2,3], [0,1,0.5], 'same') = [2.0, 3.5, 3.0]
    const r = np.correlate([1, 2, 3], [0, 1, 0.5], "same");
    expect(r.toArray()).toEqual([2.0, 3.5, 3.0]);
  });

  it("valid mode (default)", () => {
    // np.correlate([1,2,3], [0,1,0.5]) = [3.5]
    const r = np.correlate([1, 2, 3], [0, 1, 0.5]);
    expect(r.toArray()).toEqual([3.5]);
  });

  it("valid when v longer than a", () => {
    // np.correlate([1,2], [1,2,3], 'valid') = [8, 5]
    const r = np.correlate([1, 2], [1, 2, 3], "valid");
    expect(r.toArray()).toEqual([8, 5]);
  });

  it("same when v longer than a", () => {
    // np.correlate([1,2], [1,2,3], 'same') = [8, 5, 2]
    const r = np.correlate([1, 2], [1, 2, 3], "same");
    expect(r.toArray()).toEqual([8, 5, 2]);
  });

  it("full when v longer than a", () => {
    // np.correlate([1,2], [1,2,3], 'full') = [3, 8, 5, 2]
    const r = np.correlate([1, 2], [1, 2, 3], "full");
    expect(r.toArray()).toEqual([3, 8, 5, 2]);
  });

  it("equal length arrays", () => {
    // np.correlate([1,2,3], [4,5,6], 'full') = [6, 17, 32, 23, 12]
    expect(np.correlate([1, 2, 3], [4, 5, 6], "full").toArray()).toEqual([6, 17, 32, 23, 12]);
    expect(np.correlate([1, 2, 3], [4, 5, 6], "same").toArray()).toEqual([17, 32, 23]);
    expect(np.correlate([1, 2, 3], [4, 5, 6], "valid").toArray()).toEqual([32]);
  });

  it("output length rules", () => {
    const M = 5;
    const K = 3;
    const a = np.ones(M);
    const v = np.ones(K);
    expect(np.correlate(a, v, "full").size).toBe(M + K - 1);
    expect(np.correlate(a, v, "same").size).toBe(Math.max(M, K));
    expect(np.correlate(a, v, "valid").size).toBe(Math.abs(M - K) + 1);
  });

  it("dtype preservation — int", () => {
    const a = np.array([1, 2, 3], { dtype: "int32" });
    const v = np.array([1, 2], { dtype: "int32" });
    const r = np.correlate(a, v, "full");
    expect(r.dtype.name).toBe("int32");
  });

  it("dtype preservation — float32", () => {
    const a = np.array([1, 2, 3], { dtype: "float32" });
    const v = np.array([1, 2], { dtype: "float32" });
    const r = np.correlate(a, v, "full");
    expect(r.dtype.name).toBe("float32");
  });

  it("complex correlate", () => {
    // np.correlate([1+1j, 2, 3], [1+1j, 2], 'full') = [2+2j, 6+0j, 8-2j, 3-3j]
    const a = np.array([{ re: 1, im: 1 }, { re: 2, im: 0 }, { re: 3, im: 0 }], { dtype: "complex128" });
    const v = np.array([{ re: 1, im: 1 }, { re: 2, im: 0 }], { dtype: "complex128" });
    const r = np.correlate(a, v, "full");
    expect(r.size).toBe(4);
    const arr = r.toArray() as Array<{ re: number; im: number }>;
    expect(arr[0]!.re).toBeCloseTo(2);  expect(arr[0]!.im).toBeCloseTo(2);
    expect(arr[1]!.re).toBeCloseTo(6);  expect(arr[1]!.im).toBeCloseTo(0);
    expect(arr[2]!.re).toBeCloseTo(8);  expect(arr[2]!.im).toBeCloseTo(-2);
    expect(arr[3]!.re).toBeCloseTo(3);  expect(arr[3]!.im).toBeCloseTo(-3);
  });

  it("bad mode throws", () => {
    expect(() => np.correlate([1, 2], [1], "bad" as never)).toThrow(ValueError);
  });
});

describe("P10 convolve", () => {
  it("full mode (default)", () => {
    // np.convolve([1,2,3], [0,1,0.5]) = [0., 1., 2.5, 4., 1.5]
    const r = np.convolve([1, 2, 3], [0, 1, 0.5]);
    expect(r.toArray()).toEqual([0.0, 1.0, 2.5, 4.0, 1.5]);
  });

  it("same mode", () => {
    // np.convolve([1,2,3], [0,1,0.5], 'same') = [1., 2.5, 4.]
    const r = np.convolve([1, 2, 3], [0, 1, 0.5], "same");
    expect(r.toArray()).toEqual([1.0, 2.5, 4.0]);
  });

  it("valid mode", () => {
    // np.convolve([1,2,3], [0,1,0.5], 'valid') = [2.5]
    const r = np.convolve([1, 2, 3], [0, 1, 0.5], "valid");
    expect(r.toArray()).toEqual([2.5]);
  });

  it("v reversed relative to correlate", () => {
    // convolve(a, v) = correlate(a, v[::-1]) for real
    const a = [1, 2, 3, 4];
    const v = [1, 2, 3];
    // np.convolve([1,2,3,4],[1,2,3]) = [1, 4, 10, 16, 17, 12]
    expect(np.convolve(a, v, "full").toArray()).toEqual([1, 4, 10, 16, 17, 12]);
  });

  it("single-element", () => {
    expect(np.convolve([3.0], [2.0]).toArray()).toEqual([6.0]);
  });

  it("output length rules", () => {
    const M = 6;
    const K = 3;
    const a = np.ones(M);
    const v = np.ones(K);
    expect(np.convolve(a, v, "full").size).toBe(M + K - 1);
    expect(np.convolve(a, v, "same").size).toBe(Math.max(M, K));
    expect(np.convolve(a, v, "valid").size).toBe(Math.abs(M - K) + 1);
  });

  it("complex convolve", () => {
    // np.convolve([1+1j, 2+2j], [1, 1j]) = [1+1j, 1+3j, -2+2j]
    const a = np.array([{ re: 1, im: 1 }, { re: 2, im: 2 }], { dtype: "complex128" });
    const v = np.array([{ re: 1, im: 0 }, { re: 0, im: 1 }], { dtype: "complex128" });
    const r = np.convolve(a, v, "full");
    expect(r.size).toBe(3);
    const arr = r.toArray() as Array<{ re: number; im: number }>;
    expect(arr[0]!.re).toBeCloseTo(1);  expect(arr[0]!.im).toBeCloseTo(1);
    expect(arr[1]!.re).toBeCloseTo(1);  expect(arr[1]!.im).toBeCloseTo(3);
    expect(arr[2]!.re).toBeCloseTo(-2); expect(arr[2]!.im).toBeCloseTo(2);
  });

  it("bad mode throws", () => {
    expect(() => np.convolve([1, 2], [1], "bad" as never)).toThrow(ValueError);
  });
});
