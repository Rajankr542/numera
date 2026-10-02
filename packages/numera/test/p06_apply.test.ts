import { describe, expect, it } from "vitest";
import np, { IndexError, type NDArray, ValueError } from "../src/index.js";

describe("P6 applyAlongAxis / applyOverAxes", () => {
  const b = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  it("applyAlongAxis", () => {
    const mid = (v: NDArray) => ((v.get(0).item() as number) + (v.get(-1).item() as number)) / 2;
    expect(np.applyAlongAxis((v) => [mid(v)], 0, b).toArray()).toEqual([[4, 5, 6]]);
    expect(np.applyAlongAxis((v) => mid(v), 0, b).toArray()).toEqual([4, 5, 6]);
    expect(np.applyAlongAxis((v) => [mid(v)], 1, b).toArray()).toEqual([[2], [5], [8]]);
    expect(np.applyAlongAxis((v) => np.flip(v), 1, b).toArray()).toEqual([[3, 2, 1], [6, 5, 4], [9, 8, 7]]);
    const outer = np.applyAlongAxis((v) => np.stack([v, v]), -1, b);
    expect(outer.shape).toEqual([3, 2, 3]);
    expect(np.applyAlongAxis((v) => v.sum(), 0, b).toArray()).toEqual([12, 15, 18]);
    expect(np.applyAlongAxis((v, k: number) => (v.sum().item() as number) * k, 1, b, 2).toArray()).toEqual([12, 30, 48]);
    expect(np.applyAlongAxis((v) => v.sum(), 0, [1, 2, 3]).toArray()).toEqual(6);
    expect(np.applyAlongAxis((v) => [v.size, v.size], 2, np.zeros([2, 3, 4])).shape).toEqual([2, 3, 2]);
    expect(np.applyAlongAxis((v) => v.sum(), 1, np.zeros([2, 0])).toArray()).toEqual([0, 0]);
    expect(() => np.applyAlongAxis((v) => v, 1, np.zeros([0, 2]))).toThrow(/iteration dimensions are 0/);
    expect(() => np.applyAlongAxis((v) => v, 2, b)).toThrow(IndexError);
    expect(() => np.applyAlongAxis((v) => (v.get(0).item() === 1 ? [1] : [1, 2]), 1, b)).toThrow();
  });
  it("applyOverAxes", () => {
    const a = np.arange(24).reshape(2, 3, 4);
    const s = np.applyOverAxes((x, ax) => x.sum({ axis: ax }), a, [0, 2]);
    expect(s.shape).toEqual([1, 3, 1]);
    expect(s.toArray()).toEqual([[[60], [92], [124]]]);
    expect(np.applyOverAxes((x, ax) => x.sum({ axis: ax, keepdims: true }), a, -1).shape).toEqual([2, 3, 1]);
    expect(() => np.applyOverAxes((x) => x.sum(), a, 0)).toThrow(ValueError);
  });
});
