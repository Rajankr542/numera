import { describe, expect, it } from "vitest";
import np, { ValueError, DTypeError, NDArray } from "../src/index.js";

// ---- histogram ----

describe("P10 histogram", () => {
  it("basic count", () => {
    const { hist, edges } = np.histogram([1, 2, 1, 3], 3);
    expect(hist.size).toBe(3);
    expect(edges.size).toBe(4);
    expect(Number(hist.toArray().reduce((a: number, b: unknown) => a + (b as number), 0))).toBe(4);
    expect(edges.item(0)).toBe(1);
    expect(edges.item(3)).toBe(3);
  });

  it("explicit edges", () => {
    const { hist, edges } = np.histogram([1, 2, 3, 4], [1, 2, 3, 4]);
    expect(hist.toArray()).toEqual([1, 1, 2]);
    expect(edges.size).toBe(4);
  });

  it("density integrates to 1", () => {
    const { hist, edges } = np.histogram([1, 2, 3, 4, 5], 5, { density: true });
    const ea = edges.toArray() as number[];
    const ha = hist.toArray() as number[];
    const integral = ha.reduce((sum, h, i) => sum + h * (ea[i + 1]! - ea[i]!), 0);
    expect(Math.abs(integral - 1)).toBeLessThan(1e-12);
  });

  it("range clips data", () => {
    const { hist } = np.histogram([1, 2, 3, 4, 5], 2, { range: [2, 4] });
    const total = (hist.toArray() as number[]).reduce((a, b) => a + b, 0);
    expect(total).toBe(3);
  });

  it("weights", () => {
    const { hist } = np.histogram([0, 1, 2], 3, { weights: [1, 2, 3] });
    const ea = hist.toArray() as number[];
    expect(ea[0]).toBe(1);
    expect(ea[1]).toBe(2);
    expect(ea[2]).toBe(3);
  });

  it("string estimators run", () => {
    const data = np.arange(20).astype("float64");
    for (const est of ["auto", "fd", "sturges", "scott", "rice", "doane", "sqrt"] as const) {
      const { hist } = np.histogram(data, est as never);
      expect(hist.size).toBeGreaterThan(0);
    }
  });

  it("invalid bins throws", () => {
    expect(() => np.histogram([1, 2, 3], 0)).toThrow(ValueError);
  });
});

// ---- histogramBinEdges ----

describe("P10 histogramBinEdges", () => {
  it("returns edges only", () => {
    const edges = np.histogramBinEdges([1, 2, 3, 4, 5], 4);
    expect(edges.size).toBe(5);
    expect(edges.item(0)).toBe(1);
    expect(edges.item(4)).toBe(5);
  });
});

// ---- histogram2d ----

describe("P10 histogram2d", () => {
  it("basic", () => {
    const { hist, xedges, yedges } = np.histogram2d([0, 1, 2], [0, 1, 2], 3);
    expect(hist.shape).toEqual([3, 3]);
    expect(xedges.size).toBe(4);
    expect(yedges.size).toBe(4);
    const total = (hist.toArray() as number[][]).flat().reduce((a, b) => a + b, 0);
    expect(total).toBe(3);
  });

  it("per-axis bins", () => {
    const { hist } = np.histogram2d([0, 1, 2, 3], [0, 1, 2, 3], [2, 4]);
    expect(hist.shape).toEqual([2, 4]);
  });

  it("density integrates to 1", () => {
    const { hist, xedges, yedges } = np.histogram2d(
      [0.5, 1.5, 0.5, 1.5],
      [0.5, 0.5, 1.5, 1.5],
      2,
      { density: true },
    );
    const xe = xedges.toArray() as number[];
    const ye = yedges.toArray() as number[];
    const ha = hist.toArray() as number[][];
    let integral = 0;
    for (let i = 0; i < 2; i++)
      for (let j = 0; j < 2; j++)
        integral += ha[i]![j]! * (xe[i + 1]! - xe[i]!) * (ye[j + 1]! - ye[j]!);
    expect(Math.abs(integral - 1)).toBeLessThan(1e-12);
  });
});

// ---- histogramdd ----

describe("P10 histogramdd", () => {
  it("1D sample (N,) treated as 1 variable", () => {
    const { hist, edges } = np.histogramdd([1, 2, 3, 4], 4);
    expect(hist.size).toBe(4);
    expect(edges.length).toBe(1);
  });

  it("3D sample", () => {
    const { hist, edges } = np.histogramdd(
      [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
      2,
    );
    expect(hist.ndim).toBe(3);
    expect(edges.length).toBe(3);
  });
});

// ---- bincount ----

describe("P10 bincount", () => {
  it("basic", () => {
    const r = np.bincount([1, 0, 2, 0, 1]);
    expect(r.toArray()).toEqual([2, 2, 1]);
  });

  it("minlength", () => {
    const r = np.bincount([0, 1], { minlength: 5 });
    expect(r.size).toBe(5);
    expect(r.item(4)).toBe(0);
  });

  it("weights", () => {
    const r = np.bincount([0, 1, 0], { weights: [0.5, 1.0, 1.5] });
    expect(r.dtype.name).toBe("float64");
    expect(r.item(0)).toBeCloseTo(2.0, 12);
    expect(r.item(1)).toBeCloseTo(1.0, 12);
  });

  it("negative value throws", () => {
    expect(() => np.bincount([-1, 0, 1])).toThrow(ValueError);
  });

  it("float input throws", () => {
    expect(() => np.bincount(np.array([1.5, 2.5]))).toThrow(DTypeError);
  });

  it("empty array", () => {
    const r = np.bincount([], { minlength: 3 });
    expect(r.size).toBe(3);
    expect(r.toArray()).toEqual([0, 0, 0]);
  });
});

// ---- digitize ----

describe("P10 digitize", () => {
  it("right=false (default)", () => {
    const r = np.digitize([0.2, 6.4, 3.0, 1.6], [0, 1, 2.5, 4, 10]);
    expect(r.toArray()).toEqual([1, 4, 3, 2]);
  });

  it("right=true", () => {
    const r = np.digitize([1.0, 2.5, 4.0], [1, 2, 3, 4], true);
    expect(r.toArray()).toEqual([0, 2, 3]);
  });

  it("decreasing bins right=false", () => {
    const r = np.digitize([1.2, 10, 12, 0], [10, 5, 1]);
    expect(r.toArray()).toEqual([2, 0, 0, 3]);
  });

  it("decreasing bins right=true", () => {
    const r = np.digitize([1.2, 10, 12, 0], [10, 5, 1], true);
    expect(r.toArray()).toEqual([2, 1, 0, 3]);
  });

  it("out of range", () => {
    const r = np.digitize([-1, 100], [0, 10, 20]);
    expect(r.item(0)).toBe(0);
    expect(r.item(1)).toBe(3);
  });

  it("non-monotonic bins throws", () => {
    expect(() => np.digitize([1], [1, 3, 2])).toThrow(ValueError);
  });

  it("preserves input shape", () => {
    const x = np.array([[0, 1], [2, 3]]);
    const r = np.digitize(x, [0, 1, 2, 3]);
    expect(r.shape).toEqual([2, 2]);
  });
});

// ---- interp ----

describe("P10 interp", () => {
  it("basic linear interpolation", () => {
    const r = np.interp([0, 1, 1.5, 2, 2.5, 3], [1, 2, 3], [3, 2, 0]);
    const arr = r.toArray() as number[];
    expect(arr[0]).toBeCloseTo(3.0, 12);  // left fill
    expect(arr[1]).toBeCloseTo(3.0, 12);  // at xp[0]
    expect(arr[2]).toBeCloseTo(2.5, 12);
    expect(arr[3]).toBeCloseTo(2.0, 12);
    expect(arr[4]).toBeCloseTo(1.0, 12);
    expect(arr[5]).toBeCloseTo(0.0, 12);
  });

  it("left/right fill values", () => {
    const r = np.interp([-1, 5], [0, 1, 2], [0, 1, 2], { left: -99, right: 99 });
    expect(r.item(0)).toBe(-99);
    expect(r.item(1)).toBe(99);
  });

  it("preserves 2D shape", () => {
    const x = np.array([[0.5, 1.5], [2.5, 3.5]]);
    const r = np.interp(x, [0, 1, 2, 3, 4], [0, 1, 4, 9, 16]);
    expect(r.shape).toEqual([2, 2]);
  });

  it("scalar xp,fp", () => {
    const r = np.interp(0.5, [0, 1], [0, 10]);
    expect(r.item()).toBeCloseTo(5.0, 12);
  });

  it("empty xp throws", () => {
    expect(() => np.interp([1], [], [])).toThrow(ValueError);
  });

  it("xp/fp length mismatch throws", () => {
    expect(() => np.interp([1], [0, 1], [0, 1, 2])).toThrow(ValueError);
  });
});
