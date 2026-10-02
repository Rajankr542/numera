import { describe, expect, it } from "vitest";
import np, { BroadcastError, DTypeError, IndexError, NDArray, ValueError } from "../src/index.js";

const parts = (a: NDArray): number[] => Array.from(a.astype("complex128").toTypedArray() as Float64Array);
const reals = (a: NDArray): number[] => Array.from(a.astype("float64").toTypedArray() as Float64Array);
const closeTo = (got: number[], exp: number[], tol = 1e-12): void => {
  expect(got.length).toBe(exp.length);
  got.forEach((x, i) => expect(Math.abs(x - exp[i]!)).toBeLessThanOrEqual(tol * (1 + Math.abs(exp[i]!))));
};

describe("np.fft out= (P12-1, D-150)", () => {
  it("writes into out and returns the same object, positional or options", () => {
    const out = np.zeros([4], { dtype: "complex128" });
    expect(np.fft.fft([1, 2, 3, 4], { out })).toBe(out);
    closeTo(parts(out), [10, 0, -2, 2, -2, 0, -2, -2]);
    const o2 = np.zeros([4], { dtype: "complex64" });
    expect(np.fft.ifft([4, 0, 0, 0], null, -1, null, o2)).toBe(o2);
    closeTo(parts(o2), [1, 0, 1, 0, 1, 0, 1, 0]);
    const o3 = np.zeros([3]);
    expect(() => np.fft.rfft([1, 2, 3, 4], { out: o3 })).toThrow(DTypeError);
  });

  it("casts the result to out.dtype under same_kind", () => {
    const x = np.array([0.5, -1, 2, 3, -4, 1]);
    const c64 = np.zeros([6], { dtype: "complex64" });
    np.fft.fft(x, { out: c64 });
    closeTo(parts(c64), parts(np.fft.fft(x)), 1e-6);
    const f16 = np.zeros([10], { dtype: "float16" });
    np.fft.irfft(x, { out: f16 });
    closeTo(reals(f16), reals(np.fft.irfft(x)), 2e-3);
    expect(() => np.fft.irfft(x, { out: np.zeros([10], { dtype: "int64" }) })).toThrow(DTypeError);
    expect(() => np.fft.fft(x, { out: np.zeros([6]) })).toThrow(/Cannot cast ufunc 'fft' output from complex128 to float64/);
  });

  it("validates shape, ndim, writeability and type of out", () => {
    const x = np.ones([3, 4]);
    expect(() => np.fft.fft(x, { out: np.zeros([3, 5], { dtype: "complex128" }) })).toThrow(ValueError);
    expect(() => np.fft.fft(x, { out: np.zeros([4], { dtype: "complex128" }) })).toThrow("output array has wrong shape.");
    expect(() => np.fft.fft(x, { out: np.zeros([2, 4], { dtype: "complex128" }) })).toThrow(BroadcastError);
    const ro = np.broadcastTo(np.zeros([4], { dtype: "complex128" }), [3, 4]);
    expect(() => np.fft.fft(x, { out: ro })).toThrow("output array is read-only");
    expect(() => np.fft.fft(x, { out: [1, 2] as never })).toThrow(DTypeError);
    // other dims broadcast from the result into out
    const big = np.zeros([3, 4], { dtype: "complex128" });
    expect(np.fft.fft(np.ones([1, 4]), { out: big })).toBe(big);
    expect(parts(big).filter((_, i) => i % 8 === 0)).toEqual([4, 4, 4]);
  });

  it("supports in-place transforms and strided out", () => {
    const c = np.array([1, 2, 3, 4]).astype("complex128");
    np.fft.fft(c, { out: c });
    closeTo(parts(c), [10, 0, -2, 2, -2, 0, -2, -2]);
    const base = np.zeros([4, 8], { dtype: "complex128" });
    const view = base.get([], [null, null, 2]);
    np.fft.fft(np.ones([4, 4]), { out: view });
    expect(parts(view.get(0))).toEqual([4, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("N-D transforms pass out to every step", () => {
    const a = np.arange(12).reshape([3, 4]);
    const out = np.zeros([3, 4], { dtype: "complex128" });
    expect(np.fft.fftn(a, { out })).toBe(out);
    closeTo(parts(out), parts(np.fft.fftn(a)));
    const o2 = np.zeros([3, 4], { dtype: "complex128" });
    expect(np.fft.fft2(a, null, [-2, -1], "ortho", o2)).toBe(o2);
    closeTo(parts(o2), parts(np.fft.fft2(a, { norm: "ortho" })));
    // s that changes a non-final length fails the per-step shape check (NumPy)
    expect(() => np.fft.fft2(a, { s: [5, 4], out: np.zeros([5, 4], { dtype: "complex128" }) })).toThrow(BroadcastError);
    const r = np.zeros([3, 4]);
    expect(np.fft.irfftn(np.fft.rfftn(a), { s: [3, 4], out: r })).toBe(r);
    closeTo(reals(r), reals(a));
    // empty axes: input returned, out untouched
    const o3 = np.zeros([3, 4], { dtype: "complex128" });
    expect(np.fft.fftn(a, { axes: [], out: o3 })).toBe(a);
    expect(parts(o3).every((v) => v === 0)).toBe(true);
  });

  it("keeps float32/complex64 results and NumPy's loop precision", () => {
    const x = np.array([0.1, 0.2, 0.3, 0.4, 0.7], { dtype: "float32" });
    expect(np.fft.fft(x).dtype.name).toBe("complex64");
    expect(np.fft.fft(x, { norm: "ortho" }).dtype.name).toBe("complex64");
    expect(np.fft.irfft(np.array([1, 2, 3], { dtype: "float16" })).dtype.name).toBe("float16");
    expect(np.fft.rfft2(np.ones([2, 2], { dtype: "float16" })).dtype.name).toBe("complex64");
    expect(np.fft.irfft2(np.ones([2, 2], { dtype: "complex64" })).dtype.name).toBe("float32");
  });
});

describe("np.fft hfft / ihfft (P12-2, D-151)", () => {
  it("matches NumPy values and defaults", () => {
    expect(reals(np.fft.hfft([1, 2, 3]))).toEqual([8, -2, 0, -2]);
    closeTo(parts(np.fft.ihfft([8, -2, 0, -2])), [1, 0, 2, 0, 3, 0]);
    // NumPy: hfft([1, 2j, 3], norm="ortho") = [2, 1, 2, -3]
    closeTo(reals(np.fft.hfft([np.complex(1, 0), np.complex(0, 2), np.complex(3, 0)], { norm: "ortho" })), [2, 1, 2, -3]);
    // NumPy: ihfft(arange(4), norm="forward") = [6, -2-2j, -2]
    closeTo(parts(np.fft.ihfft(np.arange(4), { norm: "forward" })), [6, 0, -2, -2, -2, 0]);
    expect(np.fft.hfft([1, 2, 3], 5).shape).toEqual([5]);
    expect(np.fft.ihfft([1, 2, 3, 4, 5]).shape).toEqual([3]);
  });

  it("dtypes, out and errors", () => {
    expect(np.fft.hfft(np.ones([4], { dtype: "float16" })).dtype.name).toBe("float16");
    expect(np.fft.hfft(np.ones([4], { dtype: "complex64" })).dtype.name).toBe("float32");
    expect(np.fft.ihfft(np.ones([4], { dtype: "float32" })).dtype.name).toBe("complex64");
    expect(np.fft.ihfft(np.ones([4], { dtype: "bool" })).dtype.name).toBe("complex128");
    const out = np.zeros([3], { dtype: "complex64" });
    expect(np.fft.ihfft(np.arange(4), { out })).toBe(out);
    closeTo(parts(out), [1.5, 0, -0.5, -0.5, -0.5, 0]);
    expect(() => np.fft.ihfft([np.complex(1, 1)])).toThrow(DTypeError);
    expect(() => np.fft.hfft([1])).toThrow(ValueError);
    expect(() => np.fft.hfft([1, 2], { axis: 2 })).toThrow(IndexError);
    expect(() => np.fft.hfft(np.array(1))).toThrow(IndexError);
  });
});

describe("np.fft rfftn / irfftn / rfft2 / irfft2 (P12-2, D-151)", () => {
  const a = np.arange(24).reshape([2, 3, 4]).astype("float64");

  it("round-trips and agrees with chained 1-D transforms", () => {
    const s = np.fft.rfftn(a);
    expect(s.shape).toEqual([2, 3, 3]);
    closeTo(parts(s), parts(np.fft.fft(np.fft.fft(np.fft.rfft(a, null, 2), null, 1), null, 0)));
    closeTo(reals(np.fft.irfftn(s, { s: [2, 3, 4] })), reals(a), 1e-12);
    expect(np.fft.irfftn(s).shape).toEqual([2, 3, 4]);
    const s2 = np.fft.rfft2(a);
    expect(s2.shape).toEqual([2, 3, 3]);
    closeTo(reals(np.fft.irfft2(s2, [3, 4])), reals(a));
    expect(np.fft.rfft2(a, null, [0, 2]).shape).toEqual([2, 3, 3]);
  });

  it("s / axes rules", () => {
    expect(np.fft.rfftn(a, { s: [-1, 6], axes: [0, 1] }).shape).toEqual([2, 4, 4]);
    expect(np.fft.rfftn(a, { s: [4, 4] }).shape).toEqual([2, 4, 3]);
    expect(np.fft.irfftn(np.ones([3, 4]), { axes: [1, 1] }).shape).toEqual([3, 6]);
    expect(np.fft.irfftn(np.ones([2, 3]), { axes: [0] }).shape).toEqual([2, 3]);
    expect(() => np.fft.rfftn(a, { s: [3], axes: [0, 1] })).toThrow("Shape and axes have different lengths.");
    expect(() => np.fft.rfftn(a, { axes: [] })).toThrow(IndexError);
    expect(() => np.fft.irfftn(a, { axes: [] })).toThrow(IndexError);
    expect(() => np.fft.rfftn(np.array(1))).toThrow(IndexError);
    expect(() => np.fft.rfftn(np.ones([2], { dtype: "complex128" }))).toThrow(DTypeError);
    expect(() => np.fft.irfftn(np.ones([3, 4]), { s: [3, 0], axes: [0, 1] })).toThrow(ValueError);
  });

  it("norms and dtypes", () => {
    const x = np.ones([2, 4]);
    closeTo(parts(np.fft.rfft2(x, { norm: "forward" })), [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    closeTo(parts(np.fft.rfft2(x, { norm: "ortho" })).slice(0, 2), [Math.sqrt(8), 0]);
    expect(np.fft.irfftn(np.ones([2, 3], { dtype: "int32" })).dtype.name).toBe("float64");
    // NumPy: the intermediate ifft gives complex64, so irfft2 of float16 is float32
    expect(np.fft.irfft2(np.ones([2, 3], { dtype: "float16" })).dtype.name).toBe("float32");
    expect(np.fft.irfftn(np.ones([2, 3], { dtype: "float16" }), { axes: [1] }).dtype.name).toBe("float16");
  });
});

describe("np.fft fftshift / ifftshift (P12-3, D-152)", () => {
  it("shifts all axes by default", () => {
    expect(np.fft.fftshift([0, 1, 2, -2, -1]).toArray()).toEqual([-2, -1, 0, 1, 2]);
    expect(np.fft.ifftshift([-2, -1, 0, 1, 2]).toArray()).toEqual([0, 1, 2, -2, -1]);
    expect(np.fft.fftshift(np.fft.fftfreq(4)).toArray()).toEqual([-0.5, -0.25, 0, 0.25]);
    const m = np.arange(12).reshape([3, 4]);
    expect(np.fft.fftshift(m).toArray()).toEqual([[10, 11, 8, 9], [2, 3, 0, 1], [6, 7, 4, 5]]);
    expect(np.fft.ifftshift(np.fft.fftshift(m)).toArray()).toEqual(m.toArray());
  });

  it("axes as number, list (repeats add up) or empty", () => {
    const m = np.arange(10).reshape([2, 5]);
    expect(np.fft.fftshift(m, 1).toArray()).toEqual([[3, 4, 0, 1, 2], [8, 9, 5, 6, 7]]);
    expect(np.fft.fftshift(m, [0]).toArray()).toEqual([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]);
    expect(np.fft.fftshift(m, -1).toArray()).toEqual([[3, 4, 0, 1, 2], [8, 9, 5, 6, 7]]);
    expect(np.fft.fftshift(np.arange(6), [0, 0]).toArray()).toEqual([0, 1, 2, 3, 4, 5]);
    const e = np.fft.fftshift(m, []);
    expect(e.toArray()).toEqual(m.toArray());
    expect(e).not.toBe(m);
  });

  it("keeps dtype; handles views, empty, 0-d and bad axes", () => {
    expect(np.fft.fftshift([true, false, true]).toArray()).toEqual([true, true, false]);
    expect(np.fft.fftshift(np.ones([3], { dtype: "float16" })).dtype.name).toBe("float16");
    const c = np.fft.fftshift(np.fft.fft([1, 2, 3]));
    expect(c.dtype.name).toBe("complex128");
    const v = np.arange(8).get([null, null, -2]);
    expect(np.fft.fftshift(v).toArray()).toEqual([3, 1, 7, 5]);
    expect(np.fft.fftshift(np.arange(6).reshape([2, 3]).T).toArray()).toEqual([[5, 2], [3, 0], [4, 1]]);
    expect(np.fft.fftshift(np.zeros([0, 3])).shape).toEqual([0, 3]);
    expect(() => np.fft.fftshift(np.array(3))).toThrow(ValueError);
    expect(() => np.fft.fftshift([1, 2, 3], 1)).toThrow(IndexError);
    expect(() => np.fft.ifftshift([1, 2, 3], [0.5])).toThrow(ValueError);
  });
});
