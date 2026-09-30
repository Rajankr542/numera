import { describe, expect, it } from "vitest";
import np, { fft, NDArray } from "../src/index.js";

/** Interleaved re/im of a complex array. */
const parts = (a: NDArray): number[] => Array.from(a.toTypedArray() as Float64Array);
const closeTo = (got: number[], exp: number[], tol = 1e-12): void => {
  expect(got.length).toBe(exp.length);
  got.forEach((x, i) => expect(Math.abs(x - exp[i]!)).toBeLessThanOrEqual(tol * (1 + Math.abs(exp[i]!))));
};

describe("np.fft", () => {
  it("is exposed on the default export and as a named export", () => {
    expect(np.fft).toBe(fft);
    expect(Object.keys(np.fft).sort()).toEqual(
      ["fft", "fft2", "fftfreq", "fftn", "ifft", "ifft2", "ifftn", "irfft", "rfft", "rfftfreq"],
    );
  });

  it("fft of a delta is flat; of ones is a delta", () => {
    closeTo(parts(np.fft.fft([1, 0, 0, 0])), [1, 0, 1, 0, 1, 0, 1, 0]);
    closeTo(parts(np.fft.fft([1, 1, 1, 1])), [4, 0, 0, 0, 0, 0, 0, 0]);
  });

  it("positional and options-object forms agree", () => {
    const x = [1, 2, 3, 4, 5];
    const a = np.fft.fft(x, 8, -1, "ortho");
    const b = np.fft.fft(x, { n: 8, norm: "ortho" });
    expect(a.shape).toEqual([8]);
    closeTo(parts(a), parts(b));
    const m = np.arange(6).reshape([2, 3]);
    closeTo(parts(np.fft.fftn(m, [2, 4], [0, 1])), parts(np.fft.fftn(m, { s: [2, 4], axes: [0, 1] })));
  });

  it("rfft/irfft round trip and dtypes", () => {
    const x = np.array([0.5, -1, 2, 3, -4, 1], { dtype: "float32" });
    const r = np.fft.rfft(x);
    expect(r.dtype.name).toBe("complex64");
    expect(r.shape).toEqual([4]);
    const back = np.fft.irfft(r, 6);
    expect(back.dtype.name).toBe("float32");
    closeTo(Array.from(back.toTypedArray() as Float32Array), [0.5, -1, 2, 3, -4, 1], 1e-6);
  });

  it("fft2 defaults to the last two axes and ifft2 inverts it", () => {
    const x = np.arange(24).reshape([2, 3, 4]);
    const f = np.fft.fft2(x);
    expect(f.shape).toEqual([2, 3, 4]);
    const back = np.fft.ifft2(f);
    const p = parts(back);
    closeTo(p.filter((_, i) => i % 2 === 0), Array.from({ length: 24 }, (_, i) => i), 1e-12);
  });

  it("fftfreq / rfftfreq", () => {
    expect(np.fft.fftfreq(4).toArray()).toEqual([0, 0.25, -0.5, -0.25]);
    expect(np.fft.rfftfreq(4, 0.5).toArray()).toEqual([0, 0.5, 1]);
  });

  it("raises typed errors", () => {
    expect(() => np.fft.fft([])).toThrow(np.ValueError);
    expect(() => np.fft.fft([1, 2], 0)).toThrow(np.ValueError);
    expect(() => np.fft.fft([1, 2], 2.5)).toThrow(np.ValueError);
    expect(() => np.fft.fft([1, 2], null, 3)).toThrow(np.IndexError);
    expect(() => np.fft.fft([1, 2], null, -1, "bad" as never)).toThrow(np.ValueError);
    expect(() => np.fft.rfft(np.fft.fft([1, 2]))).toThrow(np.DTypeError);
    expect(() => np.fft.fftn([[1, 2]], [2], [0, 1])).toThrow(np.ValueError);
    expect(() => np.fft.fftfreq(0)).toThrow(np.ValueError);
  });
});
