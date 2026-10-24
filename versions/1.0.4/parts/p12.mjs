// P12 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.

const arrayArg = (name = "a") => ({ name, type: "ArrayLike", desc: "An `NDArray`, nested JS array or scalar." });
const outArg = {
  name: "[out]",
  type: "NDArray",
  desc: "Write the result here and return it. Its shape must match the result along the transformed axis, and the result dtype must cast to it under `same_kind`.",
};

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "fft",
    title: "np.fft",
    entries: [
      {
        name: "fft.out",
        sig: "np.fft.fft(a, n, axis, norm, out) · np.fft.fftn(a, { s, axes, norm, out }) · …",
        desc: "Every transform accepts `out`, either as the last positional parameter or in the options object. The result is written into `out`, and `out` itself is returned. Multi-axis transforms pass `out` to every 1-D step, as NumPy does.",
        args: [arrayArg(), outArg],
        returns: "NDArray (`out`)",
        example: `const out = np.zeros([4], { dtype: "complex64" });
np.fft.fft([1, 1, 1, 1], { out }) === out; // => true
Array.from(out.toTypedArray());            // => [4, 0, 0, 0, 0, 0, 0, 0]
const r = np.zeros([2, 2]);
np.fft.irfft2([[4, 0], [0, 0]], { s: [2, 2], out: r }) === r; // => true
r;                                          // => [[1, 1], [1, 1]]`,
      },
      {
        name: "fft.hfft",
        sig: "np.fft.hfft(a, [n], [axis=-1], [norm], [out]) · np.fft.ihfft(...)",
        desc: "FFT of a signal with Hermitian symmetry, given as its first half, which has a real spectrum. The default `n` is `2 * (m - 1)`. `ihfft` is the inverse: it takes real input and returns `n / 2 + 1` complex values.",
        args: [
          arrayArg(),
          { name: "[n], [axis], [norm]", type: "", desc: "As for `fft`." },
          outArg,
        ],
        returns: "NDArray (real for `hfft`, complex for `ihfft`)",
        example: `np.fft.hfft([1, 2, 3]);                       // => [8, -2, 0, -2]
Array.from(np.fft.ihfft([8, -2, 0, -2]).toTypedArray()); // => [1, 0, 2, 0, 3, 0]`,
      },
      {
        name: "fft.rfftn",
        sig: "np.fft.rfftn(a, [options]) · irfftn · rfft2 · irfft2",
        desc: "N-D transforms of real input: `rfft` over the last axis in `axes`, then `fft` over the others. `irfftn` inverts this, and the last output length defaults to `2 * (m - 1)`. `rfft2`/`irfft2` default to the last two axes.",
        args: [
          arrayArg(),
          { name: "[options.s]", type: "number[]", desc: "Output lengths per transformed axis (`-1` keeps the input length)." },
          { name: "[options.axes]", type: "number[]", desc: "Axes to transform." },
          { name: "[options.norm]", type: "string", desc: "Scaling convention." },
          { name: "[options.out]", type: "NDArray", desc: "Output array (see `out`)." },
        ],
        returns: "NDArray",
        example: `const spec = np.fft.rfft2([[1, 2, 3, 4], [5, 6, 7, 8]]);
spec.shape;                                     // => [2, 3]
np.fft.irfft2(spec);                            // => [[1, 2, 3, 4], [5, 6, 7, 8]]
np.fft.rfftn(np.ones([2, 3, 4])).shape;         // => [2, 3, 3]
np.fft.irfftn(np.ones([2, 3]), { axes: [0] }).shape; // => [2, 3]`,
      },
      {
        name: "fft.fftshift",
        sig: "np.fft.fftshift(x, [axes]) · np.fft.ifftshift(x, [axes])",
        desc: "Moves the zero-frequency term to the centre of the spectrum by rolling each axis in `axes` (default: all) by `shape[ax] // 2`. `ifftshift` undoes it; the two differ for odd lengths.",
        args: [
          arrayArg("x"),
          { name: "[axes]", type: "number | number[] | null", desc: "Axes to shift. Default: all." },
        ],
        returns: "NDArray (same dtype)",
        example: `np.fft.fftshift([0, 1, 2, -2, -1]);      // => [-2, -1, 0, 1, 2]
np.fft.ifftshift([-2, -1, 0, 1, 2]);     // => [0, 1, 2, -2, -1]
np.fft.fftshift([[0, 1, 2], [3, 4, 5]], 1); // => [[2, 0, 1], [5, 3, 4]]`,
      },
    ],
  },
];
