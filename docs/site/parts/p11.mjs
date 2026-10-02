// P11 API reference entries (D-056). Same Entry/Category shape as ../api.mjs;
// every example is executed by packages/numera/test/docs_site.test.ts.

const arr = (name = "a") => ({ name, type: "ArrayLike", desc: "An `NDArray` or nested JS array." });
const mats = (name = "a") => ({ name, type: "ArrayLike", desc: "Matrix or stack of matrices `(..., M, N)`." });

/** @type {import("../api.mjs").Category[]} */
export const categories = [
  {
    id: "linalg",
    title: "np.linalg",
    entries: [
      {
        name: "linalg.cholesky",
        sig: "np.linalg.cholesky(a, [options])",
        desc: "Cholesky factor of a Hermitian positive-definite matrix (stack): lower `L` with `a = L Lᴴ`, or upper `U` with `a = Uᴴ U`. Only that triangle of `a` is read. Raises `LinAlgError` if `a` is not positive definite.",
        args: [mats(), { name: "[options.upper]", type: "boolean", desc: "Return the upper factor. Default `false`." }],
        returns: "NDArray",
        example: `np.linalg.cholesky([[4, 2], [2, 5]]);                   // => [[2, 0], [1, 2]]
np.linalg.cholesky([[4, 2], [2, 5]], { upper: true });   // => [[2, 1], [0, 2]]`,
      },
      {
        name: "linalg.slogdet",
        sig: "np.linalg.slogdet(a)",
        desc: "Sign and natural log of the absolute determinant, robust against overflow. A singular matrix gives `sign = 0`, `logabsdet = -Infinity`. For complex input the sign is a complex number of modulus 1.",
        args: [mats()],
        returns: "{ sign: NDArray, logabsdet: NDArray }",
        example: `np.linalg.slogdet([[1, 2], [3, 4]]).sign.item();     // => -1
np.linalg.slogdet([[2, 0], [0, 4]]).logabsdet.item(); // => 2.0794415416798357`,
      },
      {
        name: "linalg.svdvals",
        sig: "np.linalg.svdvals(x)",
        desc: "Singular values in descending order (same as `svd(x, { computeUV: false }).S`).",
        args: [mats("x")],
        returns: "NDArray",
        example: `np.linalg.svdvals([[3, 0], [0, 4]]); // => [4, 3]`,
      },
      {
        name: "linalg.matrixPower",
        sig: "np.linalg.matrixPower(a, n)",
        desc: "Raises a square matrix (stack) to the integer power `n` by repeated squaring. `n = 0` gives the identity in `a`'s dtype; `n < 0` inverts first (float result).",
        args: [mats(), { name: "n", type: "number", desc: "Integer exponent." }],
        returns: "NDArray",
        example: `np.linalg.matrixPower([[1, 1], [0, 1]], 3);  // => [[1, 3], [0, 1]]
np.linalg.matrixPower([[1, 1], [0, 1]], -1); // => [[1, -1], [0, 1]]`,
      },
    ],
  },
];

