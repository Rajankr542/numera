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
      {
        name: "linalg.pinv",
        sig: "np.linalg.pinv(a, [options])",
        desc: "Moore–Penrose pseudo-inverse of a matrix (stack) via SVD, or via `eigh` when `hermitian`. Singular values at or below `rcond * max(s)` are treated as zero. Shape `(..., M, N)` gives `(..., N, M)`.",
        args: [
          mats(),
          { name: "[options.rcond]", type: "ArrayLike | number", desc: "Relative cutoff, broadcast over the batch. Default `1e-15`." },
          { name: "[options.rtol]", type: "ArrayLike | number | null", desc: "Array-API alias of `rcond`; `null` means `max(M, N) * eps`. Cannot be combined with `rcond`." },
          { name: "[options.hermitian]", type: "boolean", desc: "Treat `a` as Hermitian. Default `false`." },
        ],
        returns: "NDArray",
        example: `np.linalg.pinv([[1, 0], [0, 2]]);       // => [[1, 0], [0, 0.5]]
np.linalg.pinv([[2, 0, 0], [0, 4, 0]]); // => [[0.5, 0], [0, 0.25], [0, 0]]`,
      },
      {
        name: "linalg.matrixRank",
        sig: "np.linalg.matrixRank(A, [options])",
        desc: "Rank of a matrix (stack): the number of singular values above the threshold. By default the threshold is `max(s) * max(M, N) * eps`. For 0-d and 1-d input the result is 1 if any element is nonzero, else 0. Returns int64.",
        args: [
          mats("A"),
          { name: "[options.tol]", type: "ArrayLike | number", desc: "Absolute threshold." },
          { name: "[options.rtol]", type: "ArrayLike | number", desc: "Relative threshold (times the largest singular value). Cannot be combined with `tol`." },
          { name: "[options.hermitian]", type: "boolean", desc: "Use eigenvalue magnitudes of a Hermitian `A`. Default `false`." },
        ],
        returns: "NDArray",
        example: `np.linalg.matrixRank([[1, 2], [2, 4]]).item(); // => 1
np.linalg.matrixRank(np.eye(3)).item();        // => 3`,
      },
      {
        name: "linalg.cond",
        sig: "np.linalg.cond(x, [p])",
        desc: "Condition number of a matrix (stack). `null`, `2` and `-2` use singular values (any shape); `1`, `-1`, `Infinity`, `-Infinity`, `\"fro\"` and `\"nuc\"` compute `norm(x) * norm(inv(x))` and need square input. A singular matrix gives `Infinity`.",
        args: [mats("x"), { name: "[p]", type: "number | \"fro\" | \"nuc\" | null", desc: "Norm order. Default `null` (2-norm)." }],
        returns: "NDArray",
        example: `np.linalg.cond([[1, 0], [0, 2]]).item();        // => 2
np.linalg.cond([[4, 0], [0, 2]], 1).item();     // => 2`,
      },
    ],
  },
];

