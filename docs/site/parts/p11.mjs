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
      {
        name: "linalg.vectorNorm",
        sig: "np.linalg.vectorNorm(x, [options])",
        desc: "Vector norm over one axis, several axes (treated as one flattened vector), or the whole array (`axis: null`, the default). Supports any real order, `Infinity` and `-Infinity`.",
        args: [
          arr("x"),
          { name: "[options.axis]", type: "number | number[] | null", desc: "Axes to reduce. Default `null` (all)." },
          { name: "[options.keepdims]", type: "boolean", desc: "Keep reduced axes with length 1." },
          { name: "[options.ord]", type: "number", desc: "Norm order. Default `2`." },
        ],
        returns: "NDArray",
        example: `np.linalg.vectorNorm([3, 4]).item();                         // => 5
np.linalg.vectorNorm([[1, -2], [3, 4]], { axis: 1, ord: 1 }); // => [3, 7]`,
      },
      {
        name: "linalg.matrixNorm",
        sig: "np.linalg.matrixNorm(x, [options])",
        desc: "Matrix norm over the last two axes of `x` (stack).",
        args: [
          mats("x"),
          { name: "[options.keepdims]", type: "boolean", desc: "Keep the two reduced axes with length 1." },
          { name: "[options.ord]", type: "number | \"fro\" | \"nuc\"", desc: "Norm order: `\"fro\"` (default), `\"nuc\"`, 1, -1, 2, -2, `Infinity`, `-Infinity`." },
        ],
        returns: "NDArray",
        example: `np.linalg.matrixNorm([[3, 0], [0, 4]]).item();            // => 5
np.linalg.matrixNorm([[1, 2], [3, 4]], { ord: 1 }).item(); // => 6`,
      },
      {
        name: "linalg.matrixTranspose",
        sig: "np.linalg.matrixTranspose(x)",
        desc: "Swaps the last two axes (a view). Needs at least 2 dimensions.",
        args: [mats("x")],
        returns: "NDArray",
        example: `np.linalg.matrixTranspose([[1, 2, 3]]); // => [[1], [2], [3]]`,
      },
      {
        name: "linalg.diagonal",
        sig: "np.linalg.diagonal(x, [options])",
        desc: "Diagonals of the trailing matrices (`np.diagonal` with `axis1 = -2`, `axis2 = -1`).",
        args: [mats("x"), { name: "[options.offset]", type: "number", desc: "Diagonal offset. Default `0`." }],
        returns: "NDArray",
        example: `np.linalg.diagonal([[1, 2], [3, 4]]);                // => [1, 4]
np.linalg.diagonal([[1, 2], [3, 4]], { offset: 1 }); // => [2]`,
      },
      {
        name: "linalg.trace",
        sig: "np.linalg.trace(x, [options])",
        desc: "Traces of the trailing matrices (`np.trace` with `axis1 = -2`, `axis2 = -1`).",
        args: [
          mats("x"),
          { name: "[options.offset]", type: "number", desc: "Diagonal offset. Default `0`." },
          { name: "[options.dtype]", type: "DTypeLike", desc: "Accumulator/result dtype." },
        ],
        returns: "NDArray",
        example: `np.linalg.trace([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]); // => [5, 13]`,
      },
      {
        name: "linalg.outer",
        sig: "np.linalg.outer(x1, x2)",
        desc: "Outer product of two 1-D arrays. Unlike `np.outer`, other dimensions raise `ValueError`.",
        args: [arr("x1"), arr("x2")],
        returns: "NDArray",
        example: `np.linalg.outer([1, 2], [3, 4]); // => [[3, 4], [6, 8]]`,
      },
      {
        name: "linalg.tensorinv",
        sig: "np.linalg.tensorinv(a, [options])",
        desc: "Inverse of an N-d array with respect to `tensordot(·, ·, ind)`. `prod(a.shape[:ind])` must equal `prod(a.shape[ind:])`; the result has shape `a.shape[ind:] + a.shape[:ind]`.",
        args: [arr(), { name: "[options.ind]", type: "number", desc: "Number of leading indices. Default `2`." }],
        returns: "NDArray",
        example: `np.linalg.tensorinv(np.eye(4).reshape([4, 2, 2]), { ind: 1 }).shape; // => [2, 2, 4]`,
      },
      {
        name: "linalg.tensorsolve",
        sig: "np.linalg.tensorsolve(a, b, [options])",
        desc: "Solves `tensordot(a, x, x.ndim) = b` for `x`, where `x.shape = a.shape[b.ndim:]`.",
        args: [
          arr(),
          arr("b"),
          { name: "[options.axes]", type: "number[]", desc: "Axes of `a` moved to the end before solving." },
        ],
        returns: "NDArray",
        example: `np.linalg.tensorsolve(np.eye(4).reshape([4, 2, 2]), [1, 2, 3, 4]); // => [[1, 2], [3, 4]]`,
      },
    ],
  },
];

