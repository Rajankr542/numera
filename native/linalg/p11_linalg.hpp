#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "linalg.hpp"
#include "ndarray.hpp"

namespace nativpy::linalg {

// P11 linear algebra completion (D-140..D-142). All results are new arrays.

// NumPy `_commonType` result dtype of one linalg operand: float32 / complex64
// stay, complex128 stays, bool/int/float64 -> float64, float16 -> DTypeError.
DType p11_result_dtype(const NDArray& a, const char* fn);

// Cholesky factor (D-142): lower L with A = L Lᴴ, or upper U = Lᴴ.
NDArray cholesky(const NDArray& a, bool upper);

struct SlogdetResult {
  NDArray sign;
  NDArray logabsdet;
};
SlogdetResult slogdet(const NDArray& a);

// a^n by repeated squaring; n < 0 inverts first, n == 0 gives the identity.
NDArray matrix_power(const NDArray& a, std::int64_t n);

// Moore-Penrose pseudo-inverse; `rcond` (float64) broadcasts to the batch
// shape and is relative to the largest singular value (|eigenvalue| when
// `hermitian`). Empty matrices give an empty (..., N, M) array of a's dtype.
NDArray pinv(const NDArray& a, const NDArray& rcond, bool hermitian);

// Number of singular values above the threshold: `tol` absolute when given,
// else `rtol` (default max(M, N) * eps) times the largest. ndim < 2: 0-d
// "any nonzero". Result int64 with the batch shape.
NDArray matrix_rank(const NDArray& a, const std::optional<NDArray>& tol,
                    const std::optional<NDArray>& rtol, bool hermitian);

// Condition number; `p` uses the norm orders of `norm` (default = 2 via SVD).
NDArray cond(const NDArray& a, const NormOrd& p);

}  // namespace nativpy::linalg
