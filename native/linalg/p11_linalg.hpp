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

}  // namespace nativpy::linalg
