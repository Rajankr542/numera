#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>

#include "ndarray.hpp"

// numpy.polynomial series kernels (polynomial, chebyshev, legendre, laguerre,
// hermite, hermite_e), D-182. Coefficient arrays are 1-D float64 or
// complex128 (the TS layer validates and converts); results are float64, or
// complex128 when any input is complex. Each function trims its inputs where
// NumPy's `as_series` does, and follows NumPy's operation order.
namespace nativpy::poly {

enum class Basis { Power, Cheb, Leg, Lag, Herm, HermE };

std::optional<Basis> basis_from_name(std::string_view name) noexcept;

NDArray add(Basis b, const NDArray& c1, const NDArray& c2);
NDArray sub(Basis b, const NDArray& c1, const NDArray& c2);
NDArray mul(Basis b, const NDArray& c1, const NDArray& c2);
NDArray mulx(Basis b, const NDArray& c);
// Throws ValueError when the divisor is zero (NumPy: ZeroDivisionError).
std::pair<NDArray, NDArray> div(Basis b, const NDArray& c1, const NDArray& c2);
// `n >= 0`; the maxpower check is done by the caller.
NDArray pow(Basis b, const NDArray& c, std::int64_t n);
NDArray der(Basis b, const NDArray& c, std::int64_t m, double scl);
// `k`: integration constants (length <= m, padded with zeros); `lbnd` 0-d or 1-element.
NDArray integ(Basis b, const NDArray& c, std::int64_t m, const NDArray& k, const NDArray& lbnd,
              double scl);
// Evaluates the series `c` at every element of `x` (any shape).
NDArray val(Basis b, const NDArray& x, const NDArray& c);
// Pseudo-Vandermonde matrix: shape x.shape + (deg + 1,).
NDArray vander(Basis b, const NDArray& x, std::int64_t deg);
// Scaled companion matrix; throws ValueError for degree < 1.
NDArray companion(Basis b, const NDArray& c);
// Series with the given roots (sorted first, untrimmed input); empty -> [1].
NDArray fromroots(Basis b, const NDArray& roots);
// Conversions to/from the power basis (cheb2poly / poly2cheb, ...).
NDArray to_power(Basis b, const NDArray& c);
NDArray from_power(Basis b, const NDArray& c);

}  // namespace nativpy::poly
