#pragma once

#include "ndarray.hpp"

namespace nativpy {

// NumPy isclose (D-083): |a - b| <= atol + rtol * |b| (b finite) or a == b;
// with equal_nan also where both are NaN. Result is bool, broadcast shape.
NDArray isclose(const NDArray& a, const NDArray& b, double rtol, double atol, bool equal_nan);

}  // namespace nativpy
