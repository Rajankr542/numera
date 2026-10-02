#pragma once

#include <cstdint>

#include "dtype.hpp"
#include "ndarray.hpp"

namespace nativpy {

// np.finfo fields for a float/complex dtype (D-062). Float values are the
// dtype's values widened to double.
struct FloatInfo {
  DType dtype;  // component dtype for complex
  int bits, precision, iexp, nexp, nmant, machep, negep, minexp, maxexp;
  double eps, epsneg, max, min, tiny, smallest_subnormal, resolution;
};
FloatInfo float_info(DType dt);

// np.iinfo for an integer dtype.
struct IntInfo {
  int bits;
  std::int64_t min;
  std::uint64_t max;
};
IntInfo int_info(DType dt);

// Value-based np.min_scalar_type of a 0-d array (D-062).
DType min_scalar_type(const NDArray& a);

}  // namespace nativpy
