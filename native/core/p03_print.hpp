#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dtype.hpp"
#include "ndarray.hpp"

namespace nativpy {

// NumPy dragon4 trim modes: 'k' None, '.' Zeros, '0' LeaveOneZero, '-' DptZeros.
enum class TrimMode : std::uint8_t { None, Zeros, LeaveOneZero, DptZeros };

// Options of NumPy's dragon4_positional / dragon4_scientific (D-063);
// -1 means "not given".
struct Dragon4Options {
  bool scientific = false;
  bool unique = true;
  bool fractional = true;  // positional: precision counts fraction digits
  int precision = -1;
  int min_digits = -1;
  bool sign = false;
  TrimMode trim = TrimMode::None;
  int pad_left = -1;
  int pad_right = -1;
  int exp_digits = -1;
};

// Formats `value` (exactly representable in `float_dtype`: Float16/32/64)
// with NumPy's Dragon4 (dragon4.c).
std::string dragon4(double value, DType float_dtype, const Dragon4Options& opt);

// Array print options used by element formatting (NumPy format_options).
struct PrintOptions {
  int precision = 8;  // -1 = None
  std::string floatmode = "maxprec";
  bool suppress = false;
  char sign = '-';
  std::string nanstr = "nan";
  std::string infstr = "inf";
};

// NumPy _leading_trailing: the corners of `a` (each axis longer than
// 2*edgeitems keeps its first and last `edgeitems` entries), C-contiguous.
NDArray leading_trailing(const NDArray& a, std::int64_t edgeitems);

// Element strings of `data` in C order, formatted like NumPy's
// Bool/Integer/Floating/ComplexFloatingFormat built from all of `data`.
std::vector<std::string> format_elements(const NDArray& data, const PrintOptions& opt);

// NumPy `str()` of the single element of `a` (a scalar's str).
std::string scalar_str(const NDArray& a);

}  // namespace nativpy
