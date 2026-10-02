#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ndarray.hpp"

// P14 text I/O kernels (NUMPY_PARITY P14, D-171).
namespace nativpy::p14 {

struct LoadtxtOptions {
  // nullopt: runs of whitespace delimit fields.
  std::optional<std::string> delimiter;
  std::vector<std::string> comments;
  std::optional<std::string> quote;
  std::int64_t skiprows = 0;
  std::optional<std::vector<std::int64_t>> usecols;
  // < 0: read all rows.
  std::int64_t max_rows = -1;
  DType dtype = DType::Float64;
};

// np.loadtxt core: returns a 2-D (rows, cols) array; with no data rows the
// shape is (0, len(usecols) or 1). ndmin/unpack are applied by the caller.
NDArray loadtxt(const std::string& text, const LoadtxtOptions& opts);

// NumPy scalar str() of a float of dtype `dt` (Float16/32/64).
std::string float_str(double v, DType dt);
// Python repr() of a float (float64 rules) and of a complex number.
std::string py_float_repr(double v);
std::string py_complex_repr(double re, double im);

// One Python `%`-format applied to a row of values (savetxt). Values are
// read from `x` (2-D) row by row; complex rows contribute (re, im) pairs and
// "+-" is replaced by "-", as savetxt does. Each row is followed by `newline`.
std::string format_rows(const NDArray& x, const std::string& row_format, const std::string& newline);

// ndarray.tofile text mode: items in C order joined by `sep`, each written as
// Python str() of the item (or `format % item` when `format` is non-empty).
std::string tofile_text(const NDArray& a, const std::string& sep, const std::string& format);

}  // namespace nativpy::p14
