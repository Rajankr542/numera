#pragma once

#include <optional>

#include "ndarray.hpp"

namespace nativpy {

// P4 special functions (D-074). NumPy implements these in Python; numera
// ports the same formulas to native code.

// Modified Bessel I0 (Cephes Chebyshev series, as numpy.i0). float16/float32
// keep their dtype, other real inputs give float64; complex raises DTypeError.
NDArray p04_i0(const NDArray& x);

// Normalized sinc sin(pi x)/(pi x), with NumPy's eps substitution at 0.
// Integer/bool inputs give float64; float and complex keep their dtype.
NDArray p04_sinc(const NDArray& x);

// numpy.nan_to_num for one scalar replacement per class. Non-inexact input
// is returned (copied unless !copy). posinf/neginf default to the dtype's
// largest/lowest finite value.
NDArray p04_nan_to_num(const NDArray& x, bool copy, double nan, std::optional<double> posinf,
                       std::optional<double> neginf);

// True if every |imag| < tol (complex input only).
bool p04_imag_all_below(const NDArray& x, double tol);

// numpy.unwrap along `axis`. `integer` selects NumPy's integer path (integer
// input and integer period: result keeps `out` dtype, exact boundary rules).
NDArray p04_unwrap(const NDArray& p, double period, std::optional<double> discont,
                   std::int64_t axis, bool integer, DType out);

}  // namespace nativpy
