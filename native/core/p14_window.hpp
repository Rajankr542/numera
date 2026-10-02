#pragma once

#include "ndarray.hpp"

// P14 window functions (NUMPY_PARITY P14, D-172). `m` is NumPy's M (may be
// non-integral, as NumPy accepts any real M); results are float64.
namespace nativpy::p14 {

enum class Window { Bartlett, Blackman, Hamming, Hanning };

NDArray window(Window kind, double m);
NDArray kaiser(double m, double beta);
// Modified Bessel function of the first kind, order 0 (NumPy's Chebyshev
// expansion from numpy.lib._function_base_impl.i0).
double bessel_i0(double x);

}  // namespace nativpy::p14
