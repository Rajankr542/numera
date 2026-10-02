#pragma once

#include <utility>

#include "ndarray.hpp"

// P14 legacy polynomial kernels (NUMPY_PARITY P14, D-173).
namespace nativpy::p14 {

// np.convolve(a, b, mode="full") for 1-D arrays; dtype promote_types(a, b).
NDArray convolve_full(const NDArray& a, const NDArray& b);

// np.polydiv core: u and v are 1-D arrays of the same inexact dtype.
// Returns (quotient, remainder) with NumPy's remainder trimming.
std::pair<NDArray, NDArray> polydiv(const NDArray& u, const NDArray& v);

}  // namespace nativpy::p14
