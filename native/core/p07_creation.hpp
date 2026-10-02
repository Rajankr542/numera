#pragma once

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ndarray.hpp"

// P7 creation and grid kernels (NUMPY_PARITY P7, D-100, D-101).
namespace nativpy::p07 {

// np.logspace for scalar start/stop/base: linspace computed in float64
// (complex128 when `is_complex`), then base ** y, then cast to `dtype` (if any).
NDArray logspace(std::complex<double> start, std::complex<double> stop, bool is_complex,
                 std::int64_t num, bool endpoint, double base, std::optional<DType> dtype);

// np.geomspace, computed in float64 (complex128 when `cplx`), then cast to
// `dtype`. Negative/complex starts are rotated to the positive real axis.
NDArray geomspace(std::complex<double> start, std::complex<double> stop, std::int64_t num,
                  bool endpoint, bool cplx, DType dtype);

// np.tri(N, M, k): ones at and below diagonal k.
NDArray tri(std::int64_t n, std::int64_t m, std::int64_t k, DType dtype);
// np.tril / np.triu on the last two axes (1-D input is broadcast to (N, N)).
NDArray tril(const NDArray& a, std::int64_t k);
NDArray triu(const NDArray& a, std::int64_t k);
// np.diag: 1-D -> new 2-D array, 2-D -> read-only diagonal view.
NDArray diag(const NDArray& v, std::int64_t k);
// np.vander(x, N, increasing). Result dtype promote_types(x.dtype, int64).
NDArray vander(const NDArray& x, std::optional<std::int64_t> n, bool increasing);

// np.indices(dims, dtype) (dense): shape (len(dims), *dims).
NDArray indices(const Shape& dims, DType dtype);
// 1-D grid axis: i * step + start in `dtype` (int64 or float64), i < n.
NDArray grid_axis(double start, double step, std::int64_t n, DType dtype);
// Dense np.mgrid: out[k, i0, i1, ...] = i_k * steps[k] + starts[k].
NDArray mgrid(const std::vector<double>& starts, const std::vector<double>& steps,
              const Shape& sizes, DType dtype);

// np.tril_indices(n, k, m) (`upper` = triu_indices): row, column int64 arrays.
std::pair<NDArray, NDArray> tri_indices(std::int64_t n, std::int64_t m, std::int64_t k,
                                        bool upper);
// np.fill_diagonal(a, values, wrap), in place; values (already of a's dtype,
// any shape) repeat cyclically in flat order.
void fill_diagonal(const NDArray& a, const NDArray& values, bool wrap);

// np.concatenate of arrays already cast to one dtype (helper for r_/c_).
NDArray concatenate(const std::vector<NDArray>& arrays, std::int64_t axis);

// np.fromstring text mode (sep non-empty). count < 0 reads everything.
NDArray fromstring(const std::string& text, DType dtype, std::int64_t count,
                   const std::string& sep);

}  // namespace nativpy::p07
