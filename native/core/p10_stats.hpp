#pragma once

// average, cov, corrcoef, gradient, trapezoid (D-135, D-137), following
// numpy/lib/_function_base_impl.py.

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "ndarray.hpp"

namespace nativpy::p10 {

// NumPy average. Returns (avg, sum of weights broadcast to avg's shape).
// `keepdims` nullopt = NumPy's _NoValue (same result as false).
std::pair<NDArray, NDArray> average(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis,
                                    const std::optional<NDArray>& weights, bool keepdims);

struct CovOptions {
  std::optional<NDArray> y;
  bool rowvar = true;
  bool bias = false;
  std::optional<std::int64_t> ddof;
  std::optional<NDArray> fweights;
  std::optional<NDArray> aweights;
  std::optional<DType> dtype;
};

// NumPy cov. `dof_warning` is set when the degrees of freedom are <= 0
// (NumPy warns "Degrees of freedom <= 0 for slice").
NDArray cov(const NDArray& m, const CovOptions& opts, bool* dof_warning);

// NumPy corrcoef (cov normalised, real and imaginary parts clipped to [-1, 1]).
NDArray corrcoef(const NDArray& x, const std::optional<NDArray>& y, bool rowvar, std::optional<DType> dtype,
                 bool* dof_warning);

// Spacing for one gradient axis: a JS number (weak scalar), a 0-d array
// (strong scalar) or 1-d coordinates.
struct GradSpacing {
  double value = 1.0;
  std::optional<NDArray> array;
};

// NumPy gradient over `axes` (nullopt = all). `spacing` is empty (unit
// spacing), one entry (applied to every axis) or one entry per axis.
std::vector<NDArray> gradient(const NDArray& f, const std::vector<GradSpacing>& spacing,
                              const std::optional<std::vector<std::int64_t>>& axes, std::int64_t edge_order);

// NumPy trapezoid(y, x, dx, axis).
NDArray trapezoid(const NDArray& y, const std::optional<NDArray>& x, double dx, std::int64_t axis);

}  // namespace nativpy::p10
