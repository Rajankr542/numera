#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ndarray.hpp"
#include "ufunc_registry.hpp"

namespace nativpy {

// ufunc.reduce / ufunc.accumulate (D-052), driven by the D-051 registry.

struct UfuncReduceOptions {
  // Axes to reduce. nullopt = the default (axis 0; no axis for 0-d input).
  // all_axes = NumPy axis=None. An empty vector reduces nothing.
  std::optional<std::vector<std::int64_t>> axis;
  bool all_axes = false;
  std::optional<DType> dtype;
  bool keepdims = false;            // reduce only
  std::optional<double> initial;    // reduce only
  std::optional<NDArray> where;     // reduce only (bool, broadcast to a)
};

NDArray ufunc_reduce(const Ufunc& u, const NDArray& a, const NDArray* out,
                     const UfuncReduceOptions& opts);

NDArray ufunc_accumulate(const Ufunc& u, const NDArray& a, const NDArray* out,
                         const UfuncReduceOptions& opts);

}  // namespace nativpy
