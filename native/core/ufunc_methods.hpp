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

// ufunc.outer(a, b): op(a[..., None, ...], b), result shape a.shape + b.shape.
// All ufunc call parameters (dtype/casting/where/order, D-048..D-050) apply.
NDArray ufunc_outer(const Ufunc& u, const NDArray& a, const NDArray& b, const NDArray* out,
                    const UfuncParams& params);

// ufunc.reduceat(a, indices, axis): reduce a[indices[i]:indices[i+1]] along
// `axis` (or take a[indices[i]] when indices[i] >= indices[i+1]). Uses the
// reduce loop-dtype rule. `opts.axis` holds at most one axis.
NDArray ufunc_reduceat(const Ufunc& u, const NDArray& a, const NDArray& indices,
                       const NDArray* out, const UfuncReduceOptions& opts);

// ufunc.at(a, indices, b): unbuffered in-place a[idx] = op(a[idx], b[...]).
// `indices` holds one integer index array per leading axis (broadcast
// together); repeated indices apply repeatedly. `b` is null for unary ufuncs.
void ufunc_at(const Ufunc& u, const NDArray& a, const std::vector<NDArray>& indices,
              const NDArray* b);

}  // namespace nativpy
