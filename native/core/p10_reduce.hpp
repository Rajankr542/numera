#pragma once

// where= / out= for np.sum/prod/min/max/mean/var/std (D-136, P10-7).
// Wraps the existing reduce() kernel by materialising a masked copy of the
// input first; the hot path (no where/out) is unchanged.

#include <cstdint>
#include <optional>
#include <vector>

#include "ndarray.hpp"
#include "reduce.hpp"

namespace nativpy::p10 {

struct ReduceWhereOptions {
  std::optional<std::vector<std::int64_t>> axis;
  bool keepdims = false;
  std::optional<DType> dtype;
  std::optional<double> initial;
  std::int64_t ddof = 0;
  // where= : bool mask broadcast to `a`; nullopt = no mask.
  std::optional<NDArray> where;
  // out= : pre-allocated output; nullopt = allocate.
  std::optional<NDArray> out;
};

// Equivalent to np.sum/prod/min/max/mean/var/std with where= and out= support.
NDArray reduce_where(ReduceOp op, const NDArray& a, const ReduceWhereOptions& opts);

}  // namespace nativpy::p10
