#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ndarray.hpp"

namespace nativpy {

// Reductions (PLAN §17, M7). Semantics: D-017.

enum class ReduceOp { Sum, Prod, Min, Max, Mean, Var, Std };

struct ReduceOptions {
  // nullopt = all axes. Negative axes allowed.
  std::optional<std::vector<std::int64_t>> axis;
  bool keepdims = false;
  std::optional<DType> dtype;     // sum/prod/mean/var/std
  std::optional<double> initial;  // sum/prod/min/max
  std::int64_t ddof = 0;          // var/std
};

// Result dtype NumPy picks for `op` on `in` (ignoring an explicit dtype).
DType reduce_result_dtype(ReduceOp op, DType in);

NDArray reduce(ReduceOp op, const NDArray& a, const ReduceOptions& opts);

// argmin/argmax over one axis (nullopt = flattened). Result is int64.
NDArray arg_reduce(bool is_max, const NDArray& a, std::optional<std::int64_t> axis,
                   bool keepdims);

}  // namespace nativpy
