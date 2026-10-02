#pragma once

// NaN-ignoring reductions (D-133), following NumPy's _nanfunctions_impl.

#include <cstdint>
#include <optional>
#include <vector>

#include "ndarray.hpp"
#include "reduce.hpp"

namespace nativpy::p10 {

// nansum / nanprod / nanmean / nanvar / nanstd / nanmin / nanmax. `op` uses
// ReduceOptions (axis, keepdims, dtype, initial for sum/prod, ddof).
NDArray nan_reduce(ReduceOp op, const NDArray& a, const ReduceOptions& opts);

// nanargmin / nanargmax.
NDArray nan_arg_reduce(bool is_max, const NDArray& a, std::optional<std::int64_t> axis, bool keepdims);

}  // namespace nativpy::p10
