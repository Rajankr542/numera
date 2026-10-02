#pragma once

// Cumulative sums/products, diff and ptp (D-132).

#include <cstdint>
#include <optional>
#include <vector>

#include "ndarray.hpp"

namespace nativpy::p10 {

struct CumulativeOptions {
  std::optional<std::int64_t> axis;  // nullopt = flattened input
  std::optional<DType> dtype;
  bool include_initial = false;  // cumulative_sum / cumulative_prod
  bool skip_nan = false;         // nancumsum / nancumprod: NaN counts as the identity
  // Array-API cumulative_*: axis is required for ndim > 1 and 0-d input is
  // rejected. NumPy cumsum flattens when axis is omitted.
  bool array_api = false;
};

// NumPy cumsum (prod=false) / cumprod (prod=true) and their variants.
NDArray cumulative(bool prod, const NDArray& a, const CumulativeOptions& opts, const NDArray* out);

// NumPy diff(a, n, axis, prepend, append). prepend/append are already
// arrays (0-d values are broadcast along `axis` with length 1).
NDArray diff(const NDArray& a, std::int64_t n, std::int64_t axis, const std::optional<NDArray>& prepend,
             const std::optional<NDArray>& append);

// NumPy ptp: max - min in the input dtype.
NDArray ptp(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis, bool keepdims);

// Concatenation along `axis` with dtype promotion (private helper).
NDArray concat(const std::vector<NDArray>& parts, std::int64_t axis);

}  // namespace nativpy::p10
