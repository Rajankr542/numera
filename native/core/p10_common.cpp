#include "p10_common.hpp"

#include <utility>

#include "shape_ops.hpp"

namespace nativpy::p10 {

double round_to(DType dt, double v) noexcept {
  switch (dt) {
    case DType::Float16: return half_to_double(double_to_half(v));
    case DType::Float32: return static_cast<double>(static_cast<float>(v));
    default: return v;
  }
}

Rows to_rows(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis, bool keepdims,
             std::optional<DType> dt) {
  const auto nd = static_cast<std::int64_t>(a.ndim());
  std::vector<bool> reduced(static_cast<std::size_t>(nd), !axis.has_value());
  if (axis) {
    for (const auto ax : normalize_axes(*axis, nd)) reduced[static_cast<std::size_t>(ax)] = true;
  }
  std::vector<std::int64_t> perm;
  Shape out_shape;
  std::int64_t rows = 1;
  std::int64_t n = 1;
  for (std::int64_t d = 0; d < nd; ++d) {
    const std::int64_t len = a.shape()[static_cast<std::size_t>(d)];
    if (!reduced[static_cast<std::size_t>(d)]) {
      perm.push_back(d);
      out_shape.push_back(len);
      rows *= len;
    } else if (keepdims) {
      out_shape.push_back(1);
    }
  }
  for (std::int64_t d = 0; d < nd; ++d) {
    if (reduced[static_cast<std::size_t>(d)]) {
      perm.push_back(d);
      n *= a.shape()[static_cast<std::size_t>(d)];
    }
  }
  const NDArray t = nd == 0 ? a : transpose(a, perm);
  return Rows{t.astype(dt.value_or(a.dtype())), std::move(out_shape), rows, n};
}

}  // namespace nativpy::p10
