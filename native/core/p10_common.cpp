#include "p10_common.hpp"

#include <utility>

#include "error.hpp"

#include "shape_ops.hpp"

namespace nativpy::p10 {

double round_to(DType dt, double v) noexcept {
  switch (dt) {
    case DType::Float16: return half_to_double(double_to_half(v));
    case DType::Float32: return static_cast<double>(static_cast<float>(v));
    default: return v;
  }
}

// Copy of `a` with NaN (either component for complex) replaced by `v`.
NDArray replace_nan(const NDArray& a, double v) {
  const DType dt = a.dtype();
  if (dtype_info(dt).kind != 'f' && !is_complex(dt)) return a;
  NDArray c = a.copy();
  dispatch_dtype(dt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (std::is_same_v<S, float16_t> || std::is_floating_point_v<S>) {
      using C = compute_t<S>;
      for (std::int64_t i = 0; i < c.size(); ++i) {
        if (std::isnan(at<C, S>(c, i))) put<S>(c, i, v);
      }
    } else if constexpr (is_complex_v<S>) {
      for (std::int64_t i = 0; i < c.size(); ++i) {
        const S z = at<S, S>(c, i);
        if (std::isnan(z.real()) || std::isnan(z.imag())) put<S>(c, i, S(static_cast<typename S::value_type>(v)));
      }
    }
  });
  return c;
}

NDArray axis_slice(const NDArray& a, std::size_t ax, std::int64_t start, std::int64_t len) {
  Shape shape = a.shape();
  shape[ax] = len;
  return a.view(shape, a.strides(), a.offset() + start * a.strides()[ax]);
}

NDArray scalar(double v, DType dt) {
  NDArray r = NDArray::empty({}, dt);
  r.set_double(0, v);
  return r;
}

bool is_inexact(DType dt) noexcept { return dtype_info(dt).kind == 'f' || is_complex(dt); }

NDArray count_not_nan(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis, bool keepdims) {
  const Rows r = to_rows(a, axis, keepdims);
  NDArray out = NDArray::zeros(r.out_shape, DType::Int64);
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    for (std::int64_t row = 0; row < r.rows; ++row) {
      std::int64_t c = 0;
      for (std::int64_t i = 0; i < r.n; ++i) {
        const S v = at<S, S>(r.data, row * r.n + i);
        bool nan = false;
        if constexpr (is_complex_v<S>) nan = std::isnan(v.real()) || std::isnan(v.imag());
        else nan = is_nan(cast_value<compute_t<S>>(v));
        c += nan ? 0 : 1;
      }
      put<std::int64_t>(out, row, c);
    }
  });
  return out;
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
