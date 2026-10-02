#pragma once

// Shared helpers for the P10 statistics kernels (D-130).

#include <cmath>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <vector>

#include "cast.hpp"
#include "ndarray.hpp"

namespace nativpy::p10 {

// float16 is computed as float (D-014).
template <typename S>
using compute_t = std::conditional_t<std::is_same_v<S, float16_t>, float, S>;

template <typename T>
bool is_nan(T v) noexcept {
  if constexpr (std::is_floating_point_v<T>) return std::isnan(v);
  else return false;
}

// Ascending order with NaN last (NumPy sort order for real floats).
template <typename T>
bool nan_last_less(T a, T b) noexcept {
  if constexpr (std::is_floating_point_v<T>) {
    if (std::isnan(a)) return false;
    if (std::isnan(b)) return true;
  }
  return a < b;
}

// Rounds a double to the precision of a floating dtype (identity otherwise).
// Each NumPy float16/float32 operation rounds its result; computing in double
// and rounding afterwards gives the same value for + - * /.
double round_to(DType dt, double v) noexcept;

// `a` rearranged so the reduced axes are trailing and C-contiguous:
// data has shape (rows, n). out_shape is the kept shape (keepdims applied).
struct Rows {
  NDArray data;
  Shape out_shape;
  std::int64_t rows = 1;
  std::int64_t n = 1;
};

// Copy of `a` with NaN (either component for complex) replaced by `v`;
// `a` itself for non-float dtypes.
NDArray replace_nan(const NDArray& a, double v);

// Int64 count of non-NaN elements per reduced slice (keepdims applied).
NDArray count_not_nan(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis, bool keepdims);

bool is_inexact(DType dt) noexcept;

// View of `a` with axis `ax` restricted to [start, start + len).
NDArray axis_slice(const NDArray& a, std::size_t ax, std::int64_t start, std::int64_t len);

// 0-d array of dtype `dt` holding v.
NDArray scalar(double v, DType dt);

// axis nullopt = all axes. The data keeps a's dtype unless `dt` is given.
Rows to_rows(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis, bool keepdims,
             std::optional<DType> dt = std::nullopt);

// Element i of a C-contiguous array, converted to T.
template <typename T, typename S>
T at(const NDArray& a, std::int64_t i) noexcept {
  return cast_value<T>(load<S>(a.data() + static_cast<std::size_t>(i) * sizeof(S)));
}

// Stores v (converted to dtype element type) at flat index i of a C-contiguous array.
template <typename S, typename T>
void put(const NDArray& a, std::int64_t i, T v) noexcept {
  store<S>(a.data() + static_cast<std::size_t>(i) * sizeof(S), cast_value<S>(v));
}

}  // namespace nativpy::p10
