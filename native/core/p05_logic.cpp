#include "p05_logic.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <complex>
#include <type_traits>

#include "broadcast.hpp"
#include "cast.hpp"
#include "dtype.hpp"
#include "error.hpp"
#include "shape.hpp"
#include "ufunc_loops.hpp"

namespace nativpy {

namespace {

using ufunc_loops::compute_t;
using ufunc_loops::ld;

template <typename T> struct real_of { using type = T; };
template <typename R> struct real_of<std::complex<R>> { using type = R; };

template <typename T> bool any_nan(T v) noexcept { return std::isnan(v); }
template <typename R> bool any_nan(std::complex<R> v) noexcept { return std::isnan(v.real()) || std::isnan(v.imag()); }
template <typename T> bool finite(T v) noexcept { return std::isfinite(v); }
template <typename R> bool finite(std::complex<R> v) noexcept {
  return std::isfinite(v.real()) && std::isfinite(v.imag());
}
template <typename T> auto magnitude(T v) noexcept { return std::abs(v); }

template <typename S>
void isclose_loop(const NDArray& out, const NDArray& a, const NDArray& b, double rtol, double atol,
                  bool equal_nan) {
  using C = compute_t<S>;
  using R = typename real_of<C>::type;
  const R rt = static_cast<R>(rtol);
  const R at = static_cast<R>(atol);
  const auto plan = make_plan<3>(out.shape(), {&out, &a, &b});
  run_plan(plan, {out.data(), a.data(), b.data()},
           [&](const std::array<std::byte*, 3>& p, const std::array<std::int64_t, 3>& s, std::int64_t n) {
             for (std::int64_t i = 0; i < n; ++i) {
               const C x = ld<S>(p[1] + i * s[1]);
               const C y = ld<S>(p[2] + i * s[2]);
               bool r = x == y;
               if (!r && finite(y)) r = magnitude(x - y) <= at + rt * magnitude(y);
               if (!r && equal_nan) r = any_nan(x) && any_nan(y);
               store<bool>(p[0] + i * s[0], r);
             }
           });
}

}  // namespace

NDArray isclose(const NDArray& a, const NDArray& b, double rtol, double atol, bool equal_nan) {
  const DType bt = b.dtype();
  const DType b_inexact = (bt == DType::Bool || is_integer(bt)) ? DType::Float64 : bt;
  const DType dt = promote_types(a.dtype(), b_inexact);
  const Shape full = broadcast_shapes({a.shape(), b.shape()});
  const NDArray x = a.dtype() == dt ? a : a.astype(dt);
  const NDArray y = b.dtype() == dt ? b : b.astype(dt);
  NDArray out = NDArray::empty(full, DType::Bool);
  dispatch_dtype(dt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (std::is_floating_point_v<compute_t<S>> || is_complex_v<S>) {
      isclose_loop<S>(out, x, y, rtol, atol, equal_nan);
    } else {
      throw_error(ErrorKind::DType, "isclose: unsupported dtype " + std::string(dtype_name(dt)));
    }
  });
  return out;
}

namespace {

struct Split {
  NDArray arr;  // C-contiguous
  std::int64_t outer, len, inner;
  Shape shape;
  std::size_t axis;
};

// C-contiguous `a` (flattened if axis is nullopt, 0-d → (1,)) viewed as
// (outer, len, inner) around the normalized axis.
Split split_axis(const NDArray& a, DType dt, std::optional<std::int64_t> axis) {
  NDArray c = a.astype(dt);
  if (!axis || c.ndim() == 0) c = c.reshape({c.size()});
  const auto ax = static_cast<std::size_t>(normalize_axis(axis.value_or(0), static_cast<std::int64_t>(c.ndim())));
  Split s{c, 1, c.shape()[ax], 1, c.shape(), ax};
  for (std::size_t i = 0; i < ax; ++i) s.outer *= s.shape[i];
  for (std::size_t i = ax + 1; i < s.shape.size(); ++i) s.inner *= s.shape[i];
  return s;
}

}  // namespace

NDArray packbits(const NDArray& a, std::optional<std::int64_t> axis, bool little) {
  if (a.dtype() != DType::Bool && !is_integer(a.dtype())) {
    throw_error(ErrorKind::DType, "Expected an input array of integer or boolean data type");
  }
  const Split s = split_axis(a, DType::Bool, axis);
  const std::int64_t m = (s.len + 7) / 8;
  Shape os = s.shape;
  os[s.axis] = m;
  NDArray out = NDArray::zeros(os, DType::UInt8);
  const auto* in = reinterpret_cast<const std::uint8_t*>(s.arr.data());
  auto* o = reinterpret_cast<std::uint8_t*>(out.data());
  for (std::int64_t i = 0; i < s.outer; ++i) {
    for (std::int64_t k = 0; k < s.len; ++k) {
      const std::uint8_t* src = in + (i * s.len + k) * s.inner;
      std::uint8_t* dst = o + (i * m + k / 8) * s.inner;
      const int bit = little ? static_cast<int>(k % 8) : 7 - static_cast<int>(k % 8);
      for (std::int64_t j = 0; j < s.inner; ++j) {
        if (src[j] != 0) dst[j] = static_cast<std::uint8_t>(dst[j] | (1u << bit));
      }
    }
  }
  return out;
}

NDArray unpackbits(const NDArray& a, std::optional<std::int64_t> axis, std::optional<std::int64_t> count,
                   bool little) {
  if (a.dtype() != DType::UInt8) throw_error(ErrorKind::DType, "Expected an input array of unsigned byte data type");
  const Split s = split_axis(a, DType::UInt8, axis);
  const std::int64_t bits = s.len * 8;
  std::int64_t m = bits;
  if (count) {
    m = *count >= 0 ? *count : bits + *count;
    if (m < 0) throw_error(ErrorKind::Value, "-count larger than number of elements");
  }
  Shape os = s.shape;
  os[s.axis] = m;
  NDArray out = NDArray::zeros(os, DType::UInt8);
  const auto* in = reinterpret_cast<const std::uint8_t*>(s.arr.data());
  auto* o = reinterpret_cast<std::uint8_t*>(out.data());
  const std::int64_t n = std::min(m, bits);
  for (std::int64_t i = 0; i < s.outer; ++i) {
    for (std::int64_t k = 0; k < n; ++k) {
      const std::uint8_t* src = in + (i * s.len + k / 8) * s.inner;
      std::uint8_t* dst = o + (i * m + k) * s.inner;
      const int bit = little ? static_cast<int>(k % 8) : 7 - static_cast<int>(k % 8);
      for (std::int64_t j = 0; j < s.inner; ++j) dst[j] = static_cast<std::uint8_t>((src[j] >> bit) & 1u);
    }
  }
  return out;
}

}  // namespace nativpy
