#include "p05_logic.hpp"

#include <cmath>
#include <complex>
#include <type_traits>

#include "broadcast.hpp"
#include "cast.hpp"
#include "dtype.hpp"
#include "error.hpp"
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

}  // namespace nativpy
