#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "dtype.hpp"

namespace nativpy {

template <typename T> struct is_complex_type : std::false_type {};
template <typename T> struct is_complex_type<std::complex<T>> : std::true_type {};
template <typename T> inline constexpr bool is_complex_v = is_complex_type<T>::value;

// Float -> integer conversion. NumPy delegates to the C cast, which is
// undefined for NaN/out-of-range values and differs between platforms. nativpy
// defines it deterministically (matching NumPy on arm64): NaN -> 0; types of
// 32/64 bits saturate; 8/16-bit types saturate to int32 then wrap.
template <typename To, typename From>
To float_to_int(From v) noexcept {
  static_assert(std::is_integral_v<To> && std::is_floating_point_v<From>);
  if (std::isnan(v)) return To{0};
  if constexpr (sizeof(To) < 4) {
    return static_cast<To>(float_to_int<std::int32_t>(v));
  } else {
    constexpr To lo = std::numeric_limits<To>::min();
    constexpr To hi = std::numeric_limits<To>::max();
    if (v <= static_cast<From>(lo)) return lo;
    if (v >= static_cast<From>(hi)) return hi;  // (From)hi rounds up to 2^n
    return static_cast<To>(v);
  }
}

// Converts a single stored value between dtype storage types
// (NumPy "unsafe" casting semantics; complex -> real discards imaginary part).
template <typename To, typename From>
To cast_value(From v) noexcept {
  if constexpr (std::is_same_v<To, From>) {
    return v;
  } else if constexpr (std::is_same_v<From, float16_t>) {
    return cast_value<To>(half_to_double(v));
  } else if constexpr (is_complex_v<From>) {
    if constexpr (is_complex_v<To>) {
      using V = typename To::value_type;
      return To(static_cast<V>(v.real()), static_cast<V>(v.imag()));
    } else if constexpr (std::is_same_v<To, bool>) {
      return v != From{};  // NumPy: true if either component is nonzero
    } else {
      return cast_value<To>(v.real());
    }
  } else if constexpr (std::is_same_v<To, float16_t>) {
    return double_to_half(static_cast<double>(v));
  } else if constexpr (is_complex_v<To>) {
    using V = typename To::value_type;
    return To(static_cast<V>(v), V{0});
  } else if constexpr (std::is_same_v<To, bool>) {
    return v != From{0};
  } else if constexpr (std::is_integral_v<To> && std::is_floating_point_v<From>) {
    return float_to_int<To>(v);
  } else {
    return static_cast<To>(v);  // int<->int wraps (C++20 modular), int->float rounds
  }
}

template <typename T>
T load(const std::byte* p) noexcept {
  T v;
  std::memcpy(&v, p, sizeof(T));
  return v;
}

template <typename T>
void store(std::byte* p, T v) noexcept {
  std::memcpy(p, &v, sizeof(T));
}

}  // namespace nativpy
