#pragma once

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "error.hpp"
#include "fp_errors.hpp"

namespace nativpy::kernels {

// Scalar kernels on the compute type T (float16 is computed as float).
// Integer arithmetic is modular (no signed-overflow UB), D-014.

template <typename T>
using U = std::make_unsigned_t<T>;

template <typename T>
T add(T a, T b) noexcept {
  if constexpr (std::is_same_v<T, bool>) return a || b;
  else if constexpr (std::is_integral_v<T>) return static_cast<T>(static_cast<U<T>>(a) + static_cast<U<T>>(b));
  else return a + b;
}

template <typename T>
T sub(T a, T b) noexcept {
  if constexpr (std::is_integral_v<T>) return static_cast<T>(static_cast<U<T>>(a) - static_cast<U<T>>(b));
  else return a - b;
}

template <typename T>
T mul(T a, T b) noexcept {
  if constexpr (std::is_same_v<T, bool>) return a && b;
  else if constexpr (std::is_integral_v<T>) {
    // Promote to avoid int-promotion UB for 16-bit unsigned multiply.
    using W = std::conditional_t<(sizeof(T) < sizeof(unsigned)), unsigned, U<T>>;
    return static_cast<T>(static_cast<W>(static_cast<U<T>>(a)) * static_cast<W>(static_cast<U<T>>(b)));
  } else return a * b;
}

template <typename T>
T div(T a, T b) noexcept { return a / b; }  // floating only

// NumPy integer floor division: x // 0 -> 0; MIN // -1 -> MIN (wraps).
template <typename T>
T floordiv_int(T a, T b) noexcept {
  if (b == 0) {
    raise_fp_divbyzero();
    return 0;
  }
  if constexpr (std::is_signed_v<T>) {
    if (b == -1) {
      if (a == std::numeric_limits<T>::min()) raise_fp_overflow();
      return static_cast<T>(U<T>{0} - static_cast<U<T>>(a));
    }
    T q = static_cast<T>(a / b);
    if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
    return q;
  } else {
    return static_cast<T>(a / b);
  }
}

// NumPy integer remainder: sign follows divisor; x % 0 -> 0.
template <typename T>
T mod_int(T a, T b) noexcept {
  if (b == 0) {
    raise_fp_divbyzero();
    return 0;
  }
  if constexpr (std::is_signed_v<T>) {
    if (b == -1) return 0;
    T r = static_cast<T>(a % b);
    if (r != 0 && ((r < 0) != (b < 0))) r = static_cast<T>(r + b);
    return r;
  } else {
    return static_cast<T>(a % b);
  }
}

// Port of NumPy npy_divmod (npy_math_internal.h.src) for floating types.
template <typename T>
T fmod_floor(T a, T b, T* floordiv_out) noexcept {
  T mod = std::fmod(a, b);
  if (b == T{0}) {
    if (floordiv_out) *floordiv_out = a / b;
    return mod;
  }
  T div = (a - mod) / b;
  if (mod != T{0}) {
    if ((b < T{0}) != (mod < T{0})) {
      mod += b;
      div -= T{1};
    }
  } else {
    mod = std::copysign(T{0}, b);
  }
  T floordiv;
  if (div != T{0}) {
    floordiv = std::floor(div);
    if (div - floordiv > T{0.5}) floordiv += T{1};
  } else {
    floordiv = std::copysign(T{0}, a / b);
  }
  if (floordiv_out) *floordiv_out = floordiv;
  return mod;
}

template <typename T>
T mod(T a, T b) noexcept {
  if constexpr (std::is_integral_v<T>) return mod_int(a, b);
  else return fmod_floor<T>(a, b, nullptr);
}

template <typename T>
T floordiv(T a, T b) noexcept {
  if constexpr (std::is_integral_v<T>) return floordiv_int(a, b);
  else {
    if (b == T{0}) return a / b;  // NumPy: inf/nan like true divide
    T q;
    fmod_floor<T>(a, b, &q);
    return q;
  }
}

// Integer power by squaring (modular). Negative exponent is rejected before
// the loop runs (ValueError, like NumPy).
template <typename T>
T ipow(T base, T exp) noexcept {
  using W = std::conditional_t<(sizeof(T) < sizeof(unsigned)), unsigned, U<T>>;
  W r = 1;
  W b = static_cast<W>(static_cast<U<T>>(base));
  U<T> e = static_cast<U<T>>(exp);
  while (e) {
    if (e & 1U) r = static_cast<W>(static_cast<U<T>>(r * b));
    b = static_cast<W>(static_cast<U<T>>(b * b));
    e = static_cast<U<T>>(e >> 1U);
  }
  return static_cast<T>(static_cast<U<T>>(r));
}

template <typename T>
T power(T a, T b) noexcept {
  if constexpr (std::is_integral_v<T>) return ipow(a, b);
  else return std::pow(a, b);
}

template <typename T>
T absolute(T a) noexcept {
  if constexpr (std::is_same_v<T, bool> || std::is_unsigned_v<T>) return a;
  else if constexpr (std::is_integral_v<T>) return a < 0 ? static_cast<T>(U<T>{0} - static_cast<U<T>>(a)) : a;
  else return std::fabs(a);
}

template <typename T>
T negative(T a) noexcept {
  if constexpr (std::is_integral_v<T>) return static_cast<T>(U<T>{0} - static_cast<U<T>>(a));
  else return -a;
}

}  // namespace nativpy::kernels
