#pragma once

// Shared ufunc loop templates and type-resolution helpers (D-051, D-056).
// Used by ufunc_registry.cpp and by the per-family ufunc tables
// (ufunc_math.cpp for P4, ufunc_logic.cpp for P5).

#include <array>
#include <complex>
#include <cstdint>
#include <string>
#include <type_traits>
#include <utility>

#include "broadcast.hpp"
#include "cast.hpp"
#include "dtype.hpp"
#include "error.hpp"
#include "ufunc_registry.hpp"

namespace nativpy::ufunc_loops {

// ---- loop templates ----

template <typename S>
using compute_t = std::conditional_t<std::is_same_v<S, float16_t>, float, S>;

template <typename S>
compute_t<S> ld(const std::byte* p) noexcept {
  if constexpr (std::is_same_v<S, float16_t>) {
    return static_cast<float>(half_to_double(load<float16_t>(p)));
  } else {
    return load<S>(p);
  }
}

template <typename S>
void st(std::byte* p, compute_t<S> v) noexcept {
  if constexpr (std::is_same_v<S, float16_t>) {
    store(p, double_to_half(static_cast<double>(v)));
  } else {
    store(p, v);
  }
}

// Strided binary loop over S with kernel F (a stateless callable type).
template <typename S, typename F>
void bloop(const BroadcastPlan<3>& p, std::array<std::byte*, 3> base) {
  run_plan(p, base, [](const std::array<std::byte*, 3>& ptr, const std::array<std::int64_t, 3>& is,
                       std::int64_t n) {
    const F f{};
    std::byte* o = ptr[0];
    const std::byte* a = ptr[1];
    const std::byte* b = ptr[2];
    constexpr auto sz = static_cast<std::int64_t>(sizeof(S));
    if (is[0] == sz && is[1] == sz && is[2] == sz) {
      for (std::int64_t i = 0; i < n; ++i) st<S>(o + i * sz, f(ld<S>(a + i * sz), ld<S>(b + i * sz)));
    } else if (is[0] == sz && is[1] == sz && is[2] == 0) {
      const auto bv = ld<S>(b);
      for (std::int64_t i = 0; i < n; ++i) st<S>(o + i * sz, f(ld<S>(a + i * sz), bv));
    } else {
      for (std::int64_t i = 0; i < n; ++i) st<S>(o + i * is[0], f(ld<S>(a + i * is[1]), ld<S>(b + i * is[2])));
    }
  });
}

// Strided unary loop reading In and writing Out (both S unless complex->real).
template <typename In, typename Out, typename F>
void uloop(const BroadcastPlan<2>& p, std::array<std::byte*, 2> base) {
  run_plan(p, base, [](const std::array<std::byte*, 2>& ptr, const std::array<std::int64_t, 2>& is,
                       std::int64_t n) {
    const F f{};
    constexpr auto si = static_cast<std::int64_t>(sizeof(In));
    constexpr auto so = static_cast<std::int64_t>(sizeof(Out));
    if (is[0] == so && is[1] == si) {
      for (std::int64_t i = 0; i < n; ++i) st<Out>(ptr[0] + i * so, f(ld<In>(ptr[1] + i * si)));
    } else {
      for (std::int64_t i = 0; i < n; ++i) st<Out>(ptr[0] + i * is[0], f(ld<In>(ptr[1] + i * is[1])));
    }
  });
}
// ---- loop tables ----

template <typename S> constexpr bool is_bool_v = std::is_same_v<S, bool>;
template <typename S> constexpr bool is_float_v = std::is_floating_point_v<compute_t<S>>;

// Loop availability per element type, mirroring the old switch statements.
enum class Avail : std::uint8_t { All, NotBool, FloatOnly, Never };

template <typename S>
constexpr bool real_ok(Avail a) {
  switch (a) {
    case Avail::All: return true;
    case Avail::NotBool: return !is_bool_v<S>;
    case Avail::FloatOnly: return is_float_v<S>;
    case Avail::Never: return false;
  }
  return false;
}

// Binary loop table: RealF for non-complex dtypes allowed by RA; CplxF (or
// void = none) for complex dtypes.
template <typename RealF, Avail RA, typename CplxF>
constexpr std::array<BinaryLoopFn, kNumDTypes> binary_table() {
  std::array<BinaryLoopFn, kNumDTypes> t{};
  [&]<int... I>(std::integer_sequence<int, I...>) {
    ((t[I] = [] {
       using S = dtype_t<static_cast<DType>(I)>;
       if constexpr (is_complex_v<S>) {
         if constexpr (std::is_void_v<CplxF>) return BinaryLoopFn{nullptr};
         else return BinaryLoopFn{&bloop<S, CplxF>};
       } else if constexpr (real_ok<S>(RA)) {
         return BinaryLoopFn{&bloop<S, RealF>};
       } else {
         return BinaryLoopFn{nullptr};
       }
     }()),
     ...);
  }(std::make_integer_sequence<int, kNumDTypes>{});
  return t;
}

// Unary loop table (same-type loops only; complex->real loops are patched in).
template <typename RealF, Avail RA, typename CplxF>
constexpr std::array<UnaryLoopFn, kNumDTypes> unary_table() {
  std::array<UnaryLoopFn, kNumDTypes> t{};
  [&]<int... I>(std::integer_sequence<int, I...>) {
    ((t[I] = [] {
       using S = dtype_t<static_cast<DType>(I)>;
       if constexpr (is_complex_v<S>) {
         if constexpr (std::is_void_v<CplxF>) return UnaryLoopFn{nullptr};
         else return UnaryLoopFn{&uloop<S, S, CplxF>};
       } else if constexpr (real_ok<S>(RA)) {
         return UnaryLoopFn{&uloop<S, S, RealF>};
       } else {
         return UnaryLoopFn{nullptr};
       }
     }()),
     ...);
  }(std::make_integer_sequence<int, kNumDTypes>{});
  return t;
}

// Complex -> real loops (abs, angle; D-033), indexed by the complex input.
template <typename F>
constexpr std::array<UnaryLoopFn, kNumDTypes> with_complex_to_real(
    std::array<UnaryLoopFn, kNumDTypes> t) {
  t[static_cast<int>(DType::Complex64)] = &uloop<std::complex<float>, float, F>;
  t[static_cast<int>(DType::Complex128)] = &uloop<std::complex<double>, double, F>;
  return t;
}


// ---- type resolution helpers ----

[[noreturn]] inline void bad_loop(const char* name, DType dt) {
  throw_error(ErrorKind::DType, std::string("no ") + name + " loop for dtype " +
                                    std::string(dtype_name(dt)));
}

[[noreturn]] inline void no_loop(const char* name) {
  throw_error(ErrorKind::DType,
              std::string("No loop matching the specified signature and casting was found for ufunc ") +
                  name);
}

inline bool is_inexact(DType dt) noexcept {
  const char k = dtype_info(dt).kind;
  return k == 'f' || k == 'c';
}

inline DType complex_real_dtype(DType dt) noexcept {
  return dt == DType::Complex64 ? DType::Float32 : DType::Float64;
}

// NumPy's float loop for integer inputs to sqrt/exp/log (smallest float that
// holds the integer type safely).
inline DType float_for(DType in) noexcept {
  switch (in) {
    case DType::Bool:
    case DType::Int8:
    case DType::UInt8: return DType::Float16;
    case DType::Int16:
    case DType::UInt16: return DType::Float32;
    case DType::Float16:
    case DType::Float32:
    case DType::Float64: return in;
    default: return DType::Float64;
  }
}

inline LoopTypes same(DType d) noexcept { return {d, d}; }

}  // namespace nativpy::ufunc_loops
