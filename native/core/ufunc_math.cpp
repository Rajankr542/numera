#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <numbers>

#include "complex_kernels.hpp"
#include "ufunc_kernels.hpp"
#include "ufunc_loops.hpp"
#include "ufunc_registry.hpp"

namespace nativpy {

// P4: trig, exp/log, rounding, arithmetic, float bits, integer ufuncs (D-056).
// Add entries to the table below; find_ufunc picks them up by name.

using namespace ufunc_loops;

namespace {

// ---- kernels ----

// One functor for both the real and complex loops (std:: overloads).
#define P04_STD_UN(Name, fn) \
  struct Name { template <typename T> T operator()(T v) const noexcept { return std::fn(v); } };
#define P04_STD_BIN(Name, fn) \
  struct Name { template <typename T> T operator()(T a, T b) const noexcept { return std::fn(a, b); } };

P04_STD_UN(SinF, sin)
P04_STD_UN(CosF, cos)
P04_STD_UN(TanF, tan)
P04_STD_UN(ArcsinF, asin)
P04_STD_UN(ArccosF, acos)
P04_STD_UN(ArctanF, atan)
P04_STD_UN(SinhF, sinh)
P04_STD_UN(CoshF, cosh)
P04_STD_UN(TanhF, tanh)
P04_STD_UN(ArcsinhF, asinh)
P04_STD_UN(ArccoshF, acosh)
P04_STD_UN(ArctanhF, atanh)
P04_STD_BIN(Arctan2F, atan2)
P04_STD_BIN(HypotF, hypot)

// P4-2 exp / log. Complex forms follow NumPy's nc_* helpers (funcs.inc.src).
P04_STD_UN(Exp2F, exp2)
P04_STD_UN(Expm1F, expm1)
P04_STD_UN(Log2F, log2)
P04_STD_UN(Log10F, log10)
P04_STD_UN(Log1pF, log1p)
P04_STD_UN(CbrtF, cbrt)
struct CExp2F {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept {
    return std::exp(z * std::numbers::ln2_v<R>);
  }
};
struct CExpm1F {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept {
    const R a = std::sin(z.imag() / 2);
    return {std::expm1(z.real()) * std::cos(z.imag()) - 2 * a * a, std::exp(z.real()) * std::sin(z.imag())};
  }
};
struct CLog2F {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept {
    return kernels::clog<R>(z) / std::numbers::ln2_v<R>;
  }
};
struct CLog10F {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept {
    return kernels::clog<R>(z) / std::numbers::ln10_v<R>;
  }
};
struct CLog1pF {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept {
    const R x = z.real() + 1;
    return {std::log(std::hypot(x, z.imag())), std::atan2(z.imag(), x)};
  }
};
// npy_logaddexp / npy_logaddexp2.
struct LogaddexpF {
  template <typename T> T operator()(T x, T y) const noexcept {
    if (x == y) return x + std::numbers::ln2_v<T>;  // also equal infinities
    const T d = x - y;
    if (d > 0) return x + std::log1p(std::exp(-d));
    if (d <= 0) return y + std::log1p(std::exp(d));
    return d;  // nan
  }
};
struct Logaddexp2F {
  template <typename T> T operator()(T x, T y) const noexcept {
    if (x == y) return x + 1;
    const T d = x - y;
    if (d > 0) return x + std::numbers::log2e_v<T> * std::log1p(std::exp2(-d));
    if (d <= 0) return y + std::numbers::log2e_v<T> * std::log1p(std::exp2(d));
    return d;
  }
};
// Integers multiply modularly; complex uses NumPy's complex multiply.
struct SquareF {
  template <typename T> T operator()(T v) const noexcept { return kernels::mul<T>(v, v); }
};
struct CSquareF {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept { return kernels::cmul<R>(z, z); }
};
// Integer reciprocal is NumPy's 1.0 / x cast back (so 1 / 0 saturates like
// the float -> int cast, and raises divide-by-zero).
struct ReciprocalF {
  template <typename T> T operator()(T v) const noexcept {
    if constexpr (std::is_integral_v<T>) {
      return float_to_int<T>(1.0 / static_cast<double>(v));
    } else {
      return T{1} / v;
    }
  }
};
struct CReciprocalF {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept {
    return kernels::cdiv<R>(std::complex<R>{1, 0}, z);
  }
};
// P4-3 rounding. Integer/bool loops are the identity (NumPy).
struct FloorF {
  template <typename T> T operator()(T v) const noexcept {
    if constexpr (std::is_floating_point_v<T>) return std::floor(v); else return v;
  }
};
struct CeilF {
  template <typename T> T operator()(T v) const noexcept {
    if constexpr (std::is_floating_point_v<T>) return std::ceil(v); else return v;
  }
};
struct TruncF {
  template <typename T> T operator()(T v) const noexcept {
    if constexpr (std::is_floating_point_v<T>) return std::trunc(v); else return v;
  }
};
P04_STD_UN(RintF, nearbyint)
struct CRintF {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept {
    return {std::nearbyint(z.real()), std::nearbyint(z.imag())};
  }
};
struct IdentityF {
  template <typename T> T operator()(T v) const noexcept { return v; }
};
#undef P04_STD_UN
#undef P04_STD_BIN

struct Deg2radF {
  template <typename T> T operator()(T v) const noexcept { return v * static_cast<T>(std::numbers::pi / 180.0); }
};
struct Rad2degF {
  template <typename T> T operator()(T v) const noexcept { return v * static_cast<T>(180.0 / std::numbers::pi); }
};

// ---- type resolution ----

bool is_float(DType d) noexcept { return dtype_info(d).kind == 'f'; }

// Unary float ufunc with complex loops (sin, exp2, ...): ints -> float_for.
LoopTypes r_float_c(DType a, DType) { return same(is_complex(a) ? a : float_for(a)); }

// Unary/binary float ufuncs without complex loops. The binary loop is the
// first float loop both inputs cast to safely.
template <const char* Name>
LoopTypes r_float1(DType a, DType) {
  if (is_complex(a)) bad_loop(Name, a);
  return same(float_for(a));
}
template <const char* Name>
LoopTypes r_float2(DType a, DType b) {
  const DType p = promote_types(float_for(a), float_for(b));
  if (is_complex(a) || is_complex(b)) bad_loop(Name, promote_types(a, b));
  return same(p);
}

template <const char* Name>
LoopTypes d_inexact(DType, DType d) {
  if (!is_inexact(d)) no_loop(Name);
  return same(d);
}
template <const char* Name>
LoopTypes d_float(DType, DType d) {
  if (!is_float(d)) no_loop(Name);
  return same(d);
}

// Integer-preserving unary (square, reciprocal): bool -> int8 (no bool loop).
LoopTypes r_no_bool(DType a, DType) { return same(a == DType::Bool ? DType::Int8 : a); }
// positive / sign: NumPy has no bool loop and bool does not cast to one safely.
template <const char* Name>
LoopTypes r_no_bool_strict(DType a, DType) {
  if (a == DType::Bool) bad_loop(Name, a);
  return same(a);
}
template <const char* Name>
LoopTypes d_not_bool(DType, DType d) {
  if (d == DType::Bool) no_loop(Name);
  return same(d);
}

// floor/ceil/trunc keep integer and bool dtypes; no complex loop.
template <const char* Name>
LoopTypes r_keep_real(DType a, DType) {
  if (is_complex(a)) bad_loop(Name, a);
  return same(a);
}
template <const char* Name>
LoopTypes d_real(DType, DType d) {
  if (is_complex(d)) no_loop(Name);
  return same(d);
}

using V = void;  // no complex loop

#define P04_NAME(id, str) constexpr char id[] = str;
P04_NAME(kSin, "sin")
P04_NAME(kCos, "cos")
P04_NAME(kTan, "tan")
P04_NAME(kArcsin, "arcsin")
P04_NAME(kArccos, "arccos")
P04_NAME(kArctan, "arctan")
P04_NAME(kSinh, "sinh")
P04_NAME(kCosh, "cosh")
P04_NAME(kTanh, "tanh")
P04_NAME(kArcsinh, "arcsinh")
P04_NAME(kArccosh, "arccosh")
P04_NAME(kArctanh, "arctanh")
P04_NAME(kArctan2, "arctan2")
P04_NAME(kHypot, "hypot")
P04_NAME(kDeg2rad, "deg2rad")
P04_NAME(kRad2deg, "rad2deg")
P04_NAME(kRadians, "radians")
P04_NAME(kDegrees, "degrees")
P04_NAME(kExp2, "exp2")
P04_NAME(kExpm1, "expm1")
P04_NAME(kLog2, "log2")
P04_NAME(kLog10, "log10")
P04_NAME(kLog1p, "log1p")
P04_NAME(kLogaddexp, "logaddexp")
P04_NAME(kLogaddexp2, "logaddexp2")
P04_NAME(kCbrt, "cbrt")
P04_NAME(kSquare, "square")
P04_NAME(kReciprocal, "reciprocal")
P04_NAME(kFloor, "floor")
P04_NAME(kCeil, "ceil")
P04_NAME(kTrunc, "trunc")
P04_NAME(kRint, "rint")
P04_NAME(kPositive, "positive")
#undef P04_NAME

// Unary float ufunc with a complex loop sharing the same functor.
template <const char* Name, typename F, typename CF = F>
constexpr Ufunc float_c_unary() {
  return {Name, 1, std::nullopt, r_float_c, d_inexact<Name>, nullptr, {},
          unary_table<F, Avail::FloatOnly, CF>()};
}
template <const char* Name, typename F>
constexpr Ufunc float_unary() {
  return {Name, 1, std::nullopt, r_float1<Name>, d_float<Name>, nullptr, {},
          unary_table<F, Avail::FloatOnly, V>()};
}
template <const char* Name, typename F>
constexpr Ufunc keep_real_unary() {
  return {Name, 1, std::nullopt, r_keep_real<Name>, d_real<Name>, nullptr, {},
          unary_table<F, Avail::All, V>()};
}
template <const char* Name, typename F>
constexpr Ufunc float_binary(std::optional<double> identity = std::nullopt) {
  return {Name, 2, identity, r_float2<Name>, d_float<Name>, nullptr,
          binary_table<F, Avail::FloatOnly, V>(), {}};
}

constexpr std::array kTable{
    // P4-1 trig / hyperbolic
    float_c_unary<kSin, SinF>(),
    float_c_unary<kCos, CosF>(),
    float_c_unary<kTan, TanF>(),
    float_c_unary<kArcsin, ArcsinF>(),
    float_c_unary<kArccos, ArccosF>(),
    float_c_unary<kArctan, ArctanF>(),
    float_c_unary<kSinh, SinhF>(),
    float_c_unary<kCosh, CoshF>(),
    float_c_unary<kTanh, TanhF>(),
    float_c_unary<kArcsinh, ArcsinhF>(),
    float_c_unary<kArccosh, ArccoshF>(),
    float_c_unary<kArctanh, ArctanhF>(),
    float_binary<kArctan2, Arctan2F>(),
    float_binary<kHypot, HypotF>(0.0),
    float_unary<kDeg2rad, Deg2radF>(),
    float_unary<kRad2deg, Rad2degF>(),
    float_unary<kRadians, Deg2radF>(),
    float_unary<kDegrees, Rad2degF>(),
    // P4-2 exp / log
    float_c_unary<kExp2, Exp2F, CExp2F>(),
    float_c_unary<kExpm1, Expm1F, CExpm1F>(),
    float_c_unary<kLog2, Log2F, CLog2F>(),
    float_c_unary<kLog10, Log10F, CLog10F>(),
    float_c_unary<kLog1p, Log1pF, CLog1pF>(),
    float_binary<kLogaddexp, LogaddexpF>(-std::numeric_limits<double>::infinity()),
    float_binary<kLogaddexp2, Logaddexp2F>(-std::numeric_limits<double>::infinity()),
    float_unary<kCbrt, CbrtF>(),
    Ufunc{kSquare, 1, std::nullopt, r_no_bool, d_not_bool<kSquare>, nullptr, {},
          unary_table<SquareF, Avail::NotBool, CSquareF>()},
    Ufunc{kReciprocal, 1, std::nullopt, r_no_bool, d_not_bool<kReciprocal>, nullptr, {},
          unary_table<ReciprocalF, Avail::NotBool, CReciprocalF>()},
    // P4-3 rounding
    keep_real_unary<kFloor, FloorF>(),
    keep_real_unary<kCeil, CeilF>(),
    keep_real_unary<kTrunc, TruncF>(),
    float_c_unary<kRint, RintF, CRintF>(),
    // P4-4 arithmetic
    Ufunc{kPositive, 1, std::nullopt, r_no_bool_strict<kPositive>, d_not_bool<kPositive>, nullptr, {},
          unary_table<IdentityF, Avail::NotBool, IdentityF>()},
};

}  // namespace

std::span<const Ufunc> math_ufuncs() noexcept { return kTable; }

}  // namespace nativpy
