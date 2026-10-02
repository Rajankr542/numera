#include <cmath>
#include <complex>
#include <cstdint>
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
// P4-4 arithmetic.
// C fmod: sign follows the dividend; integer x % 0 -> 0 (+ divide-by-zero).
struct FmodF {
  template <typename T> T operator()(T a, T b) const noexcept {
    if constexpr (std::is_same_v<T, bool>) {
      return a && !b;  // unreachable: bool inputs use the int8 loop
    } else if constexpr (std::is_integral_v<T>) {
      if (b == 0) {
        raise_fp_divbyzero();
        return 0;
      }
      if constexpr (std::is_signed_v<T>) {
        if (b == -1) return 0;
      }
      return static_cast<T>(a % b);
    } else {
      return std::fmod(a, b);
    }
  }
};
struct FloatPowerF {
  template <typename T> T operator()(T a, T b) const noexcept { return std::pow(a, b); }
};
struct CFloatPowerF {
  template <typename R> std::complex<R> operator()(std::complex<R> a, std::complex<R> b) const noexcept {
    return kernels::cpow<R>(a, b);
  }
};
// NumPy sign: -1/0/1, NaN stays NaN (and -0 gives +0).
struct SignF {
  template <typename T> T operator()(T v) const noexcept {
    if constexpr (std::is_unsigned_v<T>) {
      return v > 0 ? T{1} : T{0};
    } else if constexpr (std::is_integral_v<T>) {
      return v > 0 ? T{1} : (v < 0 ? T{-1} : T{0});
    } else {
      return v > 0 ? T{1} : (v < 0 ? T{-1} : (v == 0 ? T{0} : v));
    }
  }
};
// NumPy 2 complex sign: z / |z| with the CDOUBLE_sign special cases.
struct CSignF {
  template <typename R> std::complex<R> operator()(std::complex<R> z) const noexcept {
    const R re = z.real(), im = z.imag();
    const R a = std::hypot(re, im);
    const R nan = std::numeric_limits<R>::quiet_NaN();
    if (std::isnan(a)) return {nan, nan};
    if (std::isinf(a)) {
      if (std::isinf(re)) {
        if (std::isinf(im)) return {nan, nan};
        return {std::copysign(R{1}, re), R{0}};
      }
      return {R{0}, std::copysign(R{1}, im)};
    }
    if (a == R{0}) return {R{0}, R{0}};
    return {re / a, im / a};
  }
};
struct HeavisideF {
  template <typename T> T operator()(T x, T h0) const noexcept {
    if (std::isnan(x)) return x;
    if (x == 0) return h0;
    return x < 0 ? T{0} : T{1};
  }
};
// maximum/minimum propagate NaN; fmax/fmin ignore it. Signed zeros: max
// prefers +0, min prefers -0 (NumPy's SIMD loops).
template <bool Max, bool PropagateNan>
struct ExtremumF {
  template <typename T> T operator()(T a, T b) const noexcept {
    if constexpr (std::is_same_v<T, bool>) {
      return Max ? (a || b) : (a && b);
    } else if constexpr (std::is_integral_v<T>) {
      return Max ? (a >= b ? a : b) : (a <= b ? a : b);
    } else {
      if (std::isnan(a)) return PropagateNan ? a : b;
      if (std::isnan(b)) return PropagateNan ? b : a;
      if (a == b) return (std::signbit(a) == Max) ? b : a;
      return Max ? (a > b ? a : b) : (a < b ? a : b);
    }
  }
};
// Complex: lexicographic order (NumPy CGE/CLE), NaN rules as for floats.
template <bool Max, bool PropagateNan>
struct CExtremumF {
  template <typename R> std::complex<R> operator()(std::complex<R> a, std::complex<R> b) const noexcept {
    const auto has_nan = [](std::complex<R> z) { return std::isnan(z.real()) || std::isnan(z.imag()); };
    const auto ge = [](std::complex<R> x, std::complex<R> y) {
      return (x.real() > y.real() && !std::isnan(x.imag()) && !std::isnan(y.imag())) ||
             (x.real() == y.real() && x.imag() >= y.imag());
    };
    const auto le = [](std::complex<R> x, std::complex<R> y) {
      return (x.real() < y.real() && !std::isnan(x.imag()) && !std::isnan(y.imag())) ||
             (x.real() == y.real() && x.imag() <= y.imag());
    };
    const bool keep_a = Max ? ge(a, b) : le(a, b);
    if constexpr (PropagateNan) {
      return (has_nan(a) || keep_a) ? a : b;
    } else {
      return (has_nan(b) || keep_a) ? a : b;
    }
  }
};
struct FabsF {
  template <typename T> T operator()(T v) const noexcept { return std::fabs(v); }
};
// P4-5 float bits / integer.
P04_STD_BIN(CopysignF, copysign)
// Exponent arrives cast to the float loop type (D-073); clamp before int.
struct LdexpF {
  template <typename T> T operator()(T x, T n) const noexcept {
    const T c = std::isnan(n) ? T{0} : std::fmax(std::fmin(n, T{100000}), T{-100000});
    return std::ldexp(x, static_cast<int>(c));
  }
};
struct NextafterF {
  template <typename T> T operator()(T x, T y) const noexcept { return std::nextafter(x, y); }
};
// NumPy npy_spacing: distance to the next value away from zero (zero -> the
// smallest subnormal); inf/nan -> nan; the largest finite -> inf (overflow).
struct SpacingF {
  template <typename T> T operator()(T x) const noexcept {
    if (!std::isfinite(x)) return std::numeric_limits<T>::quiet_NaN();
    if (x == T{0}) return std::numeric_limits<T>::denorm_min();
    const T inf = std::numeric_limits<T>::infinity();
    return std::nextafter(x, std::signbit(x) ? -inf : inf) - x;
  }
};
// float16 versions on the bit pattern (npy_half_nextafter / npy_half_spacing).
// The loop computes float16 as float; values are exact halves.
inline std::uint16_t half_bits(float v) noexcept { return double_to_half(static_cast<double>(v)).bits; }
inline float half_value(std::uint16_t b) noexcept { return static_cast<float>(half_to_double(float16_t{b})); }
inline float half_next(float x, float y) noexcept {
  if (std::isnan(x) || std::isnan(y)) return std::numeric_limits<float>::quiet_NaN();
  if (x == y) return y;
  const std::uint16_t hx = half_bits(x);
  std::uint16_t r;
  if ((hx & 0x7fffu) == 0) {
    r = static_cast<std::uint16_t>((half_bits(y) & 0x8000u) | 1u);
  } else if ((x < y) != std::signbit(x)) {
    r = static_cast<std::uint16_t>(hx + 1u);  // away from zero
  } else {
    r = static_cast<std::uint16_t>(hx - 1u);
  }
  const float out = half_value(r);
  if (std::isinf(out) && std::isfinite(x)) raise_fp_overflow();
  return out;
}
struct HalfNextafterF {
  float operator()(float x, float y) const noexcept { return half_next(x, y); }
};
// NumPy's half spacing steps toward +inf for every finite value.
struct HalfSpacingF {
  float operator()(float x) const noexcept {
    if (!std::isfinite(x)) return std::numeric_limits<float>::quiet_NaN();
    return half_next(x, std::numeric_limits<float>::infinity()) - x;
  }
};
struct SignbitF {
  template <typename T> bool operator()(T x) const noexcept { return std::signbit(x); }
};
// gcd / lcm on magnitudes (modular, like NumPy: gcd(INT_MIN, 0) == INT_MIN).
template <typename T>
std::make_unsigned_t<T> umag(T v) noexcept {
  using UT = std::make_unsigned_t<T>;
  if constexpr (std::is_signed_v<T>) {
    return v < 0 ? static_cast<UT>(UT{0} - static_cast<UT>(v)) : static_cast<UT>(v);
  } else {
    return v;
  }
}
template <typename UT>
UT ugcd(UT a, UT b) noexcept {
  while (b != 0) {
    const UT t = static_cast<UT>(a % b);
    a = b;
    b = t;
  }
  return a;
}
struct GcdF {
  template <typename T> T operator()(T a, T b) const noexcept {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      return static_cast<T>(ugcd(umag(a), umag(b)));
    } else {
      return a;  // no loop is registered for these types
    }
  }
};
struct LcmF {
  template <typename T> T operator()(T a, T b) const noexcept {
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      using UT = std::make_unsigned_t<T>;
      const UT ua = umag(a), ub = umag(b);
      const UT g = ugcd(ua, ub);
      if (g == 0) return 0;
      return static_cast<T>(kernels::mul<UT>(static_cast<UT>(ua / g), ub));
    } else {
      return a;
    }
  }
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

// fmod: NumPy loops are integer and float (bool -> int8), no complex.
template <const char* Name>
LoopTypes r_int_float(DType a, DType b) {
  const DType p = promote_types(a, b);
  if (is_complex(p)) bad_loop(Name, p);
  return same(p == DType::Bool ? DType::Int8 : p);
}
template <const char* Name>
LoopTypes d_int_float(DType, DType d) {
  if (d == DType::Bool || is_complex(d)) no_loop(Name);
  return same(d);
}
// float_power: only float64 and complex128 loops.
LoopTypes r_float_power(DType a, DType b) {
  return same(is_complex(promote_types(a, b)) ? DType::Complex128 : DType::Float64);
}
template <const char* Name>
LoopTypes d_float_power(DType, DType d) {
  if (d != DType::Float64 && d != DType::Complex128) no_loop(Name);
  return same(d);
}
// maximum/minimum/fmax/fmin: every dtype, plain promotion.
LoopTypes r_promote(DType a, DType b) { return same(promote_types(a, b)); }
LoopTypes d_any(DType, DType d) { return same(d); }

// ldexp: x picks the float loop; the exponent must be an integer that casts
// safely to int64 (NumPy's `fi`/`fl` loops). Both are cast to the float type.
template <const char* Name>
LoopTypes r_ldexp(DType a, DType b) {
  const char kb = dtype_info(b).kind;
  if (is_complex(a) || !(kb == 'b' || kb == 'i' || (kb == 'u' && b != DType::UInt64))) {
    bad_loop(Name, promote_types(a, b));
  }
  return same(float_for(a));
}
// signbit: float input, bool output.
template <const char* Name>
LoopTypes r_signbit(DType a, DType) {
  if (is_complex(a)) bad_loop(Name, a);
  return {float_for(a), DType::Bool};
}
template <const char* Name>
LoopTypes d_signbit(DType a, DType d) {
  if (d != DType::Bool || is_complex(a)) no_loop(Name);
  return {float_for(a), DType::Bool};
}
// gcd / lcm: integer loops only.
template <const char* Name>
LoopTypes r_int_only(DType a, DType b) {
  const DType p = promote_types(a, b);
  if (!is_integer(p)) bad_loop(Name, p);
  return same(p);
}
template <const char* Name>
LoopTypes d_int_only(DType, DType d) {
  if (!is_integer(d)) no_loop(Name);
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
P04_NAME(kFmod, "fmod")
P04_NAME(kFloatPower, "float_power")
P04_NAME(kSign, "sign")
P04_NAME(kHeaviside, "heaviside")
P04_NAME(kMaximum, "maximum")
P04_NAME(kMinimum, "minimum")
P04_NAME(kFmax, "fmax")
P04_NAME(kFmin, "fmin")
P04_NAME(kFabs, "fabs")
P04_NAME(kCopysign, "copysign")
P04_NAME(kLdexp, "ldexp")
P04_NAME(kNextafter, "nextafter")
P04_NAME(kSpacing, "spacing")
P04_NAME(kSignbit, "signbit")
P04_NAME(kGcd, "gcd")
P04_NAME(kLcm, "lcm")
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

constexpr int kF16 = static_cast<int>(DType::Float16);

constexpr std::array<BinaryLoopFn, kNumDTypes> nextafter_loops() {
  auto t = binary_table<NextafterF, Avail::FloatOnly, V>();
  t[kF16] = &bloop<float16_t, HalfNextafterF>;
  return t;
}
constexpr std::array<UnaryLoopFn, kNumDTypes> spacing_loops() {
  auto t = unary_table<SpacingF, Avail::FloatOnly, V>();
  t[kF16] = &uloop<float16_t, float16_t, HalfSpacingF>;
  return t;
}
constexpr std::array<UnaryLoopFn, kNumDTypes> signbit_loops() {
  std::array<UnaryLoopFn, kNumDTypes> t{};
  t[kF16] = &uloop<float16_t, bool, SignbitF>;
  t[static_cast<int>(DType::Float32)] = &uloop<float, bool, SignbitF>;
  t[static_cast<int>(DType::Float64)] = &uloop<double, bool, SignbitF>;
  return t;
}
constexpr std::array<BinaryLoopFn, kNumDTypes> int_table_gcd(bool lcm) {
  auto t = lcm ? binary_table<LcmF, Avail::NotBool, V>() : binary_table<GcdF, Avail::NotBool, V>();
  for (std::size_t i = kF16; i <= static_cast<std::size_t>(DType::Float64); ++i) t[i] = nullptr;
  return t;
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
    Ufunc{kSign, 1, std::nullopt, r_no_bool_strict<kSign>, d_not_bool<kSign>, nullptr, {},
          unary_table<SignF, Avail::NotBool, CSignF>()},
    float_unary<kFabs, FabsF>(),
    Ufunc{kFmod, 2, std::nullopt, r_int_float<kFmod>, d_int_float<kFmod>, nullptr,
          binary_table<FmodF, Avail::NotBool, V>(), {}},
    Ufunc{kFloatPower, 2, std::nullopt, r_float_power, d_float_power<kFloatPower>, nullptr,
          binary_table<FloatPowerF, Avail::FloatOnly, CFloatPowerF>(), {}},
    float_binary<kHeaviside, HeavisideF>(),
    Ufunc{kMaximum, 2, std::nullopt, r_promote, d_any, nullptr,
          binary_table<ExtremumF<true, true>, Avail::All, CExtremumF<true, true>>(), {}},
    Ufunc{kMinimum, 2, std::nullopt, r_promote, d_any, nullptr,
          binary_table<ExtremumF<false, true>, Avail::All, CExtremumF<false, true>>(), {}},
    Ufunc{kFmax, 2, std::nullopt, r_promote, d_any, nullptr,
          binary_table<ExtremumF<true, false>, Avail::All, CExtremumF<true, false>>(), {}},
    Ufunc{kFmin, 2, std::nullopt, r_promote, d_any, nullptr,
          binary_table<ExtremumF<false, false>, Avail::All, CExtremumF<false, false>>(), {}},
    // P4-5 float bits / integer
    float_binary<kCopysign, CopysignF>(),
    Ufunc{kLdexp, 2, std::nullopt, r_ldexp<kLdexp>, d_float<kLdexp>, nullptr,
          binary_table<LdexpF, Avail::FloatOnly, V>(), {}},
    Ufunc{kNextafter, 2, std::nullopt, r_float2<kNextafter>, d_float<kNextafter>, nullptr,
          nextafter_loops(), {}},
    Ufunc{kSpacing, 1, std::nullopt, r_float1<kSpacing>, d_float<kSpacing>, nullptr, {},
          spacing_loops()},
    Ufunc{kSignbit, 1, std::nullopt, r_signbit<kSignbit>, d_signbit<kSignbit>, nullptr, {},
          signbit_loops()},
    Ufunc{kGcd, 2, 0.0, r_int_only<kGcd>, d_int_only<kGcd>, nullptr, int_table_gcd(false), {}},
    Ufunc{kLcm, 2, std::nullopt, r_int_only<kLcm>, d_int_only<kLcm>, nullptr, int_table_gcd(true), {}},
};

}  // namespace

std::span<const Ufunc> math_ufuncs() noexcept { return kTable; }

}  // namespace nativpy
