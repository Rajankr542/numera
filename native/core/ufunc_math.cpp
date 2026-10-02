#include <cmath>
#include <complex>
#include <numbers>

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
#undef P04_NAME

// Unary float ufunc with a complex loop sharing the same functor.
template <const char* Name, typename F>
constexpr Ufunc float_c_unary() {
  return {Name, 1, std::nullopt, r_float_c, d_inexact<Name>, nullptr, {},
          unary_table<F, Avail::FloatOnly, F>()};
}
template <const char* Name, typename F>
constexpr Ufunc float_unary() {
  return {Name, 1, std::nullopt, r_float1<Name>, d_float<Name>, nullptr, {},
          unary_table<F, Avail::FloatOnly, V>()};
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
};

}  // namespace

std::span<const Ufunc> math_ufuncs() noexcept { return kTable; }

}  // namespace nativpy
