#include <cfenv>
#include <cmath>
#include <complex>

#include "ufunc_loops.hpp"
#include "ufunc_registry.hpp"

namespace nativpy {

// P5: comparison, logical, isnan/isinf/isfinite, bitwise ufuncs (D-056).
// Add entries to the table below; find_ufunc picks them up by name.

using namespace ufunc_loops;

namespace {

// ---- bool-output loop templates (D-080) ----

// Binary loop reading S and writing bool; F(compute_t<S>, compute_t<S>) -> bool.
template <typename S, typename F>
void bool_bloop(const BroadcastPlan<3>& p, std::array<std::byte*, 3> base) {
  run_plan(p, base, [](const std::array<std::byte*, 3>& ptr, const std::array<std::int64_t, 3>& is,
                       std::int64_t n) {
    const F f{};
    constexpr auto sz = static_cast<std::int64_t>(sizeof(S));
    if (is[0] == 1 && is[1] == sz && is[2] == sz) {
      for (std::int64_t i = 0; i < n; ++i) {
        store<bool>(ptr[0] + i, f(ld<S>(ptr[1] + i * sz), ld<S>(ptr[2] + i * sz)));
      }
    } else {
      for (std::int64_t i = 0; i < n; ++i) {
        store<bool>(ptr[0] + i * is[0], f(ld<S>(ptr[1] + i * is[1]), ld<S>(ptr[2] + i * is[2])));
      }
    }
  });
}

// Bool-output binary table: RealF for every non-complex dtype allowed by RA,
// CplxF (void = none) for complex.
template <typename RealF, Avail RA, typename CplxF>
constexpr std::array<BinaryLoopFn, kNumDTypes> bool_binary_table() {
  std::array<BinaryLoopFn, kNumDTypes> t{};
  [&]<int... I>(std::integer_sequence<int, I...>) {
    ((t[I] = [] {
       using S = dtype_t<static_cast<DType>(I)>;
       if constexpr (is_complex_v<S>) {
         if constexpr (std::is_void_v<CplxF>) return BinaryLoopFn{nullptr};
         else return BinaryLoopFn{&bool_bloop<S, CplxF>};
       } else if constexpr (real_ok<S>(RA)) {
         return BinaryLoopFn{&bool_bloop<S, RealF>};
       } else {
         return BinaryLoopFn{nullptr};
       }
     }()),
     ...);
  }(std::make_integer_sequence<int, kNumDTypes>{});
  return t;
}

// Bool-output unary table (uloop handles In != Out).
template <typename RealF, Avail RA, typename CplxF>
constexpr std::array<UnaryLoopFn, kNumDTypes> bool_unary_table() {
  std::array<UnaryLoopFn, kNumDTypes> t{};
  [&]<int... I>(std::integer_sequence<int, I...>) {
    ((t[I] = [] {
       using S = dtype_t<static_cast<DType>(I)>;
       if constexpr (is_complex_v<S>) {
         if constexpr (std::is_void_v<CplxF>) return UnaryLoopFn{nullptr};
         else return UnaryLoopFn{&uloop<S, bool, CplxF>};
       } else if constexpr (real_ok<S>(RA)) {
         return UnaryLoopFn{&uloop<S, bool, RealF>};
       } else {
         return UnaryLoopFn{nullptr};
       }
     }()),
     ...);
  }(std::make_integer_sequence<int, kNumDTypes>{});
  return t;
}

// ---- comparison kernels ----

struct EqF { template <typename T> bool operator()(T a, T b) const noexcept { return a == b; } };
struct NeF { template <typename T> bool operator()(T a, T b) const noexcept { return a != b; } };
struct LtF { template <typename T> bool operator()(T a, T b) const noexcept { return a < b; } };
struct LeF { template <typename T> bool operator()(T a, T b) const noexcept { return a <= b; } };
struct GtF { template <typename T> bool operator()(T a, T b) const noexcept { return a > b; } };
struct GeF { template <typename T> bool operator()(T a, T b) const noexcept { return a >= b; } };

// Ordered compare that raises "invalid" on NaN, like the C comparisons in
// NumPy's complex CLT/CLE/CGT/CGE macros (D-080).
template <typename Op, typename R>
bool ordered(R x, R y) noexcept {
  if (std::isnan(x) || std::isnan(y)) {
    std::feraiseexcept(FE_INVALID);
    return false;
  }
  return Op{}(x, y);
}

// NumPy lexicographic complex order: real parts first, then imaginary parts.
template <typename RealOp, typename ImagOp>
struct ComplexOrderF {
  template <typename R>
  bool operator()(std::complex<R> a, std::complex<R> b) const noexcept {
    const R xr = a.real(), xi = a.imag(), yr = b.real(), yi = b.imag();
    return (ordered<RealOp>(xr, yr) && xi == xi && yi == yi) || (xr == yr && ordered<ImagOp>(xi, yi));
  }
};
using CLtF = ComplexOrderF<LtF, LtF>;
using CLeF = ComplexOrderF<LtF, LeF>;
using CGtF = ComplexOrderF<GtF, GtF>;
using CGeF = ComplexOrderF<GtF, GeF>;

// ---- type resolution ----

// Comparisons / logical ops: loop in the promoted dtype, bool output.
LoopTypes r_to_bool(DType a, DType b) { return {promote_types(a, b), DType::Bool}; }

// dtype= is NumPy's output signature: only bool. Used by reduce (the TS call
// wrapper validates dtype= itself and keeps the default loop, D-080).
template <const char* Name>
LoopTypes d_bool_binary(DType, DType d) {
  if (d != DType::Bool) no_loop(Name);
  return same(DType::Bool);
}

constexpr char kEqual[] = "equal";
constexpr char kNotEqual[] = "notEqual";
constexpr char kLess[] = "less";
constexpr char kLessEqual[] = "lessEqual";
constexpr char kGreater[] = "greater";
constexpr char kGreaterEqual[] = "greaterEqual";

constexpr std::array<Ufunc, 6> kTable{{
    {kEqual, 2, std::nullopt, r_to_bool, d_bool_binary<kEqual>, nullptr,
     bool_binary_table<EqF, Avail::All, EqF>(), {}},
    {kNotEqual, 2, std::nullopt, r_to_bool, d_bool_binary<kNotEqual>, nullptr,
     bool_binary_table<NeF, Avail::All, NeF>(), {}},
    {kLess, 2, std::nullopt, r_to_bool, d_bool_binary<kLess>, nullptr,
     bool_binary_table<LtF, Avail::All, CLtF>(), {}},
    {kLessEqual, 2, std::nullopt, r_to_bool, d_bool_binary<kLessEqual>, nullptr,
     bool_binary_table<LeF, Avail::All, CLeF>(), {}},
    {kGreater, 2, std::nullopt, r_to_bool, d_bool_binary<kGreater>, nullptr,
     bool_binary_table<GtF, Avail::All, CGtF>(), {}},
    {kGreaterEqual, 2, std::nullopt, r_to_bool, d_bool_binary<kGreaterEqual>, nullptr,
     bool_binary_table<GeF, Avail::All, CGeF>(), {}},
}};

}  // namespace

std::span<const Ufunc> logic_ufuncs() noexcept { return kTable; }

}  // namespace nativpy
