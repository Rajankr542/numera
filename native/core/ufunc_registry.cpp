#include "ufunc_registry.hpp"

#include <cmath>
#include <complex>
#include <string>
#include <type_traits>

#include "cast.hpp"
#include "error.hpp"
#include "complex_kernels.hpp"
#include "ufunc_kernels.hpp"

namespace nativpy {

namespace {

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

// ---- kernels as stateless callables ----

// Real kernels take compute_t<S>; complex kernels take std::complex<R>.
#define NATIVPY_BIN_F(Name, fn)                                                   \
  struct Name {                                                                   \
    template <typename T>                                                         \
    T operator()(T a, T b) const noexcept { return kernels::fn<T>(a, b); }        \
  };
#define NATIVPY_CBIN_F(Name, fn)                                                  \
  struct Name {                                                                   \
    template <typename R>                                                         \
    std::complex<R> operator()(std::complex<R> a, std::complex<R> b) const noexcept { \
      return kernels::fn<R>(a, b);                                                \
    }                                                                             \
  };
#define NATIVPY_UN_F(Name, fn)                                                    \
  struct Name {                                                                   \
    template <typename T>                                                         \
    T operator()(T a) const noexcept { return kernels::fn<T>(a); }                \
  };
#define NATIVPY_CUN_F(Name, fn)                                                   \
  struct Name {                                                                   \
    template <typename R>                                                         \
    std::complex<R> operator()(std::complex<R> a) const noexcept { return kernels::fn<R>(a); } \
  };

NATIVPY_BIN_F(AddF, add)
NATIVPY_BIN_F(SubF, sub)
NATIVPY_BIN_F(MulF, mul)
NATIVPY_BIN_F(DivF, div)
NATIVPY_BIN_F(PowF, power)
NATIVPY_BIN_F(ModF, mod)
NATIVPY_BIN_F(FloorDivF, floordiv)
NATIVPY_CBIN_F(CAddF, cadd)
NATIVPY_CBIN_F(CSubF, csub)
NATIVPY_CBIN_F(CMulF, cmul)
NATIVPY_CBIN_F(CDivF, cdiv)
NATIVPY_CBIN_F(CPowF, cpow)
NATIVPY_UN_F(AbsF, absolute)
NATIVPY_UN_F(NegF, negative)
NATIVPY_CUN_F(CNegF, cneg)
NATIVPY_CUN_F(CConjF, cconj)
NATIVPY_CUN_F(CSqrtF, csqrt)
NATIVPY_CUN_F(CLogF, clog)
#undef NATIVPY_BIN_F
#undef NATIVPY_CBIN_F
#undef NATIVPY_UN_F
#undef NATIVPY_CUN_F

struct SqrtF { template <typename T> T operator()(T v) const noexcept { return std::sqrt(v); } };
struct ExpF { template <typename T> T operator()(T v) const noexcept { return std::exp(v); } };
struct LogF { template <typename T> T operator()(T v) const noexcept { return std::log(v); } };
struct IdentityF { template <typename T> T operator()(T v) const noexcept { return v; } };
// NumPy angle of real input: arctan2(0, x).
struct RealAngleF {
  template <typename T> T operator()(T v) const noexcept { return std::atan2(T{0}, v); }
};
struct ComplexAngleF {
  template <typename T>
  T operator()(std::complex<T> v) const noexcept { return std::atan2(v.imag(), v.real()); }
};
struct ComplexAbsF {
  template <typename T>
  T operator()(std::complex<T> v) const noexcept { return kernels::cabs<T>(v); }
};

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



// ---- type resolution (unchanged NumPy semantics, D-014/D-033/D-048) ----

[[noreturn]] void bad_loop(const char* name, DType dt) {
  throw_error(ErrorKind::DType, std::string("no ") + name + " loop for dtype " +
                                    std::string(dtype_name(dt)));
}

[[noreturn]] void no_loop(const char* name) {
  throw_error(ErrorKind::DType,
              std::string("No loop matching the specified signature and casting was found for ufunc ") +
                  name);
}

bool is_inexact(DType dt) noexcept {
  const char k = dtype_info(dt).kind;
  return k == 'f' || k == 'c';
}

DType complex_real_dtype(DType dt) noexcept {
  return dt == DType::Complex64 ? DType::Float32 : DType::Float64;
}

// NumPy's float loop for integer inputs to sqrt/exp/log (smallest float that
// holds the integer type safely).
DType float_for(DType in) noexcept {
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

LoopTypes same(DType d) noexcept { return {d, d}; }

// Binary default resolvers.
LoopTypes r_promote(DType a, DType b) { return same(promote_types(a, b)); }
LoopTypes r_subtract(DType a, DType b) {
  const DType p = promote_types(a, b);
  if (p == DType::Bool) {
    throw_error(ErrorKind::DType, "boolean subtract is not supported, use logicalXor instead");
  }
  return same(p);
}
LoopTypes r_divide(DType a, DType b) {
  const DType p = promote_types(a, b);
  return same((p == DType::Bool || is_integer(p)) ? DType::Float64 : p);
}
LoopTypes r_power(DType a, DType b) {
  const DType p = promote_types(a, b);
  return same(p == DType::Bool ? DType::Int8 : p);
}
template <const char* Name>
LoopTypes r_no_complex(DType a, DType b) {  // mod, floorDivide
  const DType p = promote_types(a, b);
  if (is_complex(p)) bad_loop(Name, p);  // NumPy: TypeError, no complex loop (D-033)
  return same(p == DType::Bool ? DType::Int8 : p);
}

// Binary dtype= resolvers (D-048).
LoopTypes d_any(DType, DType d) { return same(d); }
LoopTypes d_subtract(DType, DType d) {
  if (d == DType::Bool) r_subtract(d, d);  // throws the boolean-subtract message
  return same(d);
}
template <const char* Name>
LoopTypes d_inexact(DType, DType d) {
  if (!is_inexact(d)) no_loop(Name);
  return same(d);
}
template <const char* Name>
LoopTypes d_not_bool(DType, DType d) {
  if (d == DType::Bool) no_loop(Name);
  return same(d);
}
template <const char* Name>
LoopTypes d_not_bool_complex(DType, DType d) {
  if (d == DType::Bool || is_complex(d)) no_loop(Name);
  return same(d);
}


// Unary default resolvers (`b` unused). abs/angle of complex read the complex
// input and write its real dtype (D-033).
LoopTypes r_abs(DType a, DType) {
  return is_complex(a) ? LoopTypes{a, complex_real_dtype(a)} : same(a);
}
LoopTypes r_negative(DType a, DType) {
  if (a == DType::Bool) {
    throw_error(ErrorKind::DType, "boolean negative is not supported, use logicalNot instead");
  }
  return same(a);
}
LoopTypes r_float(DType a, DType) { return same(is_complex(a) ? a : float_for(a)); }
LoopTypes r_conjugate(DType a, DType) {
  return same(a == DType::Bool ? DType::Int8 : a);  // NumPy: no bool loop
}
LoopTypes r_angle(DType a, DType) {
  if (is_complex(a)) return {a, complex_real_dtype(a)};
  return same(a == DType::Bool ? DType::Float64 : float_for(a));  // arctan2(0, x)
}

constexpr char kAbs[] = "abs";
constexpr char kSqrt[] = "sqrt";
constexpr char kExp[] = "exp";
constexpr char kLog[] = "log";
constexpr char kConjugate[] = "conjugate";
constexpr char kAngle[] = "angle";
constexpr char kDivide[] = "divide";
constexpr char kPower[] = "power";
constexpr char kMod[] = "mod";
constexpr char kFloorDivide[] = "floorDivide";

// Unary dtype= resolvers (D-048); `a` is the input dtype.
LoopTypes d_abs(DType a, DType d) {
  if (is_complex(d)) no_loop(kAbs);
  if (is_complex(a) && (d == DType::Float32 || d == DType::Float64)) {
    const DType c = d == DType::Float32 ? DType::Complex64 : DType::Complex128;
    if (can_cast(a, c, Casting::Safe)) return {c, d};  // NumPy F->f / D->d
  }
  return same(d);
}
LoopTypes d_negative(DType a, DType d) {
  if (d == DType::Bool) r_negative(d, a);  // throws the boolean-negative message
  return same(d);
}
[[noreturn]] LoopTypes d_angle(DType, DType) { no_loop(kAngle); }

// power: integers to negative integer powers raise (NumPy). With where= only
// true positions are checked (D-049).
void check_power(const NDArray&, const NDArray* bp, const NDArray* mask, const Shape& full) {
  const NDArray& b = *bp;
  if (!is_integer(b.dtype()) || dtype_info(b.dtype()).kind == 'u') return;
  dispatch_dtype(b.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (std::is_integral_v<S> && std::is_signed_v<S>) {
      const auto fail = [] {
        throw_error(ErrorKind::Value, "Integers to negative integer powers are not allowed.");
      };
      if (!mask) {
        for_each_element(b, [&](const std::byte* p) {
          if (load<S>(p) < 0) fail();
        });
        return;
      }
      const auto plan = make_plan<2>(full, {&b, mask});
      run_plan(plan, {b.data(), mask->data()},
               [&](const std::array<std::byte*, 2>& p, const std::array<std::int64_t, 2>& s,
                   std::int64_t n) {
                 for (std::int64_t i = 0; i < n; ++i) {
                   if (load<bool>(p[1] + i * s[1]) && load<S>(p[0] + i * s[0]) < 0) fail();
                 }
               });
    }
  });
}


// ---- the registry (order matches BinaryOp / UnaryOp) ----

using V = void;  // no complex loop

constexpr std::array<Ufunc, 7> kBinary{{
    {"add", 2, 0.0, r_promote, d_any, nullptr, binary_table<AddF, Avail::All, CAddF>(), {}},
    {"subtract", 2, std::nullopt, r_subtract, d_subtract, nullptr,
     binary_table<SubF, Avail::NotBool, CSubF>(), {}},
    {"multiply", 2, 1.0, r_promote, d_any, nullptr, binary_table<MulF, Avail::All, CMulF>(), {}},
    {"divide", 2, std::nullopt, r_divide, d_inexact<kDivide>, nullptr,
     binary_table<DivF, Avail::FloatOnly, CDivF>(), {}},
    {"power", 2, std::nullopt, r_power, d_not_bool<kPower>, check_power,
     binary_table<PowF, Avail::NotBool, CPowF>(), {}},
    {"mod", 2, std::nullopt, r_no_complex<kMod>, d_not_bool_complex<kMod>, nullptr,
     binary_table<ModF, Avail::NotBool, V>(), {}},
    {"floorDivide", 2, std::nullopt, r_no_complex<kFloorDivide>, d_not_bool_complex<kFloorDivide>,
     nullptr, binary_table<FloorDivF, Avail::NotBool, V>(), {}},
}};

constexpr std::array<Ufunc, 7> kUnary{{
    {"abs", 1, std::nullopt, r_abs, d_abs, nullptr, {},
     with_complex_to_real<ComplexAbsF>(unary_table<AbsF, Avail::All, V>())},
    {"negative", 1, std::nullopt, r_negative, d_negative, nullptr, {},
     unary_table<NegF, Avail::NotBool, CNegF>()},
    {"sqrt", 1, std::nullopt, r_float, d_inexact<kSqrt>, nullptr, {},
     unary_table<SqrtF, Avail::FloatOnly, CSqrtF>()},
    {"exp", 1, std::nullopt, r_float, d_inexact<kExp>, nullptr, {},
     unary_table<ExpF, Avail::FloatOnly, ExpF>()},
    {"log", 1, std::nullopt, r_float, d_inexact<kLog>, nullptr, {},
     unary_table<LogF, Avail::FloatOnly, CLogF>()},
    {"conjugate", 1, std::nullopt, r_conjugate, d_not_bool<kConjugate>, nullptr, {},
     unary_table<IdentityF, Avail::All, CConjF>()},
    {"angle", 1, std::nullopt, r_angle, d_angle, nullptr, {},
     with_complex_to_real<ComplexAngleF>(unary_table<RealAngleF, Avail::FloatOnly, V>())},
}};

}  // namespace

const Ufunc* find_ufunc(std::string_view name) noexcept {
  for (const auto* table : {&kBinary, &kUnary}) {
    for (const Ufunc& u : *table) {
      if (name == u.name) return &u;
    }
  }
  return nullptr;
}

const Ufunc& get(BinaryOp op) noexcept { return kBinary[static_cast<std::size_t>(op)]; }
const Ufunc& get(UnaryOp op) noexcept { return kUnary[static_cast<std::size_t>(op)]; }


// ---- generic drivers ----

namespace {

void check_where_dtype(const NDArray& mask) {
  if (mask.dtype() != DType::Bool) {
    throw_error(ErrorKind::DType, "Cannot cast array data from " +
                                      std::string(dtype_name(mask.dtype())) +
                                      " to bool according to the rule 'safe'");
  }
}

// True if every element of `in` (broadcast to out's shape) sits at the same
// address as the out element it produces, so element-wise in-place is safe.
bool same_view(const NDArray& in, const NDArray& out) {
  return in.shares_buffer(out) && in.offset() == out.offset() && in.dtype() == out.dtype() &&
         detail::aligned_strides(in, out.shape()) == detail::aligned_strides(out, out.shape());
}

// Input prepared for a direct write into `out` (D-046): copied when it may
// share memory with out without being the same view.
NDArray safe_input(const NDArray& in, const NDArray& out) {
  return in.may_share_memory(out) && !same_view(in, out) ? in.copy() : in;
}

void check_out_writeable(const NDArray& out) {
  if (!out.writeable()) throw_error(ErrorKind::Value, "output array is read-only");
}

// `what` is "output", "input", "input 0", "input 1" (NumPy's wording).
void check_cast(const char* name, const char* what, DType from, DType to, Casting casting) {
  if (!can_cast(from, to, casting)) {
    throw_error(ErrorKind::DType, std::string("Cannot cast ufunc '") + name + "' " + what +
                                      " from " + std::string(dtype_name(from)) + " to " +
                                      std::string(dtype_name(to)) + " with casting rule '" +
                                      std::string(casting_name(casting)) + "'");
  }
}

void check_out_shape(std::vector<Shape> shapes, const Shape& out) {
  shapes.push_back(out);
  const Shape full = broadcast_shapes(shapes);  // throws, listing out last
  if (full != out) {
    shapes.pop_back();
    throw_error(ErrorKind::Broadcast, "non-broadcastable output operand with shape " +
                                          shape_to_string(out) + " doesn't match the broadcast shape " +
                                          shape_to_string(broadcast_shapes(shapes)));
  }
}

// Masked tail (D-049): the loop has already filled `tmp` (loop dtype, full
// broadcast shape). Copy the true positions into `out`, or into a zeroed result.
NDArray finish_masked(const NDArray& tmp, const NDArray* out, const NDArray& mask) {
  if (!out) {
    // Same layout as tmp (the D-050 result layout).
    NDArray res = NDArray::empty_strided(tmp.shape(), tmp.strides(), tmp.dtype(), true);
    masked_copy_into(res, tmp, mask);
    return res;
  }
  masked_copy_into(*out, tmp, mask);
  return *out;
}

// a and b have dtype lt.in; `out` has dtype lt.out and the broadcast shape.
void binary_into(const Ufunc& u, LoopTypes lt, const NDArray& a, const NDArray& b,
                 const NDArray& out) {
  const BinaryLoopFn fn = u.binary_loops[static_cast<std::size_t>(lt.in)];
  if (!fn) bad_loop(u.name, lt.in);
  fn(make_plan<3>(out.shape(), {&out, &a, &b}), {out.data(), a.data(), b.data()});
}

void unary_into(const Ufunc& u, LoopTypes lt, const NDArray& a, const NDArray& out) {
  const UnaryLoopFn fn = u.unary_loops[static_cast<std::size_t>(lt.in)];
  if (!fn) bad_loop(u.name, lt.in);
  fn(make_plan<2>(out.shape(), {&out, &a}), {out.data(), a.data()});
}

// Shared driver tail: runs `loop(dst)` into the right destination.
template <typename Loop>
NDArray run_into(DType dt, const Shape& full, const Strides& layout, const NDArray* out,
                 const UfuncParams& params, Loop&& loop) {
  if (params.where) {
    const NDArray tmp = NDArray::empty_strided(full, layout, dt);
    loop(tmp, false);
    return finish_masked(tmp, out, *params.where);
  }
  if (!out) {
    NDArray res = NDArray::empty_strided(full, layout, dt);
    loop(res, false);
    return res;
  }
  if (out->dtype() == dt) {
    loop(*out, true);  // direct write: inputs need safe_input
  } else {
    const NDArray tmp = NDArray::empty_strided(full, layout, dt);
    loop(tmp, false);
    copy_into(*out, tmp);
  }
  return *out;
}

}  // namespace


NDArray binary(const Ufunc& u, const NDArray& a_in, const NDArray& b_in, const NDArray* out,
               const UfuncParams& params) {
  if (out) check_out_writeable(*out);
  if (params.where) check_where_dtype(*params.where);
  const LoopTypes lt = params.dtype ? u.resolve_dtype(a_in.dtype(), *params.dtype)
                                    : u.resolve(a_in.dtype(), b_in.dtype());
  const DType dt = lt.out;
  check_cast(u.name, "input 0", a_in.dtype(), lt.in, params.casting);
  check_cast(u.name, "input 1", b_in.dtype(), lt.in, params.casting);
  std::vector<Shape> shapes{a_in.shape(), b_in.shape()};
  if (params.where) shapes.push_back(params.where->shape());
  if (out) {
    check_cast(u.name, "output", dt, out->dtype(), params.casting);
    check_out_shape(shapes, out->shape());
  }
  const Shape full = out ? out->shape() : broadcast_shapes(shapes);
  // D-050 layout from the operands as given (before dtype casts), like NumPy.
  const NDArray* where = params.where ? &*params.where : nullptr;
  const Strides layout = ufunc_result_strides(full, itemsize(dt), {&a_in, &b_in},
                                              {a_in.dtype() != lt.in, b_in.dtype() != lt.in},
                                              where, params.order);
  const NDArray a = a_in.dtype() == lt.in ? a_in : a_in.astype(lt.in);
  const NDArray b = b_in.dtype() == lt.in ? b_in : b_in.astype(lt.in);
  if (u.check) u.check(a, &b, where, full);
  return run_into(dt, full, layout, out, params, [&](const NDArray& dst, bool direct) {
    if (direct) binary_into(u, lt, safe_input(a, dst), safe_input(b, dst), dst);
    else binary_into(u, lt, a, b, dst);
  });
}

NDArray unary(const Ufunc& u, const NDArray& a_in, const NDArray* out, const UfuncParams& params) {
  if (out) check_out_writeable(*out);
  if (params.where) check_where_dtype(*params.where);
  const LoopTypes lt = params.dtype ? u.resolve_dtype(a_in.dtype(), *params.dtype)
                                    : u.resolve(a_in.dtype(), a_in.dtype());
  const DType dt = lt.out;
  check_cast(u.name, "input", a_in.dtype(), lt.in, params.casting);
  std::vector<Shape> shapes{a_in.shape()};
  if (params.where) shapes.push_back(params.where->shape());
  if (out) {
    check_cast(u.name, "output", dt, out->dtype(), params.casting);
    check_out_shape(shapes, out->shape());
  }
  const Shape full = out ? out->shape() : broadcast_shapes(shapes);
  const NDArray* where = params.where ? &*params.where : nullptr;
  const Strides layout = ufunc_result_strides(full, itemsize(dt), {&a_in},
                                              {a_in.dtype() != lt.in}, where, params.order);
  const NDArray a = a_in.dtype() == lt.in ? a_in : a_in.astype(lt.in);
  if (u.check) u.check(a, nullptr, where, full);
  return run_into(dt, full, layout, out, params, [&](const NDArray& dst, bool direct) {
    unary_into(u, lt, direct ? safe_input(a, dst) : a, dst);
  });
}

}  // namespace nativpy

