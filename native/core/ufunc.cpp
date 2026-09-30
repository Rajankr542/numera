#include "ufunc.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"
#include "complex_kernels.hpp"
#include "ufunc_kernels.hpp"

namespace nativpy {

namespace {

const char* binary_name(BinaryOp op) noexcept {
  switch (op) {
    case BinaryOp::Add: return "add";
    case BinaryOp::Subtract: return "subtract";
    case BinaryOp::Multiply: return "multiply";
    case BinaryOp::Divide: return "divide";
    case BinaryOp::Power: return "power";
    case BinaryOp::Mod: return "mod";
    case BinaryOp::FloorDivide: return "floorDivide";
  }
  return "?";
}

const char* unary_name(UnaryOp op) noexcept {
  switch (op) {
    case UnaryOp::Abs: return "abs";
    case UnaryOp::Negative: return "negative";
    case UnaryOp::Sqrt: return "sqrt";
    case UnaryOp::Exp: return "exp";
    case UnaryOp::Log: return "log";
    case UnaryOp::Conjugate: return "conjugate";
    case UnaryOp::Angle: return "angle";
  }
  return "?";
}

[[noreturn]] void bad_loop(const char* name, DType dt) {
  throw_error(ErrorKind::DType, std::string("no ") + name + " loop for dtype " +
                                    std::string(dtype_name(dt)));
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

}  // namespace

std::optional<BinaryOp> binary_op_from_name(std::string_view n) noexcept {
  if (n == "add") return BinaryOp::Add;
  if (n == "subtract") return BinaryOp::Subtract;
  if (n == "multiply") return BinaryOp::Multiply;
  if (n == "divide") return BinaryOp::Divide;
  if (n == "power") return BinaryOp::Power;
  if (n == "mod") return BinaryOp::Mod;
  if (n == "floorDivide") return BinaryOp::FloorDivide;
  return std::nullopt;
}

std::optional<UnaryOp> unary_op_from_name(std::string_view n) noexcept {
  if (n == "abs") return UnaryOp::Abs;
  if (n == "negative") return UnaryOp::Negative;
  if (n == "sqrt") return UnaryOp::Sqrt;
  if (n == "exp") return UnaryOp::Exp;
  if (n == "log") return UnaryOp::Log;
  if (n == "conjugate") return UnaryOp::Conjugate;
  if (n == "angle") return UnaryOp::Angle;
  return std::nullopt;
}

DType binary_result_dtype(BinaryOp op, DType a, DType b) {
  const DType p = promote_types(a, b);
  if (is_complex(p) && (op == BinaryOp::Mod || op == BinaryOp::FloorDivide)) {
    bad_loop(binary_name(op), p);  // NumPy: TypeError, no complex loop (D-033)
  }
  switch (op) {
    case BinaryOp::Subtract:
      if (p == DType::Bool) {
        throw_error(ErrorKind::DType,
                    "boolean subtract is not supported, use logicalXor instead");
      }
      return p;
    case BinaryOp::Divide:
      return (p == DType::Bool || is_integer(p)) ? DType::Float64 : p;
    case BinaryOp::Power:
    case BinaryOp::Mod:
    case BinaryOp::FloorDivide:
      return p == DType::Bool ? DType::Int8 : p;
    default:
      return p;
  }
}

DType unary_result_dtype(UnaryOp op, DType in) {
  if (is_complex(in)) {
    // abs and angle of complex are real (D-033); other ops keep the dtype.
    return (op == UnaryOp::Abs || op == UnaryOp::Angle) ? complex_real_dtype(in) : in;
  }
  if (in == DType::Bool && op == UnaryOp::Angle) return DType::Float64;  // arctan2(0, bool)
  if (in == DType::Bool && op == UnaryOp::Conjugate) return DType::Int8;  // NumPy: no bool loop
  switch (op) {
    case UnaryOp::Negative:
      if (in == DType::Bool) {
        throw_error(ErrorKind::DType, "boolean negative is not supported, use logicalNot instead");
      }
      return in;
    case UnaryOp::Abs:
    case UnaryOp::Conjugate: return in;  // NumPy conjugate of real is identity
    default: return float_for(in);
  }
}

namespace {

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

// ptr[0] = out, ptr[1] = a, ptr[2] = b.
template <typename S, typename Fn>
void binary_loop(const BroadcastPlan<3>& p, std::array<std::byte*, 3> base, Fn f) {
  run_plan(p, base, [&](const std::array<std::byte*, 3>& ptr, const std::array<std::int64_t, 3>& is,
                        std::int64_t n) {
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

template <typename S, typename Fn>
void unary_loop(const BroadcastPlan<2>& p, std::array<std::byte*, 2> base, Fn f) {
  run_plan(p, base, [&](const std::array<std::byte*, 2>& ptr, const std::array<std::int64_t, 2>& is,
                        std::int64_t n) {
    constexpr auto sz = static_cast<std::int64_t>(sizeof(S));
    if (is[0] == sz && is[1] == sz) {
      for (std::int64_t i = 0; i < n; ++i) st<S>(ptr[0] + i * sz, f(ld<S>(ptr[1] + i * sz)));
    } else {
      for (std::int64_t i = 0; i < n; ++i) st<S>(ptr[0] + i * is[0], f(ld<S>(ptr[1] + i * is[1])));
    }
  });
}

void check_nonnegative_exponent(const NDArray& b) {
  if (!is_integer(b.dtype()) || dtype_info(b.dtype()).kind == 'u') return;
  dispatch_dtype(b.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (std::is_integral_v<S> && std::is_signed_v<S>) {
      for_each_element(b, [&](const std::byte* p) {
        if (load<S>(p) < 0) {
          throw_error(ErrorKind::Value, "Integers to negative integer powers are not allowed.");
        }
      });
    }
  });
}

}  // namespace

NDArray binary(BinaryOp op, const NDArray& a_in, const NDArray& b_in) {
  const DType dt = binary_result_dtype(op, a_in.dtype(), b_in.dtype());
  const Shape out_shape = broadcast_shapes({a_in.shape(), b_in.shape()});
  const NDArray a = a_in.dtype() == dt ? a_in : a_in.astype(dt);
  const NDArray b = b_in.dtype() == dt ? b_in : b_in.astype(dt);
  if (op == BinaryOp::Power) check_nonnegative_exponent(b);
  NDArray out = NDArray::empty(out_shape, dt);
  const auto plan = make_plan<3>(out_shape, {&out, &a, &b});
  const std::array<std::byte*, 3> base{out.data(), a.data(), b.data()};
  dispatch_dtype(dt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    using C = compute_t<S>;
    if constexpr (is_complex_v<S>) {
      using R = typename S::value_type;
      namespace k = kernels;
      switch (op) {
        case BinaryOp::Add: binary_loop<S>(plan, base, k::cadd<R>); return;
        case BinaryOp::Subtract: binary_loop<S>(plan, base, k::csub<R>); return;
        case BinaryOp::Multiply: binary_loop<S>(plan, base, k::cmul<R>); return;
        case BinaryOp::Divide: binary_loop<S>(plan, base, k::cdiv<R>); return;
        case BinaryOp::Power: binary_loop<S>(plan, base, k::cpow<R>); return;
        default: break;
      }
      bad_loop(binary_name(op), dt);
    } else {
      namespace k = kernels;
      switch (op) {
        case BinaryOp::Add: binary_loop<S>(plan, base, k::add<C>); return;
        case BinaryOp::Multiply: binary_loop<S>(plan, base, k::mul<C>); return;
        case BinaryOp::Subtract:
          if constexpr (!std::is_same_v<S, bool>) { binary_loop<S>(plan, base, k::sub<C>); return; }
          break;
        case BinaryOp::Divide:
          if constexpr (std::is_floating_point_v<C>) { binary_loop<S>(plan, base, k::div<C>); return; }
          break;
        case BinaryOp::Power:
          if constexpr (!std::is_same_v<S, bool>) { binary_loop<S>(plan, base, k::power<C>); return; }
          break;
        case BinaryOp::Mod:
          if constexpr (!std::is_same_v<S, bool>) { binary_loop<S>(plan, base, k::mod<C>); return; }
          break;
        case BinaryOp::FloorDivide:
          if constexpr (!std::is_same_v<S, bool>) { binary_loop<S>(plan, base, k::floordiv<C>); return; }
          break;
      }
      bad_loop(binary_name(op), dt);
    }
  });
  return out;
}

NDArray unary(UnaryOp op, const NDArray& a_in) {
  const DType dt = unary_result_dtype(op, a_in.dtype());
  if ((op == UnaryOp::Abs || op == UnaryOp::Angle) && is_complex(a_in.dtype())) {
    // Complex -> real: input and output element types differ (D-033).
    NDArray out = NDArray::empty(a_in.shape(), dt);
    const auto plan = make_plan<2>(a_in.shape(), {&out, &a_in});
    const std::array<std::byte*, 2> base{out.data(), a_in.data()};
    const auto run = [&](auto zero) {
      using R = decltype(zero);
      using S = std::complex<R>;
      const bool angle = op == UnaryOp::Angle;
      run_plan(plan, base, [angle](const std::array<std::byte*, 2>& ptr,
                                   const std::array<std::int64_t, 2>& is, std::int64_t n) {
        for (std::int64_t i = 0; i < n; ++i) {
          const S v = load<S>(ptr[1] + i * is[1]);
          store<R>(ptr[0] + i * is[0], angle ? std::atan2(v.imag(), v.real()) : kernels::cabs<R>(v));
        }
      });
    };
    if (dt == DType::Float32) run(float{});
    else run(double{});
    return out;
  }
  const NDArray a = a_in.dtype() == dt ? a_in : a_in.astype(dt);
  NDArray out = NDArray::empty(a.shape(), dt);
  const auto plan = make_plan<2>(a.shape(), {&out, &a});
  const std::array<std::byte*, 2> base{out.data(), a.data()};
  dispatch_dtype(dt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    using C = compute_t<S>;
    if constexpr (is_complex_v<S>) {
      using R = typename S::value_type;
      switch (op) {
        case UnaryOp::Negative: unary_loop<S>(plan, base, kernels::cneg<R>); return;
        case UnaryOp::Conjugate: unary_loop<S>(plan, base, kernels::cconj<R>); return;
        case UnaryOp::Sqrt: unary_loop<S>(plan, base, kernels::csqrt<R>); return;
        case UnaryOp::Exp: unary_loop<S>(plan, base, [](S v) { return std::exp(v); }); return;
        case UnaryOp::Log: unary_loop<S>(plan, base, kernels::clog<R>); return;
        default: break;
      }
      bad_loop(unary_name(op), dt);
    } else {
      switch (op) {
        case UnaryOp::Abs: unary_loop<S>(plan, base, kernels::absolute<C>); return;
        case UnaryOp::Negative:
          if constexpr (!std::is_same_v<S, bool>) { unary_loop<S>(plan, base, kernels::negative<C>); return; }
          break;
        case UnaryOp::Sqrt:
          if constexpr (std::is_floating_point_v<C>) { unary_loop<S>(plan, base, [](C v) { return std::sqrt(v); }); return; }
          break;
        case UnaryOp::Exp:
          if constexpr (std::is_floating_point_v<C>) { unary_loop<S>(plan, base, [](C v) { return std::exp(v); }); return; }
          break;
        case UnaryOp::Log:
          if constexpr (std::is_floating_point_v<C>) { unary_loop<S>(plan, base, [](C v) { return std::log(v); }); return; }
          break;
        case UnaryOp::Conjugate: unary_loop<S>(plan, base, [](C v) { return v; }); return;
        case UnaryOp::Angle:
          // NumPy angle of real input: arctan2(0, x).
          if constexpr (std::is_floating_point_v<C>) { unary_loop<S>(plan, base, [](C v) { return std::atan2(C{0}, v); }); return; }
          break;
      }
      bad_loop(unary_name(op), dt);
    }
  });
  return out;
}

NDArray complex_part(const NDArray& a, bool imag) {
  if (is_complex(a.dtype())) {
    const DType rt = complex_real_dtype(a.dtype());
    const std::int64_t off = a.offset() + (imag ? static_cast<std::int64_t>(itemsize(rt)) : 0);
    // Same strides (bytes) and buffer; reinterpret each element's re or im.
    NDArray v{a.buffer(), rt, a.shape(), a.strides(), off};
    return a.writeable() ? v : v.as_readonly();
  }
  if (!imag) return a.view(a.shape(), a.strides(), a.offset());
  return NDArray::zeros(a.shape(), a.dtype()).as_readonly();
}

NDArray is_complex_elementwise(const NDArray& a, bool want_complex) {
  NDArray out = NDArray::empty(a.shape(), DType::Bool);
  if (!is_complex(a.dtype())) {
    for (std::int64_t i = 0; i < out.size(); ++i) out.set_int64(i, want_complex ? 0 : 1);
    return out;
  }
  const NDArray im = complex_part(a, true);
  const auto plan = make_plan<2>(a.shape(), {&out, &im});
  const std::array<std::byte*, 2> base{out.data(), im.data()};
  const auto run = [&](auto zero) {
    using R = decltype(zero);
    run_plan(plan, base, [want_complex](const std::array<std::byte*, 2>& ptr,
                                        const std::array<std::int64_t, 2>& is, std::int64_t n) {
      for (std::int64_t i = 0; i < n; ++i) {
        const bool nonzero = load<R>(ptr[1] + i * is[1]) != R{0};  // NaN counts as nonzero
        store<bool>(ptr[0] + i * is[0], nonzero == want_complex);
      }
    });
  };
  if (im.dtype() == DType::Float32) run(float{});
  else run(double{});
  return out;
}

}  // namespace nativpy
