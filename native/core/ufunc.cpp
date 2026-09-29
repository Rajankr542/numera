#include "ufunc.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"
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
  }
  return "?";
}

void reject_complex(DType dt, const char* name) {
  if (is_complex(dt)) {
    throw_error(ErrorKind::NotImplemented,
                std::string(name) + " is not implemented for complex dtypes (D-014)");
  }
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
  return std::nullopt;
}

DType binary_result_dtype(BinaryOp op, DType a, DType b) {
  reject_complex(a, binary_name(op));
  reject_complex(b, binary_name(op));
  const DType p = promote_types(a, b);
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
  reject_complex(in, unary_name(op));
  switch (op) {
    case UnaryOp::Negative:
      if (in == DType::Bool) {
        throw_error(ErrorKind::DType, "boolean negative is not supported, use logicalNot instead");
      }
      return in;
    case UnaryOp::Abs: return in;
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

[[noreturn]] void bad_loop(const char* name, DType dt) {
  throw_error(ErrorKind::DType, std::string("no ") + name + " loop for dtype " +
                                    std::string(dtype_name(dt)));
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
  const NDArray a = a_in.dtype() == dt ? a_in : a_in.astype(dt);
  NDArray out = NDArray::empty(a.shape(), dt);
  const auto plan = make_plan<2>(a.shape(), {&out, &a});
  const std::array<std::byte*, 2> base{out.data(), a.data()};
  dispatch_dtype(dt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    using C = compute_t<S>;
    if constexpr (is_complex_v<S>) {
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
      }
      bad_loop(unary_name(op), dt);
    }
  });
  return out;
}

}  // namespace nativpy
