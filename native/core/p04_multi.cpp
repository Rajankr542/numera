#include "p04_multi.hpp"

#include <cmath>
#include <string>
#include <type_traits>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"
#include "fp_errors.hpp"
#include "ufunc_kernels.hpp"
#include "ufunc_loops.hpp"

namespace nativpy {

using namespace ufunc_loops;

namespace {

const char* op_name(MultiOp op) noexcept {
  switch (op) {
    case MultiOp::Divmod: return "divmod";
    case MultiOp::Modf: return "modf";
    case MultiOp::Frexp: return "frexp";
  }
  return "?";
}

struct MultiLoop {
  DType in;
  DType out0;
  DType out1;
};

MultiLoop resolve(MultiOp op, DType a, DType b, const std::optional<DType>& dt) {
  const char* name = op_name(op);
  if (op == MultiOp::Divmod) {
    if (dt) {
      if (*dt == DType::Bool || is_complex(*dt)) no_loop(name);
      return {*dt, *dt, *dt};
    }
    const DType p = promote_types(a, b);
    if (is_complex(p)) bad_loop(name, p);
    const DType l = p == DType::Bool ? DType::Int8 : p;
    return {l, l, l};
  }
  if (dt) {
    if (dtype_info(*dt).kind != 'f') no_loop(name);
    return {*dt, *dt, op == MultiOp::Frexp ? DType::Int32 : *dt};
  }
  if (is_complex(a)) bad_loop(name, a);
  const DType f = float_for(a);
  return {f, f, op == MultiOp::Frexp ? DType::Int32 : f};
}

void check_cast(const char* name, const char* what, DType from, DType to, Casting casting) {
  if (!can_cast(from, to, casting)) {
    throw_error(ErrorKind::DType, std::string("Cannot cast ufunc '") + name + "' " + what + " from " +
                                      std::string(dtype_name(from)) + " to " +
                                      std::string(dtype_name(to)) + " with casting rule '" +
                                      std::string(casting_name(casting)) + "'");
  }
}

// divmod kernels: NumPy floor_divide / remainder pairs.
template <typename T>
void divmod_kernel(T a, T b, T& q, T& r) noexcept {
  if constexpr (std::is_integral_v<T>) {
    q = kernels::floordiv_int(a, b);
    // floordiv_int already raised divide-by-zero for b == 0.
    if (b == 0) {
      r = 0;
    } else {
      r = kernels::mod_int(a, b);
    }
  } else {
    if (b == T{0}) {
      q = a / b;
      r = std::fmod(a, b);
      return;
    }
    r = kernels::fmod_floor<T>(a, b, &q);
  }
}

template <typename S>
void run_divmod(const NDArray& q, const NDArray& r, const NDArray& a, const NDArray& b) {
  const auto plan = make_plan<4>(q.shape(), {&q, &r, &a, &b});
  run_plan(plan, {q.data(), r.data(), a.data(), b.data()},
           [](const std::array<std::byte*, 4>& p, const std::array<std::int64_t, 4>& s, std::int64_t n) {
             for (std::int64_t i = 0; i < n; ++i) {
               compute_t<S> qv{}, rv{};
               divmod_kernel<compute_t<S>>(ld<S>(p[2] + i * s[2]), ld<S>(p[3] + i * s[3]), qv, rv);
               st<S>(p[0] + i * s[0], qv);
               st<S>(p[1] + i * s[1], rv);
             }
           });
}

// modf -> (fractional, integral); frexp -> (mantissa, int32 exponent).
template <typename S, bool Frexp>
void run_unary2(const NDArray& o0, const NDArray& o1, const NDArray& a) {
  const auto plan = make_plan<3>(o0.shape(), {&o0, &o1, &a});
  run_plan(plan, {o0.data(), o1.data(), a.data()},
           [](const std::array<std::byte*, 3>& p, const std::array<std::int64_t, 3>& s, std::int64_t n) {
             for (std::int64_t i = 0; i < n; ++i) {
               const auto x = ld<S>(p[2] + i * s[2]);
               if constexpr (Frexp) {
                 int e = 0;
                 const auto m = std::frexp(x, &e);
                 st<S>(p[0] + i * s[0], m);
                 // NumPy: exponent 0 for inf/nan.
                 store<std::int32_t>(p[1] + i * s[1], std::isfinite(x) ? e : 0);
               } else {
                 compute_t<S> ip{};
                 const auto f = std::modf(x, &ip);
                 st<S>(p[0] + i * s[0], f);
                 st<S>(p[1] + i * s[1], ip);
               }
             }
           });
}

void run_loop(MultiOp op, DType dt, const NDArray& o0, const NDArray& o1, const NDArray& a,
              const NDArray* b) {
  dispatch_dtype(dt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (is_complex_v<S> || std::is_same_v<S, bool>) {
      bad_loop(op_name(op), dt);
    } else if constexpr (std::is_integral_v<S>) {
      if (op != MultiOp::Divmod) bad_loop(op_name(op), dt);
      run_divmod<S>(o0, o1, a, *b);
    } else {
      if (op == MultiOp::Divmod) run_divmod<S>(o0, o1, a, *b);
      else if (op == MultiOp::Frexp) run_unary2<S, true>(o0, o1, a);
      else run_unary2<S, false>(o0, o1, a);
    }
  });
}

void check_out(const char* name, const NDArray* out, DType loop, const Shape& full, Casting casting) {
  if (!out) return;
  if (!out->writeable()) throw_error(ErrorKind::Value, "output array is read-only");
  check_cast(name, "output", loop, out->dtype(), casting);
  if (out->shape() != full) {
    throw_error(ErrorKind::Broadcast, "non-broadcastable output operand with shape " +
                                          shape_to_string(out->shape()) +
                                          " doesn't match the broadcast shape " + shape_to_string(full));
  }
}

}  // namespace

std::array<NDArray, 2> multi_ufunc(MultiOp op, const NDArray& a_in, const NDArray* b_in,
                                   const MultiParams& params) {
  const char* name = op_name(op);
  if ((op == MultiOp::Divmod) != (b_in != nullptr)) {
    throw_error(ErrorKind::Value, std::string(name) + ": wrong number of inputs");
  }
  const MultiLoop lt = resolve(op, a_in.dtype(), b_in ? b_in->dtype() : a_in.dtype(), params.dtype);
  check_cast(name, b_in ? "input 0" : "input", a_in.dtype(), lt.in, params.casting);
  if (b_in) check_cast(name, "input 1", b_in->dtype(), lt.in, params.casting);
  std::vector<Shape> shapes{a_in.shape()};
  if (b_in) shapes.push_back(b_in->shape());
  const Shape full = broadcast_shapes(shapes);
  check_out(name, params.out0, lt.out0, full, params.casting);
  check_out(name, params.out1, lt.out1, full, params.casting);
  const NDArray a = a_in.dtype() == lt.in ? a_in : a_in.astype(lt.in);
  std::optional<NDArray> b;
  if (b_in) b = b_in->dtype() == lt.in ? *b_in : b_in->astype(lt.in);
  NDArray r0 = NDArray::empty(full, lt.out0);
  NDArray r1 = NDArray::empty(full, lt.out1);
  const FpScope fp(name);
  run_loop(op, lt.in, r0, r1, a, b ? &*b : nullptr);
  fp.check();
  if (params.out0) {
    copy_into(*params.out0, r0);
    r0 = *params.out0;
  }
  if (params.out1) {
    copy_into(*params.out1, r1);
    r1 = *params.out1;
  }
  return {r0, r1};
}

}  // namespace nativpy
