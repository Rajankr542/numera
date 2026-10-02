#include "p15_emath.hpp"

#include <cmath>
#include <complex>
#include <cstring>

#include "complex_kernels.hpp"
#include "fp_errors.hpp"

namespace nativpy {

namespace {

#if defined(_MSC_VER)
template <typename T>
std::complex<T> cacos_(std::complex<T> z) noexcept { return std::acos(z); }
template <typename T>
std::complex<T> casin_(std::complex<T> z) noexcept { return std::asin(z); }
template <typename T>
std::complex<T> catanh_(std::complex<T> z) noexcept { return std::atanh(z); }
#else
__extension__ typedef _Complex double c99d;
__extension__ typedef _Complex float c99f;

// C99 casin/cacos/catanh (NumPy's npy_casin etc. use the platform libm).
template <typename T, typename Fd, typename Ff>
std::complex<T> c99(std::complex<T> z, Fd fd, Ff ff) noexcept {
  std::complex<T> out;
  if constexpr (std::is_same_v<T, double>) {
    c99d c;
    std::memcpy(&c, &z, sizeof c);
    const c99d r = fd(c);
    std::memcpy(&out, &r, sizeof out);
  } else {
    c99f c;
    std::memcpy(&c, &z, sizeof c);
    const c99f r = ff(c);
    std::memcpy(&out, &r, sizeof out);
  }
  return out;
}

template <typename T>
std::complex<T> cacos_(std::complex<T> z) noexcept {
  return c99(z, [](c99d c) { return __builtin_cacos(c); }, [](c99f c) { return __builtin_cacosf(c); });
}
template <typename T>
std::complex<T> casin_(std::complex<T> z) noexcept {
  return c99(z, [](c99d c) { return __builtin_casin(c); }, [](c99f c) { return __builtin_casinf(c); });
}
template <typename T>
std::complex<T> catanh_(std::complex<T> z) noexcept {
  return c99(z, [](c99d c) { return __builtin_catanh(c); }, [](c99f c) { return __builtin_catanhf(c); });
}
#endif

template <typename T>
std::complex<T> scale(std::complex<T> z, T k) noexcept {
  return {z.real() * k, z.imag() * k};
}

template <typename T>
std::complex<T> complex_kernel(EmathOp op, std::complex<T> z) noexcept {
  constexpr T log2e = static_cast<T>(1.442695040888963407359924681001892137L);
  constexpr T log10e = static_cast<T>(0.434294481903251827651128918916605082L);
  switch (op) {
    case EmathOp::Sqrt: return kernels::csqrt(z);
    case EmathOp::Log: return kernels::clog(z);
    case EmathOp::Log2: return scale(kernels::clog(z), log2e);
    case EmathOp::Log10: return scale(kernels::clog(z), log10e);
    case EmathOp::Arccos: return cacos_(z);
    case EmathOp::Arcsin: return casin_(z);
    case EmathOp::Arctanh: return catanh_(z);
  }
  return z;
}

template <typename T>
T real_kernel(EmathOp op, T v) noexcept {
  switch (op) {
    case EmathOp::Sqrt: return std::sqrt(v);
    case EmathOp::Log: return std::log(v);
    case EmathOp::Log2: return std::log2(v);
    case EmathOp::Log10: return std::log10(v);
    case EmathOp::Arccos: return std::acos(v);
    case EmathOp::Arcsin: return std::asin(v);
    case EmathOp::Arctanh: return std::atanh(v);
  }
  return v;
}

const char* op_name(EmathOp op) noexcept {
  switch (op) {
    case EmathOp::Sqrt: return "sqrt";
    case EmathOp::Log: return "log";
    case EmathOp::Log2: return "log2";
    case EmathOp::Log10: return "log10";
    case EmathOp::Arccos: return "arccos";
    case EmathOp::Arcsin: return "arcsin";
    case EmathOp::Arctanh: return "arctanh";
  }
  return "emath";
}

bool out_of_domain(EmathOp op, const NDArray& x) {
  if (is_complex(x.dtype()) || x.dtype() == DType::Bool) return false;
  const bool abs_gt_1 = op == EmathOp::Arccos || op == EmathOp::Arcsin || op == EmathOp::Arctanh;
  const NDArray d = x.astype(DType::Float64);
  const auto* p = reinterpret_cast<const double*>(d.data());
  for (std::int64_t i = 0; i < d.size(); ++i) {
    if (abs_gt_1 ? std::fabs(p[i]) > 1.0 : p[i] < 0.0) return true;
  }
  return false;
}

// `_tocomplex`: complex64 for single, byte, short, ubyte, ushort, csingle.
DType to_complex(DType dt) noexcept {
  switch (dt) {
    case DType::Int8:
    case DType::UInt8:
    case DType::Int16:
    case DType::UInt16:
    case DType::Float32:
    case DType::Complex64: return DType::Complex64;
    default: return DType::Complex128;
  }
}

// The ufunc's float loop for real input (as ufunc_loops::float_for).
DType real_loop(DType dt) noexcept {
  switch (dt) {
    case DType::Bool:
    case DType::Int8:
    case DType::UInt8: return DType::Float16;
    case DType::Int16:
    case DType::UInt16:
    case DType::Float32: return DType::Float32;
    case DType::Float16: return DType::Float16;
    default: return DType::Float64;
  }
}

template <typename T>
NDArray apply_complex(EmathOp op, const NDArray& x, DType cdt) {
  NDArray a = x.astype(cdt);  // fresh C-contiguous copy, written in place
  auto* p = reinterpret_cast<std::complex<T>*>(a.data());
  for (std::int64_t i = 0; i < a.size(); ++i) p[i] = complex_kernel<T>(op, p[i]);
  return a;
}

template <typename T>
NDArray apply_real(EmathOp op, const NDArray& x, DType compute) {
  NDArray a = x.astype(compute);
  auto* p = reinterpret_cast<T*>(a.data());
  for (std::int64_t i = 0; i < a.size(); ++i) p[i] = real_kernel<T>(op, p[i]);
  return a;
}

}  // namespace

std::optional<EmathOp> emath_op_from_name(std::string_view name) noexcept {
  if (name == "sqrt") return EmathOp::Sqrt;
  if (name == "log") return EmathOp::Log;
  if (name == "log2") return EmathOp::Log2;
  if (name == "log10") return EmathOp::Log10;
  if (name == "arccos") return EmathOp::Arccos;
  if (name == "arcsin") return EmathOp::Arcsin;
  if (name == "arctanh") return EmathOp::Arctanh;
  return std::nullopt;
}

NDArray emath_unary(EmathOp op, const NDArray& x) {
  const DType dt = x.dtype();
  const FpScope fp(op_name(op));
  NDArray r = [&] {
    if (is_complex(dt) || out_of_domain(op, x)) {
      const DType cdt = to_complex(dt);
      return cdt == DType::Complex64 ? apply_complex<float>(op, x, cdt) : apply_complex<double>(op, x, cdt);
    }
    const DType rl = real_loop(dt);
    if (rl == DType::Float64) return apply_real<double>(op, x, DType::Float64);
    NDArray f = apply_real<float>(op, x, DType::Float32);
    return rl == DType::Float16 ? f.astype(DType::Float16) : f;
  }();
  fp.check();
  return r;
}

}  // namespace nativpy
