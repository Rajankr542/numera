#include "p14_poly.hpp"

#include <complex>
#include <cstring>
#include <vector>

#include "error.hpp"
#include "layout.hpp"

namespace nativpy::p14 {

namespace {

template <typename T>
std::vector<T> load(const NDArray& a, DType dt) {
  const NDArray c = copy_order(a, dt, Order::C);
  std::vector<T> v(static_cast<std::size_t>(c.size()));
  if (!v.empty()) std::memcpy(v.data(), c.data(), v.size() * sizeof(T));
  return v;
}

template <typename T>
NDArray store(const std::vector<T>& v, DType dt) {
  NDArray out = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  if (!v.empty()) std::memcpy(out.data(), v.data(), v.size() * sizeof(T));
  return out;
}

// Integer convolution in unsigned arithmetic (wraps like NumPy, no UB).
template <typename T>
NDArray conv_int(const NDArray& a, const NDArray& b, DType dt) {
  const auto x = load<T>(a, dt), y = load<T>(b, dt);
  using U = std::make_unsigned_t<T>;
  std::vector<T> out(x.size() + y.size() - 1);
  for (std::size_t i = 0; i < x.size(); ++i) {
    for (std::size_t j = 0; j < y.size(); ++j) {
      out[i + j] = static_cast<T>(static_cast<U>(static_cast<U>(out[i + j]) +
                                                 static_cast<U>(static_cast<U>(x[i]) * static_cast<U>(y[j]))));
    }
  }
  return store(out, dt);
}

template <typename T>
NDArray conv_num(const NDArray& a, const NDArray& b, DType dt) {
  const auto x = load<T>(a, dt), y = load<T>(b, dt);
  std::vector<T> out(x.size() + y.size() - 1, T{});
  for (std::size_t i = 0; i < x.size(); ++i) {
    for (std::size_t j = 0; j < y.size(); ++j) out[i + j] += x[i] * y[j];
  }
  return store(out, dt);
}

template <typename T>
double magnitude(const T& v) {
  return static_cast<double>(std::abs(v));
}

template <typename T>
std::pair<NDArray, NDArray> div_impl(const NDArray& u_in, const NDArray& v_in, DType dt) {
  const auto u = load<T>(u_in, dt), v = load<T>(v_in, dt);
  const auto m = static_cast<std::int64_t>(u.size()) - 1;
  const auto n = static_cast<std::int64_t>(v.size()) - 1;
  const T scale = T(1) / v[0];
  std::vector<T> q(static_cast<std::size_t>(std::max<std::int64_t>(m - n + 1, 1)), T{});
  std::vector<T> r = u;
  for (std::int64_t k = 0; k < m - n + 1; ++k) {
    const T d = scale * r[static_cast<std::size_t>(k)];
    q[static_cast<std::size_t>(k)] = d;
    for (std::int64_t j = 0; j <= n; ++j) r[static_cast<std::size_t>(k + j)] -= d * v[static_cast<std::size_t>(j)];
  }
  std::size_t start = 0;
  // np.allclose(r[0], 0, rtol=1e-14): |r0| <= atol (1e-8); NaN is never close.
  while (r.size() - start > 1 && magnitude(r[start]) <= 1e-8) ++start;
  return {store(q, dt), store(std::vector<T>(r.begin() + static_cast<std::ptrdiff_t>(start), r.end()), dt)};
}

}  // namespace

NDArray convolve_full(const NDArray& a, const NDArray& b) {
  if (a.ndim() != 1 || b.ndim() != 1) throw_error(ErrorKind::Value, "object too deep for desired array");
  if (a.size() == 0) throw_error(ErrorKind::Value, "a cannot be empty");
  if (b.size() == 0) throw_error(ErrorKind::Value, "v cannot be empty");
  const DType dt = promote_types(a.dtype(), b.dtype());
  switch (dt) {
    case DType::Bool: {
      const auto x = load<std::uint8_t>(a, dt), y = load<std::uint8_t>(b, dt);
      std::vector<std::uint8_t> out(x.size() + y.size() - 1, 0);
      for (std::size_t i = 0; i < x.size(); ++i) {
        for (std::size_t j = 0; j < y.size(); ++j) out[i + j] |= static_cast<std::uint8_t>(x[i] & y[j]);
      }
      return store(out, dt);
    }
    case DType::Int8: return conv_int<std::int8_t>(a, b, dt);
    case DType::UInt8: return conv_int<std::uint8_t>(a, b, dt);
    case DType::Int16: return conv_int<std::int16_t>(a, b, dt);
    case DType::UInt16: return conv_int<std::uint16_t>(a, b, dt);
    case DType::Int32: return conv_int<std::int32_t>(a, b, dt);
    case DType::UInt32: return conv_int<std::uint32_t>(a, b, dt);
    case DType::Int64: return conv_int<std::int64_t>(a, b, dt);
    case DType::UInt64: return conv_int<std::uint64_t>(a, b, dt);
    case DType::Float16: return copy_order(conv_num<float>(a, b, DType::Float32), DType::Float16, Order::C);
    case DType::Float32: return conv_num<float>(a, b, dt);
    case DType::Float64: return conv_num<double>(a, b, dt);
    case DType::Complex64: return conv_num<std::complex<float>>(a, b, dt);
    case DType::Complex128: return conv_num<std::complex<double>>(a, b, dt);
  }
  throw_error(ErrorKind::DType, "convolve: unsupported dtype");
}

std::pair<NDArray, NDArray> polydiv(const NDArray& u, const NDArray& v) {
  if (u.ndim() != 1 || v.ndim() != 1 || u.size() == 0 || v.size() == 0) {
    throw_error(ErrorKind::Value, "polydiv expects non-empty 1-D arrays");
  }
  const DType dt = u.dtype();
  switch (dt) {
    case DType::Float16: {
      auto [q, r] = div_impl<float>(u, v, DType::Float32);
      return {copy_order(q, DType::Float16, Order::C), copy_order(r, DType::Float16, Order::C)};
    }
    case DType::Float32: return div_impl<float>(u, v, dt);
    case DType::Float64: return div_impl<double>(u, v, dt);
    case DType::Complex64: return div_impl<std::complex<float>>(u, v, dt);
    case DType::Complex128: return div_impl<std::complex<double>>(u, v, dt);
    default: throw_error(ErrorKind::DType, "polydiv expects an inexact dtype");
  }
}

}  // namespace nativpy::p14
