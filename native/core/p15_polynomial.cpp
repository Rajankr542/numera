#include "p15_polynomial.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <vector>

#include "error.hpp"

namespace nativpy::poly {

namespace {

using cd = std::complex<double>;
template <typename T>
using Vec = std::vector<T>;

template <typename T>
bool is_zero(T v) noexcept {
  return v == T{0};
}

template <typename T>
Vec<T> load(const NDArray& a) {
  const NDArray c = a.astype(std::is_same_v<T, cd> ? DType::Complex128 : DType::Float64);
  Vec<T> v(static_cast<std::size_t>(c.size()));
  if (!v.empty()) std::memcpy(v.data(), c.data(), v.size() * sizeof(T));
  return v;
}

template <typename T>
NDArray store(const Vec<T>& v, Shape shape) {
  NDArray a = NDArray::empty(shape, std::is_same_v<T, cd> ? DType::Complex128 : DType::Float64);
  if (!v.empty()) std::memcpy(a.data(), v.data(), v.size() * sizeof(T));
  return a;
}

template <typename T>
NDArray store(const Vec<T>& v) {
  return store(v, Shape{static_cast<std::int64_t>(v.size())});
}

// polyutils.trimseq
template <typename T>
Vec<T> trimseq(Vec<T> v) {
  if (v.empty() || !is_zero(v.back())) return v;
  std::size_t i = v.size() - 1;
  while (i > 0 && is_zero(v[i])) --i;
  v.resize(i + 1);
  return v;
}

// polyutils.as_series for one series (non-empty is checked by the caller).
template <typename T>
Vec<T> series(const NDArray& a, bool trim = true) {
  if (a.ndim() != 1) throw_error(ErrorKind::Value, "Coefficient array is not 1-d");
  if (a.size() == 0) throw_error(ErrorKind::Value, "Coefficient array is empty");
  Vec<T> v = load<T>(a);
  return trim ? trimseq(std::move(v)) : v;
}

template <typename T>
Vec<T> scaled(Vec<T> v, T k) {
  for (auto& x : v) x *= k;
  return v;
}

template <typename T>
Vec<T> divided(Vec<T> v, T k) {
  for (auto& x : v) x /= k;
  return v;
}

template <typename T>
Vec<T> add_s(Vec<T> c1, Vec<T> c2) {
  c1 = trimseq(std::move(c1));
  c2 = trimseq(std::move(c2));
  if (c1.size() > c2.size()) {
    for (std::size_t i = 0; i < c2.size(); ++i) c1[i] += c2[i];
    return trimseq(std::move(c1));
  }
  for (std::size_t i = 0; i < c1.size(); ++i) c2[i] += c1[i];
  return trimseq(std::move(c2));
}

template <typename T>
Vec<T> sub_s(Vec<T> c1, Vec<T> c2) {
  c1 = trimseq(std::move(c1));
  c2 = trimseq(std::move(c2));
  if (c1.size() > c2.size()) {
    for (std::size_t i = 0; i < c2.size(); ++i) c1[i] -= c2[i];
    return trimseq(std::move(c1));
  }
  for (auto& x : c2) x = -x;
  for (std::size_t i = 0; i < c1.size(); ++i) c2[i] += c1[i];
  return trimseq(std::move(c2));
}

template <typename T>
Vec<T> convolve(const Vec<T>& a, const Vec<T>& b) {
  Vec<T> out(a.size() + b.size() - 1, T{0});
  for (std::size_t i = 0; i < a.size(); ++i)
    for (std::size_t j = 0; j < b.size(); ++j) out[i + j] += a[i] * b[j];
  return out;
}

template <typename T>
Vec<T> mulx_s(Basis b, Vec<T> c) {
  c = trimseq(std::move(c));
  if (c.size() == 1 && is_zero(c[0])) return c;
  const std::size_t n = c.size();
  Vec<T> p(n + 1, T{0});
  switch (b) {
    case Basis::Power:
      for (std::size_t i = 0; i < n; ++i) p[i + 1] = c[i];
      break;
    case Basis::Cheb:
      p[1] = c[0];
      for (std::size_t i = 1; i < n; ++i) p[i + 1] = c[i] / T{2};
      for (std::size_t i = 1; i < n; ++i) p[i - 1] += c[i] / T{2};
      break;
    case Basis::Leg:
      p[1] = c[0];
      for (std::size_t i = 1; i < n; ++i) {
        const double s = static_cast<double>(2 * i + 1);
        p[i + 1] = (c[i] * T(static_cast<double>(i + 1))) / T(s);
        p[i - 1] += (c[i] * T(static_cast<double>(i))) / T(s);
      }
      break;
    case Basis::Lag:
      p[0] = c[0];
      p[1] = -c[0];
      for (std::size_t i = 1; i < n; ++i) {
        p[i + 1] = -c[i] * T(static_cast<double>(i + 1));
        p[i] += c[i] * T(static_cast<double>(2 * i + 1));
        p[i - 1] -= c[i] * T(static_cast<double>(i));
      }
      break;
    case Basis::Herm:
      p[1] = c[0] / T{2};
      for (std::size_t i = 1; i < n; ++i) {
        p[i + 1] = c[i] / T{2};
        p[i - 1] += c[i] * T(static_cast<double>(i));
      }
      break;
    case Basis::HermE:
      p[1] = c[0];
      for (std::size_t i = 1; i < n; ++i) {
        p[i + 1] = c[i];
        p[i - 1] += c[i] * T(static_cast<double>(i));
      }
      break;
  }
  p[0] += c[0] * T{0};  // NumPy's `prd[0] = c[0] * 0` keeps NaN/Inf
  return p;
}

// Chebyshev z-series helpers.
template <typename T>
Vec<T> c2z(const Vec<T>& c) {
  const std::size_t n = c.size();
  Vec<T> zs(2 * n - 1, T{0});
  for (std::size_t i = 0; i < n; ++i) zs[n - 1 + i] = c[i] / T{2};
  Vec<T> out(zs.size());
  for (std::size_t i = 0; i < zs.size(); ++i) out[i] = zs[i] + zs[zs.size() - 1 - i];
  return out;
}

template <typename T>
Vec<T> z2c(const Vec<T>& zs) {
  const std::size_t n = (zs.size() + 1) / 2;
  Vec<T> c(zs.begin() + static_cast<std::ptrdiff_t>(n - 1), zs.end());
  for (std::size_t i = 1; i < n; ++i) c[i] *= T{2};
  return c;
}

template <typename T>
std::pair<Vec<T>, Vec<T>> zdiv(Vec<T> z1, Vec<T> z2) {
  const std::size_t lc1 = z1.size(), lc2 = z2.size();
  if (lc2 == 1) {
    for (auto& x : z1) x /= z2[0];
    return {z1, Vec<T>{z1[0] * T{0}}};
  }
  if (lc1 < lc2) return {Vec<T>{z1[0] * T{0}}, z1};
  const std::size_t dlen = lc1 - lc2;
  const T scl = z2[0];
  for (auto& x : z2) x /= scl;
  Vec<T> quo(dlen + 1);
  std::size_t i = 0, j = dlen;
  while (i < j) {
    const T r = z1[i];
    quo[i] = z1[i];
    quo[dlen - i] = r;
    for (std::size_t k = 0; k < lc2; ++k) z1[i + k] -= r * z2[k];
    for (std::size_t k = 0; k < lc2; ++k) z1[j + k] -= r * z2[k];
    ++i;
    --j;
  }
  const T r = z1[i];
  quo[i] = r;
  for (std::size_t k = 0; k < lc2; ++k) z1[i + k] -= r * z2[k];
  for (auto& x : quo) x /= scl;
  Vec<T> rem(z1.begin() + static_cast<std::ptrdiff_t>(i + 1),
             z1.begin() + static_cast<std::ptrdiff_t>(i - 1 + lc2));
  return {quo, rem};
}

// Clenshaw-style products of the Legendre / Laguerre / Hermite families.
template <typename T>
Vec<T> mul_s(Basis b, Vec<T> c1, Vec<T> c2) {
  c1 = trimseq(std::move(c1));
  c2 = trimseq(std::move(c2));
  if (b == Basis::Power) return trimseq(convolve(c1, c2));
  if (b == Basis::Cheb) return trimseq(z2c(convolve(c2z(c1), c2z(c2))));
  const Vec<T>& c = c1.size() > c2.size() ? c2 : c1;
  const Vec<T>& xs = c1.size() > c2.size() ? c1 : c2;
  Vec<T> a0, a1;
  if (c.size() == 1) {
    a0 = scaled(xs, c[0]);
    a1 = Vec<T>{T{0}};
  } else if (c.size() == 2) {
    a0 = scaled(xs, c[0]);
    a1 = scaled(xs, c[1]);
  } else {
    std::size_t nd = c.size();
    a0 = scaled(xs, c[c.size() - 2]);
    a1 = scaled(xs, c[c.size() - 1]);
    for (std::size_t i = 3; i <= c.size(); ++i) {
      Vec<T> tmp = a0;
      nd -= 1;
      const T ci = c[c.size() - i];
      const auto dn = static_cast<double>(nd);
      switch (b) {
        case Basis::Leg:
          a0 = sub_s(scaled(xs, ci), divided(scaled(a1, T(dn - 1)), T(dn)));
          a1 = add_s(tmp, divided(scaled(mulx_s(b, a1), T(2 * dn - 1)), T(dn)));
          break;
        case Basis::Lag:
          a0 = sub_s(scaled(xs, ci), divided(scaled(a1, T(dn - 1)), T(dn)));
          a1 = add_s(tmp, divided(sub_s(scaled(a1, T(2 * dn - 1)), mulx_s(b, a1)), T(dn)));
          break;
        case Basis::Herm:
          a0 = sub_s(scaled(xs, ci), scaled(a1, T(2 * (dn - 1))));
          a1 = add_s(tmp, scaled(mulx_s(b, a1), T{2}));
          break;
        default:  // HermE
          a0 = sub_s(scaled(xs, ci), scaled(a1, T(dn - 1)));
          a1 = add_s(tmp, mulx_s(b, a1));
          break;
      }
    }
  }
  switch (b) {
    case Basis::Lag: return add_s(a0, sub_s(a1, mulx_s(b, a1)));
    case Basis::Herm: return add_s(a0, scaled(mulx_s(b, a1), T{2}));
    default: return add_s(a0, mulx_s(b, a1));
  }
}

template <typename T>
std::pair<Vec<T>, Vec<T>> div_s(Basis b, Vec<T> c1, Vec<T> c2) {
  c1 = trimseq(std::move(c1));
  c2 = trimseq(std::move(c2));
  if (is_zero(c2.back())) throw_error(ErrorKind::Value, "polynomial division by zero");
  const std::size_t lc1 = c1.size(), lc2 = c2.size();
  if (lc1 < lc2) return {Vec<T>{c1[0] * T{0}}, c1};
  if (lc2 == 1) return {divided(c1, c2.back()), Vec<T>{c1[0] * T{0}}};
  if (b == Basis::Power) {
    const std::size_t dlen = lc1 - lc2;
    const T scl = c2.back();
    Vec<T> d(c2.begin(), c2.end() - 1);
    for (auto& x : d) x /= scl;
    std::ptrdiff_t i = static_cast<std::ptrdiff_t>(dlen), j = static_cast<std::ptrdiff_t>(lc1) - 1;
    while (i >= 0) {
      const T cj = c1[static_cast<std::size_t>(j)];
      for (std::ptrdiff_t k = i; k < j; ++k) c1[static_cast<std::size_t>(k)] -= d[static_cast<std::size_t>(k - i)] * cj;
      --i;
      --j;
    }
    Vec<T> quo(c1.begin() + j + 1, c1.end());
    for (auto& x : quo) x /= scl;
    return {quo, trimseq(Vec<T>(c1.begin(), c1.begin() + j + 1))};
  }
  if (b == Basis::Cheb) {
    auto [q, r] = zdiv(c2z(c1), c2z(c2));
    return {trimseq(z2c(q)), trimseq(z2c(r))};
  }
  Vec<T> quo(lc1 - lc2 + 1);
  Vec<T> rem = c1;
  for (std::size_t i = lc1 - lc2 + 1; i-- > 0;) {
    Vec<T> unit(i + 1, T{0});
    unit[i] = T{1};
    const Vec<T> p = mul_s(b, unit, c2);
    const T q = rem.back() / p.back();
    Vec<T> next(rem.size() - 1);
    for (std::size_t k = 0; k + 1 < rem.size(); ++k) next[k] = rem[k] - q * p[k];
    rem = std::move(next);
    quo[i] = q;
  }
  return {quo, trimseq(rem)};
}

template <typename T>
Vec<T> pow_s(Basis b, Vec<T> c, std::int64_t n) {
  c = trimseq(std::move(c));
  if (n == 0) return Vec<T>{T{1}};
  if (n == 1) return c;
  if (b == Basis::Cheb) {
    const Vec<T> zs = c2z(c);
    Vec<T> prd = zs;
    for (std::int64_t i = 2; i <= n; ++i) prd = convolve(prd, zs);
    return z2c(prd);
  }
  Vec<T> prd = c;
  for (std::int64_t i = 2; i <= n; ++i) prd = b == Basis::Power ? convolve(prd, c) : mul_s(b, prd, c);
  return prd;
}

template <typename T>
T val_s(Basis b, T x, const Vec<T>& c) {
  const std::size_t n = c.size();
  if (b == Basis::Power) {
    T c0 = c[n - 1] + x * T{0};
    for (std::size_t i = 2; i <= n; ++i) c0 = c[n - i] + c0 * x;
    return c0;
  }
  const T x2 = b == Basis::Cheb ? T{2} * x : x * T{2};
  T c0, c1;
  if (n == 1) {
    c0 = c[0];
    c1 = T{0};
  } else if (n == 2) {
    c0 = c[0];
    c1 = c[1];
  } else {
    std::size_t nd = n;
    c0 = c[n - 2];
    c1 = c[n - 1];
    for (std::size_t i = 3; i <= n; ++i) {
      const T tmp = c0;
      nd -= 1;
      const auto dn = static_cast<double>(nd);
      const T ci = c[n - i];
      switch (b) {
        case Basis::Cheb:
          c0 = ci - c1;
          c1 = tmp + c1 * x2;
          break;
        case Basis::Leg:
          c0 = ci - c1 * T((dn - 1) / dn);
          c1 = tmp + c1 * x * T((2 * dn - 1) / dn);
          break;
        case Basis::Lag:
          c0 = ci - (c1 * T(dn - 1)) / T(dn);
          c1 = tmp + (c1 * (T(2 * dn - 1) - x)) / T(dn);
          break;
        case Basis::Herm:
          c0 = ci - c1 * T(2 * (dn - 1));
          c1 = tmp + c1 * x2;
          break;
        default:  // HermE
          c0 = ci - c1 * T(dn - 1);
          c1 = tmp + c1 * x;
          break;
      }
    }
  }
  switch (b) {
    case Basis::Lag: return c0 + c1 * (T{1} - x);
    case Basis::Herm: return c0 + c1 * x2;
    default: return c0 + c1 * x;
  }
}

template <typename T>
Vec<T> der_s(Basis b, Vec<T> c, std::int64_t m, T scl) {
  if (m == 0) return c;
  std::size_t n = c.size();
  if (static_cast<std::size_t>(m) >= n) return Vec<T>{c[0] * T{0}};
  for (std::int64_t it = 0; it < m; ++it) {
    n -= 1;
    for (auto& v : c) v *= scl;
    Vec<T> d(n);
    switch (b) {
      case Basis::Power:
      case Basis::HermE:
        for (std::size_t j = n; j >= 1; --j) d[j - 1] = T(static_cast<double>(j)) * c[j];
        break;
      case Basis::Herm:
        for (std::size_t j = n; j >= 1; --j) d[j - 1] = T(static_cast<double>(2 * j)) * c[j];
        break;
      case Basis::Cheb:
        for (std::size_t j = n; j > 2; --j) {
          d[j - 1] = T(static_cast<double>(2 * j)) * c[j];
          c[j - 2] += (T(static_cast<double>(j)) * c[j]) / T(static_cast<double>(j - 2));
        }
        if (n > 1) d[1] = T{4} * c[2];
        d[0] = c[1];
        break;
      case Basis::Leg:
        for (std::size_t j = n; j > 2; --j) {
          d[j - 1] = T(static_cast<double>(2 * j - 1)) * c[j];
          c[j - 2] += c[j];
        }
        if (n > 1) d[1] = T{3} * c[2];
        d[0] = c[1];
        break;
      case Basis::Lag:
        for (std::size_t j = n; j > 1; --j) {
          d[j - 1] = -c[j];
          c[j - 1] += c[j];
        }
        d[0] = -c[1];
        break;
    }
    c = std::move(d);
  }
  return c;
}

template <typename T>
Vec<T> int_s(Basis b, Vec<T> c, std::int64_t m, const Vec<T>& k, T lbnd, T scl) {
  for (std::int64_t it = 0; it < m; ++it) {
    const std::size_t n = c.size();
    const T ki = static_cast<std::size_t>(it) < k.size() ? k[static_cast<std::size_t>(it)] : T{0};
    for (auto& v : c) v *= scl;
    if (n == 1 && is_zero(c[0])) {
      c[0] += ki;
      continue;
    }
    Vec<T> t(n + 1, T{0});
    switch (b) {
      case Basis::Power:
      case Basis::HermE:
        t[0] = c[0] * T{0};
        t[1] = c[0];
        for (std::size_t j = 1; j < n; ++j) t[j + 1] = c[j] / T(static_cast<double>(j + 1));
        break;
      case Basis::Herm:
        t[0] = c[0] * T{0};
        t[1] = c[0] / T{2};
        for (std::size_t j = 1; j < n; ++j) t[j + 1] = c[j] / T(static_cast<double>(2 * (j + 1)));
        break;
      case Basis::Cheb:
        t[0] = c[0] * T{0};
        t[1] = c[0];
        if (n > 1) t[2] = c[1] / T{4};
        for (std::size_t j = 2; j < n; ++j) {
          t[j + 1] = c[j] / T(static_cast<double>(2 * (j + 1)));
          t[j - 1] -= c[j] / T(static_cast<double>(2 * (j - 1)));
        }
        break;
      case Basis::Leg:
        t[0] = c[0] * T{0};
        t[1] = c[0];
        if (n > 1) t[2] = c[1] / T{3};
        for (std::size_t j = 2; j < n; ++j) {
          const T q = c[j] / T(static_cast<double>(2 * j + 1));
          t[j + 1] = q;
          t[j - 1] -= q;
        }
        break;
      case Basis::Lag:
        t[0] = c[0];
        t[1] = -c[0];
        for (std::size_t j = 1; j < n; ++j) {
          t[j] += c[j];
          t[j + 1] = -c[j];
        }
        break;
    }
    t[0] += ki - val_s(b, lbnd, t);
    c = std::move(t);
  }
  return c;
}

template <typename T>
Vec<T> vander_row(Basis b, T x, std::size_t deg) {
  Vec<T> v(deg + 1);
  v[0] = x * T{0} + T{1};
  if (deg == 0) return v;
  const T x2 = b == Basis::Cheb ? T{2} * x : x * T{2};
  switch (b) {
    case Basis::Lag: v[1] = T{1} - x; break;
    case Basis::Herm: v[1] = x2; break;
    default: v[1] = x; break;
  }
  for (std::size_t i = 2; i <= deg; ++i) {
    const auto di = static_cast<double>(i);
    switch (b) {
      case Basis::Power: v[i] = v[i - 1] * x; break;
      case Basis::Cheb: v[i] = v[i - 1] * x2 - v[i - 2]; break;
      case Basis::Leg: v[i] = (v[i - 1] * x * T(2 * di - 1) - v[i - 2] * T(di - 1)) / T(di); break;
      case Basis::Lag: v[i] = (v[i - 1] * (T(2 * di - 1) - x) - v[i - 2] * T(di - 1)) / T(di); break;
      case Basis::Herm: v[i] = v[i - 1] * x2 - v[i - 2] * T(2 * (di - 1)); break;
      case Basis::HermE: v[i] = v[i - 1] * x - v[i - 2] * T(di - 1); break;
    }
  }
  return v;
}

template <typename T>
NDArray companion_s(Basis b, Vec<T> c) {
  c = trimseq(std::move(c));
  if (c.size() < 2) throw_error(ErrorKind::Value, "Series must have maximum degree of at least 1.");
  if (c.size() == 2) {
    T v = -c[0] / c[1];
    if (b == Basis::Lag) v = T{1} + c[0] / c[1];
    if (b == Basis::Herm) v = T{-.5} * c[0] / c[1];
    return store(Vec<T>{v}, Shape{1, 1});
  }
  const std::size_t n = c.size() - 1;
  Vec<T> mat(n * n, T{0});
  auto at = [&](std::size_t r, std::size_t col) -> T& { return mat[r * n + col]; };
  const T last = c[n];
  Vec<double> scl(n, 1.0);
  switch (b) {
    case Basis::Power:
      for (std::size_t i = 0; i + 1 < n; ++i) at(i + 1, i) = T{1};
      for (std::size_t r = 0; r < n; ++r) at(r, n - 1) -= c[r] / last;
      break;
    case Basis::Cheb:
      for (std::size_t i = 1; i < n; ++i) scl[i] = std::sqrt(.5);
      for (std::size_t i = 0; i + 1 < n; ++i) at(i, i + 1) = at(i + 1, i) = T(i == 0 ? std::sqrt(.5) : 0.5);
      for (std::size_t r = 0; r < n; ++r) at(r, n - 1) -= (c[r] / last) * T(scl[r] / scl[n - 1]) * T{.5};
      break;
    case Basis::Leg: {
      for (std::size_t i = 0; i < n; ++i) scl[i] = 1. / std::sqrt(static_cast<double>(2 * i + 1));
      for (std::size_t i = 0; i + 1 < n; ++i)
        at(i, i + 1) = at(i + 1, i) = T(static_cast<double>(i + 1) * scl[i] * scl[i + 1]);
      const double dn = static_cast<double>(n);
      for (std::size_t r = 0; r < n; ++r) at(r, n - 1) -= (c[r] / last) * T(scl[r] / scl[n - 1]) * T(dn / (2 * dn - 1));
      break;
    }
    case Basis::Lag:
      for (std::size_t i = 0; i + 1 < n; ++i) at(i, i + 1) = at(i + 1, i) = T(-static_cast<double>(i + 1));
      for (std::size_t i = 0; i < n; ++i) at(i, i) = T(2. * static_cast<double>(i) + 1.);
      for (std::size_t r = 0; r < n; ++r) at(r, n - 1) += (c[r] / last) * T(static_cast<double>(n));
      break;
    case Basis::Herm:
    case Basis::HermE: {
      const double f = b == Basis::Herm ? 2. : 1.;
      // scl = cumprod(hstack(1, 1/sqrt(f * arange(n-1, 0, -1))))[::-1]
      Vec<double> acc(n, 1.0);
      for (std::size_t k = 1; k < n; ++k) acc[k] = acc[k - 1] * (1. / std::sqrt(f * static_cast<double>(n - k)));
      for (std::size_t k = 0; k < n; ++k) scl[k] = acc[n - 1 - k];
      for (std::size_t i = 0; i + 1 < n; ++i)
        at(i, i + 1) = at(i + 1, i) = T(std::sqrt((b == Basis::Herm ? .5 : 1.) * static_cast<double>(i + 1)));
      for (std::size_t r = 0; r < n; ++r)
        at(r, n - 1) -= b == Basis::Herm ? T(scl[r]) * c[r] / (T{2.0} * last) : T(scl[r]) * c[r] / last;
      break;
    }
  }
  return store(mat, Shape{static_cast<std::int64_t>(n), static_cast<std::int64_t>(n)});
}

template <typename T>
Vec<T> line_s(Basis b, T off, T scl) {
  switch (b) {
    case Basis::Herm: return {off, scl / T{2}};
    case Basis::Lag: return {off + scl, -scl};
    default: return {off, scl};
  }
}

template <typename T>
bool less_lex(const T& a, const T& b) {
  if constexpr (std::is_same_v<T, cd>) {
    return a.real() < b.real() || (a.real() == b.real() && a.imag() < b.imag());
  } else {
    return a < b;
  }
}

template <typename T>
Vec<T> fromroots_s(Basis b, Vec<T> r) {
  if (r.empty()) return Vec<T>{T{1}};
  std::stable_sort(r.begin(), r.end(), less_lex<T>);
  std::vector<Vec<T>> p;
  for (const T& x : r) p.push_back(line_s(b, -x, T{1}));
  std::size_t n = p.size();
  while (n > 1) {
    const std::size_t m = n / 2, rr = n % 2;
    std::vector<Vec<T>> tmp;
    for (std::size_t i = 0; i < m; ++i) tmp.push_back(mul_s(b, p[i], p[i + m]));
    if (rr) tmp[0] = mul_s(b, tmp[0], p.back());
    p = std::move(tmp);
    n = m;
  }
  return p[0];
}

template <typename T>
Vec<T> to_power_s(Basis b, Vec<T> c) {
  c = trimseq(std::move(c));
  const std::size_t n = c.size();
  const Basis P = Basis::Power;
  if (b == Basis::Power) return c;
  if (n == 1) return c;
  if (n == 2) {
    if (b == Basis::Herm) c[1] *= T{2};
    if (b != Basis::Lag) return c;
  }
  Vec<T> c0{c[n - 2]}, c1{c[n - 1]};
  for (std::size_t i = n - 1; i > 1; --i) {
    const Vec<T> tmp = c0;
    const auto di = static_cast<double>(i);
    const Vec<T> ci{c[i - 2]};
    switch (b) {
      case Basis::Cheb:
        c0 = sub_s(ci, c1);
        c1 = add_s(tmp, scaled(mulx_s(P, c1), T{2}));
        break;
      case Basis::Leg:
        c0 = sub_s(ci, divided(scaled(c1, T(di - 1)), T(di)));
        c1 = add_s(tmp, divided(scaled(mulx_s(P, c1), T(2 * di - 1)), T(di)));
        break;
      case Basis::Lag:
        c0 = sub_s(ci, divided(scaled(c1, T(di - 1)), T(di)));
        c1 = add_s(tmp, divided(sub_s(scaled(c1, T(2 * di - 1)), mulx_s(P, c1)), T(di)));
        break;
      case Basis::Herm:
        c0 = sub_s(ci, scaled(c1, T(2 * (di - 1))));
        c1 = add_s(tmp, scaled(mulx_s(P, c1), T{2}));
        break;
      default:  // HermE
        c0 = sub_s(ci, scaled(c1, T(di - 1)));
        c1 = add_s(tmp, mulx_s(P, c1));
        break;
    }
  }
  switch (b) {
    case Basis::Lag: return add_s(c0, sub_s(c1, mulx_s(P, c1)));
    case Basis::Herm: return add_s(c0, scaled(mulx_s(P, c1), T{2}));
    default: return add_s(c0, mulx_s(P, c1));
  }
}

template <typename T>
Vec<T> from_power_s(Basis b, Vec<T> pol) {
  pol = trimseq(std::move(pol));
  if (b == Basis::Power) return pol;
  Vec<T> res{T{0}};
  for (std::size_t i = pol.size(); i-- > 0;) res = add_s(mulx_s(b, res), Vec<T>{pol[i]});
  return res;
}

bool any_complex(std::initializer_list<const NDArray*> xs) {
  for (const NDArray* x : xs)
    if (is_complex(x->dtype())) return true;
  return false;
}

}  // namespace

std::optional<Basis> basis_from_name(std::string_view name) noexcept {
  if (name == "polynomial") return Basis::Power;
  if (name == "chebyshev") return Basis::Cheb;
  if (name == "legendre") return Basis::Leg;
  if (name == "laguerre") return Basis::Lag;
  if (name == "hermite") return Basis::Herm;
  if (name == "hermite_e") return Basis::HermE;
  return std::nullopt;
}

#define NATIVPY_POLY_DISPATCH(cplx, ...) \
  do {                                    \
    if (cplx) {                           \
      using T = cd;                       \
      __VA_ARGS__;                        \
    } else {                              \
      using T = double;                   \
      __VA_ARGS__;                        \
    }                                     \
  } while (0)

NDArray add(Basis, const NDArray& c1, const NDArray& c2) {
  NATIVPY_POLY_DISPATCH(any_complex({&c1, &c2}), return store(add_s(series<T>(c1), series<T>(c2))));
}

NDArray sub(Basis, const NDArray& c1, const NDArray& c2) {
  NATIVPY_POLY_DISPATCH(any_complex({&c1, &c2}), return store(sub_s(series<T>(c1), series<T>(c2))));
}

NDArray mul(Basis b, const NDArray& c1, const NDArray& c2) {
  NATIVPY_POLY_DISPATCH(any_complex({&c1, &c2}), return store(mul_s(b, series<T>(c1), series<T>(c2))));
}

NDArray mulx(Basis b, const NDArray& c) {
  NATIVPY_POLY_DISPATCH(any_complex({&c}), return store(mulx_s(b, series<T>(c))));
}

std::pair<NDArray, NDArray> div(Basis b, const NDArray& c1, const NDArray& c2) {
  NATIVPY_POLY_DISPATCH(any_complex({&c1, &c2}), {
    auto [q, r] = div_s(b, series<T>(c1), series<T>(c2));
    return {store(q), store(r)};
  });
}

NDArray pow(Basis b, const NDArray& c, std::int64_t n) {
  if (n < 0) throw_error(ErrorKind::Value, "Power must be a non-negative integer.");
  NATIVPY_POLY_DISPATCH(any_complex({&c}), return store(pow_s(b, series<T>(c), n)));
}

NDArray der(Basis b, const NDArray& c, std::int64_t m, double scl) {
  if (m < 0) throw_error(ErrorKind::Value, "The order of derivation must be non-negative");
  NATIVPY_POLY_DISPATCH(any_complex({&c}), return store(der_s(b, series<T>(c, false), m, T(scl))));
}

NDArray integ(Basis b, const NDArray& c, std::int64_t m, const NDArray& k, const NDArray& lbnd,
              double scl) {
  if (m < 0) throw_error(ErrorKind::Value, "The order of integration must be non-negative");
  if (k.size() > m) throw_error(ErrorKind::Value, "Too many integration constants");
  if (lbnd.size() != 1) throw_error(ErrorKind::Value, "lbnd must be a scalar.");
  NATIVPY_POLY_DISPATCH(any_complex({&c, &k, &lbnd}), {
    const Vec<T> kv = load<T>(k);
    return store(int_s(b, series<T>(c, false), m, kv, load<T>(lbnd)[0], T(scl)));
  });
}

NDArray val(Basis b, const NDArray& x, const NDArray& c) {
  NATIVPY_POLY_DISPATCH(any_complex({&x, &c}), {
    const Vec<T> cv = series<T>(c, false);
    Vec<T> xv = load<T>(x);
    for (auto& v : xv) v = val_s(b, v, cv);
    return store(xv, x.shape());
  });
}

NDArray vander(Basis b, const NDArray& x, std::int64_t deg) {
  if (deg < 0) throw_error(ErrorKind::Value, "deg must be non-negative");
  NATIVPY_POLY_DISPATCH(any_complex({&x}), {
    const Vec<T> xv = load<T>(x);
    const auto d = static_cast<std::size_t>(deg);
    Vec<T> out;
    out.reserve(xv.size() * (d + 1));
    for (const T& v : xv) {
      const Vec<T> row = vander_row(b, v, d);
      out.insert(out.end(), row.begin(), row.end());
    }
    Shape s = x.shape();
    s.push_back(deg + 1);
    return store(out, s);
  });
}

NDArray companion(Basis b, const NDArray& c) {
  NATIVPY_POLY_DISPATCH(any_complex({&c}), return companion_s(b, series<T>(c)));
}

NDArray fromroots(Basis b, const NDArray& roots) {
  if (roots.ndim() > 1) throw_error(ErrorKind::Value, "Coefficient array is not 1-d");
  NATIVPY_POLY_DISPATCH(any_complex({&roots}), return store(fromroots_s(b, load<T>(roots))));
}

NDArray to_power(Basis b, const NDArray& c) {
  NATIVPY_POLY_DISPATCH(any_complex({&c}), return store(to_power_s(b, series<T>(c))));
}

NDArray from_power(Basis b, const NDArray& c) {
  NATIVPY_POLY_DISPATCH(any_complex({&c}), return store(from_power_s(b, series<T>(c))));
}

#undef NATIVPY_POLY_DISPATCH

}  // namespace nativpy::poly
