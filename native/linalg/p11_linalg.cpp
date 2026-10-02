// P11 linear algebra completion (D-140..D-142).
#include "p11_linalg.hpp"

#include <cmath>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "backend.hpp"
#include "creation.hpp"
#include "error.hpp"
#include "shape.hpp"

#if defined(NATIVPY_HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#include <climits>
#endif

namespace nativpy::linalg {

namespace {

using idx = std::int64_t;
using C = std::complex<double>;

std::size_t sz(idx v) { return static_cast<std::size_t>(v); }

template <typename T>
T* ptr(const NDArray& a) {
  return reinterpret_cast<T*>(a.data());
}

void require_square(const NDArray& a, const char* fn) {
  if (a.ndim() < 2) {
    throw_error(ErrorKind::LinAlg, std::string(fn) + ": " + std::to_string(a.ndim()) +
                                       "-dimensional array given. Array must be at least "
                                       "two-dimensional");
  }
  const auto& s = a.shape();
  if (s[s.size() - 1] != s[s.size() - 2]) {
    throw_error(ErrorKind::LinAlg, "Last 2 dimensions of the array must be square");
  }
}

Shape batch_of(const Shape& s, std::size_t core) {
  return Shape(s.begin(), s.end() - static_cast<std::ptrdiff_t>(core));
}

template <typename T>
void to_colmajor(const T* src, T* dst, idx m, idx n) {
  for (idx i = 0; i < m; ++i)
    for (idx j = 0; j < n; ++j) dst[i + j * m] = src[i * n + j];
}

template <typename E>
E conj_e(E v) {
  if constexpr (std::is_floating_point_v<E>) return v;
  else return std::conj(v);
}
template <typename E>
double real_e(E v) {
  if constexpr (std::is_floating_point_v<E>) return v;
  else return v.real();
}

// Portable lower Cholesky of a row-major n×n matrix `a` (reads i >= j) into
// row-major `l` (upper part zero). Returns false when not positive definite.
template <typename E>
bool chol_lower(const E* a, E* l, idx n) {
  for (idx k = 0; k < n * n; ++k) l[k] = E{};
  for (idx j = 0; j < n; ++j) {
    double d = real_e(a[j * n + j]);
    for (idx k = 0; k < j; ++k) d -= std::norm(l[j * n + k]);
    if (!(d > 0.0)) return false;
    const double ljj = std::sqrt(d);
    l[j * n + j] = E{ljj};
    for (idx i = j + 1; i < n; ++i) {
      E s = a[i * n + j];
      for (idx k = 0; k < j; ++k) s -= l[i * n + k] * conj_e(l[j * n + k]);
      l[i * n + j] = s / ljj;
    }
  }
  return true;
}

#if defined(NATIVPY_HAVE_ACCELERATE)
// LAPACK ?potrf on a column-major copy (D-142). Returns false on info > 0.
template <typename E>
bool chol_lapack(const E* a, E* out, idx n, bool upper) {
  if (n > static_cast<idx>(INT_MAX)) throw_error(ErrorKind::Value, "matrix dimension too large");
  std::vector<E> m(sz(n * n));
  to_colmajor(a, m.data(), n, n);
  const __LAPACK_int nn = static_cast<__LAPACK_int>(n);
  const __LAPACK_int lda = nn > 0 ? nn : 1;
  __LAPACK_int info = 0;
  const char* uplo = upper ? "U" : "L";
  if constexpr (std::is_floating_point_v<E>) dpotrf_(uplo, &nn, m.data(), &lda, &info);
  else zpotrf_(uplo, &nn, m.data(), &lda, &info);
  if (info != 0) return false;
  for (idx i = 0; i < n; ++i)
    for (idx j = 0; j < n; ++j) {
      const bool keep = upper ? j >= i : j <= i;
      out[i * n + j] = keep ? m[sz(i + j * n)] : E{};
    }
  return true;
}
#endif

template <typename E>
bool chol_one(const E* a, E* out, idx n, bool upper) {
#if defined(NATIVPY_HAVE_ACCELERATE)
  if (active_backend().name() == "accelerate") return chol_lapack(a, out, n, upper);
#endif
  if (!upper) return chol_lower(a, out, n);
  // U from the upper triangle: L = chol(B) with B[i][j] = conj(A[j][i]), U = Lᴴ.
  std::vector<E> b(sz(n * n));
  std::vector<E> l(sz(n * n));
  for (idx i = 0; i < n; ++i)
    for (idx j = 0; j < n; ++j) b[sz(i * n + j)] = conj_e(a[j * n + i]);
  if (!chol_lower(b.data(), l.data(), n)) return false;
  for (idx i = 0; i < n; ++i)
    for (idx j = 0; j < n; ++j) out[i * n + j] = conj_e(l[sz(j * n + i)]);
  return true;
}

template <typename Fn>
void dispatch_rc(bool cplx, Fn&& fn) {
  if (cplx) fn(C{});
  else fn(double{});
}

}  // namespace

DType p11_result_dtype(const NDArray& a, const char* fn) {
  switch (a.dtype()) {
    case DType::Float16:
      throw_error(ErrorKind::DType, std::string(fn) + ": array type float16 is unsupported in linalg");
    case DType::Float32:
    case DType::Complex64:
    case DType::Complex128:
      return a.dtype();
    default:
      return DType::Float64;
  }
}

NDArray cholesky(const NDArray& a, bool upper) {
  const DType rt = p11_result_dtype(a, "cholesky");
  require_square(a, "cholesky");
  const bool cplx = is_complex(rt);
  const DType ct = cplx ? DType::Complex128 : DType::Float64;
  const idx n = a.shape().back();
  const NDArray ac = a.astype(ct);
  NDArray out = NDArray::empty(a.shape(), ct);
  const idx nb = shape_size(batch_of(a.shape(), 2));
  dispatch_rc(cplx, [&](auto tag) {
    using E = decltype(tag);
    for (idx t = 0; t < nb && n > 0; ++t) {
      if (!chol_one(ptr<E>(ac) + t * n * n, ptr<E>(out) + t * n * n, n, upper)) {
        throw_error(ErrorKind::LinAlg, "Matrix is not positive definite");
      }
    }
  });
  return rt == ct ? out : out.astype(rt);
}

SlogdetResult slogdet(const NDArray& a) {
  const DType rt = p11_result_dtype(a, "slogdet");
  require_square(a, "slogdet");
  const bool cplx = is_complex(rt);
  const DType ct = cplx ? DType::Complex128 : DType::Float64;
  const DType lt = rt == DType::Float32 || rt == DType::Complex64 ? DType::Float32 : DType::Float64;
  const idx n = a.shape().back();
  const Shape batch = batch_of(a.shape(), 2);
  const idx nb = shape_size(batch);
  const NDArray ac = a.astype(ct);
  NDArray sign = NDArray::empty(batch, ct);
  NDArray logdet = NDArray::empty(batch, DType::Float64);
  dispatch_rc(cplx, [&](auto tag) {
    using E = decltype(tag);
    std::vector<E> m(sz(std::max<idx>(n * n, 1)));
    std::vector<idx> piv(sz(std::max<idx>(n, 1)));
    for (idx t = 0; t < nb; ++t) {
      E s{1.0};
      double ld = 0.0;
      if (n > 0) {
        to_colmajor(ptr<E>(ac) + t * n * n, m.data(), n, n);
        int info = 0;
        if constexpr (std::is_floating_point_v<E>) info = active_backend().f64().getrf(n, m.data(), piv.data());
        else info = active_backend().f64().cgetrf(n, m.data(), piv.data());
        if (info > 0) {
          s = E{0.0};
          ld = -std::numeric_limits<double>::infinity();
        } else {
          for (idx i = 0; i < n; ++i) {
            if (piv[sz(i)] != i) s = -s;
            const E u = m[sz(i + i * n)];
            const double mag = std::abs(u);
            if constexpr (std::is_floating_point_v<E>) {
              if (u < 0) s = -s;
            } else {
              s = s * (u / mag);
            }
            ld += std::log(mag);
          }
        }
      }
      ptr<E>(sign)[t] = s;
      ptr<double>(logdet)[t] = ld;
    }
  });
  return {rt == ct ? sign : sign.astype(rt), lt == DType::Float64 ? logdet : logdet.astype(lt)};
}

NDArray matrix_power(const NDArray& a, std::int64_t n) {
  require_square(a, "matrix_power");
  if (n == 0) {
    const idx m = a.shape().back();
    const NDArray id = eye(m, m, 0, a.dtype());
    NDArray out = NDArray::empty(a.shape(), a.dtype());
    const idx nb = shape_size(batch_of(a.shape(), 2));
    const std::size_t bytes = sz(m * m) * a.itemsize();
    for (idx t = 0; t < nb; ++t) std::memcpy(out.data() + sz(t) * bytes, id.data(), bytes);
    return out;
  }
  NDArray base = n < 0 ? inv(a) : a;
  std::uint64_t e = n < 0 ? static_cast<std::uint64_t>(-(n + 1)) + 1 : static_cast<std::uint64_t>(n);
  if (e == 1) return n < 0 ? base : a.astype(a.dtype());
  if (e == 2) return matmul(base, base);
  if (e == 3) return matmul(matmul(base, base), base);
  std::optional<NDArray> z;
  std::optional<NDArray> result;
  while (e > 0) {
    z = z ? matmul(*z, *z) : base;
    const bool bit = (e & 1U) != 0;
    e >>= 1U;
    if (bit) result = result ? matmul(*result, *z) : *z;
  }
  return *result;
}

}  // namespace nativpy::linalg
