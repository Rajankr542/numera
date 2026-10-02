// P11 linear algebra completion (D-140..D-142).
#include "p11_linalg.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <utility>
#include <complex>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#include "backend.hpp"
#include "broadcast.hpp"
#include "creation.hpp"
#include "error.hpp"
#include "shape.hpp"
#include "shape_ops.hpp"

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

namespace {

// Element (flat C-order) of a real or complex array as complex<double>.
C cget(const NDArray& a, idx i) {
  if (!is_complex(a.dtype())) return {a.get_double(i), 0.0};
  if (a.dtype() == DType::Complex64) {
    const auto v = reinterpret_cast<const std::complex<float>*>(a.data())[i];
    return {v.real(), v.imag()};
  }
  return reinterpret_cast<const C*>(a.data())[i];
}

double eps_of(DType dt) {
  return dt == DType::Float32 || dt == DType::Complex64
             ? static_cast<double>(std::numeric_limits<float>::epsilon())
             : std::numeric_limits<double>::epsilon();
}

// Singular values (descending) of every matrix in `a`, as float64
// (batch..., k), plus their native dtype; hermitian uses |eigvalsh|.
std::pair<NDArray, DType> singular_values(const NDArray& a, bool hermitian) {
  NDArray s = hermitian ? eigvalsh(a) : svd(a, false, false).s;
  const DType sdt = s.dtype();
  s = s.astype(DType::Float64);
  if (hermitian) {
    const idx k = s.ndim() == 0 ? 1 : s.shape().back();
    double* p = ptr<double>(s);
    for (idx t = 0; k > 0 && t < s.size() / k; ++t) {
      std::vector<double> v(p + t * k, p + (t + 1) * k);
      for (double& x : v) x = std::abs(x);
      std::sort(v.begin(), v.end(), std::greater<>());
      std::copy(v.begin(), v.end(), p + t * k);
    }
  }
  return {s, sdt};
}

}  // namespace

NDArray pinv(const NDArray& a, const NDArray& rcond, bool hermitian) {
  if (a.ndim() < 2) {
    throw_error(ErrorKind::LinAlg, std::to_string(a.ndim()) +
                                       "-dimensional array given. Array must be at least "
                                       "two-dimensional");
  }
  if (hermitian) require_square(a, "pinv");
  const Shape& sh = a.shape();
  const idx m = sh[sh.size() - 2];
  const idx n = sh[sh.size() - 1];
  const Shape batch = batch_of(sh, 2);
  Shape out_shape = batch;
  out_shape.push_back(n);
  out_shape.push_back(m);
  if (a.size() == 0) return NDArray::empty(out_shape, a.dtype());
  const NDArray rc = broadcast_to(rcond.astype(DType::Float64), batch).astype(DType::Float64);
  const idx nb = shape_size(batch);
  const idx k = std::min(m, n);
  // Factors: A = U diag(s) Vh (hermitian: U diag(lambda) Uᴴ, Vh = Uᴴ).
  std::optional<NDArray> uo;
  std::optional<NDArray> vho;
  std::optional<NDArray> so;
  if (hermitian) {
    const EigResult e = eigh(a);
    so = e.eigenvalues.astype(DType::Float64);
    uo = e.eigenvectors;
  } else {
    SvdResult r = svd(a, false, true);
    so = r.s.astype(DType::Float64);
    uo = *r.u;
    vho = *r.vh;
  }
  const NDArray& u = *uo;
  const NDArray& s = *so;
  const NDArray vh = vho ? *vho : u;
  const DType dt = u.dtype();
  const bool cplx = is_complex(dt);
  NDArray out = NDArray::empty(out_shape, cplx ? DType::Complex128 : DType::Float64);
  const double* sp = ptr<double>(s);
  std::vector<double> inv_s(sz(k));
  for (idx t = 0; t < nb; ++t) {
    double smax = 0.0;
    for (idx i = 0; i < k; ++i) smax = std::max(smax, std::abs(sp[t * k + i]));
    const double cutoff = ptr<double>(rc)[t] * smax;
    for (idx i = 0; i < k; ++i) {
      const double v = sp[t * k + i];
      inv_s[sz(i)] = std::abs(v) > cutoff ? 1.0 / v : 0.0;
    }
    // res[j, p] = sum_i conj(Vh[i, j]) / s_i * conj(U[p, i]).
    for (idx j = 0; j < n; ++j) {
      for (idx p = 0; p < m; ++p) {
        C acc{};
        for (idx i = 0; i < k; ++i) {
          if (inv_s[sz(i)] == 0.0) continue;
          const C v = hermitian ? std::conj(cget(u, t * m * m + j * m + i))
                                : cget(vh, t * k * n + i * n + j);
          acc += std::conj(v) * inv_s[sz(i)] * std::conj(cget(u, t * m * k + p * k + i));
        }
        if (cplx) ptr<C>(out)[t * n * m + j * m + p] = acc;
        else ptr<double>(out)[t * n * m + j * m + p] = acc.real();
      }
    }
  }
  return out.astype(dt);
}

NDArray matrix_rank(const NDArray& a, const std::optional<NDArray>& tol,
                    const std::optional<NDArray>& rtol, bool hermitian) {
  if (tol && rtol) throw_error(ErrorKind::Value, "`tol` and `rtol` can't be both set.");
  if (a.ndim() < 2) {
    bool any = false;
    for (idx i = 0; i < a.size() && !any; ++i) any = cget(a, i) != C{};
    NDArray out = NDArray::empty({}, DType::Int64);
    out.set_int64(0, any ? 1 : 0);
    return out;
  }
  const auto [s, sdt] = singular_values(a, hermitian);
  const Shape& sh = a.shape();
  const idx m = sh[sh.size() - 2];
  const idx n = sh[sh.size() - 1];
  const idx k = s.shape().back();
  const Shape batch = batch_of(s.shape(), 1);
  const NDArray* thr = tol ? &*tol : (rtol ? &*rtol : nullptr);
  const Shape out_shape = thr ? broadcast_shapes({batch, thr->shape()}) : batch;
  NDArray th = thr ? broadcast_to(thr->astype(DType::Float64), out_shape).astype(DType::Float64)
                   : NDArray::empty({}, DType::Float64);
  if (!thr) th.set_double(0, static_cast<double>(std::max(m, n)) * eps_of(sdt));
  const NDArray sb = broadcast_to(s, [&] {
                       Shape x = out_shape;
                       x.push_back(k);
                       return x;
                     }()).astype(DType::Float64);
  NDArray out = NDArray::empty(out_shape, DType::Int64);
  const double* sp = ptr<double>(sb);
  for (idx t = 0; t < shape_size(out_shape); ++t) {
    double smax = 0.0;
    for (idx i = 0; i < k; ++i) smax = std::max(smax, sp[t * k + i]);
    const double tv = ptr<double>(th)[thr ? t : 0];
    const double cut = tol ? tv : smax * tv;
    std::int64_t r = 0;
    for (idx i = 0; i < k; ++i) r += sp[t * k + i] > cut ? 1 : 0;
    out.set_int64(t, r);
  }
  return out;
}

NDArray cond(const NDArray& a, const NormOrd& p) {
  if (a.ndim() >= 2 && a.size() == 0) {
    const auto& sh = a.shape();
    if (sh[sh.size() - 1] * sh[sh.size() - 2] == 0) {
      throw_error(ErrorKind::LinAlg, "cond is not defined on empty arrays");
    }
  }
  const DType rt = p11_result_dtype(a, "cond");
  const DType real_t = rt == DType::Float32 || rt == DType::Complex64 ? DType::Float32 : DType::Float64;
  std::optional<NDArray> ro;
  const bool svd_path = p.kind == "default" || (p.kind == "p" && (p.p == 2.0 || p.p == -2.0));
  if (svd_path) {
    if (a.ndim() < 2) svd(a, false, false);  // raises the NumPy LinAlgError
    const NDArray s = svd(a, false, false).s.astype(DType::Float64);
    const idx k = s.shape().back();
    const Shape batch = batch_of(s.shape(), 1);
    ro = NDArray::empty(batch, DType::Float64);
    NDArray& r = *ro;
    for (idx t = 0; t < shape_size(batch); ++t) {
      const double hi = s.get_double(t * k);
      const double lo = s.get_double(t * k + k - 1);
      r.set_double(t, p.kind == "p" && p.p == -2.0 ? lo / hi : hi / lo);
    }
  } else {
    require_square(a, "cond");
    const bool cplx = is_complex(rt);
    const DType ct = cplx ? DType::Complex128 : DType::Float64;
    const idx n = a.shape().back();
    const NDArray ac = a.astype(ct);
    NDArray invx = NDArray::empty(a.shape(), ct);
    const idx nb = shape_size(batch_of(a.shape(), 2));
    dispatch_rc(cplx, [&](auto tag) {
      using E = decltype(tag);
      std::vector<E> am(sz(n * n));
      std::vector<E> bm(sz(n * n));
      for (idx t = 0; t < nb; ++t) {
        to_colmajor(ptr<E>(ac) + t * n * n, am.data(), n, n);
        std::fill(bm.begin(), bm.end(), E{0.0});
        for (idx i = 0; i < n; ++i) bm[sz(i + i * n)] = E{1.0};
        int info = 0;
        if constexpr (std::is_floating_point_v<E>) info = active_backend().f64().gesv(n, n, am.data(), bm.data());
        else info = active_backend().f64().cgesv(n, n, am.data(), bm.data());
        E* dst = ptr<E>(invx) + t * n * n;
        for (idx i = 0; i < n; ++i)
          for (idx j = 0; j < n; ++j)
            dst[i * n + j] = info > 0 ? E{std::numeric_limits<double>::quiet_NaN()} : bm[sz(i + j * n)];
      }
    });
    const std::vector<std::int64_t> axes{-2, -1};
    const NDArray n1 = norm(a, p, axes, false).astype(DType::Float64);
    const NDArray n2 = norm(invx, p, axes, false).astype(DType::Float64);
    ro = NDArray::empty(n1.shape(), DType::Float64);
    NDArray& r = *ro;
    for (idx t = 0; t < r.size(); ++t) r.set_double(t, n1.get_double(t) * n2.get_double(t));
  }
  NDArray& r = *ro;
  // NaN -> inf unless the matrix itself contains NaN.
  const idx mn = r.size() == 0 ? 0 : a.size() / r.size();
  for (idx t = 0; t < r.size(); ++t) {
    if (!std::isnan(r.get_double(t))) continue;
    bool has_nan = false;
    for (idx i = 0; i < mn && !has_nan; ++i) {
      const C v = cget(a, t * mn + i);
      has_nan = std::isnan(v.real()) || std::isnan(v.imag());
    }
    if (!has_nan) r.set_double(t, std::numeric_limits<double>::infinity());
  }
  return real_t == DType::Float64 ? r : r.astype(real_t);
}

}  // namespace nativpy::linalg

namespace nativpy::linalg {

namespace {

using Labels = std::string;

struct Term {
  NDArray a;
  Labels labels;
};

// Repeated labels -> strided diagonal view (unique labels, first-seen order).
Term take_diagonals(const NDArray& a, const Labels& lab, std::size_t op) {
  Labels out;
  Shape shape;
  Strides strides;
  for (std::size_t d = 0; d < lab.size(); ++d) {
    const auto pos = out.find(lab[d]);
    if (pos == Labels::npos) {
      out.push_back(lab[d]);
      shape.push_back(a.shape()[d]);
      strides.push_back(a.strides()[d]);
    } else {
      if (shape[pos] != a.shape()[d]) {
        throw_error(ErrorKind::Value, "dimensions in operand " + std::to_string(op) +
                                          " for collapsing index '" + std::string(1, lab[d]) +
                                          "' don't match (" + std::to_string(shape[pos]) + " != " +
                                          std::to_string(a.shape()[d]) + ")");
      }
      strides[pos] += a.strides()[d];
    }
  }
  if (out.size() == lab.size()) return {a, lab};
  return {a.view(shape, strides, a.offset()), out};
}

std::vector<std::int64_t> perm_for(const Labels& from, const Labels& to) {
  std::vector<std::int64_t> p;
  for (const char c : to) p.push_back(static_cast<std::int64_t>(from.find(c)));
  return p;
}

// Sums out every label of `t` not in `keep` (product with a ones vector, so
// the dtype is kept: integers wrap, bool is OR-of-AND).
Term sum_out(const Term& t, const Labels& keep) {
  Labels kept;
  Labels gone;
  for (const char c : t.labels) (keep.find(c) != Labels::npos ? kept : gone).push_back(c);
  if (gone.empty()) return t;
  Shape ks;
  idx q = 1;
  for (const char c : kept) ks.push_back(t.a.shape()[t.labels.find(c)]);
  for (const char c : gone) q *= t.a.shape()[t.labels.find(c)];
  const idx p = shape_size(ks);
  const NDArray x = transpose(t.a, perm_for(t.labels, kept + gone)).reshape({p, q});
  const NDArray r = matmul(x, ones({q}, t.a.dtype()));
  return {r.reshape(ks), kept};
}

// Pairwise contraction to the labels both share or `keep` lists, via matmul
// on (batch, M, K) @ (batch, K, N). Size-1 dimensions broadcast to `sizes`.
Term contract_pair(Term x, Term y, const Labels& keep, const std::array<idx, 256>& sizes) {
  auto in = [](const Labels& l, char c) { return l.find(c) != Labels::npos; };
  Labels both = keep;
  for (const char c : y.labels) both.push_back(c);
  x = sum_out(x, both);
  both = keep + x.labels;
  y = sum_out(y, both);
  Labels batch;
  Labels contr;
  Labels fx;
  Labels fy;
  for (const char c : x.labels) {
    if (!in(y.labels, c)) fx.push_back(c);
    else if (in(keep, c)) batch.push_back(c);
    else contr.push_back(c);
  }
  for (const char c : y.labels)
    if (!in(x.labels, c)) fy.push_back(c);
  auto prod_of = [&](const Labels& l) {
    idx p = 1;
    for (const char c : l) p *= sizes[static_cast<unsigned char>(c)];
    return p;
  };
  auto arrange = [&](const Term& t, const Labels& order, const Shape& shape2) {
    NDArray v = transpose(t.a, perm_for(t.labels, order));
    Shape full;
    for (const char c : order) full.push_back(sizes[static_cast<unsigned char>(c)]);
    if (v.shape() != full) v = broadcast_to(v, full);
    return v.reshape(shape2);
  };
  const idx nb = prod_of(batch);
  const idx m = prod_of(fx);
  const idx k = prod_of(contr);
  const idx n = prod_of(fy);
  const NDArray a = arrange(x, batch + fx + contr, {nb, m, k});
  const NDArray b = arrange(y, batch + contr + fy, {nb, k, n});
  const Labels out = batch + fx + fy;
  Shape os;
  for (const char c : out) os.push_back(sizes[static_cast<unsigned char>(c)]);
  return {matmul(a, b).reshape(os), out};
}

}  // namespace

NDArray einsum(const std::vector<NDArray>& operands, const std::vector<std::string>& terms,
               const std::vector<EinsumStep>& steps) {
  if (operands.empty()) throw_error(ErrorKind::Value, "No input operands");
  if (terms.size() != operands.size()) {
    throw_error(ErrorKind::Value, "Number of einsum subscripts must be equal to the number of operands.");
  }
  DType dt = operands[0].dtype();
  for (const auto& o : operands) dt = promote_types(dt, o.dtype());
  std::array<idx, 256> sizes{};
  sizes.fill(-1);
  std::vector<Term> work;
  for (std::size_t i = 0; i < operands.size(); ++i) {
    const NDArray& o = operands[i];
    if (o.ndim() != terms[i].size()) {
      throw_error(ErrorKind::Value, "Einstein sum subscript " + terms[i] +
                                        " does not contain the correct number of indices for operand " +
                                        std::to_string(i) + ".");
    }
    for (std::size_t d = 0; d < o.ndim(); ++d) {
      idx& s = sizes[static_cast<unsigned char>(terms[i][d])];
      const idx dim = o.shape()[d];
      if (s == -1 || s == 1) s = dim;
      else if (dim != 1 && dim != s) {
        throw_error(ErrorKind::Value, std::string("Size of label '") + terms[i][d] + "' for operand " +
                                          std::to_string(i) + " (" + std::to_string(s) +
                                          ") does not match previous terms (" + std::to_string(dim) + ").");
      }
    }
    work.push_back(take_diagonals(o.dtype() == dt ? o : o.astype(dt), terms[i], i));
  }
  for (const auto& st : steps) {
    std::vector<Term> picked;
    for (const auto p : st.positions) {
      if (p < 0 || static_cast<std::size_t>(p) >= work.size()) {
        throw_error(ErrorKind::Value, "einsum_path contraction index out of range");
      }
      picked.push_back(work[sz(p)]);
      work.erase(work.begin() + p);
    }
    if (picked.empty()) throw_error(ErrorKind::Value, "einsum_path contraction is empty");
    for (const char c : st.result) {
      bool seen = false;
      for (const auto& t : picked) seen = seen || t.labels.find(c) != Labels::npos;
      if (!seen) {
        throw_error(ErrorKind::Value, std::string("einstein sum subscripts string included output subscript '") +
                                          c + "' which never appeared in an input");
      }
    }
    // Left to right over the picked operands; `keep` = result labels plus
    // the labels of picked operands not yet folded in.
    Term acc = picked[0];
    for (std::size_t j = 1; j < picked.size(); ++j) {
      Labels keep = st.result;
      for (std::size_t r = j + 1; r < picked.size(); ++r) keep += picked[r].labels;
      acc = contract_pair(acc, picked[j], keep, sizes);
    }
    acc = sum_out(acc, st.result);
    NDArray r = transpose(acc.a, perm_for(acc.labels, st.result));
    Shape full;
    for (const char c : st.result) full.push_back(sizes[static_cast<unsigned char>(c)]);
    if (r.shape() != full) r = broadcast_to(r, full);
    work.push_back({r, st.result});
  }
  if (work.size() != 1) {
    throw_error(ErrorKind::Value, "Invalid einsum_path is specified: " + std::to_string(work.size() - 1) +
                                      " more operands has to be contracted.");
  }
  return work[0].a.copy();
}

}  // namespace nativpy::linalg
