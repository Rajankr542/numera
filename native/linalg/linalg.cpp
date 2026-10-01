// NDArray-level linear algebra (M8, D-018).
#include "linalg.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend.hpp"
#include "broadcast.hpp"
#include "cast.hpp"
#include "dtype.hpp"
#include "error.hpp"
#include "shape.hpp"
#include "shape_ops.hpp"
#include "ufunc.hpp"
#include "ufunc_kernels.hpp"

namespace nativpy::linalg {

namespace {

using idx = std::int64_t;

std::size_t sz(idx v) { return static_cast<std::size_t>(v); }

// Typed pointer into a C-contiguous array.
template <typename T>
T* ptr(const NDArray& a) {
  return reinterpret_cast<T*>(a.data());
}

void reject_complex(const NDArray& a, const char* fn) {
  if (is_complex(a.dtype())) {
    throw_error(ErrorKind::NotImplemented,
                std::string(fn) + ": complex input is not supported yet (D-008)");
  }
}

// Compute dtype for decompositions: float32 stays, float16 rejected,
// bool/int -> float64.
DType decomp_dtype(const NDArray& a, const char* fn) {
  reject_complex(a, fn);
  if (a.dtype() == DType::Float16) {
    throw_error(ErrorKind::DType, std::string(fn) + ": array type float16 is unsupported in linalg");
  }
  return a.dtype() == DType::Float32 ? DType::Float32 : DType::Float64;
}

void require_2d(const NDArray& a, const char* fn) {
  if (a.ndim() < 2) {
    throw_error(ErrorKind::LinAlg, std::string(fn) + ": " + std::to_string(a.ndim()) +
                                       "-dimensional array given. Array must be at least "
                                       "two-dimensional");
  }
}

void require_square(const NDArray& a, const char* fn) {
  require_2d(a, fn);
  const auto& s = a.shape();
  if (s[s.size() - 1] != s[s.size() - 2]) {
    throw_error(ErrorKind::LinAlg, std::string(fn) + ": Last 2 dimensions of the array must be square");
  }
}

Shape batch_of(const Shape& s, std::size_t core) {
  return Shape(s.begin(), s.end() - static_cast<std::ptrdiff_t>(core));
}

Shape concat(Shape a, const Shape& b) {
  a.insert(a.end(), b.begin(), b.end());
  return a;
}

// Row-major m×n (src) -> column-major (dst), and back.
template <typename T>
void to_colmajor(const T* src, T* dst, idx m, idx n) {
  for (idx i = 0; i < m; ++i)
    for (idx j = 0; j < n; ++j) dst[i + j * m] = src[i * n + j];
}
template <typename T>
void from_colmajor(const T* src, T* dst, idx m, idx n) {
  for (idx i = 0; i < m; ++i)
    for (idx j = 0; j < n; ++j) dst[i * n + j] = src[i + j * m];
}

template <typename T>
const Routines<T>& routines() {
  if constexpr (std::is_same_v<T, float>) return active_backend().f32();
  else return active_backend().f64();
}

// Dispatches fn(T{}) for T in {float, double} based on dtype.
template <typename Fn>
decltype(auto) dispatch_real(DType dt, Fn&& fn) {
  if (dt == DType::Float32) return fn(float{});
  return fn(double{});
}

[[noreturn]] void linalg_fail(const char* what) { throw_error(ErrorKind::LinAlg, what); }

// ---- matmul ---------------------------------------------------------------

// Exact product loop for non-BLAS dtypes (integers wrap, bool OR-of-AND,
// float16 accumulates in float32).
template <typename T>
void loop_gemm(const T* a, const T* b, T* c, idx m, idx n, idx k) {
  using Acc = std::conditional_t<std::is_same_v<T, float16_t>, float, T>;
  for (idx i = 0; i < m; ++i) {
    for (idx j = 0; j < n; ++j) {
      Acc acc{};
      for (idx p = 0; p < k; ++p) {
        if constexpr (std::is_same_v<T, float16_t>) {
          acc += cast_value<float>(a[i * k + p]) * cast_value<float>(b[p * n + j]);
        } else {
          acc = kernels::add(acc, kernels::mul(a[i * k + p], b[p * n + j]));
        }
      }
      if constexpr (std::is_same_v<T, float16_t>) c[i * n + j] = cast_value<float16_t>(acc);
      else c[i * n + j] = acc;
    }
  }
}

// Core batched matmul of a (..., m, k) and b (..., k, n) (both ≥ 2-D).
NDArray matmul_2d(const NDArray& a, const NDArray& b) {
  const Shape& as = a.shape();
  const Shape& bs = b.shape();
  const idx m = as[as.size() - 2];
  const idx k = as[as.size() - 1];
  const idx k2 = bs[bs.size() - 2];
  const idx n = bs[bs.size() - 1];
  if (k != k2) {
    throw_error(ErrorKind::Shape,
                "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc "
                "signature (n?,k),(k,m?)->(n?,m?) (size " +
                    std::to_string(k2) + " is different from " + std::to_string(k) + ")");
  }
  const Shape batch = broadcast_shapes({batch_of(as, 2), batch_of(bs, 2)});
  const DType dt = promote_types(a.dtype(), b.dtype());
  // D-022: use an operand in place when it already has the loop layout.
  const auto operand = [&batch, dt](const NDArray& x, const Shape& core) {
    const Shape full = concat(batch, core);
    if (x.dtype() == dt && x.shape() == full && x.is_c_contiguous()) return x;
    return broadcast_to(x, full).astype(dt);
  };
  const NDArray ac = operand(a, {m, k});
  const NDArray bc = operand(b, {k, n});
  // Every kernel below writes all m*n outputs (D-022), so no zero fill.
  NDArray out = NDArray::empty(concat(batch, {m, n}), dt);
  const idx nb = shape_size(batch);
  if (m * n == 0) return out;
  dispatch_dtype(dt, [&](auto tag) {
    using T = dtype_t<decltype(tag)::value>;
    const T* ap = ptr<T>(ac);
    const T* bp = ptr<T>(bc);
    T* cp = ptr<T>(out);
    for (idx t = 0; t < nb; ++t) {
      const T* at = ap + t * m * k;
      const T* bt = bp + t * k * n;
      T* ct = cp + t * m * n;
      if constexpr (is_complex_v<T>) {
        routines<typename T::value_type>().cgemm(m, n, k, at, bt, ct);
      } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        routines<T>().gemm(m, n, k, at, bt, ct);
      } else {
        loop_gemm(at, bt, ct, m, n, k);
      }
    }
  });
  return out;
}

// ---- batched square-matrix helpers ------------------------------------------

// Contiguous copy of `a` in compute dtype.
NDArray as_compute(const NDArray& a, DType dt) { return a.astype(dt); }

template <typename T>
bool all_finite(const T* p, idx n) {
  for (idx i = 0; i < n; ++i)
    if (!std::isfinite(p[i])) return false;
  return true;
}
template <typename R>
bool all_finite(const std::complex<R>* p, idx n) {
  return all_finite(reinterpret_cast<const R*>(p), 2 * n);
}

template <typename T>
T det_one(const T* rowmajor, idx n, std::vector<T>& work, std::vector<idx>& piv) {
  if (n == 0) return T{1};
  to_colmajor(rowmajor, work.data(), n, n);
  const int info = routines<T>().getrf(n, work.data(), piv.data());
  if (info > 0) return T{0};
  T d{1};
  for (idx i = 0; i < n; ++i) {
    d *= work[sz(i + i * n)];
    if (piv[sz(i)] != i) d = -d;
  }
  return d;
}

// NumPy's umath_linalg complex det (D-038): sign from pivot parity times
// u_ii/|u_ii| (unfused `mult`), magnitude exp(sum log|u_ii|), then
// mult(sign, exp(logdet) + 0j). A singular getrf gives sign 0, logdet -inf.
template <typename R>
std::complex<R> cdet_one(const std::complex<R>* rowmajor, idx n,
                         std::vector<std::complex<R>>& work, std::vector<idx>& piv) {
  if (n == 0) return {R{1}, R{0}};
  to_colmajor(rowmajor, work.data(), n, n);
  const int info = routines<R>().cgetrf(n, work.data(), piv.data());
  R sr = 0;
  R si = 0;
  R logdet = -std::numeric_limits<R>::infinity();
  if (info == 0) {
    idx swaps = 0;
    for (idx i = 0; i < n; ++i) swaps += piv[sz(i)] != i ? 1 : 0;
    sr = swaps % 2 != 0 ? R{-1} : R{1};
    logdet = 0;
    for (idx i = 0; i < n; ++i) {
      const std::complex<R> u = work[sz(i + i * n)];
      const R mag = std::hypot(u.real(), u.imag());
      const R er = u.real() / mag;
      const R ei = u.imag() / mag;
      const R nr = sr * er - si * ei;
      const R ni = sr * ei + si * er;
      sr = nr;
      si = ni;
      logdet += std::log(mag);
    }
  }
  const R e = std::exp(logdet);
  return {sr * e - si * R{0}, sr * R{0} + si * e};
}

// NumPy `_commonType` when some operand is complex (D-039): complex64 only if
// every operand is float32 or complex64, else complex128. float16 rejected.
DType complex_result(std::initializer_list<const NDArray*> ops, const char* fn) {
  bool single = true;
  for (const NDArray* op : ops) {
    const DType d = op->dtype();
    if (d == DType::Float16) {
      throw_error(ErrorKind::DType, std::string(fn) + ": array type float16 is unsupported in linalg");
    }
    single = single && (d == DType::Float32 || d == DType::Complex64);
  }
  return single ? DType::Complex64 : DType::Complex128;
}

// gesv for a real or complex element type E (D-039).
template <typename E>
int gesv_e(idx n, idx r, E* a, E* b) {
  if constexpr (std::is_floating_point_v<E>) return routines<E>().gesv(n, r, a, b);
  else return routines<typename E::value_type>().cgesv(n, r, a, b);
}

// geqrf / orgqr for a real or complex element type E (D-040).
template <typename E>
int geqrf_e(idx m, idx n, E* a, E* tau) {
  if constexpr (std::is_floating_point_v<E>) return routines<E>().geqrf(m, n, a, tau);
  else return routines<typename E::value_type>().cgeqrf(m, n, a, tau);
}
template <typename E>
int orgqr_e(idx m, idx cols, idx k, E* q, const E* tau) {
  if constexpr (std::is_floating_point_v<E>) return routines<E>().orgqr(m, cols, k, q, tau);
  else return routines<typename E::value_type>().cungqr(m, cols, k, q, tau);
}

// Real scalar type of E (E itself when real).
template <typename E>
using real_of_t = decltype(std::real(E{}));

// gesdd for a real or complex element type E; s is real (D-041).
template <typename E>
int gesdd_e(idx m, idx n, E* a, real_of_t<E>* s, E* u, E* vt, bool full) {
  if constexpr (std::is_floating_point_v<E>) return routines<E>().gesdd(m, n, a, s, u, vt, full);
  else return routines<typename E::value_type>().cgesdd(m, n, a, s, u, vt, full);
}

// syevd / cheevd for a real or complex element type E; w is real (D-042).
template <typename E>
int evd_e(idx n, E* a, real_of_t<E>* w, bool vectors) {
  if constexpr (std::is_floating_point_v<E>) return routines<E>().syevd(n, a, w, vectors);
  else return routines<typename E::value_type>().cheevd(n, a, w, vectors);
}

// Dispatches fn(E{}) for the inv/solve compute dtype: float32, float64 or
// complex128 (complex input always computes in complex128, D-039).
template <typename Fn>
void dispatch_solve(DType ct, Fn&& fn) {
  if (ct == DType::Complex128) fn(std::complex<double>{});
  else if (ct == DType::Float32) fn(float{});
  else fn(double{});
}

// Shared eigh / eigvalsh (D-042): complex computes in complex128 ('D->dD'),
// eigenvalues real; `vectors` = JOBZ.
std::pair<NDArray, std::optional<NDArray>> eigh_impl(const NDArray& a, bool vectors,
                                                     const char* fn) {
  const bool cplx = is_complex(a.dtype());
  const DType dt = cplx ? a.dtype() : decomp_dtype(a, fn);
  const DType ct = cplx ? DType::Complex128 : dt;
  const DType wct = cplx ? DType::Float64 : dt;
  const DType wdt = dt == DType::Complex64 ? DType::Float32 : wct;
  require_square(a, fn);
  const idx n = a.shape().back();
  const Shape batch = batch_of(a.shape(), 2);
  const NDArray ac = as_compute(a, ct);
  NDArray w = NDArray::empty(concat(batch, {n}), wct);
  std::optional<NDArray> v;
  if (vectors) v = NDArray::empty(a.shape(), ct);
  dispatch_solve(ct, [&](auto tag) {
    using T = decltype(tag);
    using R = real_of_t<T>;
    std::vector<T> m(sz(n * n));
    for (idx t = 0; t < shape_size(batch) && n > 0; ++t) {
      // Column-major copy keeps element (i, j) in place; UPLO='L' reads i >= j.
      to_colmajor(ptr<T>(ac) + t * n * n, m.data(), n, n);
      if (evd_e(n, m.data(), ptr<R>(w) + t * n, vectors) != 0) {
        linalg_fail("Eigenvalues did not converge");
      }
      if (vectors) from_colmajor(m.data(), ptr<T>(*v) + t * n * n, n, n);
    }
  });
  if (wct != wdt) w = w.astype(wdt);
  if (v && ct != dt) v = v->astype(dt);
  return {w, v};
}

}  // namespace

NDArray matmul(const NDArray& a, const NDArray& b) {
  if (a.ndim() == 0 || b.ndim() == 0) {
    throw_error(ErrorKind::Value,
                "matmul: Input operand does not have enough dimensions (scalar operands are "
                "not allowed)");
  }
  const bool a1 = a.ndim() == 1;
  const bool b1 = b.ndim() == 1;
  const NDArray a2 = a1 ? a.reshape({1, a.shape()[0]}) : a;
  const NDArray b2 = b1 ? b.reshape({b.shape()[0], 1}) : b;
  NDArray out = matmul_2d(a2, b2);
  Shape os = out.shape();
  if (b1) os.erase(os.end() - 1);
  if (a1) os.erase(os.end() - (b1 ? 1 : 2));
  return os == out.shape() ? out : out.reshape(os);
}

NDArray dot(const NDArray& a, const NDArray& b) {
  if (a.ndim() == 0 || b.ndim() == 0) return binary(BinaryOp::Multiply, a, b);
  if (b.ndim() <= 2) return matmul(a, b);  // matches dot for these ranks
  // General case: sum over last axis of a and second-to-last of b.
  // result shape = a.shape[:-1] + b.shape[:-2] + b.shape[-1:]
  const Shape& as = a.shape();
  const Shape& bs = b.shape();
  const idx k = as.back();
  if (k != bs[bs.size() - 2]) {
    throw_error(ErrorKind::Shape, "dot: shapes " + shape_to_string(as) + " and " +
                                      shape_to_string(bs) + " not aligned");
  }
  // Move b's contraction axis to front: (k, rest...), then flatten rest.
  const idx nb = static_cast<idx>(bs.size());
  const NDArray bm = moveaxis(b, {nb - 2}, {0});
  Shape brest = bm.shape();
  brest.erase(brest.begin());
  const Shape arest = batch_of(as, 1);
  const NDArray a2 = a.reshape({shape_size(arest), k});
  const NDArray b2 = bm.reshape({k, shape_size(brest)});
  return matmul_2d(a2, b2).reshape(concat(arest, brest));
}

NDArray inner(const NDArray& a, const NDArray& b) {
  if (a.ndim() == 0 || b.ndim() == 0) return binary(BinaryOp::Multiply, a, b);
  if (a.shape().back() != b.shape().back()) {
    throw_error(ErrorKind::Shape, "inner: shapes " + shape_to_string(a.shape()) + " and " +
                                      shape_to_string(b.shape()) + " not aligned");
  }
  // inner(a, b) = dot(a, moveaxis(b, -1, 0)) generalised: contract last axes.
  const idx k = a.shape().back();
  const Shape ab = batch_of(a.shape(), 1);
  const Shape bb = batch_of(b.shape(), 1);
  const NDArray a2 = a.reshape({shape_size(ab), k});
  const NDArray bt = transpose(b.reshape({shape_size(bb), k}), {1, 0});
  return matmul_2d(a2, bt).reshape(concat(ab, bb));
}

NDArray outer(const NDArray& a, const NDArray& b) {
  const NDArray a2 = a.reshape({a.size(), 1});
  const NDArray b2 = b.reshape({1, b.size()});
  return binary(BinaryOp::Multiply, a2, b2);
}

NDArray det(const NDArray& a) {
  const bool cplx = is_complex(a.dtype());
  const DType dt = cplx ? a.dtype() : decomp_dtype(a, "det");
  require_square(a, "det");
  const idx n = a.shape().back();
  const Shape batch = batch_of(a.shape(), 2);
  if (cplx) {
    // NumPy computes every complex det in complex128 ('D->D') and casts the
    // result to the input width afterwards (D-038).
    const NDArray a128 = as_compute(a, DType::Complex128);
    NDArray o128 = NDArray::empty(batch, DType::Complex128);
    using C = std::complex<double>;
    std::vector<C> work(sz(std::max<idx>(n * n, 1)));
    std::vector<idx> piv(sz(std::max<idx>(n, 1)));
    const C* src = ptr<C>(a128);
    C* dst = ptr<C>(o128);
    for (idx t = 0; t < shape_size(batch); ++t) dst[t] = cdet_one(src + t * n * n, n, work, piv);
    return dt == DType::Complex128 ? o128 : o128.astype(dt);
  }
  const NDArray ac = as_compute(a, dt);
  NDArray out = NDArray::empty(batch, dt);
  dispatch_real(dt, [&](auto tag) {
    using T = decltype(tag);
    std::vector<T> work(sz(std::max<idx>(n * n, 1)));
    std::vector<idx> piv(sz(std::max<idx>(n, 1)));
    const T* src = ptr<T>(ac);
    T* dst = ptr<T>(out);
    for (idx t = 0; t < shape_size(batch); ++t) dst[t] = det_one(src + t * n * n, n, work, piv);
  });
  return out;
}

NDArray inv(const NDArray& a) {
  const bool cplx = is_complex(a.dtype());
  const DType dt = cplx ? complex_result({&a}, "inv") : decomp_dtype(a, "inv");
  require_square(a, "inv");
  const DType ct = cplx ? DType::Complex128 : dt;  // NumPy 'D->D' (D-039)
  const idx n = a.shape().back();
  const NDArray ac = as_compute(a, ct);
  NDArray out = NDArray::empty(a.shape(), ct);
  const idx nb = shape_size(batch_of(a.shape(), 2));
  dispatch_solve(ct, [&](auto tag) {
    using T = decltype(tag);
    std::vector<T> am(sz(n * n));
    std::vector<T> bm(sz(n * n));
    for (idx t = 0; t < nb && n > 0; ++t) {
      to_colmajor(ptr<T>(ac) + t * n * n, am.data(), n, n);
      std::fill(bm.begin(), bm.end(), T{0});
      for (idx i = 0; i < n; ++i) bm[sz(i + i * n)] = T{1};
      if (gesv_e(n, n, am.data(), bm.data()) > 0) linalg_fail("Singular matrix");
      from_colmajor(bm.data(), ptr<T>(out) + t * n * n, n, n);
    }
  });
  return ct == dt ? out : out.astype(dt);
}

NDArray solve(const NDArray& a, const NDArray& b) {
  const bool cplx = is_complex(a.dtype()) || is_complex(b.dtype());
  const DType dt = cplx ? complex_result({&a, &b}, "solve")
                        : promote_types(decomp_dtype(a, "solve"), decomp_dtype(b, "solve"));
  const DType ct = cplx ? DType::Complex128 : dt;  // NumPy 'DD->D' (D-039)
  require_square(a, "solve");
  const idx n = a.shape().back();
  // NumPy 2: b is a vector only when b.ndim == 1.
  const bool vec = b.ndim() == 1;
  const NDArray b2 = vec ? b.reshape({b.shape()[0], 1}) : b;
  if (b2.shape()[b2.ndim() - 2] != n) {
    throw_error(ErrorKind::Shape, "solve: Input operand 1 has a mismatch in its core dimension 0 "
                                  "(size " + std::to_string(b2.shape()[b2.ndim() - 2]) +
                                      " is different from " + std::to_string(n) + ")");
  }
  const idx r = b2.shape().back();
  const Shape batch = broadcast_shapes({batch_of(a.shape(), 2), batch_of(b2.shape(), 2)});
  const NDArray ac = broadcast_to(a, concat(batch, {n, n})).astype(ct);
  const NDArray bc = broadcast_to(b2, concat(batch, {n, r})).astype(ct);
  NDArray out = NDArray::empty(concat(batch, {n, r}), ct);
  dispatch_solve(ct, [&](auto tag) {
    using T = decltype(tag);
    std::vector<T> am(sz(n * n));
    std::vector<T> bm(sz(n * r));
    for (idx t = 0; t < shape_size(batch); ++t) {
      if (n == 0 || r == 0) break;
      to_colmajor(ptr<T>(ac) + t * n * n, am.data(), n, n);
      to_colmajor(ptr<T>(bc) + t * n * r, bm.data(), n, r);
      if (gesv_e(n, r, am.data(), bm.data()) > 0) linalg_fail("Singular matrix");
      from_colmajor(bm.data(), ptr<T>(out) + t * n * r, n, r);
    }
  });
  if (ct != dt) out = out.astype(dt);
  if (vec) return out.reshape(concat(batch, {n}));
  return out;
}

EigResult eigh(const NDArray& a) {
  auto [w, v] = eigh_impl(a, true, "eigh");
  return {w, *v};
}

NDArray eigvalsh(const NDArray& a) { return eigh_impl(a, false, "eigvalsh").first; }

namespace {

// Shared eig / eigvals (D-043). Real input: float32 stays (D-018), results
// complex. Complex input computes in complex128 ('D->DD') and is cast once.
// `vectors` = JOBVR ('N' for eigvals, as NumPy's eigvals gufunc).
std::pair<NDArray, std::optional<NDArray>> eig_impl(const NDArray& a, bool vectors,
                                                    const char* fn) {
  const bool cplx = is_complex(a.dtype());
  const DType dt = cplx ? a.dtype() : decomp_dtype(a, fn);
  require_square(a, fn);
  const idx n = a.shape().back();
  const Shape batch = batch_of(a.shape(), 2);
  const DType ct = cplx ? DType::Complex128 : dt;
  const DType wct = cplx ? DType::Complex128
                         : (dt == DType::Float32 ? DType::Complex64 : DType::Complex128);
  const DType wdt = dt == DType::Complex64 ? DType::Complex64 : wct;
  const NDArray ac = as_compute(a, ct);
  NDArray w = NDArray::empty(concat(batch, {n}), wct);
  std::optional<NDArray> v;
  if (vectors) v = NDArray::empty(a.shape(), wct);
  dispatch_solve(ct, [&](auto tag) {
    using E = decltype(tag);
    using R = real_of_t<E>;
    using C = std::complex<R>;
    std::vector<E> m(sz(n * n));
    std::vector<C> vc(vectors ? sz(n * n) : 0);
    for (idx t = 0; t < shape_size(batch) && n > 0; ++t) {
      const E* src = ptr<E>(ac) + t * n * n;
      if (!all_finite(src, n * n)) linalg_fail("Array must not contain infs or NaNs");
      to_colmajor(src, m.data(), n, n);
      C* wp = ptr<C>(w) + t * n;
      C* vp = vectors ? vc.data() : nullptr;
      int info = 0;
      if constexpr (std::is_floating_point_v<E>) info = routines<R>().geev(n, m.data(), wp, vp);
      else info = routines<R>().cgeev(n, m.data(), wp, vp);
      if (info != 0) linalg_fail("Eigenvalues did not converge");
      if (vectors) from_colmajor(vc.data(), ptr<C>(*v) + t * n * n, n, n);
    }
  });
  if (wct != wdt) w = w.astype(wdt);
  if (v && wct != wdt) v = v->astype(wdt);
  return {w, v};
}

}  // namespace

EigResult eig(const NDArray& a) {
  auto [w, v] = eig_impl(a, true, "eig");
  return {w, *v};
}

NDArray eigvals(const NDArray& a) { return eig_impl(a, false, "eigvals").first; }

SvdResult svd(const NDArray& a, bool full_matrices, bool compute_uv) {
  const bool cplx = is_complex(a.dtype());
  const DType dt = cplx ? a.dtype() : decomp_dtype(a, "svd");
  const DType ct = cplx ? DType::Complex128 : dt;  // NumPy 'D->DdD' (D-041)
  // Real dtypes of the computed and the returned S.
  const DType sct = cplx ? DType::Float64 : dt;
  const DType sdt = dt == DType::Complex64 ? DType::Float32 : (cplx ? DType::Float64 : dt);
  require_2d(a, "svd");
  const Shape& s = a.shape();
  const idx m = s[s.size() - 2];
  const idx n = s[s.size() - 1];
  const idx k = std::min(m, n);
  const Shape batch = batch_of(s, 2);
  const idx nb = shape_size(batch);
  const idx ur = m;
  const idx uc = full_matrices ? m : k;
  const idx vr = full_matrices ? n : k;
  const idx vc = n;
  const NDArray ac = as_compute(a, ct);
  NDArray sv = NDArray::zeros(concat(batch, {k}), sct);
  std::optional<NDArray> u;
  std::optional<NDArray> vh;
  if (compute_uv) {
    u = NDArray::zeros(concat(batch, {ur, uc}), ct);
    vh = NDArray::zeros(concat(batch, {vr, vc}), ct);
  }
  dispatch_solve(ct, [&](auto tag) {
    using T = decltype(tag);
    using R = real_of_t<T>;
    std::vector<T> am(sz(m * n));
    std::vector<T> um(sz(ur * uc));
    std::vector<T> vm(sz(vr * vc));
    for (idx t = 0; t < nb; ++t) {
      if (k == 0) {
        // Empty SVD: U / Vh are identity when full_matrices (NumPy behaviour).
        if (compute_uv && full_matrices) {
          for (idx i = 0; i < m; ++i) ptr<T>(*u)[t * m * m + i * m + i] = T{1};
          for (idx i = 0; i < n; ++i) ptr<T>(*vh)[t * n * n + i * n + i] = T{1};
        }
        continue;
      }
      const T* src = ptr<T>(ac) + t * m * n;
      // NumPy's svd_wrapper rejects non-finite input before LAPACK (D-041).
      if (!all_finite(src, m * n)) linalg_fail("SVD did not converge");
      to_colmajor(src, am.data(), m, n);
      const int info = gesdd_e(m, n, am.data(), ptr<R>(sv) + t * k,
                               compute_uv ? um.data() : nullptr,
                               compute_uv ? vm.data() : nullptr, full_matrices);
      if (info != 0) linalg_fail("SVD did not converge");
      if (compute_uv) {
        from_colmajor(um.data(), ptr<T>(*u) + t * ur * uc, ur, uc);
        from_colmajor(vm.data(), ptr<T>(*vh) + t * vr * vc, vr, vc);
      }
    }
  });
  if (ct != dt) {
    if (u) u = u->astype(dt);
    if (vh) vh = vh->astype(dt);
  }
  if (sct != sdt) sv = sv.astype(sdt);
  return {u, sv, vh};
}

QrResult qr(const NDArray& a, QrMode mode) {
  const bool cplx = is_complex(a.dtype());
  const DType dt = cplx ? a.dtype() : decomp_dtype(a, "qr");
  const DType ct = cplx ? DType::Complex128 : dt;  // NumPy 'D->D' (D-040)
  require_2d(a, "qr");
  const Shape& s = a.shape();
  const idx m = s[s.size() - 2];
  const idx n = s[s.size() - 1];
  const idx k = std::min(m, n);
  const Shape batch = batch_of(s, 2);
  const idx nb = shape_size(batch);
  const idx qc = mode == QrMode::Complete ? m : k;  // Q: m × qc
  const idx rr = mode == QrMode::Complete ? m : k;  // R: rr × n
  const NDArray ac = as_compute(a, ct);
  NDArray r = NDArray::zeros(concat(batch, {rr, n}), ct);
  std::optional<NDArray> q;
  if (mode != QrMode::R) q = NDArray::zeros(concat(batch, {m, qc}), ct);
  dispatch_solve(ct, [&](auto tag) {
    using T = decltype(tag);
    std::vector<T> am(sz(m * n));
    std::vector<T> tau(sz(std::max<idx>(k, 1)));
    std::vector<T> qm(sz(m * std::max(qc, n)));
    for (idx t = 0; t < nb; ++t) {
      T* rp = ptr<T>(r) + t * rr * n;
      if (k == 0) {
        if (q && qc > 0)
          for (idx i = 0; i < qc; ++i) ptr<T>(*q)[t * m * qc + i * qc + i] = T{1};
        continue;
      }
      to_colmajor(ptr<T>(ac) + t * m * n, am.data(), m, n);
      if (geqrf_e(m, n, am.data(), tau.data()) != 0) linalg_fail("QR failed");
      for (idx i = 0; i < std::min(rr, m); ++i)
        for (idx j = i; j < n; ++j) rp[i * n + j] = am[sz(i + j * m)];
      if (q) {
        // Reflectors live in the first k columns of am; copy then expand.
        std::fill(qm.begin(), qm.end(), T{0});
        std::copy(am.begin(), am.begin() + static_cast<std::ptrdiff_t>(m * k), qm.begin());
        if (orgqr_e(m, qc, k, qm.data(), tau.data()) != 0) linalg_fail("QR failed");
        from_colmajor(qm.data(), ptr<T>(*q) + t * m * qc, m, qc);
      }
    }
  });
  if (ct != dt) {
    r = r.astype(dt);
    if (q) q = q->astype(dt);
  }
  return {q, r};
}

LstsqResult lstsq(const NDArray& a, const NDArray& b, double rcond) {
  const DType dt = promote_types(decomp_dtype(a, "lstsq"), decomp_dtype(b, "lstsq"));
  if (a.ndim() != 2) linalg_fail("lstsq: 2-dimensional array given. Array must be two-dimensional");
  if (b.ndim() != 1 && b.ndim() != 2) linalg_fail("lstsq: b must be 1- or 2-dimensional");
  const idx m = a.shape()[0];
  const idx n = a.shape()[1];
  const bool vec = b.ndim() == 1;
  const NDArray b2 = vec ? b.reshape({b.shape()[0], 1}) : b;
  if (b2.shape()[0] != m) linalg_fail("Incompatible dimensions");
  const idx r = b2.shape()[1];
  const idx k = std::min(m, n);
  const NDArray ac = a.astype(dt);
  const NDArray bc = b2.astype(dt);
  NDArray x = NDArray::zeros({n, r}, dt);
  NDArray sv = NDArray::zeros({k}, dt);
  idx rank = 0;
  bool full_rank_overdetermined = false;
  std::vector<double> resid(sz(r), 0.0);
  dispatch_real(dt, [&](auto tag) {
    using T = decltype(tag);
    if (k == 0) return;
    // x = V · diag(1/s) · Uᵀ · b over singular values above the cutoff.
    std::vector<T> am(sz(m * n));
    std::vector<T> um(sz(m * k));
    std::vector<T> vm(sz(k * n));
    to_colmajor(ptr<T>(ac), am.data(), m, n);
    T* s = ptr<T>(sv);
    if (routines<T>().gesdd(m, n, am.data(), s, um.data(), vm.data(), false) != 0) {
      linalg_fail("SVD did not converge in Linear Least Squares");
    }
    const double eps = static_cast<double>(std::numeric_limits<T>::epsilon());
    const double rc = rcond < 0 ? eps * static_cast<double>(std::max(m, n)) : rcond;
    const double cutoff = rc * static_cast<double>(s[0]);
    const T* bp = ptr<T>(bc);
    T* xp = ptr<T>(x);
    for (idx i = 0; i < k; ++i) {
      if (static_cast<double>(s[i]) <= cutoff) continue;
      ++rank;
      for (idx c = 0; c < r; ++c) {
        double proj = 0.0;  // (u_i · b_c)
        for (idx p = 0; p < m; ++p)
          proj += static_cast<double>(um[sz(p + i * m)]) * static_cast<double>(bp[p * r + c]);
        proj /= static_cast<double>(s[i]);
        for (idx j = 0; j < n; ++j)
          xp[j * r + c] = static_cast<T>(static_cast<double>(xp[j * r + c]) +
                                         proj * static_cast<double>(vm[sz(i + j * k)]));
      }
    }
    full_rank_overdetermined = rank == n && m > n;
    if (full_rank_overdetermined) {
      for (idx c = 0; c < r; ++c) {
        double acc = 0.0;
        for (idx p = 0; p < m; ++p) {
          double ax = 0.0;
          for (idx j = 0; j < n; ++j)
            ax += static_cast<double>(ptr<T>(ac)[p * n + j]) * static_cast<double>(xp[j * r + c]);
          const double d = static_cast<double>(bp[p * r + c]) - ax;
          acc += d * d;
        }
        resid[sz(c)] = acc;
      }
    }
  });
  NDArray residuals = NDArray::zeros({full_rank_overdetermined ? r : 0}, dt);
  if (full_rank_overdetermined)
    for (idx c = 0; c < r; ++c) residuals.set_double(c, resid[sz(c)]);
  return {vec ? x.reshape({n}) : x, residuals, rank, sv};
}

namespace {

// Vector p-norm of n doubles with stride.
double vec_norm(const double* x, idx n, idx stride, const NormOrd& ord) {
  const bool dflt = ord.kind == "default";
  const double p = dflt ? 2.0 : ord.p;
  if (std::isinf(p)) {
    if (n == 0) throw_error(ErrorKind::Value, "zero-size array to reduction operation which has no identity");
    double best = std::abs(x[0]);
    for (idx i = 1; i < n; ++i) {
      const double v = std::abs(x[i * stride]);
      best = p > 0 ? std::max(best, v) : std::min(best, v);
      if (std::isnan(v)) best = v;
    }
    return best;
  }
  double acc = 0.0;
  if (p == 0.0) {
    for (idx i = 0; i < n; ++i) acc += x[i * stride] != 0.0 ? 1.0 : 0.0;
    return acc;
  }
  if (p == 2.0) {
    for (idx i = 0; i < n; ++i) acc += x[i * stride] * x[i * stride];
    return std::sqrt(acc);
  }
  if (p == 1.0) {
    for (idx i = 0; i < n; ++i) acc += std::abs(x[i * stride]);
    return acc;
  }
  for (idx i = 0; i < n; ++i) acc += std::pow(std::abs(x[i * stride]), p);
  return std::pow(acc, 1.0 / p);
}

// Matrix norm of row-major m×n doubles.
double mat_norm(const double* x, idx m, idx n, const NormOrd& ord) {
  if (ord.kind == "default" || ord.kind == "fro") return vec_norm(x, m * n, 1, NormOrd{});
  const double p = ord.p;
  if (ord.kind == "nuc" || p == 2.0 || p == -2.0) {
    NDArray a = NDArray::empty({m, n}, DType::Float64);
    std::copy(x, x + m * n, ptr<double>(a));
    const NDArray s = svd(a, false, false).s;
    const double* sp = ptr<double>(s);
    const idx k = s.size();
    if (ord.kind == "nuc") {
      double acc = 0.0;
      for (idx i = 0; i < k; ++i) acc += sp[i];
      return acc;
    }
    if (k == 0) throw_error(ErrorKind::Value, "zero-size array to reduction operation which has no identity");
    return p > 0 ? *std::max_element(sp, sp + k) : *std::min_element(sp, sp + k);
  }
  const bool cols = p == 1.0 || p == -1.0;  // max/min column abs-sum
  if (!cols && !std::isinf(p)) throw_error(ErrorKind::Value, "Invalid norm order for matrices.");
  const idx outer_n = cols ? n : m;
  const idx inner_n = cols ? m : n;
  if (outer_n == 0) throw_error(ErrorKind::Value, "zero-size array to reduction operation which has no identity");
  double best = 0.0;
  for (idx o = 0; o < outer_n; ++o) {
    double acc = 0.0;
    for (idx i = 0; i < inner_n; ++i) acc += std::abs(cols ? x[i * n + o] : x[o * n + i]);
    if (o == 0) best = acc;
    else best = p > 0 ? std::max(best, acc) : std::min(best, acc);
  }
  return best;
}

}  // namespace

NDArray norm(const NDArray& a, const NormOrd& ord,
             const std::optional<std::vector<std::int64_t>>& axis, bool keepdims) {
  reject_complex(a, "norm");
  const DType out_dt =
      (a.dtype() == DType::Float32 || a.dtype() == DType::Float16) ? a.dtype() : DType::Float64;
  const auto nd = static_cast<idx>(a.ndim());
  std::vector<idx> axes;
  if (axis) {
    axes = normalize_axes(*axis, nd);
  } else if (ord.kind == "default") {
    // Flattened 2-norm.
    const NDArray flat = a.astype(DType::Float64);
    NDArray out = NDArray::empty(keepdims ? Shape(sz(nd), 1) : Shape{}, out_dt);
    out.set_double(0, vec_norm(ptr<double>(flat), flat.size(), 1, NormOrd{}));
    return out;
  } else {
    if (nd != 1 && nd != 2) throw_error(ErrorKind::Value, "Improper number of dimensions to norm.");
    for (idx i = 0; i < nd; ++i) axes.push_back(i);
  }
  if (axes.size() != 1 && axes.size() != 2) {
    throw_error(ErrorKind::Value, "Improper number of dimensions to norm.");
  }
  if (axes.size() == 1 && (ord.kind == "fro" || ord.kind == "nuc")) {
    throw_error(ErrorKind::Value, "Invalid norm order '" + ord.kind + "' for vectors");
  }
  // Move reduced axes to the end and make a contiguous float64 copy.
  std::vector<idx> dst;
  for (std::size_t i = 0; i < axes.size(); ++i) dst.push_back(nd - static_cast<idx>(axes.size()) + static_cast<idx>(i));
  const NDArray moved = moveaxis(a, axes, dst).astype(DType::Float64);
  const Shape batch = batch_of(moved.shape(), axes.size());
  Shape out_shape = batch;
  if (keepdims) {
    out_shape = a.shape();
    for (const idx ax : axes) out_shape[sz(ax)] = 1;
  }
  NDArray out = NDArray::empty(out_shape, out_dt);
  const double* src = ptr<double>(moved);
  const idx nb = shape_size(batch);
  if (axes.size() == 1) {
    const idx n = moved.shape().back();
    for (idx t = 0; t < nb; ++t) out.set_double(t, vec_norm(src + t * n, n, 1, ord));
  } else {
    const idx m = moved.shape()[moved.ndim() - 2];
    const idx n = moved.shape().back();
    for (idx t = 0; t < nb; ++t) out.set_double(t, mat_norm(src + t * m * n, m, n, ord));
  }
  return out;
}
}  // namespace nativpy::linalg
