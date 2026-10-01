// Portable C++ linear algebra backend (D-018). Column-major storage.
// Algorithms: partial-pivot LU, cyclic Jacobi (symmetric eigen / one-sided
// SVD), Householder QR, Hessenberg reduction + shifted QR (general eigen).
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <type_traits>
#include <vector>

#include "backend.hpp"

namespace nativpy::linalg {

namespace {

using idx = std::int64_t;

template <typename T>
struct Mat {  // column-major view
  T* p;
  idx ld;
  T& operator()(idx i, idx j) const { return p[i + j * ld]; }
};

// Pivot magnitude: |x| for real, |re| + |im| for complex (LAPACK icamax, D-038).
template <typename T>
auto pivot_mag(const T& x) {
  if constexpr (std::is_floating_point_v<T>) return std::abs(x);
  else return std::abs(x.real()) + std::abs(x.imag());
}

// Real scalar type of a real or complex element type.
template <typename E>
struct real_of {
  using type = E;
};
template <typename R>
struct real_of<std::complex<R>> {
  using type = R;
};
template <typename E>
using real_t = typename real_of<E>::type;

// Conjugate (identity for real) and squared magnitude (x*x for real).
template <typename E>
E cj(const E& x) {
  if constexpr (std::is_floating_point_v<E>) return x;
  else return std::conj(x);
}
template <typename E>
real_t<E> abs2(const E& x) {
  if constexpr (std::is_floating_point_v<E>) return x * x;
  else return std::norm(x);
}

template <typename T>
int lu(idx n, T* a, idx* piv) {
  Mat<T> A{a, n};
  int info = 0;
  for (idx k = 0; k < n; ++k) {
    idx p = k;
    auto best = pivot_mag(A(k, k));
    for (idx i = k + 1; i < n; ++i) {
      if (pivot_mag(A(i, k)) > best) {
        best = pivot_mag(A(i, k));
        p = i;
      }
    }
    piv[k] = p;
    if (A(p, k) == T{0}) {
      if (info == 0) info = static_cast<int>(k + 1);
      continue;
    }
    if (p != k) {
      for (idx j = 0; j < n; ++j) std::swap(A(k, j), A(p, j));
    }
    const T inv = T{1} / A(k, k);
    for (idx i = k + 1; i < n; ++i) A(i, k) *= inv;
    for (idx j = k + 1; j < n; ++j) {
      const T f = A(k, j);
      if (f == T{0}) continue;
      for (idx i = k + 1; i < n; ++i) A(i, j) -= A(i, k) * f;
    }
  }
  return info;
}

template <typename T>
void lu_solve(idx n, idx nrhs, const T* a, const idx* piv, T* b) {
  Mat<const T> A{a, n};
  Mat<T> B{b, n};
  for (idx c = 0; c < nrhs; ++c) {
    for (idx k = 0; k < n; ++k) {
      if (piv[k] != k) std::swap(B(k, c), B(piv[k], c));
    }
    for (idx k = 0; k < n; ++k) {
      for (idx i = k + 1; i < n; ++i) B(i, c) -= A(i, k) * B(k, c);
    }
    for (idx k = n; k-- > 0;) {
      B(k, c) /= A(k, k);
      for (idx i = 0; i < k; ++i) B(i, c) -= A(i, k) * B(k, c);
    }
  }
}

// LU with partial pivoting, then substitution; real or complex E (D-039).
template <typename E>
int lu_gesv(idx n, idx nrhs, E* a, E* b) {
  std::vector<idx> piv(static_cast<std::size_t>(std::max<idx>(n, 1)));
  const int info = lu(n, a, piv.data());
  if (info != 0) return info;
  lu_solve(n, nrhs, a, piv.data(), b);
  return 0;
}

// Cyclic Jacobi on Hermitian (real: symmetric) n×n; the lower triangle is
// mirrored first and the diagonal's imaginary part ignored, as LAPACK does.
// Complex pairs are phase-aligned so a_pq is real, then rotated (D-042).
template <typename E>
int jacobi_eigh(idx n, E* a, real_t<E>* w, bool vectors) {
  using T = real_t<E>;
  Mat<E> A{a, n};
  for (idx j = 0; j < n; ++j) {
    if constexpr (!std::is_floating_point_v<E>) A(j, j) = E(A(j, j).real(), T{0});
    for (idx i = 0; i < j; ++i) A(i, j) = cj(A(j, i));
  }
  std::vector<E> vbuf(static_cast<std::size_t>(n * n), E{0});
  Mat<E> V{vbuf.data(), n};
  for (idx i = 0; i < n; ++i) V(i, i) = E{1};
  const T eps = std::numeric_limits<T>::epsilon();
  bool converged = n <= 1;
  for (int sweep = 0; sweep < 100 && !converged; ++sweep) {
    T off = 0;
    T diag = 0;
    for (idx j = 0; j < n; ++j) {
      diag += abs2(A(j, j));
      for (idx i = 0; i < j; ++i) off += abs2(A(i, j));
    }
    if (off <= eps * eps * diag || off == T{0}) {
      converged = true;
      break;
    }
    for (idx p = 0; p < n - 1; ++p) {
      for (idx q = p + 1; q < n; ++q) {
        if (A(p, q) == E{0}) continue;
        if constexpr (!std::is_floating_point_v<E>) {
          // D = diag(1, .., d, ..) at q with d = conj(a_pq)/|a_pq|: A <- Dᴴ A D, V <- V D.
          const T mag = std::abs(A(p, q));
          const E d = std::conj(A(p, q)) / mag;
          for (idx k = 0; k < n; ++k) A(k, q) *= d;
          for (idx k = 0; k < n; ++k) A(q, k) *= std::conj(d);
          A(p, q) = E(mag, T{0});
          A(q, p) = E(mag, T{0});
          A(q, q) = E(A(q, q).real(), T{0});
          for (idx k = 0; k < n; ++k) V(k, q) *= d;
        }
        const T apq = std::real(A(p, q));
        const T theta = (std::real(A(q, q)) - std::real(A(p, p))) / (T{2} * apq);
        const T t = (theta >= 0 ? T{1} : T{-1}) /
                    (std::abs(theta) + std::sqrt(theta * theta + T{1}));
        const T c = T{1} / std::sqrt(t * t + T{1});
        const T s = t * c;
        for (idx k = 0; k < n; ++k) {
          const E akp = A(k, p);
          const E akq = A(k, q);
          A(k, p) = c * akp - s * akq;
          A(k, q) = s * akp + c * akq;
        }
        for (idx k = 0; k < n; ++k) {
          const E apk = A(p, k);
          const E aqk = A(q, k);
          A(p, k) = c * apk - s * aqk;
          A(q, k) = s * apk + c * aqk;
        }
        for (idx k = 0; k < n; ++k) {
          const E vkp = V(k, p);
          const E vkq = V(k, q);
          V(k, p) = c * vkp - s * vkq;
          V(k, q) = s * vkp + c * vkq;
        }
      }
    }
  }
  if (!converged) return 1;
  std::vector<idx> order(static_cast<std::size_t>(n));
  for (idx i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::sort(order.begin(), order.end(),
            [&](idx x, idx y) { return std::real(A(x, x)) < std::real(A(y, y)); });
  for (idx c = 0; c < n; ++c) {
    const idx src = order[static_cast<std::size_t>(c)];
    w[c] = std::real(A(src, src));
  }
  if (!vectors) return 0;
  for (idx c = 0; c < n; ++c) {
    const idx src = order[static_cast<std::size_t>(c)];
    for (idx r = 0; r < n; ++r) A(r, c) = V(r, src);
  }
  return 0;
}

// Householder QR (LAPACK geqrf layout: v(0)=1 implicit, stored below diag).
template <typename T>
void householder_qr(idx m, idx n, T* a, T* tau) {
  Mat<T> A{a, m};
  const idx k = std::min(m, n);
  for (idx j = 0; j < k; ++j) {
    T norm2 = 0;
    for (idx i = j + 1; i < m; ++i) norm2 += A(i, j) * A(i, j);
    const T alpha = A(j, j);
    if (norm2 == T{0}) {
      tau[j] = T{0};
      continue;
    }
    const T beta = -std::copysign(std::sqrt(alpha * alpha + norm2), alpha);
    tau[j] = (beta - alpha) / beta;
    const T scale = T{1} / (alpha - beta);
    for (idx i = j + 1; i < m; ++i) A(i, j) *= scale;
    A(j, j) = beta;
    for (idx c = j + 1; c < n; ++c) {
      T dot = A(j, c);
      for (idx i = j + 1; i < m; ++i) dot += A(i, j) * A(i, c);
      dot *= tau[j];
      A(j, c) -= dot;
      for (idx i = j + 1; i < m; ++i) A(i, c) -= dot * A(i, j);
    }
  }
}

// Q[:, :cols] = H_0 H_1 ... H_{k-1} I[:, :cols]; reflectors read from q.
template <typename T>
void form_q(idx m, idx cols, idx k, T* q, const T* tau) {
  Mat<T> Q{q, m};
  std::vector<T> refl(static_cast<std::size_t>(m * k));
  Mat<T> R{refl.data(), m};
  for (idx j = 0; j < k; ++j) {
    for (idx i = 0; i < m; ++i) R(i, j) = Q(i, j);
  }
  for (idx j = 0; j < cols; ++j) {
    for (idx i = 0; i < m; ++i) Q(i, j) = (i == j) ? T{1} : T{0};
  }
  for (idx j = k; j-- > 0;) {
    if (tau[j] == T{0}) continue;
    for (idx c = 0; c < cols; ++c) {
      T dot = Q(j, c);
      for (idx i = j + 1; i < m; ++i) dot += R(i, j) * Q(i, c);
      dot *= tau[j];
      Q(j, c) -= dot;
      for (idx i = j + 1; i < m; ++i) Q(i, c) -= dot * R(i, j);
    }
  }
}

// One-sided Jacobi SVD of m×n A with m >= n. On return a holds U (m×n),
// s singular values (descending), v (n×n) right singular vectors (columns).
// E real or complex; complex pairs are phase-aligned so γ = u_pᴴu_q is real
// before the real rotation (D-041). The real path is unchanged.
template <typename E>
int jacobi_svd_tall(idx m, idx n, E* a, real_t<E>* s, E* v) {
  using T = real_t<E>;
  Mat<E> U{a, m};
  Mat<E> V{v, n};
  for (idx j = 0; j < n; ++j) {
    for (idx i = 0; i < n; ++i) V(i, j) = (i == j) ? E{1} : E{0};
  }
  const T eps = std::numeric_limits<T>::epsilon();
  bool converged = n <= 1;
  for (int sweep = 0; sweep < 100 && !converged; ++sweep) {
    converged = true;
    for (idx p = 0; p < n - 1; ++p) {
      for (idx q = p + 1; q < n; ++q) {
        T alpha = 0;
        T beta = 0;
        E gamma = 0;
        for (idx i = 0; i < m; ++i) {
          alpha += abs2(U(i, p));
          beta += abs2(U(i, q));
          gamma += cj(U(i, p)) * U(i, q);
        }
        T g;
        if constexpr (std::is_floating_point_v<E>) g = gamma;
        else g = std::abs(gamma);
        if (g == T{0} || std::abs(g) <= eps * std::sqrt(alpha * beta)) continue;
        converged = false;
        if constexpr (!std::is_floating_point_v<E>) {
          const E ph = std::conj(gamma) / g;  // makes column q's γ real, = g
          for (idx i = 0; i < m; ++i) U(i, q) *= ph;
          for (idx i = 0; i < n; ++i) V(i, q) *= ph;
        }
        const T zeta = (beta - alpha) / (T{2} * g);
        const T t = (zeta >= 0 ? T{1} : T{-1}) /
                    (std::abs(zeta) + std::sqrt(T{1} + zeta * zeta));
        const T c = T{1} / std::sqrt(T{1} + t * t);
        const T sn = c * t;
        for (idx i = 0; i < m; ++i) {
          const E up = U(i, p);
          const E uq = U(i, q);
          U(i, p) = c * up - sn * uq;
          U(i, q) = sn * up + c * uq;
        }
        for (idx i = 0; i < n; ++i) {
          const E vp = V(i, p);
          const E vq = V(i, q);
          V(i, p) = c * vp - sn * vq;
          V(i, q) = sn * vp + c * vq;
        }
      }
    }
  }
  if (!converged) return 1;
  for (idx j = 0; j < n; ++j) {
    T nrm = 0;
    for (idx i = 0; i < m; ++i) nrm += abs2(U(i, j));
    s[j] = std::sqrt(nrm);
  }
  // Sort descending, permuting U and V columns.
  std::vector<idx> order(static_cast<std::size_t>(n));
  for (idx i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
  std::stable_sort(order.begin(), order.end(), [&](idx x, idx y) { return s[x] > s[y]; });
  std::vector<E> ucopy(a, a + m * n);
  std::vector<E> vcopy(v, v + n * n);
  std::vector<T> scopy(s, s + n);
  for (idx c = 0; c < n; ++c) {
    const idx src = order[static_cast<std::size_t>(c)];
    s[c] = scopy[static_cast<std::size_t>(src)];
    for (idx i = 0; i < m; ++i) U(i, c) = ucopy[static_cast<std::size_t>(i + src * m)];
    for (idx i = 0; i < n; ++i) V(i, c) = vcopy[static_cast<std::size_t>(i + src * n)];
  }
  return 0;
}

// General eigenproblem via complex arithmetic: Hessenberg reduction, then
// single-shift (Wilkinson) complex QR to Schur form T = Z^H A Z, then
// eigenvectors of T by back-substitution, transformed by Z and normalised.
// E is real or complex (D-043); vout == nullptr computes values only.
template <typename E>
int complex_eig(idx n, const E* a, std::complex<real_t<E>>* w,
                std::complex<real_t<E>>* vout) {
  using T = real_t<E>;
  using C = std::complex<T>;
  std::vector<C> hb(static_cast<std::size_t>(n * n));
  std::vector<C> zb(static_cast<std::size_t>(n * n), C{0});
  Mat<C> H{hb.data(), n};
  Mat<C> Z{zb.data(), n};
  for (idx j = 0; j < n; ++j) {
    for (idx i = 0; i < n; ++i) H(i, j) = C(a[i + j * n]);
    Z(j, j) = C{1};
  }
  const T eps = std::numeric_limits<T>::epsilon();
  // Givens-based QR iterations on the full matrix (O(n^3) per sweep; fine
  // for the fallback path, Accelerate handles performance on macOS).
  auto rotate = [&](idx p, C cc, C ss, idx lo, idx hi) {
    // Left: rows p, p+1 of H over columns [lo, n); right: columns p,p+1 of H rows [0, hi].
    for (idx j = lo; j < n; ++j) {
      const C x = H(p, j);
      const C y = H(p + 1, j);
      H(p, j) = std::conj(cc) * x + std::conj(ss) * y;
      H(p + 1, j) = -ss * x + cc * y;
    }
    for (idx i = 0; i <= hi; ++i) {
      const C x = H(i, p);
      const C y = H(i, p + 1);
      H(i, p) = x * cc + y * ss;
      H(i, p + 1) = -x * std::conj(ss) + y * std::conj(cc);
    }
    for (idx i = 0; i < n; ++i) {
      const C x = Z(i, p);
      const C y = Z(i, p + 1);
      Z(i, p) = x * cc + y * ss;
      Z(i, p + 1) = -x * std::conj(ss) + y * std::conj(cc);
    }
  };
  auto givens = [](C x, C y, C& cc, C& ss) {
    const T r = std::sqrt(std::norm(x) + std::norm(y));
    if (r == T{0}) {
      cc = C{1};
      ss = C{0};
      return;
    }
    cc = x / r;
    ss = y / r;
  };
  // Reduce to Hessenberg with Givens rotations.
  for (idx j = 0; j + 2 < n; ++j) {
    for (idx i = n - 1; i > j + 1; --i) {
      C cc;
      C ss;
      givens(H(i - 1, j), H(i, j), cc, ss);
      rotate(i - 1, cc, ss, 0, n - 1);
      H(i, j) = C{0};
    }
  }
  idx hi = n - 1;
  int iter = 0;
  while (hi > 0) {
    idx lo = hi;
    while (lo > 0) {
      const T sub = std::abs(H(lo, lo - 1));
      if (sub <= eps * (std::abs(H(lo, lo)) + std::abs(H(lo - 1, lo - 1))) ||
          sub < std::numeric_limits<T>::min()) {
        H(lo, lo - 1) = C{0};
        break;
      }
      --lo;
    }
    if (lo == hi) {
      --hi;
      iter = 0;
      continue;
    }
    if (++iter > 60 * static_cast<int>(n)) return 1;
    // Wilkinson shift from trailing 2×2 (exceptional shift every 11 iters).
    const C a11 = H(hi - 1, hi - 1);
    const C a12 = H(hi - 1, hi);
    const C a21 = H(hi, hi - 1);
    const C a22 = H(hi, hi);
    C mu;
    if (iter % 11 == 0) {
      mu = a22 + C(std::abs(a21), 0);
    } else {
      const C tr = a11 + a22;
      const C det = a11 * a22 - a12 * a21;
      const C disc = std::sqrt(tr * tr / T{4} - det);
      const C l1 = tr / T{2} + disc;
      const C l2 = tr / T{2} - disc;
      mu = std::abs(l1 - a22) < std::abs(l2 - a22) ? l1 : l2;
    }
    // Implicit single-shift QR sweep on active block [lo, hi].
    C cc;
    C ss;
    givens(H(lo, lo) - mu, H(lo + 1, lo), cc, ss);
    for (idx k = lo; k < hi; ++k) {
      if (k > lo) {
        givens(H(k, k - 1), H(k + 1, k - 1), cc, ss);
      }
      rotate(k, cc, ss, 0, std::min(k + 2, n - 1));
      if (k > lo) H(k + 1, k - 1) = C{0};
    }
  }
  for (idx i = 0; i < n; ++i) w[i] = H(i, i);
  if (vout == nullptr) return 0;
  // Eigenvectors of upper-triangular H: solve (H - λ_k I) x = 0, x_k = 1.
  Mat<C> V{vout, n};
  std::vector<C> x(static_cast<std::size_t>(n));
  T hnorm = 0;
  for (idx j = 0; j < n; ++j) {
    for (idx i = 0; i <= j; ++i) hnorm = std::max(hnorm, std::abs(H(i, j)));
  }
  const T small = std::max(hnorm * eps, std::numeric_limits<T>::min());
  for (idx k = 0; k < n; ++k) {
    std::fill(x.begin(), x.end(), C{0});
    x[static_cast<std::size_t>(k)] = C{1};
    for (idx i = k; i-- > 0;) {
      C sum{0};
      for (idx j = i + 1; j <= k; ++j) sum += H(i, j) * x[static_cast<std::size_t>(j)];
      C d = H(i, i) - w[k];
      if (std::abs(d) < small) d = C(small, 0);
      x[static_cast<std::size_t>(i)] = -sum / d;
    }
    T nrm = 0;
    for (idx i = 0; i < n; ++i) {
      C s{0};
      for (idx j = 0; j <= k; ++j) s += Z(i, j) * x[static_cast<std::size_t>(j)];
      V(i, k) = s;
      nrm += std::norm(s);
    }
    nrm = std::sqrt(nrm);
    for (idx i = 0; i < n; ++i) V(i, k) /= nrm;
  }
  return 0;
}

// Extends the first `k` orthonormal columns of Q (m×m, ld m) to a full
// orthonormal basis by Gram-Schmidt against unit vectors.
// Complex Householder QR after LAPACK zlarfg/zgeqr2 (D-040): beta is real,
// tau = ((beta - Re a)/beta, -Im a/beta), v = x / (alpha - beta). The
// trailing columns get H^H = I - conj(tau) v v^H.
template <typename T>
void complex_householder_qr(idx m, idx n, std::complex<T>* a, std::complex<T>* tau) {
  using C = std::complex<T>;
  Mat<C> A{a, m};
  const idx k = std::min(m, n);
  for (idx j = 0; j < k; ++j) {
    T norm2 = 0;
    for (idx i = j + 1; i < m; ++i) norm2 += std::norm(A(i, j));
    const C alpha = A(j, j);
    if (norm2 == T{0} && alpha.imag() == T{0}) {
      tau[j] = C{0};
      continue;
    }
    const T beta = -std::copysign(std::hypot(alpha.real(), alpha.imag(), std::sqrt(norm2)),
                                  alpha.real());
    tau[j] = C((beta - alpha.real()) / beta, -alpha.imag() / beta);
    const C scale = C{1} / (alpha - beta);
    for (idx i = j + 1; i < m; ++i) A(i, j) *= scale;
    A(j, j) = beta;
    const C ct = std::conj(tau[j]);
    for (idx c = j + 1; c < n; ++c) {
      C dot = A(j, c);  // v_j = 1
      for (idx i = j + 1; i < m; ++i) dot += std::conj(A(i, j)) * A(i, c);
      dot *= ct;
      A(j, c) -= dot;
      for (idx i = j + 1; i < m; ++i) A(i, c) -= dot * A(i, j);
    }
  }
}

// Q = H(0)…H(k-1) applied to the first `cols` identity columns, with
// H(j) = I - tau_j v_j v_j^H (zung2r).
template <typename T>
void complex_form_q(idx m, idx cols, idx k, std::complex<T>* q, const std::complex<T>* tau) {
  using C = std::complex<T>;
  Mat<C> Q{q, m};
  std::vector<C> refl(static_cast<std::size_t>(m * k));
  Mat<C> R{refl.data(), m};
  for (idx j = 0; j < k; ++j) {
    for (idx i = 0; i < m; ++i) R(i, j) = Q(i, j);
  }
  for (idx j = 0; j < cols; ++j) {
    for (idx i = 0; i < m; ++i) Q(i, j) = (i == j) ? C{1} : C{0};
  }
  for (idx j = k; j-- > 0;) {
    if (tau[j] == C{0}) continue;
    for (idx c = 0; c < cols; ++c) {
      C dot = Q(j, c);
      for (idx i = j + 1; i < m; ++i) dot += std::conj(R(i, j)) * Q(i, c);
      dot *= tau[j];
      Q(j, c) -= dot;
      for (idx i = j + 1; i < m; ++i) Q(i, c) -= dot * R(i, j);
    }
  }
}

template <typename E>
void complete_basis(idx m, idx k, E* q) {
  using T = real_t<E>;
  Mat<E> Q{q, m};
  idx col = k;
  for (idx e = 0; e < m && col < m; ++e) {
    std::vector<E> v(static_cast<std::size_t>(m), E{0});
    v[static_cast<std::size_t>(e)] = E{1};
    for (int pass = 0; pass < 2; ++pass) {
      for (idx j = 0; j < col; ++j) {
        E d = 0;
        for (idx i = 0; i < m; ++i) d += cj(Q(i, j)) * v[static_cast<std::size_t>(i)];
        for (idx i = 0; i < m; ++i) v[static_cast<std::size_t>(i)] -= d * Q(i, j);
      }
    }
    T nrm = 0;
    for (const E& x : v) nrm += abs2(x);
    nrm = std::sqrt(nrm);
    if (nrm < T{0.5}) continue;
    for (idx i = 0; i < m; ++i) Q(i, col) = v[static_cast<std::size_t>(i)] / nrm;
    ++col;
  }
}

// Zero-norm U columns (rank deficiency) are replaced by an orthonormal
// completion so U stays orthonormal.
template <typename E>
void fix_null_columns(idx m, idx k, E* u, const real_t<E>* s) {
  using T = real_t<E>;
  idx good = 0;
  Mat<E> U{u, m};
  const T tol = std::numeric_limits<T>::epsilon() * (k > 0 ? s[0] : T{0}) * static_cast<T>(m);
  while (good < k && s[good] > tol && s[good] > T{0}) ++good;
  if (good == k) return;
  std::vector<E> full(static_cast<std::size_t>(m * m), E{0});
  Mat<E> F{full.data(), m};
  for (idx j = 0; j < good; ++j) {
    for (idx i = 0; i < m; ++i) F(i, j) = U(i, j);
  }
  complete_basis(m, good, full.data());
  for (idx j = good; j < k; ++j) {
    for (idx i = 0; i < m; ++i) U(i, j) = F(i, j);
  }
}

// E real or complex. Wide A works on B = Aᴴ: A = V_B Σ U_Bᴴ (D-041).
template <typename E>
int svd(idx m, idx n, E* a, real_t<E>* s, E* u, E* vt, bool full) {
  using T = real_t<E>;
  const idx k = std::min(m, n);
  const bool tall = m >= n;
  // Work on B = A (m×n) if tall, else B = A^H (n×m); B is r×c with r >= c.
  const idx r = tall ? m : n;
  const idx c = tall ? n : m;
  std::vector<E> b(static_cast<std::size_t>(r * c));
  for (idx j = 0; j < c; ++j) {
    for (idx i = 0; i < r; ++i) {
      b[static_cast<std::size_t>(i + j * r)] = tall ? a[i + j * m] : cj(a[j + i * m]);
    }
  }
  std::vector<E> v(static_cast<std::size_t>(c * c));
  if (jacobi_svd_tall(r, c, b.data(), s, v.data()) != 0) return 1;
  if (u == nullptr) return 0;
  // Left vectors of B: normalised columns of b.
  for (idx j = 0; j < c; ++j) {
    const T sj = s[j];
    for (idx i = 0; i < r; ++i) {
      b[static_cast<std::size_t>(i + j * r)] = sj > T{0} ? b[static_cast<std::size_t>(i + j * r)] / sj : E{0};
    }
  }
  fix_null_columns(r, c, b.data(), s);
  // Map to A's U (m×ucols) and Vt (vrows×n).
  const idx ucols = full ? m : k;
  const idx vrows = full ? n : k;
  std::vector<E> left(static_cast<std::size_t>(m * m), E{0});   // U candidates, ld m
  std::vector<E> right(static_cast<std::size_t>(n * n), E{0});  // V candidates, ld n
  for (idx j = 0; j < k; ++j) {
    for (idx i = 0; i < m; ++i) {
      left[static_cast<std::size_t>(i + j * m)] =
          tall ? b[static_cast<std::size_t>(i + j * r)] : v[static_cast<std::size_t>(i + j * c)];
    }
    for (idx i = 0; i < n; ++i) {
      right[static_cast<std::size_t>(i + j * n)] =
          tall ? v[static_cast<std::size_t>(i + j * c)] : b[static_cast<std::size_t>(i + j * r)];
    }
  }
  if (full) {
    complete_basis(m, k, left.data());
    complete_basis(n, k, right.data());
  }
  for (idx j = 0; j < ucols; ++j) {
    for (idx i = 0; i < m; ++i) u[i + j * m] = left[static_cast<std::size_t>(i + j * m)];
  }
  // Vh (vrows×n) column-major: Vh(i, j) = conj(V(j, i)).
  for (idx j = 0; j < n; ++j) {
    for (idx i = 0; i < vrows; ++i) vt[i + j * vrows] = cj(right[static_cast<std::size_t>(j + i * n)]);
  }
  return 0;
}

template <typename T>
class FallbackRoutines final : public Routines<T> {
 public:
  void gemm(idx m, idx n, idx k, const T* a, const T* b, T* c) const override {
    std::fill(c, c + m * n, T{0});
    for (idx i = 0; i < m; ++i) {
      T* crow = c + i * n;
      for (idx p = 0; p < k; ++p) {
        const T av = a[i * k + p];
        const T* brow = b + p * n;
        for (idx j = 0; j < n; ++j) crow[j] += av * brow[j];
      }
    }
  }
  void cgemm(idx m, idx n, idx k, const std::complex<T>* a, const std::complex<T>* b,
             std::complex<T>* c) const override {
    noblas_cgemm(m, n, k, a, b, c);
  }
  int getrf(idx n, T* a, idx* piv) const override { return lu(n, a, piv); }
  int cgetrf(idx n, std::complex<T>* a, idx* piv) const override { return lu(n, a, piv); }
  int gesv(idx n, idx nrhs, T* a, T* b) const override { return lu_gesv(n, nrhs, a, b); }
  int cgesv(idx n, idx nrhs, std::complex<T>* a, std::complex<T>* b) const override {
    return lu_gesv(n, nrhs, a, b);
  }
  int syevd(idx n, T* a, T* w, bool vectors) const override {
    return jacobi_eigh(n, a, w, vectors);
  }
  int cheevd(idx n, std::complex<T>* a, T* w, bool vectors) const override {
    return jacobi_eigh(n, a, w, vectors);
  }
  int geev(idx n, T* a, std::complex<T>* w, std::complex<T>* v) const override {
    return complex_eig(n, a, w, v);
  }
  int cgeev(idx n, std::complex<T>* a, std::complex<T>* w,
            std::complex<T>* v) const override {
    return complex_eig(n, a, w, v);
  }
  int gesdd(idx m, idx n, T* a, T* s, T* u, T* vt, bool full) const override {
    return svd(m, n, a, s, u, vt, full);
  }
  int cgesdd(idx m, idx n, std::complex<T>* a, T* s, std::complex<T>* u, std::complex<T>* vt,
             bool full) const override {
    return svd(m, n, a, s, u, vt, full);
  }
  int geqrf(idx m, idx n, T* a, T* tau) const override {
    householder_qr(m, n, a, tau);
    return 0;
  }
  int orgqr(idx m, idx cols, idx k, T* q, const T* tau) const override {
    form_q(m, cols, k, q, tau);
    return 0;
  }
  int cgeqrf(idx m, idx n, std::complex<T>* a, std::complex<T>* tau) const override {
    complex_householder_qr(m, n, a, tau);
    return 0;
  }
  int cungqr(idx m, idx cols, idx k, std::complex<T>* q,
             const std::complex<T>* tau) const override {
    complex_form_q(m, cols, k, q, tau);
    return 0;
  }
};

class FallbackBackend final : public Backend {
 public:
  [[nodiscard]] std::string_view name() const noexcept override { return "fallback"; }
  [[nodiscard]] const Routines<float>& f32() const noexcept override { return f32_; }
  [[nodiscard]] const Routines<double>& f64() const noexcept override { return f64_; }

 private:
  FallbackRoutines<float> f32_;
  FallbackRoutines<double> f64_;
};

}  // namespace

const Backend& fallback_backend() {
  static const FallbackBackend backend;
  return backend;
}

}  // namespace nativpy::linalg
