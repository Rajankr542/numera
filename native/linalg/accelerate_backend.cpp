// Apple Accelerate BLAS/LAPACK backend (D-018). Compiled only when
// NATIVPY_HAVE_ACCELERATE is defined (macOS). Uses the new LAPACK interface.
#include "backend.hpp"

#if defined(NATIVPY_HAVE_ACCELERATE)

#include <Accelerate/Accelerate.h>

#include <algorithm>
#include <climits>
#include <complex>
#include <vector>

#include "error.hpp"

namespace nativpy::linalg {

namespace {

using idx = std::int64_t;
using lint = __LAPACK_int;

lint li(idx v) {
  if (v > static_cast<idx>(INT_MAX)) {
    throw_error(ErrorKind::Value, "matrix dimension too large for the LAPACK backend");
  }
  return static_cast<lint>(v);
}

// Type-overloaded thin wrappers over the s/d LAPACK routines.
void gemm_(lint m, lint n, lint k, const float* a, const float* b, float* c) {
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}
void gemm_(lint m, lint n, lint k, const double* a, const double* b, double* c) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, k, b, n, 0.0, c, n);
}
void getrf_(const lint* m, const lint* n, float* a, const lint* lda, lint* piv, lint* info) {
  sgetrf_(m, n, a, lda, piv, info);
}
void getrf_(const lint* m, const lint* n, double* a, const lint* lda, lint* piv, lint* info) {
  dgetrf_(m, n, a, lda, piv, info);
}
void gesv_(const lint* n, const lint* r, float* a, const lint* lda, lint* p, float* b,
           const lint* ldb, lint* info) {
  sgesv_(n, r, a, lda, p, b, ldb, info);
}
void gesv_(const lint* n, const lint* r, double* a, const lint* lda, lint* p, double* b,
           const lint* ldb, lint* info) {
  dgesv_(n, r, a, lda, p, b, ldb, info);
}
void syevd_(const lint* n, float* a, float* w, float* work, const lint* lw, lint* iw,
            const lint* liw, lint* info) {
  ssyevd_("V", "L", n, a, n, w, work, lw, iw, liw, info);
}
void syevd_(const lint* n, double* a, double* w, double* work, const lint* lw, lint* iw,
            const lint* liw, lint* info) {
  dsyevd_("V", "L", n, a, n, w, work, lw, iw, liw, info);
}
void geev_(const lint* n, float* a, float* wr, float* wi, float* vr, float* work,
           const lint* lw, lint* info) {
  const lint one = 1;
  sgeev_("N", "V", n, a, n, wr, wi, nullptr, &one, vr, n, work, lw, info);
}
void geev_(const lint* n, double* a, double* wr, double* wi, double* vr, double* work,
           const lint* lw, lint* info) {
  const lint one = 1;
  dgeev_("N", "V", n, a, n, wr, wi, nullptr, &one, vr, n, work, lw, info);
}
void gesdd_(const char* job, const lint* m, const lint* n, float* a, float* s, float* u,
            const lint* ldu, float* vt, const lint* ldvt, float* work, const lint* lw,
            lint* iw, lint* info) {
  sgesdd_(job, m, n, a, m, s, u, ldu, vt, ldvt, work, lw, iw, info);
}
void gesdd_(const char* job, const lint* m, const lint* n, double* a, double* s, double* u,
            const lint* ldu, double* vt, const lint* ldvt, double* work, const lint* lw,
            lint* iw, lint* info) {
  dgesdd_(job, m, n, a, m, s, u, ldu, vt, ldvt, work, lw, iw, info);
}
void geqrf_(const lint* m, const lint* n, float* a, float* tau, float* w, const lint* lw,
            lint* info) {
  sgeqrf_(m, n, a, m, tau, w, lw, info);
}
void geqrf_(const lint* m, const lint* n, double* a, double* tau, double* w, const lint* lw,
            lint* info) {
  dgeqrf_(m, n, a, m, tau, w, lw, info);
}
void orgqr_(const lint* m, const lint* n, const lint* k, float* a, const float* tau, float* w,
            const lint* lw, lint* info) {
  sorgqr_(m, n, k, a, m, tau, w, lw, info);
}
void orgqr_(const lint* m, const lint* n, const lint* k, double* a, const double* tau,
            double* w, const lint* lw, lint* info) {
  dorgqr_(m, n, k, a, m, tau, w, lw, info);
}

template <typename T>
lint lwork_from(T q) {
  return std::max<lint>(1, static_cast<lint>(q) + 1);
}

template <typename T>
class AccelRoutines final : public Routines<T> {
 public:
  void gemm(idx m, idx n, idx k, const T* a, const T* b, T* c) const override {
    if (k == 0) {
      std::fill(c, c + m * n, T{0});
      return;
    }
    gemm_(li(m), li(n), li(k), a, b, c);
  }
  int getrf(idx n, T* a, idx* piv) const override {
    const lint nn = li(n);
    std::vector<lint> p(static_cast<std::size_t>(std::max<idx>(n, 1)));
    lint info = 0;
    getrf_(&nn, &nn, a, &nn, p.data(), &info);
    for (idx i = 0; i < n; ++i) piv[i] = p[static_cast<std::size_t>(i)] - 1;
    return static_cast<int>(info);
  }
  int gesv(idx n, idx nrhs, T* a, T* b) const override {
    const lint nn = li(n);
    const lint r = li(nrhs);
    std::vector<lint> p(static_cast<std::size_t>(std::max<idx>(n, 1)));
    lint info = 0;
    gesv_(&nn, &r, a, &nn, p.data(), b, &nn, &info);
    return static_cast<int>(info);
  }
  int syevd(idx n, T* a, T* w) const override {
    const lint nn = li(n);
    lint info = 0;
    T wq{};
    lint iwq = 0;
    const lint q = -1;
    syevd_(&nn, a, w, &wq, &q, &iwq, &q, &info);
    if (info != 0) return static_cast<int>(info);
    const lint lw = lwork_from(wq);
    const lint liw = std::max<lint>(1, iwq);
    std::vector<T> work(static_cast<std::size_t>(lw));
    std::vector<lint> iw(static_cast<std::size_t>(liw));
    syevd_(&nn, a, w, work.data(), &lw, iw.data(), &liw, &info);
    return static_cast<int>(info);
  }
  int geev(idx n, T* a, std::complex<T>* w, std::complex<T>* v) const override {
    const lint nn = li(n);
    const auto sz = static_cast<std::size_t>(n);
    std::vector<T> wr(sz);
    std::vector<T> wi(sz);
    std::vector<T> vr(sz * sz);
    lint info = 0;
    T wq{};
    const lint q = -1;
    geev_(&nn, a, wr.data(), wi.data(), vr.data(), &wq, &q, &info);
    if (info != 0) return static_cast<int>(info);
    const lint lw = lwork_from(wq);
    std::vector<T> work(static_cast<std::size_t>(lw));
    geev_(&nn, a, wr.data(), wi.data(), vr.data(), work.data(), &lw, &info);
    if (info != 0) return static_cast<int>(info);
    // Unpack LAPACK's real storage of complex-conjugate eigenvector pairs.
    for (idx j = 0; j < n; ++j) {
      const auto js = static_cast<std::size_t>(j);
      w[j] = std::complex<T>(wr[js], wi[js]);
      if (wi[js] == T{0}) {
        for (idx i = 0; i < n; ++i) v[i + j * n] = vr[static_cast<std::size_t>(i + j * n)];
      } else if (j + 1 < n) {
        for (idx i = 0; i < n; ++i) {
          const T re = vr[static_cast<std::size_t>(i + j * n)];
          const T im = vr[static_cast<std::size_t>(i + (j + 1) * n)];
          v[i + j * n] = std::complex<T>(re, im);
          v[i + (j + 1) * n] = std::complex<T>(re, -im);
        }
        w[j + 1] = std::complex<T>(wr[js + 1], wi[js + 1]);
        ++j;
      }
    }
    return 0;
  }
  int gesdd(idx m, idx n, T* a, T* s, T* u, T* vt, bool full) const override {
    const lint mm = li(m);
    const lint nn = li(n);
    const idx k = std::min(m, n);
    const char* job = u == nullptr ? "N" : (full ? "A" : "S");
    const lint ldu = std::max<lint>(1, mm);
    const lint ldvt = std::max<lint>(1, li(full ? n : k));
    std::vector<lint> iw(static_cast<std::size_t>(std::max<idx>(8 * k, 1)));
    T dummy{};
    T* up = u != nullptr ? u : &dummy;
    T* vp = vt != nullptr ? vt : &dummy;
    lint info = 0;
    T wq{};
    const lint q = -1;
    gesdd_(job, &mm, &nn, a, s, up, &ldu, vp, &ldvt, &wq, &q, iw.data(), &info);
    if (info != 0) return static_cast<int>(info);
    const lint lw = lwork_from(wq);
    std::vector<T> work(static_cast<std::size_t>(lw));
    gesdd_(job, &mm, &nn, a, s, up, &ldu, vp, &ldvt, work.data(), &lw, iw.data(), &info);
    return static_cast<int>(info);
  }
  int geqrf(idx m, idx n, T* a, T* tau) const override {
    const lint mm = li(m);
    const lint nn = li(n);
    lint info = 0;
    T wq{};
    const lint q = -1;
    geqrf_(&mm, &nn, a, tau, &wq, &q, &info);
    if (info != 0) return static_cast<int>(info);
    const lint lw = lwork_from(wq);
    std::vector<T> work(static_cast<std::size_t>(lw));
    geqrf_(&mm, &nn, a, tau, work.data(), &lw, &info);
    return static_cast<int>(info);
  }
  int orgqr(idx m, idx cols, idx k, T* q, const T* tau) const override {
    const lint mm = li(m);
    const lint cc = li(cols);
    const lint kk = li(k);
    lint info = 0;
    T wq{};
    const lint qq = -1;
    orgqr_(&mm, &cc, &kk, q, tau, &wq, &qq, &info);
    if (info != 0) return static_cast<int>(info);
    const lint lw = lwork_from(wq);
    std::vector<T> work(static_cast<std::size_t>(lw));
    orgqr_(&mm, &cc, &kk, q, tau, work.data(), &lw, &info);
    return static_cast<int>(info);
  }
};

class AccelerateBackend final : public Backend {
 public:
  [[nodiscard]] std::string_view name() const noexcept override { return "accelerate"; }
  [[nodiscard]] const Routines<float>& f32() const noexcept override { return f32_; }
  [[nodiscard]] const Routines<double>& f64() const noexcept override { return f64_; }

 private:
  AccelRoutines<float> f32_;
  AccelRoutines<double> f64_;
};

}  // namespace

const Backend& default_backend() {
  static const AccelerateBackend backend;
  return backend;
}

}  // namespace nativpy::linalg

#else

namespace nativpy::linalg {
const Backend& default_backend() { return fallback_backend(); }
}  // namespace nativpy::linalg

#endif

namespace nativpy::linalg {

namespace {
const Backend* g_active = nullptr;
}  // namespace

const Backend& active_backend() { return g_active != nullptr ? *g_active : default_backend(); }
void set_active_backend(const Backend* backend) noexcept { g_active = backend; }

}  // namespace nativpy::linalg
