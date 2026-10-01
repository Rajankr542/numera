#pragma once

#include <complex>
#include <cstdint>
#include <memory>
#include <string_view>

namespace nativpy::linalg {

// Numerical backend abstraction (PLAN §20/§45, D-018). All matrices are dense,
// COLUMN-major (LAPACK convention) with leading dimension == rows, except gemm
// which takes row-major operands. Routines return a LAPACK-style `info`:
// 0 = success, > 0 = numerical failure (singular / no convergence).
// T is float or double.

// NumPy's non-BLAS complex matmul formula (D-035), row-major: each output
// starts at +0 and adds (ar*br - ai*bi, ar*bi + ai*br) in increasing p order.
// The i-p-j loop order keeps that per-element summation order. No Annex G
// NaN recovery. Shared by the fallback backend and the BLAS backends'
// non-BLAS cases (D-036).
template <typename T>
void noblas_cgemm(std::int64_t m, std::int64_t n, std::int64_t k, const std::complex<T>* a,
                  const std::complex<T>* b, std::complex<T>* c) {
  for (std::int64_t i = 0; i < m; ++i) {
    std::complex<T>* crow = c + i * n;
    for (std::int64_t j = 0; j < n; ++j) crow[j] = std::complex<T>{};
    for (std::int64_t p = 0; p < k; ++p) {
      const T ar = a[i * k + p].real();
      const T ai = a[i * k + p].imag();
      const std::complex<T>* brow = b + p * n;
      for (std::int64_t j = 0; j < n; ++j) {
        const T br = brow[j].real();
        const T bi = brow[j].imag();
        crow[j] = {crow[j].real() + (ar * br - ai * bi), crow[j].imag() + (ar * bi + ai * br)};
      }
    }
  }
}

template <typename T>
class Routines {
 public:
  virtual ~Routines() = default;
  // Row-major C[m×n] = A[m×k] · B[k×n].
  virtual void gemm(std::int64_t m, std::int64_t n, std::int64_t k, const T* a, const T* b,
                    T* c) const = 0;
  // Complex row-major C[m×n] = A[m×k] · B[k×n], no conjugation (D-035).
  virtual void cgemm(std::int64_t m, std::int64_t n, std::int64_t k, const std::complex<T>* a,
                     const std::complex<T>* b, std::complex<T>* c) const = 0;
  // LU factorisation in place (n×n); piv receives 0-based row swaps.
  virtual int getrf(std::int64_t n, T* a, std::int64_t* piv) const = 0;
  // Solves A X = B for n×nrhs B (A overwritten by LU, B by X).
  virtual int gesv(std::int64_t n, std::int64_t nrhs, T* a, T* b) const = 0;
  // Symmetric eigen (lower triangle used): w ascending, a -> eigenvectors.
  virtual int syevd(std::int64_t n, T* a, T* w) const = 0;
  // General eigen: w complex, v complex n×n (column eigenvectors, unit 2-norm).
  virtual int geev(std::int64_t n, T* a, std::complex<T>* w, std::complex<T>* v) const = 0;
  // SVD of m×n A. s: min(m,n). If u/vt null, values only. full: U m×m and
  // Vt n×n; else U m×k and Vt k×n.
  virtual int gesdd(std::int64_t m, std::int64_t n, T* a, T* s, T* u, T* vt, bool full) const = 0;
  // Householder QR of m×n A: a -> R in upper triangle + reflectors; tau: min(m,n).
  virtual int geqrf(std::int64_t m, std::int64_t n, T* a, T* tau) const = 0;
  // Forms the first `cols` columns of Q (m×cols) from geqrf output in q
  // (q must hold the reflectors in its first k columns, ld = m).
  virtual int orgqr(std::int64_t m, std::int64_t cols, std::int64_t k, T* q,
                    const T* tau) const = 0;
};

class Backend {
 public:
  virtual ~Backend() = default;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  [[nodiscard]] virtual const Routines<float>& f32() const noexcept = 0;
  [[nodiscard]] virtual const Routines<double>& f64() const noexcept = 0;
};

// Portable C++ implementation (always available).
const Backend& fallback_backend();
// Platform-optimised backend when compiled in (Accelerate), else fallback.
const Backend& default_backend();

// Backend used by the NDArray-level API. Tests may override it.
const Backend& active_backend();
void set_active_backend(const Backend* backend) noexcept;  // nullptr = default

}  // namespace nativpy::linalg
