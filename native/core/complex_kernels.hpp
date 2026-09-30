#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>

namespace nativpy::kernels {

// Complex scalar kernels following NumPy's loops (D-033). Formulas are written
// out instead of using std::complex operators where NumPy's result differs
// (C99 Annex G NaN recovery in operator*, and division).

template <typename T>
std::complex<T> cadd(std::complex<T> a, std::complex<T> b) noexcept {
  return {a.real() + b.real(), a.imag() + b.imag()};
}

template <typename T>
std::complex<T> csub(std::complex<T> a, std::complex<T> b) noexcept {
  return {a.real() - b.real(), a.imag() - b.imag()};
}

template <typename T>
std::complex<T> cmul(std::complex<T> a, std::complex<T> b) noexcept {
  const T ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
  return {ar * br - ai * bi, ar * bi + ai * br};
}

// Smith's algorithm, as in NumPy's complex divide loop.
template <typename T>
std::complex<T> cdiv(std::complex<T> a, std::complex<T> b) noexcept {
  const T ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
  const T abr = std::fabs(br), abi = std::fabs(bi);
  if (abr >= abi) {
    if (abr == T{0} && abi == T{0}) return {ar / abr, ai / abr};
    const T rat = bi / br;
    const T scl = T{1} / (br + bi * rat);
    return {(ar + ai * rat) * scl, (ai - ar * rat) * scl};
  }
  const T rat = br / bi;
  const T scl = T{1} / (bi + br * rat);
  return {(ar * rat + ai) * scl, (ai * rat - ar) * scl};
}

// Port of NumPy npy_cpow: exact small integer powers by repeated squaring.
template <typename T>
std::complex<T> cpow(std::complex<T> a, std::complex<T> b) noexcept {
  const T ar = a.real(), ai = a.imag(), br = b.real(), bi = b.imag();
  if (br == T{0} && bi == T{0}) return {T{1}, T{0}};
  if (ar == T{0} && ai == T{0}) {
    if (br > T{0}) return {T{0}, T{0}};
    const T nan = std::numeric_limits<T>::quiet_NaN();
    return {nan, nan};
  }
  if (bi == T{0} && std::fabs(br) < T{100} && std::trunc(br) == br) {
    auto n = static_cast<std::int64_t>(br);
    if (n == 1) return a;
    if (n == 2) return cmul(a, a);
    if (n == 3) return cmul(a, cmul(a, a));
    std::complex<T> acc{T{1}, T{0}};
    std::complex<T> p = a;
    if (n < 0) n = -n;
    for (std::int64_t mask = 1;; mask <<= 1) {
      if ((n & mask) != 0) acc = cmul(acc, p);
      if (n < (mask << 1)) break;
      p = cmul(p, p);
    }
    return br < T{0} ? cdiv(std::complex<T>{T{1}, T{0}}, acc) : acc;
  }
  return std::pow(a, b);
}

template <typename T>
std::complex<T> cneg(std::complex<T> a) noexcept {
  return {-a.real(), -a.imag()};
}

template <typename T>
std::complex<T> cconj(std::complex<T> a) noexcept {
  return {a.real(), -a.imag()};
}

// NumPy's SIMD cabsolute (loops_unary_complex): inf beats nan, then a scaled
// hypot with a fused multiply-add, matching NumPy's NEON/AVX results.
template <typename T>
T cabs(std::complex<T> a) noexcept {
  const T inf = std::numeric_limits<T>::infinity();
  T re = std::fabs(a.real()), im = std::fabs(a.imag());
  if (re == inf || im == inf) return inf;
  if (std::isnan(re) || std::isnan(im)) return std::numeric_limits<T>::quiet_NaN();
  const T larger = re > im ? re : im;
  const T smaller = re > im ? im : re;
  if (larger == T{0}) return T{0};
  const T ratio = smaller / larger;
  return std::sqrt(std::fma(ratio, ratio, T{1})) * larger;
}

// Port of NumPy npy_csqrt (msun, CACM Algorithm 312).
template <typename T>
std::complex<T> csqrt(std::complex<T> z) noexcept {
  T a = z.real(), b = z.imag();
  const T inf = std::numeric_limits<T>::infinity();
  if (a == T{0} && b == T{0}) return {T{0}, b};
  if (std::isinf(b)) return {inf, b};
  if (std::isnan(a)) return {a, (b - b) / (b - b)};
  if (std::isinf(a)) {
    if (std::signbit(a)) return {std::fabs(b - b), std::copysign(a, b)};
    return {a, std::copysign(b - b, b)};
  }
  const T thresh = std::numeric_limits<T>::max() / (T{1} + std::sqrt(T{2}));
  const bool scale = std::fabs(a) >= thresh || std::fabs(b) >= thresh;
  if (scale) {
    a *= T{0.25};
    b *= T{0.25};
  }
  std::complex<T> r;
  if (a >= T{0}) {
    const T t = std::sqrt((a + std::hypot(a, b)) * T{0.5});
    r = {t, b / (2 * t)};
  } else {
    const T t = std::sqrt((-a + std::hypot(a, b)) * T{0.5});
    r = {std::fabs(b) / (2 * t), std::copysign(t, b)};
  }
  return scale ? std::complex<T>{r.real() * 2, r.imag()} : r;
}

// Port of NumPy npy_clog (CPython algorithm).
template <typename T>
std::complex<T> clog(std::complex<T> z) noexcept {
  const T ax = std::fabs(z.real()), ay = std::fabs(z.imag());
  const T ln2 = static_cast<T>(0.693147180559945309417232121458176568L);
  const int mant = std::numeric_limits<T>::digits;
  T rr;
  if (std::isnan(ax) || std::isnan(ay) || std::isinf(ax) || std::isinf(ay)) {
    rr = std::log(std::hypot(ax, ay));
  } else if (ax > std::numeric_limits<T>::max() / 4 || ay > std::numeric_limits<T>::max() / 4) {
    rr = std::log(std::hypot(ax / 2, ay / 2)) + ln2;
  } else if (ax < std::numeric_limits<T>::min() && ay < std::numeric_limits<T>::min()) {
    if (ax > T{0} || ay > T{0}) {
      rr = std::log(std::hypot(std::ldexp(ax, mant), std::ldexp(ay, mant))) -
           static_cast<T>(mant) * ln2;
    } else {
      rr = -std::numeric_limits<T>::infinity();
    }
  } else {
    const T h = std::hypot(ax, ay);
    if (0.71 <= static_cast<double>(h) && static_cast<double>(h) <= 1.73) {  // double literals, as in C
      const T am = ax > ay ? ax : ay;
      const T an = ax > ay ? ay : ax;
      rr = std::log1p((am - 1) * (am + 1) + an * an) / 2;
    } else {
      rr = std::log(h);
    }
  }
  return {rr, std::atan2(z.imag(), z.real())};
}

}  // namespace nativpy::kernels
