#include "p14_window.hpp"

#include <array>
#include <cmath>
#include <numbers>

namespace nativpy::p14 {

namespace {

constexpr std::array<double, 30> kI0A = {
    -4.4153416464793395e-18, 3.3307945188222384e-17, -2.431279846547955e-16, 1.715391285555133e-15,
    -1.1685332877993451e-14, 7.676185498604936e-14,  -4.856446783111929e-13, 2.95505266312964e-12,
    -1.726826291441556e-11,  9.675809035373237e-11,  -5.189795601635263e-10, 2.6598237246823866e-09,
    -1.300025009986248e-08,  6.046995022541919e-08,  -2.670793853940612e-07, 1.1173875391201037e-06,
    -4.4167383584587505e-06, 1.6448448070728896e-05, -5.754195010082104e-05, 0.00018850288509584165,
    -0.0005763755745385824,  0.0016394756169413357,  -0.004324309995050576,  0.010546460394594998,
    -0.02373741480589947,    0.04930528423967071,    -0.09490109704804764,   0.17162090152220877,
    -0.3046826723431984,     0.6767952744094761};

constexpr std::array<double, 25> kI0B = {
    -7.233180487874754e-18, -4.830504485944182e-18, 4.46562142029676e-17,   3.461222867697461e-17,
    -2.8276239805165836e-16, -3.425485619677219e-16, 1.7725601330565263e-15, 3.8116806693526224e-15,
    -9.554846698828307e-15, -4.150569347287222e-14, 1.54008621752141e-14,   3.8527783827421426e-13,
    7.180124451383666e-13,  -1.7941785315068062e-12, -1.3215811840447713e-11, -3.1499165279632416e-11,
    1.1889147107846439e-11, 4.94060238822497e-10,   3.3962320257083865e-09, 2.266668990498178e-08,
    2.0489185894690638e-07, 2.8913705208347567e-06, 6.889758346916825e-05,  0.0033691164782556943,
    0.8044904110141088};

template <std::size_t N>
double chbevl(double x, const std::array<double, N>& vals) {
  double b0 = vals[0], b1 = 0.0, b2 = 0.0;
  for (std::size_t i = 1; i < N; ++i) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + vals[i];
  }
  return 0.5 * (b0 - b2);
}

// Length of np.arange(start, stop, step) for float arguments.
std::int64_t arange_len(double start, double stop, double step) {
  const double n = std::ceil((stop - start) / step);
  return n > 0 ? static_cast<std::int64_t>(n) : 0;
}

}  // namespace

double bessel_i0(double x) {
  x = std::fabs(x);
  if (x <= 8.0) return std::exp(x) * chbevl(x / 2.0 - 2, kI0A);
  return std::exp(x) * chbevl(32.0 / x - 2.0, kI0B) / std::sqrt(x);
}

NDArray window(Window kind, double m) {
  if (!(m >= 1)) return NDArray::empty({0}, DType::Float64);
  if (m == 1) {
    NDArray one = NDArray::empty({1}, DType::Float64);
    one.set_double(0, 1.0);
    return one;
  }
  const std::int64_t len = arange_len(1 - m, m, 2);
  NDArray out = NDArray::empty({len}, DType::Float64);
  auto* p = reinterpret_cast<double*>(out.data());
  constexpr double pi = std::numbers::pi;
  for (std::int64_t k = 0; k < len; ++k) {
    const double n = (1 - m) + 2.0 * static_cast<double>(k);
    switch (kind) {
      case Window::Bartlett: p[k] = n <= 0 ? 1 + n / (m - 1) : 1 - n / (m - 1); break;
      case Window::Blackman: p[k] = 0.42 + 0.5 * std::cos(pi * n / (m - 1)) + 0.08 * std::cos(2.0 * pi * n / (m - 1)); break;
      case Window::Hamming: p[k] = 0.54 + 0.46 * std::cos(pi * n / (m - 1)); break;
      case Window::Hanning: p[k] = 0.5 + 0.5 * std::cos(pi * n / (m - 1)); break;
    }
  }
  return out;
}

NDArray kaiser(double m, double beta) {
  if (m == 1) {
    NDArray one = NDArray::empty({1}, DType::Float64);
    one.set_double(0, 1.0);
    return one;
  }
  const std::int64_t len = arange_len(0, m, 1);
  NDArray out = NDArray::empty({len}, DType::Float64);
  auto* p = reinterpret_cast<double*>(out.data());
  const double alpha = (m - 1) / 2.0;
  const double denom = bessel_i0(beta);
  for (std::int64_t k = 0; k < len; ++k) {
    const double r = (static_cast<double>(k) - alpha) / alpha;
    p[k] = bessel_i0(beta * std::sqrt(1 - std::pow(r, 2.0))) / denom;
  }
  return out;
}

}  // namespace nativpy::p14
