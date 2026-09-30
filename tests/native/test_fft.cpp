#include <cmath>
#include <complex>
#include <cstdint>
#include <numbers>
#include <optional>
#include <vector>

#include "error.hpp"
#include "fft.hpp"
#include "test_harness.hpp"

using namespace nativpy;
using cd = std::complex<double>;

namespace {

NDArray real_vec(const std::vector<double>& v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_double(static_cast<std::int64_t>(i), v[i]);
  return a;
}

const cd* cdata(const NDArray& a) { return reinterpret_cast<const cd*>(a.data()); }

// Naive forward DFT of length n (zero-pad or truncate x).
std::vector<cd> dft(const std::vector<double>& x, std::size_t n) {
  std::vector<cd> out(n);
  for (std::size_t k = 0; k < n; ++k) {
    cd acc = 0;
    for (std::size_t j = 0; j < n && j < x.size(); ++j) {
      const double ang = -2.0 * std::numbers::pi * static_cast<double>(j * k) /
                         static_cast<double>(n);
      acc += x[j] * cd(std::cos(ang), std::sin(ang));
    }
    out[k] = acc;
  }
  return out;
}

bool cnear(cd a, cd b, double tol = 1e-9) { return std::abs(a - b) <= tol * (1.0 + std::abs(b)); }

const std::vector<double> kX = {1.0, -2.5, 3.0, 0.5, 4.0, -1.0, 2.0};  // n = 7 (odd)
constexpr auto kB = fft::Norm::Backward;
constexpr std::nullopt_t kDef = std::nullopt;

}  // namespace

TEST_CASE("fft: 1-D fft matches naive DFT, pad and truncate") {
  for (const std::int64_t n : {7, 4, 10, 1}) {
    const NDArray r = fft::fft(real_vec(kX), n, -1, kB);
    CHECK(r.dtype() == DType::Complex128);
    CHECK_EQ(r.shape()[0], n);
    const auto ref = dft(kX, static_cast<std::size_t>(n));
    for (std::int64_t k = 0; k < n; ++k) CHECK(cnear(cdata(r)[k], ref[static_cast<std::size_t>(k)]));
  }
}

TEST_CASE("fft: ifft round trip and norms") {
  const NDArray x = real_vec(kX);
  for (const auto norm : {fft::Norm::Backward, fft::Norm::Ortho, fft::Norm::Forward}) {
    const NDArray back = fft::ifft(fft::fft(x, kDef, 0, norm), kDef, 0, norm);
    for (std::size_t k = 0; k < kX.size(); ++k) CHECK(cnear(cdata(back)[k], cd(kX[k], 0)));
  }
  const NDArray b = fft::fft(x, kDef, 0, kB);
  const NDArray o = fft::fft(x, kDef, 0, fft::Norm::Ortho);
  CHECK(cnear(cdata(o)[2], cdata(b)[2] / std::sqrt(7.0)));
  const NDArray f = fft::fft(x, kDef, 0, fft::Norm::Forward);
  CHECK(cnear(cdata(f)[3], cdata(b)[3] / 7.0));
}

TEST_CASE("fft: rfft/irfft even and odd lengths") {
  for (const std::int64_t n : {7, 6, 8, 3, 2, 1}) {
    const NDArray r = fft::rfft(real_vec(kX), n, -1, kB);
    CHECK_EQ(r.shape()[0], n / 2 + 1);
    const auto ref = dft(kX, static_cast<std::size_t>(n));
    for (std::int64_t k = 0; k <= n / 2; ++k) {
      CHECK(cnear(cdata(r)[k], ref[static_cast<std::size_t>(k)]));
    }
    const NDArray back = fft::irfft(r, n, -1, kB);
    CHECK(back.dtype() == DType::Float64);
    CHECK_EQ(back.shape()[0], n);
    for (std::int64_t k = 0; k < n && k < 7; ++k) {
      CHECK(std::abs(back.get_double(k) - kX[static_cast<std::size_t>(k)]) < 1e-9);
    }
  }
}

TEST_CASE("fft: result dtypes") {
  const NDArray x = real_vec(kX, DType::Float32);
  CHECK(fft::fft(x, kDef, 0, kB).dtype() == DType::Complex64);
  const NDArray r = fft::rfft(x, kDef, 0, kB);
  CHECK(r.dtype() == DType::Complex64);
  CHECK(fft::irfft(r, 7, 0, kB).dtype() == DType::Float32);
  const NDArray h = real_vec({1, 2, 3, 4}, DType::Float16);
  CHECK(fft::fft(h, kDef, 0, kB).dtype() == DType::Complex64);
  CHECK(fft::irfft(h, kDef, 0, kB).dtype() == DType::Float16);
  const NDArray i = real_vec({1, 2, 3}, DType::Int32);
  CHECK(fft::fft(i, kDef, 0, kB).dtype() == DType::Complex128);
  CHECK(fft::irfft(i, kDef, 0, kB).dtype() == DType::Float64);
}

TEST_CASE("fft: axis handling, strided input and fftn") {
  NDArray a = NDArray::empty({2, 3}, DType::Float64);
  for (std::int64_t i = 0; i < 6; ++i) a.set_double(i, static_cast<double>(i * i));
  // Length-2 fft along axis 0 is sum / difference.
  const NDArray r0 = fft::fft(a, kDef, 0, kB);
  for (std::int64_t j = 0; j < 3; ++j) {
    CHECK(cnear(cdata(r0)[j], cd(a.get_double(j) + a.get_double(3 + j), 0)));
    CHECK(cnear(cdata(r0)[3 + j], cd(a.get_double(j) - a.get_double(3 + j), 0)));
  }
  // Transposed (non-contiguous) view gives the same result as its copy.
  const NDArray t = a.view({3, 2}, {8, 24}, a.offset());
  const NDArray rt = fft::fft(t, kDef, -1, kB);
  const NDArray rc = fft::fft(t.copy(), kDef, -1, kB);
  for (std::int64_t k = 0; k < 6; ++k) CHECK(cnear(cdata(rt)[k], cdata(rc)[k]));
  // fftn over all axes: [0,0] is the total sum; ifftn inverts it.
  const NDArray f = fft::fftn(a, kDef, kDef, kB);
  CHECK(cnear(cdata(f)[0], cd(55, 0)));
  const NDArray fi = fft::ifftn(f, kDef, kDef, kB);
  for (std::int64_t k = 0; k < 6; ++k) CHECK(cnear(cdata(fi)[k], cd(a.get_double(k), 0)));
  // s = (-1, 5) keeps axis 0 and pads axis 1.
  const NDArray fs =
      fft::fftn(a, std::vector<std::int64_t>{-1, 5}, std::vector<std::int64_t>{0, 1}, kB);
  CHECK(fs.shape() == (Shape{2, 5}));
  // s without axes uses the last len(s) axes.
  const NDArray fl = fft::fftn(a, std::vector<std::int64_t>{4}, kDef, kB);
  CHECK(fl.shape() == (Shape{2, 4}));
  // Empty axes returns the input unchanged.
  CHECK(fft::fftn(a, kDef, std::vector<std::int64_t>{}, kB).dtype() == DType::Float64);
  // Zero-size non-transformed dim.
  CHECK(fft::fft(NDArray::zeros({2, 0}, DType::Float64), kDef, 0, kB).shape() == (Shape{2, 0}));
}

TEST_CASE("fft: errors") {
  const NDArray x = real_vec(kX);
  CHECK_THROWS_KIND(fft::fft(x, 0, 0, kB), ErrorKind::Value);
  CHECK_THROWS_KIND(fft::fft(x, -1, 0, kB), ErrorKind::Value);
  CHECK_THROWS_KIND(fft::fft(real_vec({}), kDef, 0, kB), ErrorKind::Value);
  CHECK_THROWS_KIND(fft::irfft(real_vec({1}), kDef, 0, kB), ErrorKind::Value);
  CHECK_THROWS_KIND(fft::fft(x, kDef, 1, kB), ErrorKind::Index);
  CHECK_THROWS_KIND(fft::fft(NDArray::zeros({}, DType::Float64), kDef, -1, kB), ErrorKind::Index);
  CHECK_THROWS_KIND(fft::parse_norm("bogus"), ErrorKind::Value);
  const NDArray c = fft::fft(x, kDef, 0, kB);
  CHECK_THROWS_KIND(fft::rfft(c, kDef, 0, kB), ErrorKind::DType);
  CHECK_THROWS_KIND(
      fft::fftn(x, std::vector<std::int64_t>{3}, std::vector<std::int64_t>{0, 0}, kB),
      ErrorKind::Value);
  CHECK_THROWS_KIND(fft::fftfreq(0, 1.0), ErrorKind::Value);
  CHECK_THROWS_KIND(fft::fftfreq(-1, 1.0), ErrorKind::Value);
}

TEST_CASE("fft: fftfreq / rfftfreq") {
  const NDArray f = fft::fftfreq(5, 0.1);
  const double exp5[] = {0, 2, 4, -4, -2};
  for (std::int64_t i = 0; i < 5; ++i) CHECK(std::abs(f.get_double(i) - exp5[i]) < 1e-12);
  const NDArray g = fft::fftfreq(4, 1.0);
  const double exp4[] = {0, 0.25, -0.5, -0.25};
  for (std::int64_t i = 0; i < 4; ++i) CHECK(std::abs(g.get_double(i) - exp4[i]) < 1e-15);
  const NDArray r = fft::rfftfreq(5, 1.0);
  CHECK_EQ(r.shape()[0], 3);
  CHECK(std::abs(r.get_double(2) - 0.4) < 1e-15);
}

