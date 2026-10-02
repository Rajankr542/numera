#include <complex>
#include <cstdint>
#include <optional>
#include <vector>

#include "error.hpp"
#include "fft.hpp"
#include "p12_fft.hpp"
#include "test_harness.hpp"

using namespace nativpy;
namespace p12 = nativpy::fft::p12;
using cd = std::complex<double>;

namespace {

NDArray vec(const std::vector<double>& v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_double(static_cast<std::int64_t>(i), v[i]);
  return a;
}

const cd* cdata(const NDArray& a) { return reinterpret_cast<const cd*>(a.data()); }

constexpr auto kB = fft::Norm::Backward;
constexpr std::nullopt_t kNone = std::nullopt;

}  // namespace

TEST_CASE("p12: transform matches M10 kernels for float64") {
  const NDArray x = vec({1, -2.5, 3, 0.5, 4, -1, 2});
  const NDArray a = p12::transform(p12::Op::Fft, x, kNone, -1, kB, kNone);
  const NDArray b = fft::fft(x, kNone, -1, kB);
  CHECK(a.dtype() == DType::Complex128);
  for (std::int64_t k = 0; k < 7; ++k) CHECK(cdata(a)[k] == cdata(b)[k]);
  const NDArray r = p12::transform(p12::Op::Irfft, p12::transform(p12::Op::Rfft, x, kNone, 0, kB, kNone),
                                   7, 0, kB, kNone);
  for (std::int64_t k = 0; k < 7; ++k) CHECK(std::abs(r.get_double(k) - x.get_double(k)) < 1e-12);
}

TEST_CASE("p12: result dtypes") {
  const NDArray h = vec({1, 2, 3, 4}, DType::Float16);
  CHECK(p12::transform(p12::Op::Fft, h, kNone, -1, kB, kNone).dtype() == DType::Complex64);
  CHECK(p12::transform(p12::Op::Irfft, h, kNone, -1, kB, kNone).dtype() == DType::Float16);
  CHECK(p12::transform(p12::Op::Hfft, h, kNone, -1, kB, kNone).dtype() == DType::Float16);
  CHECK(p12::transform(p12::Op::Ihfft, h, kNone, -1, kB, kNone).dtype() == DType::Complex64);
  const NDArray i = vec({1, 2, 3, 4}, DType::Int32);
  CHECK(p12::transform(p12::Op::Hfft, i, kNone, -1, kB, kNone).dtype() == DType::Float64);
  const NDArray c = vec({1, 2, 3}, DType::Complex64);
  CHECK(p12::transform(p12::Op::Hfft, c, kNone, -1, kB, kNone).dtype() == DType::Float32);
  CHECK_THROWS_KIND(p12::transform(p12::Op::Ihfft, c, kNone, -1, kB, kNone), ErrorKind::DType);
}

TEST_CASE("p12: hfft / ihfft values") {
  const NDArray h = p12::transform(p12::Op::Hfft, vec({1, 2, 3}), kNone, -1, kB, kNone);
  CHECK(h.shape() == Shape({4}));
  const std::vector<double> want = {8, -2, 0, -2};
  for (std::int64_t k = 0; k < 4; ++k) CHECK(std::abs(h.get_double(k) - want[static_cast<std::size_t>(k)]) < 1e-12);
  const NDArray ih = p12::transform(p12::Op::Ihfft, vec({8, -2, 0, -2}), kNone, -1, kB, kNone);
  CHECK(ih.shape() == Shape({3}));
  for (std::int64_t k = 0; k < 3; ++k) CHECK(std::abs(cdata(ih)[k] - cd(static_cast<double>(k + 1), 0)) < 1e-12);
  // ihfft of a non-symmetric input: conj(rfft(x)) / n.
  const NDArray x = vec({1, 2, 3});
  const NDArray a = p12::transform(p12::Op::Ihfft, x, kNone, -1, kB, kNone);
  const NDArray r = fft::rfft(x, kNone, -1, kB);
  for (std::int64_t k = 0; k < 2; ++k) CHECK(std::abs(cdata(a)[k] - std::conj(cdata(r)[k]) / 3.0) < 1e-12);
  CHECK_THROWS_KIND(p12::transform(p12::Op::Hfft, vec({1}), kNone, -1, kB, kNone), ErrorKind::Value);
  CHECK_THROWS_KIND(p12::transform(p12::Op::Hfft, vec({1, 2}), kNone, 3, kB, kNone), ErrorKind::Index);
}

TEST_CASE("p12: out= writes, casts and validates") {
  const NDArray x = vec({1, 1, 1, 1});
  NDArray out = NDArray::zeros({4}, DType::Complex64);
  const NDArray r = p12::transform(p12::Op::Fft, x, kNone, -1, kB, out);
  CHECK(r.shares_buffer(out));
  CHECK(out.dtype() == DType::Complex64);
  CHECK(reinterpret_cast<const std::complex<float>*>(out.data())[0] == std::complex<float>(4, 0));
  // Wrong length along the axis, wrong ndim.
  CHECK_THROWS_KIND(p12::transform(p12::Op::Fft, x, kNone, -1, kB, NDArray::zeros({3}, DType::Complex128)),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(p12::transform(p12::Op::Fft, x, kNone, -1, kB, NDArray::zeros({1, 4}, DType::Complex128)),
                    ErrorKind::Value);
  // complex -> real is not same_kind.
  CHECK_THROWS_KIND(p12::transform(p12::Op::Fft, x, kNone, -1, kB, NDArray::zeros({4}, DType::Float64)),
                    ErrorKind::DType);
  CHECK_THROWS_KIND(p12::transform(p12::Op::Irfft, x, 4, -1, kB, NDArray::zeros({4}, DType::Int64)),
                    ErrorKind::DType);
  CHECK_THROWS_KIND(p12::transform(p12::Op::Fft, x, kNone, -1, kB, NDArray::zeros({4}, DType::Complex128).as_readonly()),
                    ErrorKind::Value);
  // Broadcast of other dims into out, and a non-broadcastable out.
  const NDArray m = NDArray::zeros({1, 4}, DType::Float64);
  const NDArray big = NDArray::zeros({3, 4}, DType::Complex128);
  CHECK(p12::transform(p12::Op::Fft, m, kNone, -1, kB, big).shape() == Shape({3, 4}));
  CHECK_THROWS_KIND(p12::transform(p12::Op::Fft, big, kNone, -1, kB, NDArray::zeros({2, 4}, DType::Complex128)),
                    ErrorKind::Broadcast);
  // In place: the input aliases out.
  NDArray c = vec({1, 2, 3, 4}, DType::Complex128);
  p12::transform(p12::Op::Fft, c, kNone, -1, kB, c);
  CHECK(cdata(c)[0] == cd(10, 0));
}

TEST_CASE("p12: rfftn / irfftn round trip and s/axes rules") {
  NDArray a = NDArray::empty({2, 3, 4}, DType::Float64);
  for (std::int64_t i = 0; i < a.size(); ++i) a.set_double(i, static_cast<double>(i * i % 7));
  const NDArray s = p12::transform_nd(p12::NdOp::Rfftn, a, kNone, kNone, kB, kNone);
  CHECK(s.shape() == Shape({2, 3, 3}));
  const NDArray back = p12::transform_nd(p12::NdOp::Irfftn, s, std::vector<std::int64_t>{2, 3, 4}, kNone, kB, kNone);
  CHECK(back.shape() == Shape({2, 3, 4}));
  for (std::int64_t i = 0; i < a.size(); ++i) CHECK(std::abs(back.get_double(i) - a.get_double(i)) < 1e-12);
  // Default last length 2 * (m - 1).
  CHECK(p12::transform_nd(p12::NdOp::Irfftn, s, kNone, kNone, kB, kNone).shape() == Shape({2, 3, 4}));
  CHECK(p12::transform_nd(p12::NdOp::Irfftn, s, kNone, std::vector<std::int64_t>{0}, kB, kNone).shape() ==
        Shape({2, 3, 3}));
  // s = -1 keeps the input length; mismatched lengths, empty axes.
  CHECK(p12::transform_nd(p12::NdOp::Rfftn, a, std::vector<std::int64_t>{-1, 6}, std::vector<std::int64_t>{0, 1}, kB,
                          kNone).shape() == Shape({2, 4, 4}));
  CHECK_THROWS_KIND(p12::transform_nd(p12::NdOp::Rfftn, a, std::vector<std::int64_t>{2}, std::vector<std::int64_t>{0, 1},
                                      kB, kNone),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(p12::transform_nd(p12::NdOp::Rfftn, a, kNone, std::vector<std::int64_t>{}, kB, kNone),
                    ErrorKind::Index);
  CHECK_THROWS_KIND(p12::transform_nd(p12::NdOp::Irfftn, a, kNone, std::vector<std::int64_t>{}, kB, kNone),
                    ErrorKind::Index);
  // fftn over no axes returns the input.
  CHECK(p12::transform_nd(p12::NdOp::Fftn, a, kNone, std::vector<std::int64_t>{}, kB, kNone).shares_buffer(a));
}

TEST_CASE("p12: fftshift / ifftshift / roll") {
  const NDArray x = vec({0, 1, 2, 3, 4}, DType::Int64);
  const NDArray f = p12::shift(x, kNone, false);
  const std::vector<std::int64_t> wf = {3, 4, 0, 1, 2};
  for (std::int64_t k = 0; k < 5; ++k) CHECK_EQ(f.get_int64(k), wf[static_cast<std::size_t>(k)]);
  const NDArray g = p12::shift(f, kNone, true);
  for (std::int64_t k = 0; k < 5; ++k) CHECK_EQ(g.get_int64(k), k);
  // 2-D, one axis; repeated axes add up.
  NDArray m = NDArray::empty({2, 3}, DType::Float32);
  for (std::int64_t i = 0; i < 6; ++i) m.set_double(i, static_cast<double>(i));
  const NDArray r = p12::shift(m, std::vector<std::int64_t>{-1}, false);
  CHECK(r.dtype() == DType::Float32);
  const std::vector<double> wr = {2, 0, 1, 5, 3, 4};
  for (std::int64_t i = 0; i < 6; ++i) CHECK_EQ(r.get_double(i), wr[static_cast<std::size_t>(i)]);
  const NDArray rr = p12::roll(x, {1, 1}, {0, 0});
  CHECK_EQ(rr.get_int64(0), std::int64_t{3});
  // Empty arrays, empty axes, 0-d, bad axis.
  CHECK(p12::shift(NDArray::zeros({0, 3}, DType::Float64), kNone, false).shape() == Shape({0, 3}));
  const NDArray same = p12::shift(x, std::vector<std::int64_t>{}, false);
  CHECK(!same.shares_buffer(x));
  CHECK_EQ(same.get_int64(1), std::int64_t{1});
  CHECK_THROWS_KIND(p12::shift(NDArray::zeros({}, DType::Float64), kNone, false), ErrorKind::Value);
  CHECK_THROWS_KIND(p12::shift(x, std::vector<std::int64_t>{1}, false), ErrorKind::Index);
}
