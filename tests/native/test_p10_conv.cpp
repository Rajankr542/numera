#include <cmath>
#include <complex>

#include "error.hpp"
#include "p10_conv.hpp"
#include "test_harness.hpp"
#include "test_p10_util.hpp"

using namespace nativpy;
using namespace p10test;

// ---- correlate ----

TEST_CASE("p10_conv: correlate full") {
  // NumPy: np.correlate([1,2,3], [0,1,0.5], 'full') = [0.5, 2.0, 3.5, 3.0, 0.0]
  const NDArray a = dbl({1, 2, 3});
  const NDArray v = dbl({0, 1, 0.5});
  const NDArray r = p10::correlate(a, v, "full");
  CHECK_EQ(r.size(), 5);
  CHECK(close(vals(r), D{0.5, 2.0, 3.5, 3.0, 0.0}));
}

TEST_CASE("p10_conv: correlate same") {
  // NumPy: np.correlate([1,2,3], [0,1,0.5], 'same') = [2.0, 3.5, 3.0]
  const NDArray a = dbl({1, 2, 3});
  const NDArray v = dbl({0, 1, 0.5});
  const NDArray r = p10::correlate(a, v, "same");
  CHECK_EQ(r.size(), 3);
  CHECK(close(vals(r), D{2.0, 3.5, 3.0}));
}

TEST_CASE("p10_conv: correlate valid") {
  // NumPy: np.correlate([1,2,3], [0,1,0.5], 'valid') = [3.5]
  const NDArray a = dbl({1, 2, 3});
  const NDArray v = dbl({0, 1, 0.5});
  const NDArray r = p10::correlate(a, v, "valid");
  CHECK_EQ(r.size(), 1);
  CHECK_EQ(r.get_double(0), 3.5);
}

TEST_CASE("p10_conv: correlate valid v_longer") {
  // NumPy: np.correlate([1,2], [1,2,3], 'valid') = [8, 5]
  const NDArray a = dbl({1, 2});
  const NDArray v = dbl({1, 2, 3});
  const NDArray r = p10::correlate(a, v, "valid");
  CHECK_EQ(r.size(), 2);
  CHECK(vals(r) == (D{8, 5}));
}

TEST_CASE("p10_conv: correlate same v_longer") {
  // NumPy: np.correlate([1,2], [1,2,3], 'same') = [8, 5, 2]
  const NDArray a = dbl({1, 2});
  const NDArray v = dbl({1, 2, 3});
  const NDArray r = p10::correlate(a, v, "same");
  CHECK_EQ(r.size(), 3);
  CHECK(vals(r) == (D{8, 5, 2}));
}

TEST_CASE("p10_conv: correlate full v_longer") {
  // NumPy: np.correlate([1,2], [1,2,3], 'full') = [3, 8, 5, 2]
  const NDArray a = dbl({1, 2});
  const NDArray v = dbl({1, 2, 3});
  const NDArray r = p10::correlate(a, v, "full");
  CHECK_EQ(r.size(), 4);
  CHECK(vals(r) == (D{3, 8, 5, 2}));
}

TEST_CASE("p10_conv: correlate equal length") {
  // np.correlate([1,2,3], [4,5,6], 'full') = [6, 17, 32, 23, 12]
  const NDArray r = p10::correlate(dbl({1, 2, 3}), dbl({4, 5, 6}), "full");
  CHECK(vals(r) == (D{6, 17, 32, 23, 12}));
  // same = valid for equal-length full correlation
  // same: length 3, centered
  const NDArray rs = p10::correlate(dbl({1, 2, 3}), dbl({4, 5, 6}), "same");
  CHECK_EQ(rs.size(), 3);
  CHECK(vals(rs) == (D{17, 32, 23}));
  // valid: length 1
  const NDArray rv = p10::correlate(dbl({1, 2, 3}), dbl({4, 5, 6}), "valid");
  CHECK_EQ(rv.size(), 1);
  CHECK_EQ(rv.get_double(0), 32.0);
}

TEST_CASE("p10_conv: correlate complex") {
  // np.correlate([1+1j, 2, 3], [1+1j, 2], 'full') = [2+2j, 6+0j, 8-2j, 3-3j]
  NDArray a = NDArray::empty({3}, DType::Complex128);
  NDArray v = NDArray::empty({2}, DType::Complex128);
  // complex128 raw layout: [re, im, re, im, ...]
  auto* ad = reinterpret_cast<double*>(a.data());
  ad[0] = 1; ad[1] = 1;   // 1+1j
  ad[2] = 2; ad[3] = 0;   // 2+0j
  ad[4] = 3; ad[5] = 0;   // 3+0j
  auto* vd = reinterpret_cast<double*>(v.data());
  vd[0] = 1; vd[1] = 1;   // 1+1j
  vd[2] = 2; vd[3] = 0;   // 2+0j
  const NDArray r = p10::correlate(a, v, "full");
  CHECK_EQ(r.size(), 4);
  // Verify real and imaginary parts via raw pointer: [2+2j, 6+0j, 8-2j, 3-3j]
  const auto* rd = reinterpret_cast<const double*>(r.data());
  CHECK(close(D{rd[0],rd[1],rd[2],rd[3],rd[4],rd[5],rd[6],rd[7]},
              D{2, 2, 6, 0, 8, -2, 3, -3}));
}

TEST_CASE("p10_conv: correlate int dtype") {
  // np.correlate([1,2,3], [4,5], 'full') with int32 inputs → int32 output
  const NDArray a = dbl({1, 2, 3}, DType::Int32);
  const NDArray v = dbl({4, 5}, DType::Int32);
  const NDArray r = p10::correlate(a, v, "full");
  CHECK(r.dtype() == DType::Int32);
  CHECK(vals(r) == (D{5, 14, 23, 12}));
}

TEST_CASE("p10_conv: correlate bad mode") {
  CHECK_THROWS_KIND(p10::correlate(dbl({1}), dbl({1}), "bad"), ErrorKind::Value);
}

// ---- convolve ----

TEST_CASE("p10_conv: convolve full") {
  // np.convolve([1,2,3], [0,1,0.5], 'full') = [0., 1., 2.5, 4., 1.5]
  const NDArray a = dbl({1, 2, 3});
  const NDArray v = dbl({0, 1, 0.5});
  const NDArray r = p10::convolve(a, v, "full");
  CHECK_EQ(r.size(), 5);
  CHECK(close(vals(r), D{0.0, 1.0, 2.5, 4.0, 1.5}));
}

TEST_CASE("p10_conv: convolve same") {
  // np.convolve([1,2,3], [0,1,0.5], 'same') = [1., 2.5, 4.]
  const NDArray a = dbl({1, 2, 3});
  const NDArray v = dbl({0, 1, 0.5});
  const NDArray r = p10::convolve(a, v, "same");
  CHECK_EQ(r.size(), 3);
  CHECK(close(vals(r), D{1.0, 2.5, 4.0}));
}

TEST_CASE("p10_conv: convolve valid") {
  // np.convolve([1,2,3], [0,1,0.5], 'valid') = [2.5]
  const NDArray a = dbl({1, 2, 3});
  const NDArray v = dbl({0, 1, 0.5});
  const NDArray r = p10::convolve(a, v, "valid");
  CHECK_EQ(r.size(), 1);
  CHECK_EQ(r.get_double(0), 2.5);
}

TEST_CASE("p10_conv: convolve v_reversal") {
  // convolve(a, v) == correlate(a, v[::-1]) for real inputs
  const NDArray a = dbl({1, 2, 3, 4});
  const NDArray v = dbl({1, 2, 3});
  const NDArray cv = p10::convolve(a, v, "full");
  // Expected: np.convolve([1,2,3,4],[1,2,3]) = [1,4,10,16,17,12]
  CHECK_EQ(cv.size(), 6);
  CHECK(vals(cv) == (D{1, 4, 10, 16, 17, 12}));
}

TEST_CASE("p10_conv: convolve complex") {
  // np.convolve([1+1j, 2+2j], [1, 1j]) = [1+1j, 1+3j, -2+2j]
  NDArray a = NDArray::empty({2}, DType::Complex128);
  NDArray v = NDArray::empty({2}, DType::Complex128);
  // complex128 raw layout: [re, im, re, im, ...]
  auto* ad = reinterpret_cast<double*>(a.data());
  ad[0] = 1; ad[1] = 1;   // 1+1j
  ad[2] = 2; ad[3] = 2;   // 2+2j
  auto* vd = reinterpret_cast<double*>(v.data());
  vd[0] = 1; vd[1] = 0;   // 1+0j
  vd[2] = 0; vd[3] = 1;   // 0+1j
  const NDArray r = p10::convolve(a, v, "full");
  CHECK_EQ(r.size(), 3);
  // [1+1j, 1+3j, -2+2j]
  const auto* rd = reinterpret_cast<const double*>(r.data());
  CHECK(close(D{rd[0],rd[1],rd[2],rd[3],rd[4],rd[5]},
              D{1, 1, 1, 3, -2, 2}));
}

TEST_CASE("p10_conv: convolve single element") {
  // np.convolve([3.0], [2.0]) = [6.0]
  const NDArray r = p10::convolve(dbl({3.0}), dbl({2.0}), "full");
  CHECK_EQ(r.size(), 1);
  CHECK_EQ(r.get_double(0), 6.0);
}
