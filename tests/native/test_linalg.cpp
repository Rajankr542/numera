#include <cmath>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <vector>

#include "backend.hpp"
#include "creation.hpp"
#include "error.hpp"
#include "indexing.hpp"
#include "linalg.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"

using namespace nativpy;
using namespace nativpy::linalg;

namespace {

NDArray mat(std::int64_t m, std::int64_t n, std::initializer_list<double> v,
            DType dt = DType::Float64) {
  NDArray a = NDArray::empty({m, n}, dt);
  std::int64_t i = 0;
  for (const double x : v) a.set_double(i++, x);
  return a;
}

bool near(double a, double b, double tol = 1e-9) {
  return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

bool all_near(const NDArray& a, const NDArray& b, double tol = 1e-9) {
  if (a.shape() != b.shape()) return false;
  for (std::int64_t i = 0; i < a.size(); ++i)
    if (!near(a.get_double(i), b.get_double(i), tol)) return false;
  return true;
}

// Runs fn once per backend (fallback, then default).
template <typename Fn>
void each_backend(Fn&& fn) {
  set_active_backend(&fallback_backend());
  fn();
  set_active_backend(&default_backend());
  fn();
  set_active_backend(nullptr);
}

NDArray transpose2(const NDArray& a) {
  const std::int64_t m = a.shape()[0];
  const std::int64_t n = a.shape()[1];
  NDArray t = NDArray::empty({n, m}, a.dtype());
  for (std::int64_t i = 0; i < m; ++i)
    for (std::int64_t j = 0; j < n; ++j) t.set_double(j * m + i, a.get_double(i * n + j));
  return t;
}

}  // namespace

TEST_CASE("linalg: matmul shapes, dtypes and values") {
  each_backend([] {
    const NDArray a = mat(2, 3, {1, 2, 3, 4, 5, 6});
    const NDArray b = mat(3, 2, {7, 8, 9, 10, 11, 12});
    CHECK(all_near(matmul(a, b), mat(2, 2, {58, 64, 139, 154})));
    const NDArray ai = arange(0, 6, 1, DType::Int32).reshape({2, 3});
    const NDArray bi = arange(0, 6, 1, DType::Int32).reshape({3, 2});
    const NDArray ci = matmul(ai, bi);
    CHECK(ci.dtype() == DType::Int32);
    CHECK_EQ(ci.get_int64(0), 10);
    CHECK_EQ(ci.get_int64(3), 40);
    const NDArray v = arange(0, 3, 1, DType::Float64);
    CHECK(matmul(a, v).shape() == Shape({2}));
    CHECK(matmul(v, b).shape() == Shape({2}));
    CHECK(matmul(v, v).shape() == Shape{});
    CHECK(near(matmul(v, v).get_double(0), 5.0));
    const NDArray batch = arange(0, 12, 1, DType::Float32).reshape({2, 2, 3});
    const NDArray r = matmul(batch, b.astype(DType::Float32));
    CHECK(r.shape() == Shape({2, 2, 2}));
    CHECK(r.dtype() == DType::Float32);
    CHECK(near(r.get_double(7), 9 * 8 + 10 * 10 + 11 * 12, 1e-5));  // r[1,1,1]
    CHECK_THROWS_KIND(matmul(a, a), ErrorKind::Shape);
    const NDArray e = matmul(NDArray::zeros({2, 0}, DType::Float64), NDArray::zeros({0, 3}, DType::Float64));
    CHECK(e.shape() == Shape({2, 3}));
    CHECK(near(e.get_double(5), 0.0));
  });
}

TEST_CASE("linalg: matmul in-place operands and uninitialized output (D-022)") {
  each_backend([] {
    // Contiguous view with a nonzero offset: rows 1..2 of a 3x3 matrix.
    const NDArray big = arange(0, 9, 1, DType::Float64).reshape({3, 3});
    const NDArray rows = get_index(big, {IndexItem::slice(1, std::nullopt, std::nullopt)});
    CHECK(rows.is_c_contiguous());
    CHECK(rows.offset() != 0);
    const NDArray id = eye(3, 3, 0, DType::Float64);
    CHECK(all_near(matmul(rows, id), mat(2, 3, {3, 4, 5, 6, 7, 8})));
    // Non-contiguous (transposed) operand still takes the copy path.
    const NDArray t = transpose(big, {1, 0});
    CHECK(all_near(matmul(t, id), transpose2(big)));
    // Broadcast batch operand: (2,2,3) @ (3,2) with b shared across the batch.
    const NDArray b = mat(3, 2, {1, 0, 0, 1, 1, 1});
    const NDArray bat = arange(0, 12, 1, DType::Float64).reshape({2, 2, 3});
    const NDArray r = matmul(bat, b);
    CHECK(all_near(r, mat(4, 2, {2, 3, 8, 9, 14, 15, 20, 21}).reshape({2, 2, 2})));
    // Output starts as `empty`: repeat on dirty heap memory, ints and floats.
    for (int rep = 0; rep < 8; ++rep) {
      { NDArray junk = NDArray::empty({64, 64}, DType::Float64); for (std::int64_t i = 0; i < junk.size(); ++i) junk.set_double(i, 1e300); }
      const NDArray z = matmul(NDArray::zeros({64, 5}, DType::Float64), NDArray::zeros({5, 64}, DType::Float64));
      bool zero = true;
      for (std::int64_t i = 0; i < z.size(); ++i) zero = zero && z.get_double(i) == 0.0;
      CHECK(zero);
      const NDArray zi = matmul(NDArray::zeros({64, 5}, DType::Int32), NDArray::zeros({5, 64}, DType::Int32));
      bool zeroi = true;
      for (std::int64_t i = 0; i < zi.size(); ++i) zeroi = zeroi && zi.get_int64(i) == 0;
      CHECK(zeroi);
      const NDArray zk = matmul(NDArray::zeros({64, 0}, DType::Float32), NDArray::zeros({0, 64}, DType::Float32));
      bool zerok = true;
      for (std::int64_t i = 0; i < zk.size(); ++i) zerok = zerok && zk.get_double(i) == 0.0;
      CHECK(zerok);
    }
  });
}
namespace {

template <typename R>
NDArray cmat(std::int64_t m, std::int64_t n, std::initializer_list<std::complex<double>> v) {
  NDArray a = NDArray::empty({m, n}, std::is_same_v<R, float> ? DType::Complex64 : DType::Complex128);
  auto* p = reinterpret_cast<std::complex<R>*>(a.data());
  for (const auto& x : v) *p++ = {static_cast<R>(x.real()), static_cast<R>(x.imag())};
  return a;
}

template <typename R>
std::complex<R> cat(const NDArray& a, std::int64_t i) {
  return reinterpret_cast<const std::complex<R>*>(a.data())[i];
}

}  // namespace

TEST_CASE("linalg: complex matmul (D-035)") {
  using C = std::complex<double>;
  each_backend([] {
    // Small integers: every product and sum is exact in both widths.
    const auto values = [](auto tag) {
      using R = decltype(tag);
      const NDArray a = cmat<R>(2, 3, {C(1, 2), C(3, -1), C(0, 1), C(-2, 0), C(1, 1), C(2, 3)});
      const NDArray b = cmat<R>(3, 2, {C(1, 0), C(0, 1), C(2, 2), C(-1, 1), C(1, -3), C(4, 0)});
      const NDArray r = matmul(a, b);
      CHECK(r.dtype() == a.dtype());
      CHECK(r.shape() == Shape({2, 2}));
      CHECK(r.is_c_contiguous());
      // Row 0: (1+2j)*1 + (3-j)(2+2j) + j(1-3j) = 1+2j + 8+4j + 3+j = 12+7j
      //        (1+2j)j + (3-j)(-1+j) + j*4     = -2+j + -2+4j + 4j = -4+9j
      // Row 1: -2 + (1+j)(2+2j) + (2+3j)(1-3j) = -2 + 4j + 11-3j = 9+j
      //        -2j + (1+j)(-1+j) + (2+3j)4    = -2j - 2 + 8+12j = 6+10j
      CHECK(cat<R>(r, 0) == std::complex<R>(12, 7));
      CHECK(cat<R>(r, 1) == std::complex<R>(-4, 9));
      CHECK(cat<R>(r, 2) == std::complex<R>(9, 1));
      CHECK(cat<R>(r, 3) == std::complex<R>(6, 10));
      // 1-D @ 1-D: no conjugation, as in NumPy.
      const NDArray v = cmat<R>(1, 2, {C(0, 1), C(1, 1)}).reshape({2});
      const NDArray s = matmul(v, v);
      CHECK(s.shape() == Shape{});
      CHECK(cat<R>(s, 0) == std::complex<R>(-1, 2));  // j*j + (1+j)^2 = -1 + 2j
    };
    values(float{});
    values(double{});

    // Promotion: float64 @ complex64 -> complex128; int32 @ complex64 -> complex128;
    // float32 @ complex64 -> complex64.
    const NDArray c64 = cmat<float>(2, 1, {C(1, 1), C(2, -1)});
    CHECK(matmul(mat(1, 2, {3, 4}), c64).dtype() == DType::Complex128);
    CHECK(cat<double>(matmul(mat(1, 2, {3, 4}), c64), 0) == C(11, -1));
    CHECK(matmul(arange(1, 3, 1, DType::Int32).reshape({1, 2}), c64).dtype() == DType::Complex128);
    CHECK(matmul(mat(1, 2, {3, 4}, DType::Float32), c64).dtype() == DType::Complex64);

    // Batched with a broadcast operand: (2,1,2) @ (2,1).
    const NDArray bat = cmat<double>(2, 2, {C(1, 0), C(0, 1), C(0, -1), C(2, 0)}).reshape({2, 1, 2});
    const NDArray rb = matmul(bat, cmat<double>(2, 1, {C(1, 1), C(0, 2)}));
    CHECK(rb.shape() == Shape({2, 1, 1}));
    CHECK(cat<double>(rb, 0) == C(-1, 1));  // (1+j) + j*2j
    CHECK(cat<double>(rb, 1) == C(1, 3));   // -j(1+j) + 4j = 1 - j + 4j

    // NumPy's non-BLAS formula: no Annex G NaN recovery.
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const C i1 = cat<double>(matmul(cmat<double>(1, 1, {C(inf, 0)}), cmat<double>(1, 1, {C(1, 0)})), 0);
    CHECK(i1.real() == inf);
    CHECK(std::isnan(i1.imag()));
    const C i2 = cat<double>(matmul(cmat<double>(1, 1, {C(inf, inf)}), cmat<double>(1, 1, {C(0, 1)})), 0);
    CHECK(std::isnan(i2.real()));
    CHECK(std::isnan(i2.imag()));
    const C n0 = cat<double>(matmul(cmat<double>(1, 1, {C(nan, 0)}), cmat<double>(1, 1, {C(0, 0)})), 0);
    CHECK(std::isnan(n0.real()));
    CHECK(std::isnan(n0.imag()));
    // -0 products accumulate onto +0.
    const C z = cat<double>(matmul(cmat<double>(1, 1, {C(-0.0, -0.0)}), cmat<double>(1, 1, {C(1, 0)})), 0);
    CHECK(!std::signbit(z.real()));
    CHECK(!std::signbit(z.imag()));

    // Output starts as `empty`: k = 0 must still write +0+0j on dirty memory.
    for (int rep = 0; rep < 4; ++rep) {
      { NDArray junk = NDArray::empty({32, 32}, DType::Complex128); for (std::int64_t i = 0; i < junk.size() * 2; ++i) reinterpret_cast<double*>(junk.data())[i] = 1e300; }
      const NDArray zk = matmul(NDArray::zeros({16, 0}, DType::Complex128), NDArray::zeros({0, 16}, DType::Complex128));
      bool zero = true;
      for (std::int64_t i = 0; i < zk.size(); ++i) {
        const C x = cat<double>(zk, i);
        zero = zero && x == C(0, 0) && !std::signbit(x.real()) && !std::signbit(x.imag());
      }
      CHECK(zero);
    }
    CHECK_THROWS_KIND(matmul(cmat<double>(1, 2, {C(1, 0), C(1, 0)}), cmat<double>(1, 2, {C(1, 0), C(1, 0)})),
                      ErrorKind::Shape);

    // complex64 vs a double-precision reference on a larger product (tolerance, D-035).
    const std::int64_t m = 7, k = 33, n = 5;
    NDArray a = NDArray::empty({m, k}, DType::Complex64);
    NDArray b = NDArray::empty({k, n}, DType::Complex64);
    auto* ap = reinterpret_cast<std::complex<float>*>(a.data());
    auto* bp = reinterpret_cast<std::complex<float>*>(b.data());
    for (std::int64_t i = 0; i < m * k; ++i) ap[i] = {static_cast<float>(std::sin(0.7 * double(i))), static_cast<float>(std::cos(1.3 * double(i)))};
    for (std::int64_t i = 0; i < k * n; ++i) bp[i] = {static_cast<float>(std::cos(0.4 * double(i))), static_cast<float>(std::sin(2.1 * double(i)))};
    const NDArray r = matmul(a, b);
    bool close = true;
    for (std::int64_t i = 0; i < m; ++i)
      for (std::int64_t j = 0; j < n; ++j) {
        C ref{};
        for (std::int64_t p = 0; p < k; ++p) ref += C(ap[i * k + p]) * C(bp[p * n + j]);
        const C got(cat<float>(r, i * n + j));
        close = close && std::abs(got - ref) <= 2e-5 * double(k);
      }
    CHECK(close);
  });
}

TEST_CASE("linalg: complex matmul dispatch branches (D-036)") {
  using C = std::complex<double>;
  // Each (m, k, n) hits one branch of NumPy's matmul loop selection: dotu,
  // scalar_vec, gemv (vector @ matrix and matrix @ vector), column @ row, gemm.
  const std::int64_t dims[][3] = {{1, 5, 1}, {1, 1, 4}, {3, 1, 1}, {1, 4, 3},
                                  {3, 4, 1}, {3, 1, 2}, {3, 4, 2}, {2, 6, 5}};
  each_backend([&dims] {
    for (const auto& d : dims) {
      const std::int64_t m = d[0], k = d[1], n = d[2];
      NDArray a = NDArray::empty({m, k}, DType::Complex128);
      NDArray b = NDArray::empty({k, n}, DType::Complex128);
      auto* ap = reinterpret_cast<C*>(a.data());
      auto* bp = reinterpret_cast<C*>(b.data());
      // Small integers: every product and partial sum is exact, so any
      // summation order gives the same bits.
      for (std::int64_t i = 0; i < m * k; ++i) ap[i] = C(double(i % 5) - 2, double(i % 3) - 1);
      for (std::int64_t i = 0; i < k * n; ++i) bp[i] = C(double(i % 4) - 1, double(i % 7) - 3);
      std::vector<C> ref(static_cast<std::size_t>(m * n));
      noblas_cgemm<double>(m, n, k, ap, bp, ref.data());
      const NDArray r = matmul(a, b);
      bool same = r.shape() == Shape({m, n});
      for (std::int64_t i = 0; same && i < m * n; ++i) same = cat<double>(r, i) == ref[static_cast<std::size_t>(i)];
      CHECK(same);
      // complex64 takes the same branches.
      const NDArray r32 = matmul(a.astype(DType::Complex64), b.astype(DType::Complex64));
      bool same32 = r32.dtype() == DType::Complex64;
      for (std::int64_t i = 0; same32 && i < m * n; ++i)
        same32 = C(cat<float>(r32, i)) == ref[static_cast<std::size_t>(i)];
      CHECK(same32);
    }
    // inf in a 2×2 product: NumPy (cblas_zgemm and the non-BLAS loop) gives
    // inf+nanj at [0,0], i.e. inf*1 + 1*1 without Annex G recovery.
    const double inf = std::numeric_limits<double>::infinity();
    const C g = cat<double>(matmul(cmat<double>(2, 2, {C(inf, 0), C(1, 0), C(1, 0), C(1, 0)}),
                                   cmat<double>(2, 2, {C(1, 0), C(1, 0), C(1, 0), C(1, 0)})),
                            0);
    CHECK(g.real() == inf);
    CHECK(std::isnan(g.imag()));
  });
}



TEST_CASE("linalg: complex det (D-038)") {
  using C = std::complex<double>;
  const auto near_c = [](C got, C want, double tol) {
    return std::abs(got - want) <= tol * (1.0 + std::abs(want));
  };
  each_backend([&near_c] {
    const auto run = [&near_c](auto tag, double tol) {
      using R = decltype(tag);
      const DType cdt = std::is_same_v<R, float> ? DType::Complex64 : DType::Complex128;
      // (1+2j)*2 - (3-j)*0.5j = 1.5+2.5j (NumPy: 1.5000000000000002+2.5j).
      const NDArray d = det(cmat<R>(2, 2, {C(1, 2), C(3, -1), C(0, 0.5), C(2, 0)}));
      CHECK(d.dtype() == cdt);
      CHECK(d.shape() == Shape{});
      CHECK(near_c(C(cat<R>(d, 0)), C(1.5, 2.5), tol));
      // Needs a row swap: [[0, j], [2, 3]] -> -2j.
      CHECK(near_c(C(cat<R>(det(cmat<R>(2, 2, {C(0, 0), C(0, 1), C(2, 0), C(3, 0)})), 0)), C(0, -2), tol));
      // 3x3, NumPy: 5.999999999999999-2.4556958919860773e-16j.
      const NDArray d3 = det(cmat<R>(3, 3, {C(2, 0), C(0, 1), C(0, 0), C(1, -1), C(3, 0), C(1, 0),
                                            C(0, 0), C(0, 2), C(1, 1)}));
      CHECK(near_c(C(cat<R>(d3, 0)), C(6, 0), tol));
      // Singular (exact zero pivot): sign 0, logdet -inf -> exactly 0+0j.
      const C z(cat<R>(det(cmat<R>(2, 2, {C(0, 0), C(1, 0), C(0, 0), C(0, 1)})), 0));
      CHECK(z.real() == 0.0 && z.imag() == 0.0);
      // 0x0 -> 1+0j; batched (2,2,2) of k*(1+j): both determinants are -4j.
      CHECK(C(cat<R>(det(NDArray::zeros({0, 0}, cdt)), 0)) == C(1, 0));
      NDArray b = NDArray::empty({2, 2, 2}, cdt);
      auto* bp = reinterpret_cast<std::complex<R>*>(b.data());
      for (int i = 0; i < 8; ++i) bp[i] = {static_cast<R>(i), static_cast<R>(i)};
      const NDArray bd = det(b);
      CHECK(bd.shape() == Shape({2}));
      CHECK(near_c(C(cat<R>(bd, 0)), C(0, -4), tol));
      CHECK(near_c(C(cat<R>(bd, 1)), C(0, -4), tol));
      // NaN input propagates to nan+nanj, as in NumPy.
      const double nan = std::numeric_limits<double>::quiet_NaN();
      const C dn(cat<R>(det(cmat<R>(2, 2, {C(nan, 0), C(1, 0), C(1, 0), C(1, 0)})), 0));
      CHECK(std::isnan(dn.real()) && std::isnan(dn.imag()));
      // Shape errors are unchanged.
      CHECK_THROWS_KIND(det(NDArray::zeros({2, 3}, cdt)), ErrorKind::LinAlg);
    };
    run(double{}, 1e-12);
    run(float{}, 1e-5);
    // Other decompositions still reject complex until their slice lands.
    CHECK_THROWS_KIND(lstsq(cmat<double>(1, 1, {C(1, 0)}), cmat<double>(1, 1, {C(1, 0)}), -1.0),
                      ErrorKind::NotImplemented);
  });
}

TEST_CASE("linalg: complex inv/solve (D-039)") {
  using C = std::complex<double>;
  const auto near_c = [](C got, C want, double tol) {
    return std::abs(got - want) <= tol * (1.0 + std::abs(want));
  };
  each_backend([&near_c] {
    const auto run = [&near_c](auto tag, double tol) {
      using R = decltype(tag);
      const DType cdt = std::is_same_v<R, float> ? DType::Complex64 : DType::Complex128;
      // A = [[1+2j, 3-j], [0.5j, 2]], det = 1.5+2.5j.
      // inv(A) = [[2, -3+j], [-0.5j, 1+2j]] / (1.5+2.5j).
      const NDArray a = cmat<R>(2, 2, {C(1, 2), C(3, -1), C(0, 0.5), C(2, 0)});
      const C d(1.5, 2.5);
      const C want[4] = {C(2, 0) / d, C(-3, 1) / d, C(0, -0.5) / d, C(1, 2) / d};
      const NDArray ai = inv(a);
      CHECK(ai.dtype() == cdt);
      CHECK(ai.shape() == Shape({2, 2}));
      for (int i = 0; i < 4; ++i) CHECK(near_c(C(cat<R>(ai, i)), want[i], tol));
      // solve with a vector b: x = inv(A) b.
      const NDArray bv = NDArray::empty({2}, cdt);
      auto* bp = reinterpret_cast<std::complex<R>*>(bv.data());
      bp[0] = {1, 0};
      bp[1] = {0, 1};
      const NDArray x = solve(a, bv);
      CHECK(x.dtype() == cdt);
      CHECK(x.shape() == Shape({2}));
      const C j(0, 1);
      CHECK(near_c(C(cat<R>(x, 0)), want[0] + want[1] * j, tol));
      CHECK(near_c(C(cat<R>(x, 1)), want[2] + want[3] * j, tol));
      // Matrix b, broadcast batch: (2,2,2) a stack against one (2,1) b.
      NDArray st = NDArray::empty({2, 2, 2}, cdt);
      auto* sp = reinterpret_cast<std::complex<R>*>(st.data());
      for (int i = 0; i < 4; ++i) sp[i] = cat<R>(a, i);
      sp[4] = {0, 1};  // [[j, 0], [0, 2]]
      sp[5] = {0, 0};
      sp[6] = {0, 0};
      sp[7] = {2, 0};
      const NDArray xb = solve(st, bv.reshape({2, 1}));
      CHECK(xb.shape() == Shape({2, 2, 1}));
      // Second system: [[j, 0], [0, 2]] x = [1, j] -> x = [-j, 0.5j].
      CHECK(near_c(C(cat<R>(xb, 2)), C(0, -1), tol));
      CHECK(near_c(C(cat<R>(xb, 3)), C(0, 0.5), tol));
      // Row swap needed: [[0, j], [2, 3]].
      const NDArray sw = inv(cmat<R>(2, 2, {C(0, 0), C(0, 1), C(2, 0), C(3, 0)}));
      CHECK(near_c(C(cat<R>(sw, 0)), C(0, 1.5), tol));
      CHECK(near_c(C(cat<R>(sw, 1)), C(0.5, 0), tol));
      CHECK(near_c(C(cat<R>(sw, 2)), C(0, -1), tol));
      CHECK(near_c(C(cat<R>(sw, 3)), C(0, 0), tol));
      // Singular -> LinAlgError; empty -> empty; shape errors unchanged.
      CHECK_THROWS_KIND(inv(cmat<R>(2, 2, {C(1, 1), C(2, 2), C(1, 1), C(2, 2)})), ErrorKind::LinAlg);
      CHECK_THROWS_KIND(solve(cmat<R>(2, 2, {C(1, 1), C(2, 2), C(1, 1), C(2, 2)}), bv), ErrorKind::LinAlg);
      CHECK(inv(NDArray::zeros({0, 0}, cdt)).shape() == Shape({0, 0}));
      CHECK(inv(NDArray::zeros({0, 0}, cdt)).dtype() == cdt);
      CHECK_THROWS_KIND(inv(NDArray::zeros({2, 3}, cdt)), ErrorKind::LinAlg);
      CHECK_THROWS_KIND(solve(a, NDArray::zeros({3}, cdt)), ErrorKind::Shape);
    };
    run(double{}, 1e-12);
    run(float{}, 1e-5);
    // Result dtype follows NumPy _commonType (D-039).
    const NDArray a64 = cmat<float>(2, 2, {C(1, 0), C(0, 1), C(0, -1), C(3, 0)});
    CHECK(solve(a64, mat(2, 1, {1, 2}, DType::Float32)).dtype() == DType::Complex64);
    CHECK(solve(a64, mat(2, 1, {1, 2}, DType::Float64)).dtype() == DType::Complex128);
    CHECK(solve(a64, mat(2, 1, {1, 2}, DType::Int8)).dtype() == DType::Complex128);
    CHECK(solve(mat(2, 2, {1, 0, 0, 1}, DType::Float32), a64).dtype() == DType::Complex64);
    CHECK_THROWS_KIND(solve(a64, mat(2, 1, {1, 2}, DType::Float16)), ErrorKind::DType);
    // NaN input does not raise (NumPy only flags gesv info > 0).
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const NDArray ni = inv(cmat<double>(2, 2, {C(nan, 0), C(1, 0), C(1, 0), C(1, 0)}));
    CHECK(std::isnan(cat<double>(ni, 0).real()));
  });
}

TEST_CASE("linalg: complex svd (D-041)") {
  using C = std::complex<double>;
  each_backend([] {
    const auto run = [](auto tag, double tol) {
      using R = decltype(tag);
      const DType cdt = std::is_same_v<R, float> ? DType::Complex64 : DType::Complex128;
      const DType rdt = std::is_same_v<R, float> ? DType::Float32 : DType::Float64;
      const auto sat = [](const NDArray& x, std::int64_t i) {
        return static_cast<double>(reinterpret_cast<const R*>(x.data())[i]);
      };
      const auto near = [tol](double got, double want) {
        return std::abs(got - want) <= tol * (1.0 + std::abs(want));
      };
      // A (3x2) = [[1+2j, 3-j], [0.5j, 2], [1, j]]; NumPy S = [4.352..., 1.5198...].
      const NDArray a = cmat<R>(3, 2, {C(1, 2), C(3, -1), C(0, 0.5), C(2, 0), C(1, 0), C(0, 1)});
      const double ws[2] = {4.352020701327292, 1.5198407203449662};
      for (const bool full : {true, false}) {
        const SvdResult r = svd(a, full, true);
        CHECK(r.u->dtype() == cdt);
        CHECK(r.vh->dtype() == cdt);
        CHECK(r.s.dtype() == rdt);
        const std::int64_t uc = full ? 3 : 2;
        CHECK(r.u->shape() == Shape({3, uc}));
        CHECK(r.vh->shape() == Shape({2, 2}));
        CHECK(near(sat(r.s, 0), ws[0]));
        CHECK(near(sat(r.s, 1), ws[1]));
        // U^H U = I.
        for (int i = 0; i < uc; ++i)
          for (int j = 0; j < uc; ++j) {
            C d{};
            for (int p = 0; p < 3; ++p)
              d += std::conj(C(cat<R>(*r.u, p * uc + i))) * C(cat<R>(*r.u, p * uc + j));
            CHECK(std::abs(d - (i == j ? C(1, 0) : C(0, 0))) <= 10 * tol);
          }
        // U[:, :2] diag(S) Vh reconstructs A.
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 2; ++j) {
            C acc{};
            for (int p = 0; p < 2; ++p)
              acc += C(cat<R>(*r.u, i * uc + p)) * sat(r.s, p) * C(cat<R>(*r.vh, p * 2 + j));
            CHECK(std::abs(acc - C(cat<R>(a, i * 2 + j))) <= 10 * tol);
          }
      }
      // Wide input Aᴴ (2x3): same S, reconstructs.
      const NDArray w = cmat<R>(2, 3, {C(1, -2), C(0, -0.5), C(1, 0), C(3, 1), C(2, 0), C(0, -1)});
      const SvdResult rw = svd(w, true, true);
      CHECK(rw.u->shape() == Shape({2, 2}));
      CHECK(rw.vh->shape() == Shape({3, 3}));
      CHECK(near(sat(rw.s, 0), ws[0]));
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j) {
          C acc{};
          for (int p = 0; p < 2; ++p)
            acc += C(cat<R>(*rw.u, i * 2 + p)) * sat(rw.s, p) * C(cat<R>(*rw.vh, p * 3 + j));
          CHECK(std::abs(acc - C(cat<R>(w, i * 3 + j))) <= 10 * tol);
        }
      const SvdResult nv = svd(a, true, false);
      CHECK(!nv.u.has_value());
      CHECK(!nv.vh.has_value());
      CHECK(nv.s.dtype() == rdt);
      CHECK(near(sat(nv.s, 1), ws[1]));
      // [[j]] -> S = [1]; batched [[1j,0],[0,2]], [[1,1j],[1j,1]] -> [2,1], [√2,√2].
      CHECK(near(sat(svd(cmat<R>(1, 1, {C(0, 1)}), true, false).s, 0), 1.0));
      NDArray b = NDArray::empty({2, 2, 2}, cdt);
      auto* bp = reinterpret_cast<std::complex<R>*>(b.data());
      const C bv[8] = {C(0, 1), C(0, 0), C(0, 0), C(2, 0), C(1, 0), C(0, 1), C(0, 1), C(1, 0)};
      for (int i = 0; i < 8; ++i) bp[i] = {static_cast<R>(bv[i].real()), static_cast<R>(bv[i].imag())};
      const SvdResult bs = svd(b, false, false);
      CHECK(bs.s.shape() == Shape({2, 2}));
      CHECK(near(sat(bs.s, 0), 2.0));
      CHECK(near(sat(bs.s, 1), 1.0));
      CHECK(near(sat(bs.s, 2), std::sqrt(2.0)));
      CHECK(near(sat(bs.s, 3), std::sqrt(2.0)));
      // Rank-deficient: U stays unitary.
      const SvdResult rd = svd(cmat<R>(2, 2, {C(1, 1), C(1, 1), C(1, 1), C(1, 1)}), true, true);
      CHECK(near(sat(rd.s, 0), 2.0 * std::sqrt(2.0)));
      CHECK(std::abs(sat(rd.s, 1)) <= 10 * tol);
      C d01{};
      for (int p = 0; p < 2; ++p) d01 += std::conj(C(cat<R>(*rd.u, p * 2))) * C(cat<R>(*rd.u, p * 2 + 1));
      CHECK(std::abs(d01) <= 10 * tol);
      // Empty: (3,0) full -> U = I3 (complex), Vh (0,0), S (0,) real.
      const SvdResult e = svd(NDArray::zeros({3, 0}, cdt), true, true);
      CHECK(e.u->shape() == Shape({3, 3}));
      CHECK(e.u->dtype() == cdt);
      CHECK(C(cat<R>(*e.u, 4)) == C(1, 0));
      CHECK(e.vh->shape() == Shape({0, 0}));
      CHECK(e.s.dtype() == rdt);
      CHECK(svd(NDArray::zeros({0, 2}, cdt), false, false).s.shape() == Shape({0}));
      CHECK_THROWS_KIND(svd(NDArray::zeros({3}, cdt), true, true), ErrorKind::LinAlg);
    };
    run(double{}, 1e-12);
    run(float{}, 1e-5);
    // Non-finite input raises before LAPACK (NumPy svd_wrapper), real and complex.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();
    CHECK_THROWS_KIND(svd(cmat<double>(2, 2, {C(nan, 0), C(1, 0), C(1, 0), C(1, 0)}), true, true),
                      ErrorKind::LinAlg);
    CHECK_THROWS_KIND(svd(cmat<double>(1, 2, {C(1, 0), C(0, inf)}), true, false), ErrorKind::LinAlg);
    NDArray rn = NDArray::zeros({2, 2}, DType::Float64);
    reinterpret_cast<double*>(rn.data())[3] = nan;
    CHECK_THROWS_KIND(svd(rn, false, false), ErrorKind::LinAlg);
  });
}

TEST_CASE("linalg: complex eigh/eigvalsh (D-042)") {
  using C = std::complex<double>;
  each_backend([] {
    const auto run = [](auto tag, double tol) {
      using R = decltype(tag);
      const DType cdt = std::is_same_v<R, float> ? DType::Complex64 : DType::Complex128;
      const DType rdt = std::is_same_v<R, float> ? DType::Float32 : DType::Float64;
      const auto wat = [](const NDArray& x, std::int64_t i) {
        return static_cast<double>(reinterpret_cast<const R*>(x.data())[i]);
      };
      const auto near = [tol](double got, double want) {
        return std::abs(got - want) <= tol * (1.0 + std::abs(want));
      };
      // H = [[2, 1-j], [1+j, 3]] -> eigenvalues [1, 4].
      const NDArray h = cmat<R>(2, 2, {C(2, 0), C(1, -1), C(1, 1), C(3, 0)});
      const EigResult e = eigh(h);
      CHECK(e.eigenvalues.dtype() == rdt);
      CHECK(e.eigenvectors.dtype() == cdt);
      CHECK(e.eigenvalues.shape() == Shape({2}));
      CHECK(e.eigenvectors.shape() == Shape({2, 2}));
      CHECK(near(wat(e.eigenvalues, 0), 1.0));
      CHECK(near(wat(e.eigenvalues, 1), 4.0));
      const NDArray vh = eigvalsh(h);
      CHECK(vh.dtype() == rdt);
      CHECK(near(wat(vh, 0), 1.0));
      CHECK(near(wat(vh, 1), 4.0));
      // Only the lower triangle is read: upper garbage and diagonal imag ignored.
      const NDArray g = cmat<R>(2, 2, {C(2, 5), C(9, 9), C(1, 1), C(3, -7)});
      const NDArray gw = eigvalsh(g);
      CHECK(near(wat(gw, 0), 1.0));
      CHECK(near(wat(gw, 1), 4.0));
      // 3x3 Hermitian: A V = V diag(w), V^H V = I.
      const NDArray a = cmat<R>(3, 3, {C(4, 0), C(1, -2), C(0, 1), C(1, 2), C(3, 0), C(2, -1),
                                       C(0, -1), C(2, 1), C(5, 0)});
      const EigResult r = eigh(a);
      for (int i = 0; i < 2; ++i) CHECK(wat(r.eigenvalues, i) <= wat(r.eigenvalues, i + 1));
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
          C av{};
          C vv{};
          for (int p = 0; p < 3; ++p) {
            av += C(cat<R>(a, i * 3 + p)) * C(cat<R>(r.eigenvectors, p * 3 + j));
            vv += std::conj(C(cat<R>(r.eigenvectors, p * 3 + i))) *
                  C(cat<R>(r.eigenvectors, p * 3 + j));
          }
          const C want = C(cat<R>(r.eigenvectors, i * 3 + j)) * wat(r.eigenvalues, j);
          CHECK(std::abs(av - want) <= 20 * tol);
          CHECK(std::abs(vv - (i == j ? C(1, 0) : C(0, 0))) <= 10 * tol);
        }
      // Batched (2,2,2): [[1,0],[0,-2]] and [[0,j],[-j,0]] -> [-2,1], [-1,1].
      NDArray b = NDArray::empty({2, 2, 2}, cdt);
      auto* bp = reinterpret_cast<std::complex<R>*>(b.data());
      const C bv[8] = {C(1, 0), C(0, 0), C(0, 0), C(-2, 0), C(0, 0), C(0, 1), C(0, -1), C(0, 0)};
      for (int i = 0; i < 8; ++i) bp[i] = {static_cast<R>(bv[i].real()), static_cast<R>(bv[i].imag())};
      const NDArray bw = eigvalsh(b);
      CHECK(bw.shape() == Shape({2, 2}));
      CHECK(near(wat(bw, 0), -2.0));
      CHECK(near(wat(bw, 1), 1.0));
      CHECK(near(wat(bw, 2), -1.0));
      CHECK(near(wat(bw, 3), 1.0));
      // Empty 0x0 and errors.
      const EigResult z = eigh(NDArray::zeros({0, 0}, cdt));
      CHECK(z.eigenvalues.shape() == Shape({0}));
      CHECK(z.eigenvalues.dtype() == rdt);
      CHECK(z.eigenvectors.dtype() == cdt);
      CHECK_THROWS_KIND(eigh(NDArray::zeros({2, 3}, cdt)), ErrorKind::LinAlg);
      CHECK_THROWS_KIND(eigvalsh(NDArray::zeros({3}, cdt)), ErrorKind::LinAlg);
    };
    run(double{}, 1e-12);
    run(float{}, 1e-5);
  });
}

TEST_CASE("linalg: complex eig/eigvals (D-043)") {
  using C = std::complex<double>;
  each_backend([] {
    const auto run = [](auto tag, double tol) {
      using R = decltype(tag);
      const DType cdt = std::is_same_v<R, float> ? DType::Complex64 : DType::Complex128;
      // Upper triangular, eigenvalues 1+j, 2-j, -1 (any order).
      const NDArray t = cmat<R>(3, 3, {C(1, 1), C(2, 0), C(0, 0), C(0, 0), C(2, -1), C(0, 1),
                                       C(0, 0), C(0, 0), C(-1, 0)});
      const C want[3] = {C(1, 1), C(2, -1), C(-1, 0)};
      const auto has = [&](const NDArray& w) {
        for (const C& x : want) {
          bool found = false;
          for (int i = 0; i < 3; ++i) found = found || std::abs(C(cat<R>(w, i)) - x) <= tol;
          if (!found) return false;
        }
        return true;
      };
      const EigResult e = eig(t);
      CHECK(e.eigenvalues.dtype() == cdt);
      CHECK(e.eigenvectors.dtype() == cdt);
      CHECK(e.eigenvalues.shape() == Shape({3}));
      CHECK(e.eigenvectors.shape() == Shape({3, 3}));
      CHECK(has(e.eigenvalues));
      const NDArray ev = eigvals(t);
      CHECK(ev.dtype() == cdt);
      CHECK(has(ev));
      // Dense non-normal matrix: A v_j = w_j v_j, |v_j| = 1; eigvals agree.
      const NDArray a = cmat<R>(3, 3, {C(1, 2), C(3, -1), C(0, 1), C(-2, 0), C(1, 1), C(2, 3),
                                       C(0.5, 0), C(-1, 2), C(4, -2)});
      const EigResult r = eig(a);
      const NDArray rv = eigvals(a);
      for (int j = 0; j < 3; ++j) {
        double nrm = 0;
        bool found = false;
        for (int k = 0; k < 3; ++k)
          found = found || std::abs(C(cat<R>(rv, k)) - C(cat<R>(r.eigenvalues, j))) <= 50 * tol;
        CHECK(found);
        for (int i = 0; i < 3; ++i) {
          C av{};
          for (int p = 0; p < 3; ++p) av += C(cat<R>(a, i * 3 + p)) * C(cat<R>(r.eigenvectors, p * 3 + j));
          const C vij = C(cat<R>(r.eigenvectors, i * 3 + j));
          CHECK(std::abs(av - vij * C(cat<R>(r.eigenvalues, j))) <= 50 * tol);
          nrm += std::norm(vij);
        }
        CHECK(std::abs(nrm - 1.0) <= 10 * tol);
      }
      // Batched (2,2,2): diag(1, j) and [[0,1],[-1,0]] (eigenvalues ±j).
      NDArray b = NDArray::empty({2, 2, 2}, cdt);
      auto* bp = reinterpret_cast<std::complex<R>*>(b.data());
      const C bv[8] = {C(1, 0), C(0, 0), C(0, 0), C(0, 1), C(0, 0), C(1, 0), C(-1, 0), C(0, 0)};
      for (int i = 0; i < 8; ++i) bp[i] = {static_cast<R>(bv[i].real()), static_cast<R>(bv[i].imag())};
      const NDArray bw = eigvals(b);
      CHECK(bw.shape() == Shape({2, 2}));
      CHECK(std::abs(C(cat<R>(bw, 0)) + C(cat<R>(bw, 1)) - C(1, 1)) <= tol);
      CHECK(std::abs(C(cat<R>(bw, 2)) + C(cat<R>(bw, 3))) <= tol);
      CHECK(std::abs(std::abs(C(cat<R>(bw, 2)).imag()) - 1.0) <= tol);
      CHECK(eig(b).eigenvectors.shape() == Shape({2, 2, 2}));
      // Empty, non-finite and shape errors.
      const EigResult z = eig(NDArray::zeros({0, 0}, cdt));
      CHECK(z.eigenvalues.shape() == Shape({0}));
      CHECK(z.eigenvalues.dtype() == cdt);
      CHECK(eigvals(NDArray::zeros({0, 0}, cdt)).dtype() == cdt);
      const double nan = std::numeric_limits<double>::quiet_NaN();
      CHECK_THROWS_KIND(eig(cmat<R>(2, 2, {C(0, nan), C(0, 0), C(0, 0), C(1, 0)})), ErrorKind::LinAlg);
      CHECK_THROWS_KIND(eigvals(cmat<R>(1, 1, {C(INFINITY, 0)})), ErrorKind::LinAlg);
      CHECK_THROWS_KIND(eig(NDArray::zeros({2, 3}, cdt)), ErrorKind::LinAlg);
      CHECK_THROWS_KIND(eigvals(NDArray::zeros({3}, cdt)), ErrorKind::LinAlg);
    };
    run(double{}, 1e-12);
    run(float{}, 1e-5);
    // Real eigvals (values-only path): same values as eig, complex dtype.
    const NDArray rot = mat(2, 2, {0, -1, 1, 0});
    const NDArray rw = eigvals(rot);
    CHECK(rw.dtype() == DType::Complex128);
    CHECK(std::abs(std::abs(cat<double>(rw, 0).imag()) - 1.0) <= 1e-12);
    CHECK(eigvals(rot.astype(DType::Float32)).dtype() == DType::Complex64);
  });
}


TEST_CASE("linalg: complex qr (D-040)") {
  using C = std::complex<double>;
  const auto near_c = [](C got, C want, double tol) {
    return std::abs(got - want) <= tol * (1.0 + std::abs(want));
  };
  each_backend([&near_c] {
    const auto run = [&near_c](auto tag, double tol) {
      using R = decltype(tag);
      const DType cdt = std::is_same_v<R, float> ? DType::Complex64 : DType::Complex128;
      // A (3x2) = [[1+2j, 3-j], [0.5j, 2], [1, j]]; NumPy reference values.
      const NDArray a = cmat<R>(3, 2, {C(1, 2), C(3, -1), C(0, 0.5), C(2, 0), C(1, 0), C(0, 1)});
      const QrResult red = qr(a, QrMode::Reduced);
      CHECK(red.q->dtype() == cdt);
      CHECK(red.r.dtype() == cdt);
      CHECK(red.q->shape() == Shape({3, 2}));
      CHECK(red.r.shape() == Shape({2, 2}));
      const C wq[6] = {C(-0.4, -0.8), C(-2.26778684e-01, 7.55928946e-02),
                       C(0, -0.2), C(-5.44268841e-01, 3.02371578e-02),
                       C(-0.4, 0), C(6.04743157e-02, -8.01284683e-01)};
      const C wr[4] = {C(-2.5, 0), C(-0.4, 2.8), C(0, 0), C(-2.6457513110645907, 0)};
      for (int i = 0; i < 6; ++i) CHECK(near_c(C(cat<R>(*red.q, i)), wq[i], std::max(tol, 1e-8)));
      for (int i = 0; i < 4; ++i) CHECK(near_c(C(cat<R>(red.r, i)), wr[i], tol));
      // R's diagonal is exactly real (zlarfg) and the lower triangle is zero.
      CHECK(cat<R>(red.r, 0).imag() == R{0});
      CHECK(cat<R>(red.r, 3).imag() == R{0});
      CHECK(cat<R>(red.r, 2) == std::complex<R>{});
      // Complete: Q is 3x3 unitary (Q^H Q = I), R is 3x2 with a zero last row.
      const QrResult cq = qr(a, QrMode::Complete);
      CHECK(cq.q->shape() == Shape({3, 3}));
      CHECK(cq.r.shape() == Shape({3, 2}));
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
          C dot{};
          for (int p = 0; p < 3; ++p)
            dot += std::conj(C(cat<R>(*cq.q, p * 3 + i))) * C(cat<R>(*cq.q, p * 3 + j));
          CHECK(near_c(dot, i == j ? C(1, 0) : C(0, 0), tol));
        }
      }
      // Q R reconstructs A.
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
          C s{};
          for (int p = 0; p < 3; ++p) s += C(cat<R>(*cq.q, i * 3 + p)) * C(cat<R>(cq.r, p * 2 + j));
          CHECK(near_c(s, C(cat<R>(a, i * 2 + j)), tol));
        }
      }
      // Mode 'r' on the wide transpose; 1x1 [[j]] -> Q = [[-j]], R = [[-1]].
      const QrResult rr = qr(transpose(a, {1, 0}), QrMode::R);
      CHECK(!rr.q.has_value());
      CHECK(rr.r.shape() == Shape({2, 3}));
      CHECK(near_c(C(cat<R>(rr.r, 0)), C(-3.872983346207417, 0), tol));
      const QrResult one = qr(cmat<R>(1, 1, {C(0, 1)}), QrMode::Reduced);
      CHECK(near_c(C(cat<R>(*one.q, 0)), C(0, -1), tol));
      CHECK(near_c(C(cat<R>(one.r, 0)), C(-1, 0), tol));
      // Batched input factors each matrix separately.
      NDArray st = NDArray::empty({2, 1, 1}, cdt);
      auto* sp = reinterpret_cast<std::complex<R>*>(st.data());
      sp[0] = {0, 1};
      sp[1] = {2, 0};
      const QrResult bq = qr(st, QrMode::Reduced);
      CHECK(bq.q->shape() == Shape({2, 1, 1}));
      CHECK(C(cat<R>(bq.r, 1)) == C(2, 0));  // real 1x1: tau = 0 (zlarfg)
      CHECK(C(cat<R>(*bq.q, 1)) == C(1, 0));
      // Empty shapes: (0,3) reduced -> Q (0,0), R (0,3); (3,0) complete -> Q = I3.
      const QrResult e0 = qr(NDArray::zeros({0, 3}, cdt), QrMode::Reduced);
      CHECK(e0.q->shape() == Shape({0, 0}));
      CHECK(e0.r.shape() == Shape({0, 3}));
      const QrResult e1 = qr(NDArray::zeros({3, 0}, cdt), QrMode::Complete);
      CHECK(e1.q->dtype() == cdt);
      CHECK(C(cat<R>(*e1.q, 4)) == C(1, 0));
      CHECK(C(cat<R>(*e1.q, 1)) == C(0, 0));
      CHECK_THROWS_KIND(qr(NDArray::zeros({3}, cdt), QrMode::Reduced), ErrorKind::LinAlg);
    };
    run(double{}, 1e-12);
    run(float{}, 1e-5);
    // NaN input does not raise (NumPy returns NaN factors).
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const QrResult nq = qr(cmat<double>(2, 2, {C(nan, 0), C(1, 0), C(1, 0), C(1, 0)}), QrMode::Reduced);
    CHECK(std::isnan(cat<double>(nq.r, 0).real()));
  });
}

TEST_CASE("linalg: dot/inner/outer") {
  const NDArray a = arange(0, 24, 1, DType::Float64).reshape({2, 3, 4});
  const NDArray b = arange(0, 24, 1, DType::Float64).reshape({3, 4, 2});
  const NDArray d = dot(a, b);
  CHECK(d.shape() == Shape({2, 3, 3, 2}));
  // d[1,2,0,1] = sum_k a[1,2,k] * b[0,k,1]
  double ref = 0;
  for (int k = 0; k < 4; ++k) ref += (12 + 8 + k) * (k * 2 + 1);
  CHECK(near(d.get_double(((1 * 3 + 2) * 3 + 0) * 2 + 1), ref));
  const NDArray in = inner(a, arange(0, 8, 1, DType::Float64).reshape({2, 4}));
  CHECK(in.shape() == Shape({2, 3, 2}));
  CHECK(near(in.get_double(1), 0 * 4 + 1 * 5 + 2 * 6 + 3 * 7));
  const NDArray o = outer(arange(1, 3, 1, DType::Int64), arange(1, 4, 1, DType::Int64));
  CHECK(o.shape() == Shape({2, 3}));
  CHECK_EQ(o.get_int64(5), 6);
}

TEST_CASE("linalg: det/inv/solve") {
  each_backend([] {
    const NDArray a = mat(3, 3, {2, -1, 0, -1, 2, -1, 0, -1, 2});
    CHECK(near(det(a).get_double(0), 4.0));
    const NDArray ai = inv(a);
    CHECK(all_near(matmul(a, ai), eye(3, 3, 0, DType::Float64), 1e-12));
    const NDArray b = mat(3, 1, {1, 0, 1});
    const NDArray x = solve(a, b);
    CHECK(x.shape() == Shape({3, 1}));
    CHECK(all_near(matmul(a, x), b, 1e-12));
    const NDArray xv = solve(a, arange(1, 4, 1, DType::Int64));
    CHECK(xv.shape() == Shape({3}));
    CHECK(near(det(mat(2, 2, {1, 2, 2, 4})).get_double(0), 0.0));
    CHECK_THROWS_KIND(inv(mat(2, 2, {1, 2, 2, 4})), ErrorKind::LinAlg);
    CHECK_THROWS_KIND(det(arange(0, 6, 1, DType::Float64).reshape({2, 3})), ErrorKind::LinAlg);
    CHECK_THROWS_KIND(det(arange(0, 3, 1, DType::Float64)), ErrorKind::LinAlg);
    CHECK_THROWS_KIND(det(mat(1, 1, {1}, DType::Float16)), ErrorKind::DType);
    CHECK(det(mat(2, 2, {1, 2, 3, 4}, DType::Float32)).dtype() == DType::Float32);
    CHECK(det(mat(2, 2, {1, 2, 3, 4}, DType::Int64)).dtype() == DType::Float64);
    // Batched det; empty matrix det == 1.
    const NDArray bd = det(arange(0, 8, 1, DType::Float64).reshape({2, 2, 2}));
    CHECK(bd.shape() == Shape({2}));
    CHECK(near(bd.get_double(1), 4 * 7 - 5 * 6.0));
    CHECK(near(det(NDArray::zeros({0, 0}, DType::Float64)).get_double(0), 1.0));
  });
}

TEST_CASE("linalg: eigh/eig reconstruct") {
  each_backend([] {
    const NDArray s = mat(3, 3, {4, 1, 2, 1, 3, 0, 2, 0, 5});
    const EigResult e = eigh(s);
    for (int i = 0; i + 1 < 3; ++i) CHECK(e.eigenvalues.get_double(i) <= e.eigenvalues.get_double(i + 1));
    // A V = V diag(w)
    const NDArray av = matmul(s, e.eigenvectors);
    for (std::int64_t i = 0; i < 3; ++i)
      for (std::int64_t j = 0; j < 3; ++j)
        CHECK(near(av.get_double(i * 3 + j),
                   e.eigenvectors.get_double(i * 3 + j) * e.eigenvalues.get_double(j), 1e-9));
    // Rotation: eigenvalues ±i.
    const EigResult g = eig(mat(2, 2, {0, -1, 1, 0}));
    CHECK(g.eigenvalues.dtype() == DType::Complex128);
    const auto* w = reinterpret_cast<const std::complex<double>*>(g.eigenvalues.data());
    const auto* v = reinterpret_cast<const std::complex<double>*>(g.eigenvectors.data());
    CHECK(near(std::abs(w[0].imag()), 1.0, 1e-9));
    CHECK(near(w[0].imag(), -w[1].imag(), 1e-9));
    const double ar[4] = {0, -1, 1, 0};
    for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i) {
        const std::complex<double> lhs = ar[i * 2] * v[0 * 2 + j] + ar[i * 2 + 1] * v[1 * 2 + j];
        CHECK(std::abs(lhs - w[j] * v[i * 2 + j]) < 1e-9);
      }
    CHECK_THROWS_KIND(eig(mat(2, 2, {NAN, 0, 0, 1})), ErrorKind::LinAlg);
  });
}

TEST_CASE("linalg: svd/qr/lstsq/norm") {
  each_backend([] {
    const NDArray a = mat(3, 2, {1, 2, 3, 4, 5, 6});
    const SvdResult r = svd(a, false, true);
    CHECK(r.u->shape() == Shape({3, 2}));
    CHECK(r.vh->shape() == Shape({2, 2}));
    CHECK(r.s.get_double(0) >= r.s.get_double(1));
    NDArray us = r.u->copy();
    for (std::int64_t i = 0; i < 3; ++i)
      for (std::int64_t j = 0; j < 2; ++j) us.set_double(i * 2 + j, us.get_double(i * 2 + j) * r.s.get_double(j));
    CHECK(all_near(matmul(us, *r.vh), a, 1e-10));
    const SvdResult f = svd(a, true, true);
    CHECK(f.u->shape() == Shape({3, 3}));
    CHECK(all_near(matmul(transpose2(*f.u), *f.u), eye(3, 3, 0, DType::Float64), 1e-10));
    CHECK(!svd(a, true, false).u.has_value());

    const QrResult q = qr(a, QrMode::Reduced);
    CHECK(q.q->shape() == Shape({3, 2}));
    CHECK(q.r.shape() == Shape({2, 2}));
    CHECK(near(q.r.get_double(2), 0.0));
    CHECK(all_near(matmul(*q.q, q.r), a, 1e-10));
    const QrResult qc = qr(a, QrMode::Complete);
    CHECK(qc.q->shape() == Shape({3, 3}));
    CHECK(all_near(matmul(*qc.q, qc.r), a, 1e-10));
    const QrResult wide = qr(transpose2(a), QrMode::Reduced);
    CHECK(all_near(matmul(*wide.q, wide.r), transpose2(a), 1e-10));

    const LstsqResult l = lstsq(a, arange(1, 4, 1, DType::Float64), -1);
    CHECK_EQ(l.rank, 2);
    CHECK(l.x.shape() == Shape({2}));
    CHECK(near(l.x.get_double(0), 0.0, 1e-9));
    CHECK(near(l.x.get_double(1), 0.5, 1e-9));
    CHECK(l.residuals.shape() == Shape({1}));

    CHECK(near(norm(a, {}, std::nullopt, false).get_double(0), std::sqrt(91.0)));
    CHECK(near(norm(a, {"p", 1}, std::nullopt, false).get_double(0), 12.0));
    CHECK(near(norm(a, {"p", INFINITY}, std::nullopt, false).get_double(0), 11.0));
    CHECK(near(norm(a, {"nuc", 0}, std::nullopt, false).get_double(0),
               r.s.get_double(0) + r.s.get_double(1), 1e-10));
    CHECK(near(norm(a, {"p", 2}, std::nullopt, false).get_double(0), r.s.get_double(0), 1e-10));
    const NDArray n1 = norm(a, {"p", 1}, std::vector<std::int64_t>{1}, true);
    CHECK(n1.shape() == Shape({3, 1}));
    CHECK(near(n1.get_double(2), 11.0));
    CHECK_THROWS_KIND(norm(arange(0, 3, 1, DType::Float64), {"fro", 0}, std::nullopt, false),
                      ErrorKind::Value);
  });
}
