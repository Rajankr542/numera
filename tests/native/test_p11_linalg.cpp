#include <cmath>
#include <complex>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include "backend.hpp"
#include "creation.hpp"
#include "error.hpp"
#include "linalg.hpp"
#include "p11_linalg.hpp"
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

NDArray cmat(std::int64_t m, std::int64_t n, std::initializer_list<std::complex<double>> v) {
  NDArray a = NDArray::empty({m, n}, DType::Complex128);
  auto* p = reinterpret_cast<std::complex<double>*>(a.data());
  for (const auto& x : v) *p++ = x;
  return a;
}

std::complex<double> cat(const NDArray& a, std::int64_t i) {
  const NDArray c = a.astype(DType::Complex128);
  return reinterpret_cast<const std::complex<double>*>(c.data())[i];
}

bool near(double a, double b, double tol = 1e-9) {
  return std::abs(a - b) <= tol * (1.0 + std::abs(b));
}

bool all_near(const NDArray& a, std::initializer_list<double> v, double tol = 1e-9) {
  if (a.size() != static_cast<std::int64_t>(v.size())) return false;
  std::int64_t i = 0;
  for (const double x : v)
    if (!near(a.get_double(i++), x, tol)) return false;
  return true;
}

NDArray scal(double v) {
  NDArray a = NDArray::empty({}, DType::Float64);
  a.set_double(0, v);
  return a;
}

template <typename Fn>
void each_backend(Fn&& fn) {
  set_active_backend(&fallback_backend());
  fn();
  set_active_backend(&default_backend());
  fn();
  set_active_backend(nullptr);
}

}  // namespace

TEST_CASE("p11: cholesky lower/upper, complex, errors") {
  each_backend([] {
    const NDArray a = mat(2, 2, {4, 100, 2, 3});  // upper triangle ignored for lower
    CHECK(all_near(cholesky(a, false), {2, 0, 1, std::sqrt(2.0)}));
    const NDArray at = mat(2, 2, {4, 2, 100, 3});
    CHECK(all_near(cholesky(at, true), {2, 1, 0, std::sqrt(2.0)}));
    const NDArray f = cholesky(mat(2, 2, {4, 2, 2, 3}, DType::Float32), false);
    CHECK(f.dtype() == DType::Float32);
    CHECK(cholesky(mat(1, 1, {4}, DType::Int64), false).dtype() == DType::Float64);
    const NDArray c = cmat(2, 2, {{4, 0}, {1, -1}, {1, 1}, {3, 0}});
    const NDArray lo = cholesky(c, false);
    CHECK(near(cat(lo, 2).real(), 0.5) && near(cat(lo, 2).imag(), 0.5));
    CHECK(near(cat(lo, 3).real(), std::sqrt(2.5)));
    CHECK(cat(lo, 1) == std::complex<double>(0, 0));
    const NDArray up = cholesky(c, true);
    CHECK(near(cat(up, 1).real(), 0.5) && near(cat(up, 1).imag(), -0.5));
    CHECK(cat(up, 2) == std::complex<double>(0, 0));
    CHECK_THROWS_KIND(cholesky(mat(2, 2, {1, 2, 2, 1}), false), ErrorKind::LinAlg);
    CHECK_THROWS_KIND(cholesky(mat(2, 2, {std::nan(""), 0, 0, 1}), false), ErrorKind::LinAlg);
    CHECK_THROWS_KIND(cholesky(mat(2, 3, {1, 0, 0, 0, 1, 0}), false), ErrorKind::LinAlg);
    CHECK_THROWS_KIND(cholesky(mat(1, 1, {1}, DType::Float16), false), ErrorKind::DType);
    CHECK(cholesky(NDArray::zeros({0, 0}, DType::Float64), false).shape() == Shape({0, 0}));
    const NDArray batch = cholesky(eye(2, 2, 0, DType::Float64).reshape({1, 2, 2}), false);
    CHECK(batch.shape() == Shape({1, 2, 2}));
  });
}

TEST_CASE("p11: slogdet real, complex, singular, empty") {
  each_backend([] {
    auto r = slogdet(mat(2, 2, {1, 2, 3, 4}));
    CHECK(near(r.sign.get_double(0), -1.0));
    CHECK(near(r.logabsdet.get_double(0), std::log(2.0)));
    r = slogdet(mat(2, 2, {0, 0, 0, 0}));
    CHECK(r.sign.get_double(0) == 0.0);
    CHECK(std::isinf(r.logabsdet.get_double(0)) && r.logabsdet.get_double(0) < 0);
    r = slogdet(NDArray::zeros({2, 0, 0}, DType::Float64));
    CHECK(r.sign.shape() == Shape({2}));
    CHECK(r.sign.get_double(1) == 1.0 && r.logabsdet.get_double(1) == 0.0);
    r = slogdet(mat(2, 2, {1, 2, 3, 4}, DType::Float32));
    CHECK(r.sign.dtype() == DType::Float32 && r.logabsdet.dtype() == DType::Float32);
    const auto c = slogdet(cmat(2, 2, {{0, 1}, {2, 0}, {3, 0}, {4, 0}}));
    CHECK(c.sign.dtype() == DType::Complex128 && c.logabsdet.dtype() == DType::Float64);
    CHECK(near(cat(c.sign, 0).real(), -0.8320502943378437, 1e-12));
    CHECK(near(cat(c.sign, 0).imag(), 0.554700196225229, 1e-12));
    CHECK(near(c.logabsdet.get_double(0), 1.9756218592907138, 1e-12));
    const auto c64 = slogdet(cmat(1, 1, {{0, 2}}).astype(DType::Complex64));
    CHECK(c64.sign.dtype() == DType::Complex64 && c64.logabsdet.dtype() == DType::Float32);
    CHECK(near(cat(c64.sign, 0).imag(), 1.0, 1e-6));
    CHECK_THROWS_KIND(slogdet(NDArray::zeros({3}, DType::Float64)), ErrorKind::LinAlg);
  });
}

TEST_CASE("p11: matrix_power") {
  each_backend([] {
    const NDArray a = mat(2, 2, {1, 2, 3, 4}, DType::Int8);
    const NDArray p5 = matrix_power(a, 5);
    CHECK(p5.dtype() == DType::Int8);
    CHECK_EQ(p5.get_int64(0), 45);  // 1069 wraps
    CHECK_EQ(p5.get_int64(3), 78);
    const NDArray p0 = matrix_power(a, 0);
    CHECK(p0.dtype() == DType::Int8 && p0.get_int64(0) == 1 && p0.get_int64(1) == 0);
    const NDArray m1 = matrix_power(mat(2, 2, {1, 1, 0, 1}, DType::Int64), -1);
    CHECK(m1.dtype() == DType::Float64);
    CHECK(all_near(m1, {1, -1, 0, 1}));
    CHECK(all_near(matrix_power(mat(2, 2, {1, 1, 0, 1}), -6), {1, -6, 0, 1}));
    CHECK(all_near(matrix_power(mat(2, 2, {1, 1, 0, 1}), 9), {1, 9, 0, 1}));
    CHECK(all_near(matrix_power(mat(2, 2, {2, 0, 0, 3}), 1), {2, 0, 0, 3}));
    CHECK_THROWS_KIND(matrix_power(NDArray::zeros({2, 3}, DType::Float64), 2), ErrorKind::LinAlg);
    CHECK_THROWS_KIND(matrix_power(mat(2, 2, {1, 2, 2, 4}), -1), ErrorKind::LinAlg);
    CHECK(matrix_power(NDArray::zeros({3, 0, 0}, DType::Float64), 0).shape() == Shape({3, 0, 0}));
  });
}

TEST_CASE("p11: pinv, matrix_rank, cond") {
  each_backend([] {
    const NDArray rc = scal(1e-15);
    const NDArray p = pinv(mat(2, 2, {1, 2, 3, 4}), rc, false);
    CHECK(all_near(p, {-2, 1, 1.5, -0.5}, 1e-9));
    const NDArray ph = pinv(mat(2, 2, {2, 1, 1, 2}), rc, true);
    CHECK(all_near(ph, {2.0 / 3, -1.0 / 3, -1.0 / 3, 2.0 / 3}, 1e-9));
    const NDArray pr = pinv(mat(2, 3, {1, 0, 0, 0, 2, 0}), rc, false);
    CHECK(pr.shape() == Shape({3, 2}));
    CHECK(all_near(pr, {1, 0, 0, 0.5, 0, 0}, 1e-9));
    const NDArray pc = pinv(cmat(1, 1, {{0, 2}}), rc, false);
    CHECK(near(cat(pc, 0).imag(), -0.5));
    CHECK(pinv(NDArray::zeros({2, 0, 3}, DType::Float32), rc, false).shape() == Shape({2, 3, 0}));
    CHECK_THROWS_KIND(pinv(NDArray::zeros({3}, DType::Float64), rc, false), ErrorKind::LinAlg);

    CHECK_EQ(matrix_rank(mat(2, 2, {1, 2, 2, 4}), std::nullopt, std::nullopt, false).get_int64(0), 1);
    CHECK_EQ(matrix_rank(eye(3, 3, 0, DType::Float64), std::nullopt, std::nullopt, true).get_int64(0), 3);
    CHECK_EQ(matrix_rank(mat(2, 2, {1, 0, 0, 1e-3}), scal(1e-2),
                         std::nullopt, false).get_int64(0), 1);
    CHECK_EQ(matrix_rank(mat(2, 2, {1, 0, 0, 1e-3}), std::nullopt,
                         scal(1e-2), false).get_int64(0), 1);
    CHECK_EQ(matrix_rank(NDArray::zeros({3}, DType::Float64), std::nullopt, std::nullopt, false).get_int64(0), 0);
    CHECK_THROWS_KIND(matrix_rank(eye(2, 2, 0, DType::Float64), rc, rc, false), ErrorKind::Value);

    CHECK(near(cond(mat(2, 2, {1, 0, 0, 2}), NormOrd{}).get_double(0), 2.0));
    CHECK(near(cond(mat(2, 2, {1, 0, 0, 2}), NormOrd{"p", -2.0}).get_double(0), 0.5));
    CHECK(near(cond(mat(2, 2, {1, 2, 3, 4}), NormOrd{"fro", 0}).get_double(0), 15.0));
    CHECK(near(cond(mat(2, 2, {1, 2, 3, 4}), NormOrd{"p", 1.0}).get_double(0), 21.0));
    CHECK(near(cond(mat(2, 2, {1, 2, 3, 4}),
                    NormOrd{"p", std::numeric_limits<double>::infinity()}).get_double(0), 21.0));
    CHECK(std::isinf(cond(mat(2, 2, {1, 2, 2, 4}), NormOrd{"p", 1.0}).get_double(0)));
    CHECK(cond(mat(2, 2, {1, 0, 0, 2}, DType::Float32), NormOrd{}).dtype() == DType::Float32);
    CHECK_THROWS_KIND(cond(NDArray::zeros({0, 0}, DType::Float64), NormOrd{}), ErrorKind::LinAlg);
    CHECK_THROWS_KIND(cond(NDArray::zeros({2, 3}, DType::Float64), NormOrd{"p", 1.0}), ErrorKind::LinAlg);
  });
}

TEST_CASE("p11: einsum core") {
  const NDArray a = mat(2, 3, {1, 2, 3, 4, 5, 6});
  const NDArray b = mat(3, 2, {1, 0, 0, 1, 1, 1});
  const std::vector<EinsumStep> one{{{1, 0}, "ik"}};
  CHECK(all_near(einsum({a, b}, {"ij", "jk"}, one), {4, 5, 10, 11}));
  CHECK(all_near(einsum({a}, {"ij"}, {{{0}, "ji"}}), {1, 4, 2, 5, 3, 6}));
  CHECK(all_near(einsum({a}, {"ij"}, {{{0}, ""}}), {21}));
  const NDArray sq = mat(2, 2, {1, 2, 3, 4});
  CHECK(all_near(einsum({sq}, {"ii"}, {{{0}, "i"}}), {1, 4}));
  CHECK(all_near(einsum({sq}, {"ii"}, {{{0}, ""}}), {5}));
  CHECK(all_near(einsum({a, a}, {"ij", "ij"}, {{{1, 0}, "ij"}}), {1, 4, 9, 16, 25, 36}));
  // three operands in one step, and a two-step path
  const NDArray v = mat(1, 2, {1, 1}).reshape({2});
  CHECK(all_near(einsum({v, a, b}, {"i", "ij", "jk"}, {{{2, 1, 0}, "k"}}), {14, 16}));
  CHECK(all_near(einsum({v, a, b}, {"i", "ij", "jk"}, {{{2, 1}, "ik"}, {{1, 0}, "k"}}), {14, 16}));
  // broadcasting a size-1 dim, int wrap, bool
  CHECK(all_near(einsum({mat(1, 3, {1, 1, 1}), a}, {"ij", "ij"}, {{{1, 0}, "ij"}}), {1, 2, 3, 4, 5, 6}));
  const NDArray i8 = mat(1, 2, {100, 100}, DType::Int8).reshape({2});
  const NDArray r8 = einsum({i8}, {"i"}, {{{0}, ""}});
  CHECK(r8.dtype() == DType::Int8 && r8.get_int64(0) == -56);
  CHECK_THROWS_KIND(einsum({mat(2, 3, {1, 2, 3, 4, 5, 6})}, {"ii"}, {{{0}, "i"}}), ErrorKind::Value);
  CHECK_THROWS_KIND(einsum({a, sq}, {"ij", "jk"}, one), ErrorKind::Value);
  CHECK_THROWS_KIND(einsum({a, b}, {"ij", "jk"}, {{{1}, "jk"}}), ErrorKind::Value);
}
