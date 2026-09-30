#include <cmath>
#include <complex>
#include <cstdint>
#include <initializer_list>
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
