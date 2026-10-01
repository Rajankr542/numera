#include <cmath>
#include <complex>
#include <limits>

#include "complex_kernels.hpp"
#include "error.hpp"
#include "ndarray.hpp"
#include "reduce.hpp"
#include "test_harness.hpp"
#include "ufunc.hpp"

using namespace nativpy;

namespace {
using C = std::complex<double>;
constexpr double kInf = std::numeric_limits<double>::infinity();

bool same(double a, double b) {
  if (std::isnan(a) || std::isnan(b)) return std::isnan(a) && std::isnan(b);
  return a == b && std::signbit(a) == std::signbit(b);
}
bool same(C a, C b) { return same(a.real(), b.real()) && same(a.imag(), b.imag()); }

NDArray cvec(std::initializer_list<C> v) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, DType::Complex128);
  auto* p = reinterpret_cast<C*>(a.data());
  for (auto x : v) *p++ = x;
  return a;
}
C at(const NDArray& a, std::int64_t i) { return reinterpret_cast<const C*>(a.data())[i]; }
}  // namespace

TEST_CASE("complex kernels: division (Smith) and zero divisors") {
  CHECK(same(kernels::cdiv(C{1, 2}, C{2, -1}), C{0, 1}));
  const C z = kernels::cdiv(C{-3, 0.5}, C{0, 0});
  CHECK(same(z, C{-kInf, kInf}));
  CHECK(same(kernels::cdiv(C{1e300, 1e300}, C{1e300, -1e300}), C{0, 1}));
}

TEST_CASE("complex kernels: power special cases (npy_cpow)") {
  CHECK(same(kernels::cpow(C{5, 3}, C{0, 0}), C{1, 0}));
  CHECK(same(kernels::cpow(C{0, 0}, C{0, 0}), C{1, 0}));
  CHECK(same(kernels::cpow(C{0, 0}, C{2, 5}), C{0, 0}));
  const C n = kernels::cpow(C{0, 0}, C{-1, 0});
  CHECK(std::isnan(n.real()) && std::isnan(n.imag()));
  CHECK(same(kernels::cpow(C{1, 2}, C{2, 0}), C{-3, 4}));
  CHECK(same(kernels::cpow(C{1, 1}, C{-2, 0}), kernels::cdiv(C{1, 0}, C{0, 2})));
}

TEST_CASE("complex kernels: abs, sqrt and log special cases") {
  CHECK_EQ(kernels::cabs(C{3, 4}), 5.0);
  CHECK_EQ(kernels::cabs(C{kInf, std::nan("")}), kInf);
  CHECK(std::isnan(kernels::cabs(C{std::nan(""), 1})));
  CHECK_EQ(kernels::cabs(C{0, 0}), 0.0);
  CHECK(same(kernels::csqrt(C{-1, 0}), C{0, 1}));
  CHECK(same(kernels::csqrt(C{-4, -0.0}), C{0, -2}));
  CHECK(same(kernels::csqrt(C{0, -0.0}), C{0, -0.0}));
  CHECK(same(kernels::csqrt(C{-kInf, 1}), C{0, kInf}));
  CHECK(same(kernels::csqrt(C{1, kInf}), C{kInf, kInf}));
  CHECK(same(kernels::clog(C{0, 0}), C{-kInf, 0}));
  CHECK(same(kernels::clog(C{-1, 0}), C{0, std::atan2(0.0, -1.0)}));
  CHECK(same(kernels::clog(C{1, 0}), C{0, 0}));
}

TEST_CASE("complex ufuncs: dtypes, mod rejection and real/imag views") {
  CHECK(unary_result_dtype(UnaryOp::Abs, DType::Complex64) == DType::Float32);
  CHECK(unary_result_dtype(UnaryOp::Angle, DType::Complex128) == DType::Float64);
  CHECK(unary_result_dtype(UnaryOp::Conjugate, DType::Complex64) == DType::Complex64);
  CHECK(unary_result_dtype(UnaryOp::Conjugate, DType::Bool) == DType::Int8);
  CHECK_THROWS_KIND(binary_result_dtype(BinaryOp::Mod, DType::Complex128, DType::Float64),
                    ErrorKind::DType);
  NDArray a = cvec({C{1, 2}, C{-3, 0.5}});
  NDArray m = binary(BinaryOp::Multiply, a, a);
  CHECK(same(at(m, 0), C{-3, 4}));
  NDArray cj = unary(UnaryOp::Conjugate, a);
  CHECK(same(at(cj, 1), C{-3, -0.5}));
  NDArray re = complex_part(a, false);
  NDArray im = complex_part(a, true);
  CHECK(re.dtype() == DType::Float64 && im.strides() == Strides({16}));
  CHECK_EQ(im.get_double(1), 0.5);
  im.set_double(0, 9.0);
  CHECK(same(at(a, 0), C{1, 9}));
  NDArray flags = is_complex_elementwise(cvec({C{1, 0}, C{0, std::nan("")}}), true);
  CHECK_EQ(flags.get_int64(0), std::int64_t{0});
  CHECK_EQ(flags.get_int64(1), std::int64_t{1});
  NDArray ri = complex_part(NDArray::zeros({2}, DType::Int8), true);
  CHECK(!ri.writeable() && ri.dtype() == DType::Int8);
}

TEST_CASE("complex reductions: sum/prod (P1-3a)") {
  CHECK(reduce_result_dtype(ReduceOp::Sum, DType::Complex64) == DType::Complex64);
  CHECK(reduce_result_dtype(ReduceOp::Prod, DType::Complex128) == DType::Complex128);
  CHECK_THROWS_KIND(reduce_result_dtype(ReduceOp::Mean, DType::Complex128),
                    ErrorKind::NotImplemented);
  // [[1+2j, 3-1j, 0.5j], [2, -1j, 1+1j]]
  const NDArray m =
      cvec({C{1, 2}, C{3, -1}, C{0, 0.5}, C{2, 0}, C{0, -1}, C{1, 1}}).reshape({2, 3});
  ReduceOptions a0;
  a0.axis = std::vector<std::int64_t>{0};
  const NDArray s0 = reduce(ReduceOp::Sum, m, a0);
  CHECK(s0.dtype() == DType::Complex128 && s0.shape() == Shape({3}));
  CHECK(same(at(s0, 0), C{3, 2}) && same(at(s0, 2), C{1, 1.5}));
  const NDArray p0 = reduce(ReduceOp::Prod, m, a0);
  CHECK(same(at(p0, 0), C{2, 4}) && same(at(p0, 1), C{-1, -3}) && same(at(p0, 2), C{-0.5, 0.5}));
  ReduceOptions a1;
  a1.axis = std::vector<std::int64_t>{1};
  a1.initial = 1;
  const NDArray s1 = reduce(ReduceOp::Sum, m, a1);
  CHECK(same(at(s1, 0), C{5, 1.5}) && same(at(s1, 1), C{4, 0}));
  // Empty input gives the identity; an all -0.0 sum is +0.0 (NumPy).
  CHECK(same(at(reduce(ReduceOp::Prod, cvec({}), {}), 0), C{1, 0}));
  CHECK(same(at(reduce(ReduceOp::Sum, cvec({C{-0.0, -0.0}}), {}), 0), C{0, 0}));
  // NumPy's multiply formula: (inf+0j)^2 -> nan+nanj (no Annex G recovery).
  const C pi = at(reduce(ReduceOp::Prod, cvec({C{kInf, 0}, C{kInf, 0}}), {}), 0);
  CHECK(std::isnan(pi.real()) && std::isnan(pi.imag()));
  // Pairwise along the trailing axis, sequential along a leading one (NumPy).
  NDArray big = NDArray::empty({200}, DType::Complex128);
  auto* bp = reinterpret_cast<C*>(big.data());
  bp[0] = C{1e16, 1e16};
  for (int i = 1; i < 200; ++i) bp[i] = C{1, 1};
  CHECK(same(at(reduce(ReduceOp::Sum, big, {}), 0),
             C{1.0000000000000188e16, 1.0000000000000188e16}));
  CHECK(same(at(reduce(ReduceOp::Sum, big.reshape({200, 1}), a0), 0), C{1e16, 1e16}));
  // Real input with a complex64 dtype override.
  ReduceOptions d;
  d.dtype = DType::Complex64;
  const NDArray f = reduce(ReduceOp::Sum, NDArray::zeros({3}, DType::Int8), d);
  CHECK(f.dtype() == DType::Complex64 && f.ndim() == 0);
}

