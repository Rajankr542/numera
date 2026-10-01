#include <cmath>
#include <complex>
#include <limits>

#include "complex_kernels.hpp"
#include "creation.hpp"
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
  CHECK_THROWS_KIND(reduce_result_dtype(ReduceOp::Std, DType::Complex128),
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
  // Pairwise along the inner loop, sequential along a leading one. NumPy drops
  // size-1 axes first, so (200, 1) over axis 0 is still pairwise (verified).
  NDArray big = NDArray::empty({200}, DType::Complex128);
  auto* bp = reinterpret_cast<C*>(big.data());
  bp[0] = C{1e16, 1e16};
  for (int i = 1; i < 200; ++i) bp[i] = C{1, 1};
  const C pw{1.0000000000000188e16, 1.0000000000000188e16};
  CHECK(same(at(reduce(ReduceOp::Sum, big, {}), 0), pw));
  CHECK(same(at(reduce(ReduceOp::Sum, big.reshape({200, 1}), a0), 0), pw));
  NDArray big2 = NDArray::empty({200, 2}, DType::Complex128);
  auto* b2 = reinterpret_cast<C*>(big2.data());
  for (int i = 0; i < 400; ++i) b2[i] = bp[i / 2];
  CHECK(same(at(reduce(ReduceOp::Sum, big2, a0), 0), C{1e16, 1e16}));
  // Real input with a complex64 dtype override.
  ReduceOptions d;
  d.dtype = DType::Complex64;
  const NDArray f = reduce(ReduceOp::Sum, NDArray::zeros({3}, DType::Int8), d);
  CHECK(f.dtype() == DType::Complex64 && f.ndim() == 0);
}

TEST_CASE("complex reductions: mean (P1-3b)") {
  CHECK(reduce_result_dtype(ReduceOp::Mean, DType::Complex64) == DType::Complex64);
  CHECK_THROWS_KIND(reduce_result_dtype(ReduceOp::Var, DType::Complex128),
                    ErrorKind::NotImplemented);
  // [[1+1j, 2], [3, 4j]] over axis 0, keepdims -> [[2+0.5j, 1+2j]]
  const NDArray m = cvec({C{1, 1}, C{2, 0}, C{3, 0}, C{0, 4}}).reshape({2, 2});
  ReduceOptions k;
  k.axis = std::vector<std::int64_t>{0};
  k.keepdims = true;
  const NDArray m0 = reduce(ReduceOp::Mean, m, k);
  CHECK(m0.shape() == Shape({1, 2}) && m0.dtype() == DType::Complex128);
  CHECK(same(at(m0, 0), C{2, 0.5}) && same(at(m0, 1), C{1, 2}));
  // Empty -> nan+nanj; inf+1j -> inf+nanj (NumPy's Smith divide by count).
  const C e = at(reduce(ReduceOp::Mean, cvec({}), {}), 0);
  CHECK(std::isnan(e.real()) && std::isnan(e.imag()));
  CHECK(same(at(reduce(ReduceOp::Mean, cvec({C{kInf, 1}, C{1, 0}}), {}), 0), C{kInf, std::nan("")}));
  // Pairwise sum then divide: NumPy gives 50000000000000.94 (+same imag).
  NDArray big = NDArray::empty({200}, DType::Complex128);
  auto* bp = reinterpret_cast<C*>(big.data());
  bp[0] = C{1e16, 1e16};
  for (int i = 1; i < 200; ++i) bp[i] = C{1, 1};
  CHECK(same(at(reduce(ReduceOp::Mean, big, {}), 0), C{50000000000000.94, 50000000000000.94}));
  // complex64 (divide done in complex128, then rounded): [1+1j, 2, 4j] -> 1+1.6666666j
  NDArray f = NDArray::empty({3}, DType::Complex64);
  auto* fp = reinterpret_cast<std::complex<float>*>(f.data());
  fp[0] = {1, 1};
  fp[1] = {2, 0};
  fp[2] = {0, 4};
  const auto fm = reinterpret_cast<const std::complex<float>*>(reduce(ReduceOp::Mean, f, {}).data())[0];
  CHECK_EQ(fm.real(), 1.0f);
  CHECK_EQ(fm.imag(), 1.6666666269302368f);
  // Real input with a complex dtype: int8 [0, 1, 2] -> 1+0j (complex64).
  ReduceOptions d;
  d.dtype = DType::Complex64;
  const NDArray ri = reduce(ReduceOp::Mean, arange(0, 3, 1, DType::Int8), d);
  CHECK(ri.dtype() == DType::Complex64);
  CHECK_EQ(reinterpret_cast<const std::complex<float>*>(ri.data())[0].real(), 1.0f);
}



TEST_CASE("complex reductions: min/max/argmin/argmax (P1-3c)") {
  const double nan = std::nan("");
  const auto mx = [](const NDArray& a) { return at(reduce(ReduceOp::Max, a, {}), 0); };
  const auto mn = [](const NDArray& a) { return at(reduce(ReduceOp::Min, a, {}), 0); };
  const auto amax = [](const NDArray& a) { return arg_reduce(true, a, std::nullopt, false).get_int64(0); };
  const auto amin = [](const NDArray& a) { return arg_reduce(false, a, std::nullopt, false).get_int64(0); };
  // Lexicographic order: real part, then imag.
  const NDArray lex = cvec({C{1, 5}, C{2, 0}, C{2, -1}, C{1, 9}});
  CHECK(same(mx(lex), C{2, 0}) && same(mn(lex), C{1, 5}));
  CHECK_EQ(amax(lex), std::int64_t{1});
  CHECK_EQ(amin(lex), std::int64_t{0});
  // The first NaN in either part wins (values below are from NumPy 2.5.3).
  const NDArray ni = cvec({C{1, 1}, C{0, nan}, C{nan, 0}});
  CHECK(same(mx(ni), C{0, nan}) && same(mn(ni), C{0, nan}));
  CHECK_EQ(amax(ni), std::int64_t{1});
  CHECK_EQ(amin(ni), std::int64_t{1});
  const NDArray nr = cvec({C{1, 1}, C{nan, 0}, C{0, nan}});
  CHECK(same(mx(nr), C{nan, 0}) && same(mn(nr), C{nan, 0}));
  // Signed zeros compare equal, so ties keep the first value.
  CHECK(same(mx(cvec({C{-0.0, -0.0}, C{0.0, -0.0}})), C{-0.0, -0.0}));
  CHECK(same(mn(cvec({C{0.0, 0.0}, C{-0.0, 0.0}})), C{0.0, 0.0}));
  const NDArray ties = cvec({C{3, 1}, C{3, 1}, C{1, 0}});
  CHECK_EQ(amax(ties), std::int64_t{0});
  CHECK_EQ(amin(ties), std::int64_t{2});
  // Axis reductions: [[1+2j, 3-1j], [3, -1j]].
  const NDArray m = cvec({C{1, 2}, C{3, -1}, C{3, 0}, C{0, -1}}).reshape({2, 2});
  ReduceOptions a0;
  a0.axis = std::vector<std::int64_t>{0};
  const NDArray m0 = reduce(ReduceOp::Max, m, a0);
  CHECK(same(at(m0, 0), C{3, 0}) && same(at(m0, 1), C{3, -1}));
  const NDArray n0 = reduce(ReduceOp::Min, m, a0);
  CHECK(same(at(n0, 0), C{1, 2}) && same(at(n0, 1), C{0, -1}));
  const NDArray am0 = arg_reduce(true, m, 0, false);
  CHECK_EQ(am0.get_int64(0), std::int64_t{1});
  CHECK_EQ(am0.get_int64(1), std::int64_t{0});
  const NDArray an1 = arg_reduce(false, m, 1, true);
  CHECK(an1.shape() == Shape({2, 1}));
  CHECK_EQ(an1.get_int64(1), std::int64_t{1});
  // initial, empty input and complex64.
  ReduceOptions init;
  init.initial = 5;
  CHECK(same(at(reduce(ReduceOp::Max, cvec({C{1, 1}}), init), 0), C{5, 0}));
  CHECK(same(at(reduce(ReduceOp::Max, cvec({}), init), 0), C{5, 0}));
  CHECK_THROWS_KIND(reduce(ReduceOp::Max, cvec({}), {}), ErrorKind::Value);
  CHECK_THROWS_KIND(arg_reduce(true, cvec({}), std::nullopt, false), ErrorKind::Value);
  NDArray f = NDArray::empty({2}, DType::Complex64);
  auto* fp = reinterpret_cast<std::complex<float>*>(f.data());
  fp[0] = {1, 2};
  fp[1] = {1, 3};
  const NDArray fm = reduce(ReduceOp::Max, f, {});
  CHECK(fm.dtype() == DType::Complex64);
  CHECK_EQ(reinterpret_cast<const std::complex<float>*>(fm.data())[0].imag(), 3.0f);
}

