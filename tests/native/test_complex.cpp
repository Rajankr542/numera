#include <cmath>
#include <complex>
#include <limits>

#include "complex_kernels.hpp"
#include "error.hpp"
#include "ndarray.hpp"
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
