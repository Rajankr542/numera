#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

#include "error.hpp"
#include "p15_emath.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
NDArray vec(std::vector<double> v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_double(static_cast<std::int64_t>(i), v[i]);
  return a;
}
std::complex<double> cget(const NDArray& a, std::int64_t i) {
  if (a.dtype() == DType::Complex64) {
    const auto v = reinterpret_cast<const std::complex<float>*>(a.data())[i];
    return {static_cast<double>(v.real()), static_cast<double>(v.imag())};
  }
  return reinterpret_cast<const std::complex<double>*>(a.data())[i];
}
bool near(double a, double b) { return std::fabs(a - b) <= 1e-6 * std::fmax(1.0, std::fabs(b)); }
}  // namespace

TEST_CASE("p15 emath: in-domain real input stays real") {
  const NDArray r = emath_unary(EmathOp::Sqrt, vec({4.0, 0.25}));
  CHECK(r.dtype() == DType::Float64);
  CHECK_EQ(r.get_double(0), 2.0);
  CHECK_EQ(r.get_double(1), 0.5);
  const NDArray f = emath_unary(EmathOp::Log2, vec({8.0}, DType::Float32));
  CHECK(f.dtype() == DType::Float32);
  CHECK_EQ(f.get_double(0), 3.0);
  CHECK(emath_unary(EmathOp::Sqrt, vec({4.0}, DType::Int8)).dtype() == DType::Float16);
  CHECK(emath_unary(EmathOp::Sqrt, vec({4.0}, DType::Int64)).dtype() == DType::Float64);
}

TEST_CASE("p15 emath: out-of-domain real input becomes complex") {
  const NDArray r = emath_unary(EmathOp::Sqrt, vec({-4.0, 4.0}));
  CHECK(r.dtype() == DType::Complex128);
  CHECK(cget(r, 0) == std::complex<double>(0.0, 2.0));
  CHECK(cget(r, 1) == std::complex<double>(2.0, 0.0));
  CHECK(emath_unary(EmathOp::Log, vec({-1.0}, DType::Float32)).dtype() == DType::Complex64);
  CHECK(emath_unary(EmathOp::Log, vec({-1.0}, DType::Int16)).dtype() == DType::Complex64);
  CHECK(emath_unary(EmathOp::Log, vec({-1.0}, DType::Int32)).dtype() == DType::Complex128);
  CHECK(emath_unary(EmathOp::Log, vec({-1.0}, DType::Float16)).dtype() == DType::Complex128);
  const NDArray l = emath_unary(EmathOp::Log10, vec({-100.0}));
  CHECK(near(cget(l, 0).real(), 2.0));
  CHECK(near(cget(l, 0).imag(), 1.3643763538418412));
  const NDArray a = emath_unary(EmathOp::Arcsin, vec({-4.0}));
  CHECK(near(cget(a, 0).real(), -1.5707963267948966));
  CHECK(near(cget(a, 0).imag(), 2.0634370688955608));
  const NDArray c = emath_unary(EmathOp::Arccos, vec({2.0}));
  CHECK(near(cget(c, 0).imag(), -1.3169578969248166));
  const NDArray t = emath_unary(EmathOp::Arctanh, vec({-4.0}));
  CHECK(near(cget(t, 0).imag(), 1.5707963267948966));
}

TEST_CASE("p15 emath: edge cases") {
  CHECK_EQ(emath_unary(EmathOp::Sqrt, vec({})).size(), std::int64_t{0});
  CHECK(std::isnan(emath_unary(EmathOp::Sqrt, vec({NAN})).get_double(0)));
  CHECK(std::isinf(emath_unary(EmathOp::Log, vec({0.0})).get_double(0)));
  CHECK(emath_unary(EmathOp::Sqrt, NDArray::zeros({}, DType::Bool)).dtype() == DType::Float16);
  CHECK(emath_op_from_name("log2") == EmathOp::Log2);
  CHECK(!emath_op_from_name("tan"));
}
