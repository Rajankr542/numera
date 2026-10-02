#include <cmath>
#include <limits>

#include "error.hpp"
#include "p03_dtypes.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
NDArray scalar_d(double v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({}, dt);
  a.set_double(0, v);
  return a;
}
NDArray scalar_i(std::int64_t v) {
  NDArray a = NDArray::empty({}, DType::Int64);
  a.set_int64(0, v);
  return a;
}
}  // namespace

TEST_CASE("p03: float_info") {
  const FloatInfo h = float_info(DType::Float16);
  CHECK_EQ(h.eps, std::ldexp(1.0, -10));
  CHECK_EQ(h.max, 65504.0);
  CHECK_EQ(h.precision, 3);
  const FloatInfo f = float_info(DType::Complex64);
  CHECK(f.dtype == DType::Float32);
  CHECK_EQ(f.nmant, 23);
  CHECK_EQ(f.minexp, -126);
  CHECK_EQ(f.maxexp, 128);
  CHECK_EQ(f.negep, -24);
  const FloatInfo d = float_info(DType::Float64);
  CHECK_EQ(d.precision, 15);
  CHECK_EQ(d.resolution, 1e-15);
  CHECK_EQ(d.smallest_subnormal, 5e-324);
  CHECK_THROWS_KIND(float_info(DType::Int8), ErrorKind::Value);
}

TEST_CASE("p03: int_info") {
  CHECK_EQ(int_info(DType::Int8).min, -128);
  CHECK_EQ(int_info(DType::Int8).max, 127u);
  CHECK_EQ(int_info(DType::UInt64).max, 18446744073709551615ull);
  CHECK_EQ(int_info(DType::Int64).min, std::numeric_limits<std::int64_t>::min());
  CHECK_THROWS_KIND(int_info(DType::Bool), ErrorKind::Value);
  CHECK_THROWS_KIND(int_info(DType::Float32), ErrorKind::Value);
}

TEST_CASE("p03: min_scalar_type") {
  CHECK(min_scalar_type(scalar_i(10)) == DType::UInt8);
  CHECK(min_scalar_type(scalar_i(-10)) == DType::Int8);
  CHECK(min_scalar_type(scalar_i(300)) == DType::UInt16);
  CHECK(min_scalar_type(scalar_i(-129)) == DType::Int16);
  CHECK(min_scalar_type(scalar_i(-2147483649LL)) == DType::Int64);
  CHECK(min_scalar_type(scalar_d(3.1)) == DType::Float16);
  CHECK(min_scalar_type(scalar_d(65000.0)) == DType::Float32);
  CHECK(min_scalar_type(scalar_d(1e50)) == DType::Float64);
  CHECK(min_scalar_type(scalar_d(NAN)) == DType::Float16);
  CHECK(min_scalar_type(scalar_d(7e4, DType::Float32)) == DType::Float32);
  CHECK(min_scalar_type(scalar_d(1.0, DType::Complex128)) == DType::Complex64);
  CHECK(min_scalar_type(scalar_d(1e50, DType::Complex128)) == DType::Complex128);
  CHECK(min_scalar_type(NDArray::empty({1}, DType::Int64)) == DType::Int64);
}
