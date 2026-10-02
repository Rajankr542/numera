#include <cmath>
#include <limits>

#include "error.hpp"
#include "p03_print.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
Dragon4Options pos() { return Dragon4Options{}; }
Dragon4Options sci() {
  Dragon4Options o;
  o.scientific = true;
  return o;
}
NDArray vec(std::initializer_list<double> v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t k = 0;
  for (const double x : v) a.set_double(k++, x);
  return a;
}
}  // namespace

TEST_CASE("p03: dragon4 positional") {
  CHECK_EQ(dragon4(1.0, DType::Float64, pos()), std::string("1."));
  CHECK_EQ(dragon4(0.1, DType::Float64, pos()), std::string("0.1"));
  CHECK_EQ(dragon4(-0.0, DType::Float64, pos()), std::string("-0."));
  CHECK_EQ(dragon4(1e20, DType::Float64, pos()), std::string("100000000000000000000."));
  CHECK_EQ(dragon4(0.1, DType::Float32, pos()), std::string("0.1"));
  Dragon4Options o = pos();
  o.unique = false;
  o.precision = 10;
  CHECK_EQ(dragon4(static_cast<double>(0.1f), DType::Float32, o), std::string("0.1000000015"));
  o = pos();
  o.precision = 0;
  CHECK_EQ(dragon4(2.5, DType::Float64, o), std::string("2."));
  CHECK_EQ(dragon4(3.5, DType::Float64, o), std::string("4."));
  o = pos();
  o.precision = 2;
  CHECK_EQ(dragon4(0.0001, DType::Float64, o), std::string("0.00"));
  o = pos();
  o.precision = 3;
  o.fractional = false;
  CHECK_EQ(dragon4(1234.5678, DType::Float64, o), std::string("1230."));
  o = pos();
  o.min_digits = 20;
  CHECK_EQ(dragon4(0.3, DType::Float64, o), std::string("0.29999999999999998890"));
  o = pos();
  o.trim = TrimMode::DptZeros;
  o.pad_right = 3;
  CHECK_EQ(dragon4(1.0, DType::Float64, o), std::string("1    "));
  o = pos();
  o.pad_left = 2;
  CHECK_EQ(dragon4(0.5, DType::Float64, o), std::string(" 0.5"));
  CHECK_EQ(dragon4(5e-324, DType::Float64, pos()).substr(0, 6), std::string("0.0000"));
  CHECK_EQ(dragon4(std::numeric_limits<double>::infinity(), DType::Float64, pos()), std::string("inf"));
  CHECK_EQ(dragon4(-std::numeric_limits<double>::quiet_NaN(), DType::Float64, pos()), std::string("nan"));
  CHECK_THROWS_KIND(dragon4(1.0, DType::Int8, pos()), ErrorKind::DType);
}

TEST_CASE("p03: dragon4 scientific") {
  CHECK_EQ(dragon4(1.0, DType::Float64, sci()), std::string("1.e+00"));
  Dragon4Options o = sci();
  o.trim = TrimMode::DptZeros;
  CHECK_EQ(dragon4(1.0, DType::Float64, o), std::string("1e+00"));
  o = sci();
  o.precision = 2;
  CHECK_EQ(dragon4(123.456, DType::Float64, o), std::string("1.23e+02"));
  CHECK_EQ(dragon4(9.9999, DType::Float64, o), std::string("1.e+01"));
  o = sci();
  o.exp_digits = 4;
  CHECK_EQ(dragon4(1e100, DType::Float64, o), std::string("1.e+0100"));
  o = sci();
  o.pad_left = 3;
  CHECK_EQ(dragon4(-1.5, DType::Float64, o), std::string(" -1.5e+00"));
  CHECK_EQ(dragon4(1e-5, DType::Float16, sci()), std::string("1.e-05"));
}

TEST_CASE("p03: format_elements and scalar_str") {
  PrintOptions po;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  const auto f = format_elements(vec({1.5, -2, nan, inf}), po);
  CHECK_EQ(f[0], std::string(" 1.5"));
  CHECK_EQ(f[1], std::string("-2. "));
  CHECK_EQ(f[2], std::string(" nan"));
  CHECK_EQ(f[3], std::string(" inf"));
  const auto e = format_elements(vec({1e-5, 1}), po);
  CHECK_EQ(e[0], std::string("1.e-05"));
  CHECK_EQ(e[1], std::string("1.e+00"));
  const auto b = format_elements(vec({1, 0}, DType::Bool), po);
  CHECK_EQ(b[0], std::string(" True"));
  CHECK_EQ(b[1], std::string("False"));
  const auto n = format_elements(vec({-1, 10}, DType::Int8), po);
  CHECK_EQ(n[0], std::string("-1"));
  CHECK_EQ(n[1], std::string("10"));
  CHECK_EQ(scalar_str(vec({1e16})), std::string("1e+16"));
  CHECK_EQ(scalar_str(vec({1000}, DType::Float16)), std::string("1e+03"));
  CHECK_EQ(scalar_str(vec({0.5}, DType::Float32)), std::string("0.5"));
  CHECK_EQ(scalar_str(vec({2})), std::string("2.0"));
  NDArray big = NDArray::empty({10}, DType::Int64);
  for (std::int64_t k = 0; k < 10; ++k) big.set_int64(k, k);
  const NDArray lt = leading_trailing(big, 2);
  CHECK_EQ(lt.size(), 4);
  CHECK_EQ(lt.get_int64(2), 8);
}
