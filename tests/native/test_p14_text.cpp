#include <cmath>
#include <string>

#include "creation.hpp"
#include "error.hpp"
#include "p14_text.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
p14::LoadtxtOptions opts(DType dt = DType::Float64) {
  p14::LoadtxtOptions o;
  o.comments = {"#"};
  o.dtype = dt;
  return o;
}
}  // namespace

TEST_CASE("p14: loadtxt parses whitespace tables with comments") {
  NDArray a = p14::loadtxt("# head\n1 2.5\n\n  -3 inf # tail\n", opts());
  CHECK(a.shape() == (Shape{2, 2}));
  CHECK_EQ(a.get_double(1), 2.5);
  CHECK_EQ(a.get_double(2), -3.0);
  CHECK(std::isinf(a.get_double(3)));
  NDArray e = p14::loadtxt("", opts());
  CHECK(e.shape() == (Shape{0, 1}));
}

TEST_CASE("p14: loadtxt delimiter, quotes, usecols, integer ranges") {
  auto o = opts(DType::Int8);
  o.delimiter = ",";
  o.quote = "\"";
  o.usecols = std::vector<std::int64_t>{-1};
  NDArray a = p14::loadtxt("\"1,x\",7\n2,-128\n", o);
  CHECK(a.shape() == (Shape{2, 1}));
  CHECK_EQ(a.get_int64(1), std::int64_t{-128});
  CHECK_THROWS_KIND(p14::loadtxt("1,300\n", o), ErrorKind::Value);
  CHECK_THROWS_KIND(p14::loadtxt("1,1.5\n", o), ErrorKind::Value);
  auto u = opts(DType::UInt64);
  CHECK_EQ(p14::loadtxt("18446744073709551615\n", u).get_uint64(0), UINT64_MAX);
  CHECK_THROWS_KIND(p14::loadtxt("-0\n", u), ErrorKind::Value);
  CHECK_THROWS_KIND(p14::loadtxt("1 2\n3\n", opts()), ErrorKind::Value);
}

TEST_CASE("p14: float_str follows NumPy scalar str") {
  CHECK_EQ(p14::float_str(1e16, DType::Float64), std::string("1e+16"));
  CHECK_EQ(p14::float_str(1e7, DType::Float64), std::string("10000000.0"));
  CHECK_EQ(p14::float_str(1e6, DType::Float32), std::string("1e+06"));
  CHECK_EQ(p14::float_str(static_cast<float>(0.1), DType::Float32), std::string("0.1"));
  CHECK_EQ(p14::float_str(1e3, DType::Float16), std::string("1e+03"));
  CHECK_EQ(p14::float_str(1e-4, DType::Float32), std::string("0.0001"));
  CHECK_EQ(p14::float_str(-0.0, DType::Float64), std::string("-0.0"));
  CHECK_EQ(p14::py_complex_repr(0, 1), std::string("1j"));
  CHECK_EQ(p14::py_complex_repr(1, -2.5), std::string("(1-2.5j)"));
}

TEST_CASE("p14: format_rows applies Python %-formatting") {
  NDArray a = arange(0, 4, 1, DType::Int64).reshape({2, 2});
  CHECK_EQ(p14::format_rows(a, "%03d,%-3x|", "\n"), std::string("000,1  |\n002,3  |\n"));
  NDArray f = arange(0, 2, 1, DType::Int64).astype(DType::Float64).reshape({1, 2});
  CHECK_EQ(p14::format_rows(f, "%.2e %g", "\r\n"), std::string("0.00e+00 1\r\n"));
  CHECK_THROWS_KIND(p14::format_rows(f, "%d", "\n"), ErrorKind::DType);
  CHECK_THROWS_KIND(p14::format_rows(f, "%r %r", "\n"), ErrorKind::Value);
  CHECK_EQ(p14::tofile_text(f, ", ", ""), std::string("0.0, 1.0"));
  CHECK_EQ(p14::tofile_text(f, " ", "%.1f"), std::string("0.0 1.0"));
}
