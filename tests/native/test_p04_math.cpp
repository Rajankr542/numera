#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numbers>

#include "cast.hpp"

#include "error.hpp"
#include "test_harness.hpp"
#include "ufunc_registry.hpp"

using namespace nativpy;

namespace {
NDArray vec_d(std::initializer_list<double> v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t i = 0;
  for (auto x : v) a.set_double(i++, x);
  return a;
}
NDArray vec_i(std::initializer_list<std::int64_t> v, DType dt) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t i = 0;
  for (auto x : v) a.set_int64(i++, x);
  return a;
}
const Ufunc& U(const char* name) {
  const Ufunc* u = find_ufunc(name);
  if (!u) throw_error(ErrorKind::Value, std::string("missing ufunc ") + name);
  return *u;
}
NDArray un(const char* name, const NDArray& a) { return unary(U(name), a, nullptr, {}); }
NDArray bin(const char* name, const NDArray& a, const NDArray& b) {
  return binary(U(name), a, b, nullptr, {});
}
bool near(double a, double b, double tol = 1e-12) {
  if (std::isnan(a) && std::isnan(b)) return true;
  if (std::isinf(a) || std::isinf(b)) return a == b;
  return std::fabs(a - b) <= tol * std::fmax(1.0, std::fabs(b));
}
}  // namespace

TEST_CASE("p04 trig: values and dtypes") {
  const NDArray x = vec_d({0.0, 0.5, -1.0});
  const NDArray s = un("sin", x);
  CHECK(s.dtype() == DType::Float64);
  CHECK(near(s.get_double(1), std::sin(0.5)));
  CHECK(near(un("arccos", x).get_double(2), std::acos(-1.0)));
  CHECK(near(un("arctanh", x).get_double(1), std::atanh(0.5)));
  CHECK(std::isinf(un("arctanh", vec_d({1.0})).get_double(0)));
  CHECK(std::isnan(un("arcsin", vec_d({2.0})).get_double(0)));
  // ints/bool -> smallest safe float (NumPy)
  CHECK(un("cos", vec_i({1}, DType::Int8)).dtype() == DType::Float16);
  CHECK(un("cos", vec_i({1}, DType::Int16)).dtype() == DType::Float32);
  CHECK(un("cos", vec_i({1}, DType::Int64)).dtype() == DType::Float64);
  CHECK(un("tanh", vec_i({1}, DType::Bool)).dtype() == DType::Float16);
  CHECK(near(un("rad2deg", vec_d({std::numbers::pi})).get_double(0), 180.0));
  CHECK(near(un("deg2rad", vec_d({180.0})).get_double(0), std::numbers::pi));
}

TEST_CASE("p04 trig: complex loops") {
  NDArray z = NDArray::empty({1}, DType::Complex128);
  z.set_double(0, 2.0);  // 2+0j
  const NDArray r = un("arcsin", z);
  CHECK(r.dtype() == DType::Complex128);
  const auto v = load<std::complex<double>>(r.data());
  CHECK(near(v.real(), std::numbers::pi / 2));
  CHECK(near(std::fabs(v.imag()), 1.3169578969248166));
  CHECK(un("sinh", NDArray::empty({2}, DType::Complex64)).dtype() == DType::Complex64);
  CHECK_THROWS_KIND(un("deg2rad", z), ErrorKind::DType);
  CHECK_THROWS_KIND(bin("arctan2", z, z), ErrorKind::DType);
}

TEST_CASE("p04 trig: binary") {
  const NDArray a = vec_d({1.0, -1.0, 0.0});
  const NDArray b = vec_d({1.0, -1.0, -0.0});
  const NDArray r = bin("arctan2", a, b);
  CHECK(near(r.get_double(0), std::numbers::pi / 4));
  CHECK(near(r.get_double(1), -3 * std::numbers::pi / 4));
  CHECK(near(r.get_double(2), std::numbers::pi));
  CHECK(near(bin("hypot", vec_d({3.0}), vec_d({4.0})).get_double(0), 5.0));
  CHECK(bin("hypot", vec_i({3}, DType::Int8), vec_i({4}, DType::UInt8)).dtype() == DType::Float16);
  CHECK(bin("arctan2", vec_d({1}, DType::Float16), vec_i({1}, DType::Int16)).dtype() == DType::Float32);
  CHECK(U("hypot").identity.has_value());
  CHECK(!U("arctan2").identity.has_value());
}
