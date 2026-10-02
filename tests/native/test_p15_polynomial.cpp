#include <cstdint>
#include <vector>

#include "error.hpp"
#include "p15_polynomial.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
NDArray vec(std::vector<double> v) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, DType::Float64);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_double(static_cast<std::int64_t>(i), v[i]);
  return a;
}
std::vector<double> get(const NDArray& a) {
  const auto* p = reinterpret_cast<const double*>(a.data());
  return {p, p + a.size()};
}
}  // namespace

TEST_CASE("p15 polynomial: power-basis multiply, divide and trim") {
  using poly::Basis;
  CHECK(get(poly::mul(Basis::Power, vec({1, 2}), vec({1, 2}))) == (std::vector<double>{1, 4, 4}));
  auto [q, r] = poly::div(Basis::Power, vec({1, 2, 3, 4}), vec({1, 2}));
  CHECK(get(q) == (std::vector<double>{0.75, 0.5, 2}));
  CHECK(get(r) == (std::vector<double>{0.25}));
  CHECK(get(poly::add(Basis::Power, vec({1, 2, 3}), vec({0, 0, -3}))) == (std::vector<double>{1, 2}));
}

TEST_CASE("p15 polynomial: chebyshev / hermite kernels") {
  using poly::Basis;
  CHECK(get(poly::to_power(Basis::Cheb, vec({1, 2, 3}))) == (std::vector<double>{-2, 2, 6}));
  CHECK(get(poly::pow(Basis::Herm, vec({1, 2}), 3)) == (std::vector<double>{25, 54, 12, 8}));
  CHECK(get(poly::val(Basis::Cheb, vec({0.5}), vec({1, 2, 3}))) == (std::vector<double>{0.5}));
  CHECK(get(poly::fromroots(Basis::Herm, vec({1, 2}))) == (std::vector<double>{2.5, -1.5, 0.25}));
  CHECK_EQ(poly::vander(Basis::Cheb, vec({0.5, 2}), 3).shape()[1], 4);
}

TEST_CASE("p15 polynomial: errors") {
  using poly::Basis;
  CHECK_THROWS_KIND(poly::div(Basis::Power, vec({1}), vec({0})), ErrorKind::Value);
  CHECK_THROWS_KIND(poly::companion(Basis::Leg, vec({1})), ErrorKind::Value);
  CHECK_THROWS_KIND(poly::der(Basis::Power, vec({1, 2}), -1, 1.0), ErrorKind::Value);
  CHECK(!poly::basis_from_name("nope").has_value());
}
