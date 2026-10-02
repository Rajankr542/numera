#include <complex>

#include "creation.hpp"
#include "error.hpp"
#include "p14_poly.hpp"
#include "test_harness.hpp"

using namespace nativpy;

TEST_CASE("p14: convolve_full promotes and wraps like NumPy") {
  NDArray a = arange(1, 3, 1, DType::Int64);  // [1, 2]
  NDArray b = arange(3, 5, 1, DType::Int64);  // [3, 4]
  NDArray c = p14::convolve_full(a, b);
  CHECK(c.shape() == (Shape{3}));
  CHECK_EQ(c.get_int64(0), std::int64_t{3});
  CHECK_EQ(c.get_int64(1), std::int64_t{10});
  CHECK_EQ(c.get_int64(2), std::int64_t{8});
  NDArray i8 = arange(100, 102, 1, DType::Int64).astype(DType::Int8);
  CHECK(p14::convolve_full(i8, i8).dtype() == DType::Int8);
  CHECK_EQ(p14::convolve_full(i8, i8).get_int64(0), std::int64_t{static_cast<std::int8_t>(10000 & 0xff)});
  NDArray f = arange(0, 2, 1, DType::Int64).astype(DType::Float32);
  CHECK(p14::convolve_full(f, a).dtype() == DType::Float64);
  CHECK_THROWS_KIND(p14::convolve_full(NDArray::empty({0}, DType::Float64), a), ErrorKind::Value);
}

TEST_CASE("p14: polydiv quotient and trimmed remainder") {
  NDArray u = NDArray::empty({3}, DType::Float64);  // x^2 - 3x + 2
  u.set_double(0, 1);
  u.set_double(1, -3);
  u.set_double(2, 2);
  NDArray v = NDArray::empty({2}, DType::Float64);
  v.set_double(0, 1);
  v.set_double(1, -1);
  auto [q, r] = p14::polydiv(u, v);
  CHECK(q.shape() == (Shape{2}));
  CHECK_EQ(q.get_double(1), -2.0);
  CHECK(r.shape() == (Shape{1}));
  CHECK_EQ(r.get_double(0), 0.0);
  CHECK_THROWS_KIND(p14::polydiv(u.astype(DType::Int64), v.astype(DType::Int64)), ErrorKind::DType);
}
