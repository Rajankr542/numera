#include <cstdint>
#include <vector>

#include "error.hpp"
#include "p03_methods.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
NDArray iota(const Shape& shape, DType dt = DType::Int64) {
  NDArray a = NDArray::empty(shape, dt);
  for (std::int64_t i = 0; i < a.size(); ++i) a.set_int64(i, i);
  return a;
}
std::vector<std::int64_t> values(const NDArray& a) {
  std::vector<std::int64_t> v;
  for (std::int64_t i = 0; i < a.size(); ++i) v.push_back(a.get_int64(i));
  return v;
}
NDArray scalar(std::int64_t v, DType dt = DType::Int64) {
  NDArray s = NDArray::empty({}, dt);
  s.set_int64(0, v);
  return s;
}
}  // namespace

TEST_CASE("p03: fill") {
  NDArray a = iota({2, 3}, DType::Int8);
  fill(a, scalar(7));
  CHECK(values(a) == std::vector<std::int64_t>(6, 7));
  NDArray f = NDArray::empty({2}, DType::Float64);
  NDArray v = NDArray::empty({}, DType::Float64);
  v.set_double(0, 2.5);
  fill(f, v);
  CHECK_EQ(f.get_double(1), 2.5);
  fill(NDArray::empty({0}, DType::Int32), scalar(1));  // empty: no-op
  CHECK_THROWS_KIND(fill(a, NDArray::empty({0}, DType::Int64)), ErrorKind::Value);
  CHECK_THROWS_KIND(fill(a.as_readonly(), scalar(1)), ErrorKind::Value);
}

TEST_CASE("p03: to_bytes orders") {
  const NDArray a = iota({2, 3}, DType::Int8);
  const auto c = to_bytes(a, Order::C);
  const auto f = to_bytes(a, Order::F);
  CHECK(c == std::vector<std::byte>({std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}, std::byte{4}, std::byte{5}}));
  CHECK(f == std::vector<std::byte>({std::byte{0}, std::byte{3}, std::byte{1}, std::byte{4}, std::byte{2}, std::byte{5}}));
  const NDArray t = transpose(a, {});
  CHECK(to_bytes(t, Order::K) == to_bytes(t, Order::C));
  CHECK(to_bytes(t, Order::A) == to_bytes(a, Order::C));
  CHECK(to_bytes(NDArray::empty({0}, DType::Float64), Order::C).empty());
}

TEST_CASE("p03: view_as itemsize rules") {
  const NDArray a = iota({2, 3}, DType::Int32);
  const NDArray b = view_as(a, DType::UInt8);
  CHECK(b.shape() == Shape({2, 12}));
  CHECK(b.strides() == Strides({12, 1}));
  CHECK_EQ(b.get_int64(4), 1);
  CHECK(view_as(a, DType::Float32).shape() == Shape({2, 3}));
  CHECK(view_as(b, DType::Int32).shape() == Shape({2, 3}));
  CHECK_THROWS_KIND(view_as(a, DType::Int64), ErrorKind::Value);  // 12 bytes % 8
  CHECK_THROWS_KIND(view_as(transpose(a, {}), DType::Int16), ErrorKind::Value);
  CHECK_THROWS_KIND(view_as(scalar(1, DType::Int32), DType::Int16), ErrorKind::Value);
  CHECK(view_as(NDArray::empty({3, 0}, DType::Int32), DType::Int16).shape() == Shape({3, 0}));
  CHECK(!view_as(a.as_readonly(), DType::UInt32).writeable());
}

TEST_CASE("p03: byteswap") {
  const NDArray a = iota({2}, DType::Int16);
  const NDArray s = byteswap(a, false);
  CHECK_EQ(s.get_int64(1), 256);
  CHECK_EQ(a.get_int64(1), 1);
  CHECK(s.owns_data());
  byteswap(a, true);
  CHECK_EQ(a.get_int64(1), 256);
  CHECK_THROWS_KIND(byteswap(a.as_readonly(), true), ErrorKind::Value);
  const NDArray t = byteswap(transpose(iota({2, 3}, DType::Int16), {}), false);
  CHECK(t.strides() == Strides({2, 6}));  // keeps the input layout
  NDArray z = NDArray::empty({1}, DType::Complex64);
  z.set_double(0, 1.0);
  const NDArray zs = byteswap(z, false);
  CHECK(std::to_integer<int>(zs.data()[0]) == 0x3f);  // 1.0f = 3f800000, first component swapped
  CHECK(std::to_integer<int>(zs.data()[4]) == 0);
}

TEST_CASE("p03: flat_assign") {
  NDArray a = iota({2, 2});
  NDArray pos = NDArray::empty({3}, DType::Int64);
  pos.set_int64(0, 0);
  pos.set_int64(1, -1);
  pos.set_int64(2, 1);
  NDArray vals = NDArray::empty({2}, DType::Int64);
  vals.set_int64(0, 10);
  vals.set_int64(1, 20);
  flat_assign(a, &pos, vals);
  CHECK(values(a) == std::vector<std::int64_t>({10, 10, 2, 20}));
  flat_assign(a, nullptr, scalar(5));
  CHECK(values(a) == std::vector<std::int64_t>(4, 5));
  NDArray bad = NDArray::empty({1}, DType::Int64);
  bad.set_int64(0, 4);
  CHECK_THROWS_KIND(flat_assign(a, &bad, vals), ErrorKind::Index);
  CHECK_THROWS_KIND(flat_assign(a.as_readonly(), nullptr, vals), ErrorKind::Value);
}
