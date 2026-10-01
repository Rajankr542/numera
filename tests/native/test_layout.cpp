#include <cstdint>
#include <vector>

#include "error.hpp"
#include "layout.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
NDArray iota(const Shape& shape) {
  NDArray a = NDArray::empty(shape, DType::Int64);
  for (std::int64_t i = 0; i < a.size(); ++i) a.set_int64(i, i);
  return a;
}
std::vector<std::int64_t> values(const NDArray& a) {
  std::vector<std::int64_t> v;
  for (std::int64_t i = 0; i < a.size(); ++i) v.push_back(a.get_int64(i));
  return v;
}
}  // namespace

TEST_CASE("layout: empty_order C/F") {
  const NDArray f = empty_order({2, 3}, DType::Float64, Order::F);
  CHECK(f.strides() == Strides({8, 16}));
  CHECK(f.is_f_contiguous());
  const NDArray z = empty_order({2, 3}, DType::Int32, Order::C, true);
  CHECK(z.strides() == Strides({12, 4}));
  CHECK_EQ(z.get_int64(5), 0);
  CHECK(empty_order({0, 3}, DType::Float64, Order::F).strides() == Strides({0, 0}));
  CHECK_THROWS_KIND(empty_order({2}, DType::Float64, Order::K), ErrorKind::Value);
  CHECK_THROWS_KIND(empty_order({-1}, DType::Float64, Order::C), ErrorKind::Value);
}

TEST_CASE("layout: copy_order C/F/A/K") {
  const NDArray a = iota({2, 3});
  const NDArray f = copy_order(a, a.dtype(), Order::F);
  CHECK(f.strides() == Strides({8, 16}));
  CHECK(values(f) == values(a));
  CHECK(copy_order(f, f.dtype(), Order::A).strides() == Strides({8, 16}));
  CHECK(copy_order(a, a.dtype(), Order::A).strides() == Strides({24, 8}));
  CHECK(copy_order(f, f.dtype(), Order::K).strides() == Strides({8, 16}));
  // K keeps a permuted layout.
  const NDArray t = transpose(iota({2, 3, 4}), {1, 0, 2});  // strides (32, 96, 8)
  const NDArray k = copy_order(t, DType::Float32, Order::K);
  CHECK(k.strides() == Strides({16, 48, 4}));
  CHECK(values(k) == values(t));
  CHECK(copy_order(t, t.dtype(), Order::C).is_c_contiguous());
}

TEST_CASE("layout: reshape/ravel/flatten order") {
  const NDArray a = iota({2, 3});
  CHECK(values(ravel_order(a, Order::F)) == std::vector<std::int64_t>({0, 3, 1, 4, 2, 5}));
  CHECK(values(flatten_order(a, Order::F)) == std::vector<std::int64_t>({0, 3, 1, 4, 2, 5}));
  const NDArray r = reshape_order(a, {3, 2}, Order::F);
  CHECK(values(r) == std::vector<std::int64_t>({0, 4, 3, 2, 1, 5}));
  const NDArray f = copy_order(a, a.dtype(), Order::F);
  const NDArray rk = ravel_order(f, Order::K);
  CHECK(rk.shares_buffer(f));  // memory order: a view
  CHECK(values(rk) == std::vector<std::int64_t>({0, 3, 1, 4, 2, 5}));
  CHECK(values(ravel_order(f, Order::A)) == std::vector<std::int64_t>({0, 3, 1, 4, 2, 5}));
  CHECK(values(ravel_order(f, Order::C)) == values(a));
  CHECK(values(flatten_order(transpose(a, {}), Order::K)) == values(a));
  CHECK_THROWS_KIND(reshape_order(a, {6}, Order::K), ErrorKind::Value);
}
