#include <vector>

#include "error.hpp"
#include "layout.hpp"
#include "p03_iter.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"

using namespace nativpy;

TEST_CASE("p03: iter_plan orders") {
  const NDArray a = NDArray::empty({2, 3}, DType::Int64);
  const NDArray t = transpose(a, {});
  IterPlan c = iter_plan({t}, Order::C);
  CHECK(c.shape == Shape({3, 2}));
  CHECK(c.axes == std::vector<std::size_t>({0, 1}));
  CHECK(iter_plan({t}, Order::F).axes == std::vector<std::size_t>({1, 0}));
  CHECK(iter_plan({t}, Order::K).axes == std::vector<std::size_t>({1, 0}));
  CHECK(iter_plan({t}, Order::A).axes == std::vector<std::size_t>({1, 0}));
  CHECK(iter_plan({t, a.reshape({3, 2})}, Order::A).axes == std::vector<std::size_t>({0, 1}));
  const IterPlan z = iter_plan({NDArray::empty({}, DType::Int8)}, Order::K);
  CHECK(z.shape.empty());
}

TEST_CASE("p03: iter_plan flips negative strides and broadcasts") {
  const NDArray a = NDArray::empty({2, 3}, DType::Int64);
  const NDArray r = a.view({2, 3}, {24, -8}, 16);
  const IterPlan k = iter_plan({r}, Order::K);
  CHECK(k.axes == std::vector<std::size_t>({0, 1}));
  CHECK(k.flipped == std::vector<bool>({false, true}));
  CHECK(iter_plan({r, a}, Order::K).flipped == std::vector<bool>({false, false}));
  CHECK(iter_plan({r}, Order::C).flipped == std::vector<bool>({false, false}));
  const IterPlan b = iter_plan({a, NDArray::empty({3}, DType::Int8)}, Order::K);
  CHECK(b.shape == Shape({2, 3}));
  CHECK_THROWS_KIND(iter_plan({a, NDArray::empty({2}, DType::Int8)}, Order::K), ErrorKind::Broadcast);
}
