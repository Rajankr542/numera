#include "creation.hpp"
#include "error.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"

using namespace nativpy;

TEST_CASE("creation: ones/full") {
  NDArray o = ones({2, 3}, DType::Int16);
  CHECK(o.shape() == Shape({2, 3}));
  for (int i = 0; i < 6; ++i) CHECK_EQ(o.get_int64(i), std::int64_t{1});
  NDArray v = NDArray::empty({}, DType::Float32);
  v.set_double(0, 2.5);
  NDArray f = full({0, 4}, v);
  CHECK_EQ(f.size(), std::int64_t{0});
  CHECK(f.strides() == Strides({0, 0}));
  CHECK_EQ(full({3}, v).get_double(2), 2.5);
  CHECK_THROWS_KIND(full({2}, o), ErrorKind::Value);
}

TEST_CASE("creation: arange") {
  NDArray a = arange(10, 0, -3, DType::Int64);
  CHECK(a.shape() == Shape({4}));
  CHECK_EQ(a.get_int64(3), std::int64_t{1});
  CHECK_EQ(arange(0, 1, 0.3, DType::Int32).size(), std::int64_t{4});
  CHECK_EQ(arange(5, 0, 1, DType::Float64).size(), std::int64_t{0});
  CHECK_THROWS_KIND(arange(0, 5, 0, DType::Float64), ErrorKind::Value);
  CHECK_THROWS_KIND(arange(0, 3, 1, DType::Bool), ErrorKind::Value);
  CHECK_EQ(arange(0, 2, 1, DType::Bool).get_int64(1), std::int64_t{1});
}

TEST_CASE("creation: linspace/eye") {
  NDArray l = linspace(-1, 1, 5, true, DType::Int32);
  const std::int64_t expected[] = {-1, -1, 0, 0, 1};
  for (int i = 0; i < 5; ++i) CHECK_EQ(l.get_int64(i), expected[i]);
  CHECK_EQ(linspace(0, 1, 0, true, DType::Float64).size(), std::int64_t{0});
  CHECK_THROWS_KIND(linspace(0, 1, -1, true, DType::Float64), ErrorKind::Value);
  NDArray e = eye(3, 4, -1, DType::Int32);
  CHECK_EQ(e.get_int64(4), std::int64_t{1});
  CHECK_EQ(e.get_int64(9), std::int64_t{1});
  CHECK_EQ(e.get_int64(0), std::int64_t{0});
}

TEST_CASE("shape_ops: transpose/swapaxes/moveaxis") {
  NDArray a = arange(0, 24, 1, DType::Float64).reshape({2, 3, 4});
  NDArray t = transpose(a, {});
  CHECK(t.shape() == Shape({4, 3, 2}));
  CHECK(t.strides() == Strides({8, 32, 96}));
  CHECK(t.shares_buffer(a));
  CHECK_THROWS_KIND(transpose(a, {0, 0, 1}), ErrorKind::Value);
  CHECK_THROWS_KIND(transpose(a, {0, 1}), ErrorKind::Value);
  CHECK(swapaxes(a, 0, -1).shape() == Shape({4, 3, 2}));
  CHECK(moveaxis(a, {0}, {-1}).shape() == Shape({3, 4, 2}));
  CHECK(moveaxis(a, {0, 1}, {-1, -2}).shape() == Shape({4, 3, 2}));
}

TEST_CASE("shape_ops: squeeze/expand_dims/ravel/flatten") {
  NDArray a = NDArray::zeros({1, 3, 1}, DType::Float64);
  CHECK(squeeze(a, std::nullopt).shape() == Shape({3}));
  CHECK(squeeze(a, std::vector<std::int64_t>{-1}).shape() == Shape({1, 3}));
  CHECK_THROWS_KIND(squeeze(a, std::vector<std::int64_t>{1}), ErrorKind::Value);
  NDArray b = NDArray::zeros({2, 3}, DType::Float64);
  NDArray e = expand_dims(b, {-1});
  CHECK(e.shape() == Shape({2, 3, 1}));
  CHECK(e.strides() == Strides({24, 8, 8}));
  CHECK_THROWS_KIND(expand_dims(b, {0, 0}), ErrorKind::Value);
  NDArray t = transpose(arange(0, 6, 1, DType::Int64).reshape({2, 3}), {});
  NDArray r = ravel(t);
  CHECK_EQ(r.get_int64(1), std::int64_t{3});
  NDArray c = arange(0, 6, 1, DType::Int64);
  CHECK(ravel(c).shares_buffer(c));
  CHECK(!flatten(c).shares_buffer(c));
}
