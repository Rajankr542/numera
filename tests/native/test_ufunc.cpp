#include <cmath>
#include <cstdint>

#include "broadcast.hpp"
#include "creation.hpp"
#include "error.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"
#include "ufunc.hpp"

using namespace nativpy;

namespace {
NDArray vec_i(std::initializer_list<std::int64_t> v, DType dt) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t i = 0;
  for (auto x : v) a.set_int64(i++, x);
  return a;
}
NDArray vec_d(std::initializer_list<double> v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t i = 0;
  for (auto x : v) a.set_double(i++, x);
  return a;
}
}  // namespace

TEST_CASE("broadcast: shapes") {
  CHECK(broadcast_shapes({{3, 1}, {4}}) == Shape({3, 4}));
  CHECK(broadcast_shapes({{}, {2, 3}}) == Shape({2, 3}));
  CHECK(broadcast_shapes({{0, 1}, {1, 5}}) == Shape({0, 5}));
  CHECK(broadcast_shapes({}) == Shape({}));
  CHECK_THROWS_KIND(broadcast_shapes({{3}, {4}}), ErrorKind::Broadcast);
  CHECK_THROWS_KIND(broadcast_shapes({{2, 0}, {3}}), ErrorKind::Broadcast);
}

TEST_CASE("broadcast: to / plan coalescing") {
  NDArray a = vec_d({1, 2, 3});
  NDArray b = broadcast_to(a, {4, 3});
  CHECK(b.strides() == Strides({0, 8}));
  CHECK(!b.owns_data());
  CHECK_EQ(b.get_double(10), 2.0);
  CHECK_THROWS_KIND(broadcast_to(a, {4}), ErrorKind::Broadcast);
  CHECK_THROWS_KIND(broadcast_to(ones({2, 3}, DType::Float64), {3}), ErrorKind::Broadcast);

  NDArray x = ones({2, 3, 4}, DType::Float64);
  NDArray y = ones({2, 3, 4}, DType::Float64);
  auto p = make_plan<2>(x.shape(), {&x, &y});
  CHECK(p.shape == Shape({24}));  // fully contiguous -> one loop
  NDArray t = transpose(x, {});
  auto q = make_plan<2>(t.shape(), {&t, &t});
  CHECK_EQ(q.shape.size(), std::size_t{3});  // no merge possible
  NDArray r = ones({3, 1}, DType::Float64);
  auto s = make_plan<2>({3, 4}, {&r, &r});
  CHECK(s.shape == Shape({3, 4}));
  CHECK(s.strides[0] == Strides({8, 0}));
}

TEST_CASE("ufunc: result dtypes") {
  CHECK(binary_result_dtype(BinaryOp::Divide, DType::Int32, DType::Int32) == DType::Float64);
  CHECK(binary_result_dtype(BinaryOp::Divide, DType::Float16, DType::Float16) == DType::Float16);
  CHECK(binary_result_dtype(BinaryOp::Power, DType::Bool, DType::Bool) == DType::Int8);
  CHECK(binary_result_dtype(BinaryOp::Add, DType::Bool, DType::Bool) == DType::Bool);
  CHECK(binary_result_dtype(BinaryOp::Add, DType::Int64, DType::UInt64) == DType::Float64);
  CHECK_THROWS_KIND(binary_result_dtype(BinaryOp::Subtract, DType::Bool, DType::Bool), ErrorKind::DType);
  CHECK_THROWS_KIND(binary_result_dtype(BinaryOp::Add, DType::Complex64, DType::Float32),
                    ErrorKind::NotImplemented);
  CHECK(unary_result_dtype(UnaryOp::Sqrt, DType::UInt8) == DType::Float16);
  CHECK(unary_result_dtype(UnaryOp::Sqrt, DType::Int16) == DType::Float32);
  CHECK(unary_result_dtype(UnaryOp::Log, DType::Int32) == DType::Float64);
  CHECK_THROWS_KIND(unary_result_dtype(UnaryOp::Negative, DType::Bool), ErrorKind::DType);
}

TEST_CASE("ufunc: integer edge cases") {
  NDArray a = vec_i({7, -7, 0, 5}, DType::Int32);
  NDArray b = vec_i({2, 2, 0, -3}, DType::Int32);
  NDArray m = binary(BinaryOp::Mod, a, b);
  NDArray f = binary(BinaryOp::FloorDivide, a, b);
  const std::int64_t em[] = {1, 1, 0, -1};
  const std::int64_t ef[] = {3, -4, 0, -2};
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(m.get_int64(i), em[i]);
    CHECK_EQ(f.get_int64(i), ef[i]);
  }
  NDArray mn = vec_i({-128}, DType::Int8);
  NDArray neg1 = vec_i({-1}, DType::Int8);
  CHECK_EQ(binary(BinaryOp::FloorDivide, mn, neg1).get_int64(0), std::int64_t{-128});
  CHECK_EQ(binary(BinaryOp::Mod, mn, neg1).get_int64(0), std::int64_t{0});
  CHECK_EQ(unary(UnaryOp::Abs, mn).get_int64(0), std::int64_t{-128});
  CHECK_EQ(binary(BinaryOp::Add, vec_i({127}, DType::Int8), vec_i({1}, DType::Int8)).get_int64(0),
           std::int64_t{-128});
  CHECK_EQ(binary(BinaryOp::Subtract, vec_i({1}, DType::UInt8), vec_i({2}, DType::UInt8)).get_int64(0),
           std::int64_t{255});
  CHECK_EQ(binary(BinaryOp::Multiply, vec_i({65535}, DType::UInt16), vec_i({65535}, DType::UInt16))
               .get_int64(0),
           std::int64_t{1});
  NDArray p = binary(BinaryOp::Power, vec_i({2, 3}, DType::Int8), vec_i({7, 5}, DType::Int8));
  CHECK_EQ(p.get_int64(0), std::int64_t{-128});
  CHECK_EQ(p.get_int64(1), std::int64_t{-13});
  CHECK_THROWS_KIND(binary(BinaryOp::Power, vec_i({2}, DType::Int64), vec_i({-1}, DType::Int64)),
                    ErrorKind::Value);
}

TEST_CASE("ufunc: float semantics and broadcasting") {
  NDArray m = binary(BinaryOp::Mod, vec_d({-7.5, 7.5, 1.0}), vec_d({2.0, -2.0, 0.0}));
  CHECK_EQ(m.get_double(0), 0.5);
  CHECK_EQ(m.get_double(1), -0.5);
  CHECK(std::isnan(m.get_double(2)));
  NDArray d = binary(BinaryOp::Divide, vec_i({1, 0, -1}, DType::Int64), vec_i({0, 0, 0}, DType::Int64));
  CHECK(d.dtype() == DType::Float64);
  CHECK(std::isinf(d.get_double(0)) && d.get_double(0) > 0);
  CHECK(std::isnan(d.get_double(1)));
  NDArray h = binary(BinaryOp::Add, vec_d({1.5}, DType::Float16), vec_d({0.001}, DType::Float16));
  CHECK_EQ(h.get_double(0), 1.5009765625);
  NDArray col = arange(0, 3, 1, DType::Float64).reshape({3, 1});
  NDArray row = arange(0, 4, 1, DType::Float64);
  NDArray s = binary(BinaryOp::Add, col, row);
  CHECK(s.shape() == Shape({3, 4}));
  CHECK(s.is_c_contiguous());
  CHECK_EQ(s.get_double(11), 5.0);
  NDArray t = binary(BinaryOp::Subtract, transpose(s, {}), vec_d({1}));
  CHECK(t.shape() == Shape({4, 3}));
  CHECK_EQ(t.get_double(1), 0.0);  // s[1,0] - 1
  CHECK_EQ(binary(BinaryOp::Add, ones({0, 3}, DType::Float64), ones({1, 3}, DType::Float64)).size(),
           std::int64_t{0});
  CHECK(std::isnan(unary(UnaryOp::Sqrt, vec_d({-1})).get_double(0)));
  NDArray lg = unary(UnaryOp::Log, vec_d({0}));
  CHECK(std::isinf(lg.get_double(0)) && lg.get_double(0) < 0);
  NDArray z = NDArray::empty({}, DType::Float64);
  z.set_double(0, 4.0);
  NDArray r0 = unary(UnaryOp::Sqrt, z);
  CHECK(r0.shape() == Shape({}));
  CHECK_EQ(r0.get_double(0), 2.0);
}

