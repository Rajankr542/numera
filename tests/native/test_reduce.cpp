#include <cmath>
#include <cstdint>
#include <limits>

#include "creation.hpp"
#include "error.hpp"
#include "reduce.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
NDArray ar(std::int64_t n, DType dt = DType::Int64) {
  return arange(0, static_cast<double>(n), 1, dt);
}
ReduceOptions ax(std::vector<std::int64_t> axes, bool keep = false) {
  ReduceOptions o;
  o.axis = std::move(axes);
  o.keepdims = keep;
  return o;
}
}  // namespace

TEST_CASE("reduce: result dtypes follow NumPy (D-017)") {
  CHECK(reduce_result_dtype(ReduceOp::Sum, DType::Int8) == DType::Int64);
  CHECK(reduce_result_dtype(ReduceOp::Sum, DType::UInt8) == DType::UInt64);
  CHECK(reduce_result_dtype(ReduceOp::Prod, DType::Bool) == DType::Int64);
  CHECK(reduce_result_dtype(ReduceOp::Sum, DType::Float32) == DType::Float32);
  CHECK(reduce_result_dtype(ReduceOp::Max, DType::UInt16) == DType::UInt16);
  CHECK(reduce_result_dtype(ReduceOp::Mean, DType::Int32) == DType::Float64);
  CHECK(reduce_result_dtype(ReduceOp::Var, DType::Float16) == DType::Float16);
  CHECK_THROWS_KIND(reduce_result_dtype(ReduceOp::Sum, DType::Complex128),
                    ErrorKind::NotImplemented);
}

TEST_CASE("reduce: sum/prod/min/max with axis and keepdims") {
  const NDArray a = ar(6).reshape({2, 3});
  CHECK_EQ(reduce(ReduceOp::Sum, a, {}).get_int64(0), 15);
  const NDArray s0 = reduce(ReduceOp::Sum, a, ax({0}));
  CHECK(s0.shape() == Shape({3}));
  CHECK_EQ(s0.get_int64(2), 7);
  const NDArray s1 = reduce(ReduceOp::Sum, a, ax({-1}, true));
  CHECK(s1.shape() == Shape({2, 1}));
  CHECK_EQ(s1.get_int64(1), 12);
  CHECK(reduce(ReduceOp::Sum, a, ax({0, 1}, true)).shape() == Shape({1, 1}));
  CHECK(reduce(ReduceOp::Sum, a, ax({})).shape() == Shape({2, 3}));
  ReduceOptions p;
  p.initial = 2;
  CHECK_EQ(reduce(ReduceOp::Prod, ar(5, DType::Int64).reshape({5}), p).get_int64(0), 0);
  CHECK_EQ(reduce(ReduceOp::Max, a, ax({1})).get_int64(0), 2);
  CHECK_EQ(reduce(ReduceOp::Min, a, ax({0})).get_int64(1), 1);
  // Transposed (non-contiguous) input.
  const NDArray t = reduce(ReduceOp::Sum, a.view({3, 2}, {8, 24}, 0), ax({1}));
  CHECK_EQ(t.get_int64(0), 3);
  // int8 dtype override wraps.
  ReduceOptions d;
  d.dtype = DType::Int8;
  d.initial = 127;
  CHECK_EQ(reduce(ReduceOp::Sum, ar(3, DType::Int8), d).get_int64(0), -126);
}

TEST_CASE("reduce: empty inputs, NaN and signed zero") {
  const NDArray e = NDArray::zeros({0, 3}, DType::Float64);
  CHECK_THROWS_KIND(reduce(ReduceOp::Max, e, ax({0})), ErrorKind::Value);
  CHECK(reduce(ReduceOp::Max, e, ax({1})).shape() == Shape({0}));
  ReduceOptions init;
  init.initial = 5;
  CHECK_EQ(reduce(ReduceOp::Max, e, init).get_double(0), 5.0);
  CHECK_EQ(reduce(ReduceOp::Sum, e, ax({0})).get_double(1), 0.0);
  CHECK(std::isnan(reduce(ReduceOp::Mean, e, {}).get_double(0)));
  NDArray n = NDArray::empty({3}, DType::Float64);
  n.set_double(0, 1);
  n.set_double(1, std::numeric_limits<double>::quiet_NaN());
  n.set_double(2, 3);
  CHECK(std::isnan(reduce(ReduceOp::Max, n, {}).get_double(0)));
  CHECK_EQ(arg_reduce(true, n, std::nullopt, false).get_int64(0), 1);
  NDArray z = NDArray::empty({2}, DType::Float64);
  z.set_double(0, 0.0);
  z.set_double(1, -0.0);
  CHECK(std::signbit(reduce(ReduceOp::Min, z, {}).get_double(0)));
  CHECK(!std::signbit(reduce(ReduceOp::Max, z, {}).get_double(0)));
  CHECK_THROWS_KIND(arg_reduce(false, e, 0, false), ErrorKind::Value);
  CHECK(arg_reduce(false, e, 1, false).shape() == Shape({0}));
  CHECK_THROWS_KIND(reduce(ReduceOp::Sum, e, ax({2})), ErrorKind::Index);
  CHECK_THROWS_KIND(reduce(ReduceOp::Sum, e, ax({0, 0})), ErrorKind::Value);
}

TEST_CASE("reduce: mean/var/std and argmin/argmax") {
  const NDArray a = ar(6).reshape({2, 3});
  CHECK_EQ(reduce(ReduceOp::Mean, a, ax({1})).get_double(1), 4.0);
  ReduceOptions v;
  v.ddof = 1;
  CHECK_EQ(reduce(ReduceOp::Var, a, v).get_double(0), 3.5);
  CHECK_EQ(reduce(ReduceOp::Std, ar(3), {}).get_double(0), std::sqrt(2.0 / 3.0));
  CHECK(reduce(ReduceOp::Mean, ar(3, DType::Float32), {}).dtype() == DType::Float32);
  const NDArray am = arg_reduce(true, a, 1, true);
  CHECK(am.shape() == Shape({2, 1}));
  CHECK_EQ(am.get_int64(0), 2);
  CHECK_EQ(arg_reduce(false, a, std::nullopt, false).get_int64(0), 0);
  CHECK(arg_reduce(false, a, std::nullopt, true).shape() == Shape({1, 1}));
}
