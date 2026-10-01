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
  CHECK_THROWS_KIND(reduce_result_dtype(ReduceOp::Var, DType::Complex128),
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

TEST_CASE("reduce: D-021 fast paths (lanes, pairwise, column sweep)") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  // Long rows take the multi-lane min/max path; the NaN is at each position
  // class (lane seed, lane body, scalar tail).
  for (const std::int64_t pos : {0LL, 5LL, 16LL, 500LL, 998LL}) {
    NDArray x = arange(0, 999, 1, DType::Float64);
    x.set_double(pos, nan);
    CHECK(std::isnan(reduce(ReduceOp::Max, x, {}).get_double(0)));
    CHECK(std::isnan(reduce(ReduceOp::Min, x, {}).get_double(0)));
  }
  // ±inf without NaN: the v - v NaN probe fires, the rescan must not.
  NDArray f = arange(0, 100, 1, DType::Float64);
  f.set_double(3, inf);
  f.set_double(70, -inf);
  CHECK_EQ(reduce(ReduceOp::Max, f, {}).get_double(0), inf);
  CHECK_EQ(reduce(ReduceOp::Min, f, {}).get_double(0), -inf);
  // Signed zero across lanes: -0.0 < +0.0 regardless of position.
  NDArray z = NDArray::zeros({200}, DType::Float64);
  for (std::int64_t i = 0; i < 200; ++i) z.set_double(i, -0.0);
  z.set_double(150, 0.0);
  CHECK(!std::signbit(reduce(ReduceOp::Max, z, {}).get_double(0)));
  CHECK(std::signbit(reduce(ReduceOp::Min, z, {}).get_double(0)));
  // Integer lanes.
  NDArray iv = arange(0, 1000, 1, DType::Int32);
  iv.set_double(637, -5);
  CHECK_EQ(reduce(ReduceOp::Min, iv, {}).get_int64(0), -5);
  CHECK_EQ(reduce(ReduceOp::Max, iv, {}).get_int64(0), 999);
  // Pairwise sum: exact on integers-as-floats for every block-size regime,
  // and -0.0 only sums to +0.0 (0 + pairwise, as NumPy).
  for (const std::int64_t n : {3LL, 8LL, 128LL, 129LL, 5000LL}) {
    const double expect = static_cast<double>(n) * static_cast<double>(n - 1) / 2;
    CHECK_EQ(reduce(ReduceOp::Sum, arange(0, static_cast<double>(n), 1, DType::Float64), {}).get_double(0),
             expect);
  }
  NDArray nz = NDArray::zeros({300}, DType::Float64);
  for (std::int64_t i = 0; i < 300; ++i) nz.set_double(i, -0.0);
  CHECK(!std::signbit(reduce(ReduceOp::Sum, nz, {}).get_double(0)));
  // Column sweep (leading-axis reduction without a transpose copy) agrees
  // with the transposed-input path for every op, including initial and NaN.
  NDArray m = arange(0, 60, 1, DType::Float64).reshape({12, 5});
  m.set_double(17, nan);
  const NDArray mt = m.view({5, 12}, {8, 40}, 0);  // m.T, non-contiguous
  for (const ReduceOp op : {ReduceOp::Sum, ReduceOp::Prod, ReduceOp::Min, ReduceOp::Max,
                            ReduceOp::Mean, ReduceOp::Var, ReduceOp::Std}) {
    const NDArray c = reduce(op, m, ax({0}));
    const NDArray r = reduce(op, mt, ax({1}));
    CHECK(c.shape() == Shape({5}));
    for (std::int64_t j = 0; j < 5; ++j) {
      const double a = c.get_double(j);
      const double b = r.get_double(j);
      CHECK((std::isnan(a) && std::isnan(b)) || a == b);
    }
  }
  CHECK(std::isnan(reduce(ReduceOp::Max, m, ax({0})).get_double(2)));
  CHECK_EQ(reduce(ReduceOp::Max, m, ax({0})).get_double(1), 56.0);
  ReduceOptions init = ax({0});
  init.initial = 1000;
  CHECK_EQ(reduce(ReduceOp::Min, arange(0, 12, 1, DType::Int64).reshape({4, 3}), init).get_int64(0), 0);
  CHECK_EQ(reduce(ReduceOp::Sum, arange(0, 12, 1, DType::Int64).reshape({4, 3}), init).get_int64(2), 1026);
  // Bool column sweep and 3-d leading axes.
  NDArray bm = NDArray::zeros({3, 2}, DType::Bool);
  bm.set_double(2, 1);
  CHECK_EQ(reduce(ReduceOp::Max, bm, ax({0})).get_int64(0), 1);
  CHECK_EQ(reduce(ReduceOp::Max, bm, ax({0})).get_int64(1), 0);
  const NDArray s3 = reduce(ReduceOp::Sum, arange(0, 24, 1, DType::Int64).reshape({2, 3, 4}), ax({0, 1}));
  CHECK(s3.shape() == Shape({4}));
  CHECK_EQ(s3.get_int64(0), 60);
}
