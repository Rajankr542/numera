#include <cmath>
#include <optional>
#include <vector>

#include "error.hpp"
#include "p10_nan.hpp"
#include "test_harness.hpp"
#include "test_p10_util.hpp"

using namespace nativpy;
using namespace p10test;

namespace {
ReduceOptions ax1() {
  ReduceOptions o;
  o.axis = std::vector<std::int64_t>{1};
  return o;
}
}  // namespace

TEST_CASE("p10: nansum / nanprod / nanmean / nanvar / nanstd") {
  const NDArray x = dbl({1, NAN, 3, NAN, NAN, NAN}, DType::Float64, {2, 3});
  CHECK(vals(p10::nan_reduce(ReduceOp::Sum, x, ax1())) == (D{4, 0}));
  CHECK(vals(p10::nan_reduce(ReduceOp::Prod, x, ax1())) == (D{3, 1}));
  CHECK(close(vals(p10::nan_reduce(ReduceOp::Mean, x, ax1())), D{2, NAN}));
  CHECK(close(vals(p10::nan_reduce(ReduceOp::Var, x, ax1())), D{1, NAN}));
  CHECK(close(vals(p10::nan_reduce(ReduceOp::Std, x, ax1())), D{1, NAN}));
  ReduceOptions d1;
  d1.ddof = 1;
  CHECK(close(vals(p10::nan_reduce(ReduceOp::Var, dbl({1, NAN, 2}), d1)), D{0.5}));
  CHECK(std::isnan(p10::nan_reduce(ReduceOp::Var, dbl({1, NAN}), d1).get_double(0)));
  const NDArray f32 = p10::nan_reduce(ReduceOp::Var, dbl({1, NAN, 2}, DType::Float32), ReduceOptions{});
  CHECK(f32.dtype() == DType::Float32);
  CHECK_EQ(f32.get_double(0), 0.25);
  CHECK(p10::nan_reduce(ReduceOp::Sum, dbl({100, 100}, DType::Int8), ReduceOptions{}).dtype() == DType::Int64);
  ReduceOptions bad;
  bad.dtype = DType::Int64;
  CHECK_THROWS_KIND(p10::nan_reduce(ReduceOp::Mean, dbl({1, NAN}), bad), ErrorKind::DType);
}

TEST_CASE("p10: nanmin / nanmax / nanargmin / nanargmax") {
  const NDArray x = dbl({1, NAN, 3, NAN, NAN, NAN}, DType::Float64, {2, 3});
  CHECK(close(vals(p10::nan_reduce(ReduceOp::Min, x, ax1())), D{1, NAN}));
  CHECK(close(vals(p10::nan_reduce(ReduceOp::Max, x, ax1())), D{3, NAN}));
  CHECK(std::signbit(p10::nan_reduce(ReduceOp::Min, dbl({0.0, -0.0}), ReduceOptions{}).get_double(0)));
  CHECK(!std::signbit(p10::nan_reduce(ReduceOp::Max, dbl({-0.0, 0.0}), ReduceOptions{}).get_double(0)));
  ReduceOptions init = ax1();
  init.initial = 0;
  CHECK(vals(p10::nan_reduce(ReduceOp::Min, x, init)) == (D{0, 0}));
  CHECK_THROWS_KIND(p10::nan_reduce(ReduceOp::Min, NDArray::empty({0}, DType::Float64), ReduceOptions{}),
                    ErrorKind::Value);
  CHECK_EQ(p10::nan_arg_reduce(true, dbl({NAN, 2, 5, NAN}), std::nullopt, false).get_int64(0), 2);
  CHECK_EQ(p10::nan_arg_reduce(false, dbl({1, NAN, 0.5}), std::nullopt, false).get_int64(0), 2);
  CHECK_THROWS_KIND(p10::nan_arg_reduce(false, x, 1, false), ErrorKind::Value);
  CHECK_EQ(p10::nan_arg_reduce(false, dbl({3, 1}, DType::Int8), std::nullopt, false).get_int64(0), 1);
}
