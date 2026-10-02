#include <cmath>
#include <optional>
#include <vector>

#include "error.hpp"
#include "p10_cumdiff.hpp"
#include "test_harness.hpp"
#include "test_p10_util.hpp"

using namespace nativpy;
using namespace p10test;

TEST_CASE("p10: cumsum / cumprod / cumulative_* / nancum*") {
  p10::CumulativeOptions o;
  const NDArray m = dbl({1, 2, 3, 4}, DType::Int8, {2, 2});
  NDArray r = p10::cumulative(false, m, o, nullptr);
  CHECK(r.dtype() == DType::Int64);
  CHECK(vals(r) == (D{1, 3, 6, 10}));
  CHECK(vals(p10::cumulative(true, m, o, nullptr)) == (D{1, 2, 6, 24}));
  o.axis = 1;
  CHECK(vals(p10::cumulative(false, m, o, nullptr)) == (D{1, 3, 3, 7}));
  o.include_initial = true;
  r = p10::cumulative(false, m, o, nullptr);
  CHECK(r.shape() == (Shape{2, 3}));
  CHECK(vals(r) == (D{0, 1, 3, 0, 3, 7}));
  p10::CumulativeOptions api;
  api.array_api = true;
  CHECK_THROWS_KIND(p10::cumulative(false, m, api, nullptr), ErrorKind::Value);
  p10::CumulativeOptions nan;
  nan.skip_nan = true;
  CHECK(vals(p10::cumulative(false, dbl({1, NAN, 2}), nan, nullptr)) == (D{1, 1, 3}));
  CHECK(vals(p10::cumulative(true, dbl({2, NAN, 2}), nan, nullptr)) == (D{2, 2, 4}));
  NDArray out = NDArray::zeros({3}, DType::Int64);
  p10::cumulative(false, dbl({1.5, 1, 1}), p10::CumulativeOptions{}, &out);
  CHECK(vals(out) == (D{1, 2, 3}));
}

TEST_CASE("p10: diff") {
  CHECK(vals(p10::diff(dbl({1, 4, 9}), 2, -1, std::nullopt, std::nullopt)) == (D{2}));
  CHECK(p10::diff(dbl({1, 2, 3}), 5, -1, std::nullopt, std::nullopt).size() == 0);
  const NDArray b = p10::diff(dbl({1, 0, 0}, DType::Bool), 1, -1, std::nullopt, std::nullopt);
  CHECK(b.dtype() == DType::Bool);
  CHECK(vals(b) == (D{1, 0}));
  CHECK(vals(p10::diff(dbl({5, 1}, DType::UInt8), 1, 0, std::nullopt, std::nullopt)) == (D{252}));
  const NDArray z = dbl({7}).reshape({});
  const NDArray r = p10::diff(dbl({1, 1, 1, 1, 1, 1}, DType::Float64, {2, 3}), 1, 0, std::nullopt, z);
  CHECK(r.shape() == (Shape{2, 3}));
  CHECK(vals(r) == (D{0, 0, 0, 6, 6, 6}));
  CHECK_THROWS_KIND(p10::diff(z, 1, 0, std::nullopt, std::nullopt), ErrorKind::Value);
  CHECK_THROWS_KIND(p10::diff(dbl({1}), -1, 0, std::nullopt, std::nullopt), ErrorKind::Value);
}

TEST_CASE("p10: ptp") {
  const NDArray r = p10::ptp(dbl({-100, 100}, DType::Int8), std::nullopt, false);
  CHECK(r.dtype() == DType::Int8);
  CHECK_EQ(r.get_double(0), -56.0);
  CHECK(vals(p10::ptp(dbl({1, 5, 2, 9}, DType::Float64, {2, 2}), std::vector<std::int64_t>{0}, true)) == (D{1, 4}));
  CHECK_THROWS_KIND(p10::ptp(dbl({1}, DType::Bool), std::nullopt, false), ErrorKind::DType);
  CHECK_THROWS_KIND(p10::ptp(NDArray::empty({0}, DType::Float64), std::nullopt, false), ErrorKind::Value);
}
