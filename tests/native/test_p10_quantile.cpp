#include <cmath>
#include <optional>
#include <vector>

#include "error.hpp"
#include "p10_quantile.hpp"
#include "test_harness.hpp"
#include "test_p10_util.hpp"

using namespace nativpy;
using namespace p10test;

namespace {
NDArray scalar(double v) { return dbl({v}).reshape({}); }
p10::QuantileOptions with(p10::QMethod m) {
  p10::QuantileOptions o;
  o.method = m;
  return o;
}
}  // namespace

TEST_CASE("p10: quantile methods (NumPy 2.5 values)") {
  // a = arange(10) ** 1.5 is not needed: simple sample 0..9, q list.
  const NDArray a = dbl({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  const NDArray q = dbl({0, 0.01, 0.1, 0.33, 0.5, 0.55, 0.9, 0.99, 1});
  CHECK(close(vals(p10::quantile(a, q, with(p10::QMethod::Linear))),
              D{0, 0.09, 0.9, 2.97, 4.5, 4.95, 8.1, 8.91, 9}));
  CHECK(close(vals(p10::quantile(a, q, with(p10::QMethod::Hazen))),
              D{0, 0, 0.5, 2.8000000000000003, 4.5, 5, 8.5, 9, 9}));
  CHECK(vals(p10::quantile(a, q, with(p10::QMethod::Lower))) == (D{0, 0, 0, 2, 4, 4, 8, 8, 9}));
  CHECK(vals(p10::quantile(a, q, with(p10::QMethod::Higher))) == (D{0, 1, 1, 3, 5, 5, 9, 9, 9}));
  CHECK(vals(p10::quantile(a, q, with(p10::QMethod::InvertedCdf))) == (D{0, 0, 0, 3, 4, 5, 8, 9, 9}));
  CHECK_THROWS_KIND(p10::qmethod_from_name("bad"), ErrorKind::Value);
  CHECK(p10::qmethod_from_name("nearest") == p10::QMethod::Nearest);
}

TEST_CASE("p10: quantile dtypes, NaN, errors") {
  const NDArray i8 = dbl({1, 2, 3, 4}, DType::Int8);
  p10::QuantileOptions weak;
  weak.weak_q = true;
  const NDArray r1 = p10::quantile(i8, scalar(0.5), weak);
  CHECK(r1.dtype() == DType::Float64);
  CHECK_EQ(r1.get_double(0), 2.5);
  CHECK(p10::quantile(i8, scalar(0.5), with(p10::QMethod::Lower)).dtype() == DType::Int8);
  CHECK(p10::quantile(dbl({1, 2, 3}, DType::Float32), scalar(0.3), weak).dtype() == DType::Float32);
  // int8 differences wrap like NumPy's b - a in the input dtype.
  CHECK_EQ(p10::quantile(dbl({-100, 100}, DType::Int8), scalar(0.75), weak).get_double(0), 114.0);
  CHECK(std::isnan(p10::quantile(dbl({1, NAN, 3}), scalar(0.5), weak).get_double(0)));
  p10::QuantileOptions nanq = weak;
  nanq.ignore_nan = true;
  CHECK_EQ(p10::quantile(dbl({1, NAN, 3}), scalar(0.5), nanq).get_double(0), 2.0);
  CHECK_THROWS_KIND(p10::quantile(dbl({1, 2}), scalar(1.5), weak), ErrorKind::Value);
  CHECK_THROWS_KIND(p10::quantile(NDArray::empty({0}, DType::Float64), scalar(0.5), weak), ErrorKind::Index);
  CHECK_THROWS_KIND(p10::quantile(dbl({1, 2}, DType::Complex128), scalar(0.5), weak), ErrorKind::DType);
  CHECK_THROWS_KIND(p10::quantile(dbl({1, 0}, DType::Bool), scalar(0.5), weak), ErrorKind::DType);
  p10::QuantileOptions w = weak;
  w.weights = dbl({1, 1, 1, 1});
  CHECK_THROWS_KIND(p10::quantile(i8, scalar(0.5), w), ErrorKind::Value);  // linear + weights
}

TEST_CASE("p10: quantile axis, keepdims, weights") {
  const NDArray a = dbl({0, 1, 2, 3, 4, 5}, DType::Float64, {2, 3});
  p10::QuantileOptions o;
  o.axis = std::vector<std::int64_t>{1};
  o.keepdims = true;
  const NDArray r = p10::quantile(a, dbl({0.1, 0.5}, DType::Float64, {1, 2}), o);
  CHECK(r.shape() == (Shape{1, 2, 2, 1}));
  CHECK(close(vals(r), D{0.2, 3.2, 1, 4}));
  p10::QuantileOptions w;
  w.axis = std::vector<std::int64_t>{1};
  w.method = p10::QMethod::InvertedCdf;
  w.weights = dbl({1, 2, 3});
  w.weak_q = true;
  CHECK(vals(p10::quantile(a, scalar(0.5), w)) == (D{1, 4}));
  w.weights = dbl({0, 0, 0});
  CHECK_THROWS_KIND(p10::quantile(a, scalar(0.5), w), ErrorKind::Value);
  w.weights = dbl({1, -1, 0});
  CHECK_THROWS_KIND(p10::quantile(a, scalar(0.5), w), ErrorKind::Value);
}

TEST_CASE("p10: median / nanmedian") {
  CHECK_EQ(p10::median(dbl({100, 120}, DType::Int8), std::nullopt, false, false).get_double(0), 110.0);
  const NDArray m = dbl({1, NAN, 3, 4}, DType::Float64, {2, 2});
  const NDArray r = p10::median(m, std::vector<std::int64_t>{1}, false, false);
  CHECK(close(vals(r), D{NAN, 3.5}));
  CHECK(close(vals(p10::median(m, std::vector<std::int64_t>{1}, true, true)), D{1, 3.5}));
  CHECK(p10::median(m, std::vector<std::int64_t>{1}, true, true).shape() == (Shape{2, 1}));
  CHECK(std::isnan(p10::median(NDArray::empty({0}, DType::Float64), std::nullopt, false, false).get_double(0)));
  CHECK(p10::median(dbl({1, 2}, DType::Float16), std::nullopt, false, false).dtype() == DType::Float16);
  CHECK_THROWS_KIND(p10::median(dbl({1}, DType::Complex128), std::nullopt, false, false),
                    ErrorKind::NotImplemented);
}
