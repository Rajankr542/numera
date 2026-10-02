#include <cmath>
#include <optional>
#include <vector>

#include "error.hpp"
#include "p10_stats.hpp"
#include "test_harness.hpp"
#include "test_p10_util.hpp"

using namespace nativpy;
using namespace p10test;

TEST_CASE("p10: average") {
  const NDArray x = dbl({1, 1, 1, 1, 1, 1}, DType::Float64, {2, 3});
  auto [avg, scl] = p10::average(x, std::vector<std::int64_t>{1}, dbl({1, 2, 3}), false);
  CHECK(vals(avg) == (D{1, 1}));
  CHECK(vals(scl) == (D{6, 6}));
  auto [m, n] = p10::average(dbl({0, 1, 2, 3, 4, 5}, DType::Int64), std::nullopt, std::nullopt, false);
  CHECK_EQ(m.get_double(0), 2.5);
  CHECK_EQ(n.get_double(0), 6.0);
  CHECK_THROWS_KIND(p10::average(dbl({1, 2}), std::nullopt, dbl({1, -1}), false), ErrorKind::Value);
  CHECK_THROWS_KIND(p10::average(x, std::nullopt, dbl({1, 2, 3}), false), ErrorKind::DType);
}

TEST_CASE("p10: cov / corrcoef") {
  CHECK_EQ(p10::cov(dbl({1, 2, 3}), p10::CovOptions{}, nullptr).get_double(0), 1.0);
  p10::CovOptions o;
  o.y = dbl({1, 5, 2});
  CHECK(close(vals(p10::cov(dbl({1, 2, 3}), o, nullptr)), D{1, 0.5, 0.5, 13.0 / 3}));
  p10::CovOptions w;
  w.rowvar = false;
  w.aweights = dbl({1, 2});
  w.fweights = dbl({2, 1});
  CHECK(close(vals(p10::cov(dbl({1, 2, 3, 4}, DType::Float64, {2, 2}), w, nullptr)), D{1.6, 1.6, 1.6, 1.6}));
  p10::CovOptions d3;
  d3.ddof = 3;
  bool warn = false;
  CHECK(std::isinf(p10::cov(dbl({1, 2, 3}, DType::Float64, {1, 3}), d3, &warn).get_double(0)));
  CHECK(warn);
  p10::CovOptions fw;
  fw.fweights = dbl({1.5, 1});
  CHECK_THROWS_KIND(p10::cov(dbl({1, 2}), fw, nullptr), ErrorKind::DType);
  const NDArray r = p10::corrcoef(dbl({1, 2, 3, 1, 5, 2}, DType::Float64, {2, 3}), std::nullopt, true,
                                  std::nullopt, nullptr);
  CHECK(close(vals(r), D{1, 0.24019223070763066, 0.24019223070763066, 1}, 1e-15));
}

TEST_CASE("p10: gradient / trapezoid") {
  const NDArray f = dbl({1, 2, 4, 7, 11});
  CHECK(vals(p10::gradient(f, {}, std::nullopt, 1)[0]) == (D{1, 1.5, 2.5, 3.5, 4}));
  p10::GradSpacing two;
  two.value = 2;
  CHECK(vals(p10::gradient(f, {two}, std::nullopt, 1)[0]) == (D{0.5, 0.75, 1.25, 1.75, 2}));
  CHECK(vals(p10::gradient(f, {}, std::nullopt, 2)[0]) == (D{0.5, 1.5, 2.5, 3.5, 4.5}));
  p10::GradSpacing xs;
  xs.array = dbl({0, 1, 3, 4, 7});
  CHECK(close(vals(p10::gradient(f, {xs}, std::nullopt, 1)[0]), D{1, 1, 7.0 / 3, 31.0 / 12, 4.0 / 3}, 1e-14));
  const auto g2 = p10::gradient(dbl({1, 2, 6, 3, 4, 5}, DType::Float64, {2, 3}), {}, std::nullopt, 1);
  CHECK_EQ(g2.size(), std::size_t{2});
  CHECK(vals(g2[1]) == (D{1, 2.5, 4, 1, 1, 1}));
  CHECK_THROWS_KIND(p10::gradient(dbl({1, 2}), {}, std::nullopt, 2), ErrorKind::Value);
  CHECK(p10::gradient(dbl({1, 2, 4}, DType::Float32), {}, std::nullopt, 1)[0].dtype() == DType::Float32);
  CHECK(p10::gradient(dbl({1, 2, 4}, DType::Int8), {}, std::nullopt, 1)[0].dtype() == DType::Float64);

  CHECK_EQ(p10::trapezoid(dbl({1, 2, 3}), std::nullopt, 1, -1).get_double(0), 4.0);
  CHECK_EQ(p10::trapezoid(dbl({1, 2, 3}), dbl({0, 1, 3}), 1, -1).get_double(0), 6.5);
  CHECK(vals(p10::trapezoid(dbl({1, 2, 3, 4}, DType::Float64, {2, 2}), std::nullopt, 1, 0)) == (D{2, 3}));
}
