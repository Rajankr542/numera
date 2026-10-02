#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "error.hpp"
#include "p10_hist.hpp"
#include "test_harness.hpp"
#include "test_p10_util.hpp"

using namespace nativpy;
using namespace p10test;

// ---- histogram ----

TEST_CASE("p10_hist: histogram basic count") {
  const NDArray a = dbl({1, 2, 1, 3});
  p10::HistBins bins;
  bins.count = 3;
  std::vector<std::string> warns;
  auto [hist, edges] = p10::histogram(a, bins, std::nullopt, false, std::nullopt, false, warns);
  // bins: [1, 4/3), [4/3, 5/3), [5/3, 2) ... no; [1, rmax=3+0 => 3]
  // rmin=1, rmax=3; 3 bins => edges at 1, 5/3, 7/3, 3
  CHECK_EQ(hist.size(), std::int64_t{3});
  // values 1,1 go in bin 0; 2 in bin 1; 3 in bin 2 (last bin closed)
  CHECK_EQ(hist.get_int64(0) + hist.get_int64(1) + hist.get_int64(2), std::int64_t{4});
  CHECK_EQ(edges.size(), std::int64_t{4});
  CHECK_EQ(edges.get_double(0), 1.0);
  CHECK_EQ(edges.get_double(3), 3.0);
}

TEST_CASE("p10_hist: histogram with explicit edges") {
  const NDArray a = dbl({1, 2, 3, 4});
  const NDArray e = dbl({1, 2, 3, 4});
  p10::HistBins bins;
  bins.edges = e;
  std::vector<std::string> warns;
  auto [hist, edges] = p10::histogram(a, bins, std::nullopt, false, std::nullopt, false, warns);
  CHECK_EQ(hist.size(), std::int64_t{3});
  // 1 in [1,2), 2 in [2,3), 3 and 4 in [3,4] — last bin closed
  CHECK_EQ(hist.get_int64(0), std::int64_t{1});
  CHECK_EQ(hist.get_int64(1), std::int64_t{1});
  CHECK_EQ(hist.get_int64(2), std::int64_t{2});
}

TEST_CASE("p10_hist: histogram density") {
  const NDArray a = dbl({1, 2, 3, 4, 5});
  p10::HistBins bins;
  bins.count = 5;
  std::vector<std::string> warns;
  auto [hist, edges] = p10::histogram(a, bins, std::nullopt, true, std::nullopt, false, warns);
  // integral should be 1
  double integral = 0;
  for (std::int64_t i = 0; i < hist.size(); ++i) {
    const double width = edges.get_double(i + 1) - edges.get_double(i);
    integral += hist.get_double(i) * width;
  }
  CHECK(std::fabs(integral - 1.0) < 1e-12);
}

TEST_CASE("p10_hist: histogram range") {
  const NDArray a = dbl({1, 2, 3, 4, 5});
  p10::HistBins bins;
  bins.count = 2;
  std::vector<std::string> warns;
  auto [hist, edges] = p10::histogram(a, bins, std::make_optional(std::make_pair(2.0, 4.0)), false, std::nullopt, false, warns);
  // Only values 2, 3, 4 fall inside [2, 4].
  CHECK_EQ(hist.get_int64(0) + hist.get_int64(1), std::int64_t{3});
}

TEST_CASE("p10_hist: histogram_bin_edges (edges_only=true)") {
  const NDArray a = dbl({0, 1, 2, 3});
  p10::HistBins bins;
  bins.count = 4;
  std::vector<std::string> warns;
  auto [hist, edges] = p10::histogram(a, bins, std::nullopt, false, std::nullopt, true, warns);
  CHECK_EQ(hist.size(), std::int64_t{0});
  CHECK_EQ(edges.size(), std::int64_t{5});
}

TEST_CASE("p10_hist: histogram estimators") {
  // At least check they run without error for a normal-ish sample.
  const std::vector<double> vals = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  const NDArray a = dbl({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  for (const char* est : {"auto", "fd", "sturges", "scott", "rice", "doane", "sqrt"}) {
    p10::HistBins bins;
    bins.estimator = est;
    std::vector<std::string> warns;
    auto [hist, edges] = p10::histogram(a, bins, std::nullopt, false, std::nullopt, false, warns);
    CHECK(hist.size() > 0);
  }
}

TEST_CASE("p10_hist: histogram errors") {
  const NDArray a = dbl({1, 2, 3});
  // Non-monotonic edges.
  const NDArray bad_edges = dbl({3, 1, 2});
  p10::HistBins bins;
  bins.edges = bad_edges;
  std::vector<std::string> warns;
  bool threw = false;
  try { p10::histogram(a, bins, std::nullopt, false, std::nullopt, false, warns); }
  catch (const nativpy::Error& e) { threw = e.kind() == ErrorKind::Value; }
  CHECK(threw);
}

// ---- histogramdd ----

TEST_CASE("p10_hist: histogramdd 2D") {
  const NDArray x = dbl({0, 1, 2, 1});
  const NDArray y = dbl({0, 0, 1, 1});
  p10::HistBins bins;
  bins.count = 2;
  auto [hist, edges] = p10::histogramdd({x, y}, {bins}, {}, false, std::nullopt);
  CHECK_EQ(hist.shape()[0], std::int64_t{2});
  CHECK_EQ(hist.shape()[1], std::int64_t{2});
  CHECK_EQ(hist.size(), std::int64_t{4});
  // Total = 4 samples.
  std::int64_t total = 0;
  for (std::int64_t i = 0; i < 4; ++i) total += hist.get_int64(i);
  CHECK_EQ(total, std::int64_t{4});
  CHECK_EQ(static_cast<std::int64_t>(edges.size()), std::int64_t{2});
}

TEST_CASE("p10_hist: sample_columns 1D") {
  const NDArray a = dbl({1, 2, 3});
  const auto cols = p10::sample_columns(a, false);
  CHECK_EQ(static_cast<std::int64_t>(cols.size()), std::int64_t{1});
  CHECK_EQ(cols[0].size(), std::int64_t{3});
}

TEST_CASE("p10_hist: sample_columns 2D") {
  // (3, 2) -> two columns of length 3
  const NDArray m = dbl({1, 2, 3, 4, 5, 6}, DType::Float64, {3, 2});
  const auto cols = p10::sample_columns(m, false);
  CHECK_EQ(static_cast<std::int64_t>(cols.size()), std::int64_t{2});
  CHECK_EQ(cols[0].size(), std::int64_t{3});
  CHECK_EQ(cols[0].get_double(0), 1.0);
  CHECK_EQ(cols[1].get_double(0), 2.0);
}

// ---- bincount ----

TEST_CASE("p10_hist: bincount basic") {
  const NDArray x = dbl({1, 0, 2, 0, 1}, DType::Int64);
  const NDArray r = p10::bincount(x, std::nullopt, 0, false);
  CHECK_EQ(r.size(), std::int64_t{3});
  CHECK_EQ(r.get_int64(0), std::int64_t{2});
  CHECK_EQ(r.get_int64(1), std::int64_t{2});
  CHECK_EQ(r.get_int64(2), std::int64_t{1});
}

TEST_CASE("p10_hist: bincount minlength") {
  const NDArray x = dbl({1, 2}, DType::Int64);
  const NDArray r = p10::bincount(x, std::nullopt, 5, false);
  CHECK_EQ(r.size(), std::int64_t{5});
}

TEST_CASE("p10_hist: bincount weights") {
  const NDArray x = dbl({0, 1, 0}, DType::Int64);
  const NDArray w = dbl({0.5, 1.0, 1.5});
  const NDArray r = p10::bincount(x, w, 0, false);
  CHECK(r.dtype() == DType::Float64);
  CHECK_EQ(r.get_double(0), 2.0);
  CHECK_EQ(r.get_double(1), 1.0);
}

TEST_CASE("p10_hist: bincount negative raises") {
  const NDArray x = dbl({-1, 0}, DType::Int64);
  CHECK_THROWS_KIND(p10::bincount(x, std::nullopt, 0, false), ErrorKind::Value);
}

TEST_CASE("p10_hist: bincount empty fromList") {
  const NDArray x = NDArray::empty({0}, DType::Int64);
  const NDArray r = p10::bincount(x, std::nullopt, 3, true);
  CHECK_EQ(r.size(), std::int64_t{3});
}

// ---- digitize ----

TEST_CASE("p10_hist: digitize right=false (default)") {
  const NDArray x = dbl({0.2, 6.4, 3.0, 1.6});
  const NDArray bins = dbl({0, 1, 2.5, 4, 10});
  const NDArray r = p10::digitize(x, bins, false);
  // numpy: digitize([0.2,6.4,3.0,1.6], [0,1,2.5,4,10]) => [1,4,3,2]
  CHECK_EQ(r.get_int64(0), std::int64_t{1});
  CHECK_EQ(r.get_int64(1), std::int64_t{4});
  CHECK_EQ(r.get_int64(2), std::int64_t{3});
  CHECK_EQ(r.get_int64(3), std::int64_t{2});
}

TEST_CASE("p10_hist: digitize right=true") {
  const NDArray x = dbl({1.0, 2.5, 4.0});
  const NDArray bins = dbl({1, 2, 3, 4});
  const NDArray r = p10::digitize(x, bins, true);
  // right=True: x <= bin right edge -> bin = index of first bin > x
  // 1.0 -> lower_bound(bins, 1.0) = 0 index -> 0
  // 2.5 -> 2
  // 4.0 -> 3
  CHECK_EQ(r.get_int64(0), std::int64_t{0});
  CHECK_EQ(r.get_int64(1), std::int64_t{2});
  CHECK_EQ(r.get_int64(2), std::int64_t{3});
}

TEST_CASE("p10_hist: digitize decreasing bins") {
  const NDArray x = dbl({1.2, 10, 12, 0});
  const NDArray bins = dbl({10, 5, 1});  // decreasing
  const NDArray r = p10::digitize(x, bins, false);
  // numpy: digitize([1.2,10,12,0], [10,5,1]) => [2,0,0,3]
  CHECK_EQ(r.get_int64(0), std::int64_t{2});
  CHECK_EQ(r.get_int64(1), std::int64_t{0});
  CHECK_EQ(r.get_int64(2), std::int64_t{0});
  CHECK_EQ(r.get_int64(3), std::int64_t{3});
  // Also check right=True
  const NDArray r2 = p10::digitize(x, bins, true);
  // numpy: digitize([1.2,10,12,0], [10,5,1], right=True) => [2,1,0,3]
  CHECK_EQ(r2.get_int64(0), std::int64_t{2});
  CHECK_EQ(r2.get_int64(1), std::int64_t{1});
  CHECK_EQ(r2.get_int64(2), std::int64_t{0});
  CHECK_EQ(r2.get_int64(3), std::int64_t{3});
}

TEST_CASE("p10_hist: digitize non-monotonic bins raises") {
  const NDArray x = dbl({1, 2});
  const NDArray bins = dbl({1, 3, 2});
  CHECK_THROWS_KIND(p10::digitize(x, bins, false), ErrorKind::Value);
}

// ---- interp ----

TEST_CASE("p10_hist: interp basic") {
  const NDArray x = dbl({0, 1, 1.5, 2, 2.5, 3});
  const NDArray xp = dbl({1, 2, 3});
  const NDArray fp = dbl({3, 2, 0});
  const NDArray r = p10::interp(x, xp, fp, std::nullopt, std::nullopt, std::nullopt);
  // np.interp([0,1,1.5,2,2.5,3],[1,2,3],[3,2,0]) = [3, 3, 2.5, 2, 1, 0]
  CHECK_EQ(r.get_double(0), 3.0);   // clamped left
  CHECK_EQ(r.get_double(1), 3.0);   // exactly at xp[0]
  CHECK_EQ(r.get_double(2), 2.5);
  CHECK_EQ(r.get_double(3), 2.0);
  CHECK_EQ(r.get_double(4), 1.0);
  CHECK_EQ(r.get_double(5), 0.0);
}

TEST_CASE("p10_hist: interp left/right fill") {
  const NDArray x = dbl({-1, 5});
  const NDArray xp = dbl({0, 1, 2});
  const NDArray fp = dbl({0, 1, 2});
  const NDArray r = p10::interp(x, xp, fp,
                                std::make_optional(std::complex<double>{-99, 0}),
                                std::make_optional(std::complex<double>{99, 0}),
                                std::nullopt);
  CHECK_EQ(r.get_double(0), -99.0);
  CHECK_EQ(r.get_double(1), 99.0);
}

TEST_CASE("p10_hist: interp preserves shape") {
  const NDArray x = dbl({1, 2}, DType::Float64, {2});
  const NDArray xp = dbl({0, 1, 2, 3});
  const NDArray fp = dbl({0, 1, 4, 9});
  const NDArray r = p10::interp(x, xp, fp, std::nullopt, std::nullopt, std::nullopt);
  CHECK(r.shape() == x.shape());
}
