#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "cast.hpp"
#include "error.hpp"
#include "p09_sets.hpp"
#include "p09_sort_kernels.hpp"
#include "p09_sorting.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {

template <typename T>
NDArray vec(const std::vector<T>& v, DType dt, Shape shape = {}) {
  if (shape.empty()) shape = {static_cast<std::int64_t>(v.size())};
  NDArray a = NDArray::empty(shape, dt);
  for (std::size_t i = 0; i < v.size(); ++i) store<T>(a.data() + i * sizeof(T), v[i]);
  return a;
}

std::vector<double> dbl(const NDArray& a) {
  std::vector<double> out;
  for (std::int64_t i = 0; i < a.size(); ++i) out.push_back(a.get_double(i));
  return out;
}

std::vector<std::int64_t> ints(const NDArray& a) {
  std::vector<std::int64_t> out;
  for (std::int64_t i = 0; i < a.size(); ++i) out.push_back(a.get_int64(i));
  return out;
}

const std::vector<std::int64_t> kPart = {5, 1, 4, 1, 3, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9, 3,
                                         2, 3, 8, 4, 6, 2, 6, 4, 3, 3, 8, 3, 2, 7, 9, 5};

}  // namespace

TEST_CASE("p09: comparators order NaN last, also in reverse") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  CHECK(p09::lt(1.0, nan));
  CHECK(!p09::lt(nan, 1.0));
  CHECK(p09::gt(1.0, nan));
  CHECK(!p09::lt(-0.0, 0.0));
  using C = std::complex<double>;
  CHECK(p09::lt(C(1, 0), C(1, nan)));
  CHECK(p09::lt(C(1, nan), C(nan, 2)));
  CHECK(p09::lt(C(1, 2), C(1, 3)));
  const float16_t h1 = double_to_half(1.0), hn = double_to_half(nan), hz = double_to_half(-0.0);
  CHECK(p09::lt(h1, hn));
  CHECK(!p09::lt(hz, double_to_half(0.0)));
  CHECK(p09::lt(double_to_half(-2.0), double_to_half(-1.0)));
}

TEST_CASE("p09: sort and argsort for every dtype") {
  for (const DType dt : {DType::Bool, DType::Int8, DType::UInt8, DType::Int16, DType::UInt16, DType::Int32,
                         DType::UInt32, DType::Int64, DType::UInt64, DType::Float16, DType::Float32,
                         DType::Float64, DType::Complex64, DType::Complex128}) {
    const NDArray a = vec<std::int64_t>({3, 0, 2, 1, 0}, DType::Int64).astype(dt);
    const NDArray s = sort_copy(a, -1, SortKind::Quick, false);
    const bool b = dt == DType::Bool;
    CHECK(dbl(s) == (b ? std::vector<double>{0, 0, 1, 1, 1} : std::vector<double>{0, 0, 1, 2, 3}));
    const auto idx = ints(argsort(a, -1, SortKind::Stable, false));
    CHECK(idx == (b ? std::vector<std::int64_t>{1, 4, 0, 2, 3} : std::vector<std::int64_t>{1, 4, 3, 2, 0}));
    const NDArray d = sort_copy(a, -1, SortKind::Stable, true);
    CHECK(dbl(d) == (b ? std::vector<double>{1, 1, 1, 0, 0} : std::vector<double>{3, 2, 1, 0, 0}));
  }
}

TEST_CASE("p09: quicksort survives long inputs (heapsort fallback path)") {
  std::vector<std::int64_t> v;
  for (std::int64_t i = 0; i < 5000; ++i) v.push_back((i * 7919) % 1013);
  const NDArray a = vec(v, DType::Int64);
  const auto s = ints(sort_copy(a, std::nullopt, SortKind::Quick, false));
  for (std::size_t i = 1; i < s.size(); ++i) CHECK(s[i - 1] <= s[i]);
  const auto idx = ints(argsort(a, 0, SortKind::Quick, false));
  for (std::size_t i = 1; i < idx.size(); ++i) {
    CHECK(v[static_cast<std::size_t>(idx[i - 1])] <= v[static_cast<std::size_t>(idx[i])]);
  }
}

TEST_CASE("p09: sort along an axis and in place on a strided view") {
  NDArray a = vec<std::int64_t>({3, 1, 2, 9, 7, 8}, DType::Int64, {2, 3});
  CHECK(ints(sort_copy(a, 0, SortKind::Quick, false)) == (std::vector<std::int64_t>{3, 1, 2, 9, 7, 8}));
  sort_inplace(a, 1, SortKind::Quick, false);
  CHECK(ints(a) == (std::vector<std::int64_t>{1, 2, 3, 7, 8, 9}));
  CHECK_THROWS_KIND(sort_inplace(NDArray::zeros({}, DType::Int64), -1, SortKind::Quick, false), ErrorKind::Index);
}

TEST_CASE("p09: partition / argpartition match NumPy introselect") {
  const NDArray a = vec(kPart, DType::Int64);
  CHECK(ints(partition_copy(a, {10}, -1)) ==
        (std::vector<std::int64_t>{2, 1, 2, 1, 2, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4,
                                   5, 9, 8, 7, 6, 9, 6, 8, 5, 5, 8, 6, 9, 7, 9, 5}));
  CHECK(ints(argpartition(a, {10}, -1)) ==
        (std::vector<std::int64_t>{6, 1, 21, 3, 16, 28, 15, 24, 17, 4, 25, 27, 9, 19, 23, 2,
                                   0, 14, 18, 13, 20, 12, 22, 11, 10, 8, 26, 7, 5, 29, 30, 31}));
  const auto multi = ints(partition_copy(a, {3, -1, 20}, -1));
  std::vector<std::int64_t> sorted = kPart;
  std::sort(sorted.begin(), sorted.end());
  for (const std::size_t k : {3u, 20u, 31u}) CHECK_EQ(multi[k], sorted[k]);
  CHECK_THROWS_KIND(partition_copy(a, {32}, -1), ErrorKind::Value);
}

TEST_CASE("p09: lexsort and searchsorted") {
  const NDArray k0 = vec<std::int64_t>({1, 2}, DType::Int64);
  const NDArray k1 = vec<std::int64_t>({3, 1}, DType::Int64);
  CHECK(ints(lexsort({k0, k1}, -1)) == (std::vector<std::int64_t>{1, 0}));
  CHECK_THROWS_KIND(lexsort({}, -1), ErrorKind::DType);
  const NDArray s = vec<double>({1, 2, 2, 3}, DType::Float64);
  const NDArray keys = vec<double>({2, 0, 4}, DType::Float64);
  CHECK(ints(searchsorted(s, keys, false, std::nullopt)) == (std::vector<std::int64_t>{1, 0, 4}));
  CHECK(ints(searchsorted(s, keys, true, std::nullopt)) == (std::vector<std::int64_t>{3, 0, 4}));
  const NDArray u = vec<double>({3, 1, 2}, DType::Float64);
  const NDArray perm = vec<std::int64_t>({1, 2, 0}, DType::Int64);
  CHECK(ints(searchsorted(u, vec<double>({2.5}, DType::Float64), false, perm)) == (std::vector<std::int64_t>{2}));
  CHECK_THROWS_KIND(searchsorted(u, u, false, vec<std::int64_t>({0, 1, 7}, DType::Int64)), ErrorKind::Value);
}

TEST_CASE("p09: unique1d with indices, inverse and counts") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const auto r = unique1d(vec<double>({nan, 2, 1, nan, 2}, DType::Float64), true, true, true, true);
  const auto v = dbl(r.values);
  CHECK(v.size() == 3 && v[0] == 1 && v[1] == 2 && std::isnan(v[2]));
  CHECK(ints(*r.indices) == (std::vector<std::int64_t>{2, 1, 0}));
  CHECK(ints(*r.inverse) == (std::vector<std::int64_t>{2, 1, 0, 2, 1}));
  CHECK(ints(*r.counts) == (std::vector<std::int64_t>{1, 2, 2}));
  CHECK_EQ(unique1d(vec<double>({nan, nan}, DType::Float64), false, false, false, false).values.size(), 2);
  const auto rows = unique_rows(vec<std::int64_t>({1, 0, 0, 1, 1, 0}, DType::Int64, {3, 2}), true, true, true);
  CHECK(ints(rows.values) == (std::vector<std::int64_t>{0, 1, 1, 0}));
  CHECK(ints(*rows.counts) == (std::vector<std::int64_t>{1, 2}));
}

TEST_CASE("p09: set functions and ediff1d") {
  const NDArray a = vec<std::int64_t>({1, 3, 4, 3}, DType::Int64);
  const NDArray b = vec<std::int64_t>({3, 1, 2, 1}, DType::Int64);
  const auto i = intersect1d(a, b, false, true);
  CHECK(ints(i.values) == (std::vector<std::int64_t>{1, 3}));
  CHECK(ints(*i.indices1) == (std::vector<std::int64_t>{0, 1}));
  CHECK(ints(*i.indices2) == (std::vector<std::int64_t>{1, 0}));
  CHECK(ints(union1d(a, b)) == (std::vector<std::int64_t>{1, 2, 3, 4}));
  CHECK(ints(setxor1d(a, b, false)) == (std::vector<std::int64_t>{2, 4}));
  CHECK(ints(setdiff1d(a, b, false)) == (std::vector<std::int64_t>{4}));
  CHECK(ints(isin(a, b, false)) == (std::vector<std::int64_t>{1, 1, 0, 1}));
  CHECK(ints(ediff1d(vec<std::uint8_t>({1, 0}, DType::UInt8), std::nullopt, std::nullopt)) ==
        (std::vector<std::int64_t>{255}));
  CHECK_THROWS_KIND(ediff1d(vec<bool>({true, false}, DType::Bool), std::nullopt, std::nullopt), ErrorKind::DType);
}
