#include <cstdint>
#include <optional>

#include "creation.hpp"
#include "error.hpp"
#include "indexing.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
using K = IndexItem;
const auto all = K::slice(std::nullopt, std::nullopt, std::nullopt);
NDArray ar(std::int64_t n) { return arange(0, static_cast<double>(n), 1, DType::Int64); }
NDArray ints(std::initializer_list<std::int64_t> v) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, DType::Int64);
  std::int64_t i = 0;
  for (auto x : v) a.set_int64(i++, x);
  return a;
}
}  // namespace

TEST_CASE("indexing: slice_indices matches Python") {
  auto b = slice_indices(std::nullopt, std::nullopt, -1, 5);
  CHECK_EQ(b.start, 4);
  CHECK_EQ(b.length, 5);
  b = slice_indices(-100, 100, 2, 5);
  CHECK_EQ(b.start, 0);
  CHECK_EQ(b.length, 3);
  b = slice_indices(3, 1, 1, 5);
  CHECK_EQ(b.length, 0);
  b = slice_indices(-1, -6, -2, 5);
  CHECK_EQ(b.start, 4);
  CHECK_EQ(b.length, 3);
  CHECK_THROWS_KIND(slice_indices(0, 1, 0, 5), ErrorKind::Value);
}

TEST_CASE("indexing: basic views") {
  NDArray a = ar(24).reshape({2, 3, 4});
  NDArray r = get_index(a, {K::integer_(1)});
  CHECK(r.shape() == Shape({3, 4}));
  CHECK(r.shares_buffer(a));
  CHECK_EQ(r.get_int64(0), 12);
  r = get_index(a, {K::integer_(-1), K::integer_(-1), K::integer_(-1)});
  CHECK(r.shape() == Shape({}));
  CHECK_EQ(r.get_int64(0), 23);
  r = get_index(a, {all, K::slice(std::nullopt, std::nullopt, -1)});
  CHECK(r.strides() == Strides({96, -32, 8}));
  CHECK_EQ(r.get_int64(0), 8);
  r = get_index(a, {K::ellipsis(), K::integer_(0)});
  CHECK(r.shape() == Shape({2, 3}));
  CHECK_EQ(r.get_int64(1), 4);
  r = get_index(a, {K::new_axis(), K::ellipsis(), K::new_axis()});
  CHECK(r.shape() == Shape({1, 2, 3, 4, 1}));
  r = get_index(a, {all, K::slice(5, 9, std::nullopt)});
  CHECK(r.shape() == Shape({2, 0, 4}));
  CHECK_THROWS_KIND(get_index(a, {K::integer_(2)}), ErrorKind::Index);
  CHECK_THROWS_KIND(get_index(a, {all, all, all, all}), ErrorKind::Index);
  CHECK_THROWS_KIND(get_index(a, {K::ellipsis(), K::ellipsis()}), ErrorKind::Index);
}

TEST_CASE("indexing: advanced and boolean") {
  NDArray a = ar(24).reshape({2, 3, 4});
  NDArray r = get_index(a, {all, K::array_(ints({0, 2}))});
  CHECK(r.shape() == Shape({2, 2, 4}));
  CHECK(!r.shares_buffer(a));
  CHECK_EQ(r.get_int64(4), 8);
  r = get_index(a, {K::array_(ints({0, 1})), all, K::array_(ints({1, 2}))});
  CHECK(r.shape() == Shape({2, 3}));  // non-adjacent: broadcast dims first
  CHECK_EQ(r.get_int64(3), 14);  // a[1, 0, 2]
  r = get_index(a, {K::integer_(0), all, K::array_(ints({1, 2}))});
  CHECK(r.shape() == Shape({2, 3}));
  CHECK_EQ(r.get_int64(1), 5);
  r = get_index(a, {all, K::array_(ints({-1}))});
  CHECK_EQ(r.get_int64(0), 8);
  CHECK_THROWS_KIND(get_index(a, {K::array_(ints({2}))}), ErrorKind::Index);
  NDArray mask = NDArray::zeros({2, 3, 4}, DType::Bool);
  mask.set_int64(5, 1);
  mask.set_int64(20, 1);
  r = get_index(a, {K::array_(mask)});
  CHECK(r.shape() == Shape({2}));
  CHECK_EQ(r.get_int64(1), 20);
  CHECK_THROWS_KIND(get_index(a, {K::array_(NDArray::zeros({2, 2}, DType::Bool))}),
                    ErrorKind::Index);
  NDArray t = NDArray::zeros({}, DType::Bool);
  t.set_int64(0, 1);
  CHECK(get_index(a, {K::array_(t)}).shape() == Shape({1, 2, 3, 4}));
  CHECK_THROWS_KIND(get_index(a, {K::array_(NDArray::zeros({1}, DType::Float64))}),
                    ErrorKind::Index);
}

TEST_CASE("indexing: assignment, nonzero, take, where") {
  NDArray a = ar(12).reshape({3, 4});
  set_index(a, {all, K::integer_(0)}, ones({}, DType::Float64));
  CHECK_EQ(a.get_int64(4), 1);
  set_index(a, {K::array_(ints({0, 0})), K::array_(ints({1, 1}))}, ints({7, 9}));
  CHECK_EQ(a.get_int64(1), 9);  // last write wins
  NDArray b = ar(5);
  set_index(b, {K::slice(1, std::nullopt, std::nullopt)},
            get_index(b, {K::slice(std::nullopt, -1, std::nullopt)}));  // overlap
  CHECK_EQ(b.get_int64(4), 3);
  CHECK_THROWS_KIND(set_index(a, {all}, ints({1, 2})), ErrorKind::Broadcast);
  auto nz = nonzero(ints({0, 3, 0, 5}));
  CHECK_EQ(nz.size(), std::size_t{1});
  CHECK_EQ(nz[0].get_int64(1), 3);
  CHECK_EQ(take(ar(12).reshape({3, 4}), ints({-1}), std::nullopt).get_int64(0), 11);
  CHECK(take(ar(12).reshape({3, 4}), ints({0, 2}), 1).shape() == Shape({3, 2}));
  NDArray w = where(ints({1, 0}), ints({5, 5}), ones({}, DType::Float64));
  CHECK(w.dtype() == DType::Float64);
  CHECK_EQ(w.get_double(1), 1.0);
}
