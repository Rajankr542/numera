#include <cstdint>
#include <optional>
#include <vector>

#include "creation.hpp"
#include "error.hpp"
#include "p08_indexing.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
NDArray ar(std::int64_t n, DType dt = DType::Int64) { return arange(0, static_cast<double>(n), 1, dt); }
NDArray ints(std::initializer_list<std::int64_t> v, Shape shape = {}) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, DType::Int64);
  std::int64_t i = 0;
  for (auto x : v) a.set_int64(i++, x);
  return shape.empty() ? a : a.reshape(shape);
}
NDArray bools(std::initializer_list<int> v) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, DType::Bool);
  std::int64_t i = 0;
  for (auto x : v) a.set_int64(i++, x);
  return a;
}
std::vector<std::int64_t> vals(const NDArray& a) {
  std::vector<std::int64_t> out;
  for (std::int64_t i = 0; i < a.size(); ++i) out.push_back(a.get_int64(i));
  return out;
}
using V = std::vector<std::int64_t>;
}  // namespace

TEST_CASE("p08: clip_mode_from_name") {
  CHECK(clip_mode_from_name("raise") == ClipMode::Raise);
  CHECK(clip_mode_from_name("wrap") == ClipMode::Wrap);
  CHECK(clip_mode_from_name("clip") == ClipMode::Clip);
  CHECK_THROWS_KIND(clip_mode_from_name("bad"), ErrorKind::Value);
}

TEST_CASE("p08: take modes") {
  const NDArray a = ar(6).reshape({2, 3});
  CHECK(vals(take_mode(a, ints({-1, 7}), std::nullopt, ClipMode::Wrap)) == (V{5, 1}));
  CHECK(vals(take_mode(a, ints({-9, 7}), std::nullopt, ClipMode::Clip)) == (V{0, 5}));
  CHECK(vals(take_mode(a, ints({-1}), std::nullopt, ClipMode::Raise)) == (V{5}));
  CHECK_THROWS_KIND(take_mode(a, ints({6}), std::nullopt, ClipMode::Raise), ErrorKind::Index);
  const NDArray t = take_mode(a, ints({2, 0}, {1, 2}), 1, ClipMode::Raise);
  CHECK(t.shape() == (Shape{2, 1, 2}));
  CHECK(vals(t) == (V{2, 0, 5, 3}));
  // Transposed (non-contiguous) source.
  CHECK(vals(take_mode(transpose(a, {}), ints({1}), 0, ClipMode::Raise)) == (V{1, 4}));
  CHECK_THROWS_KIND(take_mode(NDArray::empty({0}, DType::Float64), ints({0}), 0, ClipMode::Wrap),
                    ErrorKind::Index);
  CHECK_EQ(take_mode(NDArray::empty({0}, DType::Float64), NDArray::empty({0}, DType::Int64), 0,
                     ClipMode::Wrap).size(), 0);
  CHECK_THROWS_KIND(take_mode(a, ar(2, DType::Float64), 0, ClipMode::Raise), ErrorKind::DType);
  CHECK_THROWS_KIND(take_mode(a, ints({0}), 2, ClipMode::Raise), ErrorKind::Index);
}

TEST_CASE("p08: take_along_axis / put_along_axis") {
  const NDArray a = ar(6).reshape({2, 3});
  CHECK(vals(take_along_axis(a, ints({2, 0}, {2, 1}), 1)) == (V{2, 3}));
  CHECK(vals(take_along_axis(a, ints({5, 0}), std::nullopt)) == (V{5, 0}));
  CHECK(take_along_axis(ar(3).reshape({1, 3}), ints({0, 2}, {2, 1}), 1).shape() == (Shape{2, 1}));
  CHECK_THROWS_KIND(take_along_axis(a, ints({1}), 1), ErrorKind::Value);
  CHECK_THROWS_KIND(take_along_axis(a, ints({1, 2}, {1, 2}), std::nullopt), ErrorKind::Value);
  CHECK_THROWS_KIND(take_along_axis(a, ints({3}, {1, 1}), 1), ErrorKind::Index);
  CHECK_THROWS_KIND(take_along_axis(a, ar(2, DType::Float64).reshape({2, 1}), 1), ErrorKind::Index);
  NDArray b = a.copy();
  NDArray v = NDArray::empty({}, DType::Int64);
  v.set_int64(0, 99);
  put_along_axis(b, ints({2, 0}, {2, 1}), v, 1);
  CHECK(vals(b) == (V{0, 1, 99, 99, 4, 5}));
  NDArray c = a.copy();
  put_along_axis(c, ints({5}), v, std::nullopt);
  CHECK_EQ(c.get_int64(5), 99);
  NDArray ro = diagonal(a.copy(), 0, 0, 1);  // read-only view
  CHECK_THROWS_KIND(put_along_axis(ro, ints({0}), v, 0), ErrorKind::Value);
}

TEST_CASE("p08: put / putmask / place") {
  NDArray x = ar(5, DType::Float64);
  put(x, ints({0, 7}), ints({9, 8}), ClipMode::Wrap);
  CHECK(vals(x) == (V{9, 1, 8, 3, 4}));
  put(x, ints({-1}), ints({7}), ClipMode::Raise);
  CHECK_EQ(x.get_int64(4), 7);
  CHECK_THROWS_KIND(put(x, ints({5}), ints({1}), ClipMode::Raise), ErrorKind::Index);
  put(x, ints({0}), NDArray::empty({0}, DType::Int64), ClipMode::Raise);  // no-op
  NDArray empty = NDArray::empty({0}, DType::Float64);
  CHECK_THROWS_KIND(put(empty, ints({0}), ints({1}), ClipMode::Raise), ErrorKind::Index);
  // Through a transposed view: flat C order of the view.
  NDArray base = ar(6).reshape({2, 3});
  NDArray tv = transpose(base, {});
  put(tv, ints({1}), ints({9}), ClipMode::Raise);
  CHECK_EQ(base.get_int64(3), 9);

  NDArray y = ar(5, DType::Float64);
  putmask(y, bools({0, 0, 1, 1, 1}), ints({10, 20}), false);
  CHECK(vals(y) == (V{0, 1, 10, 20, 10}));
  NDArray z = ar(5, DType::Float64);
  place(z, bools({0, 0, 1, 1, 1}), ints({10, 20}), false);
  CHECK(vals(z) == (V{0, 1, 10, 20, 10}));
  NDArray w = ar(5, DType::Int32);
  CHECK_THROWS_KIND(putmask(w, bools({1, 0, 0, 0, 0}), ar(1, DType::Float64), true), ErrorKind::DType);
  CHECK_THROWS_KIND(place(w, bools({1, 0, 0, 0, 0}), ar(1, DType::Int64), true), ErrorKind::DType);
  CHECK_THROWS_KIND(putmask(w, bools({1}), ints({1}), false), ErrorKind::Value);
  CHECK_THROWS_KIND(place(w, bools({1, 0, 0, 0, 0}), NDArray::empty({0}, DType::Int32), false),
                    ErrorKind::Value);
  putmask(w, bools({1, 0, 0, 0, 0}), NDArray::empty({0}, DType::Int32), false);  // no-op
  place(w, bools({0, 0, 0, 0, 0}), NDArray::empty({0}, DType::Int32), false);    // no-op
  CHECK(vals(w) == (V{0, 1, 2, 3, 4}));
}

TEST_CASE("p08: choose / compress / extract / select") {
  const std::vector<NDArray> ch{ints({1, 2, 3}), ints({4, 5, 6}), ints({7, 8, 9})};
  CHECK(vals(choose(ints({0, 1, 2}), ch, ClipMode::Raise)) == (V{1, 5, 9}));
  CHECK(vals(choose(ints({-1, 5, 1}), ch, ClipMode::Clip)) == (V{1, 8, 6}));
  CHECK(vals(choose(ints({-1, 5, 1}), ch, ClipMode::Wrap)) == (V{7, 8, 6}));
  CHECK_THROWS_KIND(choose(ints({0, 3, 0}), ch, ClipMode::Raise), ErrorKind::Value);
  CHECK_THROWS_KIND(choose(ints({0}), {}, ClipMode::Raise), ErrorKind::Value);
  CHECK_THROWS_KIND(choose(ar(1, DType::Float64), ch, ClipMode::Raise), ErrorKind::DType);
  // Broadcasting (2,1) against (3,) and dtype promotion.
  const NDArray r = choose(ints({0, 1}, {2, 1}), {ints({1, 2, 3}), ar(3, DType::Float32)}, ClipMode::Raise);
  CHECK(r.shape() == (Shape{2, 3}));
  CHECK(r.dtype() == DType::Float64);

  const NDArray a = ar(6).reshape({2, 3});
  CHECK(vals(compress(bools({0, 1}), a, 0)) == (V{3, 4, 5}));
  CHECK(vals(compress(bools({1, 0, 1, 1}), a, std::nullopt)) == (V{0, 2, 3}));
  CHECK(vals(compress(bools({1, 0}), a, 1)) == (V{0, 3}));
  CHECK_THROWS_KIND(compress(bools({1, 0, 1, 1}), a, 1), ErrorKind::Index);
  CHECK_THROWS_KIND(compress(bools({1}).reshape({1, 1}), a, 0), ErrorKind::Value);
  CHECK(vals(extract(ints({0, 0, 0, 1, 1, 1}, {2, 3}), a)) == (V{3, 4, 5}));
  CHECK(vals(extract(bools({1, 0, 1}), ints({1, 2, 3, 4}))) == (V{1, 3}));

  NDArray dflt = NDArray::empty({}, DType::Int64);
  dflt.set_int64(0, 9);
  const NDArray s = select({bools({1, 0}), bools({1, 1})}, {ints({1, 2}), ints({3, 4})}, dflt);
  CHECK(vals(s) == (V{1, 4}));
  CHECK(vals(select({bools({0, 0})}, {ints({1, 2})}, dflt)) == (V{9, 9}));
  CHECK_THROWS_KIND(select({ints({1, 0})}, {ints({1, 2})}, dflt), ErrorKind::DType);
  CHECK_THROWS_KIND(select({}, {}, dflt), ErrorKind::Value);
  CHECK_THROWS_KIND(select({bools({1})}, {}, dflt), ErrorKind::Value);
  CHECK(vals(none_of({bools({1, 0, 0}), bools({0, 0, 1})})) == (V{0, 1, 0}));
}

TEST_CASE("p08: argwhere / flatnonzero / count_nonzero") {
  const NDArray a = ints({0, 3, 4, 0}, {2, 2});
  const NDArray w = argwhere(a);
  CHECK(w.shape() == (Shape{2, 2}));
  CHECK(vals(w) == (V{0, 1, 1, 0}));
  CHECK(argwhere(ints({5}).reshape({})).shape() == (Shape{1, 0}));
  CHECK(argwhere(ints({0}).reshape({})).shape() == (Shape{0, 0}));
  CHECK(vals(flatnonzero(a)) == (V{1, 2}));
  CHECK(vals(flatnonzero(ints({3}).reshape({}))) == (V{0}));
  CHECK_EQ(count_nonzero(a, std::nullopt, false).get_int64(0), 2);
  CHECK(count_nonzero(a, std::nullopt, false).ndim() == 0);
  const NDArray c = count_nonzero(a, std::vector<std::int64_t>{0}, true);
  CHECK(c.shape() == (Shape{1, 2}));
  CHECK(vals(c) == (V{1, 1}));
  NDArray f = NDArray::empty({2}, DType::Float64);
  f.set_double(0, std::nan(""));
  f.set_double(1, -0.0);
  CHECK_EQ(count_nonzero(f, std::nullopt, false).get_int64(0), 1);
  CHECK_EQ(count_nonzero(NDArray::empty({0}, DType::Int8), std::nullopt, false).get_int64(0), 0);
}

TEST_CASE("p08: ravel_multi_index / unravel_index") {
  const NDArray r = ravel_multi_index({ints({1, 2}), ints({3, 9})}, {3, 4},
                                      {ClipMode::Raise, ClipMode::Clip}, Order::C);
  CHECK(vals(r) == (V{7, 11}));
  CHECK_EQ(ravel_multi_index({ints({1}), ints({2})}, {3, 4}, {ClipMode::Raise}, Order::F).get_int64(0), 7);
  CHECK_EQ(ravel_multi_index({ints({-1}), ints({5})}, {3, 4}, {ClipMode::Wrap}, Order::C).get_int64(0), 9);
  CHECK_THROWS_KIND(ravel_multi_index({ints({3}), ints({1})}, {3, 4}, {ClipMode::Raise}, Order::C),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(ravel_multi_index({ints({1})}, {3, 4}, {ClipMode::Raise}, Order::C), ErrorKind::Value);
  CHECK_THROWS_KIND(ravel_multi_index({ints({1}), ints({1})}, {3, 4}, {ClipMode::Raise}, Order::K),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(ravel_multi_index({ints({1}), ints({1})}, {3, 4},
                                      {ClipMode::Raise, ClipMode::Raise, ClipMode::Raise}, Order::C),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(ravel_multi_index({ar(1, DType::Float64), ints({1})}, {3, 4}, {ClipMode::Raise},
                                      Order::C),
                    ErrorKind::DType);
  const auto u = unravel_index(ints({5, 7}), {3, 4}, Order::C);
  CHECK_EQ(u.size(), 2u);
  CHECK(vals(u[0]) == (V{1, 1}));
  CHECK(vals(u[1]) == (V{1, 3}));
  const auto uf = unravel_index(ints({5}).reshape({}), {3, 4}, Order::F);
  CHECK(uf[0].ndim() == 0);
  CHECK_EQ(uf[0].get_int64(0), 2);
  CHECK_EQ(uf[1].get_int64(0), 1);
  CHECK_THROWS_KIND(unravel_index(ints({12}), {3, 4}, Order::C), ErrorKind::Value);
  CHECK_THROWS_KIND(unravel_index(ints({-1}), {3, 4}, Order::C), ErrorKind::Value);
  CHECK_THROWS_KIND(unravel_index(ar(1, DType::Float64), {3, 4}, Order::C), ErrorKind::DType);
  CHECK_EQ(unravel_index(NDArray::empty({0}, DType::Int64), {3, 4}, Order::C)[0].size(), 0);
}

TEST_CASE("p08: diagonal view / trace") {
  NDArray a = ar(24).reshape({2, 3, 4});
  const NDArray d = diagonal(a, 1, 0, 2);
  CHECK(d.shape() == (Shape{3, 2}));
  CHECK(vals(d) == (V{1, 14, 5, 18, 9, 22}));
  CHECK(!d.writeable());
  CHECK(d.shares_buffer(a));
  CHECK(d.strides() == (Strides{32, 104}));
  const NDArray m = ar(9, DType::Float64).reshape({3, 3});
  CHECK(vals(diagonal(m, -1, 0, 1)) == (V{3, 7}));
  CHECK_EQ(diagonal(m, 5, 0, 1).size(), 0);
  CHECK_EQ(diagonal(m, -5, 0, 1).size(), 0);
  CHECK(vals(diagonal(m, 0, 1, 0)) == (V{0, 4, 8}));
  CHECK_THROWS_KIND(diagonal(ar(3), 0, 0, 1), ErrorKind::Value);
  CHECK_THROWS_KIND(diagonal(m, 0, 1, -1), ErrorKind::Value);
  CHECK_THROWS_KIND(diagonal(m, 0, 0, 2), ErrorKind::Index);
  const NDArray t = trace(a, 1, 0, 2, std::nullopt);
  CHECK(vals(t) == (V{15, 23, 31}));
  CHECK(trace(eye(3, 3, 0, DType::Int8), 0, 0, 1, std::nullopt).dtype() == DType::Int64);
  CHECK(trace(eye(2, 2, 0, DType::UInt8), 0, 0, 1, std::nullopt).dtype() == DType::UInt64);
  CHECK(trace(eye(2, 2, 0, DType::Bool), 0, 0, 1, std::nullopt).dtype() == DType::Int64);
  CHECK(trace(eye(2, 2, 0, DType::UInt8), 0, 0, 1, DType::Float32).dtype() == DType::Float32);
  CHECK_EQ(trace(NDArray::empty({0, 0}, DType::Float64), 0, 0, 1, std::nullopt).get_double(0), 0.0);
}
