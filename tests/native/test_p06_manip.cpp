#include <limits>

#include "broadcast.hpp"
#include "creation.hpp"
#include "error.hpp"
#include "layout.hpp"
#include "p06_manip.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {

NDArray iota(const Shape& shape, DType dt = DType::Int64) {
  return arange(0, static_cast<double>(shape_size(shape)), 1, dt).reshape(shape);
}

std::vector<std::int64_t> values(const NDArray& a) {
  std::vector<std::int64_t> v;
  for (std::int64_t i = 0; i < a.size(); ++i) v.push_back(a.get_int64(i));
  return v;
}

}  // namespace

TEST_CASE("p06: concatenate axis/None/dtype/casting/out") {
  const NDArray a = iota({2, 2});
  const NDArray b = iota({1, 2});
  NDArray c = concatenate({a, b}, 0, std::nullopt, Casting::SameKind, nullptr);
  CHECK(c.shape() == Shape({3, 2}));
  CHECK(values(c) == std::vector<std::int64_t>({0, 1, 2, 3, 0, 1}));
  NDArray d = concatenate({a, iota({2, 1})}, -1, std::nullopt, Casting::SameKind, nullptr);
  CHECK(values(d) == std::vector<std::int64_t>({0, 1, 0, 2, 3, 1}));
  NDArray f = concatenate({a, iota({3})}, std::nullopt, std::nullopt, Casting::SameKind, nullptr);
  CHECK(f.shape() == Shape({7}));
  CHECK((concatenate({iota({2}, DType::Int8), iota({1}, DType::Float32)}, 0, std::nullopt,
                       Casting::SameKind, nullptr).dtype() == DType::Float32));
  CHECK_THROWS_KIND(concatenate({iota({2}, DType::Float32)}, 0, DType::Int8, Casting::SameKind,
                                nullptr), ErrorKind::DType);
  CHECK((concatenate({iota({2}, DType::Float32)}, 0, DType::Int8, Casting::Unsafe, nullptr)
               .dtype() == DType::Int8));
  // F-ordered inputs give an F-ordered result (multi-sorted strides).
  const NDArray fa = copy_order(a, DType::Float64, Order::F);
  CHECK(concatenate({fa, fa}, 0, std::nullopt, Casting::SameKind, nullptr).strides() ==
        Strides({8, 32}));
  NDArray out = NDArray::zeros({4}, DType::Int8);
  concatenate({iota({2}), iota({2})}, 0, std::nullopt, Casting::Unsafe, &out);
  CHECK(values(out) == std::vector<std::int64_t>({0, 1, 0, 1}));
  NDArray bad = NDArray::zeros({3}, DType::Int64);
  CHECK_THROWS_KIND(concatenate({iota({2})}, 0, std::nullopt, Casting::SameKind, &bad),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(concatenate({}, 0, std::nullopt, Casting::SameKind, nullptr), ErrorKind::Value);
  CHECK_THROWS_KIND(concatenate({iota({})}, 0, std::nullopt, Casting::SameKind, nullptr),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(concatenate({a, iota({3})}, 0, std::nullopt, Casting::SameKind, nullptr),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(concatenate({a, iota({3, 3})}, 1, std::nullopt, Casting::SameKind, nullptr),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(concatenate({a}, 2, std::nullopt, Casting::SameKind, nullptr), ErrorKind::Index);
  // Empty pieces.
  CHECK(concatenate({iota({0, 2}), a}, 0, std::nullopt, Casting::SameKind, nullptr).shape() ==
        Shape({2, 2}));
}

TEST_CASE("p06: stack / split / unstack") {
  const NDArray a = iota({3});
  NDArray s = stack({a, a}, 1, std::nullopt, Casting::SameKind, nullptr);
  CHECK(s.shape() == Shape({3, 2}));
  CHECK(values(s) == std::vector<std::int64_t>({0, 0, 1, 1, 2, 2}));
  CHECK_THROWS_KIND(stack({a, iota({2})}, 0, std::nullopt, Casting::SameKind, nullptr),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(stack({}, 0, std::nullopt, Casting::SameKind, nullptr), ErrorKind::Value);
  const NDArray v = iota({7});
  auto parts = split_sections(v, 3, 0, false);
  CHECK_EQ(parts.size(), std::size_t{3});
  CHECK(values(parts[0]) == std::vector<std::int64_t>({0, 1, 2}));
  CHECK(values(parts[2]) == std::vector<std::int64_t>({5, 6}));
  CHECK_THROWS_KIND(split_sections(v, 3, 0, true), ErrorKind::Value);
  CHECK_THROWS_KIND(split_sections(v, 0, 0, false), ErrorKind::Value);
  auto at = split_at(iota({6}), {2, 10}, 0);
  CHECK_EQ(at.size(), std::size_t{3});
  CHECK_EQ(at[1].size(), std::int64_t{4});
  CHECK_EQ(at[2].size(), std::int64_t{0});
  auto rev = split_at(iota({5}), {3, 1}, 0);
  CHECK_EQ(rev[1].size(), std::int64_t{0});
  CHECK(values(rev[2]) == std::vector<std::int64_t>({1, 2, 3, 4}));
  auto u = unstack(iota({2, 3}), 1);
  CHECK_EQ(u.size(), std::size_t{3});
  CHECK(values(u[2]) == std::vector<std::int64_t>({2, 5}));
  CHECK(u[0].shares_buffer(u[1]));
  CHECK_THROWS_KIND(unstack(iota({}), 0), ErrorKind::Value);
}

TEST_CASE("p06: tile / repeat / resize") {
  const NDArray a = iota({3});
  NDArray t = tile(a, {2, 2});
  CHECK(t.shape() == Shape({2, 6}));
  CHECK(values(t) == std::vector<std::int64_t>({0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2}));
  CHECK(tile(iota({2, 2}), {2}).shape() == Shape({2, 4}));
  CHECK(tile(iota({2}), {}).shape() == Shape({2}));
  NDArray t1 = tile(a, {1});
  CHECK(!t1.shares_buffer(a));
  CHECK(tile(a, {0}).shape() == Shape({0}));
  CHECK(tile(iota({}), {3}).shape() == Shape({3}));
  CHECK_THROWS_KIND(tile(a, {-1}), ErrorKind::Value);
  CHECK(values(repeat(a, {1, 0, 2}, 0)) == std::vector<std::int64_t>({0, 2, 2}));
  CHECK(values(repeat(iota({2, 2}), {2}, std::nullopt)) ==
        std::vector<std::int64_t>({0, 0, 1, 1, 2, 2, 3, 3}));
  NDArray r1 = repeat(iota({2, 2}), {1, 2}, 1);
  CHECK(values(r1) == std::vector<std::int64_t>({0, 1, 1, 2, 3, 3}));
  NDArray rt = repeat(transpose(iota({2, 2}), {}), {2}, 0);
  CHECK(values(rt) == std::vector<std::int64_t>({0, 2, 0, 2, 1, 3, 1, 3}));
  CHECK(repeat(iota({}), {3}, std::nullopt).shape() == Shape({3}));
  CHECK_THROWS_KIND(repeat(a, {1, 2}, 0), ErrorKind::Value);
  CHECK_THROWS_KIND(repeat(a, {-1}, 0), ErrorKind::Value);
  CHECK_THROWS_KIND(repeat(a, {1}, 1), ErrorKind::Index);
  CHECK(values(resize(a, {2, 4})) == std::vector<std::int64_t>({0, 1, 2, 0, 1, 2, 0, 1}));
  CHECK(values(resize(iota({0}), {2})) == std::vector<std::int64_t>({0, 0}));
  CHECK_THROWS_KIND(resize(a, {-1}), ErrorKind::Value);
  const NDArray f = copy_order(iota({2, 3}), DType::Int64, Order::F);
  NDArray ri = resize_inplace_data(f, {3, 3});
  CHECK(ri.strides() == Strides({8, 24}));
  CHECK(values(ri) == std::vector<std::int64_t>({0, 4, 0, 3, 2, 0, 1, 5, 0}));
  CHECK_THROWS_KIND(resize_inplace_data(slice_axis(a, 0, 0, 3, 2), {2}), ErrorKind::Value);
  CHECK_THROWS_KIND(resize_inplace_data(a.reshape({3, 1}), {2}), ErrorKind::Value);
}

namespace {
PadOptions pad_opts(PadMode m, std::int64_t l, std::int64_t r, std::size_t nd = 1) {
  PadOptions o;
  o.mode = m;
  o.width.assign(nd, {l, r});
  o.stat_length.assign(nd, {-1, -1});
  return o;
}
}  // namespace

TEST_CASE("p06: pad modes") {
  const NDArray a = iota({4});  // 0 1 2 3
  CHECK(values(pad(a, pad_opts(PadMode::Constant, 1, 2))) ==
        std::vector<std::int64_t>({0, 0, 1, 2, 3, 0, 0}));
  CHECK(values(pad(a, pad_opts(PadMode::Edge, 2, 1))) ==
        std::vector<std::int64_t>({0, 0, 0, 1, 2, 3, 3}));
  CHECK(values(pad(a, pad_opts(PadMode::Reflect, 5, 5))) ==
        std::vector<std::int64_t>({1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2}));
  CHECK(values(pad(a, pad_opts(PadMode::Symmetric, 5, 5))) ==
        std::vector<std::int64_t>({3, 3, 2, 1, 0, 0, 1, 2, 3, 3, 2, 1, 0, 0}));
  CHECK(values(pad(a, pad_opts(PadMode::Wrap, 5, 5))) ==
        std::vector<std::int64_t>({3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0}));
  PadOptions odd = pad_opts(PadMode::Reflect, 2, 2);
  odd.odd = true;
  CHECK(values(pad(a, odd)) == std::vector<std::int64_t>({-2, -1, 0, 1, 2, 3, 4, 5}));
  CHECK(values(pad(a, pad_opts(PadMode::Maximum, 1, 1))) ==
        std::vector<std::int64_t>({3, 0, 1, 2, 3, 3}));
  CHECK(values(pad(a, pad_opts(PadMode::Mean, 1, 1))) ==
        std::vector<std::int64_t>({2, 0, 1, 2, 3, 2}));  // 1.5 rounds to even
  CHECK(values(pad(a, pad_opts(PadMode::Median, 1, 0))) ==
        std::vector<std::int64_t>({2, 0, 1, 2, 3}));
  PadOptions sl = pad_opts(PadMode::Minimum, 1, 1);
  sl.stat_length = {{2, 1}};
  CHECK(values(pad(a, sl)) == std::vector<std::int64_t>({0, 0, 1, 2, 3, 3}));
  sl.stat_length = {{0, 1}};
  CHECK_THROWS_KIND(pad(a, sl), ErrorKind::Value);
  PadOptions lr = pad_opts(PadMode::LinearRamp, 2, 2);
  lr.values = arange(0, 2, 1, DType::Float64).reshape({1, 2}).astype(DType::Float64);
  CHECK(values(pad(a, lr)) == std::vector<std::int64_t>({0, 0, 0, 1, 2, 3, 2, 1}));
  NDArray m = pad(iota({2, 2}), pad_opts(PadMode::Edge, 1, 1, 2));
  CHECK(m.shape() == Shape({4, 4}));
  CHECK(values(m) == std::vector<std::int64_t>({0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3}));
  CHECK(pad(copy_order(iota({2, 3}), DType::Int64, Order::F), pad_opts(PadMode::Edge, 1, 1, 2))
            .is_f_contiguous());
  CHECK(values(pad(iota({0}), pad_opts(PadMode::Constant, 1, 1))) == std::vector<std::int64_t>({0, 0}));
  CHECK_THROWS_KIND(pad(iota({0}), pad_opts(PadMode::Edge, 1, 1)), ErrorKind::Value);
  CHECK(pad(iota({0}), pad_opts(PadMode::Edge, 0, 0)).shape() == Shape({0}));
  CHECK_THROWS_KIND(pad(a, pad_opts(PadMode::Edge, -1, 0)), ErrorKind::Value);
}

TEST_CASE("p06: insert_along / delete_along / trim_zeros") {
  const NDArray a = iota({4});
  const NDArray v = arange(10, 12, 1, DType::Float64);
  CHECK(values(insert_along(a, 0, {1, 2}, v)) == std::vector<std::int64_t>({0, 10, 11, 1, 2, 3}));
  CHECK(values(insert_along(a, 0, {5, 0}, v)) == std::vector<std::int64_t>({11, 0, 1, 2, 3, 10}));
  NDArray m = insert_along(iota({2, 2}), 1, {1}, arange(7, 8, 1, DType::Int64));
  CHECK(values(m) == std::vector<std::int64_t>({0, 7, 1, 2, 7, 3}));
  const NDArray f = copy_order(iota({2, 3}), DType::Int64, Order::F);
  CHECK(insert_along(f, 0, {0}, iota({3})).is_f_contiguous());
  CHECK(insert_along(iota({0}), 0, {}, iota({0})).shape() == Shape({0}));
  CHECK_THROWS_KIND(insert_along(a, 0, {6}, v), ErrorKind::Index);
  CHECK_THROWS_KIND(insert_along(a, 0, {1, 1}, v), ErrorKind::Value);
  CHECK_THROWS_KIND(insert_along(a, 1, {0}, v), ErrorKind::Index);
  CHECK(values(delete_along(a, 0, {true, false, false, true})) == std::vector<std::int64_t>({0, 3}));
  CHECK(values(delete_along(iota({2, 3}), 1, {false, true, true})) ==
        std::vector<std::int64_t>({1, 2, 4, 5}));
  CHECK(delete_along(f, 1, {true, false, true}).is_f_contiguous());
  CHECK(delete_along(a, 0, {false, false, false, false}).shape() == Shape({0}));
  CHECK_THROWS_KIND(delete_along(a, 0, {true}), ErrorKind::Value);
  NDArray z = NDArray::zeros({5}, DType::Float64);
  z.set_double(1, 1.0);
  z.set_double(3, 2.0);
  NDArray t = trim_zeros(z, true, true, {true});
  CHECK(t.shape() == Shape({3}));
  CHECK(t.shares_buffer(z));
  CHECK(trim_zeros(z, true, false, {true}).shape() == Shape({4}));
  CHECK(trim_zeros(z, false, true, {true}).shape() == Shape({4}));
  CHECK(trim_zeros(NDArray::zeros({3}, DType::Int8), true, true, {true}).shape() == Shape({0}));
  NDArray z2 = NDArray::zeros({3, 4}, DType::Int32);
  z2.set_int64(5, 1);  // (1, 1)
  z2.set_int64(7, 1);  // (1, 3)
  CHECK(trim_zeros(z2, true, true, {true, true}).shape() == Shape({1, 3}));
  CHECK(trim_zeros(z2, true, true, {false, true}).shape() == Shape({3, 3}));
}

TEST_CASE("p06: flip / roll") {
  const NDArray a = iota({2, 3});
  NDArray f = flip(a, {1}, false);
  CHECK(f.shares_buffer(a));
  CHECK(values(f) == std::vector<std::int64_t>({2, 1, 0, 5, 4, 3}));
  CHECK(values(flip(a, {}, true)) == std::vector<std::int64_t>({5, 4, 3, 2, 1, 0}));
  CHECK(values(flip(flip(a, {0, -1}, false), {}, true)) == values(a));
  CHECK(flip(iota({0, 2}), {0}, false).shape() == Shape({0, 2}));
  CHECK_THROWS_KIND(flip(a, {0, 0}, false), ErrorKind::Value);
  CHECK_THROWS_KIND(flip(a, {2}, false), ErrorKind::Index);
  CHECK(values(roll(a, {1, 1})) == std::vector<std::int64_t>({5, 3, 4, 2, 0, 1}));
  CHECK(values(roll(a, {0, -1})) == std::vector<std::int64_t>({1, 2, 0, 4, 5, 3}));
  CHECK(values(roll(a, {4, 0})) == values(a));
  const NDArray F = copy_order(a, DType::Int64, Order::F);
  CHECK(roll(F, {1, 0}).strides() == Strides({8, 16}));
  CHECK(roll(iota({0}), {3}).shape() == Shape({0}));
}

TEST_CASE("p06: copyto / all_finite") {
  NDArray d = NDArray::zeros({2, 3}, DType::Float64);
  copyto(d, iota({3}), Casting::SameKind, nullptr);
  CHECK(values(d) == std::vector<std::int64_t>({0, 1, 2, 0, 1, 2}));
  NDArray mask = NDArray::zeros({3}, DType::Bool);
  mask.set_int64(1, 1);
  copyto(d, arange(7, 8, 1, DType::Int64), Casting::SameKind, &mask);
  CHECK(values(d) == std::vector<std::int64_t>({0, 7, 2, 0, 7, 2}));
  NDArray i8 = NDArray::zeros({3}, DType::Int8);
  CHECK_THROWS_KIND(copyto(i8, iota({3}, DType::Float32), Casting::SameKind, nullptr), ErrorKind::DType);
  copyto(i8, iota({3}, DType::Float32), Casting::Unsafe, nullptr);
  CHECK_THROWS_KIND(copyto(i8, iota({2}), Casting::Unsafe, nullptr), ErrorKind::Value);
  CHECK_THROWS_KIND(copyto(i8, iota({3}), Casting::Unsafe, &i8), ErrorKind::DType);
  CHECK_THROWS_KIND(copyto(iota({3}), iota({2, 3}), Casting::Unsafe, nullptr), ErrorKind::Value);
  CHECK_THROWS_KIND(copyto(broadcast_to(iota({1}), {3}), iota({3}), Casting::Unsafe, nullptr),
                    ErrorKind::Value);
  CHECK(all_finite(iota({3})));
  NDArray f = NDArray::zeros({3}, DType::Float16);
  CHECK(all_finite(f));
  f.set_double(2, std::numeric_limits<double>::infinity());
  CHECK(!all_finite(f));
  NDArray c = NDArray::zeros({2}, DType::Complex64);
  CHECK(all_finite(c));
}

TEST_CASE("p06: copyto leading length-1 axes") {
  NDArray d = NDArray::zeros({3}, DType::Int64);
  copyto(d, iota({1, 1, 3}), Casting::SameKind, nullptr);
  CHECK(values(d) == std::vector<std::int64_t>({0, 1, 2}));
}
