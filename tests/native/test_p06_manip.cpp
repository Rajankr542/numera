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
