#include <cstdint>
#include <vector>

#include "error.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"
#include "ufunc_methods.hpp"
#include "ufunc_registry.hpp"

using namespace nativpy;

namespace {
NDArray mat_d(const Shape& shape, std::vector<double> v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty(shape, dt);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_double(static_cast<std::int64_t>(i), v[i]);
  return a;
}
NDArray bools(const Shape& shape, std::vector<int> v) {
  NDArray a = NDArray::empty(shape, DType::Bool);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_int64(static_cast<std::int64_t>(i), v[i]);
  return a;
}
const Ufunc& U(const char* name) { return *find_ufunc(name); }
UfuncReduceOptions ax(std::vector<std::int64_t> axes) {
  UfuncReduceOptions o;
  o.axis = std::move(axes);
  return o;
}
}  // namespace

TEST_CASE("ufunc.reduce: default axis 0 and explicit axes") {
  const NDArray a = mat_d({2, 3}, {1, 2, 3, 4, 5, 6});
  const NDArray r0 = ufunc_reduce(U("add"), a, nullptr, {});
  CHECK(r0.shape() == Shape({3}));
  CHECK_EQ(r0.get_double(0), 5.0);
  CHECK_EQ(r0.get_double(2), 9.0);
  const NDArray r1 = ufunc_reduce(U("subtract"), a, nullptr, ax({1}));
  CHECK(r1.shape() == Shape({2}));
  CHECK_EQ(r1.get_double(0), -4.0);  // 1 - 2 - 3
  CHECK_EQ(r1.get_double(1), -7.0);  // 4 - 5 - 6
  const NDArray rn = ufunc_reduce(U("add"), a, nullptr, ax({-1}));
  CHECK_EQ(rn.get_double(1), 15.0);
  UfuncReduceOptions all;
  all.all_axes = true;
  const NDArray rall = ufunc_reduce(U("multiply"), a, nullptr, all);
  CHECK(rall.ndim() == 0);
  CHECK_EQ(rall.get_double(0), 720.0);
  const NDArray rt = ufunc_reduce(U("add"), a, nullptr, ax({0, 1}));
  CHECK_EQ(rt.get_double(0), 21.0);
}

TEST_CASE("ufunc.reduce: non-contiguous input and divide/power") {
  const NDArray a = mat_d({2, 3}, {8, 2, 3, 4, 5, 6});
  const NDArray t = transpose(a, {});  // 3 x 2, strided
  const NDArray r = ufunc_reduce(U("divide"), t, nullptr, ax({1}));
  CHECK_EQ(r.get_double(0), 2.0);  // 8 / 4
  CHECK_EQ(r.get_double(2), 0.5);  // 3 / 6
  const NDArray p = ufunc_reduce(U("power"), mat_d({3}, {2, 3, 2}), nullptr, {});
  CHECK_EQ(p.get_double(0), 64.0);  // (2^3)^2
}

TEST_CASE("ufunc.reduce: keepdims, axis=() and 0-d input") {
  const NDArray a = mat_d({2, 3}, {1, 2, 3, 4, 5, 6});
  UfuncReduceOptions o = ax({1});
  o.keepdims = true;
  const NDArray r = ufunc_reduce(U("add"), a, nullptr, o);
  CHECK(r.shape() == Shape({2, 1}));
  CHECK_EQ(r.get_double(1), 15.0);
  const NDArray none = ufunc_reduce(U("subtract"), a, nullptr, ax({}));
  CHECK(none.shape() == Shape({2, 3}));
  CHECK_EQ(none.get_double(4), 5.0);
  const NDArray s = ufunc_reduce(U("subtract"), mat_d({}, {7}), nullptr, {});
  CHECK(s.ndim() == 0);
  CHECK_EQ(s.get_double(0), 7.0);
}

TEST_CASE("ufunc.reduce: dtypes") {
  NDArray i8 = NDArray::empty({3}, DType::Int8);
  for (std::int64_t k = 0; k < 3; ++k) i8.set_int64(k, 100);
  CHECK(ufunc_reduce(U("add"), i8, nullptr, {}).dtype() == DType::Int64);
  CHECK_EQ(ufunc_reduce(U("add"), i8, nullptr, {}).get_int64(0), 300);
  CHECK(ufunc_reduce(U("subtract"), i8, nullptr, {}).dtype() == DType::Int8);
  CHECK(ufunc_reduce(U("divide"), i8, nullptr, {}).dtype() == DType::Float64);
  UfuncReduceOptions o;
  o.dtype = DType::Int8;
  const NDArray w = ufunc_reduce(U("add"), i8, nullptr, o);
  CHECK(w.dtype() == DType::Int8);
  CHECK_EQ(w.get_int64(0), 44);  // 300 wraps to 44
  NDArray b = bools({2}, {1, 1});
  CHECK(ufunc_reduce(U("add"), b, nullptr, {}).dtype() == DType::Int64);
  CHECK_THROWS_KIND(ufunc_reduce(U("subtract"), b, nullptr, {}), ErrorKind::DType);
}

TEST_CASE("ufunc.reduce: initial and empty input") {
  const NDArray a = mat_d({3}, {1, 2, 3});
  UfuncReduceOptions o;
  o.initial = 10.0;
  CHECK_EQ(ufunc_reduce(U("subtract"), a, nullptr, o).get_double(0), 4.0);
  CHECK_EQ(ufunc_reduce(U("add"), a, nullptr, o).get_double(0), 16.0);
  const NDArray e = NDArray::empty({0, 3}, DType::Float64);
  const NDArray s = ufunc_reduce(U("add"), e, nullptr, {});
  CHECK(s.shape() == Shape({3}));
  CHECK_EQ(s.get_double(0), 0.0);
  CHECK_EQ(ufunc_reduce(U("multiply"), e, nullptr, {}).get_double(1), 1.0);
  CHECK_THROWS_KIND(ufunc_reduce(U("subtract"), e, nullptr, {}), ErrorKind::Value);
  CHECK_EQ(ufunc_reduce(U("subtract"), e, nullptr, o).get_double(2), 10.0);
  // Empty result: nothing to compute, so no identity is needed.
  CHECK(ufunc_reduce(U("subtract"), e, nullptr, ax({1})).shape() == Shape({0}));
}

TEST_CASE("ufunc.reduce: where") {
  const NDArray a = mat_d({2, 3}, {1, 2, 3, 4, 5, 6});
  UfuncReduceOptions o = ax({1});
  o.where = bools({3}, {1, 0, 1});
  const NDArray r = ufunc_reduce(U("add"), a, nullptr, o);
  CHECK_EQ(r.get_double(0), 4.0);
  CHECK_EQ(r.get_double(1), 10.0);
  CHECK_THROWS_KIND(ufunc_reduce(U("subtract"), a, nullptr, o), ErrorKind::Value);
  o.initial = 0.0;
  const NDArray s = ufunc_reduce(U("subtract"), a, nullptr, o);
  CHECK_EQ(s.get_double(0), -4.0);  // 0 - 1 - 3
  CHECK_EQ(s.get_double(1), -10.0);
  UfuncReduceOptions bad = ax({1});
  bad.where = mat_d({3}, {1, 0, 1});
  CHECK_THROWS_KIND(ufunc_reduce(U("add"), a, nullptr, bad), ErrorKind::DType);
}

TEST_CASE("ufunc.reduce: out") {
  const NDArray a = mat_d({2, 3}, {1, 2, 3, 4, 5, 6});
  NDArray out = NDArray::zeros({3}, DType::Float32);
  const NDArray r = ufunc_reduce(U("add"), a, &out, {});
  CHECK(r.shares_buffer(out));
  CHECK_EQ(out.get_double(1), 7.0);
  NDArray wrong = NDArray::zeros({2}, DType::Float64);
  CHECK_THROWS_KIND(ufunc_reduce(U("add"), a, &wrong, {}), ErrorKind::Value);
  const NDArray ro = NDArray::zeros({3}, DType::Float64).as_readonly();
  CHECK_THROWS_KIND(ufunc_reduce(U("add"), a, &ro, {}), ErrorKind::Value);
  // In place: out overlaps the input.
  NDArray v = mat_d({2, 2}, {1, 2, 3, 4});
  const NDArray row0 = v.view({2}, {8}, 0);
  ufunc_reduce(U("add"), v, &row0, {});
  CHECK_EQ(v.get_double(0), 4.0);
  CHECK_EQ(v.get_double(1), 6.0);
}

TEST_CASE("ufunc.reduce: errors") {
  const NDArray a = mat_d({2, 3}, {1, 2, 3, 4, 5, 6});
  CHECK_THROWS_KIND(ufunc_reduce(U("negative"), a, nullptr, {}), ErrorKind::Value);
  CHECK_THROWS_KIND(ufunc_reduce(U("subtract"), a, nullptr, ax({0, 1})), ErrorKind::Value);
  CHECK_THROWS_KIND(ufunc_reduce(U("add"), a, nullptr, ax({2})), ErrorKind::Index);
  CHECK_THROWS_KIND(ufunc_reduce(U("add"), a, nullptr, ax({0, 0})), ErrorKind::Value);
  NDArray neg = NDArray::empty({2}, DType::Int64);
  neg.set_int64(0, 2);
  neg.set_int64(1, -1);
  CHECK_THROWS_KIND(ufunc_reduce(U("power"), neg, nullptr, {}), ErrorKind::Value);
}

TEST_CASE("ufunc.accumulate") {
  const NDArray a = mat_d({2, 3}, {1, 2, 3, 4, 5, 6});
  const NDArray r0 = ufunc_accumulate(U("add"), a, nullptr, {});
  CHECK(r0.shape() == Shape({2, 3}));
  CHECK_EQ(r0.get_double(3), 5.0);
  CHECK_EQ(r0.get_double(5), 9.0);
  const NDArray r1 = ufunc_accumulate(U("subtract"), a, nullptr, ax({1}));
  CHECK_EQ(r1.get_double(1), -1.0);
  CHECK_EQ(r1.get_double(2), -4.0);
  CHECK_EQ(r1.get_double(5), -7.0);
  const NDArray t = ufunc_accumulate(U("multiply"), transpose(a, {}), nullptr, ax({1}));
  CHECK_EQ(t.get_double(1), 4.0);   // [1, 1*4]
  CHECK_EQ(t.get_double(5), 18.0);  // [3, 3*6]
  NDArray i8 = NDArray::empty({2}, DType::Int8);
  i8.set_int64(0, 100);
  i8.set_int64(1, 100);
  CHECK(ufunc_accumulate(U("add"), i8, nullptr, {}).dtype() == DType::Int64);
  CHECK(ufunc_accumulate(U("add"), NDArray::empty({0, 3}, DType::Float64), nullptr, {}).shape() ==
        Shape({0, 3}));
  NDArray out = NDArray::zeros({2, 3}, DType::Float64);
  ufunc_accumulate(U("add"), a, &out, {});
  CHECK_EQ(out.get_double(4), 7.0);
}

TEST_CASE("ufunc.accumulate: errors") {
  const NDArray a = mat_d({2, 3}, {1, 2, 3, 4, 5, 6});
  CHECK_THROWS_KIND(ufunc_accumulate(U("abs"), a, nullptr, {}), ErrorKind::Value);
  CHECK_THROWS_KIND(ufunc_accumulate(U("add"), mat_d({}, {1}), nullptr, {}), ErrorKind::DType);
  CHECK_THROWS_KIND(ufunc_accumulate(U("add"), a, nullptr, ax({0, 1})), ErrorKind::Value);
  UfuncReduceOptions all;
  all.all_axes = true;
  CHECK_THROWS_KIND(ufunc_accumulate(U("add"), a, nullptr, all), ErrorKind::Value);
  CHECK_THROWS_KIND(ufunc_accumulate(U("add"), a, nullptr, ax({5})), ErrorKind::Index);
  NDArray wrong = NDArray::zeros({3}, DType::Float64);
  CHECK_THROWS_KIND(ufunc_accumulate(U("add"), a, &wrong, {}), ErrorKind::Value);
}

namespace {
NDArray ints(const Shape& shape, std::vector<std::int64_t> v) {
  NDArray a = NDArray::empty(shape, DType::Int64);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_int64(static_cast<std::int64_t>(i), v[i]);
  return a;
}
}  // namespace

TEST_CASE("ufunc.outer") {
  const NDArray r = ufunc_outer(U("multiply"), mat_d({2}, {1, 2}), mat_d({3}, {3, 4, 5}), nullptr, {});
  CHECK(r.shape() == Shape({2, 3}));
  CHECK_EQ(r.get_double(5), 10.0);
  const NDArray s = ufunc_outer(U("subtract"), mat_d({2, 1}, {1, 2}), mat_d({2}, {10, 20}), nullptr, {});
  CHECK(s.shape() == Shape({2, 1, 2}));
  CHECK_EQ(s.get_double(3), -18.0);
  CHECK_THROWS_KIND(ufunc_outer(U("negative"), mat_d({1}, {1}), mat_d({1}, {1}), nullptr, {}),
                    ErrorKind::Value);
}

TEST_CASE("ufunc.reduceat") {
  const NDArray a = mat_d({8}, {0, 1, 2, 3, 4, 5, 6, 7});
  const NDArray r = ufunc_reduceat(U("add"), a, ints({4}, {0, 4, 1, 5}), nullptr, {});
  CHECK(r.shape() == Shape({4}));
  CHECK_EQ(r.get_double(0), 6.0);   // 0..3
  CHECK_EQ(r.get_double(1), 4.0);   // 4 > 1: a[4]
  CHECK_EQ(r.get_double(2), 10.0);  // 1..4
  CHECK_EQ(r.get_double(3), 18.0);  // 5..7
  const NDArray m = mat_d({2, 3}, {1, 2, 3, 4, 5, 6});
  const NDArray c = ufunc_reduceat(U("multiply"), m, ints({2}, {0, 2}), nullptr, ax({1}));
  CHECK(c.shape() == Shape({2, 2}));
  CHECK_EQ(c.get_double(0), 2.0);
  CHECK_EQ(c.get_double(3), 6.0);
  CHECK_THROWS_KIND(ufunc_reduceat(U("add"), a, ints({1}, {8}), nullptr, {}), ErrorKind::Index);
}

TEST_CASE("ufunc.at") {
  NDArray a = mat_d({4}, {1, 2, 3, 4});
  const NDArray one = mat_d({}, {1});
  ufunc_at(U("add"), a, {ints({3}, {0, 0, 2})}, &one);
  CHECK_EQ(a.get_double(0), 3.0);  // repeated index applies twice
  CHECK_EQ(a.get_double(2), 4.0);
  ufunc_at(U("negative"), a, {ints({1}, {1})}, nullptr);
  CHECK_EQ(a.get_double(1), -2.0);
  NDArray m = mat_d({2, 2}, {1, 2, 3, 4});
  const NDArray v = mat_d({2}, {10, 20});
  ufunc_at(U("multiply"), m, {ints({1}, {1})}, &v);
  CHECK_EQ(m.get_double(2), 30.0);
  CHECK_EQ(m.get_double(3), 80.0);
  CHECK_THROWS_KIND(ufunc_at(U("add"), a, {ints({1}, {9})}, &one), ErrorKind::Index);
  CHECK_EQ(a.get_double(0), 3.0);
  CHECK_THROWS_KIND(ufunc_at(U("add"), a, {ints({1}, {0})}, nullptr), ErrorKind::Value);
}
