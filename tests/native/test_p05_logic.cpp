#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "error.hpp"
#include "fp_errors.hpp"
#include "test_harness.hpp"
#include "ufunc_methods.hpp"
#include "ufunc_registry.hpp"

using namespace nativpy;

namespace {
const Ufunc& U(const char* name) { return *find_ufunc(name); }
NDArray vec_d(std::vector<double> v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_double(static_cast<std::int64_t>(i), v[i]);
  return a;
}
NDArray vec_i(std::vector<std::int64_t> v, DType dt = DType::Int64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  for (std::size_t i = 0; i < v.size(); ++i) a.set_int64(static_cast<std::int64_t>(i), v[i]);
  return a;
}
NDArray cvec(std::vector<std::complex<double>> v) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, DType::Complex128);
  std::memcpy(a.data(), v.data(), v.size() * sizeof(std::complex<double>));
  return a;
}
std::vector<std::int64_t> ints(const NDArray& a) {
  std::vector<std::int64_t> r;
  for (std::int64_t i = 0; i < a.size(); ++i) r.push_back(a.get_int64(i));
  return r;
}
using IV = std::vector<std::int64_t>;
NDArray bin(const char* name, const NDArray& a, const NDArray& b) {
  return binary(U(name), a, b, nullptr, {});
}
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
}  // namespace

TEST_CASE("p05 compare: bool output for every real dtype") {
  for (DType dt : {DType::Bool, DType::Int8, DType::UInt8, DType::Int16, DType::UInt16, DType::Int32,
                   DType::UInt32, DType::Int64, DType::UInt64, DType::Float16, DType::Float32,
                   DType::Float64}) {
    const NDArray a = vec_d({0, 1, 1}, dt);
    const NDArray b = vec_d({1, 1, 0}, dt);
    const NDArray r = bin("less", a, b);
    CHECK(r.dtype() == DType::Bool);
    CHECK(ints(r) == IV({1, 0, 0}));
    CHECK(ints(bin("equal", a, b)) == IV({0, 1, 0}));
    CHECK(ints(bin("notEqual", a, b)) == IV({1, 0, 1}));
    CHECK(ints(bin("lessEqual", a, b)) == IV({1, 1, 0}));
    CHECK(ints(bin("greater", a, b)) == IV({0, 0, 1}));
    CHECK(ints(bin("greaterEqual", a, b)) == IV({0, 1, 1}));
  }
}

TEST_CASE("p05 compare: mixed dtypes promote, NaN compares false") {
  CHECK(ints(bin("less", vec_i({1, 2}, DType::Int8), vec_d({1.5, 1.5}))) == IV({1, 0}));
  const NDArray n = vec_d({kNaN, 1.0});
  const NDArray m = vec_d({1.0, kNaN});
  for (const char* op : {"equal", "less", "lessEqual", "greater", "greaterEqual"}) {
    CHECK(ints(bin(op, n, m)) == IV({0, 0}));
  }
  CHECK(ints(bin("notEqual", n, n)) == IV({1, 0}));
}

TEST_CASE("p05 compare: complex lexicographic order (NumPy CLT/CLE)") {
  using C = std::complex<double>;
  const NDArray a = cvec({C(1, 2), C(1, 1), C(1, 1), C(0, 5), C(1, kNaN), C(kNaN, 0)});
  const NDArray b = cvec({C(1, 1), C(1, 2), C(1, 1), C(1, -5), C(2, 0), C(1, 0)});
  CHECK(ints(bin("less", a, b)) == IV({0, 1, 0, 1, 0, 0}));
  CHECK(ints(bin("lessEqual", a, b)) == IV({0, 1, 1, 1, 0, 0}));
  CHECK(ints(bin("greater", a, b)) == IV({1, 0, 0, 0, 0, 0}));
  CHECK(ints(bin("greaterEqual", a, b)) == IV({1, 0, 1, 0, 0, 0}));
  CHECK(ints(bin("equal", a, b)) == IV({0, 0, 1, 0, 0, 0}));
}

TEST_CASE("p05 compare: complex NaN raises invalid like NumPy") {
  const ErrState old = get_errstate();
  ErrState s = old;
  s.invalid = FpMode::Raise;
  set_errstate(s);
  using C = std::complex<double>;
  CHECK_THROWS_KIND(bin("less", cvec({C(kNaN, 0)}), cvec({C(1, 0)})), ErrorKind::FloatingPoint);
  // equal is quiet; real float compares are quiet.
  CHECK(ints(bin("equal", cvec({C(kNaN, 0)}), cvec({C(1, 0)}))) == IV({0}));
  CHECK(ints(bin("less", vec_d({kNaN}), vec_d({1}))) == IV({0}));
  set_errstate(old);
}

TEST_CASE("p05 compare: dtype= only bool; out casting") {
  UfuncParams p;
  p.dtype = DType::Float64;
  CHECK_THROWS_KIND(binary(U("equal"), vec_d({1}), vec_d({1}), nullptr, p), ErrorKind::DType);
  NDArray out = NDArray::zeros({2}, DType::Float64);
  binary(U("less"), vec_d({0, 2}), vec_d({1, 1}), &out, {});
  CHECK_EQ(out.get_double(0), 1.0);
  CHECK_EQ(out.get_double(1), 0.0);
}

TEST_CASE("p05 compare: reduce needs bool input") {
  const NDArray b = vec_i({1, 0, 0}, DType::Bool);
  const NDArray r = ufunc_reduce(U("equal"), b, nullptr, {});
  CHECK(r.dtype() == DType::Bool);
  CHECK_EQ(r.get_int64(0), 1);  // (1 == 0) == 0
  CHECK_THROWS_KIND(ufunc_reduce(U("equal"), vec_i({1, 1}), nullptr, {}), ErrorKind::DType);
  UfuncReduceOptions o;
  o.dtype = DType::Bool;
  CHECK_EQ(ufunc_reduce(U("equal"), vec_i({1, 2}), nullptr, o).get_int64(0), 1);  // 1, 2 -> True, True
  const NDArray acc = ufunc_accumulate(U("equal"), b, nullptr, {});
  CHECK(ints(acc) == IV({1, 0, 1}));
  UfuncReduceOptions two;
  two.axis = std::vector<std::int64_t>{0, 0};
  CHECK_THROWS_KIND(ufunc_reduce(U("less"), b, nullptr, two), ErrorKind::Value);
}

TEST_CASE("p05 compare: empty and broadcast") {
  const NDArray e = NDArray::empty({0}, DType::Float64);
  CHECK_EQ(bin("less", e, vec_d({1})).size(), 0);
  NDArray col = NDArray::empty({2, 1}, DType::Int64);
  col.set_int64(0, 1);
  col.set_int64(1, 3);
  const NDArray r = bin("less", col, vec_i({2, 4}));
  CHECK(r.shape() == Shape({2, 2}));
  CHECK(ints(r) == IV({1, 1, 0, 1}));
}

TEST_CASE("p05 logical: and/or/xor/not on every dtype") {
  for (DType dt : {DType::Bool, DType::Int8, DType::UInt64, DType::Float16, DType::Float64,
                   DType::Complex64, DType::Complex128}) {
    const NDArray a = vec_d({0, 0, 2, 3}, dt);
    const NDArray b = vec_d({0, 1, 0, 1}, dt);
    CHECK(ints(bin("logicalAnd", a, b)) == IV({0, 0, 0, 1}));
    CHECK(ints(bin("logicalOr", a, b)) == IV({0, 1, 1, 1}));
    CHECK(ints(bin("logicalXor", a, b)) == IV({0, 1, 1, 0}));
    const NDArray n = unary(U("logicalNot"), a, nullptr, {});
    CHECK(n.dtype() == DType::Bool);
    CHECK(ints(n) == IV({1, 1, 0, 0}));
  }
  using C = std::complex<double>;
  CHECK(ints(unary(U("logicalNot"), cvec({C(0, 1), C(0, 0)}), nullptr, {})) == IV({0, 1}));
  CHECK(ints(unary(U("logicalNot"), vec_d({kNaN}), nullptr, {})) == IV({0}));
}

TEST_CASE("p05 logical: reduce casts to bool and uses identities") {
  const NDArray r = ufunc_reduce(U("logicalAnd"), vec_i({1, 2, 0}), nullptr, {});
  CHECK(r.dtype() == DType::Bool);
  CHECK_EQ(r.get_int64(0), 0);
  CHECK_EQ(ufunc_reduce(U("logicalXor"), vec_i({1, 1, 1}), nullptr, {}).get_int64(0), 1);
  const NDArray e = NDArray::empty({0}, DType::Float64);
  CHECK_EQ(ufunc_reduce(U("logicalAnd"), e, nullptr, {}).get_int64(0), 1);
  CHECK_EQ(ufunc_reduce(U("logicalOr"), e, nullptr, {}).get_int64(0), 0);
  NDArray m = NDArray::empty({2, 2}, DType::Int64);
  for (int i = 0; i < 4; ++i) m.set_int64(i, i == 1 ? 0 : 1);
  UfuncReduceOptions all;
  all.all_axes = true;
  CHECK_EQ(ufunc_reduce(U("logicalAnd"), m, nullptr, all).get_int64(0), 0);
  CHECK_EQ(ufunc_reduce(U("logicalOr"), m, nullptr, all).get_int64(0), 1);
  NDArray out = NDArray::zeros({}, DType::Int64);
  ufunc_reduce(U("logicalAnd"), vec_d({1.5, 2}), &out, {});
  CHECK_EQ(out.get_int64(0), 1);
  CHECK(ints(ufunc_accumulate(U("logicalAnd"), vec_i({1, 2, 0, 3}), nullptr, {})) == IV({1, 1, 0, 0}));
  UfuncReduceOptions bad;
  bad.dtype = DType::Int64;
  CHECK_THROWS_KIND(ufunc_reduce(U("logicalAnd"), vec_i({1}), nullptr, bad), ErrorKind::DType);
}

TEST_CASE("p05 isnan/isinf/isfinite/isposinf/isneginf") {
  constexpr double inf = std::numeric_limits<double>::infinity();
  for (DType dt : {DType::Float16, DType::Float32, DType::Float64}) {
    const NDArray a = vec_d({kNaN, inf, -inf, 1.0}, dt);
    const auto un = [&](const char* n) { return ints(unary(U(n), a, nullptr, {})); };
    CHECK(un("isnan") == IV({1, 0, 0, 0}));
    CHECK(un("isinf") == IV({0, 1, 1, 0}));
    CHECK(un("isfinite") == IV({0, 0, 0, 1}));
    CHECK(un("isposinf") == IV({0, 1, 0, 0}));
    CHECK(un("isneginf") == IV({0, 0, 1, 0}));
  }
  for (DType dt : {DType::Bool, DType::Int8, DType::UInt64}) {
    const NDArray a = vec_i({0, 1}, dt);
    CHECK(ints(unary(U("isnan"), a, nullptr, {})) == IV({0, 0}));
    CHECK(ints(unary(U("isfinite"), a, nullptr, {})) == IV({1, 1}));
    CHECK(ints(unary(U("isneginf"), a, nullptr, {})) == IV({0, 0}));
  }
  using C = std::complex<double>;
  const NDArray c = cvec({C(1, kNaN), C(inf, 0), C(1, 2)});
  CHECK(ints(unary(U("isnan"), c, nullptr, {})) == IV({1, 0, 0}));
  CHECK(ints(unary(U("isinf"), c, nullptr, {})) == IV({0, 1, 0}));
  CHECK(ints(unary(U("isfinite"), c, nullptr, {})) == IV({0, 0, 1}));
  CHECK_THROWS_KIND(unary(U("isposinf"), c, nullptr, {}), ErrorKind::DType);
  CHECK_THROWS_KIND(unary(U("isnat"), vec_d({1}), nullptr, {}), ErrorKind::DType);
  UfuncParams p;
  p.dtype = DType::Float64;
  CHECK_THROWS_KIND(unary(U("isnan"), vec_d({1}), nullptr, p), ErrorKind::DType);
  p.dtype = DType::Bool;
  CHECK(ints(unary(U("isnan"), vec_d({kNaN}), nullptr, p)) == IV({1}));
}
