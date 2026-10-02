#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "cast.hpp"
#include "error.hpp"
#include "fp_errors.hpp"
#include "p05_logic.hpp"
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
  const ErrStateGuard guard;  // restores errstate on exit, even if an exception escapes
  ErrState s = get_errstate();
  s.invalid = FpMode::Raise;
  set_errstate(s);
  using C = std::complex<double>;
  CHECK_THROWS_KIND(bin("less", cvec({C(kNaN, 0)}), cvec({C(1, 0)})), ErrorKind::FloatingPoint);
  // equal is quiet; real float compares are quiet.
  CHECK(ints(bin("equal", cvec({C(kNaN, 0)}), cvec({C(1, 0)}))) == IV({0}));
  CHECK(ints(bin("less", vec_d({kNaN}), vec_d({1}))) == IV({0}));
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

TEST_CASE("p05 bitwise and/or/xor/invert") {
  for (DType dt : {DType::Int8, DType::UInt8, DType::Int16, DType::UInt16, DType::Int32, DType::UInt32,
                   DType::Int64, DType::UInt64}) {
    const NDArray a = vec_i({12, 10}, dt);
    const NDArray b = vec_i({10, 6}, dt);
    CHECK(bin("bitwiseAnd", a, b).dtype() == dt);
    CHECK(ints(bin("bitwiseAnd", a, b)) == IV({8, 2}));
    CHECK(ints(bin("bitwiseOr", a, b)) == IV({14, 14}));
    CHECK(ints(bin("bitwiseXor", a, b)) == IV({6, 12}));
  }
  CHECK(ints(unary(U("invert"), vec_i({5}, DType::Int8), nullptr, {})) == IV({-6}));
  CHECK(ints(unary(U("invert"), vec_i({1}, DType::UInt8), nullptr, {})) == IV({254}));
  const NDArray bl = vec_i({1, 0}, DType::Bool);
  CHECK(unary(U("invert"), bl, nullptr, {}).dtype() == DType::Bool);
  CHECK(ints(unary(U("invert"), bl, nullptr, {})) == IV({0, 1}));
  CHECK(ints(bin("bitwiseXor", bl, vec_i({1, 1}, DType::Bool))) == IV({0, 1}));
  CHECK_THROWS_KIND(bin("bitwiseAnd", vec_d({1}), vec_i({1})), ErrorKind::DType);
  CHECK_THROWS_KIND(bin("bitwiseAnd", vec_i({1}, DType::UInt64), vec_i({1})), ErrorKind::DType);
  CHECK_THROWS_KIND(unary(U("invert"), vec_d({1}), nullptr, {}), ErrorKind::DType);
}

TEST_CASE("p05 bitwise reduce identities") {
  const NDArray e8 = NDArray::empty({0}, DType::UInt8);
  CHECK_EQ(ufunc_reduce(U("bitwiseAnd"), e8, nullptr, {}).get_int64(0), 255);
  const NDArray e64 = NDArray::empty({0}, DType::UInt64);
  const NDArray r = ufunc_reduce(U("bitwiseAnd"), e64, nullptr, {});
  CHECK(load<std::uint64_t>(r.data()) == ~std::uint64_t{0});
  CHECK_EQ(ufunc_reduce(U("bitwiseAnd"), NDArray::empty({0}, DType::Int8), nullptr, {}).get_int64(0), -1);
  CHECK_EQ(ufunc_reduce(U("bitwiseAnd"), NDArray::empty({0}, DType::Bool), nullptr, {}).get_int64(0), 1);
  CHECK_EQ(ufunc_reduce(U("bitwiseOr"), vec_i({1, 2, 4}), nullptr, {}).get_int64(0), 7);
  CHECK_EQ(ufunc_reduce(U("bitwiseXor"), vec_i({1, 3}), nullptr, {}).get_int64(0), 2);
  CHECK_EQ(ufunc_reduce(U("leftShift"), vec_i({1, 2, 3}), nullptr, {}).get_int64(0), 32);
}

TEST_CASE("p05 shifts follow npy_lshift/npy_rshift") {
  const auto sh = [](const char* op, std::int64_t a, std::int64_t b, DType dt) {
    return bin(op, vec_i({a}, dt), vec_i({b}, dt)).get_int64(0);
  };
  CHECK_EQ(sh("leftShift", 1, 3, DType::Int8), 8);
  CHECK_EQ(sh("leftShift", 1, 7, DType::Int8), -128);
  CHECK_EQ(sh("leftShift", 1, 9, DType::Int8), 0);
  CHECK_EQ(sh("leftShift", 1, -1, DType::Int8), 0);
  CHECK_EQ(sh("leftShift", 1, 64, DType::Int64), 0);
  CHECK_EQ(sh("rightShift", -128, 10, DType::Int8), -1);
  CHECK_EQ(sh("rightShift", -5, -1, DType::Int8), -1);
  CHECK_EQ(sh("rightShift", 5, -1, DType::Int8), 0);
  CHECK_EQ(sh("rightShift", 200, 10, DType::UInt8), 0);
  CHECK_EQ(sh("rightShift", -8, 1, DType::Int32), -4);
  const NDArray t = vec_i({1}, DType::Bool);
  const NDArray r = bin("leftShift", t, t);
  CHECK(r.dtype() == DType::Int8);
  CHECK_EQ(r.get_int64(0), 2);
  CHECK(bin("leftShift", vec_i({1}, DType::UInt8), vec_i({1}, DType::Int8)).dtype() == DType::Int16);
}

TEST_CASE("p05 bitwiseCount") {
  const NDArray r = unary(U("bitwiseCount"), vec_i({-1, -128, 7, 0}, DType::Int8), nullptr, {});
  CHECK(r.dtype() == DType::UInt8);
  CHECK(ints(r) == IV({1, 1, 3, 0}));
  CHECK(ints(unary(U("bitwiseCount"), vec_i({255}, DType::UInt8), nullptr, {})) == IV({8}));
  CHECK(ints(unary(U("bitwiseCount"), vec_i({1}, DType::Bool), nullptr, {})) == IV({1}));
  NDArray big = NDArray::empty({1}, DType::UInt64);
  store<std::uint64_t>(big.data(), ~std::uint64_t{0});
  CHECK(ints(unary(U("bitwiseCount"), big, nullptr, {})) == IV({64}));
  CHECK(ints(unary(U("bitwiseCount"), vec_i({std::numeric_limits<std::int64_t>::min()}), nullptr, {})) == IV({1}));
  CHECK_THROWS_KIND(unary(U("bitwiseCount"), vec_d({1}), nullptr, {}), ErrorKind::DType);
}

TEST_CASE("p05 isclose kernel") {
  constexpr double inf = std::numeric_limits<double>::infinity();
  const NDArray a = vec_d({1.0, 1.0, inf, -inf, kNaN, 1e10});
  const NDArray b = vec_d({1.0 + 1e-6, 1.1, inf, inf, kNaN, 1.00001e10});
  NDArray r = isclose(a, b, 1e-5, 1e-8, false);
  CHECK(r.dtype() == DType::Bool);
  CHECK(ints(r) == IV({1, 0, 1, 0, 0, 1}));
  CHECK(ints(isclose(a, b, 1e-5, 1e-8, true)) == IV({1, 0, 1, 0, 1, 1}));
  CHECK(ints(isclose(vec_i({1, 2}, DType::Int8), vec_i({1, 3}), 1e-5, 1e-8, false)) == IV({1, 0}));
  CHECK(ints(isclose(vec_i({1, 2}), vec_i({2, 2}), 1.0, 0.0, false)) == IV({1, 1}));
  using C = std::complex<double>;
  CHECK(ints(isclose(cvec({C(1, 1), C(kNaN, 0)}), cvec({C(1, 1 + 1e-9), C(0, kNaN)}), 1e-5, 1e-8, true)) ==
        IV({1, 1}));
  const NDArray f16 = vec_d({1.0}, DType::Float16);
  CHECK(ints(isclose(f16, vec_d({1.001}, DType::Float16), 1e-5, 1e-8, false)) == IV({0}));
  CHECK(isclose(vec_d({1.0, 2.0}), vec_d({1.0}), 1e-5, 1e-8, false).shape() == Shape{2});
}

TEST_CASE("p05 packbits/unpackbits") {
  const NDArray a = vec_i({2, 7, 23, 4, 5, 0}, DType::UInt8).reshape({2, 3});
  CHECK(ints(packbits(a, std::nullopt, false)) == IV({248}));
  const NDArray r1 = packbits(a, 1, false);
  CHECK(r1.shape() == (Shape{2, 1}));
  CHECK(ints(r1) == IV({224, 192}));
  CHECK(ints(packbits(a, 0, false)) == IV({192, 192, 128}));
  CHECK(ints(packbits(a, -1, true)) == IV({7, 3}));
  CHECK(ints(packbits(vec_i({-1, 0, 2}), std::nullopt, false)) == IV({160}));
  CHECK(ints(packbits(vec_i({1, 1, 1, 1, 1, 1, 1, 1, 1}, DType::Bool), std::nullopt, false)) == IV({255, 128}));
  CHECK(packbits(NDArray::zeros({2, 0}, DType::Bool), 0, false).shape() == (Shape{1, 0}));
  CHECK_THROWS_KIND(packbits(vec_d({1}), std::nullopt, false), ErrorKind::DType);
  CHECK_THROWS_KIND(packbits(vec_i({1}), 1, false), ErrorKind::Index);

  const NDArray u = vec_i({2, 7, 23}, DType::UInt8).reshape({3, 1});
  CHECK(unpackbits(u, 1, std::nullopt, false).shape() == (Shape{3, 8}));
  CHECK(ints(unpackbits(u, 1, -3, true)) == IV({0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1}));
  CHECK(ints(unpackbits(vec_i({5}, DType::UInt8), std::nullopt, 10, false)) == IV({0, 0, 0, 0, 0, 1, 0, 1, 0, 0}));
  CHECK(unpackbits(u, std::nullopt, 0, false).shape() == Shape{0});
  CHECK(unpackbits(u, 0, std::nullopt, false).shape() == (Shape{24, 1}));
  CHECK_THROWS_KIND(unpackbits(u, 1, -9, false), ErrorKind::Value);
  CHECK_THROWS_KIND(unpackbits(vec_i({1}, DType::Int8), std::nullopt, std::nullopt, false), ErrorKind::DType);
  // Round trip on a strided input.
  const NDArray big = vec_i({1, 0, 1, 1, 0, 0, 1, 0, 1, 1}, DType::Bool);
  CHECK(ints(unpackbits(packbits(big, 0, true), 0, 10, true)) == IV({1, 0, 1, 1, 0, 0, 1, 0, 1, 1}));
}
