#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numbers>

#include "cast.hpp"

#include "error.hpp"
#include "p04_multi.hpp"
#include "p04_special.hpp"
#include "test_harness.hpp"
#include "ufunc_registry.hpp"

using namespace nativpy;

namespace {
NDArray vec_d(std::initializer_list<double> v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t i = 0;
  for (auto x : v) a.set_double(i++, x);
  return a;
}
NDArray vec_i(std::initializer_list<std::int64_t> v, DType dt) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t i = 0;
  for (auto x : v) a.set_int64(i++, x);
  return a;
}
const Ufunc& U(const char* name) {
  const Ufunc* u = find_ufunc(name);
  if (!u) throw_error(ErrorKind::Value, std::string("missing ufunc ") + name);
  return *u;
}
NDArray un(const char* name, const NDArray& a) { return unary(U(name), a, nullptr, {}); }
NDArray bin(const char* name, const NDArray& a, const NDArray& b) {
  return binary(U(name), a, b, nullptr, {});
}
bool near(double a, double b, double tol = 1e-12) {
  if (std::isnan(a) && std::isnan(b)) return true;
  if (std::isinf(a) || std::isinf(b)) return a == b;
  return std::fabs(a - b) <= tol * std::fmax(1.0, std::fabs(b));
}
}  // namespace

TEST_CASE("p04 trig: values and dtypes") {
  const NDArray x = vec_d({0.0, 0.5, -1.0});
  const NDArray s = un("sin", x);
  CHECK(s.dtype() == DType::Float64);
  CHECK(near(s.get_double(1), std::sin(0.5)));
  CHECK(near(un("arccos", x).get_double(2), std::acos(-1.0)));
  CHECK(near(un("arctanh", x).get_double(1), std::atanh(0.5)));
  CHECK(std::isinf(un("arctanh", vec_d({1.0})).get_double(0)));
  CHECK(std::isnan(un("arcsin", vec_d({2.0})).get_double(0)));
  // ints/bool -> smallest safe float (NumPy)
  CHECK(un("cos", vec_i({1}, DType::Int8)).dtype() == DType::Float16);
  CHECK(un("cos", vec_i({1}, DType::Int16)).dtype() == DType::Float32);
  CHECK(un("cos", vec_i({1}, DType::Int64)).dtype() == DType::Float64);
  CHECK(un("tanh", vec_i({1}, DType::Bool)).dtype() == DType::Float16);
  CHECK(near(un("rad2deg", vec_d({std::numbers::pi})).get_double(0), 180.0));
  CHECK(near(un("deg2rad", vec_d({180.0})).get_double(0), std::numbers::pi));
}

TEST_CASE("p04 trig: complex loops") {
  NDArray z = NDArray::empty({1}, DType::Complex128);
  z.set_double(0, 2.0);  // 2+0j
  const NDArray r = un("arcsin", z);
  CHECK(r.dtype() == DType::Complex128);
  const auto v = load<std::complex<double>>(r.data());
  CHECK(near(v.real(), std::numbers::pi / 2));
  CHECK(near(std::fabs(v.imag()), 1.3169578969248166));
  CHECK(un("sinh", NDArray::empty({2}, DType::Complex64)).dtype() == DType::Complex64);
  CHECK_THROWS_KIND(un("deg2rad", z), ErrorKind::DType);
  CHECK_THROWS_KIND(bin("arctan2", z, z), ErrorKind::DType);
}

TEST_CASE("p04 trig: binary") {
  const NDArray a = vec_d({1.0, -1.0, 0.0});
  const NDArray b = vec_d({1.0, -1.0, -0.0});
  const NDArray r = bin("arctan2", a, b);
  CHECK(near(r.get_double(0), std::numbers::pi / 4));
  CHECK(near(r.get_double(1), -3 * std::numbers::pi / 4));
  CHECK(near(r.get_double(2), std::numbers::pi));
  CHECK(near(bin("hypot", vec_d({3.0}), vec_d({4.0})).get_double(0), 5.0));
  CHECK(bin("hypot", vec_i({3}, DType::Int8), vec_i({4}, DType::UInt8)).dtype() == DType::Float16);
  CHECK(bin("arctan2", vec_d({1}, DType::Float16), vec_i({1}, DType::Int16)).dtype() == DType::Float32);
  CHECK(U("hypot").identity.has_value());
  CHECK(!U("arctan2").identity.has_value());
}

TEST_CASE("p04 exp/log: values, dtypes, identities") {
  const NDArray x = vec_d({0.0, 1.0, 3.0});
  CHECK(near(un("exp2", x).get_double(2), 8.0));
  CHECK(near(un("log2", vec_d({8.0})).get_double(0), 3.0));
  CHECK(near(un("log10", vec_d({1000.0})).get_double(0), 3.0));
  CHECK(near(un("expm1", vec_d({1e-10})).get_double(0), 1.00000000005e-10));
  CHECK(near(un("log1p", vec_d({1e-10})).get_double(0), 9.9999999995e-11));
  CHECK(un("log1p", vec_d({-1.0})).get_double(0) == -std::numeric_limits<double>::infinity());
  CHECK(near(un("cbrt", vec_d({-8.0})).get_double(0), -2.0));
  CHECK(un("cbrt", vec_i({8}, DType::Int8)).dtype() == DType::Float16);
  const double inf = std::numeric_limits<double>::infinity();
  const NDArray l = bin("logaddexp", vec_d({1.0, inf, -inf, inf}), vec_d({2.0, inf, -inf, -inf}));
  CHECK(near(l.get_double(0), 2.313261687518223));
  CHECK(l.get_double(1) == inf);
  CHECK(l.get_double(2) == -inf);
  CHECK(l.get_double(3) == inf);
  CHECK(near(bin("logaddexp2", vec_d({1.0}), vec_d({1.0})).get_double(0), 2.0));
  CHECK(*U("logaddexp").identity == -inf);
}

TEST_CASE("p04 exp/log: complex loops") {
  NDArray z = NDArray::empty({1}, DType::Complex128);
  store(z.data(), std::complex<double>(-1.0, 0.0));
  const auto v = load<std::complex<double>>(un("log2", z).data());
  CHECK(near(v.real(), 0.0));
  CHECK(near(v.imag(), 4.532360141827194));
  store(z.data(), std::complex<double>(1.0, 1.0));
  const auto e = load<std::complex<double>>(un("exp2", z).data());
  CHECK(near(e.real(), 1.5384778027279442));
  CHECK(near(e.imag(), 1.2779225526272695));
  const auto s = load<std::complex<double>>(un("square", z).data());
  CHECK(s == std::complex<double>(0.0, 2.0));
  CHECK_THROWS_KIND(un("cbrt", z), ErrorKind::DType);
  CHECK_THROWS_KIND(bin("logaddexp", z, z), ErrorKind::DType);
}

TEST_CASE("p04 square/reciprocal: integer loops") {
  const NDArray s = un("square", vec_i({200, 3}, DType::UInt8));
  CHECK(s.dtype() == DType::UInt8);
  CHECK_EQ(s.get_int64(0), std::int64_t{64});  // modular
  CHECK(un("square", vec_i({1}, DType::Bool)).dtype() == DType::Int8);
  const NDArray r = un("reciprocal", vec_i({2, -1, 1}, DType::Int64));
  CHECK_EQ(r.get_int64(0), std::int64_t{0});
  CHECK_EQ(r.get_int64(1), std::int64_t{-1});
  CHECK_EQ(r.get_int64(2), std::int64_t{1});
  CHECK(near(un("reciprocal", vec_d({4.0})).get_double(0), 0.25));
}

TEST_CASE("p04 rounding: floor ceil trunc rint") {
  const NDArray x = vec_d({-1.5, -0.5, 0.5, 1.5, 2.5});
  const NDArray f = un("floor", x), c = un("ceil", x), t = un("trunc", x), r = un("rint", x);
  const double ef[] = {-2, -1, 0, 1, 2}, ec[] = {-1, -0.0, 1, 2, 3}, et[] = {-1, -0.0, 0, 1, 2},
               er[] = {-2, -0.0, 0, 2, 2};
  for (int i = 0; i < 5; ++i) {
    CHECK_EQ(f.get_double(i), ef[i]);
    CHECK_EQ(c.get_double(i), ec[i]);
    CHECK_EQ(t.get_double(i), et[i]);
    CHECK_EQ(r.get_double(i), er[i]);
  }
  CHECK(std::signbit(t.get_double(1)));
  CHECK(un("floor", vec_i({3}, DType::Int16)).dtype() == DType::Int16);
  CHECK(un("trunc", vec_i({1}, DType::Bool)).dtype() == DType::Bool);
  CHECK(un("rint", vec_i({1}, DType::Int16)).dtype() == DType::Float32);
  NDArray z = NDArray::empty({1}, DType::Complex128);
  store(z.data(), std::complex<double>(1.5, 2.5));
  CHECK(load<std::complex<double>>(un("rint", z).data()) == std::complex<double>(2.0, 2.0));
  CHECK_THROWS_KIND(un("floor", z), ErrorKind::DType);
  CHECK_EQ(un("positive", vec_i({-3}, DType::Int8)).get_int64(0), std::int64_t{-3});
  CHECK_THROWS_KIND(un("positive", vec_i({1}, DType::Bool)), ErrorKind::DType);
}

TEST_CASE("p04 arithmetic: fmod, float_power, sign, heaviside") {
  const NDArray f = bin("fmod", vec_i({-7, 7, -128, 5}, DType::Int8), vec_i({3, -3, -1, 0}, DType::Int8));
  CHECK(f.dtype() == DType::Int8);
  const std::int64_t ef[] = {-1, 1, 0, 0};
  for (int i = 0; i < 4; ++i) CHECK_EQ(f.get_int64(i), ef[i]);
  const NDArray ff = bin("fmod", vec_d({-7.5, 7.5}), vec_d({2.0, -2.0}));
  CHECK_EQ(ff.get_double(0), -1.5);
  CHECK_EQ(ff.get_double(1), 1.5);
  CHECK(bin("fmod", vec_i({1}, DType::Bool), vec_i({1}, DType::Bool)).dtype() == DType::Int8);
  CHECK(bin("float_power", vec_i({2}, DType::Int8), vec_i({3}, DType::Int8)).dtype() == DType::Float64);
  CHECK(bin("float_power", vec_d({2}, DType::Float32), vec_d({3}, DType::Float32)).dtype() == DType::Float64);
  CHECK(bin("float_power", NDArray::empty({1}, DType::Complex64), vec_d({1})).dtype() == DType::Complex128);
  const NDArray s = un("sign", vec_d({-2.0, -0.0, 3.0, std::nan("")}));
  CHECK_EQ(s.get_double(0), -1.0);
  CHECK(s.get_double(1) == 0.0 && !std::signbit(s.get_double(1)));
  CHECK_EQ(s.get_double(2), 1.0);
  CHECK(std::isnan(s.get_double(3)));
  CHECK_EQ(un("sign", vec_i({-5}, DType::Int16)).get_int64(0), std::int64_t{-1});
  CHECK_THROWS_KIND(un("sign", vec_i({1}, DType::Bool)), ErrorKind::DType);
  NDArray z = NDArray::empty({1}, DType::Complex128);
  store(z.data(), std::complex<double>(3.0, 4.0));
  const auto sz = load<std::complex<double>>(un("sign", z).data());
  CHECK(near(sz.real(), 0.6) && near(sz.imag(), 0.8));
  const NDArray h = bin("heaviside", vec_d({-1.0, 0.0, 2.0}), vec_d({0.5}));
  CHECK_EQ(h.get_double(0), 0.0);
  CHECK_EQ(h.get_double(1), 0.5);
  CHECK_EQ(h.get_double(2), 1.0);
  CHECK(un("fabs", vec_i({-1}, DType::Int8)).dtype() == DType::Float16);
}

TEST_CASE("p04 arithmetic: maximum minimum fmax fmin") {
  const double nan = std::nan("");
  const NDArray a = vec_d({nan, 1.0, -0.0, 0.0});
  const NDArray b = vec_d({1.0, nan, 0.0, -0.0});
  const NDArray mx = bin("maximum", a, b), mn = bin("minimum", a, b);
  const NDArray fx = bin("fmax", a, b), fn_ = bin("fmin", a, b);
  CHECK(std::isnan(mx.get_double(0)) && std::isnan(mx.get_double(1)));
  CHECK(std::isnan(mn.get_double(0)) && std::isnan(mn.get_double(1)));
  CHECK_EQ(fx.get_double(0), 1.0);
  CHECK_EQ(fn_.get_double(1), 1.0);
  CHECK(!std::signbit(mx.get_double(2)) && !std::signbit(mx.get_double(3)));
  CHECK(std::signbit(mn.get_double(2)) && std::signbit(mn.get_double(3)));
  const NDArray bi = bin("maximum", vec_i({1, 0}, DType::Bool), vec_i({0, 0}, DType::Bool));
  CHECK(bi.dtype() == DType::Bool);
  CHECK_EQ(bi.get_int64(0), std::int64_t{1});
  CHECK(bin("minimum", vec_i({1}, DType::Int8), vec_i({1}, DType::UInt8)).dtype() == DType::Int16);
  NDArray z = NDArray::empty({1}, DType::Complex128), w = NDArray::empty({1}, DType::Complex128);
  store(z.data(), std::complex<double>(1.0, 2.0));
  store(w.data(), std::complex<double>(1.0, 3.0));
  CHECK(load<std::complex<double>>(bin("maximum", z, w).data()) == std::complex<double>(1.0, 3.0));
  CHECK(load<std::complex<double>>(bin("fmin", z, w).data()) == std::complex<double>(1.0, 2.0));
}

TEST_CASE("p04 float bits: copysign ldexp nextafter spacing signbit") {
  const NDArray c = bin("copysign", vec_d({3.0, 2.0}), vec_d({-0.0, 1.0}));
  CHECK_EQ(c.get_double(0), -3.0);
  CHECK_EQ(c.get_double(1), 2.0);
  CHECK(bin("copysign", vec_i({1}, DType::Int8), vec_i({-1}, DType::Int8)).dtype() == DType::Float16);
  const NDArray l = bin("ldexp", vec_d({1.5, 1.0}), vec_i({3, -1}, DType::Int64));
  CHECK_EQ(l.get_double(0), 12.0);
  CHECK_EQ(l.get_double(1), 0.5);
  CHECK(bin("ldexp", vec_i({1}, DType::Int8), vec_i({3}, DType::Int64)).dtype() == DType::Float16);
  CHECK(std::isinf(bin("ldexp", vec_d({1.0}), vec_i({std::int64_t{1} << 40}, DType::Int64)).get_double(0)));
  CHECK_THROWS_KIND(bin("ldexp", vec_d({1.0}), vec_d({2.5})), ErrorKind::DType);
  CHECK_THROWS_KIND(bin("ldexp", vec_d({1.0}), vec_i({2}, DType::UInt64)), ErrorKind::DType);
  const NDArray n = bin("nextafter", vec_d({1.0, 0.0}), vec_d({2.0, -1.0}));
  CHECK_EQ(n.get_double(0), std::nextafter(1.0, 2.0));
  CHECK_EQ(n.get_double(1), -std::numeric_limits<double>::denorm_min());
  const NDArray nh = bin("nextafter", vec_d({1.0, 0.0}, DType::Float16), vec_d({2.0, -1.0}, DType::Float16));
  CHECK_EQ(nh.get_double(0), 1.0 + 1.0 / 1024);
  CHECK(near(nh.get_double(1), -5.960464477539063e-08));
  const NDArray sp = un("spacing", vec_d({1.0, -1.0, 0.0, std::numeric_limits<double>::infinity()}));
  CHECK_EQ(sp.get_double(0), std::numeric_limits<double>::epsilon());
  CHECK_EQ(sp.get_double(1), -std::numeric_limits<double>::epsilon());
  CHECK_EQ(sp.get_double(2), std::numeric_limits<double>::denorm_min());
  CHECK(std::isnan(sp.get_double(3)));
  const NDArray sh = un("spacing", vec_d({1.0, -1.0, -65504.0}, DType::Float16));
  CHECK_EQ(sh.get_double(0), 1.0 / 1024);
  CHECK_EQ(sh.get_double(1), 1.0 / 2048);
  CHECK_EQ(sh.get_double(2), 32.0);
  CHECK(un("spacing", vec_i({1}, DType::Int8)).dtype() == DType::Float16);
  const NDArray sb = un("signbit", vec_d({-0.0, 1.0, -3.0}));
  CHECK(sb.dtype() == DType::Bool);
  CHECK_EQ(sb.get_int64(0), std::int64_t{1});
  CHECK_EQ(sb.get_int64(1), std::int64_t{0});
  CHECK(un("signbit", vec_i({-1}, DType::Int8)).get_int64(0) == 1);
  CHECK_THROWS_KIND(un("signbit", NDArray::empty({1}, DType::Complex128)), ErrorKind::DType);
}

TEST_CASE("p04 integer: gcd lcm") {
  const NDArray g = bin("gcd", vec_i({12, -12, 0, 7}, DType::Int64), vec_i({18, 18, 0, 0}, DType::Int64));
  const std::int64_t eg[] = {6, 6, 0, 7};
  for (int i = 0; i < 4; ++i) CHECK_EQ(g.get_int64(i), eg[i]);
  const NDArray l = bin("lcm", vec_i({4, 0, -3}, DType::Int64), vec_i({6, 5, 7}, DType::Int64));
  const std::int64_t el[] = {12, 0, 21};
  for (int i = 0; i < 3; ++i) CHECK_EQ(l.get_int64(i), el[i]);
  CHECK_EQ(bin("lcm", vec_i({100}, DType::Int8), vec_i({3}, DType::Int8)).get_int64(0), std::int64_t{44});
  CHECK(bin("gcd", vec_i({1}, DType::Int8), vec_i({1}, DType::UInt8)).dtype() == DType::Int16);
  CHECK_EQ(*U("gcd").identity, 0.0);
  CHECK_THROWS_KIND(bin("gcd", vec_d({1.0}), vec_d({2.0})), ErrorKind::DType);
  CHECK_THROWS_KIND(bin("gcd", vec_i({1}, DType::Bool), vec_i({1}, DType::Bool)), ErrorKind::DType);
}

TEST_CASE("p04 multi-output: divmod modf frexp") {
  const NDArray div2 = vec_i({2, 2, 0}, DType::Int64);
  const auto dm = multi_ufunc(MultiOp::Divmod, vec_i({7, -7, 5}, DType::Int64),
                              &div2, {});
  const std::int64_t eq[] = {3, -4, 0}, er[] = {1, 1, 0};
  for (int i = 0; i < 3; ++i) {
    CHECK_EQ(dm[0].get_int64(i), eq[i]);
    CHECK_EQ(dm[1].get_int64(i), er[i]);
  }
  const NDArray fb = vec_d({2.0, 2.0, 0.0});
  const auto df = multi_ufunc(MultiOp::Divmod, vec_d({7.5, -7.5, 1.0}), &fb, {});
  CHECK_EQ(df[0].get_double(0), 3.0);
  CHECK_EQ(df[1].get_double(1), 0.5);
  CHECK_EQ(df[0].get_double(1), -4.0);
  CHECK(std::isinf(df[0].get_double(2)) && std::isnan(df[1].get_double(2)));
  const NDArray bb = vec_i({1}, DType::Bool);
  CHECK(multi_ufunc(MultiOp::Divmod, bb, &bb, {})[0].dtype() == DType::Int8);
  NDArray z = NDArray::empty({1}, DType::Complex128);
  CHECK_THROWS_KIND(multi_ufunc(MultiOp::Divmod, z, &z, {}), ErrorKind::DType);

  const auto mf = multi_ufunc(MultiOp::Modf, vec_d({-2.5, std::numeric_limits<double>::infinity()}), nullptr, {});
  CHECK_EQ(mf[0].get_double(0), -0.5);
  CHECK_EQ(mf[1].get_double(0), -2.0);
  CHECK_EQ(mf[0].get_double(1), 0.0);
  CHECK(std::isinf(mf[1].get_double(1)));
  CHECK(multi_ufunc(MultiOp::Modf, vec_i({8}, DType::Int8), nullptr, {})[0].dtype() == DType::Float16);

  const auto fr = multi_ufunc(MultiOp::Frexp, vec_d({8.0, 0.0, -3.0, std::nan("")}), nullptr, {});
  CHECK(fr[1].dtype() == DType::Int32);
  CHECK_EQ(fr[0].get_double(0), 0.5);
  CHECK_EQ(fr[1].get_int64(0), std::int64_t{4});
  CHECK_EQ(fr[0].get_double(2), -0.75);
  CHECK_EQ(fr[1].get_int64(2), std::int64_t{2});
  CHECK_EQ(fr[1].get_int64(3), std::int64_t{0});
  CHECK_THROWS_KIND(multi_ufunc(MultiOp::Frexp, z, nullptr, {}), ErrorKind::DType);

  // out= with casting, and a bad shape
  NDArray o0 = NDArray::empty({2}, DType::Float64), o1 = NDArray::empty({2}, DType::Int64);
  MultiParams p;
  p.out0 = &o0;
  p.out1 = &o1;
  const NDArray three = vec_i({3}, DType::Int64);
  const auto res = multi_ufunc(MultiOp::Divmod, vec_i({7, 8}, DType::Int64), &three, p);
  CHECK(res[0].shares_buffer(o0));
  CHECK_EQ(o0.get_double(1), 2.0);
  CHECK_EQ(o1.get_int64(1), std::int64_t{2});
  NDArray bad = NDArray::empty({3}, DType::Float64);
  p.out0 = &bad;
  CHECK_THROWS_KIND(multi_ufunc(MultiOp::Divmod, vec_i({7, 8}, DType::Int64), &three, p), ErrorKind::Broadcast);
  NDArray io = NDArray::empty({2}, DType::Int64);
  MultiParams q;
  q.out0 = &io;
  CHECK_THROWS_KIND(multi_ufunc(MultiOp::Modf, vec_d({1.5, 2.5}), nullptr, q), ErrorKind::DType);
}

TEST_CASE("p04 special: i0 sinc nan_to_num unwrap") {
  const NDArray i = p04_i0(vec_d({0.0, 1.0, -2.5, 30.0}));
  CHECK(near(i.get_double(0), 1.0));
  CHECK(near(i.get_double(1), 1.2660658777520082));
  CHECK(near(i.get_double(2), 3.2898391440501231));
  CHECK(near(i.get_double(3), 781672297823.9775, 1e-13));
  CHECK(p04_i0(vec_i({1}, DType::Int8)).dtype() == DType::Float64);
  CHECK_THROWS_KIND(p04_i0(NDArray::empty({1}, DType::Complex128)), ErrorKind::DType);
  const NDArray s = p04_sinc(vec_d({0.0, 0.5}));
  CHECK(near(s.get_double(0), 1.0));
  CHECK(near(s.get_double(1), 0.6366197723675814));
  const double inf = std::numeric_limits<double>::infinity();
  const NDArray n = p04_nan_to_num(vec_d({std::nan(""), inf, -inf}), true, 0.0, std::nullopt, -1.0);
  CHECK_EQ(n.get_double(0), 0.0);
  CHECK_EQ(n.get_double(1), std::numeric_limits<double>::max());
  CHECK_EQ(n.get_double(2), -1.0);
  const NDArray u = p04_unwrap(vec_i({0, 7, 14}, DType::Int64), 10, std::nullopt, -1, true, DType::Int64);
  CHECK_EQ(u.get_int64(1), std::int64_t{-3});
  CHECK_EQ(u.get_int64(2), std::int64_t{-6});
  const NDArray uf = p04_unwrap(vec_d({0.0, 3.0, 6.5}), 2 * std::numbers::pi, std::nullopt, 0, false, DType::Float64);
  CHECK(near(uf.get_double(2), 6.5 - 2 * std::numbers::pi));
  CHECK_THROWS_KIND(p04_unwrap(vec_d({1.0}), 1, std::nullopt, 1, false, DType::Float64), ErrorKind::Index);
}
