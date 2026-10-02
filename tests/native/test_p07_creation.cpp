#include <cmath>
#include <complex>

#include "creation.hpp"
#include "error.hpp"
#include "p07_creation.hpp"
#include "test_harness.hpp"

using namespace nativpy;
using cd = std::complex<double>;

namespace {
NDArray iota(std::int64_t n, DType dt = DType::Int64) { return arange(0, static_cast<double>(n), 1, dt); }
}  // namespace

TEST_CASE("p07: logspace/geomspace") {
  NDArray l = p07::logspace({0, 0}, {2, 0}, false, 3, true, 10.0, std::nullopt);
  CHECK(l.dtype() == DType::Float64);
  CHECK_EQ(l.get_double(2), 100.0);
  NDArray li = p07::logspace({0, 0}, {2.5, 0}, false, 3, true, 10.0, DType::Int64);
  CHECK_EQ(li.get_int64(1), std::int64_t{17});
  CHECK_EQ(p07::logspace({0, 0}, {1, 0}, false, 0, true, 2.0, std::nullopt).size(), std::int64_t{0});
  CHECK_THROWS_KIND(p07::logspace({0, 0}, {1, 0}, false, -1, true, 2.0, std::nullopt), ErrorKind::Value);
  NDArray g = p07::geomspace({1, 0}, {1000, 0}, 4, true, false, DType::Float64);
  CHECK_EQ(g.get_double(0), 1.0);
  CHECK_EQ(g.get_double(3), 1000.0);
  CHECK(std::fabs(g.get_double(1) - 10.0) < 1e-12);
  NDArray gn = p07::geomspace({-1, 0}, {-1000, 0}, 4, true, false, DType::Float64);
  CHECK_EQ(gn.get_double(3), -1000.0);
  CHECK(std::isnan(p07::geomspace({-1, 0}, {1, 0}, 3, true, false, DType::Float64).get_double(1)));
  CHECK_THROWS_KIND(p07::geomspace({0, 0}, {1, 0}, 3, true, false, DType::Float64), ErrorKind::Value);
  NDArray gc = p07::geomspace({0, 1}, {0, 1000}, 4, true, true, DType::Complex128);
  const auto* pc = reinterpret_cast<const cd*>(gc.data());
  CHECK_EQ(pc[3].imag(), 1000.0);
  CHECK_EQ(pc[3].real(), 0.0);
  CHECK(std::fabs(pc[1].imag() - 10.0) < 1e-12);
}

TEST_CASE("p07: tri/tril/triu/diag/vander") {
  NDArray t = p07::tri(2, 3, 1, DType::Float64);
  CHECK(t.shape() == Shape({2, 3}));
  CHECK_EQ(t.get_double(2), 0.0);
  CHECK_EQ(t.get_double(5), 1.0);
  CHECK(p07::tri(-1, -1, 0, DType::Bool).shape() == Shape({0, 0}));
  NDArray m = iota(9).reshape({3, 3});
  NDArray lo = p07::tril(m, -1);
  CHECK_EQ(lo.get_int64(3), std::int64_t{3});
  CHECK_EQ(lo.get_int64(4), std::int64_t{0});
  NDArray up = p07::triu(m, 1);
  CHECK_EQ(up.get_int64(1), std::int64_t{1});
  CHECK_EQ(up.get_int64(4), std::int64_t{0});
  CHECK(p07::tril(iota(3), 0).shape() == Shape({3, 3}));
  CHECK(p07::triu(NDArray::zeros({2, 0, 3}, DType::Int8), 0).shape() == Shape({2, 0, 3}));
  CHECK_THROWS_KIND(p07::tril(NDArray::zeros({}, DType::Int8), 0), ErrorKind::DType);
  NDArray d = p07::diag(iota(2), -1);
  CHECK(d.shape() == Shape({3, 3}));
  CHECK_EQ(d.get_int64(3), std::int64_t{0});
  CHECK_EQ(d.get_int64(7), std::int64_t{1});
  NDArray dv = p07::diag(m, 1);
  CHECK(dv.shape() == Shape({2}));
  CHECK_EQ(dv.get_int64(1), std::int64_t{5});
  CHECK(!dv.writeable());
  CHECK(dv.shares_buffer(m));
  CHECK_EQ(p07::diag(m, 5).size(), std::int64_t{0});
  CHECK_EQ(p07::diag(iota(6).reshape({2, 3}), -1).get_int64(0), std::int64_t{3});
  CHECK_THROWS_KIND(p07::diag(NDArray::zeros({2, 2, 2}, DType::Int8), 0), ErrorKind::Value);
  NDArray v = p07::vander(arange(1, 4, 1, DType::Int8), 3, false);
  CHECK(v.dtype() == DType::Int64);
  CHECK_EQ(v.get_int64(6), std::int64_t{9});
  CHECK_EQ(v.get_int64(8), std::int64_t{1});
  NDArray vi = p07::vander(arange(1, 4, 1, DType::Float32), 2, true);
  CHECK(vi.dtype() == DType::Float64);
  CHECK_EQ(vi.get_double(5), 3.0);
  CHECK(p07::vander(iota(2), 0, false).shape() == Shape({2, 0}));
  CHECK_THROWS_KIND(p07::vander(iota(2), -1, false), ErrorKind::Value);
  CHECK_THROWS_KIND(p07::vander(m, std::nullopt, false), ErrorKind::Value);
}

TEST_CASE("p07: grids") {
  NDArray g = p07::mgrid({0, 1}, {1, 0.5}, {2, 3}, DType::Float64);
  CHECK(g.shape() == Shape({2, 2, 3}));
  CHECK_EQ(g.get_double(3), 1.0);
  CHECK_EQ(g.get_double(7), 1.5);
  NDArray ix = p07::indices({2, 3}, DType::Int8);
  CHECK(ix.dtype() == DType::Int8);
  CHECK_EQ(ix.get_int64(4), std::int64_t{1});
  CHECK_EQ(ix.get_int64(11), std::int64_t{2});
  CHECK(p07::indices({}, DType::Int64).shape() == Shape({0}));
  CHECK(p07::indices({0, 2}, DType::Int64).shape() == Shape({2, 0, 2}));
  CHECK_EQ(p07::grid_axis(5, -2, 3, DType::Int64).get_int64(2), std::int64_t{1});
  CHECK_THROWS_KIND(p07::grid_axis(0, 1, -1, DType::Int64), ErrorKind::Value);
}

TEST_CASE("p07: tri_indices/fill_diagonal/concatenate") {
  auto [r, c] = p07::tri_indices(3, 3, 0, false);
  CHECK_EQ(r.size(), std::int64_t{6});
  CHECK_EQ(r.get_int64(5), std::int64_t{2});
  CHECK_EQ(c.get_int64(4), std::int64_t{1});
  auto [ru, cu] = p07::tri_indices(3, 4, -1, true);
  CHECK_EQ(ru.size(), std::int64_t{11});
  CHECK_EQ(cu.get_int64(10), std::int64_t{3});
  CHECK_EQ(p07::tri_indices(0, 0, 0, false).first.size(), std::int64_t{0});

  NDArray a = NDArray::zeros({7, 3}, DType::Int64);
  NDArray seven = NDArray::empty({}, DType::Int64);
  seven.set_int64(0, 7);
  p07::fill_diagonal(a, seven, true);
  CHECK_EQ(a.get_int64(12), std::int64_t{7});
  CHECK_EQ(a.get_int64(20), std::int64_t{7});
  NDArray b = NDArray::zeros({7, 3}, DType::Int64);
  p07::fill_diagonal(b, seven, false);
  CHECK_EQ(b.get_int64(12), std::int64_t{0});
  CHECK_EQ(b.get_int64(8), std::int64_t{7});
  NDArray cube = NDArray::zeros({2, 2, 2}, DType::Float32);
  p07::fill_diagonal(cube, iota(2, DType::Float32), false);
  CHECK_EQ(cube.get_double(7), 1.0);
  CHECK_THROWS_KIND(p07::fill_diagonal(NDArray::zeros({2, 3, 2}, DType::Float32), iota(1, DType::Float32), false),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(p07::fill_diagonal(iota(3), iota(1), false), ErrorKind::Value);

  NDArray cat = p07::concatenate({iota(2), iota(3)}, 0);
  CHECK(cat.shape() == Shape({5}));
  CHECK_EQ(cat.get_int64(4), std::int64_t{2});
  NDArray c2 = p07::concatenate({iota(2).reshape({2, 1}), iota(4).reshape({2, 2})}, -1);
  CHECK(c2.shape() == Shape({2, 3}));
  CHECK_EQ(c2.get_int64(3), std::int64_t{1});
  CHECK_THROWS_KIND(p07::concatenate({}, 0), ErrorKind::Value);
  CHECK_THROWS_KIND(p07::concatenate({iota(2), iota(4).reshape({2, 2})}, 0), ErrorKind::Value);
  CHECK_THROWS_KIND(p07::concatenate({iota(2)}, 1), ErrorKind::Index);
}

TEST_CASE("p07: fromstring") {
  NDArray a = p07::fromstring(" 1 , 2 ,3 ", DType::Float64, -1, ",");
  CHECK(a.shape() == Shape({3}));
  CHECK_EQ(a.get_double(2), 3.0);
  CHECK_EQ(p07::fromstring("", DType::Float64, -1, " ").size(), std::int64_t{0});
  CHECK_EQ(p07::fromstring("1 2 3", DType::Int64, 2, " ").size(), std::int64_t{2});
  CHECK_THROWS_KIND(p07::fromstring("1 2", DType::Float64, 3, " "), ErrorKind::Value);
  CHECK_THROWS_KIND(p07::fromstring("1,2,,3", DType::Float64, -1, ","), ErrorKind::Value);
  CHECK_THROWS_KIND(p07::fromstring("1.5 2", DType::Int64, -1, " "), ErrorKind::Value);
  CHECK_THROWS_KIND(p07::fromstring("1", DType::Float64, -1, ""), ErrorKind::Value);
  NDArray f = p07::fromstring("inf -Infinity nan(x) 1e400 -.5e-3", DType::Float64, -1, " ");
  CHECK(std::isinf(f.get_double(0)));
  CHECK(f.get_double(1) < 0);
  CHECK(std::isnan(f.get_double(2)));
  CHECK(std::isinf(f.get_double(3)));
  CHECK_EQ(f.get_double(4), -0.0005);
  NDArray i8 = p07::fromstring("300 -129 - 5", DType::Int8, -1, " ");
  CHECK_EQ(i8.get_int64(0), std::int64_t{44});
  CHECK_EQ(i8.get_int64(1), std::int64_t{127});
  CHECK_EQ(i8.get_int64(2), std::int64_t{-5});
  CHECK_EQ(p07::fromstring("99999999999999999999", DType::UInt64, -1, " ").get_uint64(0),
           std::uint64_t{18446744073709551615ULL});
  CHECK_EQ(p07::fromstring("-9223372036854775809", DType::Int64, -1, " ").get_int64(0),
           std::int64_t{9223372036854775807LL});
  CHECK_THROWS_KIND(p07::fromstring("-1", DType::UInt8, -1, " "), ErrorKind::Value);
  NDArray c = p07::fromstring("1-1j 2j j 1+2", DType::Complex128, -1, " ");
  const auto* pc = reinterpret_cast<const cd*>(c.data());
  CHECK(pc[0] == cd(1, -1));
  CHECK(pc[1] == cd(0, 2));
  CHECK(pc[2] == cd(0, -1));
  CHECK(pc[3] == cd(1, 0));
  CHECK_EQ(c.size(), std::int64_t{4});
  CHECK_EQ(p07::fromstring("   ", DType::Float64, -1, " ").get_double(0), -1.0);
  CHECK_EQ(p07::fromstring("1x2", DType::Float64, -1, " x ").get_double(1), 2.0);
  NDArray b = p07::fromstring("1.5 0 nan", DType::Bool, -1, " ");
  CHECK_EQ(b.get_int64(0), std::int64_t{1});
  CHECK_EQ(b.get_int64(1), std::int64_t{0});
  CHECK_EQ(b.get_int64(2), std::int64_t{1});
}
