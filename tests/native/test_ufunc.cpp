#include <cmath>
#include <cstdint>

#include "broadcast.hpp"
#include "creation.hpp"
#include "error.hpp"
#include "shape_ops.hpp"
#include "test_harness.hpp"
#include "ufunc.hpp"

using namespace nativpy;

namespace {
NDArray vec_i(std::initializer_list<std::int64_t> v, DType dt) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t i = 0;
  for (auto x : v) a.set_int64(i++, x);
  return a;
}
NDArray vec_d(std::initializer_list<double> v, DType dt = DType::Float64) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  std::int64_t i = 0;
  for (auto x : v) a.set_double(i++, x);
  return a;
}
}  // namespace

TEST_CASE("broadcast: shapes") {
  CHECK(broadcast_shapes({{3, 1}, {4}}) == Shape({3, 4}));
  CHECK(broadcast_shapes({{}, {2, 3}}) == Shape({2, 3}));
  CHECK(broadcast_shapes({{0, 1}, {1, 5}}) == Shape({0, 5}));
  CHECK(broadcast_shapes({}) == Shape({}));
  CHECK_THROWS_KIND(broadcast_shapes({{3}, {4}}), ErrorKind::Broadcast);
  CHECK_THROWS_KIND(broadcast_shapes({{2, 0}, {3}}), ErrorKind::Broadcast);
}

TEST_CASE("broadcast: to / plan coalescing") {
  NDArray a = vec_d({1, 2, 3});
  NDArray b = broadcast_to(a, {4, 3});
  CHECK(b.strides() == Strides({0, 8}));
  CHECK(!b.owns_data());
  CHECK_EQ(b.get_double(10), 2.0);
  CHECK_THROWS_KIND(broadcast_to(a, {4}), ErrorKind::Broadcast);
  CHECK_THROWS_KIND(broadcast_to(ones({2, 3}, DType::Float64), {3}), ErrorKind::Broadcast);

  NDArray x = ones({2, 3, 4}, DType::Float64);
  NDArray y = ones({2, 3, 4}, DType::Float64);
  auto p = make_plan<2>(x.shape(), {&x, &y});
  CHECK(p.shape == Shape({24}));  // fully contiguous -> one loop
  NDArray t = transpose(x, {});
  auto q = make_plan<2>(t.shape(), {&t, &t});
  CHECK_EQ(q.shape.size(), std::size_t{3});  // no merge possible
  NDArray r = ones({3, 1}, DType::Float64);
  auto s = make_plan<2>({3, 4}, {&r, &r});
  CHECK(s.shape == Shape({3, 4}));
  CHECK(s.strides[0] == Strides({8, 0}));
}

TEST_CASE("ufunc: result dtypes") {
  CHECK(binary_result_dtype(BinaryOp::Divide, DType::Int32, DType::Int32) == DType::Float64);
  CHECK(binary_result_dtype(BinaryOp::Divide, DType::Float16, DType::Float16) == DType::Float16);
  CHECK(binary_result_dtype(BinaryOp::Power, DType::Bool, DType::Bool) == DType::Int8);
  CHECK(binary_result_dtype(BinaryOp::Add, DType::Bool, DType::Bool) == DType::Bool);
  CHECK(binary_result_dtype(BinaryOp::Add, DType::Int64, DType::UInt64) == DType::Float64);
  CHECK_THROWS_KIND(binary_result_dtype(BinaryOp::Subtract, DType::Bool, DType::Bool), ErrorKind::DType);
  // P1 (D-033): complex loops exist; complex mod/floorDivide have none.
  CHECK(binary_result_dtype(BinaryOp::Add, DType::Complex64, DType::Float32) == DType::Complex64);
  CHECK(binary_result_dtype(BinaryOp::Divide, DType::Complex64, DType::Float64) == DType::Complex128);
  CHECK_THROWS_KIND(binary_result_dtype(BinaryOp::FloorDivide, DType::Complex64, DType::Float32),
                    ErrorKind::DType);
  CHECK(unary_result_dtype(UnaryOp::Sqrt, DType::UInt8) == DType::Float16);
  CHECK(unary_result_dtype(UnaryOp::Sqrt, DType::Int16) == DType::Float32);
  CHECK(unary_result_dtype(UnaryOp::Log, DType::Int32) == DType::Float64);
  CHECK_THROWS_KIND(unary_result_dtype(UnaryOp::Negative, DType::Bool), ErrorKind::DType);
}

TEST_CASE("ufunc: integer edge cases") {
  NDArray a = vec_i({7, -7, 0, 5}, DType::Int32);
  NDArray b = vec_i({2, 2, 0, -3}, DType::Int32);
  NDArray m = binary(BinaryOp::Mod, a, b);
  NDArray f = binary(BinaryOp::FloorDivide, a, b);
  const std::int64_t em[] = {1, 1, 0, -1};
  const std::int64_t ef[] = {3, -4, 0, -2};
  for (int i = 0; i < 4; ++i) {
    CHECK_EQ(m.get_int64(i), em[i]);
    CHECK_EQ(f.get_int64(i), ef[i]);
  }
  NDArray mn = vec_i({-128}, DType::Int8);
  NDArray neg1 = vec_i({-1}, DType::Int8);
  CHECK_EQ(binary(BinaryOp::FloorDivide, mn, neg1).get_int64(0), std::int64_t{-128});
  CHECK_EQ(binary(BinaryOp::Mod, mn, neg1).get_int64(0), std::int64_t{0});
  CHECK_EQ(unary(UnaryOp::Abs, mn).get_int64(0), std::int64_t{-128});
  CHECK_EQ(binary(BinaryOp::Add, vec_i({127}, DType::Int8), vec_i({1}, DType::Int8)).get_int64(0),
           std::int64_t{-128});
  CHECK_EQ(binary(BinaryOp::Subtract, vec_i({1}, DType::UInt8), vec_i({2}, DType::UInt8)).get_int64(0),
           std::int64_t{255});
  CHECK_EQ(binary(BinaryOp::Multiply, vec_i({65535}, DType::UInt16), vec_i({65535}, DType::UInt16))
               .get_int64(0),
           std::int64_t{1});
  NDArray p = binary(BinaryOp::Power, vec_i({2, 3}, DType::Int8), vec_i({7, 5}, DType::Int8));
  CHECK_EQ(p.get_int64(0), std::int64_t{-128});
  CHECK_EQ(p.get_int64(1), std::int64_t{-13});
  CHECK_THROWS_KIND(binary(BinaryOp::Power, vec_i({2}, DType::Int64), vec_i({-1}, DType::Int64)),
                    ErrorKind::Value);
}

TEST_CASE("ufunc: float semantics and broadcasting") {
  NDArray m = binary(BinaryOp::Mod, vec_d({-7.5, 7.5, 1.0}), vec_d({2.0, -2.0, 0.0}));
  CHECK_EQ(m.get_double(0), 0.5);
  CHECK_EQ(m.get_double(1), -0.5);
  CHECK(std::isnan(m.get_double(2)));
  NDArray d = binary(BinaryOp::Divide, vec_i({1, 0, -1}, DType::Int64), vec_i({0, 0, 0}, DType::Int64));
  CHECK(d.dtype() == DType::Float64);
  CHECK(std::isinf(d.get_double(0)) && d.get_double(0) > 0);
  CHECK(std::isnan(d.get_double(1)));
  NDArray h = binary(BinaryOp::Add, vec_d({1.5}, DType::Float16), vec_d({0.001}, DType::Float16));
  CHECK_EQ(h.get_double(0), 1.5009765625);
  NDArray col = arange(0, 3, 1, DType::Float64).reshape({3, 1});
  NDArray row = arange(0, 4, 1, DType::Float64);
  NDArray s = binary(BinaryOp::Add, col, row);
  CHECK(s.shape() == Shape({3, 4}));
  CHECK(s.is_c_contiguous());
  CHECK_EQ(s.get_double(11), 5.0);
  NDArray t = binary(BinaryOp::Subtract, transpose(s, {}), vec_d({1}));
  CHECK(t.shape() == Shape({4, 3}));
  CHECK_EQ(t.get_double(1), 0.0);  // s[1,0] - 1
  CHECK_EQ(binary(BinaryOp::Add, ones({0, 3}, DType::Float64), ones({1, 3}, DType::Float64)).size(),
           std::int64_t{0});
  CHECK(std::isnan(unary(UnaryOp::Sqrt, vec_d({-1})).get_double(0)));
  NDArray lg = unary(UnaryOp::Log, vec_d({0}));
  CHECK(std::isinf(lg.get_double(0)) && lg.get_double(0) < 0);
  NDArray z = NDArray::empty({}, DType::Float64);
  z.set_double(0, 4.0);
  NDArray r0 = unary(UnaryOp::Sqrt, z);
  CHECK(r0.shape() == Shape({}));
  CHECK_EQ(r0.get_double(0), 2.0);
}


// Expected values from NumPy 2.5.3 (D-046).
TEST_CASE("ufunc: out= dtype casts and broadcast") {
  NDArray o16 = NDArray::zeros({2}, DType::Int16);
  NDArray r = binary(BinaryOp::Add, vec_i({100, 100}, DType::Int8), vec_i({100, 100}, DType::Int8), o16);
  CHECK(r.shares_buffer(o16));
  CHECK_EQ(o16.get_int64(0), std::int64_t{-56});  // loop dtype int8 wraps
  NDArray f32 = NDArray::zeros({2}, DType::Float32);
  binary(BinaryOp::Add, vec_d({1.5, 2.0}), vec_d({1.0}), f32);
  CHECK_EQ(f32.get_double(0), 2.5);
  NDArray c = NDArray::zeros({2}, DType::Complex128);
  binary(BinaryOp::Add, vec_d({1.5, 2.0}), vec_d({1.0}), c);
  CHECK_EQ(c.get_double(1), 3.0);
  NDArray o23 = NDArray::zeros({2, 3}, DType::Float64);
  binary(BinaryOp::Add, vec_d({1, 2, 3}), vec_d({1}), o23);
  CHECK_EQ(o23.get_double(5), 4.0);
  NDArray u = NDArray::zeros({2, 3}, DType::Float64);
  unary(UnaryOp::Negative, vec_d({1}), u);
  CHECK_EQ(u.get_double(4), -1.0);
  NDArray s = NDArray::zeros({2}, DType::Float32);
  unary(UnaryOp::Sqrt, vec_i({4, 9}, DType::Int64), s);
  CHECK_EQ(s.get_double(1), 3.0);
  NDArray cz = NDArray::empty({1}, DType::Complex128);
  cz.set_double(0, 0.0);
  NDArray ab = binary(BinaryOp::Add, cz, vec_d({0}), NDArray::zeros({1}, DType::Complex128));
  CHECK_EQ(ab.get_double(0), 0.0);
  NDArray absf = NDArray::zeros({1}, DType::Float32);
  unary(UnaryOp::Abs, cz, absf);
  CHECK_EQ(absf.get_double(0), 0.0);
  NDArray empty = NDArray::zeros({0}, DType::Float64);
  CHECK_EQ(binary(BinaryOp::Add, empty, vec_d({1}), empty).size(), std::int64_t{0});
}

TEST_CASE("ufunc: out= strided and overlapping") {
  NDArray a = arange(0, 6, 1, DType::Float64);
  const auto st = a.strides()[0];
  NDArray head = a.view({5}, {st}, 0), tail = a.view({5}, {st}, st);
  binary(BinaryOp::Add, head, tail, tail);  // NumPy: [0,1,3,5,7,9]
  const double want1[] = {0, 1, 3, 5, 7, 9};
  for (int i = 0; i < 6; ++i) CHECK_EQ(a.get_double(i), want1[i]);
  a = arange(0, 6, 1, DType::Float64);
  binary(BinaryOp::Add, a.view({5}, {st}, st), a.view({5}, {st}, 0), a.view({5}, {st}, 0));
  const double want2[] = {1, 3, 5, 7, 9, 5};
  for (int i = 0; i < 6; ++i) CHECK_EQ(a.get_double(i), want2[i]);
  a = arange(0, 6, 1, DType::Float64);
  binary(BinaryOp::Add, a, a, a);
  CHECK_EQ(a.get_double(5), 10.0);
  a = arange(0, 6, 1, DType::Float64);
  unary(UnaryOp::Negative, a.view({6}, {-st}, 5 * st), a);  // reversed overlap
  CHECK_EQ(a.get_double(0), -5.0);
  CHECK_EQ(a.get_double(5), 0.0);
  a = arange(0, 4, 1, DType::Float64);
  binary(BinaryOp::Add, a.view({1}, {st}, 0), a, a);  // broadcast input overlaps out
  CHECK_EQ(a.get_double(3), 3.0);
  a = arange(0, 6, 1, DType::Float64);
  binary(BinaryOp::Multiply, a.view({3}, {2 * st}, 0), a.view({3}, {2 * st}, st), a.view({3}, {st}, 0));
  CHECK_EQ(a.get_double(1), 6.0);
  CHECK_EQ(a.get_double(2), 20.0);
  NDArray z = NDArray::zeros({6}, DType::Float64);
  binary(BinaryOp::Add, ones({3}, DType::Float64), vec_d({2}), z.view({3}, {2 * st}, 0));
  CHECK_EQ(z.get_double(2), 3.0);
  CHECK_EQ(z.get_double(1), 0.0);
  NDArray i8 = arange(0, 4, 1, DType::Int8);
  NDArray f = NDArray::zeros({4}, DType::Float64);
  binary(BinaryOp::Add, i8, i8, f);  // cast path
  CHECK_EQ(f.get_double(3), 6.0);
}

TEST_CASE("ufunc: out= errors in NumPy order") {
  NDArray ro = NDArray::zeros({3}, DType::Int64).as_readonly();
  CHECK_THROWS_KIND(binary(BinaryOp::Add, vec_d({1, 1, 1}), vec_d({1.5}), ro), ErrorKind::Value);
  CHECK_THROWS_KIND(unary(UnaryOp::Sqrt, vec_d({1}), ro), ErrorKind::Value);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, vec_d({1}), vec_d({1}), broadcast_to(vec_d({0}), {3})),
                    ErrorKind::Value);
  NDArray i64 = NDArray::zeros({3}, DType::Int64);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, ones({2, 3}, DType::Float64), vec_d({1}), i64), ErrorKind::DType);
  CHECK_THROWS_KIND(binary(BinaryOp::Divide, vec_i({1, 2, 3}, DType::Int64), vec_i({1}, DType::Int64), i64),
                    ErrorKind::DType);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, vec_i({1}, DType::Int64), vec_i({1}, DType::Int64),
                           NDArray::zeros({1}, DType::UInt8)),
                    ErrorKind::DType);
  CHECK_THROWS_KIND(binary(BinaryOp::Power, vec_i({1, 1, 1}, DType::Int64), vec_i({-1}, DType::Int64),
                           NDArray::zeros({3}, DType::Bool)),
                    ErrorKind::DType);
  CHECK_THROWS_KIND(binary(BinaryOp::Power, vec_i({1, 1, 1}, DType::Int64), vec_i({-1}, DType::Int64),
                           NDArray::zeros({4}, DType::Float64)),
                    ErrorKind::Broadcast);
  CHECK_THROWS_KIND(binary(BinaryOp::Power, vec_i({1}, DType::Int64), vec_i({-1}, DType::Int64),
                           NDArray::zeros({1}, DType::Float64)),
                    ErrorKind::Value);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, ones({2, 3}, DType::Float64), vec_d({1}),
                           NDArray::zeros({3}, DType::Float64)),
                    ErrorKind::Broadcast);
  CHECK_THROWS_KIND(unary(UnaryOp::Negative, ones({2, 3}, DType::Float64), NDArray::zeros({3}, DType::Float64)),
                    ErrorKind::Broadcast);
  CHECK_THROWS_KIND(unary(UnaryOp::Abs, NDArray::zeros({1}, DType::Complex128), NDArray::zeros({1}, DType::Int64)),
                    ErrorKind::DType);
  CHECK_THROWS_KIND(unary(UnaryOp::Negative, NDArray::zeros({1}, DType::Bool), NDArray::zeros({1}, DType::Int64)),
                    ErrorKind::DType);
}

// dtype= / casting= (D-048). Expected values from NumPy 2.5.3.
TEST_CASE("ufunc: dtype= selects the loop") {
  const NDArray i8 = vec_i({100, 2}, DType::Int8);
  UfuncParams p{DType::Int16, Casting::SameKind};
  NDArray r = binary(BinaryOp::Add, i8, i8, nullptr, p);  // no int8 wrap
  CHECK(r.dtype() == DType::Int16);
  CHECK_EQ(r.get_int64(0), 200);
  // Unsafe input cast truncates: int8(1) + int8(2.7) = 3.
  p = {DType::Int8, Casting::Unsafe};
  r = binary(BinaryOp::Add, vec_i({1}, DType::Int8), vec_d({2.7}), nullptr, p);
  CHECK(r.dtype() == DType::Int8);
  CHECK_EQ(r.get_int64(0), 3);
  // sqrt int64 with dtype=float32 under same_kind (int64 -> float32 is same_kind).
  p = {DType::Float32, Casting::SameKind};
  r = unary(UnaryOp::Sqrt, vec_i({4}, DType::Int64), nullptr, p);
  CHECK(r.dtype() == DType::Float32);
  CHECK_EQ(r.get_double(0), 2.0);
  // divide with float32 loop
  r = binary(BinaryOp::Divide, vec_i({1}, DType::Int8), vec_i({4}, DType::Int8), nullptr, p);
  CHECK(r.dtype() == DType::Float32);
  CHECK_EQ(r.get_double(0), 0.25);
}

TEST_CASE("ufunc: dtype= abs of complex") {
  // c = 3+4j built as 3 + 4*sqrt(-1) via complex ops.
  NDArray j = unary(UnaryOp::Sqrt, vec_d({-1}).astype(DType::Complex128));
  const NDArray c = binary(BinaryOp::Add, vec_d({3}), binary(BinaryOp::Multiply, vec_d({4}), j));
  // complex128 -> float64 uses the D->d loop: |3+4j| = 5.
  NDArray r = unary(UnaryOp::Abs, c, nullptr, {DType::Float64, Casting::SameKind});
  CHECK(r.dtype() == DType::Float64);
  CHECK_EQ(r.get_double(0), 5.0);
  // complex64 -> float64 also uses D->d (c8 casts safely to c16).
  r = unary(UnaryOp::Abs, c.astype(DType::Complex64), nullptr, {DType::Float64, Casting::SameKind});
  CHECK_EQ(r.get_double(0), 5.0);
  // complex128 -> float32: needs unsafe, then the f->f loop on the real part.
  CHECK_THROWS_KIND(unary(UnaryOp::Abs, c, nullptr, {DType::Float32, Casting::SameKind}), ErrorKind::DType);
  r = unary(UnaryOp::Abs, c, nullptr, {DType::Float32, Casting::Unsafe});
  CHECK(r.dtype() == DType::Float32);
  CHECK_EQ(r.get_double(0), 3.0);
  CHECK_THROWS_KIND(unary(UnaryOp::Abs, c, nullptr, {DType::Complex128, Casting::Unsafe}), ErrorKind::DType);
}

TEST_CASE("ufunc: casting= checks inputs and out") {
  const NDArray i8 = vec_i({1}, DType::Int8);
  const NDArray i16 = vec_i({1}, DType::Int16);
  UfuncParams no{std::nullopt, Casting::No};
  CHECK_EQ(binary(BinaryOp::Add, i8, i8, nullptr, no).get_int64(0), 2);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, i8, i16, nullptr, no), ErrorKind::DType);
  CHECK_THROWS_KIND(unary(UnaryOp::Sqrt, i8, nullptr, no), ErrorKind::DType);
  // complex abs reads its input directly, so casting=no passes.
  CHECK(unary(UnaryOp::Abs, NDArray::zeros({1}, DType::Complex64), nullptr, no).dtype() == DType::Float32);
  // out: float64 loop into int8 out requires unsafe.
  NDArray out = NDArray::zeros({1}, DType::Int8);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, vec_d({1.5}), vec_d({1.5}), &out, {}), ErrorKind::DType);
  binary(BinaryOp::Add, vec_d({1.5}), vec_d({1.5}), &out, {std::nullopt, Casting::Unsafe});
  CHECK_EQ(out.get_int64(0), 3);
  // safe forbids int64 -> int8 out even though same_kind allows it.
  NDArray o8 = NDArray::zeros({1}, DType::Int8);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, vec_i({1}, DType::Int64), vec_i({1}, DType::Int64), &o8,
                           {std::nullopt, Casting::Safe}),
                    ErrorKind::DType);
}

TEST_CASE("ufunc: dtype= missing loops and error order") {
  const NDArray i8 = vec_i({1}, DType::Int8);
  const UfuncParams u{DType::Int8, Casting::Unsafe};
  CHECK_THROWS_KIND(binary(BinaryOp::Divide, i8, i8, nullptr, u), ErrorKind::DType);
  CHECK_THROWS_KIND(unary(UnaryOp::Sqrt, i8, nullptr, u), ErrorKind::DType);
  const UfuncParams b{DType::Bool, Casting::Unsafe};
  CHECK_THROWS_KIND(binary(BinaryOp::Power, i8, i8, nullptr, b), ErrorKind::DType);
  CHECK_THROWS_KIND(binary(BinaryOp::Subtract, i8, i8, nullptr, b), ErrorKind::DType);
  CHECK_THROWS_KIND(unary(UnaryOp::Negative, i8, nullptr, b), ErrorKind::DType);
  const UfuncParams c{DType::Complex128, Casting::Unsafe};
  CHECK_THROWS_KIND(binary(BinaryOp::Mod, i8, i8, nullptr, c), ErrorKind::DType);
  CHECK_THROWS_KIND(binary(BinaryOp::FloorDivide, i8, i8, nullptr, c), ErrorKind::DType);
  // dtype=bool add/multiply exist (logical or / and).
  CHECK_EQ(binary(BinaryOp::Add, i8, vec_i({0}, DType::Int8), nullptr, b).get_int64(0), 1);
  // read-only out wins over a missing loop (NumPy order).
  NDArray ro = NDArray::zeros({1}, DType::Int8).as_readonly();
  CHECK_THROWS_KIND(binary(BinaryOp::Divide, i8, i8, &ro, u), ErrorKind::Value);
  // negative exponent check still runs with dtype=.
  CHECK_THROWS_KIND(binary(BinaryOp::Power, i8, vec_i({-1}, DType::Int8), nullptr,
                           {DType::Int16, Casting::SameKind}),
                    ErrorKind::Value);
}


// where= (D-049). Expected values from NumPy 2.5.3.
TEST_CASE("ufunc: where= with and without out") {
  const NDArray a = vec_d({1, 2, 3});
  const NDArray b = vec_d({10, 20, 30});
  const NDArray m = vec_i({1, 0, 1}, DType::Int64).astype(DType::Bool);
  UfuncParams p;
  p.where = m;
  NDArray out = vec_d({7.5, 7.5, 7.5});
  binary(BinaryOp::Add, a, b, &out, p);
  CHECK_EQ(out.get_double(0), 11.0);
  CHECK_EQ(out.get_double(1), 7.5);  // untouched
  CHECK_EQ(out.get_double(2), 33.0);
  // Without out: masked positions are zero (numera's deterministic choice).
  NDArray r = binary(BinaryOp::Add, a, b, nullptr, p);
  CHECK_EQ(r.get_double(1), 0.0);
  CHECK_EQ(r.get_double(2), 33.0);
  // Unary into a narrower out with an unsafe cast.
  NDArray o8 = vec_i({5, 5, 5}, DType::Int8);
  p.casting = Casting::Unsafe;
  unary(UnaryOp::Sqrt, vec_d({4, 9, 16}), &o8, p);
  CHECK_EQ(o8.get_int64(0), 2);
  CHECK_EQ(o8.get_int64(1), 5);
  CHECK_EQ(o8.get_int64(2), 4);
}

TEST_CASE("ufunc: where= broadcasting and shape errors") {
  // The mask expands the result shape: add(1., 2., where=ones((2,3))).
  UfuncParams p;
  NDArray mrow = NDArray::zeros({2, 1}, DType::Bool);
  mrow.set_int64(0, 1);
  p.where = mrow;
  NDArray r = binary(BinaryOp::Add, vec_d({1, 2, 3}), vec_d({10}), nullptr, p);
  CHECK((r.shape() == Shape{2, 3}));
  CHECK_EQ(r.get_double(2), 13.0);
  CHECK_EQ(r.get_double(3), 0.0);
  // Incompatible mask, and a mask larger than out.
  p.where = NDArray::zeros({2}, DType::Bool);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, vec_d({1, 2, 3}), vec_d({1}), nullptr, p), ErrorKind::Broadcast);
  CHECK_THROWS_KIND(unary(UnaryOp::Negative, vec_d({1, 2, 3}), nullptr, p), ErrorKind::Broadcast);
  p.where = NDArray::zeros({2, 3}, DType::Bool);
  NDArray out = NDArray::zeros({3}, DType::Float64);
  CHECK_THROWS_KIND(binary(BinaryOp::Add, vec_d({1, 2, 3}), vec_d({1}), &out, p), ErrorKind::Broadcast);
}

TEST_CASE("ufunc: where= dtype, error order and masked values") {
  const NDArray i64 = vec_i({2, 2}, DType::Int64);
  UfuncParams p;
  p.where = vec_i({1, 0}, DType::Int64);  // not bool: refused (safe cast)
  CHECK_THROWS_KIND(binary(BinaryOp::Add, i64, i64, nullptr, p), ErrorKind::DType);
  // read-only out is reported before the mask dtype.
  NDArray ro = NDArray::zeros({2}, DType::Int64).as_readonly();
  CHECK_THROWS_KIND(binary(BinaryOp::Add, i64, i64, &ro, p), ErrorKind::Value);
  // Input cast errors come before mask shape errors.
  p.where = NDArray::zeros({3}, DType::Bool);
  p.casting = Casting::No;
  CHECK_THROWS_KIND(binary(BinaryOp::Add, i64, vec_d({1.5}), nullptr, p), ErrorKind::DType);
  // Negative exponent only matters where the mask is true.
  const NDArray e = vec_i({1, -1}, DType::Int64);
  p = UfuncParams{};
  p.where = vec_i({1, 0}, DType::Int64).astype(DType::Bool);
  NDArray out = NDArray::zeros({2}, DType::Int64);
  binary(BinaryOp::Power, i64, e, &out, p);
  CHECK_EQ(out.get_int64(0), 2);
  CHECK_EQ(out.get_int64(1), 0);
  p.where = vec_i({1, 1}, DType::Int64).astype(DType::Bool);
  CHECK_THROWS_KIND(binary(BinaryOp::Power, i64, e, &out, p), ErrorKind::Value);
  // A mask aliasing out: computed as if the mask were read first.
  NDArray o = vec_i({1, 0, 1}, DType::Int8);
  p.where = NDArray{o.buffer(), DType::Bool, o.shape(), o.strides(), o.offset()};
  binary(BinaryOp::Add, o, o, &o, p);
  CHECK_EQ(o.get_int64(0), 2);
  CHECK_EQ(o.get_int64(1), 0);
  CHECK_EQ(o.get_int64(2), 2);
}

