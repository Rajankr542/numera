#include "error.hpp"
#include "ndarray.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
NDArray arange_f64(std::int64_t n) {
  NDArray a = NDArray::empty({n}, DType::Float64);
  for (std::int64_t i = 0; i < n; ++i) a.set_double(i, static_cast<double>(i));
  return a;
}
}  // namespace

TEST_CASE("ndarray: basic metadata") {
  NDArray a = NDArray::zeros({3, 4}, DType::Float32);
  CHECK_EQ(a.ndim(), std::size_t{2});
  CHECK_EQ(a.size(), std::int64_t{12});
  CHECK_EQ(a.itemsize(), std::size_t{4});
  CHECK_EQ(a.nbytes(), std::int64_t{48});
  CHECK(a.strides() == Strides({16, 4}));
  CHECK_EQ(a.offset(), std::int64_t{0});
  CHECK(a.is_c_contiguous());
  CHECK(a.owns_data());
  CHECK_EQ(a.get_double(11), 0.0);
}

TEST_CASE("ndarray: zero-d and empty arrays") {
  NDArray s = NDArray::zeros({}, DType::Int32);
  CHECK_EQ(s.size(), std::int64_t{1});
  CHECK_EQ(s.ndim(), std::size_t{0});
  s.set_int64(0, 42);
  CHECK_EQ(s.get_int64(0), std::int64_t{42});
  NDArray e = NDArray::zeros({0, 3}, DType::Float64);
  CHECK_EQ(e.size(), std::int64_t{0});
  CHECK_THROWS_KIND(e.get_double(0), ErrorKind::Index);
  CHECK_THROWS_KIND(NDArray::zeros({-1}, DType::Float64), ErrorKind::Value);
}

TEST_CASE("ndarray: views share memory (PLAN §9 example)") {
  // a = arange(12).reshape(3,4); b = a[:, 1:3]
  NDArray a = arange_f64(12).reshape({3, 4});
  NDArray b = a.view({3, 2}, {a.strides()[0], a.strides()[1]}, 1 * 8);
  CHECK(b.shares_buffer(a));
  CHECK(!b.owns_data());
  CHECK(!b.is_c_contiguous());
  CHECK(b.strides() == Strides({32, 8}));
  const double expected[] = {1, 2, 5, 6, 9, 10};
  for (int i = 0; i < 6; ++i) CHECK_EQ(b.get_double(i), expected[i]);
  b.set_double(0, 100.0);  // writes through to a
  CHECK_EQ(a.get_double(1), 100.0);
}

TEST_CASE("ndarray: view outlives original handle") {
  const auto before = MemoryBuffer::live_buffers();
  NDArray* view = nullptr;
  {
    NDArray a = arange_f64(10);
    view = new NDArray(a.view({5}, {16}, 0));
  }
  CHECK_EQ(MemoryBuffer::live_buffers(), before + 1);
  CHECK_EQ(view->get_double(4), 8.0);
  delete view;
  CHECK_EQ(MemoryBuffer::live_buffers(), before);
}

TEST_CASE("ndarray: reversed and broadcast views") {
  NDArray a = arange_f64(5);
  NDArray r = a.view({5}, {-8}, 32);
  CHECK_EQ(r.get_double(0), 4.0);
  CHECK_EQ(r.get_double(4), 0.0);
  NDArray bc = a.view({3, 5}, {0, 8}, 0);
  CHECK_EQ(bc.get_double(12), 2.0);
  CHECK_THROWS_KIND(a.view({6}, {8}, 0), ErrorKind::Value);
  CHECK_THROWS_KIND(a.view({2}, {8}, -8), ErrorKind::Value);
}

TEST_CASE("ndarray: copy produces contiguous independent data") {
  NDArray a = arange_f64(12).reshape({3, 4});
  NDArray t = a.view({4, 3}, {8, 32}, 0);  // transpose
  NDArray c = t.copy();
  CHECK(!c.shares_buffer(a));
  CHECK(c.is_c_contiguous());
  CHECK(c.strides() == Strides({24, 8}));
  const double expected[] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11};
  for (int i = 0; i < 12; ++i) CHECK_EQ(c.get_double(i), expected[i]);
}

TEST_CASE("ndarray: reshape view vs copy") {
  NDArray a = arange_f64(12);
  NDArray r = a.reshape({2, -1});
  CHECK(r.shares_buffer(a));
  CHECK(r.shape() == Shape({2, 6}));
  NDArray t = r.view({6, 2}, {8, 48}, 0);  // non-contiguous
  NDArray rt = t.reshape({12});
  CHECK(!rt.shares_buffer(a));
  CHECK_EQ(rt.get_double(1), 6.0);
  CHECK_THROWS_KIND(a.reshape({5, -1}), ErrorKind::Shape);
}

TEST_CASE("ndarray: astype conversions") {
  NDArray a = NDArray::empty({4}, DType::Float64);
  a.set_double(0, 1.5);
  a.set_double(1, -2.7);
  a.set_double(2, 300.0);
  a.set_double(3, 0.0);
  NDArray i8 = a.astype(DType::Int8);
  CHECK_EQ(i8.get_int64(0), std::int64_t{1});
  CHECK_EQ(i8.get_int64(1), std::int64_t{-2});
  CHECK_EQ(i8.get_int64(2), std::int64_t{44});
  NDArray b = a.astype(DType::Bool);
  CHECK_EQ(b.get_int64(3), std::int64_t{0});
  CHECK_EQ(b.get_int64(1), std::int64_t{1});
  NDArray h = a.astype(DType::Float16);
  CHECK_EQ(h.get_double(0), 1.5);
  NDArray u = NDArray::empty({1}, DType::UInt64);
  u.set_uint64(0, 18446744073709551615ULL);
  CHECK_EQ(u.get_uint64(0), 18446744073709551615ULL);
}

TEST_CASE("ndarray: byte_offset_of with negative indices") {
  NDArray a = arange_f64(12).reshape({3, 4});
  CHECK_EQ(a.byte_offset_of({1, 2}), std::int64_t{48});
  CHECK_EQ(a.byte_offset_of({-1, -1}), std::int64_t{88});
  CHECK_THROWS_KIND(a.byte_offset_of({3, 0}), ErrorKind::Index);
  CHECK_THROWS_KIND(a.byte_offset_of({0}), ErrorKind::Index);
}

TEST_CASE("ndarray: strided reshape returns a view when possible") {
  NDArray a = arange_f64(24);
  NDArray every_other = a.view({12}, {16}, 0);
  NDArray r = every_other.reshape({3, 4});
  CHECK(r.shares_buffer(a));
  CHECK(r.strides() == Strides({64, 16}));
  CHECK_EQ(r.get_double(5), 10.0);
}

TEST_CASE("ndarray: may_share_memory uses byte extents") {
  NDArray a = arange_f64(24);
  NDArray head = a.view({5}, {8}, 0);
  NDArray tail = a.view({5}, {8}, 80);
  NDArray none = a.view({0}, {8}, 0);
  CHECK(head.may_share_memory(a));
  CHECK(!head.may_share_memory(tail));
  CHECK(!none.may_share_memory(a));
  CHECK(!a.may_share_memory(a.copy()));
  CHECK(NDArray::zeros({2, 0, 3}, DType::Float64).strides() == Strides({0, 0, 0}));
}

