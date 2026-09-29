#include "error.hpp"
#include "shape.hpp"
#include "strides.hpp"
#include "test_harness.hpp"

using namespace nativpy;

TEST_CASE("shape: size, validation, overflow") {
  CHECK_EQ(shape_size({}), std::int64_t{1});
  CHECK_EQ(shape_size({3, 4}), std::int64_t{12});
  CHECK_EQ(shape_size({3, 0, 4}), std::int64_t{0});
  CHECK_THROWS_KIND(validate_shape({2, -1}), ErrorKind::Value);
  CHECK_THROWS_KIND(shape_size({1LL << 40, 1LL << 40}), ErrorKind::Value);
  CHECK_THROWS_KIND(validate_shape(Shape(65, 1)), ErrorKind::Value);
  validate_shape(Shape(64, 1));
}

TEST_CASE("shape: normalize_axis") {
  CHECK_EQ(normalize_axis(-1, 3), std::int64_t{2});
  CHECK_EQ(normalize_axis(0, 3), std::int64_t{0});
  CHECK_THROWS_KIND(normalize_axis(3, 3), ErrorKind::Index);
  CHECK_THROWS_KIND(normalize_axis(-4, 3), ErrorKind::Index);
}

TEST_CASE("shape: resolve_reshape") {
  CHECK(resolve_reshape({3, -1}, 12) == Shape({3, 4}));
  CHECK(resolve_reshape({-1}, 0) == Shape({0}));
  CHECK(resolve_reshape({0, 5}, 0) == Shape({0, 5}));
  CHECK_THROWS_KIND(resolve_reshape({0, -1}, 0), ErrorKind::Shape);
  CHECK_THROWS_KIND(resolve_reshape({5, -1}, 12), ErrorKind::Shape);
  CHECK_THROWS_KIND(resolve_reshape({-1, -1}, 12), ErrorKind::Value);
  CHECK_THROWS_KIND(resolve_reshape({5}, 12), ErrorKind::Shape);
  CHECK_EQ(shape_to_string({3}), std::string("(3,)"));
  CHECK_EQ(shape_to_string({}), std::string("()"));
}

TEST_CASE("strides: c-contiguous strides match NumPy") {
  CHECK(c_contiguous_strides({3, 4}, 8) == Strides({32, 8}));
  CHECK(c_contiguous_strides({2, 3, 4}, 4) == Strides({48, 16, 4}));
  CHECK(c_contiguous_strides({}, 8).empty());
  CHECK(c_contiguous_strides({2, 0, 3}, 8) == Strides({24, 24, 8}));
}

TEST_CASE("strides: contiguity flags") {
  CHECK(is_c_contiguous({3, 4}, {32, 8}, 8));
  CHECK(!is_f_contiguous({3, 4}, {32, 8}, 8));
  CHECK(is_f_contiguous({3, 4}, {8, 24}, 8));
  CHECK(is_c_contiguous({3, 1}, {8, 999}, 8));  // size-1 dims ignored
  CHECK(!is_c_contiguous({3, 2}, {32, 8}, 8));  // sliced columns
  CHECK(is_c_contiguous({0, 4}, {1, 1}, 8));     // empty
}

TEST_CASE("strides: view bounds checking") {
  check_view_bounds({3, 4}, {32, 8}, 0, 8, 96);
  CHECK_THROWS_KIND(check_view_bounds({3, 4}, {32, 8}, 8, 8, 96), ErrorKind::Value);
  // Reversed view starting at the last element.
  check_view_bounds({4}, {-8}, 24, 8, 32);
  CHECK_THROWS_KIND(check_view_bounds({5}, {-8}, 24, 8, 32), ErrorKind::Value);
  check_view_bounds({0}, {8}, 32, 8, 32);  // empty at end
  // Broadcast (zero stride) view of a single element.
  check_view_bounds({1000, 1000}, {0, 0}, 0, 8, 8);
}

TEST_CASE("strides: allocation strides zero for empty arrays (NumPy)") {
  CHECK(allocation_strides({2, 0, 3}, 8) == Strides({0, 0, 0}));
  CHECK(allocation_strides({0}, 1) == Strides({0}));
  CHECK(allocation_strides({2, 3}, 8) == Strides({24, 8}));
  CHECK(allocation_strides({}, 8).empty());
}

TEST_CASE("strides: nocopy reshape matches NumPy") {
  Strides s;
  // arange(24)[::2].reshape(3,4) -> (64,16)
  CHECK(attempt_nocopy_reshape({12}, {16}, {3, 4}, 8, s));
  CHECK(s == Strides({64, 16}));
  // arange(24).reshape(4,6).T.reshape(6,2,2) -> (8,96,48); flatten copies.
  CHECK(attempt_nocopy_reshape({6, 4}, {8, 48}, {6, 2, 2}, 8, s));
  CHECK(s == Strides({8, 96, 48}));
  CHECK(!attempt_nocopy_reshape({6, 4}, {8, 48}, {24}, 8, s));
  // (2,3,4)[:, ::2, :] -> (2,2,2,2) view (96,64,16,8); (2,8) copies.
  CHECK(attempt_nocopy_reshape({2, 2, 4}, {96, 64, 8}, {2, 2, 2, 2}, 8, s));
  CHECK(s == Strides({96, 64, 16, 8}));
  CHECK(!attempt_nocopy_reshape({2, 2, 4}, {96, 64, 8}, {2, 8}, 8, s));
  // Broadcast rows (2,3)/(0,8): trailing size-1 dim is a view, flatten copies.
  CHECK(attempt_nocopy_reshape({2, 3}, {0, 8}, {2, 3, 1}, 8, s));
  CHECK(s == Strides({0, 8, 8}));
  CHECK(!attempt_nocopy_reshape({2, 3}, {0, 8}, {6}, 8, s));
  // Scalar -> (1,1) and (1,) -> ().
  CHECK(attempt_nocopy_reshape({}, {}, {1, 1}, 8, s));
  CHECK(s == Strides({8, 8}));
  CHECK(attempt_nocopy_reshape({1}, {8}, {}, 8, s));
  CHECK(s.empty());
}

