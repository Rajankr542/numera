#include <cmath>
#include <limits>

#include "cast.hpp"
#include "dtype.hpp"
#include "test_harness.hpp"

using namespace nativpy;

TEST_CASE("dtype: item sizes and names") {
  CHECK_EQ(itemsize(DType::Bool), std::size_t{1});
  CHECK_EQ(itemsize(DType::Float16), std::size_t{2});
  CHECK_EQ(itemsize(DType::Int64), std::size_t{8});
  CHECK_EQ(itemsize(DType::Complex64), std::size_t{8});
  CHECK_EQ(itemsize(DType::Complex128), std::size_t{16});
  CHECK(dtype_name(DType::UInt32) == "uint32");
  CHECK(dtype_from_name("float32") == DType::Float32);
  CHECK(!dtype_from_name("float128").has_value());
  for (int i = 0; i < kNumDTypes; ++i) {
    const auto dt = static_cast<DType>(i);
    CHECK(dtype_from_name(dtype_name(dt)) == dt);
  }
}

TEST_CASE("dtype: promotion matches NumPy samples") {
  CHECK(promote_types(DType::Int8, DType::UInt8) == DType::Int16);
  CHECK(promote_types(DType::Int64, DType::UInt64) == DType::Float64);
  CHECK(promote_types(DType::Float16, DType::Int16) == DType::Float32);
  CHECK(promote_types(DType::Complex64, DType::Float64) == DType::Complex128);
  CHECK(promote_types(DType::Bool, DType::Bool) == DType::Bool);
  for (int i = 0; i < kNumDTypes; ++i) {
    for (int j = 0; j < kNumDTypes; ++j) {
      const auto a = static_cast<DType>(i);
      const auto b = static_cast<DType>(j);
      CHECK(promote_types(a, b) == promote_types(b, a));
    }
  }
}

TEST_CASE("float16: round trip and rounding") {
  CHECK_EQ(half_to_double(double_to_half(1.0)), 1.0);
  CHECK_EQ(half_to_double(double_to_half(-2.5)), -2.5);
  CHECK_EQ(half_to_double(double_to_half(65504.0)), 65504.0);
  CHECK(std::isinf(half_to_double(double_to_half(65520.0))));  // rounds to inf
  CHECK_EQ(half_to_double(double_to_half(65519.0)), 65504.0);
  CHECK_EQ(half_to_double(double_to_half(6e-8)), 5.960464477539063e-08);
  CHECK_EQ(half_to_double(double_to_half(1e-8)), 0.0);
  CHECK(std::isnan(half_to_double(double_to_half(NAN))));
  CHECK_EQ(double_to_half(-0.0).bits, std::uint16_t{0x8000});
  CHECK_EQ(double_to_half(1.0 + 1.0 / 2048).bits, std::uint16_t{0x3C00});  // tie->even
  CHECK_EQ(double_to_half(1.0 + 3.0 / 2048).bits, std::uint16_t{0x3C02});  // tie->even
}

TEST_CASE("cast: float to int is deterministic") {
  CHECK_EQ((cast_value<std::int32_t>(NAN)), 0);
  CHECK_EQ((cast_value<std::int32_t>(INFINITY)), std::numeric_limits<std::int32_t>::max());
  CHECK_EQ((cast_value<std::int64_t>(-1e30)), std::numeric_limits<std::int64_t>::min());
  CHECK_EQ((cast_value<std::uint32_t>(-1.0)), std::uint32_t{0});
  CHECK_EQ(static_cast<int>(cast_value<std::int8_t>(300.0)), 44);
  CHECK_EQ(static_cast<int>(cast_value<std::uint8_t>(-2.7)), 254);
  CHECK_EQ((cast_value<std::int16_t>(-2.7)), std::int16_t{-2});
  CHECK_EQ((cast_value<bool>(-0.0)), false);
  CHECK_EQ((cast_value<bool>(NAN)), true);
  CHECK_EQ((cast_value<std::int8_t>(std::int64_t{200})), std::int8_t{-56});
}
