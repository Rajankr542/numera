#include "dtype.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>

namespace nativpy {

namespace {

constexpr std::array<DTypeInfo, kNumDTypes> kInfo{{
    {"bool", 1, 1, 'b'},
    {"int8", 1, 1, 'i'},
    {"uint8", 1, 1, 'u'},
    {"int16", 2, 2, 'i'},
    {"uint16", 2, 2, 'u'},
    {"int32", 4, 4, 'i'},
    {"uint32", 4, 4, 'u'},
    {"int64", 8, 8, 'i'},
    {"uint64", 8, 8, 'u'},
    {"float16", 2, 2, 'f'},
    {"float32", 4, 4, 'f'},
    {"float64", 8, 8, 'f'},
    {"complex64", 8, 4, 'c'},
    {"complex128", 16, 8, 'c'},
}};

// Generated from numpy.promote_types (NumPy 2.x); index order == DType order.
constexpr std::uint8_t kPromote[kNumDTypes][kNumDTypes] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
    {1, 1, 3, 3, 5, 5, 7, 7, 11, 9, 10, 11, 12, 13},
    {2, 3, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13},
    {3, 3, 3, 3, 5, 5, 7, 7, 11, 10, 10, 11, 12, 13},
    {4, 5, 4, 5, 4, 5, 6, 7, 8, 10, 10, 11, 12, 13},
    {5, 5, 5, 5, 5, 5, 7, 7, 11, 11, 11, 11, 13, 13},
    {6, 7, 6, 7, 6, 7, 6, 7, 8, 11, 11, 11, 13, 13},
    {7, 7, 7, 7, 7, 7, 7, 7, 11, 11, 11, 11, 13, 13},
    {8, 11, 8, 11, 8, 11, 8, 11, 8, 11, 11, 11, 13, 13},
    {9, 9, 9, 10, 10, 11, 11, 11, 11, 9, 10, 11, 12, 13},
    {10, 10, 10, 10, 10, 11, 11, 11, 11, 10, 10, 11, 12, 13},
    {11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 13, 13},
    {12, 12, 12, 12, 12, 13, 13, 13, 13, 12, 12, 13, 12, 13},
    {13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13},
};

}  // namespace

const DTypeInfo& dtype_info(DType dt) noexcept {
  return kInfo[static_cast<std::size_t>(dt)];
}

std::optional<DType> dtype_from_name(std::string_view name) noexcept {
  for (int i = 0; i < kNumDTypes; ++i) {
    if (kInfo[static_cast<std::size_t>(i)].name == name) {
      return static_cast<DType>(i);
    }
  }
  return std::nullopt;
}

DType promote_types(DType a, DType b) noexcept {
  return static_cast<DType>(
      kPromote[static_cast<std::size_t>(a)][static_cast<std::size_t>(b)]);
}

namespace {
// Kind order used by same_kind casting: b < u < i < f < c.
int kind_rank(DType dt) noexcept {
  switch (dtype_info(dt).kind) {
    case 'b': return 0;
    case 'u': return 1;
    case 'i': return 2;
    case 'f': return 3;
    default: return 4;  // 'c'
  }
}
}  // namespace

std::optional<Casting> casting_from_name(std::string_view name) noexcept {
  if (name == "no") return Casting::No;
  if (name == "equiv") return Casting::Equiv;
  if (name == "safe") return Casting::Safe;
  if (name == "same_kind") return Casting::SameKind;
  if (name == "unsafe") return Casting::Unsafe;
  return std::nullopt;
}

std::string_view casting_name(Casting c) noexcept {
  switch (c) {
    case Casting::No: return "no";
    case Casting::Equiv: return "equiv";
    case Casting::Safe: return "safe";
    case Casting::SameKind: return "same_kind";
    case Casting::Unsafe: return "unsafe";
  }
  return "?";
}

bool can_cast(DType from, DType to, Casting casting) noexcept {
  switch (casting) {
    case Casting::No:
    case Casting::Equiv: return from == to;
    case Casting::Unsafe: return true;
    case Casting::Safe: return promote_types(from, to) == to;
    case Casting::SameKind:
      return promote_types(from, to) == to || kind_rank(from) <= kind_rank(to);
  }
  return false;
}

// Conversion via float32 intermediate would double-round; convert directly
// from the double bit pattern with round-to-nearest-even.
float16_t double_to_half(double value) noexcept {
  const auto bits = std::bit_cast<std::uint64_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 48) & 0x8000U);
  const auto exp = static_cast<int>((bits >> 52) & 0x7FFU);
  std::uint64_t mant = bits & 0xFFFFFFFFFFFFFULL;

  if (exp == 0x7FF) {  // Inf / NaN
    if (mant == 0) return {static_cast<std::uint16_t>(sign | 0x7C00U)};
    // Preserve top mantissa bits, force quiet NaN.
    return {static_cast<std::uint16_t>(sign | 0x7E00U | (mant >> 42))};
  }
  const int e = exp - 1023 + 15;  // rebias
  if (e >= 0x1F) {                // overflow -> inf
    return {static_cast<std::uint16_t>(sign | 0x7C00U)};
  }
  if (e <= 0) {  // subnormal half or zero
    if (e < -10) return {sign};  // underflows to (signed) zero
    mant |= 1ULL << 52;          // implicit leading bit
    const int shift = 42 + 1 - e;  // bits to discard
    const std::uint64_t half_mant = mant >> shift;
    const std::uint64_t rem = mant & ((1ULL << shift) - 1);
    const std::uint64_t halfway = 1ULL << (shift - 1);
    std::uint64_t result = half_mant;
    if (rem > halfway || (rem == halfway && (half_mant & 1U))) ++result;
    return {static_cast<std::uint16_t>(sign | result)};
  }
  std::uint64_t half_mant = mant >> 42;
  const std::uint64_t rem = mant & ((1ULL << 42) - 1);
  const std::uint64_t halfway = 1ULL << 41;
  std::uint32_t result =
      (static_cast<std::uint32_t>(e) << 10) | static_cast<std::uint32_t>(half_mant);
  if (rem > halfway || (rem == halfway && (half_mant & 1U))) {
    ++result;  // may carry into exponent (and up to inf) correctly
  }
  return {static_cast<std::uint16_t>(sign | result)};
}

double half_to_double(float16_t value) noexcept {
  const std::uint16_t h = value.bits;
  const bool neg = (h & 0x8000U) != 0;
  const int exp = (h >> 10) & 0x1F;
  const int mant = h & 0x3FF;
  double result = 0.0;
  if (exp == 0) {
    result = std::ldexp(static_cast<double>(mant), -24);
  } else if (exp == 0x1F) {
    result = mant == 0 ? INFINITY : NAN;
  } else {
    result = std::ldexp(static_cast<double>(mant | 0x400), exp - 25);
  }
  return neg ? -result : result;
}

}  // namespace nativpy
