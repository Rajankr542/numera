#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <type_traits>

#include "error.hpp"

namespace nativpy {

enum class DType : std::uint8_t {
  Bool,
  Int8,
  UInt8,
  Int16,
  UInt16,
  Int32,
  UInt32,
  Int64,
  UInt64,
  Float16,
  Float32,
  Float64,
  Complex64,
  Complex128,
};

inline constexpr int kNumDTypes = 14;

// NumPy "kind" characters: b, i, u, f, c.
struct DTypeInfo {
  std::string_view name;
  std::size_t itemsize;
  std::size_t alignment;
  char kind;
};

const DTypeInfo& dtype_info(DType dt) noexcept;
inline std::size_t itemsize(DType dt) noexcept { return dtype_info(dt).itemsize; }
inline std::string_view dtype_name(DType dt) noexcept { return dtype_info(dt).name; }
std::optional<DType> dtype_from_name(std::string_view name) noexcept;

inline bool is_integer(DType dt) noexcept {
  const char k = dtype_info(dt).kind;
  return k == 'i' || k == 'u';
}
inline bool is_complex(DType dt) noexcept { return dtype_info(dt).kind == 'c'; }

// Equivalent of numpy.promote_types for the supported dtypes.
DType promote_types(DType a, DType b) noexcept;

// NumPy casting rules (D-045). All dtypes are native byte order, so No and
// Equiv behave the same.
enum class Casting : std::uint8_t { No, Equiv, Safe, SameKind, Unsafe };

std::optional<Casting> casting_from_name(std::string_view name) noexcept;
std::string_view casting_name(Casting c) noexcept;

// Equivalent of numpy.can_cast(from, to, casting) for dtype arguments.
bool can_cast(DType from, DType to, Casting casting) noexcept;

// IEEE-754 binary16 storage type (bit pattern), conversions round-to-nearest-even.
struct float16_t {
  std::uint16_t bits;
};
float16_t double_to_half(double value) noexcept;
double half_to_double(float16_t value) noexcept;

// Maps DType -> C++ storage type.
template <DType D> struct dtype_traits;
#define NATIVPY_DTYPE_TRAIT(D, T) \
  template <> struct dtype_traits<DType::D> { using type = T; };
NATIVPY_DTYPE_TRAIT(Bool, bool)
NATIVPY_DTYPE_TRAIT(Int8, std::int8_t)
NATIVPY_DTYPE_TRAIT(UInt8, std::uint8_t)
NATIVPY_DTYPE_TRAIT(Int16, std::int16_t)
NATIVPY_DTYPE_TRAIT(UInt16, std::uint16_t)
NATIVPY_DTYPE_TRAIT(Int32, std::int32_t)
NATIVPY_DTYPE_TRAIT(UInt32, std::uint32_t)
NATIVPY_DTYPE_TRAIT(Int64, std::int64_t)
NATIVPY_DTYPE_TRAIT(UInt64, std::uint64_t)
NATIVPY_DTYPE_TRAIT(Float16, float16_t)
NATIVPY_DTYPE_TRAIT(Float32, float)
NATIVPY_DTYPE_TRAIT(Float64, double)
NATIVPY_DTYPE_TRAIT(Complex64, std::complex<float>)
NATIVPY_DTYPE_TRAIT(Complex128, std::complex<double>)
#undef NATIVPY_DTYPE_TRAIT

template <DType D> using dtype_t = typename dtype_traits<D>::type;

// Calls fn(std::integral_constant<DType, D>{}) for the runtime dtype.
template <typename Fn>
decltype(auto) dispatch_dtype(DType dt, Fn&& fn) {
  switch (dt) {
#define NATIVPY_CASE(D) \
  case DType::D:        \
    return fn(std::integral_constant<DType, DType::D>{});
    NATIVPY_CASE(Bool)
    NATIVPY_CASE(Int8)
    NATIVPY_CASE(UInt8)
    NATIVPY_CASE(Int16)
    NATIVPY_CASE(UInt16)
    NATIVPY_CASE(Int32)
    NATIVPY_CASE(UInt32)
    NATIVPY_CASE(Int64)
    NATIVPY_CASE(UInt64)
    NATIVPY_CASE(Float16)
    NATIVPY_CASE(Float32)
    NATIVPY_CASE(Float64)
    NATIVPY_CASE(Complex64)
    NATIVPY_CASE(Complex128)
#undef NATIVPY_CASE
  }
  throw_error(ErrorKind::DType, "invalid dtype");
}

}  // namespace nativpy
