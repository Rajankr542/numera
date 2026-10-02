#include "p03_dtypes.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include "error.hpp"

namespace nativpy {

namespace {

template <typename T>
FloatInfo info_of(DType dt) {
  using L = std::numeric_limits<T>;
  FloatInfo f{};
  f.dtype = dt;
  f.bits = static_cast<int>(sizeof(T) * 8);
  f.nmant = L::digits - 1;
  f.nexp = f.bits - 1 - f.nmant;
  f.iexp = f.nexp;
  f.machep = -f.nmant;
  f.negep = -L::digits;
  f.minexp = L::min_exponent - 1;
  f.maxexp = L::max_exponent;
  f.precision = static_cast<int>(std::floor(f.nmant * std::log10(2.0)));
  f.eps = static_cast<double>(L::epsilon());
  f.epsneg = f.eps / 2;
  f.max = static_cast<double>(L::max());
  f.min = -f.max;
  f.tiny = static_cast<double>(L::min());
  f.smallest_subnormal = static_cast<double>(L::denorm_min());
  f.resolution = static_cast<double>(static_cast<T>(std::pow(10.0, -f.precision)));
  return f;
}

FloatInfo half_info() {
  FloatInfo f{};
  f.dtype = DType::Float16;
  f.bits = 16;
  f.nmant = 10;
  f.nexp = 5;
  f.iexp = 5;
  f.machep = -10;
  f.negep = -11;
  f.minexp = -14;
  f.maxexp = 16;
  f.precision = 3;
  f.eps = std::ldexp(1.0, -10);
  f.epsneg = std::ldexp(1.0, -11);
  f.max = 65504.0;
  f.min = -65504.0;
  f.tiny = std::ldexp(1.0, -14);
  f.smallest_subnormal = std::ldexp(1.0, -24);
  f.resolution = half_to_double(double_to_half(1e-3));
  return f;
}

// NumPy's float thresholds for value-based float/complex sizing.
DType min_float(double v) {
  if (!std::isfinite(v) || (v > -65000.0 && v < 65000.0)) return DType::Float16;
  if (v > -3.4e38 && v < 3.4e38) return DType::Float32;
  return DType::Float64;
}

bool fits_f32(double v) { return !std::isfinite(v) || (v > -3.4e38 && v < 3.4e38); }

DType min_unsigned(std::uint64_t v) {
  if (v <= 0xFFu) return DType::UInt8;
  if (v <= 0xFFFFu) return DType::UInt16;
  if (v <= 0xFFFFFFFFu) return DType::UInt32;
  return DType::UInt64;
}

DType min_signed(std::int64_t v) {
  if (v >= std::numeric_limits<std::int8_t>::min()) return DType::Int8;
  if (v >= std::numeric_limits<std::int16_t>::min()) return DType::Int16;
  if (v >= std::numeric_limits<std::int32_t>::min()) return DType::Int32;
  return DType::Int64;
}

}  // namespace

FloatInfo float_info(DType dt) {
  switch (dt) {
    case DType::Float16: return half_info();
    case DType::Float32:
    case DType::Complex64: return info_of<float>(DType::Float32);
    case DType::Float64:
    case DType::Complex128: return info_of<double>(DType::Float64);
    default:
      throw_error(ErrorKind::Value,
                  "data type dtype('" + std::string(dtype_name(dt)) + "') not compatible with finfo");
  }
}

IntInfo int_info(DType dt) {
  if (!is_integer(dt)) {
    throw_error(ErrorKind::Value, std::string("Invalid integer data type '") + dtype_info(dt).kind + "'.");
  }
  const int bits = static_cast<int>(itemsize(dt) * 8);
  if (dtype_info(dt).kind == 'u') {
    const std::uint64_t max = bits == 64 ? std::numeric_limits<std::uint64_t>::max() : (std::uint64_t{1} << bits) - 1;
    return {bits, 0, max};
  }
  const std::uint64_t max = (std::uint64_t{1} << (bits - 1)) - 1;
  return {bits, -static_cast<std::int64_t>(max) - 1, max};
}

DType min_scalar_type(const NDArray& a) {
  if (a.ndim() != 0) return a.dtype();
  const DType dt = a.dtype();
  const char kind = dtype_info(dt).kind;
  if (kind == 'b') return DType::Bool;
  if (kind == 'u') return min_unsigned(a.get_uint64(0));
  if (kind == 'i') {
    const std::int64_t v = a.get_int64(0);
    return v >= 0 ? min_unsigned(static_cast<std::uint64_t>(v)) : min_signed(v);
  }
  if (kind == 'f') {
    const DType m = min_float(a.get_double(0));
    return itemsize(m) < itemsize(dt) ? m : dt;
  }
  // complex: real and imaginary parts are the two doubles of the element
  const NDArray c = a.astype(DType::Complex128);
  double parts[2];
  std::memcpy(parts, c.data(), sizeof parts);
  if (dt == DType::Complex64 || (fits_f32(parts[0]) && fits_f32(parts[1]))) return DType::Complex64;
  return DType::Complex128;
}

}  // namespace nativpy
