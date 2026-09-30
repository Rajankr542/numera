#include "distributions.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "error.hpp"
#include "shape.hpp"
#include "ziggurat_tables.hpp"

namespace nativpy::random {

namespace {

std::int64_t count_of(const Shape& shape) {
  std::int64_t n = 1;
  for (const auto d : shape) {
    if (d < 0) throw_error(ErrorKind::Value, "negative dimensions are not allowed");
    n *= d;
  }
  return n;
}

std::uint64_t gen_mask(std::uint64_t max) {
  std::uint64_t mask = max;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;
  return mask;
}

// Buffered draws (NumPy buffered_uint8/16, buffered_bounded_bool).
struct Buf {
  std::uint32_t buf = 0;
  int bcnt = 0;
  std::uint16_t u16(BitGen& g) {
    if (!bcnt) {
      buf = g.next_uint32();
      bcnt = 1;
    } else {
      buf >>= 16;
      bcnt -= 1;
    }
    return static_cast<std::uint16_t>(buf);
  }
  std::uint8_t u8(BitGen& g) {
    if (!bcnt) {
      buf = g.next_uint32();
      bcnt = 3;
    } else {
      buf >>= 8;
      bcnt -= 1;
    }
    return static_cast<std::uint8_t>(buf);
  }
  bool bit(BitGen& g) {
    if (!bcnt) {
      buf = g.next_uint32();
      bcnt = 31;
    } else {
      buf >>= 1;
      bcnt -= 1;
    }
    return (buf & 1U) != 0;
  }
};

std::uint64_t lemire64(BitGen& g, std::uint64_t rng) {
  const std::uint64_t rng_excl = rng + 1;
  u128 m = static_cast<u128>(g.next_uint64()) * rng_excl;
  auto leftover = static_cast<std::uint64_t>(m);
  if (leftover < rng_excl) {
    const std::uint64_t threshold = (std::numeric_limits<std::uint64_t>::max() - rng) % rng_excl;
    while (leftover < threshold) {
      m = static_cast<u128>(g.next_uint64()) * rng_excl;
      leftover = static_cast<std::uint64_t>(m);
    }
  }
  return static_cast<std::uint64_t>(m >> 64);
}

std::uint32_t lemire32(BitGen& g, std::uint32_t rng) {
  const std::uint32_t rng_excl = rng + 1;
  std::uint64_t m = static_cast<std::uint64_t>(g.next_uint32()) * rng_excl;
  auto leftover = static_cast<std::uint32_t>(m);
  if (leftover < rng_excl) {
    const std::uint32_t threshold = (std::numeric_limits<std::uint32_t>::max() - rng) % rng_excl;
    while (leftover < threshold) {
      m = static_cast<std::uint64_t>(g.next_uint32()) * rng_excl;
      leftover = static_cast<std::uint32_t>(m);
    }
  }
  return static_cast<std::uint32_t>(m >> 32);
}

std::uint16_t lemire16(BitGen& g, std::uint16_t rng, Buf& b) {
  const auto rng_excl = static_cast<std::uint16_t>(rng + 1);
  std::uint32_t m = static_cast<std::uint32_t>(b.u16(g)) * rng_excl;
  auto leftover = static_cast<std::uint16_t>(m);
  if (leftover < rng_excl) {
    const auto threshold = static_cast<std::uint16_t>((0xFFFFU - rng) % rng_excl);
    while (leftover < threshold) {
      m = static_cast<std::uint32_t>(b.u16(g)) * rng_excl;
      leftover = static_cast<std::uint16_t>(m);
    }
  }
  return static_cast<std::uint16_t>(m >> 16);
}

std::uint8_t lemire8(BitGen& g, std::uint8_t rng, Buf& b) {
  const auto rng_excl = static_cast<std::uint8_t>(rng + 1);
  std::uint16_t m = static_cast<std::uint16_t>(b.u8(g) * rng_excl);
  auto leftover = static_cast<std::uint8_t>(m);
  if (leftover < rng_excl) {
    const auto threshold = static_cast<std::uint8_t>((0xFFU - rng) % rng_excl);
    while (leftover < threshold) {
      m = static_cast<std::uint16_t>(b.u8(g) * rng_excl);
      leftover = static_cast<std::uint8_t>(m);
    }
  }
  return static_cast<std::uint8_t>(m >> 8);
}

}  // namespace

float next_float(BitGen& g) {
  return static_cast<float>(g.next_uint32() >> 8) * (1.0F / 16777216.0F);
}

double standard_normal(BitGen& g) {
  using namespace zig;
  for (;;) {
    std::uint64_t r = g.next_uint64();
    const auto idx = static_cast<std::size_t>(r & 0xffU);
    r >>= 8;
    const bool sign = (r & 0x1U) != 0;
    const std::uint64_t rabs = (r >> 1) & 0x000fffffffffffffULL;
    double x = static_cast<double>(rabs) * wi_double[idx];
    if (sign) x = -x;
    if (rabs < ki_double[idx]) return x;
    if (idx == 0) {
      for (;;) {
        const double xx = -ziggurat_nor_inv_r * std::log1p(-g.next_double());
        const double yy = -std::log1p(-g.next_double());
        if (yy + yy > xx * xx) {
          return ((rabs >> 8) & 0x1U) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;
        }
      }
    } else {
      if (((fi_double[idx - 1] - fi_double[idx]) * g.next_double() + fi_double[idx]) <
          std::exp(-0.5 * x * x)) {
        return x;
      }
    }
  }
}

float standard_normal_f(BitGen& g) {
  using namespace zig;
  for (;;) {
    const std::uint32_t r = g.next_uint32();
    const auto idx = static_cast<std::size_t>(r & 0xffU);
    const bool sign = ((r >> 8) & 0x1U) != 0;
    const std::uint32_t rabs = (r >> 9) & 0x0007fffffU;
    float x = static_cast<float>(rabs) * wi_float[idx];
    if (sign) x = -x;
    if (rabs < ki_float[idx]) return x;
    if (idx == 0) {
      for (;;) {
        const float xx = -ziggurat_nor_inv_r_f * std::log1p(-next_float(g));
        const float yy = -std::log1p(-next_float(g));
        if (yy + yy > xx * xx) {
          return ((rabs >> 8) & 0x1U) ? -(ziggurat_nor_r_f + xx) : ziggurat_nor_r_f + xx;
        }
      }
    } else {
      // NumPy compares in double here (exp() of a float promoted to double).
      if (static_cast<double>((fi_float[idx - 1] - fi_float[idx]) * next_float(g) +
                              fi_float[idx]) < std::exp(-0.5 * static_cast<double>(x) * x)) {
        return x;
      }
    }
  }
}

std::uint64_t random_interval(BitGen& g, std::uint64_t max) {
  if (max == 0) return 0;
  const std::uint64_t mask = gen_mask(max);
  std::uint64_t value = 0;
  if (max <= 0xffffffffULL) {
    while ((value = (g.next_uint32() & mask)) > max) {
    }
  } else {
    while ((value = (g.next_uint64() & mask)) > max) {
    }
  }
  return value;
}

std::uint64_t bounded_uint64(BitGen& g, std::uint64_t off, std::uint64_t rng, bool masked) {
  if (rng == 0) return off;
  if (rng <= 0xFFFFFFFFULL) {
    if (rng == 0xFFFFFFFFULL) return off + g.next_uint32();
    if (masked) {
      const auto mask = static_cast<std::uint32_t>(gen_mask(rng));
      std::uint32_t v = 0;
      while ((v = (g.next_uint32() & mask)) > rng) {
      }
      return off + v;
    }
    return off + lemire32(g, static_cast<std::uint32_t>(rng));
  }
  if (rng == 0xFFFFFFFFFFFFFFFFULL) return off + g.next_uint64();
  if (masked) {
    const std::uint64_t mask = gen_mask(rng);
    std::uint64_t v = 0;
    while ((v = (g.next_uint64() & mask)) > rng) {
    }
    return off + v;
  }
  return off + lemire64(g, rng);
}

double LegacyGauss::next(BitGen& g) {
  if (has_gauss) {
    const double temp = gauss;
    has_gauss = false;
    gauss = 0.0;
    return temp;
  }
  double x1 = 0.0;
  double x2 = 0.0;
  double r2 = 0.0;
  do {
    x1 = 2.0 * g.next_double() - 1.0;
    x2 = 2.0 * g.next_double() - 1.0;
    r2 = x1 * x1 + x2 * x2;
  } while (r2 >= 1.0 || r2 == 0.0);
  const double f = std::sqrt(-2.0 * std::log(r2) / r2);
  gauss = f * x1;
  has_gauss = true;
  return f * x2;
}

namespace {

// Fill `cnt` values of unsigned width T: off + bounded(rng). Mirrors
// random_bounded_uint{8,16,32,64}_fill exactly (including buffering).
template <typename T>
void fill_bounded(BitGen& g, T off, T rng, std::int64_t cnt, bool masked, T* out) {
  constexpr T kMax = std::numeric_limits<T>::max();
  if (rng == 0) {
    for (std::int64_t i = 0; i < cnt; ++i) out[i] = off;
    return;
  }
  if constexpr (sizeof(T) == 8) {
    if (rng <= 0xFFFFFFFFULL) {
      std::vector<std::uint32_t> narrow(static_cast<std::size_t>(cnt));
      fill_bounded<std::uint32_t>(g, 0, static_cast<std::uint32_t>(rng), cnt, masked,
                                  narrow.data());
      for (std::int64_t i = 0; i < cnt; ++i) out[i] = off + narrow[static_cast<std::size_t>(i)];
      return;
    }
    if (rng == kMax) {
      for (std::int64_t i = 0; i < cnt; ++i) out[i] = off + g.next_uint64();
      return;
    }
    const std::uint64_t mask = gen_mask(rng);
    for (std::int64_t i = 0; i < cnt; ++i) {
      std::uint64_t v = 0;
      if (masked) {
        while ((v = (g.next_uint64() & mask)) > rng) {
        }
      } else {
        v = lemire64(g, rng);
      }
      out[i] = off + v;
    }
  } else {
    Buf b;
    const auto draw = [&]() -> T {
      if constexpr (sizeof(T) == 4) return g.next_uint32();
      if constexpr (sizeof(T) == 2) return b.u16(g);
      if constexpr (sizeof(T) == 1) return b.u8(g);
    };
    if (rng == kMax) {
      for (std::int64_t i = 0; i < cnt; ++i) out[i] = static_cast<T>(off + draw());
      return;
    }
    const auto mask = static_cast<T>(gen_mask(rng));
    for (std::int64_t i = 0; i < cnt; ++i) {
      T v = 0;
      if (masked) {
        while ((v = static_cast<T>(draw() & mask)) > rng) {
        }
      } else if constexpr (sizeof(T) == 4) {
        v = lemire32(g, rng);
      } else if constexpr (sizeof(T) == 2) {
        v = lemire16(g, rng, b);
      } else {
        v = lemire8(g, rng, b);
      }
      out[i] = static_cast<T>(off + v);
    }
  }
}

struct Bounds {
  i128 lb;
  i128 ub;
};

Bounds bounds_of(DType dt) {
  switch (dt) {
    case DType::Bool: return {0, 1};
    case DType::Int8: return {-128, 127};
    case DType::UInt8: return {0, 255};
    case DType::Int16: return {-32768, 32767};
    case DType::UInt16: return {0, 65535};
    case DType::Int32: return {-2147483648LL, 2147483647LL};
    case DType::UInt32: return {0, 4294967295LL};
    case DType::Int64:
      return {static_cast<i128>(std::numeric_limits<std::int64_t>::min()),
              static_cast<i128>(std::numeric_limits<std::int64_t>::max())};
    case DType::UInt64:
      return {0, static_cast<i128>(std::numeric_limits<std::uint64_t>::max())};
    default:
      throw_error(ErrorKind::DType,
                  "Unsupported dtype '" + std::string(dtype_name(dt)) + "' for integers");
  }
}

}  // namespace

NDArray bounded_integers(BitGen& g, i128 low, i128 high, bool closed, const Shape& shape,
                         DType dtype, bool masked) {
  const auto [lb, ub] = bounds_of(dtype);
  const std::string name(dtype_name(dtype));
  if (!closed) high -= 1;
  if (low < lb) throw_error(ErrorKind::Value, "low is out of bounds for " + name);
  if (high > ub) throw_error(ErrorKind::Value, "high is out of bounds for " + name);
  if (low > high) {
    if (low == 0) throw_error(ErrorKind::Value, closed ? "high < 0" : "high <= 0");
    throw_error(ErrorKind::Value, closed ? "low > high" : "low >= high");
  }
  const std::int64_t cnt = count_of(shape);
  NDArray out = NDArray::empty(shape, dtype);
  const auto rng = static_cast<std::uint64_t>(high - low);
  const auto off = static_cast<std::uint64_t>(low);  // two's complement wrap, as NumPy
  std::byte* data = out.data();
  switch (itemsize(dtype)) {
    case 1: {
      std::vector<std::uint8_t> v(static_cast<std::size_t>(cnt));
      if (dtype == DType::Bool) {
        Buf b;
        for (auto& x : v) x = rng == 0 ? static_cast<std::uint8_t>(off) : static_cast<std::uint8_t>(b.bit(g));
      } else {
        fill_bounded<std::uint8_t>(g, static_cast<std::uint8_t>(off), static_cast<std::uint8_t>(rng),
                                   cnt, masked, v.data());
      }
      std::memcpy(data, v.data(), v.size());
      break;
    }
    case 2: {
      std::vector<std::uint16_t> v(static_cast<std::size_t>(cnt));
      fill_bounded<std::uint16_t>(g, static_cast<std::uint16_t>(off),
                                  static_cast<std::uint16_t>(rng), cnt, masked, v.data());
      std::memcpy(data, v.data(), v.size() * 2);
      break;
    }
    case 4: {
      std::vector<std::uint32_t> v(static_cast<std::size_t>(cnt));
      fill_bounded<std::uint32_t>(g, static_cast<std::uint32_t>(off),
                                  static_cast<std::uint32_t>(rng), cnt, masked, v.data());
      std::memcpy(data, v.data(), v.size() * 4);
      break;
    }
    default: {
      std::vector<std::uint64_t> v(static_cast<std::size_t>(cnt));
      fill_bounded<std::uint64_t>(g, off, rng, cnt, masked, v.data());
      std::memcpy(data, v.data(), v.size() * 8);
      break;
    }
  }
  return out;
}

namespace {

template <typename T, typename Fn>
NDArray fill_with(const Shape& shape, DType dtype, Fn&& fn) {
  const std::int64_t cnt = count_of(shape);
  NDArray out = NDArray::empty(shape, dtype);
  std::vector<T> v(static_cast<std::size_t>(cnt));
  for (auto& x : v) x = fn();
  if (cnt > 0) std::memcpy(out.data(), v.data(), v.size() * sizeof(T));
  return out;
}

void check_dtype_float(DType dtype, const char* fn) {
  if (dtype != DType::Float64 && dtype != DType::Float32) {
    throw_error(ErrorKind::DType, std::string("Unsupported dtype \"") + std::string(dtype_name(dtype)) +
                                      "\" for " + fn);
  }
}

}  // namespace

NDArray random_doubles(BitGen& g, const Shape& shape, DType dtype) {
  check_dtype_float(dtype, "random");
  if (dtype == DType::Float32) return fill_with<float>(shape, dtype, [&] { return next_float(g); });
  return fill_with<double>(shape, dtype, [&] { return g.next_double(); });
}

NDArray uniform(BitGen& g, double low, double high, const Shape& shape) {
  const double range = high - low;
  return fill_with<double>(shape, DType::Float64, [&] { return low + range * g.next_double(); });
}

NDArray normal(BitGen& g, double loc, double scale, const Shape& shape, DType dtype) {
  check_dtype_float(dtype, "standard_normal");
  if (dtype == DType::Float32) {
    // NumPy exposes float32 only via standard_normal (Generator.normal has no
    // dtype); refuse rather than silently dropping loc/scale.
    if (loc != 0.0 || scale != 1.0) {
      throw_error(ErrorKind::Value, "float32 normal supports only loc=0, scale=1");
    }
    return fill_with<float>(shape, dtype, [&] { return standard_normal_f(g); });
  }
  return fill_with<double>(shape, dtype, [&] { return loc + scale * standard_normal(g); });
}

NDArray legacy_normal(BitGen& g, LegacyGauss& gs, double loc, double scale, const Shape& shape) {
  return fill_with<double>(shape, DType::Float64, [&] { return loc + scale * gs.next(g); });
}

void shuffle(BitGen& g, const NDArray& a) {
  a.check_writeable();
  if (a.ndim() == 0) throw_error(ErrorKind::Value, "shuffle requires an array with ndim >= 1");
  const std::int64_t n = a.shape()[0];
  if (n <= 1 || a.size() == 0) {
    // NumPy still consumes nothing for n <= 1 (loop is empty).
    return;
  }
  const std::int64_t s0 = a.strides()[0];
  const std::size_t isz = a.itemsize();
  const Shape sub_shape(a.shape().begin() + 1, a.shape().end());
  const Strides sub_strides(a.strides().begin() + 1, a.strides().end());
  std::byte tmp[16];
  for (std::int64_t i = n - 1; i >= 1; --i) {
    const auto j = static_cast<std::int64_t>(random_interval(g, static_cast<std::uint64_t>(i)));
    if (i == j) continue;
    const NDArray row = a.view(sub_shape, sub_strides, a.offset() + i * s0);
    const std::int64_t delta = (j - i) * s0;
    for_each_element(row, [&](std::byte* p) {
      std::memcpy(tmp, p, isz);
      std::memcpy(p, p + delta, isz);
      std::memcpy(p + delta, tmp, isz);
    });
  }
}

NDArray generator_choice_indices(BitGen& g, std::int64_t pop_size, const Shape& shape,
                                 bool replace, bool shuffle_result) {
  const std::int64_t size = count_of(shape);
  if (replace) {
    return bounded_integers(g, 0, pop_size, false, shape, DType::Int64, false);
  }
  if (size > pop_size) {
    throw_error(ErrorKind::Value,
                "Cannot take a larger sample than population when replace is False");
  }
  NDArray out = NDArray::empty(shape, DType::Int64);
  std::vector<std::int64_t> idx;
  const std::int64_t cutoff = shuffle_result ? 50 : 20;
  if (pop_size > 10000 && size > pop_size / cutoff) {
    // _shuffle_int(pop, max(pop - size, 1)) on arange, keep the tail.
    std::vector<std::int64_t> all(static_cast<std::size_t>(pop_size));
    for (std::int64_t i = 0; i < pop_size; ++i) all[static_cast<std::size_t>(i)] = i;
    const std::int64_t first = std::max<std::int64_t>(pop_size - size, 1);
    for (std::int64_t i = pop_size - 1; i >= first; --i) {
      const auto j = static_cast<std::size_t>(bounded_uint64(g, 0, static_cast<std::uint64_t>(i), false));
      std::swap(all[j], all[static_cast<std::size_t>(i)]);
    }
    idx.assign(all.end() - size, all.end());
  } else {
    // Floyd's algorithm with an open-addressing hash set (NumPy).
    idx.resize(static_cast<std::size_t>(size));
    const auto set_size0 = static_cast<std::uint64_t>(1.2 * static_cast<double>(size));
    const std::uint64_t mask = gen_mask(set_size0);
    std::vector<std::uint64_t> hash_set(static_cast<std::size_t>(mask + 1), ~0ULL);
    for (std::int64_t j = pop_size - size; j < pop_size; ++j) {
      const std::uint64_t val = bounded_uint64(g, 0, static_cast<std::uint64_t>(j), false);
      std::uint64_t loc = val & mask;
      while (hash_set[loc] != ~0ULL && hash_set[loc] != val) loc = (loc + 1) & mask;
      const auto pos = static_cast<std::size_t>(j - pop_size + size);
      if (hash_set[loc] == ~0ULL) {
        hash_set[loc] = val;
        idx[pos] = static_cast<std::int64_t>(val);
      } else {
        loc = static_cast<std::uint64_t>(j) & mask;
        while (hash_set[loc] != ~0ULL) loc = (loc + 1) & mask;
        hash_set[loc] = static_cast<std::uint64_t>(j);
        idx[pos] = j;
      }
    }
    if (shuffle_result) {
      for (std::int64_t i = size - 1; i >= 1; --i) {
        const auto j = static_cast<std::size_t>(bounded_uint64(g, 0, static_cast<std::uint64_t>(i), false));
        std::swap(idx[j], idx[static_cast<std::size_t>(i)]);
      }
    }
  }
  if (size > 0) std::memcpy(out.data(), idx.data(), idx.size() * sizeof(std::int64_t));
  return out;
}

}  // namespace nativpy::random
