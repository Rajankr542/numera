#include "creation.hpp"

#include <algorithm>
#include <cmath>
#include <string>

#include "cast.hpp"
#include "error.hpp"

namespace nativpy {

namespace {
// Fills a freshly allocated contiguous array with one value of type T.
template <typename T>
void fill_contiguous(NDArray& out, T value) {
  auto* p = reinterpret_cast<T*>(out.data());
  std::fill(p, p + out.size(), value);
}
}  // namespace

NDArray full(const Shape& shape, const NDArray& value) {
  if (value.size() != 1) throw_error(ErrorKind::Value, "fill value must have exactly one element");
  NDArray out = NDArray::empty(shape, value.dtype());
  if (out.size() == 0) return out;
  dispatch_dtype(value.dtype(), [&](auto tag) {
    using T = dtype_t<decltype(tag)::value>;
    fill_contiguous<T>(out, load<T>(value.data()));
  });
  return out;
}

NDArray ones(const Shape& shape, DType dtype) {
  NDArray one = NDArray::empty({}, dtype);
  one.set_double(0, 1.0);
  return full(shape, one);
}

NDArray arange(double start, double stop, double step, DType dtype) {
  if (step == 0.0) throw_error(ErrorKind::Value, "arange: step must not be zero");
  const double len_d = std::ceil((stop - start) / step);
  if (!std::isfinite(len_d)) throw_error(ErrorKind::Value, "arange: cannot compute length");
  if (len_d > 9007199254740991.0) throw_error(ErrorKind::Memory, "arange: array is too big");
  const auto len = len_d > 0 ? static_cast<std::int64_t>(len_d) : std::int64_t{0};
  if (dtype == DType::Bool && len > 2) {
    throw_error(ErrorKind::Value,
                "arange() is only supported for booleans when the result has at most length 2");
  }
  NDArray out = NDArray::empty({len}, dtype);
  if (len == 0) return out;
  dispatch_dtype(dtype, [&](auto tag) {
    using T = dtype_t<decltype(tag)::value>;
    auto* p = reinterpret_cast<T*>(out.data());
    p[0] = cast_value<T>(start);
    if (len == 1) return;
    p[1] = cast_value<T>(start + step);
    // NumPy <TYPE>_fill: start + i * (buf[1] - buf[0]) in the element type
    // (float16 computes in float32).
    if constexpr (std::is_same_v<T, float16_t>) {
      const auto s = static_cast<float>(half_to_double(p[0]));
      const float d = static_cast<float>(half_to_double(p[1])) - s;
      for (std::int64_t i = 2; i < len; ++i) {
        p[i] = double_to_half(static_cast<double>(s + static_cast<float>(i) * d));
      }
    } else if constexpr (!std::is_same_v<T, bool>) {
      const T s = p[0];
      const T d = static_cast<T>(p[1] - s);
      for (std::int64_t i = 2; i < len; ++i) {
        if constexpr (is_complex_v<T>) {
          p[i] = s + static_cast<typename T::value_type>(i) * d;
        } else if constexpr (std::is_integral_v<T>) {
          // uint64 arithmetic gives C-style wraparound without signed UB
          // (small types would otherwise promote to int and overflow).
          const auto su = static_cast<std::uint64_t>(s);
          const auto du = static_cast<std::uint64_t>(d);
          p[i] = static_cast<T>(su + static_cast<std::uint64_t>(i) * du);
        } else {
          p[i] = s + static_cast<T>(i) * d;
        }
      }
    }
  });
  return out;
}

NDArray linspace(double start, double stop, std::int64_t num, bool endpoint, DType dtype) {
  if (num < 0) {
    throw_error(ErrorKind::Value,
                "Number of samples, " + std::to_string(num) + ", must be non-negative.");
  }
  NDArray y = NDArray::empty({num}, DType::Float64);
  auto* p = reinterpret_cast<double*>(y.data());
  const std::int64_t div = endpoint ? num - 1 : num;
  const double delta = stop - start;
  if (div > 0) {
    const double step = delta / static_cast<double>(div);
    for (std::int64_t i = 0; i < num; ++i) {
      const auto fi = static_cast<double>(i);
      // NumPy: if step == 0, compute (i / div) * delta instead of i * step.
      p[i] = (step == 0.0 ? (fi / static_cast<double>(div)) * delta : fi * step) + start;
    }
  } else {
    for (std::int64_t i = 0; i < num; ++i) p[i] = static_cast<double>(i) * delta + start;
  }
  if (endpoint && num > 1) p[num - 1] = stop;
  if (is_integer(dtype)) {
    for (std::int64_t i = 0; i < num; ++i) p[i] = std::floor(p[i]);
  }
  return dtype == DType::Float64 ? y : y.astype(dtype);
}

NDArray eye(std::int64_t n, std::int64_t m, std::int64_t k, DType dtype) {
  NDArray out = NDArray::zeros({n, m}, dtype);
  if (out.size() == 0) return out;
  NDArray one = ones({}, dtype);
  const std::size_t isz = out.itemsize();
  for (std::int64_t i = 0; i < n; ++i) {
    const std::int64_t j = i + k;
    if (j < 0 || j >= m) continue;
    std::memcpy(out.data() + static_cast<std::size_t>(i * m + j) * isz, one.data(), isz);
  }
  return out;
}

}  // namespace nativpy
