#include "p07_creation.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>

#include "broadcast.hpp"
#include "cast.hpp"
#include "complex_kernels.hpp"
#include "creation.hpp"
#include "error.hpp"
#include "ufunc.hpp"

namespace nativpy::p07 {

namespace {

using cdouble = std::complex<double>;

void check_dims(const Shape& shape) {
  for (const auto d : shape) {
    if (d < 0) throw_error(ErrorKind::Value, "negative dimensions are not allowed");
  }
}

// NumPy linspace for complex128 start/stop (arange(0, num) * step + start).
NDArray linspace_complex(cdouble start, cdouble stop, std::int64_t num, bool endpoint) {
  if (num < 0) {
    throw_error(ErrorKind::Value,
                "Number of samples, " + std::to_string(num) + ", must be non-negative.");
  }
  NDArray y = NDArray::empty({num}, DType::Complex128);
  auto* p = reinterpret_cast<cdouble*>(y.data());
  const std::int64_t div = endpoint ? num - 1 : num;
  const cdouble delta = kernels::csub(stop, start);
  if (div > 0) {
    const cdouble fdiv{static_cast<double>(div), 0.0};
    const cdouble step = kernels::cdiv(delta, fdiv);
    const bool step_zero = step.real() == 0.0 && step.imag() == 0.0;
    for (std::int64_t i = 0; i < num; ++i) {
      const cdouble fi{static_cast<double>(i), 0.0};
      const cdouble scaled =
          step_zero ? kernels::cmul(kernels::cdiv(fi, fdiv), delta) : kernels::cmul(fi, step);
      p[i] = kernels::cadd(scaled, start);
    }
  } else {
    for (std::int64_t i = 0; i < num; ++i) {
      const cdouble fi{static_cast<double>(i), 0.0};
      p[i] = kernels::cadd(kernels::cmul(fi, delta), start);
    }
  }
  if (endpoint && num > 1) p[num - 1] = stop;
  return y;
}

NDArray scalar_array(cdouble v, DType dt) {
  NDArray a = NDArray::empty({}, dt);
  if (dt == DType::Complex128) {
    store<cdouble>(a.data(), v);
  } else {
    store<double>(a.data(), v.real());
  }
  return a;
}

// base ** y with NumPy's power ufunc (y is float64 or complex128).
NDArray power_of(double base, const NDArray& y) {
  return binary(BinaryOp::Power, scalar_array({base, 0.0}, y.dtype()), y);
}

// NumPy np.sign for complex128 (z / |z| with the inf/nan special cases).
cdouble complex_sign(cdouble z) {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double r = z.real(), i = z.imag();
  const double a = std::hypot(r, i);
  if (std::isnan(a)) return {nan, nan};
  if (std::isinf(a)) {
    if (std::isinf(r)) return std::isinf(i) ? cdouble{nan, nan} : cdouble{std::copysign(1.0, r), 0.0};
    return {0.0, std::copysign(1.0, i)};
  }
  if (a == 0.0) return {0.0, 0.0};
  return {r / a, i / a};
}

double real_sign(double x) {
  if (std::isnan(x)) return x;
  return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0);
}

constexpr double kLog10E = 0.434294481903251827651128918916605082;

}  // namespace

NDArray logspace(cdouble start, cdouble stop, bool is_complex, std::int64_t num, bool endpoint,
                 double base, std::optional<DType> dtype) {
  NDArray y = is_complex ? linspace_complex(start, stop, num, endpoint)
                         : linspace(start.real(), stop.real(), num, endpoint, DType::Float64);
  NDArray r = power_of(base, y);
  return dtype && *dtype != r.dtype() ? r.astype(*dtype) : r;
}

NDArray geomspace(cdouble start, cdouble stop, std::int64_t num, bool endpoint, bool cplx,
                  DType dtype) {
  if ((start.real() == 0.0 && start.imag() == 0.0) || (stop.real() == 0.0 && stop.imag() == 0.0)) {
    throw_error(ErrorKind::Value, "Geometric sequence cannot include zero");
  }
  NDArray result = NDArray::empty({0}, DType::Float64);
  if (cplx) {
    const cdouble sign = complex_sign(start);
    const cdouble s = kernels::cdiv(start, sign);
    const cdouble e = kernels::cdiv(stop, sign);
    const cdouble ls = kernels::clog(s), le = kernels::clog(e);
    NDArray y = linspace_complex({ls.real() * kLog10E, ls.imag() * kLog10E},
                                 {le.real() * kLog10E, le.imag() * kLog10E}, num, endpoint);
    result = power_of(10.0, y);
    auto* p = reinterpret_cast<cdouble*>(result.data());
    if (num > 0) {
      p[0] = s;
      if (num > 1 && endpoint) p[num - 1] = e;
    }
    for (std::int64_t i = 0; i < num; ++i) p[i] = kernels::cmul(p[i], sign);
  } else {
    const double sign = real_sign(start.real());
    const double s = start.real() / sign;
    const double e = stop.real() / sign;
    NDArray y = linspace(std::log10(s), std::log10(e), num, endpoint, DType::Float64);
    result = power_of(10.0, y);
    auto* p = reinterpret_cast<double*>(result.data());
    if (num > 0) {
      p[0] = s;
      if (num > 1 && endpoint) p[num - 1] = e;
    }
    for (std::int64_t i = 0; i < num; ++i) p[i] *= sign;
  }
  return dtype != result.dtype() ? result.astype(dtype) : result;
}

NDArray tri(std::int64_t n, std::int64_t m, std::int64_t k, DType dtype) {
  // NumPy builds tri from arange(N) / arange(M), so negative sizes give 0.
  n = std::max<std::int64_t>(n, 0);
  m = std::max<std::int64_t>(m, 0);
  NDArray out = NDArray::zeros({n, m}, dtype);
  if (out.size() == 0) return out;
  NDArray one = ones({}, dtype);
  const std::size_t isz = out.itemsize();
  for (std::int64_t i = 0; i < n; ++i) {
    // j <= i + k
    const std::int64_t last = std::min(m - 1, i + k);
    for (std::int64_t j = 0; j <= last; ++j) {
      std::memcpy(out.data() + static_cast<std::size_t>(i * m + j) * isz, one.data(), isz);
    }
  }
  return out;
}

namespace {

// Copies `a` (1-D input broadcast to (N, N)) and zeroes the elements outside
// the kept triangle: lower keeps j <= i + k, upper keeps j >= i + k.
NDArray triangle(const NDArray& a, std::int64_t k, bool upper) {
  if (a.ndim() == 0) throw_error(ErrorKind::DType, "tril/triu: input must be at least 1-d");
  Shape shape = a.shape();
  if (a.ndim() == 1) shape = {shape[0], shape[0]};
  NDArray out = NDArray::empty(shape, a.dtype());
  copy_into(out, a);
  if (out.size() == 0) return out;
  const std::size_t nd = shape.size();
  const std::int64_t rows = shape[nd - 2], cols = shape[nd - 1];
  const std::int64_t mats = out.size() / (rows * cols);
  const std::size_t isz = out.itemsize();
  for (std::int64_t b = 0; b < mats; ++b) {
    std::byte* base = out.data() + static_cast<std::size_t>(b * rows * cols) * isz;
    for (std::int64_t i = 0; i < rows; ++i) {
      std::byte* row = base + static_cast<std::size_t>(i * cols) * isz;
      // Zero range [lo, hi) of columns.
      std::int64_t lo = 0, hi = 0;
      if (upper) {
        hi = std::clamp<std::int64_t>(i + k, 0, cols);
      } else {
        lo = std::clamp<std::int64_t>(i + k + 1, 0, cols);
        hi = cols;
      }
      if (hi > lo) {
        std::memset(row + static_cast<std::size_t>(lo) * isz, 0, static_cast<std::size_t>(hi - lo) * isz);
      }
    }
  }
  return out;
}

}  // namespace

NDArray tril(const NDArray& a, std::int64_t k) { return triangle(a, k, false); }
NDArray triu(const NDArray& a, std::int64_t k) { return triangle(a, k, true); }

NDArray diag(const NDArray& v, std::int64_t k) {
  if (v.ndim() == 1) {
    const std::int64_t len = v.shape()[0];
    const std::int64_t n = len + (k < 0 ? -k : k);
    NDArray out = NDArray::zeros({n, n}, v.dtype());
    if (len == 0) return out;
    const std::int64_t row0 = k >= 0 ? 0 : -k, col0 = k >= 0 ? k : 0;
    NDArray dst = out.view({len}, {static_cast<std::int64_t>(out.itemsize()) * (n + 1)},
                           out.offset() + static_cast<std::int64_t>(out.itemsize()) * (row0 * n + col0));
    copy_into(dst, v);
    return out;
  }
  if (v.ndim() == 2) {
    const std::int64_t rows = v.shape()[0], cols = v.shape()[1];
    const std::int64_t len =
        k >= 0 ? std::min(rows, cols - k) : std::min(rows + k, cols);
    if (len <= 0) return NDArray::empty({0}, v.dtype()).as_readonly();
    const std::int64_t off = v.offset() + (k >= 0 ? k * v.strides()[1] : -k * v.strides()[0]);
    return v.view({len}, {v.strides()[0] + v.strides()[1]}, off).as_readonly();
  }
  throw_error(ErrorKind::Value, "Input must be 1- or 2-d.");
}

NDArray vander(const NDArray& x, std::optional<std::int64_t> n_opt, bool increasing) {
  if (x.ndim() != 1) throw_error(ErrorKind::Value, "x must be a one-dimensional array or sequence.");
  const std::int64_t len = x.shape()[0];
  const std::int64_t n = n_opt.value_or(len);
  check_dims({len, n});
  const DType dt = promote_types(x.dtype(), DType::Int64);
  NDArray xs = x.astype(dt);
  NDArray out = NDArray::empty({len, n}, dt);
  if (out.size() == 0) return out;
  dispatch_dtype(dt, [&](auto tag) {
    using T = dtype_t<decltype(tag)::value>;
    if constexpr (std::is_same_v<T, std::int64_t> || std::is_same_v<T, double> ||
                  std::is_same_v<T, cdouble>) {
      const auto* xp = reinterpret_cast<const T*>(xs.data());
      auto* o = reinterpret_cast<T*>(out.data());
      for (std::int64_t i = 0; i < len; ++i) {
        T* row = o + i * n;
        const T v = xp[i];
        T acc = T{1};
        for (std::int64_t j = 0; j < n; ++j) {
          if (j > 0) {
            if constexpr (std::is_same_v<T, cdouble>) {
              acc = j == 1 ? v : kernels::cmul(acc, v);
            } else if constexpr (std::is_same_v<T, std::int64_t>) {
              acc = j == 1 ? v
                           : static_cast<T>(static_cast<std::uint64_t>(acc) * static_cast<std::uint64_t>(v));
            } else {
              acc = j == 1 ? v : acc * v;
            }
          }
          row[increasing ? j : n - 1 - j] = acc;
        }
      }
    } else {
      throw_error(ErrorKind::DType, "vander: unsupported dtype");
    }
  });
  return out;
}

NDArray grid_axis(double start, double step, std::int64_t n, DType dtype) {
  check_dims({n});
  NDArray out = NDArray::empty({n}, dtype);
  if (dtype == DType::Int64) {
    auto* p = reinterpret_cast<std::int64_t*>(out.data());
    const auto s = static_cast<std::int64_t>(start), d = static_cast<std::int64_t>(step);
    for (std::int64_t i = 0; i < n; ++i) p[i] = i * d + s;
  } else if (dtype == DType::Float64) {
    auto* p = reinterpret_cast<double*>(out.data());
    for (std::int64_t i = 0; i < n; ++i) p[i] = static_cast<double>(i) * step + start;
  } else {
    throw_error(ErrorKind::DType, "grid: dtype must be int64 or float64");
  }
  return out;
}

NDArray mgrid(const std::vector<double>& starts, const std::vector<double>& steps,
              const Shape& sizes, DType dtype) {
  check_dims(sizes);
  const auto nd = static_cast<std::int64_t>(sizes.size());
  Shape shape{nd};
  shape.insert(shape.end(), sizes.begin(), sizes.end());
  NDArray out = NDArray::empty(shape, dtype);
  if (out.size() == 0) return out;
  const std::int64_t per = out.size() / nd;
  for (std::int64_t k = 0; k < nd; ++k) {
    const auto ku = static_cast<std::size_t>(k);
    std::int64_t inner = 1;
    for (std::size_t d = ku + 1; d < sizes.size(); ++d) inner *= sizes[d];
    const std::int64_t len = sizes[ku];
    NDArray axis = grid_axis(starts[ku], steps[ku], len, dtype);
    const std::size_t isz = out.itemsize();
    std::byte* dst = out.data() + static_cast<std::size_t>(k * per) * isz;
    for (std::int64_t f = 0; f < per; ++f) {
      const std::int64_t i = (f / inner) % len;
      std::memcpy(dst + static_cast<std::size_t>(f) * isz, axis.data() + static_cast<std::size_t>(i) * isz, isz);
    }
  }
  return out;
}

NDArray indices(const Shape& dims, DType dtype) {
  NDArray g = mgrid(std::vector<double>(dims.size(), 0.0), std::vector<double>(dims.size(), 1.0), dims,
                    DType::Int64);
  return dtype == DType::Int64 ? g : g.astype(dtype);
}

std::pair<NDArray, NDArray> tri_indices(std::int64_t n, std::int64_t m, std::int64_t k, bool upper) {
  n = std::max<std::int64_t>(n, 0);
  m = std::max<std::int64_t>(m, 0);
  auto keep = [&](std::int64_t i, std::int64_t j) { return upper ? j >= i + k : j <= i + k; };
  std::int64_t count = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    // Number of kept columns in row i.
    const std::int64_t lim = i + k;
    count += upper ? std::clamp<std::int64_t>(m - lim, 0, m) : std::clamp<std::int64_t>(lim + 1, 0, m);
  }
  NDArray rows = NDArray::empty({count}, DType::Int64);
  NDArray cols = NDArray::empty({count}, DType::Int64);
  auto* r = reinterpret_cast<std::int64_t*>(rows.data());
  auto* c = reinterpret_cast<std::int64_t*>(cols.data());
  std::int64_t pos = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t j = 0; j < m; ++j) {
      if (keep(i, j)) {
        r[pos] = i;
        c[pos] = j;
        ++pos;
      }
    }
  }
  return {rows, cols};
}

void fill_diagonal(const NDArray& a, const NDArray& values, bool wrap) {
  if (a.ndim() < 2) throw_error(ErrorKind::Value, "array must be at least 2-d");
  a.check_writeable();
  const auto& shape = a.shape();
  std::int64_t step = 0;
  std::int64_t end = a.size();
  if (a.ndim() == 2) {
    step = shape[1] + 1;
    if (!wrap) end = std::min(end, shape[1] * shape[1]);
  } else {
    for (std::size_t d = 1; d < shape.size(); ++d) {
      if (shape[d] != shape[0]) {
        throw_error(ErrorKind::Value, "All dimensions of input must be of equal length");
      }
    }
    step = 1;
    std::int64_t prod = 1;
    for (std::size_t d = 0; d + 1 < shape.size(); ++d) {
      prod *= shape[d];
      step += prod;
    }
  }
  if (values.dtype() != a.dtype()) throw_error(ErrorKind::DType, "fill_diagonal: dtype mismatch");
  const std::int64_t nv = values.size();
  if (nv == 0 || a.size() == 0) return;
  NDArray vals = values.is_c_contiguous() ? values : values.copy();
  const std::size_t isz = a.itemsize();
  const std::size_t nd = a.ndim();
  std::int64_t j = 0;
  for (std::int64_t f = 0; f < end; f += step, ++j) {
    std::int64_t rem = f;
    std::int64_t off = 0;
    for (std::size_t d = nd; d-- > 0;) {
      off += (rem % shape[d]) * a.strides()[d];
      rem /= shape[d];
    }
    std::memcpy(a.data() + off, vals.data() + static_cast<std::size_t>(j % nv) * isz, isz);
  }
}

NDArray concatenate(const std::vector<NDArray>& arrays, std::int64_t axis) {
  if (arrays.empty()) throw_error(ErrorKind::Value, "need at least one array to concatenate");
  const NDArray& first = arrays[0];
  const auto nd = static_cast<std::int64_t>(first.ndim());
  if (nd == 0) throw_error(ErrorKind::Value, "zero-dimensional arrays cannot be concatenated");
  if (axis < -nd || axis >= nd) {
    throw_error(ErrorKind::Index, "axis " + std::to_string(axis) + " is out of bounds for array of dimension " +
                                      std::to_string(nd));
  }
  if (axis < 0) axis += nd;
  const auto ax = static_cast<std::size_t>(axis);
  Shape shape = first.shape();
  shape[ax] = 0;
  for (std::size_t i = 0; i < arrays.size(); ++i) {
    const NDArray& a = arrays[i];
    if (a.dtype() != first.dtype()) throw_error(ErrorKind::DType, "concatenate: dtype mismatch");
    if (static_cast<std::int64_t>(a.ndim()) != nd) {
      throw_error(ErrorKind::Value,
                  "all the input arrays must have same number of dimensions, but the array at index 0 has " +
                      std::to_string(nd) + " dimension(s) and the array at index " + std::to_string(i) +
                      " has " + std::to_string(a.ndim()) + " dimension(s)");
    }
    for (std::size_t d = 0; d < a.ndim(); ++d) {
      if (d != ax && a.shape()[d] != first.shape()[d]) {
        throw_error(ErrorKind::Value,
                    "all the input array dimensions except for the concatenation axis must match exactly, "
                    "but along dimension " + std::to_string(d) + ", the array at index 0 has size " +
                        std::to_string(first.shape()[d]) + " and the array at index " + std::to_string(i) +
                        " has size " + std::to_string(a.shape()[d]));
      }
    }
    shape[ax] += a.shape()[ax];
  }
  NDArray out = NDArray::empty(shape, first.dtype());
  if (out.size() == 0) return out;
  std::int64_t pos = 0;
  for (const NDArray& a : arrays) {
    if (a.size() > 0) {
      NDArray dst = out.view(a.shape(), out.strides(), out.offset() + pos * out.strides()[ax]);
      copy_into(dst, a);
    }
    pos += a.shape()[ax];
  }
  return out;
}

// ---- np.fromstring text mode: port of NumPy's array_from_text (D-101) ----
namespace {

bool is_space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }
bool is_digit(char c) { return c >= '0' && c <= '9'; }
bool is_alnum(char c) {
  return is_digit(c) || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
bool starts_with_ci(const char* p, const char* word) {
  for (; *word != '\0'; ++p, ++word) {
    char c = *p;
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (c != *word) return false;
  }
  return true;
}

// CPython PyOS_string_to_double(s, &end, NULL): decimal float syntax, overflow
// to +-inf, underflow to +-0. On failure returns -1.0 and *end = s.
double plain_strtod(const char* s, const char** end) {
  const char* p = s;
  bool neg = false;
  if (*p == '+' || *p == '-') neg = *p++ == '-';
  const char* digits = p;
  std::int64_t int_digits = 0, lead_frac_zeros = 0;
  bool nonzero_int = false, seen_nonzero = false, any = false;
  while (is_digit(*p)) {
    any = true;
    if (*p != '0') nonzero_int = true;
    if (nonzero_int) ++int_digits;
    ++p;
  }
  if (*p == '.') {
    ++p;
    while (is_digit(*p)) {
      any = true;
      if (*p != '0') seen_nonzero = true;
      if (!nonzero_int && !seen_nonzero) ++lead_frac_zeros;
      ++p;
    }
  }
  if (!any) {
    *end = s;
    return -1.0;
  }
  std::int64_t exp = 0;
  if (*p == 'e' || *p == 'E') {
    const char* q = p + 1;
    bool eneg = false;
    if (*q == '+' || *q == '-') eneg = *q++ == '-';
    if (is_digit(*q)) {
      while (is_digit(*q)) {
        if (exp < 100000000) exp = exp * 10 + (*q - '0');
        ++q;
      }
      if (eneg) exp = -exp;
      p = q;
    }
  }
  *end = p;
  double v = 0.0;
  const auto r = std::from_chars(digits, p, v, std::chars_format::general);
  if (r.ec == std::errc::result_out_of_range) {
    const std::int64_t magnitude = (nonzero_int ? int_digits : -lead_frac_zeros) + exp;
    v = magnitude > 0 ? std::numeric_limits<double>::infinity() : 0.0;
  } else if (r.ec != std::errc{}) {
    *end = s;
    return -1.0;
  }
  return neg ? -v : v;
}

// NumPyOS_ascii_strtod: skips whitespace, POSIX inf/nan forms, else plain.
double numpy_strtod(const char* s, const char** end) {
  while (is_space(*s)) ++s;
  const char* p = s;
  double sign = 1.0;
  if (*p == '-') {
    sign = -1.0;
    ++p;
  } else if (*p == '+') {
    ++p;
  }
  if (starts_with_ci(p, "nan")) {
    p += 3;
    if (*p == '(') {
      ++p;
      while (is_alnum(*p) || *p == '_') ++p;
      if (*p == ')') ++p;
    }
    *end = p;
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (starts_with_ci(p, "inf")) {
    p += 3;
    if (starts_with_ci(p, "inity")) p += 5;
    *end = p;
    return sign * std::numeric_limits<double>::infinity();
  }
  return plain_strtod(s, end);
}

// CPython PyOS_strtoul(str, ptr, 10) (64-bit unsigned long).
std::uint64_t py_strtoul(const char* s, const char** end) {
  while (is_space(*s)) ++s;
  while (*s == '0') ++s;
  std::uint64_t r = 0;
  bool overflow = false;
  while (is_digit(*s)) {
    const auto d = static_cast<std::uint64_t>(*s - '0');
    if (!overflow) {
      if (r > (std::numeric_limits<std::uint64_t>::max() - d) / 10) {
        overflow = true;
      } else {
        r = r * 10 + d;
      }
    }
    ++s;
  }
  *end = s;
  return overflow ? std::numeric_limits<std::uint64_t>::max() : r;
}

// CPython PyOS_strtol(str, ptr, 10) (64-bit long).
std::int64_t py_strtol(const char* s, const char** end) {
  while (is_space(*s)) ++s;
  const char sign = *s;
  if (sign == '+' || sign == '-') ++s;
  const std::uint64_t u = py_strtoul(s, end);
  constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (u <= kMax) {
    const auto v = static_cast<std::int64_t>(u);
    return sign == '-' ? -v : v;
  }
  if (sign == '-' && u == kMax + 1) return std::numeric_limits<std::int64_t>::min();
  return std::numeric_limits<std::int64_t>::max();
}

// The dtype's `fromstr`: parses one element at `s` into `out`, sets *end.
template <typename T>
void fromstr(const char* s, std::byte* out, const char** end) {
  if constexpr (is_complex_v<T>) {
    using V = typename T::value_type;
    double re = numpy_strtod(s, end);
    double im = 0.0;
    if (**end == '+' || **end == '-') {
      const char* str = *end;
      const double r2 = numpy_strtod(str, end);
      if (**end == 'j') {
        im = r2;
        ++*end;
      }
    } else if (**end == 'j') {
      im = re;
      re = 0.0;
      ++*end;
    }
    store<T>(out, T(static_cast<V>(re), static_cast<V>(im)));
  } else if constexpr (std::is_same_v<T, bool>) {
    store<bool>(out, numpy_strtod(s, end) != 0.0);
  } else if constexpr (std::is_same_v<T, float16_t>) {
    store<float16_t>(out, double_to_half(numpy_strtod(s, end)));
  } else if constexpr (std::is_floating_point_v<T>) {
    store<T>(out, static_cast<T>(numpy_strtod(s, end)));
  } else if constexpr (std::is_signed_v<T>) {
    // C conversion from long: modular truncation.
    store<T>(out, static_cast<T>(py_strtol(s, end)));
  } else {
    store<T>(out, static_cast<T>(py_strtoul(s, end)));
  }
}

// NumPy swab_separator.
std::string swab_separator(const std::string& sep) {
  std::string out;
  bool skip_space = false;
  if (!sep.empty() && !is_space(sep[0])) out.push_back(' ');
  for (const char c : sep) {
    if (is_space(c)) {
      if (!skip_space) {
        out.push_back(' ');
        skip_space = true;
      }
    } else {
      out.push_back(c);
      skip_space = false;
    }
  }
  if (!out.empty() && out.back() == ' ') out.push_back(' ');
  return out;
}

// NumPy fromstr_skip_separator: 0 matched, -1 end of string, -2 mismatch.
int skip_separator(const char** s, const char* sep, const char* end) {
  const char* str = *s;
  int result = 0;
  while (true) {
    const char c = *str;
    if (str >= end) {
      result = -1;
      break;
    }
    if (*sep == '\0') {
      result = str != *s ? 0 : -2;
      break;
    }
    if (*sep == ' ') {
      if (!is_space(c)) {
        ++sep;
        continue;
      }
    } else if (*sep != c) {
      result = -2;
      break;
    } else {
      ++sep;
    }
    ++str;
  }
  *s = str;
  return result;
}

}  // namespace

NDArray fromstring(const std::string& text, DType dtype, std::int64_t count, const std::string& sep) {
  if (sep.empty()) {
    throw_error(ErrorKind::Value, "The binary mode of fromstring is removed, use frombuffer instead");
  }
  const std::string clean = swab_separator(sep);
  const char* s = text.c_str();
  const char* end = s + text.size();
  const std::size_t isz = itemsize(dtype);
  std::vector<std::byte> buf;
  std::int64_t nread = 0;
  int flag = 0;
  dispatch_dtype(dtype, [&](auto tag) {
    using T = dtype_t<decltype(tag)::value>;
    for (std::int64_t i = 0; count < 0 || i < count; ++i) {
      buf.resize(static_cast<std::size_t>(nread + 1) * isz);
      const char* e = s;
      fromstr<T>(s, buf.data() + static_cast<std::size_t>(nread) * isz, &e);
      if (e == s) {
        flag = s >= end ? -1 : -2;
        break;
      }
      s = e;
      ++nread;
      flag = skip_separator(&s, clean.c_str(), end);
      if (flag < 0) {
        if (count == i + 1) flag = -1;
        break;
      }
    }
  });
  if (flag == -2) {
    throw_error(ErrorKind::Value, "string or file could not be read to its end due to unmatched data");
  }
  if (count >= 0 && nread < count) throw_error(ErrorKind::Value, "string is smaller than requested size");
  NDArray out = NDArray::empty({nread}, dtype);
  if (nread > 0) std::memcpy(out.data(), buf.data(), static_cast<std::size_t>(nread) * isz);
  return out;
}

}  // namespace nativpy::p07
