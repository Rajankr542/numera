// NumPy-compatible FFT (M10, D-020). The 1-D lane loops port NumPy's
// numpy/fft/_pocketfft_umath.cpp (BSD-3) onto nativpy NDArrays.
#include "fft.hpp"

#include <cmath>
#include <complex>
#include <cstddef>
#include <string>
#include <type_traits>
#include <vector>

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

#include "dtype.hpp"
#include "error.hpp"
#include "shape.hpp"

namespace nativpy::fft {

namespace {

using idx = std::int64_t;

std::size_t sz(idx v) { return static_cast<std::size_t>(v); }

template <typename T>
DType dtype_of() {
  if constexpr (std::is_same_v<T, float>) {
    return DType::Float32;
  } else if constexpr (std::is_same_v<T, double>) {
    return DType::Float64;
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    return DType::Complex64;
  } else {
    static_assert(std::is_same_v<T, std::complex<double>>);
    return DType::Complex128;
  }
}

// float16/float32/complex64 compute in float32; everything else in float64.
bool single_precision(DType dt) {
  return dt == DType::Float16 || dt == DType::Float32 || dt == DType::Complex64;
}

[[noreturn]] void bad_points(idx n) {
  throw_error(ErrorKind::Value,
              "Invalid number of FFT data points (" + std::to_string(n) + ") specified.");
}

// Normalization factor `fct` for a transform of length n (numpy _raw_fft).
// Inverse transforms swap backward <-> forward.
template <typename T>
T factor(Norm norm, idx n, bool forward) {
  if (!forward) {
    if (norm == Norm::Backward) {
      norm = Norm::Forward;
    } else if (norm == Norm::Forward) {
      norm = Norm::Backward;
    }
  }
  switch (norm) {
    case Norm::Backward: return T(1);
    case Norm::Ortho: return T(1) / std::sqrt(static_cast<T>(n));
    case Norm::Forward: return T(1) / static_cast<T>(n);
  }
  return T(1);
}

// NumPy computes fct in result_type(a.real.dtype, 1.0), i.e. float16 for
// float16 input: reciprocal(sqrt(n, dtype=f16)) / reciprocal(n, dtype=f16),
// each step rounded to half precision.
float factor_half(Norm norm, idx n, bool forward) {
  const auto h = [](double v) { return half_to_double(double_to_half(v)); };
  if (!forward) {
    if (norm == Norm::Backward) {
      norm = Norm::Forward;
    } else if (norm == Norm::Forward) {
      norm = Norm::Backward;
    }
  }
  const double hn = h(static_cast<double>(n));
  switch (norm) {
    case Norm::Backward: return 1.0f;
    case Norm::Ortho: return static_cast<float>(h(1.0 / h(std::sqrt(hn))));
    case Norm::Forward: return static_cast<float>(h(1.0 / hn));
  }
  return 1.0f;
}

// C-contiguous input viewed as [outer, len, inner] around `axis`.
struct Lanes {
  idx outer = 1;
  idx len = 0;
  idx inner = 1;
};

Lanes lanes_of(const Shape& s, std::size_t axis) {
  Lanes l;
  for (std::size_t d = 0; d < axis; ++d) l.outer *= s[d];
  l.len = s[axis];
  for (std::size_t d = axis + 1; d < s.size(); ++d) l.inner *= s[d];
  return l;
}

// Calls fn(lane_in_offset, lane_out_offset) (element offsets) for every lane.
template <typename Fn>
void for_each_lane(const Lanes& in, idx nout, Fn&& fn) {
  for (idx o = 0; o < in.outer; ++o) {
    for (idx i = 0; i < in.inner; ++i) {
      fn(o * in.len * in.inner + i, o * nout * in.inner + i);
    }
  }
}

// Complex forward/backward along `axis` with output length nout (zero-pad or
// truncate, numpy fft_loop). `a` is C-contiguous std::complex<T>.
template <typename T>
NDArray c2c_axis(const NDArray& a, std::size_t axis, idx nout, bool forward, T fct) {
  Shape out_shape = a.shape();
  out_shape[axis] = nout;
  NDArray out = NDArray::empty(out_shape, dtype_of<std::complex<T>>());
  if (out.size() == 0) return out;
  const Lanes l = lanes_of(a.shape(), axis);
  const auto* ip = reinterpret_cast<const std::complex<T>*>(a.data());
  auto* op = reinterpret_cast<std::complex<T>*>(out.data());
  auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_c<T>>(sz(nout));
  std::vector<std::complex<T>> buf(sz(nout));
  const idx ncopy = l.len < nout ? l.len : nout;
  for_each_lane(l, nout, [&](idx in0, idx out0) {
    idx k = 0;
    for (; k < ncopy; ++k) buf[sz(k)] = ip[in0 + k * l.inner];
    for (; k < nout; ++k) buf[sz(k)] = std::complex<T>(0);
    plan->exec(reinterpret_cast<pocketfft::detail::cmplx<T>*>(buf.data()), fct, forward);
    for (k = 0; k < nout; ++k) op[out0 + k * l.inner] = buf[sz(k)];
  });
  return out;
}

// Real forward transform: npts points -> npts/2+1 complex (numpy rfft_impl).
template <typename T>
NDArray r2c_axis(const NDArray& a, std::size_t axis, idx npts, T fct) {
  const idx nout = npts / 2 + 1;
  Shape out_shape = a.shape();
  out_shape[axis] = nout;
  NDArray out = NDArray::empty(out_shape, dtype_of<std::complex<T>>());
  if (out.size() == 0) return out;
  const Lanes l = lanes_of(a.shape(), axis);
  const auto* ip = reinterpret_cast<const T*>(a.data());
  auto* op = reinterpret_cast<std::complex<T>*>(out.data());
  auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(sz(npts));
  // 2*nout reals, data at offset 1: after exec the FFTpack order R0,R1,I1,...
  // becomes complex pairs once R0 is moved to slot 0 and I0 is set to 0.
  std::vector<T> buf(sz(2 * nout));
  const idx ncopy = l.len < npts ? l.len : npts;
  for_each_lane(l, nout, [&](idx in0, idx out0) {
    T* b = buf.data() + 1;
    idx k = 0;
    for (; k < ncopy; ++k) b[k] = ip[in0 + k * l.inner];
    for (; k < 2 * nout - 1; ++k) b[k] = T(0);
    plan->exec(b, fct, pocketfft::FORWARD);
    buf[0] = buf[1];
    buf[1] = T(0);
    for (k = 0; k < nout; ++k) {
      op[out0 + k * l.inner] = std::complex<T>(buf[sz(2 * k)], buf[sz(2 * k + 1)]);
    }
  });
  return out;
}

// Hermitian half-spectrum -> nout real points (numpy irfft_loop).
template <typename T>
NDArray c2r_axis(const NDArray& a, std::size_t axis, idx nout, T fct) {
  Shape out_shape = a.shape();
  out_shape[axis] = nout;
  NDArray out = NDArray::empty(out_shape, dtype_of<T>());
  if (out.size() == 0) return out;
  const Lanes l = lanes_of(a.shape(), axis);
  const auto* ip = reinterpret_cast<const std::complex<T>*>(a.data());
  auto* op = reinterpret_cast<T*>(out.data());
  auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(sz(nout));
  std::vector<T> buf(sz(nout));
  const idx npairs = (nout - 1) / 2;  // R1,I1 .. Rk,Ik
  for_each_lane(l, nout, [&](idx in0, idx out0) {
    const auto at = [&](idx k) { return ip[in0 + k * l.inner]; };
    buf[0] = l.len > 0 ? at(0).real() : T(0);
    for (idx k = 1; k <= npairs; ++k) {
      const std::complex<T> v = k < l.len ? at(k) : std::complex<T>(0);
      buf[sz(2 * k - 1)] = v.real();
      buf[sz(2 * k)] = v.imag();
    }
    if (nout > 1 && nout % 2 == 0) {
      buf[sz(nout - 1)] = nout / 2 >= l.len ? T(0) : at(nout / 2).real();
    }
    plan->exec(buf.data(), fct, pocketfft::BACKWARD);
    for (idx k = 0; k < nout; ++k) op[out0 + k * l.inner] = buf[sz(k)];
  });
  return out;
}

// C-contiguous copy of `a` in the compute dtype (complex or real).
NDArray prepare(const NDArray& a, bool single, bool complex_in) {
  DType target;
  if (complex_in) {
    target = single ? DType::Complex64 : DType::Complex128;
  } else {
    target = single ? DType::Float32 : DType::Float64;
  }
  if (a.dtype() == target && a.is_c_contiguous()) return a;
  return a.astype(target);
}

std::size_t check_axis(const NDArray& a, idx axis) {
  if (a.ndim() == 0) {
    throw_error(ErrorKind::Index, "FFT requires an array with at least one dimension");
  }
  return static_cast<std::size_t>(normalize_axis(axis, static_cast<idx>(a.ndim())));
}

enum class Kind { C2C, R2C, C2R };

NDArray raw_fft(const NDArray& a, std::optional<idx> n_opt, idx axis, Norm norm, Kind kind,
                bool forward) {
  // NumPy order: an explicit n is validated before the axis.
  if (n_opt && *n_opt < 1) bad_points(*n_opt);
  const std::size_t ax = check_axis(a, axis);
  if (kind == Kind::R2C && is_complex(a.dtype())) {
    throw_error(ErrorKind::DType, "rfft: complex input is not supported (use fft)");
  }
  idx n = 0;
  if (n_opt) {
    n = *n_opt;
  } else {
    n = kind == Kind::C2R ? 2 * (a.shape()[ax] - 1) : a.shape()[ax];
  }
  if (n < 1) bad_points(n);
  const bool single = single_precision(a.dtype());
  const NDArray in = prepare(a, single, kind != Kind::R2C);
  if (single) {
    const float f = a.dtype() == DType::Float16 ? factor_half(norm, n, forward)
                                                 : factor<float>(norm, n, forward);
    switch (kind) {
      case Kind::C2C: return c2c_axis<float>(in, ax, n, forward, f);
      case Kind::R2C: return r2c_axis<float>(in, ax, n, f);
      case Kind::C2R: {
        NDArray r = c2r_axis<float>(in, ax, n, f);
        return a.dtype() == DType::Float16 ? r.astype(DType::Float16) : r;
      }
    }
  }
  const double f = factor<double>(norm, n, forward);
  switch (kind) {
    case Kind::C2C: return c2c_axis<double>(in, ax, n, forward, f);
    case Kind::R2C: return r2c_axis<double>(in, ax, n, f);
    case Kind::C2R: return c2r_axis<double>(in, ax, n, f);
  }
  return in;  // unreachable
}

NDArray raw_fftn(const NDArray& a, const std::optional<std::vector<idx>>& s,
                 const std::optional<std::vector<idx>>& axes_opt, Norm norm, bool forward) {
  const idx nd = static_cast<idx>(a.ndim());
  std::vector<idx> axes;
  if (axes_opt) {
    axes = *axes_opt;
  } else {
    const idx k = s ? static_cast<idx>(s->size()) : nd;
    for (idx i = -k; i < 0; ++i) axes.push_back(i);
  }
  std::vector<idx> sizes;
  if (s) {
    if (s->size() != axes.size()) {
      throw_error(ErrorKind::Value, "Shape and axes have different lengths.");
    }
    sizes = *s;
  } else {
    sizes.assign(axes.size(), -1);
  }
  if (axes.empty()) return a;  // NumPy returns the input unchanged
  NDArray cur = a;
  for (std::size_t i = axes.size(); i-- > 0;) {
    std::optional<idx> n;
    if (sizes[i] != -1) n = sizes[i];
    cur = raw_fft(cur, n, axes[i], norm, Kind::C2C, forward);
  }
  return cur;
}

}  // namespace

Norm parse_norm(const std::string& s) {
  if (s.empty() || s == "backward") return Norm::Backward;
  if (s == "ortho") return Norm::Ortho;
  if (s == "forward") return Norm::Forward;
  throw_error(ErrorKind::Value, "Invalid norm value " + s +
                                    "; should be \"backward\", \"ortho\" or \"forward\".");
}

NDArray fft(const NDArray& a, std::optional<idx> n, idx axis, Norm norm) {
  return raw_fft(a, n, axis, norm, Kind::C2C, true);
}
NDArray ifft(const NDArray& a, std::optional<idx> n, idx axis, Norm norm) {
  return raw_fft(a, n, axis, norm, Kind::C2C, false);
}
NDArray rfft(const NDArray& a, std::optional<idx> n, idx axis, Norm norm) {
  return raw_fft(a, n, axis, norm, Kind::R2C, true);
}
NDArray irfft(const NDArray& a, std::optional<idx> n, idx axis, Norm norm) {
  return raw_fft(a, n, axis, norm, Kind::C2R, false);
}

NDArray fftn(const NDArray& a, const std::optional<std::vector<idx>>& s,
             const std::optional<std::vector<idx>>& axes, Norm norm) {
  return raw_fftn(a, s, axes, norm, true);
}
NDArray ifftn(const NDArray& a, const std::optional<std::vector<idx>>& s,
              const std::optional<std::vector<idx>>& axes, Norm norm) {
  return raw_fftn(a, s, axes, norm, false);
}

// numpy.fft.fftfreq: results = [0, 1, ..., (n-1)//2, -(n//2), ..., -1] / (d*n)
NDArray fftfreq(idx n, double d) {
  if (n < 0) throw_error(ErrorKind::Value, "negative dimensions are not allowed");
  if (n == 0 || d == 0.0) throw_error(ErrorKind::Value, "float division by zero");
  NDArray out = NDArray::empty({n}, DType::Float64);
  auto* p = reinterpret_cast<double*>(out.data());
  const double val = 1.0 / (static_cast<double>(n) * d);
  const idx half = (n - 1) / 2 + 1;
  for (idx i = 0; i < half; ++i) p[i] = static_cast<double>(i) * val;
  for (idx i = half; i < n; ++i) p[i] = static_cast<double>(-(n / 2) + (i - half)) * val;
  return out;
}

// numpy.fft.rfftfreq: [0, 1, ..., n//2] / (d*n). Negative n gives an empty
// result (NumPy: arange(n//2 + 1) with a non-positive length).
NDArray rfftfreq(idx n, double d) {
  if (n == 0 || d == 0.0) throw_error(ErrorKind::Value, "float division by zero");
  idx m = (n >= 0 ? n / 2 : -((-n + 1) / 2)) + 1;  // Python floor division
  if (m < 0) m = 0;
  NDArray out = NDArray::empty({m}, DType::Float64);
  auto* p = reinterpret_cast<double*>(out.data());
  const double val = 1.0 / (static_cast<double>(n) * d);
  for (idx i = 0; i < m; ++i) p[i] = static_cast<double>(i) * val;
  return out;
}

}  // namespace nativpy::fft

