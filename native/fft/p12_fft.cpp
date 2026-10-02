// P12 FFT completion (D-150..D-152). The 1-D lane loops are ported from the
// M10 kernels in fft.cpp (which port NumPy's _pocketfft_umath.cpp, BSD-3),
// with the loop type chosen the way NumPy's ufunc type resolution does.
#include "p12_fft.hpp"

#include <complex>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"

#include "broadcast.hpp"
#include "cast.hpp"
#include "dtype.hpp"
#include "error.hpp"
#include "shape.hpp"

namespace nativpy::fft::p12 {

namespace {

using idx = std::int64_t;

std::size_t sz(idx v) { return static_cast<std::size_t>(v); }

enum class Kind { C2C, R2C, C2R };

[[noreturn]] void bad_points(idx n) {
  throw_error(ErrorKind::Value,
              "Invalid number of FFT data points (" + std::to_string(n) + ") specified.");
}

DType real_of(DType dt) {
  switch (dt) {
    case DType::Float16: return DType::Float16;
    case DType::Float32:
    case DType::Complex64: return DType::Float32;
    default: return DType::Float64;
  }
}

// numpy _raw_fft: fct = 1 (Python int) for "backward" after the inverse swap,
// else reciprocal(sqrt(n)) / reciprocal(n) in result_type(a.real.dtype, 1.0).
struct Factor {
  bool is_one = true;
  DType dtype = DType::Float64;  // dtype of a non-unit factor
  double value = 1.0;
};

Factor factor_of(Norm norm, idx n, bool forward, DType a_dtype) {
  if (!forward) {
    if (norm == Norm::Backward) {
      norm = Norm::Forward;
    } else if (norm == Norm::Forward) {
      norm = Norm::Backward;
    }
  }
  Factor f;
  if (norm == Norm::Backward) return f;
  f.is_one = false;
  f.dtype = real_of(a_dtype);
  const bool ortho = norm == Norm::Ortho;
  if (f.dtype == DType::Float16) {
    const auto h = [](double v) { return half_to_double(double_to_half(v)); };
    const double hn = h(static_cast<double>(n));
    f.value = ortho ? h(1.0 / h(std::sqrt(hn))) : h(1.0 / hn);
  } else if (f.dtype == DType::Float32) {
    const float fn = static_cast<float>(n);
    f.value = static_cast<double>(ortho ? 1.0f / std::sqrt(fn) : 1.0f / fn);
  } else {
    const double dn = static_cast<double>(n);
    f.value = ortho ? 1.0 / std::sqrt(dn) : 1.0 / dn;
  }
  return f;
}

template <typename T>
DType complex_dtype() {
  return std::is_same_v<T, float> ? DType::Complex64 : DType::Complex128;
}
template <typename T>
DType real_dtype() {
  return std::is_same_v<T, float> ? DType::Float32 : DType::Float64;
}

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

template <typename Fn>
void for_each_lane(const Lanes& in, idx nout, Fn&& fn) {
  for (idx o = 0; o < in.outer; ++o) {
    for (idx i = 0; i < in.inner; ++i) fn(o * in.len * in.inner + i, o * nout * in.inner + i);
  }
}

// `a` is C-contiguous complex<T>; zero-pad or truncate to nout (numpy fft_loop).
template <typename T>
NDArray c2c_axis(const NDArray& a, std::size_t axis, idx nout, bool forward, T fct) {
  Shape out_shape = a.shape();
  out_shape[axis] = nout;
  NDArray out = NDArray::empty(out_shape, complex_dtype<T>());
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

// `a` is C-contiguous T; npts real points -> npts/2+1 complex (numpy rfft_impl).
template <typename T>
NDArray r2c_axis(const NDArray& a, std::size_t axis, idx npts, T fct) {
  const idx nout = npts / 2 + 1;
  Shape out_shape = a.shape();
  out_shape[axis] = nout;
  NDArray out = NDArray::empty(out_shape, complex_dtype<T>());
  if (out.size() == 0) return out;
  const Lanes l = lanes_of(a.shape(), axis);
  const auto* ip = reinterpret_cast<const T*>(a.data());
  auto* op = reinterpret_cast<std::complex<T>*>(out.data());
  auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(sz(npts));
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

// `a` is C-contiguous complex<T>; Hermitian half-spectrum -> nout reals.
template <typename T>
NDArray c2r_axis(const NDArray& a, std::size_t axis, idx nout, T fct) {
  Shape out_shape = a.shape();
  out_shape[axis] = nout;
  NDArray out = NDArray::empty(out_shape, real_dtype<T>());
  if (out.size() == 0) return out;
  const Lanes l = lanes_of(a.shape(), axis);
  const auto* ip = reinterpret_cast<const std::complex<T>*>(a.data());
  auto* op = reinterpret_cast<T*>(out.data());
  auto plan = pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<T>>(sz(nout));
  std::vector<T> buf(sz(nout));
  const idx npairs = (nout - 1) / 2;
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

NDArray prepare(const NDArray& a, DType target) {
  if (a.dtype() == target && a.is_c_contiguous()) return a;
  return a.astype(target);
}

template <typename T>
NDArray run_loop(Kind kind, const NDArray& a, std::size_t ax, idx n, bool forward, T fct) {
  switch (kind) {
    case Kind::C2C: return c2c_axis<T>(prepare(a, complex_dtype<T>()), ax, n, forward, fct);
    case Kind::R2C: return r2c_axis<T>(prepare(a, real_dtype<T>()), ax, n, fct);
    case Kind::C2R: return c2r_axis<T>(prepare(a, complex_dtype<T>()), ax, n, fct);
  }
  return a;  // unreachable
}

const char* ufunc_name(Kind kind, bool forward, idx n) {
  if (kind == Kind::C2C) return forward ? "fft" : "ifft";
  if (kind == Kind::C2R) return "irfft";
  return n % 2 == 0 ? "rfft_n_even" : "rfft_n_odd";
}

// numpy _raw_fft with an optional `out` (D-150).
NDArray raw_fft(const NDArray& a, std::optional<idx> n_opt, idx axis, Norm norm, Kind kind,
                bool forward, const std::optional<NDArray>& out) {
  if (n_opt && *n_opt < 1) bad_points(*n_opt);
  if (a.ndim() == 0) {
    throw_error(ErrorKind::Index, "FFT requires an array with at least one dimension");
  }
  const auto ax = static_cast<std::size_t>(normalize_axis(axis, static_cast<idx>(a.ndim())));
  idx n = 0;
  if (n_opt) {
    n = *n_opt;
  } else {
    n = kind == Kind::C2R ? 2 * (a.shape()[ax] - 1) : a.shape()[ax];
  }
  if (n < 1) bad_points(n);
  const idx n_out = kind == Kind::R2C ? n / 2 + 1 : n;
  if (out && (out->ndim() != a.ndim() || out->shape()[ax] != n_out)) {
    throw_error(ErrorKind::Value, "output array has wrong shape.");
  }
  if (kind == Kind::R2C && is_complex(a.dtype())) {
    throw_error(ErrorKind::DType, std::string("ufunc '") + ufunc_name(kind, forward, n) +
                                      "' not supported for complex input (use fft)");
  }
  const Factor f = factor_of(norm, n, forward, a.dtype());
  const DType single_in = kind == Kind::R2C ? DType::Float32 : DType::Complex64;
  const bool single = !f.is_one && f.dtype == DType::Float32 && a.dtype() == single_in;
  // Result dtype without `out` (numpy: result_type(a, 1j) or the real dtype).
  DType res;
  if (kind == Kind::C2R) {
    res = real_of(a.dtype());
  } else {
    res = real_of(a.dtype()) == DType::Float64 ? DType::Complex128 : DType::Complex64;
  }
  const DType loop_out = kind == Kind::C2R ? (single ? DType::Float32 : DType::Float64)
                                           : (single ? DType::Complex64 : DType::Complex128);
  if (out) {
    if (!can_cast(loop_out, out->dtype(), Casting::SameKind)) {
      throw_error(ErrorKind::DType, std::string("Cannot cast ufunc '") +
                                        ufunc_name(kind, forward, n) + "' output from " +
                                        std::string(dtype_name(loop_out)) + " to " +
                                        std::string(dtype_name(out->dtype())) +
                                        " with casting rule 'same_kind'");
    }
    if (!out->writeable()) throw_error(ErrorKind::Value, "output array is read-only");
    Shape rshape = a.shape();
    rshape[ax] = n_out;
    // Checked before computing, so a failure leaves `out` unchanged.
    const Shape b = broadcast_shapes({rshape, out->shape()});
    if (b != out->shape()) {
      throw_error(ErrorKind::Broadcast, "non-broadcastable output operand with shape " +
                                            shape_to_string(out->shape()) +
                                            " doesn't match the broadcast shape " +
                                            shape_to_string(b));
    }
  }
  NDArray r = single ? run_loop<float>(kind, a, ax, n, forward, static_cast<float>(f.value))
                     : run_loop<double>(kind, a, ax, n, forward, f.value);
  if (out) {
    copy_into(*out, r);
    return *out;
  }
  return r.dtype() == res ? r : r.astype(res);
}

// In-place complex conjugate (ihfft writes conj into its result / out).
void conj_inplace(const NDArray& a) {
  if (!is_complex(a.dtype())) return;
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = dtype_t<decltype(tag)::value>;
    if constexpr (is_complex_v<T>) {
      for_each_element(a, [](std::byte* p) { store<T>(p, std::conj(load<T>(p))); });
    }
  });
}

NDArray conj_copy(const NDArray& a) {
  if (!is_complex(a.dtype())) return a;
  NDArray c = a.copy();
  conj_inplace(c);
  return c;
}

Norm swap(Norm n) {
  if (n == Norm::Backward) return Norm::Forward;
  if (n == Norm::Forward) return Norm::Backward;
  return n;
}

std::int64_t axis_len(const NDArray& a, idx axis) {
  const idx nd = static_cast<idx>(a.ndim());
  if (axis < -nd || axis >= nd) {
    throw_error(ErrorKind::Index, "axis " + std::to_string(axis) +
                                      " is out of bounds for array of dimension " +
                                      std::to_string(nd));
  }
  return a.shape()[sz(axis < 0 ? axis + nd : axis)];
}

// numpy.fft._cook_nd_args.
std::pair<std::vector<idx>, std::vector<idx>> cook_nd(const NDArray& a,
                                                      const std::optional<std::vector<idx>>& s_opt,
                                                      const std::optional<std::vector<idx>>& axes_opt,
                                                      bool invreal) {
  const bool shapeless = !s_opt;
  std::vector<idx> s;
  std::vector<idx> axes;
  if (shapeless) {
    if (axes_opt) {
      for (const idx ax : *axes_opt) s.push_back(axis_len(a, ax));
    } else {
      s.assign(a.shape().begin(), a.shape().end());
    }
  } else {
    s = *s_opt;
  }
  if (axes_opt) {
    axes = *axes_opt;
  } else {
    const idx k = static_cast<idx>(s.size());
    for (idx i = -k; i < 0; ++i) axes.push_back(i);
  }
  if (s.size() != axes.size()) {
    throw_error(ErrorKind::Value, "Shape and axes have different lengths.");
  }
  if (invreal && shapeless) {
    if (axes.empty()) throw_error(ErrorKind::Index, "list index out of range");
    s.back() = (axis_len(a, axes.back()) - 1) * 2;
  }
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] == -1) s[i] = axis_len(a, axes[i]);
  }
  return {s, axes};
}

}  // namespace

NDArray transform(Op op, const NDArray& a, std::optional<idx> n, idx axis, Norm norm,
                  const std::optional<NDArray>& out) {
  switch (op) {
    case Op::Fft: return raw_fft(a, n, axis, norm, Kind::C2C, true, out);
    case Op::Ifft: return raw_fft(a, n, axis, norm, Kind::C2C, false, out);
    case Op::Rfft: return raw_fft(a, n, axis, norm, Kind::R2C, true, out);
    case Op::Irfft: return raw_fft(a, n, axis, norm, Kind::C2R, false, out);
    case Op::Hfft: {
      if (!n) n = (axis_len(a, axis) - 1) * 2;
      return raw_fft(conj_copy(a), n, axis, swap(norm), Kind::C2R, false, out);
    }
    case Op::Ihfft: {
      if (!n) n = axis_len(a, axis);
      NDArray r = raw_fft(a, n, axis, swap(norm), Kind::R2C, true, out);
      conj_inplace(r);
      return r;
    }
  }
  return a;  // unreachable
}

NDArray transform_nd(NdOp op, const NDArray& a, const std::optional<std::vector<idx>>& s_opt,
                     const std::optional<std::vector<idx>>& axes_opt, Norm norm,
                     const std::optional<NDArray>& out) {
  const auto [s, axes] = cook_nd(a, s_opt, axes_opt, op == NdOp::Irfftn);
  const std::size_t k = axes.size();
  switch (op) {
    case NdOp::Fftn:
    case NdOp::Ifftn: {
      const Op one = op == NdOp::Fftn ? Op::Fft : Op::Ifft;
      NDArray cur = a;
      for (std::size_t i = k; i-- > 0;) cur = transform(one, cur, s[i], axes[i], norm, out);
      return cur;
    }
    case NdOp::Rfftn: {
      if (k == 0) throw_error(ErrorKind::Index, "list index out of range");
      NDArray cur = transform(Op::Rfft, a, s[k - 1], axes[k - 1], norm, out);
      for (std::size_t i = k - 1; i-- > 0;) cur = transform(Op::Fft, cur, s[i], axes[i], norm, out);
      return cur;
    }
    case NdOp::Irfftn: {
      if (k == 0) throw_error(ErrorKind::Index, "list index out of range");
      NDArray cur = a;
      for (std::size_t i = 0; i + 1 < k; ++i) {
        cur = transform(Op::Ifft, cur, s[i], axes[i], norm, std::nullopt);
      }
      return transform(Op::Irfft, cur, s[k - 1], axes[k - 1], norm, out);
    }
  }
  return a;  // unreachable
}

NDArray roll(const NDArray& a, const std::vector<idx>& shifts, const std::vector<idx>& axes) {
  const idx nd = static_cast<idx>(a.ndim());
  std::vector<idx> total(a.ndim(), 0);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    total[sz(normalize_axis(axes[i], nd))] += shifts[i];
  }
  if (a.size() == 0) return a.copy();
  NDArray cur = a;
  bool copied = false;
  for (std::size_t d = 0; d < a.ndim(); ++d) {
    const idx len = a.shape()[d];
    if (len == 0) continue;
    const idx k = ((total[d] % len) + len) % len;
    if (k == 0) continue;
    NDArray dst = NDArray::empty(cur.shape(), cur.dtype());
    const auto part = [](const NDArray& x, std::size_t dim, idx start, idx count) {
      Shape sh = x.shape();
      sh[dim] = count;
      return x.view(sh, x.strides(), x.offset() + start * x.strides()[dim]);
    };
    copy_into(part(dst, d, k, len - k), part(cur, d, 0, len - k));
    copy_into(part(dst, d, 0, k), part(cur, d, len - k, k));
    cur = std::move(dst);
    copied = true;
  }
  return copied ? cur : a.copy();
}

NDArray shift(const NDArray& a, const std::optional<std::vector<idx>>& axes_opt, bool inverse) {
  std::vector<idx> axes;
  if (axes_opt) {
    axes = *axes_opt;
  } else {
    for (idx d = 0; d < static_cast<idx>(a.ndim()); ++d) axes.push_back(d);
  }
  if (a.ndim() == 0 && axes.empty()) {
    throw_error(ErrorKind::Value, "not enough values to unpack (expected 2, got 0)");
  }
  std::vector<idx> shifts;
  for (const idx ax : axes) {
    const idx h = axis_len(a, ax) / 2;
    shifts.push_back(inverse ? -h : h);
  }
  return roll(a, shifts, axes);
}

}  // namespace nativpy::fft::p12
