#include "reduce.hpp"

#include <algorithm>
#include <cmath>
#include <string>
#include <type_traits>

#include "cast.hpp"
#include "error.hpp"
#include "shape.hpp"
#include "shape_ops.hpp"
#include "ufunc_kernels.hpp"

namespace nativpy {

namespace {

// float16 is computed as float (like the ufunc kernels, D-014).
template <typename S>
using compute_t = std::conditional_t<std::is_same_v<S, float16_t>, float, S>;

template <typename S>
compute_t<S> ld(const std::byte* p) noexcept {
  return cast_value<compute_t<S>>(load<S>(p));
}

template <typename T>
bool is_nan(T v) noexcept {
  if constexpr (std::is_floating_point_v<T>) return std::isnan(v);
  else return false;
}

// NumPy min/max treat -0.0 < +0.0 (order-independent result).
template <typename T>
bool sign_bit(T v) noexcept {
  if constexpr (std::is_floating_point_v<T>) return std::signbit(v);
  else return false;
}

// Input rearranged so that reduced axes are last and C-contiguous.
struct Work {
  NDArray data;       // shape = kept dims + reduced dims, C-contiguous
  Shape out_shape;    // kept dims (keepdims applied)
  std::int64_t rows;  // number of output elements
  std::int64_t n;     // elements reduced per output
};

Work prepare(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis,
             bool keepdims, DType work_dtype) {
  const auto nd = static_cast<std::int64_t>(a.ndim());
  std::vector<bool> red(static_cast<std::size_t>(nd), !axis.has_value());
  if (axis) {
    for (const auto ax : normalize_axes(*axis, nd)) red[static_cast<std::size_t>(ax)] = true;
  }
  std::vector<std::int64_t> perm;
  Shape out_shape;
  std::int64_t rows = 1;
  std::int64_t n = 1;
  for (std::int64_t d = 0; d < nd; ++d) {
    const auto dim = a.shape()[static_cast<std::size_t>(d)];
    if (red[static_cast<std::size_t>(d)]) {
      n *= dim;
      if (keepdims) out_shape.push_back(1);
    } else {
      perm.push_back(d);
      out_shape.push_back(dim);
      rows *= dim;
    }
  }
  for (std::int64_t d = 0; d < nd; ++d) {
    if (red[static_cast<std::size_t>(d)]) perm.push_back(d);
  }
  const NDArray t = nd == 0 ? a : transpose(a, perm);
  return {t.astype(work_dtype), std::move(out_shape), rows, n};
}

void reject_complex(DType dt) {
  if (is_complex(dt)) {
    throw_error(ErrorKind::NotImplemented, "reductions on complex arrays are not implemented");
  }
}

const char* op_name(ReduceOp op) {
  switch (op) {
    case ReduceOp::Sum: return "add";
    case ReduceOp::Prod: return "multiply";
    case ReduceOp::Min: return "minimum";
    case ReduceOp::Max: return "maximum";
    default: return "mean";
  }
}

// Sum/prod/min/max over each row of `w` (dtype W), written as dtype W.
template <typename W>
void fold_rows(ReduceOp op, const Work& w, const std::optional<double>& initial, NDArray& out) {
  using C = compute_t<W>;
  const std::byte* src = w.data.data();
  std::byte* dst = out.data();
  const std::size_t isz = sizeof(W);
  if (w.n == 0 && !initial && (op == ReduceOp::Min || op == ReduceOp::Max)) {
    throw_error(ErrorKind::Value, std::string("zero-size array to reduction operation ") +
                                      op_name(op) + " which has no identity");
  }
  for (std::int64_t r = 0; r < w.rows; ++r) {
    const std::byte* row = src + static_cast<std::size_t>(r * w.n) * isz;
    C acc{};
    std::int64_t start = 0;
    if (initial) {
      acc = cast_value<C>(*initial);
    } else if (op == ReduceOp::Sum) {
      acc = C{0};
    } else if (op == ReduceOp::Prod) {
      acc = C{1};
    } else {
      acc = ld<W>(row);
      start = 1;
    }
    for (std::int64_t i = start; i < w.n; ++i) {
      const C v = ld<W>(row + static_cast<std::size_t>(i) * isz);
      switch (op) {
        case ReduceOp::Sum: acc = kernels::add<C>(acc, v); break;
        case ReduceOp::Prod: acc = kernels::mul<C>(acc, v); break;
        case ReduceOp::Min:
          if (!is_nan(acc) && (is_nan(v) || v < acc || (v == acc && sign_bit(v)))) acc = v;
          break;
        case ReduceOp::Max:
          if (!is_nan(acc) && (is_nan(v) || v > acc || (v == acc && !sign_bit(v)))) acc = v;
          break;
        default: break;
      }
    }
    store<W>(dst + static_cast<std::size_t>(r) * isz, cast_value<W>(acc));
  }
}

// Mean/var/std in accumulator dtype W (float32 or float64).
template <typename W>
void moment_rows(ReduceOp op, const Work& w, std::int64_t ddof, NDArray& out) {
  const auto* src = reinterpret_cast<const W*>(w.data.data());
  std::byte* dst = out.data();
  const W count = static_cast<W>(w.n);
  for (std::int64_t r = 0; r < w.rows; ++r) {
    const W* row = src + r * w.n;
    W s = 0;
    for (std::int64_t i = 0; i < w.n; ++i) s += row[i];
    W res = s / count;  // empty -> NaN (D-017)
    if (op != ReduceOp::Mean) {
      const W mean = res;
      W ss = 0;
      for (std::int64_t i = 0; i < w.n; ++i) {
        const W d = row[i] - mean;
        ss += d * d;
      }
      const W denom = static_cast<W>(std::max<std::int64_t>(w.n - ddof, 0));
      res = ss / denom;
      if (op == ReduceOp::Std) res = std::sqrt(res);
    }
    store<W>(dst + static_cast<std::size_t>(r) * sizeof(W), res);
  }
}

}  // namespace

DType reduce_result_dtype(ReduceOp op, DType in) {
  reject_complex(in);
  const bool intlike = in == DType::Bool || is_integer(in);
  switch (op) {
    case ReduceOp::Sum:
    case ReduceOp::Prod:
      if (!intlike) return in;
      return dtype_info(in).kind == 'u' ? DType::UInt64 : DType::Int64;
    case ReduceOp::Min:
    case ReduceOp::Max: return in;
    default: return intlike ? DType::Float64 : in;
  }
}

NDArray reduce(ReduceOp op, const NDArray& a, const ReduceOptions& opts) {
  reject_complex(a.dtype());
  const DType rdt = opts.dtype.value_or(reduce_result_dtype(op, a.dtype()));
  reject_complex(rdt);
  if (op == ReduceOp::Min || op == ReduceOp::Max || op == ReduceOp::Sum ||
      op == ReduceOp::Prod) {
    const Work w = prepare(a, opts.axis, opts.keepdims, rdt);
    NDArray out = NDArray::empty(w.out_shape, rdt);
    dispatch_dtype(rdt, [&](auto tag) {
      using W = dtype_t<decltype(tag)::value>;
      if constexpr (!is_complex_v<W>) fold_rows<W>(op, w, opts.initial, out);
    });
    return out;
  }
  // mean/var/std: accumulate in float64, or float32 for float16/float32
  // results (NumPy keeps float32 and upcasts float16 to float32).
  const DType acc = (rdt == DType::Float16 || rdt == DType::Float32) ? DType::Float32
                                                                      : DType::Float64;
  const Work w = prepare(a, opts.axis, opts.keepdims, acc);
  NDArray tmp = NDArray::empty(w.out_shape, acc);
  if (acc == DType::Float32) moment_rows<float>(op, w, opts.ddof, tmp);
  else moment_rows<double>(op, w, opts.ddof, tmp);
  return tmp.dtype() == rdt ? tmp : tmp.astype(rdt);
}

NDArray arg_reduce(bool is_max, const NDArray& a, std::optional<std::int64_t> axis,
                   bool keepdims) {
  reject_complex(a.dtype());
  std::optional<std::vector<std::int64_t>> axes;
  if (axis) axes = std::vector<std::int64_t>{*axis};
  const Work w = prepare(a, axes, false, a.dtype());
  const char* name = is_max ? "argmax" : "argmin";
  if (w.n == 0) {
    throw_error(ErrorKind::Value, std::string("attempt to get ") + name + " of an empty sequence");
  }
  Shape out_shape = w.out_shape;
  if (keepdims) {
    if (axis) {
      out_shape.insert(out_shape.begin() + normalize_axis(*axis, static_cast<std::int64_t>(a.ndim())), 1);
    } else {
      out_shape.assign(a.ndim(), 1);
    }
  }
  NDArray out = NDArray::empty(out_shape, DType::Int64);
  auto* dst = reinterpret_cast<std::int64_t*>(out.data());
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (!is_complex_v<S>) {
      using C = compute_t<S>;
      const std::byte* src = w.data.data();
      for (std::int64_t r = 0; r < w.rows; ++r) {
        const std::byte* row = src + static_cast<std::size_t>(r * w.n) * sizeof(S);
        C best = ld<S>(row);
        std::int64_t bi = 0;
        for (std::int64_t i = 1; i < w.n && !is_nan(best); ++i) {
          const C v = ld<S>(row + static_cast<std::size_t>(i) * sizeof(S));
          if (is_nan(v) || (is_max ? v > best : v < best)) {
            best = v;
            bi = i;
          }
        }
        dst[r] = bi;
      }
    }
  });
  return out;
}

}  // namespace nativpy
