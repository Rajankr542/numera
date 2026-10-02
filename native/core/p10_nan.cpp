#include "p10_nan.hpp"

#include <cmath>
#include <limits>
#include <string>

#include "error.hpp"
#include "fp_errors.hpp"
#include "p10_common.hpp"
#include "ufunc.hpp"

namespace nativpy::p10 {

namespace {

bool any_nan(const NDArray& a) {
  bool found = false;
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    for_each_element(a, [&](const std::byte* p) {
      const S v = load<S>(p);
      if constexpr (is_complex_v<S>) found = found || std::isnan(v.real()) || std::isnan(v.imag());
      else found = found || is_nan(cast_value<compute_t<S>>(v));
    });
  });
  return found;
}

// dst[i] = v where mask[i] (same shape, C-contiguous dst).
void fill_where(NDArray& dst, const NDArray& mask, double v) {
  const NDArray m = mask.astype(DType::Bool);
  for (std::int64_t i = 0; i < dst.size(); ++i) {
    if (at<bool, bool>(m, i)) dst.set_double(i, v);
  }
}

// Bool mask of NaN positions (either component for complex).
NDArray nan_mask(const NDArray& a) {
  const NDArray c = a.copy();
  NDArray m = NDArray::empty(a.shape(), DType::Bool);
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    for (std::int64_t i = 0; i < c.size(); ++i) {
      const S v = at<S, S>(c, i);
      bool nan = false;
      if constexpr (is_complex_v<S>) nan = std::isnan(v.real()) || std::isnan(v.imag());
      else nan = is_nan(cast_value<compute_t<S>>(v));
      put<bool>(m, i, nan);
    }
  });
  return m;
}

// np.errstate(invalid='ignore', divide='ignore') for the guard's lifetime.
class IgnoreDivide {
 public:
  IgnoreDivide() noexcept : saved_(get_errstate()) {
    ErrState s = saved_;
    s.divide = FpMode::Ignore;
    s.invalid = FpMode::Ignore;
    set_errstate(s);
  }
  ~IgnoreDivide() { set_errstate(saved_); }
  IgnoreDivide(const IgnoreDivide&) = delete;
  IgnoreDivide& operator=(const IgnoreDivide&) = delete;

 private:
  ErrState saved_;
};

// NumPy _divide_by_count: a / b cast (unsafe) back to a's dtype, FP errors ignored.
NDArray divide_by_count(const NDArray& a, const NDArray& cnt) {
  const IgnoreDivide guard;
  const NDArray q = binary(BinaryOp::Divide, a, cnt);
  return q.dtype() == a.dtype() ? q : q.astype(a.dtype());
}

// fmin/fmax reduction: NaNs skipped, all-NaN (or empty along a non-empty
// result) slices give NaN. Real -0.0 < +0.0; complex compares lexicographically.
NDArray nan_minmax(bool is_max, const NDArray& a, const ReduceOptions& opts) {
  const Rows r = to_rows(a, opts.axis, opts.keepdims);
  if (r.n == 0 && r.rows > 0 && !opts.initial) {
    throw_error(ErrorKind::Value, std::string("zero-size array to reduction operation ") +
                                      (is_max ? "fmax" : "fmin") + " which has no identity");
  }
  NDArray out = NDArray::empty(r.out_shape, a.dtype());
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (is_complex_v<S>) {
      using V = typename S::value_type;
      for (std::int64_t row = 0; row < r.rows; ++row) {
        bool have = false;
        S best{};
        for (std::int64_t i = 0; i < r.n; ++i) {
          const S v = at<S, S>(r.data, row * r.n + i);
          if (std::isnan(v.real()) || std::isnan(v.imag())) continue;
          const bool better = is_max ? (v.real() > best.real() || (v.real() == best.real() && v.imag() > best.imag()))
                                     : (v.real() < best.real() || (v.real() == best.real() && v.imag() < best.imag()));
          if (!have || better) best = v;
          have = true;
        }
        if (!have) best = S(std::numeric_limits<V>::quiet_NaN(), V{0});
        put<S>(out, row, best);
      }
    } else {
      using C = compute_t<S>;
      for (std::int64_t row = 0; row < r.rows; ++row) {
        bool have = opts.initial.has_value();
        C best = have ? static_cast<C>(*opts.initial) : C{};
        for (std::int64_t i = 0; i < r.n; ++i) {
          const C v = at<C, S>(r.data, row * r.n + i);
          if (is_nan(v)) continue;
          bool better = is_max ? v > best : v < best;
          if constexpr (std::is_floating_point_v<C>) {
            if (v == best && std::signbit(v) != std::signbit(best)) better = is_max ? !std::signbit(v) : std::signbit(v);
          }
          if (!have || better) best = v;
          have = true;
        }
        if constexpr (std::is_floating_point_v<C>) {
          if (!have) best = std::numeric_limits<C>::quiet_NaN();
        }
        put<S>(out, row, best);
      }
    }
  });
  return out;
}

}  // namespace

NDArray nan_reduce(ReduceOp op, const NDArray& a, const ReduceOptions& opts) {
  const DType dt = a.dtype();
  if (op == ReduceOp::Min || op == ReduceOp::Max) {
    if (dtype_info(dt).kind != 'f' && !is_complex(dt)) return reduce(op, a, opts);
    if (opts.initial && is_complex(dt)) {
      throw_error(ErrorKind::NotImplemented, "complex nanmin/nanmax with initial= is not implemented");
    }
    return nan_minmax(op == ReduceOp::Max, a, opts);
  }
  if (op == ReduceOp::Sum || op == ReduceOp::Prod) {
    return reduce(op, replace_nan(a, op == ReduceOp::Prod ? 1.0 : 0.0), opts);
  }
  // mean / var / std: NumPy uses the plain reduction when nothing can be NaN.
  if (!is_inexact(dt) || !any_nan(a)) return reduce(op, a, opts);
  if (opts.dtype && !is_inexact(*opts.dtype)) {
    throw_error(ErrorKind::DType, "If a is inexact, then dtype must be inexact");
  }
  ReduceOptions sum_opts;
  sum_opts.axis = opts.axis;
  sum_opts.dtype = opts.dtype;
  NDArray arr = replace_nan(a, 0.0);
  if (op == ReduceOp::Mean) {
    sum_opts.keepdims = opts.keepdims;
    const NDArray cnt = count_not_nan(a, opts.axis, opts.keepdims);
    return divide_by_count(reduce(ReduceOp::Sum, arr, sum_opts), cnt);
  }
  sum_opts.keepdims = true;
  const NDArray mask = nan_mask(a);
  const NDArray cnt_k = count_not_nan(a, opts.axis, true);
  const NDArray avg = divide_by_count(reduce(ReduceOp::Sum, arr, sum_opts), cnt_k);
  NDArray dev = binary(BinaryOp::Subtract, arr, avg);
  if (dev.dtype() != arr.dtype()) dev = dev.astype(arr.dtype());  // NumPy out=arr, casting unsafe
  if (!dev.is_c_contiguous()) dev = dev.copy();
  fill_where(dev, mask, 0.0);
  NDArray sqr = is_complex(dev.dtype()) ? complex_part(binary(BinaryOp::Multiply, dev, unary(UnaryOp::Conjugate, dev)), false)
                                        : binary(BinaryOp::Multiply, dev, dev);
  sum_opts.keepdims = opts.keepdims;
  NDArray var = reduce(ReduceOp::Sum, sqr, sum_opts);
  const NDArray cnt = count_not_nan(a, opts.axis, opts.keepdims);
  NDArray dof = cnt.copy();
  for (std::int64_t i = 0; i < dof.size(); ++i) dof.set_int64(i, cnt.get_int64(i) - opts.ddof);
  var = divide_by_count(var, dof);
  if (!var.is_c_contiguous() || !var.owns_data()) var = var.copy();
  for (std::int64_t i = 0; i < dof.size(); ++i) {
    if (dof.get_int64(i) <= 0) var.set_double(i, std::nan(""));
  }
  if (op == ReduceOp::Std) var = unary(UnaryOp::Sqrt, var);
  return var;
}

NDArray nan_arg_reduce(bool is_max, const NDArray& a, std::optional<std::int64_t> axis, bool keepdims) {
  if (!is_inexact(a.dtype())) return arg_reduce(is_max, a, axis, keepdims);
  const double inf = std::numeric_limits<double>::infinity();
  const NDArray r = arg_reduce(is_max, replace_nan(a, is_max ? -inf : inf), axis, keepdims);
  std::optional<std::vector<std::int64_t>> axes;
  if (axis) axes = std::vector<std::int64_t>{*axis};
  const NDArray cnt = count_not_nan(a, axes, false);
  for (std::int64_t i = 0; i < cnt.size(); ++i) {
    if (cnt.get_int64(i) == 0) throw_error(ErrorKind::Value, "All-NaN slice encountered");
  }
  return r;
}

}  // namespace nativpy::p10
