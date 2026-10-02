#include "p10_cumdiff.hpp"

#include <string>

#include "broadcast.hpp"
#include "error.hpp"
#include "p10_common.hpp"
#include "reduce.hpp"
#include "shape_ops.hpp"
#include "ufunc.hpp"
#include "ufunc_methods.hpp"
#include "ufunc_registry.hpp"

namespace nativpy::p10 {

namespace {

[[noreturn]] void bool_subtract_error() {
  throw_error(ErrorKind::DType,
              "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, the `^` "
              "operator, or the logical_xor function instead.");
}

}  // namespace

NDArray concat(const std::vector<NDArray>& parts, std::int64_t axis) {
  DType dt = parts.front().dtype();
  for (const auto& p : parts) dt = promote_types(dt, p.dtype());
  const auto nd = static_cast<std::int64_t>(parts.front().ndim());
  const auto ax = static_cast<std::size_t>(normalize_axis(axis, nd));
  Shape shape = parts.front().shape();
  shape[ax] = 0;
  for (const auto& p : parts) {
    if (static_cast<std::int64_t>(p.ndim()) != nd) {
      throw_error(ErrorKind::Value, "all the input arrays must have same number of dimensions");
    }
    for (std::size_t d = 0; d < p.ndim(); ++d) {
      if (d != ax && p.shape()[d] != parts.front().shape()[d]) {
        throw_error(ErrorKind::Value, "all the input array dimensions except for the concatenation axis must match exactly");
      }
    }
    shape[ax] += p.shape()[ax];
  }
  NDArray out = NDArray::empty(shape, dt);
  std::int64_t at_ = 0;
  for (const auto& p : parts) {
    const std::int64_t len = p.shape()[ax];
    if (len > 0 && out.size() > 0) copy_into(axis_slice(out, ax, at_, len), p);
    at_ += len;
  }
  return out;
}

NDArray cumulative(bool prod, const NDArray& a_in, const CumulativeOptions& opts, const NDArray* out) {
  NDArray a = a_in;
  std::int64_t axis = 0;
  if (opts.axis) {
    axis = *opts.axis;
    if (a.ndim() == 0) a = a.reshape({1});
  } else {
    if (opts.array_api && a.ndim() > 1) {
      throw_error(ErrorKind::Value,
                  "For arrays which have more than one dimension ``axis`` argument is required.");
    }
    a = ravel(a);
  }
  if (opts.skip_nan) a = replace_nan(a, prod ? 1.0 : 0.0);
  const Ufunc& u = get(prod ? BinaryOp::Multiply : BinaryOp::Add);
  UfuncReduceOptions uo;
  uo.axis = std::vector<std::int64_t>{axis};
  uo.dtype = opts.dtype;
  if (!opts.include_initial) return ufunc_accumulate(u, a, out, uo);
  NDArray r = ufunc_accumulate(u, a, nullptr, uo);
  const auto ax = static_cast<std::size_t>(normalize_axis(axis, static_cast<std::int64_t>(a.ndim())));
  Shape s = r.shape();
  s[ax] = 1;
  NDArray init = NDArray::empty(s, r.dtype());
  for (std::int64_t i = 0; i < init.size(); ++i) init.set_double(i, prod ? 1.0 : 0.0);
  NDArray res = concat({init, r}, static_cast<std::int64_t>(ax));
  if (!out) return res;
  out->check_writeable();
  if (out->shape() != res.shape()) {
    throw_error(ErrorKind::Value, "provided out is the wrong size for the accumulation.");
  }
  copy_into(*out, res);
  return *out;
}

NDArray diff(const NDArray& a_in, std::int64_t n, std::int64_t axis, const std::optional<NDArray>& prepend,
             const std::optional<NDArray>& append) {
  if (n == 0) return a_in;
  if (n < 0) throw_error(ErrorKind::Value, "order must be non-negative but got " + std::to_string(n));
  if (a_in.ndim() == 0) throw_error(ErrorKind::Value, "diff requires input that is at least one dimensional");
  const auto nd = static_cast<std::int64_t>(a_in.ndim());
  const auto ax = static_cast<std::size_t>(normalize_axis(axis, nd));
  NDArray a = a_in;
  if (prepend || append) {
    std::vector<NDArray> parts;
    const auto edge = [&](const NDArray& v) {
      if (v.ndim() > 0) return v;
      Shape s = a_in.shape();
      s[ax] = 1;
      return broadcast_to(v, s);
    };
    if (prepend) parts.push_back(edge(*prepend));
    parts.push_back(a_in);
    if (append) parts.push_back(edge(*append));
    a = concat(parts, static_cast<std::int64_t>(ax));
  }
  for (std::int64_t k = 0; k < n; ++k) {
    const std::int64_t len = a.shape()[ax];
    if (len == 0) break;
    const NDArray hi = axis_slice(a, ax, 1, len - 1);
    const NDArray lo = axis_slice(a, ax, 0, len - 1);
    if (a.dtype() == DType::Bool) {  // NumPy uses not_equal for bool
      NDArray r = NDArray::empty(hi.shape(), DType::Bool);
      const NDArray hc = hi.copy();
      const NDArray lc = lo.copy();
      for (std::int64_t i = 0; i < r.size(); ++i) {
        put<bool>(r, i, at<bool, bool>(hc, i) != at<bool, bool>(lc, i));
      }
      a = r;
    } else {
      a = binary(BinaryOp::Subtract, hi, lo);
    }
  }
  return a;
}

NDArray ptp(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis, bool keepdims) {
  if (a.dtype() == DType::Bool) bool_subtract_error();
  ReduceOptions o;
  o.axis = axis;
  o.keepdims = keepdims;
  const NDArray hi = reduce(ReduceOp::Max, a, o);
  const NDArray lo = reduce(ReduceOp::Min, a, o);
  return binary(BinaryOp::Subtract, hi, lo);
}

}  // namespace nativpy::p10
