#include "p10_reduce.hpp"

#include <cstdint>
#include <string>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"
#include "p10_common.hpp"
#include "reduce.hpp"
#include "shape.hpp"
#include "shape_ops.hpp"
#include "ufunc.hpp"

namespace nativpy::p10 {

namespace {

// Return C-contiguous copy of `a` with masked-out positions replaced by `fill`.
NDArray apply_mask(const NDArray& a, const NDArray& mask_bool, double fill) {
  NDArray r = a.copy();
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    for (std::int64_t i = 0; i < r.size(); ++i) {
      if (!at<bool, bool>(mask_bool, i))
        put<S>(r, i, fill);
    }
  });
  return r;
}

// Count of true values per reduced slice (result shape, keepdims applied).
NDArray masked_count(const NDArray& mask_bool,
                     const std::optional<std::vector<std::int64_t>>& axis,
                     bool keepdims) {
  const Rows r = to_rows(mask_bool, axis, keepdims, DType::Bool);
  NDArray out = NDArray::zeros(r.out_shape, DType::Int64);
  for (std::int64_t row = 0; row < r.rows; ++row) {
    std::int64_t c = 0;
    for (std::int64_t i = 0; i < r.n; ++i) {
      if (at<bool, bool>(r.data, row * r.n + i)) ++c;
    }
    put<std::int64_t>(out, row, c);
  }
  return out;
}

// Build a ReduceOptions from the where-opts fields (no where/out).
ReduceOptions base_opts(const ReduceWhereOptions& opts) {
  ReduceOptions ro;
  ro.axis     = opts.axis;
  ro.keepdims = opts.keepdims;
  ro.dtype    = opts.dtype;
  ro.initial  = opts.initial;
  ro.ddof     = opts.ddof;
  return ro;
}

// Copy `result` into `*opts.out` (shape-checked, unsafe cast) and return it.
NDArray write_out(NDArray result, const ReduceWhereOptions& opts) {
  if (!opts.out) return result;
  const NDArray& out = *opts.out;
  if (!out.writeable())
    throw_error(ErrorKind::Value, "output array is read-only");
  if (out.shape() != result.shape()) {
    throw_error(ErrorKind::Value,
                "output parameter shape mismatch: expected " +
                    shape_to_string(result.shape()) + ", got " +
                    shape_to_string(out.shape()));
  }
  copy_into(out, result);
  return out;
}

}  // namespace

NDArray reduce_where(ReduceOp op, const NDArray& a, const ReduceWhereOptions& opts) {
  // Fast path: no where= and no out= — delegate directly to the core kernel.
  if (!opts.where && !opts.out)
    return reduce(op, a, base_opts(opts));

  // ---- validate where= and build masked input ----
  std::optional<NDArray> mask_bc;
  NDArray input = a;  // replaced below when where= is given

  if (opts.where) {
    const NDArray& raw = *opts.where;
    if (raw.dtype() != DType::Bool) {
      throw_error(ErrorKind::DType,
                  "where must be a boolean array, not " +
                      std::string(dtype_name(raw.dtype())));
    }

    // min/max require initial= when a mask is present (NumPy rule, D-136).
    if ((op == ReduceOp::Min || op == ReduceOp::Max) && !opts.initial) {
      throw_error(ErrorKind::Value,
                  std::string("reduction operation '") +
                      (op == ReduceOp::Min ? "minimum" : "maximum") +
                      "' does not have an identity, so to use a where mask one "
                      "has to specify 'initial'");
    }

    // Broadcast both to a common shape.
    const Shape bshape = broadcast_shapes({a.shape(), raw.shape()});
    const NDArray a_bc = broadcast_to(a, bshape);
    mask_bc = broadcast_to(raw, bshape);

    // Replace masked-out positions with the reduction identity:
    //   sum / mean / var / std -> 0
    //   prod                   -> 1
    //   min / max              -> *opts.initial (validated above)
    double fill = 0.0;
    if (op == ReduceOp::Prod)
      fill = 1.0;
    else if ((op == ReduceOp::Min || op == ReduceOp::Max) && opts.initial)
      fill = *opts.initial;

    const NDArray a_flat =
        a_bc.is_c_contiguous() ? a_bc.copy() : a_bc.astype(a_bc.dtype());
    input = apply_mask(a_flat, *mask_bc, fill);
  }

  // ---- mean / var / std with where= : divide by masked count ----
  if (opts.where && (op == ReduceOp::Mean || op == ReduceOp::Var || op == ReduceOp::Std)) {
    const DType rdt    = opts.dtype.value_or(reduce_result_dtype(op, a.dtype()));
    const DType acc_dt = (rdt == DType::Float16 || rdt == DType::Float32)
                             ? DType::Float32
                             : DType::Float64;

    const NDArray cnt   = masked_count(*mask_bc, opts.axis, opts.keepdims);
    const NDArray cnt_f = cnt.astype(acc_dt);

    ReduceOptions sum_ro;
    sum_ro.axis     = opts.axis;
    sum_ro.keepdims = opts.keepdims;
    sum_ro.dtype    = acc_dt;

    if (op == ReduceOp::Mean) {
      NDArray s = reduce(ReduceOp::Sum, input, sum_ro);
      NDArray q = binary(BinaryOp::Divide, s, cnt_f);
      NDArray r = q.dtype() == rdt ? std::move(q) : q.astype(rdt);
      return write_out(std::move(r), opts);
    }

    // var / std: mean -> deviations -> masked zero-fill -> sum-of-squares / denom.
    NDArray s    = reduce(ReduceOp::Sum, input, sum_ro);
    NDArray mean = binary(BinaryOp::Divide, s, cnt_f);

    const NDArray input_acc = input.astype(acc_dt);
    const NDArray mean_bc   = broadcast_to(mean, input_acc.shape());
    NDArray dev = binary(BinaryOp::Subtract, input_acc, mean_bc);

    // Zero masked-off deviations so they don't contribute to the variance.
    dev = apply_mask(dev, *mask_bc, 0.0);

    NDArray sq = binary(BinaryOp::Multiply, dev, dev);
    NDArray ss = reduce(ReduceOp::Sum, sq, sum_ro);

    // denom = clamp(count - ddof, 0, ∞).
    NDArray denom = cnt_f.copy();
    for (std::int64_t i = 0; i < denom.size(); ++i) {
      const double v = denom.get_double(i) - static_cast<double>(opts.ddof);
      denom.set_double(i, v < 0.0 ? 0.0 : v);
    }

    NDArray var_arr = binary(BinaryOp::Divide, ss, denom);
    if (op == ReduceOp::Std) var_arr = unary(UnaryOp::Sqrt, var_arr);
    NDArray r = var_arr.dtype() == rdt ? std::move(var_arr) : var_arr.astype(rdt);
    return write_out(std::move(r), opts);
  }

  // ---- sum / prod / min / max (with or without where=) ----
  // input is already masked; pass initial for min/max seeding.
  ReduceOptions ro = base_opts(opts);
  if (op != ReduceOp::Min && op != ReduceOp::Max) ro.initial = std::nullopt;
  NDArray result = reduce(op, input, ro);
  return write_out(std::move(result), opts);
}

}  // namespace nativpy::p10
