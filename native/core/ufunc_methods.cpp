#include "ufunc_methods.hpp"

#include <array>
#include <string>
#include <string_view>

#include "broadcast.hpp"
#include "error.hpp"
#include "reduce.hpp"
#include "shape.hpp"
#include "shape_ops.hpp"

namespace nativpy {

namespace {

void require_binary(const Ufunc& u, const char* method) {
  if (u.nin != 2) {
    throw_error(ErrorKind::Value, std::string(method) + " only supported for binary functions");
  }
}

void check_out_writeable(const NDArray* out) {
  if (out && !out->writeable()) throw_error(ErrorKind::Value, "output array is read-only");
}

// Loop dtype (D-052): dtype= wins, then out's dtype, then the sum/prod rule
// for add/multiply, then the ufunc's own resolver.
DType loop_dtype(const Ufunc& u, const NDArray& a, const NDArray* out,
                 const std::optional<DType>& dtype) {
  if (dtype) return u.resolve_dtype(a.dtype(), *dtype).out;
  if (out) return u.resolve(out->dtype(), a.dtype()).out;
  const std::string_view name = u.name;
  if (name == "add" || name == "multiply") return reduce_result_dtype(ReduceOp::Sum, a.dtype());
  return u.resolve(a.dtype(), a.dtype()).out;
}

BinaryLoopFn loop_for(const Ufunc& u, DType dt) {
  const BinaryLoopFn fn = u.binary_loops[static_cast<std::size_t>(dt)];
  if (!fn) {
    throw_error(ErrorKind::DType,
                std::string("no ") + u.name + " loop for dtype " + std::string(dtype_name(dt)));
  }
  return fn;
}

// dst = op(x, y) over dst's shape (x, y broadcast). dst may alias x.
void run_loop(BinaryLoopFn fn, const NDArray& dst, const NDArray& x, const NDArray& y) {
  fn(make_plan<3>(dst.shape(), {&dst, &x, &y}), {dst.data(), x.data(), y.data()});
}

void fill_value(const NDArray& dst, double v) {
  NDArray s = NDArray::empty({}, DType::Float64);
  s.set_double(0, v);
  copy_into(dst, s);
}

// The input cast to the loop dtype. Results are built in a fresh buffer and
// copied into `out` last, so an input overlapping `out` needs no copy.
NDArray as_loop_dtype(const NDArray& a, DType dt) { return a.dtype() == dt ? a : a.astype(dt); }

// View of `a` with axis `ax` restricted to [start, start + len).
NDArray axis_slice(const NDArray& a, std::size_t ax, std::int64_t start, std::int64_t len) {
  Shape shape = a.shape();
  shape[ax] = len;
  return a.view(shape, a.strides(), a.offset() + start * a.strides()[ax]);
}

// Same buffer as `acc` (keepdims shape), but shaped like the full input with
// stride 0 on every reduced axis, so the strided loop accumulates in place.
NDArray reduction_view(const NDArray& acc, const Shape& full, const std::vector<bool>& red) {
  Strides st = acc.strides();
  for (std::size_t d = 0; d < full.size(); ++d) {
    if (red[d]) st[d] = 0;
  }
  return acc.view(full, st, acc.offset());
}

}  // namespace

NDArray ufunc_reduce(const Ufunc& u, const NDArray& a_in, const NDArray* out,
                     const UfuncReduceOptions& opts) {
  require_binary(u, "reduce");
  check_out_writeable(out);
  const auto nd = static_cast<std::int64_t>(a_in.ndim());
  const auto und = static_cast<std::size_t>(nd);

  // Reduced axes.
  std::vector<bool> red(und, false);
  std::size_t n_axes = 0;
  if (opts.all_axes) {
    red.assign(und, true);
    n_axes = und;
  } else {
    const std::vector<std::int64_t> axes =
        opts.axis ? *opts.axis : (nd == 0 ? std::vector<std::int64_t>{} : std::vector<std::int64_t>{0});
    for (const auto ax : normalize_axes(axes, nd)) red[static_cast<std::size_t>(ax)] = true;
    n_axes = axes.size();
  }
  if (n_axes > 1 && !u.identity) {
    throw_error(ErrorKind::Value, std::string("reduction operation '") + u.name +
                                      "' is not reorderable, so at most one axis may be specified");
  }

  const DType dt = loop_dtype(u, a_in, out, opts.dtype);
  const BinaryLoopFn fn = loop_for(u, dt);

  // where= mask (bool, broadcast to the input).
  std::optional<NDArray> mask;
  if (opts.where) {
    if (opts.where->dtype() != DType::Bool) {
      throw_error(ErrorKind::DType, "Cannot cast array data from " +
                                        std::string(dtype_name(opts.where->dtype())) +
                                        " to bool according to the rule 'safe'");
    }
    if (!u.identity && !opts.initial) {
      throw_error(ErrorKind::Value, std::string("reduction operation '") + u.name +
                                        "' does not have an identity, so to use a where mask one "
                                        "has to specify 'initial'");
    }
    mask = broadcast_to(*opts.where, a_in.shape());
  }

  // Result shapes: `kept` has size-1 reduced axes; `res` applies keepdims.
  Shape kept;
  Shape res;
  std::int64_t n = 1;
  for (std::size_t d = 0; d < und; ++d) {
    const auto dim = a_in.shape()[d];
    if (red[d]) {
      n *= dim;
      kept.push_back(1);
      if (opts.keepdims) res.push_back(1);
    } else {
      kept.push_back(dim);
      res.push_back(dim);
    }
  }
  if (out && out->shape() != res) {
    throw_error(ErrorKind::Value, std::string("output parameter for reduction operation ") +
                                      u.name + " has the wrong shape: expected " +
                                      shape_to_string(res) + ", got " +
                                      shape_to_string(out->shape()));
  }
  const std::optional<double> start = opts.initial ? opts.initial : u.identity;
  if (n == 0 && shape_size(kept) > 0 && !start) {
    throw_error(ErrorKind::Value, std::string("zero-size array to reduction operation ") +
                                      u.name + " which has no identity");
  }

  const NDArray a = as_loop_dtype(a_in, dt);
  NDArray acc = NDArray::empty(kept, dt);

  if (acc.size() > 0) {
    if (mask) {
      // Masked: one step per reduced position; only true positions update acc.
      fill_value(acc, *start);
      std::vector<std::size_t> rdims;
      for (std::size_t d = 0; d < und; ++d) {
        if (red[d]) rdims.push_back(d);
      }
      NDArray tmp = NDArray::empty(kept, dt);
      std::vector<std::int64_t> idx(rdims.size(), 0);
      for (std::int64_t step = 0; step < n; ++step) {
        NDArray s = a;
        NDArray m = *mask;
        for (std::size_t k = 0; k < rdims.size(); ++k) {
          s = axis_slice(s, rdims[k], idx[k], 1);
          m = axis_slice(m, rdims[k], idx[k], 1);
        }
        if (u.check) u.check(acc, &s, &m, kept);
        run_loop(fn, tmp, acc, s);
        masked_copy_into(acc, tmp, m);
        for (std::size_t k = rdims.size(); k-- > 0;) {
          if (++idx[k] < a.shape()[rdims[k]]) break;
          idx[k] = 0;
        }
      }
    } else if (n_axes == 0 && !opts.initial) {
      copy_into(acc, a);  // nothing to reduce: the cast input
    } else if (n_axes == 1 && !opts.initial && n > 0) {
      // NumPy: the first element along the axis seeds the accumulator.
      std::size_t ax = 0;
      while (!red[ax]) ++ax;
      copy_into(acc, axis_slice(a, ax, 0, 1));
      const std::int64_t len = a.shape()[ax];
      if (len > 1) {
        const NDArray rest = axis_slice(a, ax, 1, len - 1);
        if (u.check) u.check(rest, &rest, nullptr, rest.shape());
        const NDArray view = reduction_view(acc, rest.shape(), red);
        run_loop(fn, view, view, rest);
      }
    } else {
      fill_value(acc, *start);
      if (n > 0) {
        if (u.check) u.check(a, &a, nullptr, a.shape());
        const NDArray view = reduction_view(acc, a.shape(), red);
        run_loop(fn, view, view, a);
      }
    }
  }

  const NDArray result = opts.keepdims ? acc : acc.reshape(res);
  if (!out) return result;
  copy_into(*out, result);
  return *out;
}

NDArray ufunc_accumulate(const Ufunc& u, const NDArray& a_in, const NDArray* out,
                         const UfuncReduceOptions& opts) {
  require_binary(u, "accumulate");
  check_out_writeable(out);
  if (a_in.ndim() == 0) throw_error(ErrorKind::DType, "cannot accumulate on a scalar");
  if (opts.all_axes || (opts.axis && opts.axis->size() != 1)) {
    throw_error(ErrorKind::Value, "accumulate does not allow multiple axes");
  }
  const auto nd = static_cast<std::int64_t>(a_in.ndim());
  const auto ax = static_cast<std::size_t>(normalize_axis(opts.axis ? opts.axis->front() : 0, nd));

  const DType dt = loop_dtype(u, a_in, out, opts.dtype);
  const BinaryLoopFn fn = loop_for(u, dt);
  if (out && out->shape() != a_in.shape()) {
    throw_error(ErrorKind::Value, "provided out is the wrong size for the accumulation.");
  }

  const NDArray a = as_loop_dtype(a_in, dt);
  NDArray r = NDArray::empty(a.shape(), dt);
  const std::int64_t len = a.shape()[ax];
  if (r.size() > 0) {
    copy_into(axis_slice(r, ax, 0, 1), axis_slice(a, ax, 0, 1));
    if (len > 1) {
      // r[k] = op(r[k-1], a[k]); C-order iteration writes r[k-1] before r[k].
      const NDArray rest = axis_slice(a, ax, 1, len - 1);
      if (u.check) u.check(rest, &rest, nullptr, rest.shape());
      run_loop(fn, axis_slice(r, ax, 1, len - 1), axis_slice(r, ax, 0, len - 1), rest);
    }
  }
  if (!out) return r;
  copy_into(*out, r);
  return *out;
}

}  // namespace nativpy
