#include "p10_stats.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numeric>
#include <string>

#include "broadcast.hpp"
#include "error.hpp"
#include "p10_common.hpp"
#include "p10_cumdiff.hpp"
#include "reduce.hpp"
#include "shape_ops.hpp"
#include "ufunc.hpp"

namespace nativpy::p10 {

namespace {

using Axes = std::optional<std::vector<std::int64_t>>;

// A Python float combined with an array of dtype `other` (NEP 50 weak scalar).
NDArray weak(double v, DType other) { return scalar(v, is_inexact(other) ? other : DType::Float64); }

NDArray add(const NDArray& a, const NDArray& b) { return binary(BinaryOp::Add, a, b); }
NDArray sub(const NDArray& a, const NDArray& b) { return binary(BinaryOp::Subtract, a, b); }
NDArray mul(const NDArray& a, const NDArray& b) { return binary(BinaryOp::Multiply, a, b); }
NDArray div(const NDArray& a, const NDArray& b) { return binary(BinaryOp::Divide, a, b); }

DType real_dtype(DType dt) {
  if (dt == DType::Complex64) return DType::Float32;
  if (dt == DType::Complex128) return DType::Float64;
  return dt;
}

bool any_zero(const NDArray& a) {
  const NDArray c = a.copy();
  bool found = false;
  dispatch_dtype(c.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    for (std::int64_t i = 0; i < c.size() && !found; ++i) {
      const S v = at<S, S>(c, i);
      if constexpr (is_complex_v<S>) found = v.real() == 0 && v.imag() == 0;
      else found = cast_value<double>(v) == 0.0;
    }
  });
  return found;
}

// Sequential c[i, j] = sum_k x[i, k] * conj(y[j, k]) (NumPy dot(X, Y.T.conj())).
NDArray dot_conj(const NDArray& x_in, const NDArray& y_in) {
  const DType dt = promote_types(x_in.dtype(), y_in.dtype());
  const NDArray x = x_in.astype(dt);
  const NDArray y = y_in.astype(dt);
  const std::int64_t r = x.shape()[0];
  const std::int64_t n = x.shape()[1];
  const std::int64_t r2 = y.shape()[0];
  NDArray c = NDArray::empty({r, r2}, dt);
  dispatch_dtype(dt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    using C = compute_t<S>;
    for (std::int64_t i = 0; i < r; ++i) {
      for (std::int64_t j = 0; j < r2; ++j) {
        C acc{};
        for (std::int64_t k = 0; k < n; ++k) {
          const C a = at<C, S>(x, i * n + k);
          C b = at<C, S>(y, j * n + k);
          if constexpr (is_complex_v<C>) b = std::conj(b);
          if constexpr (std::is_same_v<C, bool>) acc = acc || (a && b);
          else acc = static_cast<C>(acc + a * b);
        }
        put<S>(c, i * r2 + j, acc);
      }
    }
  });
  return c;
}

// NumPy array(m, ndmin=2, dtype=dt) (a fresh C-contiguous copy).
NDArray as_2d(const NDArray& m, DType dt) {
  NDArray x = m.astype(dt);
  if (x.ndim() == 0) return x.reshape({1, 1});
  if (x.ndim() == 1) return x.reshape({1, x.shape()[0]});
  return x;
}

NDArray transposed_copy(const NDArray& x) { return transpose(x, {}).copy(); }

NDArray sample_weights(const NDArray& w_in, std::int64_t n, const char* what, bool integral) {
  const NDArray w = w_in.astype(DType::Float64);
  const std::string name(what);
  if (integral) {
    for (std::int64_t i = 0; i < w.size(); ++i) {
      const double v = w.get_double(i);
      if (v != std::nearbyint(v)) throw_error(ErrorKind::DType, name + " must be integer");
    }
  }
  if (w.ndim() > 1) throw_error(ErrorKind::Value, "cannot handle multidimensional " + name);
  if (w.ndim() == 0 || w.shape()[0] != n) {
    throw_error(ErrorKind::Value, "incompatible numbers of samples and " + name);
  }
  for (std::int64_t i = 0; i < w.size(); ++i) {
    if (w.get_double(i) < 0) throw_error(ErrorKind::Value, name + " cannot be negative");
  }
  return w;
}

[[noreturn]] void index_error(std::int64_t i, std::size_t ax, std::int64_t len) {
  throw_error(ErrorKind::Index, "index " + std::to_string(i) + " is out of bounds for axis " + std::to_string(ax) +
                                    " with size " + std::to_string(len));
}

}  // namespace

std::pair<NDArray, NDArray> average(const NDArray& a, const Axes& axis_in, const std::optional<NDArray>& weights,
                                    bool keepdims) {
  const auto nd = static_cast<std::int64_t>(a.ndim());
  Axes axis;
  if (axis_in) axis = normalize_axes(*axis_in, nd);
  ReduceOptions ro;
  ro.axis = axis;
  ro.keepdims = keepdims;
  if (!weights) {
    const NDArray avg = reduce(ReduceOp::Mean, a, ro);
    if (avg.size() == 0) throw_error(ErrorKind::Value, "division by zero");
    const double ratio = static_cast<double>(a.size()) / static_cast<double>(avg.size());
    NDArray scl = NDArray::empty(avg.shape(), avg.dtype());
    for (std::int64_t i = 0; i < scl.size(); ++i) scl.set_double(i, ratio);
    return {avg, scl};
  }
  NDArray wgt = *weights;
  if (wgt.shape() != a.shape()) {
    if (!axis) throw_error(ErrorKind::DType, "Axis must be specified when shapes of a and weights differ.");
    Shape want;
    for (const auto ax : *axis) want.push_back(a.shape()[static_cast<std::size_t>(ax)]);
    if (wgt.shape() != want) {
      throw_error(ErrorKind::Value, "Shape of weights must be consistent with shape of a along specified axis.");
    }
    std::vector<std::int64_t> order(axis->size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::int64_t i, std::int64_t j) { return (*axis)[static_cast<std::size_t>(i)] < (*axis)[static_cast<std::size_t>(j)]; });
    wgt = transpose(wgt, order);
    Shape s;
    for (std::int64_t d = 0; d < nd; ++d) {
      const bool in = std::find(axis->begin(), axis->end(), d) != axis->end();
      s.push_back(in ? a.shape()[static_cast<std::size_t>(d)] : 1);
    }
    wgt = wgt.reshape(s);
  }
  DType rdt = promote_types(a.dtype(), wgt.dtype());
  if (!is_inexact(a.dtype())) rdt = promote_types(rdt, DType::Float64);
  ReduceOptions so = ro;
  so.dtype = rdt;
  NDArray scl = reduce(ReduceOp::Sum, wgt, so);
  if (any_zero(scl)) throw_error(ErrorKind::Value, "Weights sum to zero, can't be normalized");
  UfuncParams p;
  p.dtype = rdt;
  const NDArray prod = binary(BinaryOp::Multiply, a, wgt, nullptr, p);
  const NDArray avg = div(reduce(ReduceOp::Sum, prod, ro), scl);
  if (scl.shape() != avg.shape()) scl = broadcast_to(scl, avg.shape()).copy();
  return {avg, scl};
}

NDArray cov(const NDArray& m, const CovOptions& o, bool* dof_warning) {
  if (dof_warning) *dof_warning = false;
  if (m.ndim() > 2) throw_error(ErrorKind::Value, "m has more than 2 dimensions");
  if (o.y && o.y->ndim() > 2) throw_error(ErrorKind::Value, "y has more than 2 dimensions");
  DType dt = DType::Float64;
  if (o.dtype) {
    dt = *o.dtype;
  } else {
    dt = promote_types(m.dtype(), DType::Float64);
    if (o.y) dt = promote_types(dt, o.y->dtype());
  }
  NDArray x = as_2d(m, dt);
  if (!o.rowvar && m.ndim() != 1) x = transposed_copy(x);
  if (x.shape()[0] == 0) return NDArray::empty({0, 0}, DType::Float64);
  if (o.y) {
    NDArray y = as_2d(*o.y, dt);
    if (!o.rowvar && y.shape()[0] != 1) y = transposed_copy(y);
    x = concat({x, y}, 0);
  }
  const std::int64_t ddof = o.ddof ? *o.ddof : (o.bias ? 0 : 1);
  const std::int64_t n = x.shape()[1];
  std::optional<NDArray> w;
  std::optional<NDArray> aw;
  if (o.fweights) w = sample_weights(*o.fweights, n, "fweights", true);
  if (o.aweights) {
    aw = sample_weights(*o.aweights, n, "aweights", false);
    w = w ? mul(*w, *aw) : *aw;
  }
  auto [avg, wsum_arr] = average(x, std::vector<std::int64_t>{1}, w, false);
  double fact = 0;
  if (!w) {
    fact = static_cast<double>(n - ddof);
  } else {
    const double wsum = wsum_arr.get_double(0);
    if (ddof == 0) {
      fact = wsum;
    } else if (!aw) {
      fact = wsum - static_cast<double>(ddof);
    } else {
      double s = 0;  // Python builtin sum
      for (std::int64_t i = 0; i < n; ++i) {
        const double wa = w->get_double(i) * aw->get_double(i);
        s += wa;
      }
      const double t = static_cast<double>(ddof) * s;
      fact = wsum - t / wsum;
    }
  }
  if (fact <= 0) {
    if (dof_warning) *dof_warning = true;
    fact = 0.0;
  }
  const NDArray centre = avg.reshape({avg.shape()[0], 1});
  x = binary(BinaryOp::Subtract, x, centre, x);
  const NDArray xt = w ? mul(x, w->reshape({1, n})) : x;
  NDArray c = dot_conj(x, xt);
  const NDArray inv = div(scalar(1.0, DType::Float64), scalar(fact, DType::Float64));
  c = binary(BinaryOp::Multiply, c, inv, c);
  return squeeze(c, std::nullopt);
}

NDArray corrcoef(const NDArray& xin, const std::optional<NDArray>& y, bool rowvar, std::optional<DType> dtype,
                 bool* dof_warning) {
  CovOptions o;
  o.y = y;
  o.rowvar = rowvar;
  o.dtype = dtype;
  NDArray c = cov(xin, o, dof_warning);
  if (c.ndim() != 2) return div(c, c);
  const std::int64_t k = c.shape()[0];
  NDArray d = NDArray::empty({k}, c.dtype());
  for (std::int64_t i = 0; i < k; ++i) copy_into(d.view({}, {}, d.offset() + i * static_cast<std::int64_t>(d.itemsize())),
                                                 c.view({}, {}, c.offset() + i * (c.strides()[0] + c.strides()[1])));
  const NDArray sd = unary(UnaryOp::Sqrt, complex_part(d, false).copy());
  c = binary(BinaryOp::Divide, c, sd.reshape({k, 1}), c);
  c = binary(BinaryOp::Divide, c, sd.reshape({1, k}), c);
  dispatch_dtype(c.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    using C = compute_t<S>;
    const auto clip = [](auto v) { return v < -1 ? decltype(v)(-1) : v > 1 ? decltype(v)(1) : v; };
    for (std::int64_t i = 0; i < c.size(); ++i) {
      const C v = at<C, S>(c, i);
      if constexpr (is_complex_v<C>) put<S>(c, i, C(clip(v.real()), clip(v.imag())));
      else if constexpr (std::is_floating_point_v<C>) put<S>(c, i, clip(v));
    }
  });
  return c;
}

std::vector<NDArray> gradient(const NDArray& f_in, const std::vector<GradSpacing>& spacing, const Axes& axes_in,
                              std::int64_t edge_order) {
  const auto nd = static_cast<std::int64_t>(f_in.ndim());
  std::vector<std::int64_t> axes;
  if (axes_in) {
    axes = normalize_axes(*axes_in, nd);
  } else {
    for (std::int64_t i = 0; i < nd; ++i) axes.push_back(i);
  }
  // Per axis: a scalar spacing (weak double or strong 0-d) or 1-d differences.
  struct Dx {
    bool weak = true;
    double value = 1.0;
    std::optional<NDArray> scalar0;  // strong 0-d
    std::optional<NDArray> diffs;    // non-uniform
  };
  std::vector<Dx> dx(axes.size());
  const auto scalar_dx = [](const GradSpacing& g) {
    Dx d;
    if (g.array) {
      d.weak = false;
      d.scalar0 = g.array->copy();
    } else {
      d.value = g.value;
    }
    return d;
  };
  if (spacing.size() == 1 && (!spacing[0].array || spacing[0].array->ndim() == 0)) {
    for (auto& d : dx) d = scalar_dx(spacing[0]);
  } else if (!spacing.empty()) {
    if (spacing.size() != axes.size()) throw_error(ErrorKind::DType, "invalid number of arguments");
    for (std::size_t i = 0; i < axes.size(); ++i) {
      const GradSpacing& g = spacing[i];
      if (!g.array || g.array->ndim() == 0) {
        dx[i] = scalar_dx(g);
        continue;
      }
      NDArray dist = *g.array;
      if (dist.ndim() != 1) throw_error(ErrorKind::Value, "distances must be either scalars or 1d");
      if (dist.shape()[0] != f_in.shape()[static_cast<std::size_t>(axes[i])]) {
        throw_error(ErrorKind::Value, "when 1d, distances must match the length of the corresponding dimension");
      }
      if (is_integer(dist.dtype())) dist = dist.astype(DType::Float64);
      const NDArray dd = diff(dist, 1, -1, std::nullopt, std::nullopt).copy();
      if (dd.size() == 0) index_error(0, 0, 0);
      bool uniform = true;
      dispatch_dtype(dd.dtype(), [&](auto tag) {
        using S = dtype_t<decltype(tag)::value>;
        using C = compute_t<S>;
        const C first = at<C, S>(dd, 0);
        for (std::int64_t k = 0; k < dd.size(); ++k) uniform = uniform && at<C, S>(dd, k) == first;
      });
      Dx d;
      d.weak = false;
      if (uniform) d.scalar0 = axis_slice(dd, 0, 0, 1).reshape({});
      else d.diffs = dd;
      dx[i] = d;
    }
  }
  if (edge_order > 2) throw_error(ErrorKind::Value, "'edge_order' greater than 2 not supported");
  NDArray f = f_in;
  DType otype = f.dtype();
  if (!is_inexact(otype)) {
    if (is_integer(otype)) f = f.astype(DType::Float64);
    otype = DType::Float64;
  }
  const DType rt = real_dtype(otype);
  std::vector<NDArray> outs;
  for (std::size_t i = 0; i < axes.size(); ++i) {
    const auto ax = static_cast<std::size_t>(axes[i]);
    const std::int64_t len = f.shape()[ax];
    if (len < edge_order + 1) {
      throw_error(ErrorKind::Value,
                  "Shape of array too small to calculate a numerical gradient, at least (edge_order + 1) elements "
                  "are required.");
    }
    const Dx& d = dx[i];
    NDArray out = NDArray::empty(f.shape(), otype);
    const auto sl = [&](std::int64_t start, std::int64_t n) { return axis_slice(f, ax, start, n); };
    const auto at_ = [&](std::int64_t k) { return axis_slice(f, ax, k < 0 ? len + k : k, 1); };
    const auto set = [&](std::int64_t start, std::int64_t n, const NDArray& v) {
      copy_into(axis_slice(out, ax, start, n), v);
    };
    // Coefficient `num / den(dx)` for a scalar spacing, as NumPy computes it.
    const auto coef = [&](double num, double mult) {  // num / (mult * dx)
      if (d.weak) return scalar(num / (mult * d.value), rt);
      const NDArray s = *d.scalar0;
      const NDArray den = mult == 1.0 ? s : mul(weak(mult, s.dtype()), s);
      return div(weak(num, den.dtype()), den);
    };
    Shape cshape(f.ndim(), 1);
    const auto col = [&](const NDArray& v) {
      Shape s = cshape;
      s[ax] = v.size();
      return v.reshape(s);
    };
    if (len > 2) {
      if (!d.diffs) {
        // (f[2:] - f[:-2]) / (2. * dx)
        const NDArray num = sub(sl(2, len - 2), sl(0, len - 2));
        const NDArray den = d.weak ? scalar(2.0 * d.value, rt) : mul(weak(2.0, d.scalar0->dtype()), *d.scalar0);
        set(1, len - 2, div(num, den));
      } else {
        const NDArray& h = *d.diffs;
        const std::int64_t m = h.size();
        const NDArray dx1 = axis_slice(h, 0, 0, m - 1);
        const NDArray dx2 = axis_slice(h, 0, 1, m - 1);
        const NDArray s12 = add(dx1, dx2);
        const NDArray a = div(unary(UnaryOp::Negative, dx2), mul(dx1, s12));
        const NDArray b = div(sub(dx2, dx1), mul(dx1, dx2));
        const NDArray c = div(dx1, mul(dx2, s12));
        set(1, len - 2,
            add(add(mul(col(a), sl(0, len - 2)), mul(col(b), sl(1, len - 2))), mul(col(c), sl(2, len - 2))));
      }
    }
    if (edge_order == 1) {
      const auto step = [&](bool first) {
        if (d.diffs) return axis_slice(*d.diffs, 0, first ? 0 : d.diffs->size() - 1, 1).reshape({});
        return d.weak ? scalar(d.value, rt) : *d.scalar0;
      };
      set(0, 1, div(sub(at_(1), at_(0)), step(true)));
      set(len - 1, 1, div(sub(at_(-1), at_(-2)), step(false)));
    } else {
      if (len < 3) index_error(len < 2 ? 1 : 2, ax, len);
      const auto edge = [&](bool first) -> std::vector<NDArray> {
        if (!d.diffs) {
          if (first) return {coef(-1.5, 1.0), coef(2.0, 1.0), coef(-0.5, 1.0)};
          return {coef(0.5, 1.0), coef(-2.0, 1.0), coef(1.5, 1.0)};
        }
        const std::int64_t m = d.diffs->size();
        const NDArray dx1 = axis_slice(*d.diffs, 0, first ? 0 : m - 2, 1).reshape({});
        const NDArray dx2 = axis_slice(*d.diffs, 0, first ? 1 : m - 1, 1).reshape({});
        const NDArray s12 = add(dx1, dx2);
        if (first) {
          return {div(unary(UnaryOp::Negative, add(mul(weak(2.0, dx1.dtype()), dx1), dx2)), mul(dx1, s12)),
                  div(s12, mul(dx1, dx2)), div(unary(UnaryOp::Negative, dx1), mul(dx2, s12))};
        }
        return {div(dx2, mul(dx1, s12)), div(unary(UnaryOp::Negative, s12), mul(dx1, dx2)),
                div(add(mul(weak(2.0, dx2.dtype()), dx2), dx1), mul(dx2, s12))};
      };
      const auto lo = edge(true);
      set(0, 1, add(add(mul(lo[0], at_(0)), mul(lo[1], at_(1))), mul(lo[2], at_(2))));
      const auto hi = edge(false);
      set(len - 1, 1, add(add(mul(hi[0], at_(-3)), mul(hi[1], at_(-2))), mul(hi[2], at_(-1))));
    }
    outs.push_back(out);
  }
  return outs;
}

NDArray trapezoid(const NDArray& y, const std::optional<NDArray>& x, double dx, std::int64_t axis) {
  if (y.ndim() == 0) throw_error(ErrorKind::Index, "too many indices for array: array is 0-dimensional, but 1 were indexed");
  const auto nd = static_cast<std::int64_t>(y.ndim());
  const auto ax = static_cast<std::size_t>(normalize_axis(axis, nd));
  const std::int64_t len = y.shape()[ax];
  const std::int64_t m = len > 0 ? len - 1 : 0;
  const NDArray mid = add(axis_slice(y, ax, std::min<std::int64_t>(1, len), m), axis_slice(y, ax, 0, m));
  const auto scaled = [&]() {
    if (!x) return mul(weak(dx, mid.dtype()), mid);
    if (x->ndim() == 1) {
      const NDArray d = diff(*x, 1, -1, std::nullopt, std::nullopt);
      Shape s(y.ndim(), 1);
      s[ax] = d.shape()[0];
      return mul(d.reshape(s), mid);
    }
    return mul(diff(*x, 1, axis, std::nullopt, std::nullopt), mid);
  };
  const NDArray s0 = scaled();
  const NDArray t = div(s0, weak(2.0, s0.dtype()));
  ReduceOptions ro;
  ro.axis = std::vector<std::int64_t>{axis};
  return reduce(ReduceOp::Sum, t, ro);
}

}  // namespace nativpy::p10
