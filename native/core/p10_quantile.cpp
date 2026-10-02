#include "p10_quantile.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>

#include "broadcast.hpp"
#include "error.hpp"
#include "p10_common.hpp"
#include "shape_ops.hpp"
#include "ufunc_kernels.hpp"

namespace nativpy::p10 {

namespace {

constexpr std::array<std::string_view, 13> kMethodNames = {
    "inverted_cdf", "averaged_inverted_cdf", "closest_observation", "interpolated_inverted_cdf",
    "hazen",        "weibull",               "linear",              "median_unbiased",
    "normal_unbiased", "lower",              "higher",              "midpoint",
    "nearest"};

bool is_discrete(QMethod m) noexcept {
  return m == QMethod::InvertedCdf || m == QMethod::ClosestObservation || m == QMethod::Lower ||
         m == QMethod::Higher || m == QMethod::Nearest;
}

bool is_float(DType dt) noexcept { return dtype_info(dt).kind == 'f'; }

// Python's floor-modulo by 2 of an integral double.
bool is_odd(double v) noexcept {
  const double m = std::fmod(v, 2.0);
  return m != 0.0;
}

// Where the quantile falls in a sorted sample of n values (NumPy
// _QuantileMethods + _get_indexes + _get_gamma). G is the dtype in which
// NumPy computes the virtual index.
struct Pick {
  std::int64_t lo = 0;  // discrete methods use only lo
  std::int64_t hi = 0;
  double gamma = 0;
};

Pick pick(QMethod m, std::int64_t n, double q, DType g) {
  const auto rg = [g](double v) { return round_to(g, v); };
  const auto nd = static_cast<double>(n);
  const auto clampi = [n](double v) {
    const auto i = static_cast<std::int64_t>(v);
    return std::clamp<std::int64_t>(i, 0, n - 1);
  };
  const auto boundaries = [&](double index, bool odd_rule) {
    const double prev = std::floor(index);
    const double gamma = rg(index - prev);
    const bool use_prev = gamma == 0 && (!odd_rule || is_odd(prev));
    double res = use_prev ? prev : rg(prev + 1);
    if (res < 0) res = 0;
    return Pick{clampi(res), 0, 0};
  };
  switch (m) {
    case QMethod::InvertedCdf: return boundaries(rg(rg(nd * q) - 1), false);
    case QMethod::ClosestObservation: return boundaries(rg(rg(rg(nd * q) - 1) - 0.5), true);
    case QMethod::Lower: return Pick{clampi(std::floor(rg((nd - 1) * q))), 0, 0};
    case QMethod::Higher: return Pick{clampi(std::ceil(rg((nd - 1) * q))), 0, 0};
    case QMethod::Nearest: return Pick{clampi(std::nearbyint(rg((nd - 1) * q))), 0, 0};
    default: break;
  }
  double vi = 0;
  const auto cvi = [&](double alpha, double beta) {
    // alpha and 1 - alpha - beta are Python floats: weakly cast to G first.
    const double c = rg(1 - alpha - beta);
    return rg(rg(rg(nd * q) + rg(rg(alpha) + rg(q * c))) - 1);
  };
  switch (m) {
    case QMethod::AveragedInvertedCdf: vi = rg(rg(nd * q) - 1); break;
    case QMethod::InterpolatedInvertedCdf: vi = cvi(0, 1); break;
    case QMethod::Hazen: vi = cvi(0.5, 0.5); break;
    case QMethod::Weibull: vi = cvi(0, 0); break;
    case QMethod::Linear: vi = rg((nd - 1) * q); break;
    case QMethod::MedianUnbiased: vi = cvi(1 / 3.0, 1 / 3.0); break;
    case QMethod::NormalUnbiased: vi = cvi(3 / 8.0, 3 / 8.0); break;
    case QMethod::Midpoint: {
      const double x = rg((nd - 1) * q);
      vi = rg(0.5 * rg(std::floor(x) + std::ceil(x)));
      break;
    }
    default: break;
  }
  // _get_indexes: -1 (the last element) above the range, 0 below it.
  double prev = std::floor(vi);
  double next = prev + 1;
  if (vi >= nd - 1) prev = next = -1;
  if (vi < 0) prev = next = 0;
  if (std::isnan(vi)) prev = next = -1;
  double gamma = rg(vi - prev);
  if (m == QMethod::AveragedInvertedCdf) gamma = gamma == 0 ? 0.5 : 1.0;
  if (m == QMethod::Midpoint) gamma = std::fmod(vi, 1.0) == 0 ? 0.0 : 0.5;
  const auto idx = [n](double v) { return v < 0 ? n - 1 : static_cast<std::int64_t>(v); };
  return Pick{idx(prev), idx(next), gamma};
}

template <typename S>
double as_double(S v) noexcept {
  return cast_value<double>(v);
}

// Per-row worker for one input dtype S.
template <typename S>
struct QuantileRun {
  using C = compute_t<S>;
  const Rows& rows;
  const std::vector<double>& qs;  // q values in [0, 1], rounded to G
  const QuantileOptions& opts;
  DType a_dt, g_dt, r_dt;
  bool discrete;
  bool int_index;  // integer q array with an integer virtual index: element, cast to r_dt
  const Rows* weights;
  NDArray& out;

  void store_result(std::int64_t qi, std::int64_t r, double v) const {
    const std::int64_t flat = qi * rows.rows + r;
    dispatch_dtype(r_dt, [&](auto tag) {
      using R = dtype_t<decltype(tag)::value>;
      if constexpr (!is_complex_v<R>) put<R>(out, flat, v);
    });
  }
  void store_value(std::int64_t qi, std::int64_t r, S v) const {
    store<S>(out.data() + static_cast<std::size_t>(qi * rows.rows + r) * sizeof(S), v);
  }

  // NumPy _lerp in the result dtype; b - a is taken in the input dtype.
  double lerp(S sa, S sb, double t) const {
    const auto rr = [this](double v) { return round_to(r_dt, v); };
    const C a = cast_value<C>(sa);
    const C b = cast_value<C>(sb);
    double diff = 0;
    if constexpr (std::is_same_v<C, bool>) diff = a != b;  // unreachable: bool is rejected
    else diff = rr(round_to(a_dt, as_double(kernels::sub<C>(b, a))));
    const double ad = rr(as_double(a));
    const double bd = rr(as_double(b));
    const double tt = rr(t);
    double res = rr(ad + rr(diff * tt));
    if (t >= 0.5) {
      const double om = opts.weak_q ? rr(1 - t) : rr(round_to(g_dt, 1 - t));
      res = rr(bd - rr(diff * om));
    }
    return res;
  }

  void run() const {
    const auto nq = static_cast<std::int64_t>(qs.size());
    std::vector<S> buf;
    buf.reserve(static_cast<std::size_t>(rows.n));
    std::vector<double> wbuf;
    const auto* base = reinterpret_cast<const std::byte*>(rows.data.data());
    for (std::int64_t r = 0; r < rows.rows; ++r) {
      buf.clear();
      wbuf.clear();
      bool has_nan = false;
      for (std::int64_t i = 0; i < rows.n; ++i) {
        const S v = load<S>(base + static_cast<std::size_t>(r * rows.n + i) * sizeof(S));
        const bool nan = is_nan(cast_value<C>(v));
        has_nan = has_nan || nan;
        if (nan && opts.ignore_nan) continue;
        buf.push_back(v);
        if (weights) wbuf.push_back(at<double, double>(weights->data, r * rows.n + i));
      }
      const auto n = static_cast<std::int64_t>(buf.size());
      if (n == 0 || (has_nan && !opts.ignore_nan)) {
        if (n == 0 && !opts.ignore_nan) {
          throw_error(ErrorKind::Index, "index -1 is out of bounds for axis 0 with size 0");
        }
        for (std::int64_t qi = 0; qi < nq; ++qi) store_result(qi, r, std::nan(""));
        continue;
      }
      const auto less = [](S x, S y) { return nan_last_less(cast_value<C>(x), cast_value<C>(y)); };
      if (weights) {
        run_weighted(r, buf, wbuf, less);
        continue;
      }
      // Selection: nth_element per needed index, or a full sort for many q.
      const bool sort_all = nq > 4;
      if (sort_all) std::sort(buf.begin(), buf.end(), less);
      const auto kth = [&](std::int64_t k) {
        if (!sort_all) std::nth_element(buf.begin(), buf.begin() + k, buf.end(), less);
        return buf[static_cast<std::size_t>(k)];
      };
      for (std::int64_t qi = 0; qi < nq; ++qi) {
        const Pick p = pick(opts.method, n, qs[static_cast<std::size_t>(qi)], g_dt);
        if (discrete) {
          store_value(qi, r, kth(p.lo));
        } else if (int_index) {
          const S v = kth(p.lo);
          dispatch_dtype(r_dt, [&](auto tag) {
            using R = dtype_t<decltype(tag)::value>;
            if constexpr (!is_complex_v<R>) put<R>(out, qi * rows.rows + r, v);
          });
        } else {
          const S lo = kth(p.lo);
          const S hi = kth(p.hi);
          store_result(qi, r, lerp(lo, hi, p.gamma));
        }
      }
    }
  }

  template <typename Less>
  void run_weighted(std::int64_t r, const std::vector<S>& buf, const std::vector<double>& w,
                    const Less& less) const {
    const auto n = static_cast<std::int64_t>(buf.size());
    std::vector<std::size_t> order(buf.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t x, std::size_t y) { return less(buf[x], buf[y]); });
    std::vector<double> cdf(buf.size());
    double acc = 0;
    for (std::size_t i = 0; i < order.size(); ++i) {
      acc += w[order[i]];
      cdf[i] = acc;
    }
    const double total = cdf.back();
    for (auto& c : cdf) c /= total;
    if (std::isnan(cdf.back())) throw_error(ErrorKind::Value, "Weights included NaN, inf or were all zero.");
    for (auto& c : cdf) {
      c = round_to(g_dt, c);
      if (c == 0) c = -1;
    }
    for (std::size_t qi = 0; qi < qs.size(); ++qi) {
      const auto it = std::lower_bound(cdf.begin(), cdf.end(), qs[qi]);
      const auto k = std::min<std::int64_t>(it - cdf.begin(), n - 1);
      store_value(static_cast<std::int64_t>(qi), r, buf[order[static_cast<std::size_t>(k)]]);
    }
  }
};

// Weights broadcast to a.shape (NumPy _weights_are_valid), as float64 rows.
Rows weight_rows(const NDArray& a, const NDArray& w, const QuantileOptions& opts) {
  NDArray wb = w;
  if (w.shape() != a.shape()) {
    if (!opts.axis) {
      throw_error(ErrorKind::DType, "Axis must be specified when shapes of a and weights differ.");
    }
    const auto axes = normalize_axes(*opts.axis, static_cast<std::int64_t>(a.ndim()));
    Shape want;
    for (const auto ax : axes) want.push_back(a.shape()[static_cast<std::size_t>(ax)]);
    if (w.shape() != want) {
      throw_error(ErrorKind::Value,
                  "Shape of weights must be consistent with shape of a along specified axis.");
    }
    std::vector<std::int64_t> perm(axes.size());
    std::iota(perm.begin(), perm.end(), std::int64_t{0});
    std::sort(perm.begin(), perm.end(), [&](std::int64_t x, std::int64_t y) {
      return axes[static_cast<std::size_t>(x)] < axes[static_cast<std::size_t>(y)];
    });
    const NDArray t = perm.empty() ? w : transpose(w, perm);
    Shape s;
    for (std::size_t d = 0; d < a.ndim(); ++d) {
      s.push_back(std::find(axes.begin(), axes.end(), static_cast<std::int64_t>(d)) != axes.end()
                      ? a.shape()[d]
                      : 1);
    }
    wb = broadcast_to(t.reshape(s), a.shape());
  }
  if (is_complex(wb.dtype())) throw_error(ErrorKind::DType, "weights must be real");
  Rows wr = to_rows(wb, opts.axis, opts.keepdims, DType::Float64);
  for (std::int64_t i = 0; i < wr.data.size(); ++i) {
    if (at<double, double>(wr.data, i) < 0) throw_error(ErrorKind::Value, "Weights must be non-negative.");
  }
  return wr;
}

NDArray nan_filled(const Shape& shape, DType dt) {
  NDArray out = NDArray::empty(shape, dt);
  for (std::int64_t i = 0; i < out.size(); ++i) out.set_double(i, std::nan(""));
  return out;
}

DType mean_dtype(DType a) { return (a == DType::Bool || is_integer(a)) ? DType::Float64 : a; }

}  // namespace

QMethod qmethod_from_name(std::string_view name) {
  for (std::size_t i = 0; i < kMethodNames.size(); ++i) {
    if (kMethodNames[i] == name) return static_cast<QMethod>(i);
  }
  std::string list;
  for (const auto m : kMethodNames) list += (list.empty() ? "'" : ", '") + std::string(m) + "'";
  throw_error(ErrorKind::Value, "'" + std::string(name) + "' is not a valid method. Use one of: dict_keys([" +
                                    list + "])");
}

NDArray quantile(const NDArray& a, const NDArray& q, const QuantileOptions& opts) {
  const DType adt = a.dtype();
  if (is_complex(adt)) throw_error(ErrorKind::DType, "a must be an array of real numbers");
  if (is_complex(q.dtype())) throw_error(ErrorKind::DType, "q must be real");
  const DType g = (!opts.weak_q && is_float(q.dtype())) ? q.dtype() : DType::Float64;
  // An integer q array keeps integer virtual indexes (NumPy): `linear` takes
  // the element; averaged/interpolated_inverted_cdf and weibull truncate gamma
  // to an int64 0 (result dtype result_type(a, int64)).
  // (A weak JS integer q behaves like a Python int: `linear` takes the element.)
  const bool int_q = !opts.percentile && !is_float(q.dtype());
  const bool int_index = int_q && !opts.weak_q && (opts.method == QMethod::AveragedInvertedCdf ||
                                   opts.method == QMethod::InterpolatedInvertedCdf ||
                                   opts.method == QMethod::Weibull);
  std::vector<double> qs(static_cast<std::size_t>(q.size()));
  const NDArray qc = q.astype(DType::Float64);
  for (std::int64_t i = 0; i < q.size(); ++i) {
    double v = qc.get_double(i);
    if (opts.percentile) v = round_to(g, v / 100);
    if (!(v >= 0.0 && v <= 1.0)) {
      throw_error(ErrorKind::Value, opts.percentile ? "Percentiles must be in the range [0, 100]"
                                                    : "Quantiles must be in the range [0, 1]");
    }
    qs[static_cast<std::size_t>(i)] = v;
  }
  if (opts.weights && opts.method != QMethod::InvertedCdf) {
    throw_error(ErrorKind::Value, "Only method 'inverted_cdf' supports weights. Got: " +
                                      std::string(kMethodNames[static_cast<std::size_t>(opts.method)]) + ".");
  }
  std::optional<Rows> wrows;
  if (opts.weights) wrows = weight_rows(a, *opts.weights, opts);
  if (q.ndim() > 2) throw_error(ErrorKind::Value, "q must be a scalar or 1d");
  const bool discrete = opts.weights || is_discrete(opts.method) || (opts.method == QMethod::Linear && int_q);
  DType rdt = adt;
  if (!discrete) {
    if (adt == DType::Bool) {
      throw_error(ErrorKind::DType,
                  "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, "
                  "the `^` operator, or the logical_xor function instead.");
    }
    if (opts.weak_q) rdt = is_float(adt) ? adt : DType::Float64;
    else if (int_index) rdt = promote_types(adt, DType::Int64);
    else rdt = promote_types(adt, g);
  }
  if (opts.ignore_nan && a.size() == 0) {  // NumPy falls back to nanmean (no q dims)
    const Rows r = to_rows(a, opts.axis, opts.keepdims);
    return nan_filled(r.out_shape, mean_dtype(adt));
  }
  const Rows rows = to_rows(a, opts.axis, opts.keepdims);
  Shape shape = q.shape();
  shape.insert(shape.end(), rows.out_shape.begin(), rows.out_shape.end());
  NDArray out = NDArray::empty(shape, rdt);
  if (out.size() == 0) return out;
  dispatch_dtype(adt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (!is_complex_v<S>) {
      QuantileRun<S>{rows, qs, opts, adt, g, rdt, discrete, int_index, wrows ? &*wrows : nullptr, out}.run();
    }
  });
  return out;
}

NDArray median(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis, bool keepdims,
               bool ignore_nan) {
  const DType adt = a.dtype();
  if (is_complex(adt)) throw_error(ErrorKind::NotImplemented, "median of complex input is not implemented");
  const DType rdt = mean_dtype(adt);
  const Rows rows = to_rows(a, axis, keepdims);
  NDArray out = NDArray::empty(rows.out_shape, rdt);
  if (out.size() == 0) return out;
  dispatch_dtype(adt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (!is_complex_v<S>) {
      using C = compute_t<S>;
      // NumPy takes np.mean of the middle element(s): float32 accumulation
      // for float16/float32, float64 otherwise; pairwise seed -0.0, plus 0.
      using W = std::conditional_t<std::is_same_v<C, float>, float, double>;
      std::vector<C> buf;
      for (std::int64_t r = 0; r < rows.rows; ++r) {
        buf.clear();
        bool has_nan = false;
        for (std::int64_t i = 0; i < rows.n; ++i) {
          const C v = at<C, S>(rows.data, r * rows.n + i);
          const bool nan = is_nan(v);
          has_nan = has_nan || nan;
          if (!(nan && ignore_nan)) buf.push_back(v);
        }
        const auto n = static_cast<std::int64_t>(buf.size());
        double res = std::nan("");
        if (n > 0 && !(has_nan && !ignore_nan)) {
          const auto less = [](C x, C y) { return nan_last_less(x, y); };
          const auto h = static_cast<std::size_t>(n / 2);
          std::nth_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(h), buf.end(), less);
          const W hi = cast_value<W>(buf[h]);
          W s = W(-0.0);
          if (n % 2 == 0) {
            const W lo = cast_value<W>(*std::max_element(buf.begin(), buf.begin() + static_cast<std::ptrdiff_t>(h), less));
            s += lo;
            s += hi;
            s = W(0) + s;
            s /= W(2);
          } else {
            s += hi;
            s = W(0) + s;
          }
          res = static_cast<double>(s);
        }
        dispatch_dtype(rdt, [&](auto rtag) {
          using R = dtype_t<decltype(rtag)::value>;
          if constexpr (!is_complex_v<R>) put<R>(out, r, res);
        });
      }
    }
  });
  return out;
}

}  // namespace nativpy::p10
