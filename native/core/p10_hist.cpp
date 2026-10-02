// histogram, histogram_bin_edges, histogramdd, bincount, digitize, interp
// (D-134, D-138), following numpy/lib/_histograms_impl.py and
// numpy/_core/src/multiarray/compiled_base.c.

#include "p10_hist.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"
#include "p10_common.hpp"
#include "p10_cumdiff.hpp"
#include "shape_ops.hpp"

namespace nativpy::p10 {

namespace {

// ---- bin-edge estimators (NumPy _histograms_impl.py) ----

// Freedman-Diaconis estimator: 2 * IQR / n^(1/3).
double fd_width(const std::vector<double>& sorted, double /*range_width*/) {
  const auto n = static_cast<std::int64_t>(sorted.size());
  if (n == 0) return 0;
  const double q1 = sorted[static_cast<std::size_t>(std::max<std::int64_t>(0, (n - 1) / 4))];
  const double q3 = sorted[static_cast<std::size_t>(std::min<std::int64_t>(n - 1, (3 * (n - 1)) / 4 + 1))];
  const double iqr = q3 - q1;
  if (iqr == 0) return 0;
  return 2.0 * iqr * std::pow(static_cast<double>(n), -1.0 / 3.0);
}

// Sturges: ceil(log2(n) + 1).
std::int64_t sturges_count(std::int64_t n) {
  return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(std::log2(static_cast<double>(n)) + 1)));
}

// Scott's rule: 3.5 * std * n^(-1/3).
double scott_width(const std::vector<double>& vals_in, double /*range_width*/) {
  const auto n = static_cast<std::int64_t>(vals_in.size());
  if (n < 2) return 0;
  double mean = 0;
  for (double v : vals_in) mean += v;
  mean /= static_cast<double>(n);
  double var = 0;
  for (double v : vals_in) var += (v - mean) * (v - mean);
  var /= static_cast<double>(n - 1);
  const double std_dev = std::sqrt(var);
  return 3.5 * std_dev * std::pow(static_cast<double>(n), -1.0 / 3.0);
}

// Rice: 2 * n^(1/3).
std::int64_t rice_count(std::int64_t n) {
  return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(2.0 * std::pow(static_cast<double>(n), 1.0 / 3.0))));
}

// Doane: Sturges extended for non-normal data.
std::int64_t doane_count(const std::vector<double>& vals_in) {
  const auto n = static_cast<std::int64_t>(vals_in.size());
  if (n < 3) return 1;
  double mean = 0;
  for (double v : vals_in) mean += v;
  mean /= static_cast<double>(n);
  double m3 = 0, m2 = 0;
  for (double v : vals_in) {
    const double d = v - mean;
    m2 += d * d;
    m3 += d * d * d;
  }
  m2 /= static_cast<double>(n);
  m3 /= static_cast<double>(n);
  if (m2 <= 0) return sturges_count(n);
  const double sg1 = m3 / std::pow(m2, 1.5);
  const double sg1_se = std::sqrt(6.0 * static_cast<double>(n - 2) / (static_cast<double>(n + 1) * static_cast<double>(n + 3)));
  const double result = std::log2(static_cast<double>(n)) + 1 + std::log2(1 + std::fabs(sg1) / sg1_se);
  return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(result)));
}

// Extract finite sorted values from an NDArray as float64.
std::vector<double> sorted_finite(const NDArray& a) {
  std::vector<double> v;
  v.reserve(static_cast<std::size_t>(a.size()));
  for (std::int64_t i = 0; i < a.size(); ++i) {
    const double d = a.get_double(i);
    if (std::isfinite(d)) v.push_back(d);
  }
  std::sort(v.begin(), v.end());
  return v;
}

// Compute number of bins from an estimator name and the data.
std::int64_t estimate_bins(const std::string& name, const NDArray& a, double range_width,
                           std::vector<std::string>& warnings) {
  const auto n = a.size();
  if (n == 0) return 1;
  const std::vector<double> sv = sorted_finite(a);
  if (sv.empty()) {
    warnings.push_back("Input array has no finite values. Number of bins set to 1.");
    return 1;
  }
  if (name == "auto") {
    // max(fd, sturges)
    const double fw = fd_width(sv, range_width);
    const std::int64_t fd_bins = fw > 0 ? static_cast<std::int64_t>(std::ceil(range_width / fw)) : 0;
    const std::int64_t st_bins = sturges_count(n);
    return std::max(fd_bins, st_bins);
  }
  if (name == "fd") {
    const double fw = fd_width(sv, range_width);
    if (fw <= 0) {
      warnings.push_back("The bin width is too small. Bins are set to 1.");
      return 1;
    }
    return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(range_width / fw)));
  }
  if (name == "sturges") return sturges_count(n);
  if (name == "scott") {
    const double sw = scott_width(sv, range_width);
    if (sw <= 0) {
      warnings.push_back("The bin width is too small. Bins are set to 1.");
      return 1;
    }
    return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(range_width / sw)));
  }
  if (name == "rice") return rice_count(n);
  if (name == "doane") return doane_count(sv);
  if (name == "sqrt") return std::max<std::int64_t>(1, static_cast<std::int64_t>(std::ceil(std::sqrt(static_cast<double>(n)))));
  throw_error(ErrorKind::Value, "'" + name + "' is not a valid estimator for `bins`");
}

// Build uniformly-spaced edges for [rmin, rmax] with `nbins` bins.
NDArray uniform_edges(double rmin, double rmax, std::int64_t nbins) {
  NDArray edges = NDArray::empty({nbins + 1}, DType::Float64);
  for (std::int64_t i = 0; i <= nbins; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(nbins);
    edges.set_double(i, std::fma(t, rmax - rmin, rmin));
  }
  // Force exact endpoints.
  edges.set_double(0, rmin);
  edges.set_double(nbins, rmax);
  return edges;
}

// Determine the [rmin, rmax] range from `a` and the optional hint.
// Returns {rmin, rmax} or throws on empty/NaN input.
std::pair<double, double> determine_range(const NDArray& a, const Range& range,
                                          std::vector<std::string>& warnings) {
  double rmin, rmax;
  if (range) {
    rmin = range->first;
    rmax = range->second;
    if (!(rmin <= rmax))
      throw_error(ErrorKind::Value, "max must be larger than min in range parameter.");
    if (!std::isfinite(rmin) || !std::isfinite(rmax))
      throw_error(ErrorKind::Value, "supplied range of [{min}, {max}] is not finite");
  } else {
    // NumPy: compute from the data (ignoring NaN/Inf per NumPy).
    if (a.size() == 0) {
      warnings.push_back("Input array is empty.");
      return {0.0, 1.0};
    }
    rmin = std::numeric_limits<double>::infinity();
    rmax = -std::numeric_limits<double>::infinity();
    bool has_nan = false;
    for (std::int64_t i = 0; i < a.size(); ++i) {
      const double v = a.get_double(i);
      if (std::isnan(v)) { has_nan = true; continue; }
      if (v < rmin) rmin = v;
      if (v > rmax) rmax = v;
    }
    if (has_nan) warnings.push_back("Input array contains NaN values.");
    if (!std::isfinite(rmin)) {
      // all NaN
      warnings.push_back("Input array is empty.");
      return {0.0, 1.0};
    }
    if (!std::isfinite(rmin) || !std::isfinite(rmax))
      throw_error(ErrorKind::Value, "autodetected range of [{min}, {max}] is not finite");
  }
  if (rmin == rmax) {
    rmin -= 0.5;
    rmax += 0.5;
  }
  return {rmin, rmax};
}

// ---- 1-D histogram bin search ----

// Returns the bin index (0-based) for value v given edges, closed on right
// for the last bin (NumPy logic).  Returns -1 if v is outside [edges[0], edges[-1]].
std::int64_t find_bin(double v, const std::vector<double>& edges) {
  const std::int64_t n = static_cast<std::int64_t>(edges.size()) - 1;
  if (n <= 0) return -1;
  if (v < edges[0] || v > edges[static_cast<std::size_t>(n)]) return -1;
  // Last bin is closed on the right.
  if (v == edges[static_cast<std::size_t>(n)]) return n - 1;
  // Binary search for the correct bin.
  const auto it = std::upper_bound(edges.begin(), edges.end(), v);
  const std::int64_t idx = static_cast<std::int64_t>(it - edges.begin()) - 1;
  return std::clamp<std::int64_t>(idx, 0, n - 1);
}

}  // namespace

// ---- histogram / histogram_bin_edges ----

HistResult histogram(const NDArray& a, const HistBins& bins, const Range& range, bool density,
                     const std::optional<NDArray>& weights, bool edges_only,
                     std::vector<std::string>& warnings) {
  // Flatten and cast to float64.
  const NDArray flat = a.reshape({a.size()}).astype(DType::Float64);

  // Determine range.
  auto [rmin, rmax] = determine_range(flat, range, warnings);

  // Determine edges.
  NDArray edges = NDArray::empty({0}, DType::Float64);  // placeholder
  if (bins.edges) {
    edges = bins.edges->astype(DType::Float64);
    if (edges.ndim() != 1 || edges.size() < 2)
      throw_error(ErrorKind::Value, "`bins` must be 1d and have at least two values");
    // Validate monotonically increasing.
    for (std::int64_t i = 0; i < edges.size() - 1; ++i) {
      if (edges.get_double(i) >= edges.get_double(i + 1))
        throw_error(ErrorKind::Value, "`bins` must be monotonically increasing, when an array");
    }
  } else {
    std::int64_t nbins;
    if (bins.count) {
      nbins = *bins.count;
      if (nbins <= 0) throw_error(ErrorKind::Value, "`bins` must be positive");
    } else {
      // Estimator name.
      const double range_width = rmax - rmin;
      nbins = estimate_bins(*bins.estimator, flat, range_width, warnings);
    }
    edges = uniform_edges(rmin, rmax, nbins);
  }

  if (edges_only) return {NDArray::empty({0}, DType::Float64), edges};

  const std::int64_t nbins = edges.size() - 1;

  // Build vector of edges for fast lookup.
  std::vector<double> edge_vals(static_cast<std::size_t>(edges.size()));
  for (std::int64_t i = 0; i < edges.size(); ++i)
    edge_vals[static_cast<std::size_t>(i)] = edges.get_double(i);

  // Choose hist dtype: float64 if density or weights, else float64 (NumPy
  // returns float64 when density=True, int64 when no weights, float64 with weights).
  const bool use_float = density || weights.has_value();
  NDArray hist = NDArray::zeros({nbins}, use_float ? DType::Float64 : DType::Int64);

  // Accumulate counts / weighted counts.
  for (std::int64_t i = 0; i < flat.size(); ++i) {
    const double v = flat.get_double(i);
    if (std::isnan(v)) continue;
    const std::int64_t bin = find_bin(v, edge_vals);
    if (bin < 0) continue;
    if (weights) {
      const double old = hist.get_double(bin);
      hist.set_double(bin, old + weights->get_double(i % weights->size()));
    } else {
      hist.set_int64(bin, hist.get_int64(bin) + 1);
    }
  }

  if (density) {
    // hist / (n * bin_width) — normalise so integral == 1.
    double total = 0;
    for (std::int64_t i = 0; i < hist.size(); ++i) total += hist.get_double(i);
    if (total != 0) {
      for (std::int64_t i = 0; i < hist.size(); ++i) {
        const double width = edge_vals[static_cast<std::size_t>(i + 1)] - edge_vals[static_cast<std::size_t>(i)];
        hist.set_double(i, hist.get_double(i) / (total * width));
      }
    }
  }

  return {hist, edges};
}

// ---- histogramdd ----

HistDdResult histogramdd(const std::vector<NDArray>& columns, const std::vector<HistBins>& bins_in,
                         const std::vector<Range>& range_in, bool density,
                         const std::optional<NDArray>& weights) {
  const auto D = columns.size();
  if (D == 0) throw_error(ErrorKind::Value, "histogramdd: sample must have at least one variable");

  const std::int64_t N = columns[0].size();
  for (std::size_t d = 1; d < D; ++d) {
    if (columns[d].size() != N)
      throw_error(ErrorKind::Value, "histogramdd: all columns must have the same length");
  }

  // ---- build per-axis edges ----
  std::vector<NDArray> all_edges;
  all_edges.reserve(D);
  // Initialise with placeholder arrays (will be replaced below).
  for (std::size_t d = 0; d < D; ++d)
    all_edges.push_back(NDArray::empty({0}, DType::Float64));
  std::vector<std::vector<double>> edge_vals(D);

  for (std::size_t d = 0; d < D; ++d) {
    const NDArray& col = columns[d];
    const NDArray flat = col.reshape({col.size()}).astype(DType::Float64);
    const HistBins& b = bins_in.size() == 1 ? bins_in[0] : bins_in[d];
    const Range& r = range_in.empty() ? Range{} : range_in[d];

    std::vector<std::string> tmp_warns;
    auto [rmin, rmax] = determine_range(flat, r, tmp_warns);

    if (b.edges) {
      all_edges[d] = b.edges->astype(DType::Float64);
    } else {
      std::int64_t nbins;
      if (b.count) {
        nbins = *b.count;
        if (nbins <= 0) throw_error(ErrorKind::Value, "histogramdd: `bins` must be positive");
      } else {
        double range_width = rmax - rmin;
        tmp_warns.clear();
        nbins = estimate_bins(*b.estimator, flat, range_width, tmp_warns);
      }
      all_edges[d] = uniform_edges(rmin, rmax, nbins);
    }
    const NDArray& ev = all_edges[d];
    edge_vals[d].resize(static_cast<std::size_t>(ev.size()));
    for (std::int64_t i = 0; i < ev.size(); ++i)
      edge_vals[d][static_cast<std::size_t>(i)] = ev.get_double(i);
  }

  // ---- build output shape ----
  Shape out_shape(D);
  for (std::size_t d = 0; d < D; ++d)
    out_shape[d] = static_cast<std::int64_t>(edge_vals[d].size()) - 1;

  // ---- strides for flat index calculation ----
  std::vector<std::int64_t> stride(D, 1);
  for (std::size_t d = D - 1; d-- > 0;)
    stride[d] = stride[d + 1] * out_shape[d + 1];

  const bool use_float = density || weights.has_value();
  NDArray hist = NDArray::zeros(out_shape, use_float ? DType::Float64 : DType::Int64);

  // ---- accumulate ----
  for (std::int64_t i = 0; i < N; ++i) {
    std::int64_t flat_idx = 0;
    bool in_range = true;
    for (std::size_t d = 0; d < D; ++d) {
      const double v = columns[d].get_double(i);
      if (std::isnan(v)) { in_range = false; break; }
      const std::int64_t bin = find_bin(v, edge_vals[d]);
      if (bin < 0) { in_range = false; break; }
      flat_idx += stride[d] * bin;
    }
    if (!in_range) continue;
    if (weights) {
      hist.set_double(flat_idx, hist.get_double(flat_idx) + weights->get_double(i));
    } else {
      hist.set_int64(flat_idx, hist.get_int64(flat_idx) + 1);
    }
  }

  if (density) {
    double total = 0;
    for (std::int64_t i = 0; i < hist.size(); ++i) total += hist.get_double(i);
    if (total != 0) {
      // Divide each cell by total * prod of its bin widths.
      // Iterate using multi-index.
      const std::int64_t ncells = hist.size();
      for (std::int64_t idx = 0; idx < ncells; ++idx) {
        double bin_vol = 1.0;
        std::int64_t tmp = idx;
        for (std::size_t d = 0; d < D; ++d) {
          const std::int64_t bin = tmp / stride[d];
          tmp %= stride[d];
          bin_vol *= edge_vals[d][static_cast<std::size_t>(bin + 1)] - edge_vals[d][static_cast<std::size_t>(bin)];
        }
        hist.set_double(idx, hist.get_double(idx) / (total * bin_vol));
      }
    }
  }

  return {hist, all_edges};
}

std::vector<NDArray> sample_columns(const NDArray& sample, bool from_list) {
  if (from_list || sample.ndim() == 1) {
    // sample is (N,) — treat as 1 variable.
    NDArray col = sample.reshape({sample.size()}).astype(DType::Float64);
    return {col};
  }
  // sample is (N, D) — columns.
  if (sample.ndim() != 2)
    throw_error(ErrorKind::Value, "histogramdd: sample must be 1-d or 2-d");
  const std::int64_t N = sample.shape()[0];
  const std::int64_t Dv = sample.shape()[1];
  const NDArray s = sample.astype(DType::Float64);
  std::vector<NDArray> cols;
  cols.reserve(static_cast<std::size_t>(Dv));
  for (std::int64_t d = 0; d < Dv; ++d) {
    NDArray col = NDArray::empty({N}, DType::Float64);
    for (std::int64_t r = 0; r < N; ++r)
      col.set_double(r, s.get_double(r * Dv + d));
    cols.push_back(col);
  }
  return cols;
}

// ---- bincount ----

NDArray bincount(const NDArray& x, const std::optional<NDArray>& weights, std::int64_t minlength,
                 bool from_list) {
  if (from_list && x.size() == 0) {
    // empty list: return zeros of length max(0, minlength)
    const std::int64_t len = std::max<std::int64_t>(0, minlength);
    return NDArray::zeros({len}, weights ? DType::Float64 : DType::Int64);
  }

  // x must be 1-d non-negative integers.
  if (x.ndim() > 1) throw_error(ErrorKind::Value, "object too deep for desired array");
  const DType xdt = x.dtype();
  if (dtype_info(xdt).kind == 'f' || is_complex(xdt) || xdt == DType::Bool)
    throw_error(ErrorKind::DType, "Cannot cast array data from dtype('" +
                std::string(1, dtype_info(xdt).kind) + "') to dtype('int64') according to the rule 'safe'");

  // Find maximum value.
  std::int64_t max_val = -1;
  for (std::int64_t i = 0; i < x.size(); ++i) {
    const std::int64_t v = x.get_int64(i);
    if (v < 0)
      throw_error(ErrorKind::Value, "First argument of bincount must be non-negative");
    if (v > max_val) max_val = v;
  }
  const std::int64_t len = std::max<std::int64_t>(max_val + 1, minlength);

  if (weights) {
    if (weights->size() != x.size())
      throw_error(ErrorKind::Value, "The weights and list don't have the same length.");
    NDArray out = NDArray::zeros({len}, DType::Float64);
    for (std::int64_t i = 0; i < x.size(); ++i) {
      const std::int64_t idx = x.get_int64(i);
      out.set_double(idx, out.get_double(idx) + weights->get_double(i));
    }
    return out;
  }
  NDArray out = NDArray::zeros({len}, DType::Int64);
  for (std::int64_t i = 0; i < x.size(); ++i) {
    const std::int64_t idx = x.get_int64(i);
    out.set_int64(idx, out.get_int64(idx) + 1);
  }
  return out;
}

// ---- digitize ----

NDArray digitize(const NDArray& x, const NDArray& bins, bool right) {
  if (bins.ndim() != 1) throw_error(ErrorKind::Value, "bins must be one-dimensional");
  const std::int64_t n = bins.size();

  // Determine if bins are monotonically increasing or decreasing.
  bool increasing = true, decreasing = true;
  for (std::int64_t i = 0; i < n - 1; ++i) {
    const double a = bins.get_double(i), b = bins.get_double(i + 1);
    if (a > b) increasing = false;
    if (a < b) decreasing = false;
  }
  if (n > 1 && !increasing && !decreasing)
    throw_error(ErrorKind::Value, "bins must be monotonically increasing or decreasing");

  // Copy bins as double for search.
  std::vector<double> bv(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) bv[static_cast<std::size_t>(i)] = bins.get_double(i);

  const NDArray xflat = x.reshape({x.size()}).astype(DType::Float64);
  NDArray out = NDArray::empty({x.size()}, DType::Int64);

  if (increasing) {
    for (std::int64_t i = 0; i < xflat.size(); ++i) {
      const double v = xflat.get_double(i);
      std::int64_t idx;
      if (right) {
        // np.digitize with right=True: use lower_bound (left-closed: v < bin).
        idx = static_cast<std::int64_t>(std::lower_bound(bv.begin(), bv.end(), v) - bv.begin());
      } else {
        // right=False (default): use upper_bound (v <= bin not counted).
        idx = static_cast<std::int64_t>(std::upper_bound(bv.begin(), bv.end(), v) - bv.begin());
      }
      out.set_int64(i, idx);
    }
  } else {
    // Decreasing: flip the bins, compute, then mirror the index.
    std::vector<double> rbv(bv.rbegin(), bv.rend());
    for (std::int64_t i = 0; i < xflat.size(); ++i) {
      const double v = xflat.get_double(i);
      std::int64_t idx;
      if (!right) {
        // digitize(right=False, decreasing) = mirror of upper_bound in reversed bins
        idx = static_cast<std::int64_t>(std::upper_bound(rbv.begin(), rbv.end(), v) - rbv.begin());
      } else {
        // digitize(right=True, decreasing) = mirror of lower_bound in reversed bins
        idx = static_cast<std::int64_t>(std::lower_bound(rbv.begin(), rbv.end(), v) - rbv.begin());
      }
      // Mirror index back to decreasing bins ordering.
      out.set_int64(i, n - idx);
    }
  }
  return out.reshape(x.shape());
}

// ---- interp ----

NDArray interp(const NDArray& x, const NDArray& xp, const NDArray& fp,
               const std::optional<std::complex<double>>& left_val,
               const std::optional<std::complex<double>>& right_val,
               std::optional<double> period) {
  if (xp.ndim() != 1) throw_error(ErrorKind::Value, "xp must be one-dimensional");
  if (fp.ndim() != 1) throw_error(ErrorKind::Value, "fp must be one-dimensional");
  if (xp.size() != fp.size()) throw_error(ErrorKind::Value, "xp and fp must be of the same length");
  if (xp.size() == 0) throw_error(ErrorKind::Value, "array of sample points is empty");

  const bool complex_fp = is_complex(fp.dtype());
  const DType out_dtype = complex_fp ? DType::Complex128 : DType::Float64;

  // Flatten x.
  const NDArray xflat = x.reshape({x.size()}).astype(DType::Float64);
  const std::int64_t nx = xflat.size();
  const std::int64_t nxp = xp.size();

  // Build xp/fp as double/complex128 vectors.
  std::vector<double> xpv(static_cast<std::size_t>(nxp));
  for (std::int64_t i = 0; i < nxp; ++i) xpv[static_cast<std::size_t>(i)] = xp.get_double(i);

  const NDArray fp64 = fp.astype(out_dtype);

  // Default left/right fill values.
  const std::complex<double> fill_left = left_val ? *left_val : (complex_fp ?
      [&]() -> std::complex<double> {
        return {fp64.get_double(0), fp64.get_double(1)};
      }() :
      std::complex<double>{fp64.get_double(0), 0.0});
  const std::complex<double> fill_right = right_val ? *right_val : (complex_fp ?
      [&]() -> std::complex<double> {
        return {fp64.get_double((nxp - 1) * 2), fp64.get_double((nxp - 1) * 2 + 1)};
      }() :
      std::complex<double>{fp64.get_double(nxp - 1), 0.0});

  // For complex fp we need per-element access to the complex values.
  auto get_fp = [&](std::int64_t i) -> std::complex<double> {
    if (complex_fp) {
      const NDArray fp128 = fp.astype(DType::Complex128);
      // complex128 stores real+imag as adjacent doubles.
      const double re = fp128.get_double(i * 2);
      const double im = fp128.get_double(i * 2 + 1);
      return {re, im};
    }
    return {fp.get_double(i), 0.0};
  };

  NDArray out = NDArray::empty({nx}, out_dtype);

  for (std::int64_t i = 0; i < nx; ++i) {
    double xi = xflat.get_double(i);

    // Handle period.
    if (period) {
      if (*period <= 0) throw_error(ErrorKind::Value, "period must be a positive value");
      xi = std::fmod(xi, *period);
      if (xi < 0) xi += *period;
      // Wrap xp too — this is done by ensuring the search works in [0, period).
    }

    // Binary search for the insertion point in xpv.
    const auto it = std::lower_bound(xpv.begin(), xpv.end(), xi);
    const std::int64_t j = static_cast<std::int64_t>(it - xpv.begin());

    std::complex<double> result;
    if (j == 0) {
      if (xi < xpv[0]) result = fill_left;
      else result = get_fp(0);
    } else if (j >= nxp) {
      result = fill_right;
    } else {
      // Interpolate between j-1 and j.
      const double x0 = xpv[static_cast<std::size_t>(j - 1)];
      const double x1 = xpv[static_cast<std::size_t>(j)];
      const std::complex<double> y0 = get_fp(j - 1);
      const std::complex<double> y1 = get_fp(j);
      const double slope = (x1 - x0);
      double t;
      if (slope == 0) {
        t = 1.0;
      } else {
        // Use std::fma for the interpolation weight (matches NumPy's fused multiply-add).
        t = (xi - x0) / slope;
      }
      result = {std::fma(t, y1.real() - y0.real(), y0.real()),
                std::fma(t, y1.imag() - y0.imag(), y0.imag())};
    }

    if (complex_fp) {
      out.set_double(i * 2, result.real());
      out.set_double(i * 2 + 1, result.imag());
    } else {
      out.set_double(i, result.real());
    }
  }

  return out.reshape(x.shape());
}

}  // namespace nativpy::p10
