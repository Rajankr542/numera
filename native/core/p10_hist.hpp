#pragma once

// histogram, histogram_bin_edges, histogramdd, bincount, digitize, interp
// (D-134, D-138), following numpy/lib/_histograms_impl.py and
// numpy/_core/src/multiarray/compiled_base.c.

#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "ndarray.hpp"

namespace nativpy::p10 {

// One `bins=` argument: an integer count, an estimator name or 1-d edges.
struct HistBins {
  std::optional<std::int64_t> count;
  std::optional<std::string> estimator;
  std::optional<NDArray> edges;
};

using Range = std::optional<std::pair<double, double>>;

struct HistResult {
  NDArray hist;
  NDArray edges;
};

// NumPy histogram (edges_only: histogram_bin_edges; `hist` is then empty).
// RuntimeWarning messages are appended to `warnings`.
HistResult histogram(const NDArray& a, const HistBins& bins, const Range& range, bool density,
                     const std::optional<NDArray>& weights, bool edges_only, std::vector<std::string>& warnings);

struct HistDdResult {
  NDArray hist;
  std::vector<NDArray> edges;
};

// NumPy histogramdd. `columns` holds the D coordinate arrays (each of length
// N); `bins` and `range` have D entries (bins: count or edges only).
HistDdResult histogramdd(const std::vector<NDArray>& columns, const std::vector<HistBins>& bins,
                         const std::vector<Range>& range, bool density, const std::optional<NDArray>& weights);

// Splits a sample into histogramdd columns: an (N, D) array, or (from_list)
// NumPy's atleast_2d(sample).T for a nested JS list.
std::vector<NDArray> sample_columns(const NDArray& sample, bool from_list);

// NumPy bincount. `from_list`: x came from a JS array (an empty list is int).
NDArray bincount(const NDArray& x, const std::optional<NDArray>& weights, std::int64_t minlength, bool from_list);

// NumPy digitize.
NDArray digitize(const NDArray& x, const NDArray& bins, bool right);

// NumPy interp (float64, or complex128 for complex fp).
NDArray interp(const NDArray& x, const NDArray& xp, const NDArray& fp,
               const std::optional<std::complex<double>>& left, const std::optional<std::complex<double>>& right,
               std::optional<double> period);

}  // namespace nativpy::p10
