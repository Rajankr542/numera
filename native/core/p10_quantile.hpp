#pragma once

// Quantiles, percentiles and medians, including the nan-variants (D-131).

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "ndarray.hpp"

namespace nativpy::p10 {

enum class QMethod {
  InvertedCdf,
  AveragedInvertedCdf,
  ClosestObservation,
  InterpolatedInvertedCdf,
  Hazen,
  Weibull,
  Linear,
  MedianUnbiased,
  NormalUnbiased,
  Lower,
  Higher,
  Midpoint,
  Nearest,
};

// Throws ValueError (NumPy's message) for an unknown name.
QMethod qmethod_from_name(std::string_view name);

struct QuantileOptions {
  std::optional<std::vector<std::int64_t>> axis;  // nullopt = flattened
  bool keepdims = false;
  QMethod method = QMethod::Linear;
  std::optional<NDArray> weights;  // inverted_cdf only
  bool weak_q = false;             // q was a JS number (NEP 50 weak scalar)
  bool percentile = false;         // q is in [0, 100]
  bool ignore_nan = false;         // nanquantile / nanpercentile
};

// NumPy quantile/percentile (and nan-variants). The result shape is
// q.shape + the reduced shape of `a`.
NDArray quantile(const NDArray& a, const NDArray& q, const QuantileOptions& opts);

// NumPy median / nanmedian.
NDArray median(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis,
               bool keepdims, bool ignore_nan);

}  // namespace nativpy::p10
