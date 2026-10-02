#pragma once

// P9 unique and set functions (D-123, D-124). Inputs that combine two arrays
// must already share a dtype (the TS layer promotes them).

#include <optional>

#include "ndarray.hpp"

namespace nativpy {

struct UniqueResult {
  NDArray values;
  std::optional<NDArray> indices;
  std::optional<NDArray> inverse;  // 1-D, length a.size()
  std::optional<NDArray> counts;
};

// NumPy _unique1d over the flattened array (always sorted output).
UniqueResult unique1d(const NDArray& a, bool return_index, bool return_inverse, bool return_counts,
                      bool equal_nan);
// unique(axis=): `a` has shape (n, m); rows compare lexicographically.
// values has shape (k, m).
UniqueResult unique_rows(const NDArray& a, bool return_index, bool return_inverse, bool return_counts);

// Bool array of elem's shape: elem value == some test value (NumPy ==).
NDArray isin(const NDArray& elem, const NDArray& test, bool invert);

struct IntersectResult {
  NDArray values;
  std::optional<NDArray> indices1;
  std::optional<NDArray> indices2;
};
IntersectResult intersect1d(const NDArray& a, const NDArray& b, bool assume_unique, bool return_indices);
NDArray union1d(const NDArray& a, const NDArray& b);
NDArray setxor1d(const NDArray& a, const NDArray& b, bool assume_unique);
NDArray setdiff1d(const NDArray& a, const NDArray& b, bool assume_unique);

// Differences of the flattened array, with optional prefix/suffix arrays of
// the same dtype. Bool input raises DType.
NDArray ediff1d(const NDArray& a, const std::optional<NDArray>& to_begin,
                const std::optional<NDArray>& to_end);

}  // namespace nativpy
