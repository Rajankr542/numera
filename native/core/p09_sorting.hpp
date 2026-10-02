#pragma once

// P9 sorting, selection and searching (D-120, D-121, D-122).

#include <cstdint>
#include <optional>
#include <vector>

#include "ndarray.hpp"

namespace nativpy {

enum class SortKind : std::uint8_t { Quick, Stable };

// np.sort: sorted copy. axis nullopt flattens (C order) first.
NDArray sort_copy(const NDArray& a, std::optional<std::int64_t> axis, SortKind kind, bool descending);
// ndarray.sort: in place along `axis` (array must be writeable).
void sort_inplace(const NDArray& a, std::int64_t axis, SortKind kind, bool descending);
// np.argsort: int64 indices (flattened input when axis is nullopt).
NDArray argsort(const NDArray& a, std::optional<std::int64_t> axis, SortKind kind, bool descending);

// np.partition / ndarray.partition / np.argpartition (NumPy introselect).
NDArray partition_copy(const NDArray& a, const std::vector<std::int64_t>& kth,
                       std::optional<std::int64_t> axis);
void partition_inplace(const NDArray& a, const std::vector<std::int64_t>& kth, std::int64_t axis);
NDArray argpartition(const NDArray& a, const std::vector<std::int64_t>& kth,
                     std::optional<std::int64_t> axis);

// np.lexsort: keys of equal shape; the last key is primary.
NDArray lexsort(const std::vector<NDArray>& keys, std::int64_t axis);

// np.searchsorted: `a` 1-D and `v` of the same dtype; sorter optional int64 1-D.
NDArray searchsorted(const NDArray& a, const NDArray& v, bool right,
                     const std::optional<NDArray>& sorter);

}  // namespace nativpy
