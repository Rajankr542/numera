#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ndarray.hpp"

namespace nativpy {

// Shape manipulation (PLAN §12, M3). All return views except flatten.

// Permutes axes (reverses them when `axes` is empty). Throws on repeated or
// out-of-range axes.
NDArray transpose(const NDArray& a, const std::vector<std::int64_t>& axes);

// Removes the given size-1 axes, or every size-1 axis when `axes` is nullopt.
NDArray squeeze(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axes);

// Inserts size-1 axes at the given positions (normalized against the
// output ndim, like NumPy).
NDArray expand_dims(const NDArray& a, const std::vector<std::int64_t>& axes);

// NumPy swapaxes / moveaxis.
NDArray swapaxes(const NDArray& a, std::int64_t axis1, std::int64_t axis2);
NDArray moveaxis(const NDArray& a, const std::vector<std::int64_t>& source,
                 const std::vector<std::int64_t>& destination);

// ravel: view when possible (reshape(-1)); flatten: always a copy.
NDArray ravel(const NDArray& a);
NDArray flatten(const NDArray& a);

// Normalizes a list of axes against ndim; throws Value on duplicates
// (NumPy "repeated axis") and Index when out of range.
std::vector<std::int64_t> normalize_axes(const std::vector<std::int64_t>& axes, std::int64_t ndim);

}  // namespace nativpy
