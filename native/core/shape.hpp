#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nativpy {

using Shape = std::vector<std::int64_t>;
using Strides = std::vector<std::int64_t>;  // in bytes (D-003)

// Maximum dimensions, matching NumPy 2.x (NPY_MAXDIMS = 64).
inline constexpr std::size_t kMaxDims = 64;

// Throws ShapeError/ValueError for negative dims, too many dims, or overflow.
void validate_shape(const Shape& shape);

// Product of dimensions (1 for 0-d). Throws ValueError on int64 overflow.
std::int64_t shape_size(const Shape& shape);

// Normalizes a possibly-negative axis. Throws IndexError (AxisError analogue).
std::int64_t normalize_axis(std::int64_t axis, std::int64_t ndim);

// Resolves a single -1 entry in a reshape target. Throws ShapeError.
Shape resolve_reshape(const Shape& target, std::int64_t size);

std::string shape_to_string(const Shape& shape);

}  // namespace nativpy
