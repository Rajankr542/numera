#pragma once

#include <cstdint>
#include <optional>

#include "ndarray.hpp"

namespace nativpy {

// NumPy isclose (D-083): |a - b| <= atol + rtol * |b| (b finite) or a == b;
// with equal_nan also where both are NaN. Result is bool, broadcast shape.
NDArray isclose(const NDArray& a, const NDArray& b, double rtol, double atol, bool equal_nan);

// NumPy packbits (D-084): int/bool input, nonzero → 1 bit, uint8 output.
// axis nullopt flattens; 0-d input acts like shape (1,).
NDArray packbits(const NDArray& a, std::optional<std::int64_t> axis, bool little);

// NumPy unpackbits (D-084): uint8 input, 8 bits per byte along axis (nullopt
// flattens); count keeps the first `count` bits (negative: drops -count, too
// negative → ValueError), padding with zeros when count exceeds the bits.
NDArray unpackbits(const NDArray& a, std::optional<std::int64_t> axis, std::optional<std::int64_t> count,
                   bool little);

}  // namespace nativpy
