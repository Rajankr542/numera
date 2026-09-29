#pragma once

#include <cstdint>

#include "ndarray.hpp"

namespace nativpy {

// Array creation routines (PLAN §11, M2). Semantics follow NumPy 2.x; see
// DECISIONS D-012 for divergences caused by JS numbers.

// New C-contiguous array whose every element equals `value` (size 1, any
// shape). Result dtype is value.dtype(); the caller casts beforehand.
NDArray full(const Shape& shape, const NDArray& value);

// Like full() with the value 1 cast to `dtype` (valid for every dtype).
NDArray ones(const Shape& shape, DType dtype);

// NumPy arange: length ceil((stop - start) / step); values computed as
// v0 + i * (v1 - v0) where v0 = cast(start), v1 = cast(start + step), in the
// target dtype (NumPy's `fill` semantics). Throws Value on step == 0 or a
// non-finite length; bool results may have at most 2 elements.
NDArray arange(double start, double stop, double step, DType dtype);

// NumPy linspace computed in float64, then floored for integer dtypes and
// cast to `dtype`. Throws Value when num < 0.
NDArray linspace(double start, double stop, std::int64_t num, bool endpoint, DType dtype);

// NumPy eye: N x M array with ones on diagonal k.
NDArray eye(std::int64_t n, std::int64_t m, std::int64_t k, DType dtype);

}  // namespace nativpy
