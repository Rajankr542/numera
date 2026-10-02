#pragma once

#include <vector>

#include "ndarray.hpp"
#include "ufunc.hpp"

namespace nativpy {

// NDArray methods of P3-2 (D-060).

// a.fill(value): `value` (any shape, size >= 1) is cast to a's dtype; its first
// element is written to every element of `a`.
void fill(const NDArray& a, const NDArray& value);

// The bytes of `a` in C or F order (NumPy tobytes; K is treated as C, A as F
// when `a` is F- and not C-contiguous).
std::vector<std::byte> to_bytes(const NDArray& a, Order order);

// a.view(dtype): reinterprets the bytes (NumPy 2 rules for itemsize changes).
NDArray view_as(const NDArray& a, DType dtype);

// a.byteswap(): a K-layout copy with every element's bytes reversed (complex:
// each component); `inplace` swaps `a` itself and returns it.
NDArray byteswap(const NDArray& a, bool inplace);

// a.flat[positions] = values. `positions` holds C-order flat indices (int64,
// any shape) or is null for every element; `values` (cast to a's dtype) is
// repeated cyclically, as NumPy's flatiter assignment.
void flat_assign(const NDArray& a, const NDArray* positions, const NDArray& values);

}  // namespace nativpy
