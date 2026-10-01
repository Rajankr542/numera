#pragma once

#include "ndarray.hpp"
#include "ufunc.hpp"

namespace nativpy {

// Memory-order aware creation and copies (P3-1, D-055). `Order` is the
// D-050 enum (C/F/A/K).

// Strides of a fresh dense C- or F-ordered array (zero dims → zero strides).
Strides order_strides(const Shape& shape, std::size_t itemsize, Order order);

// New uninitialized / zeroed array in C or F order (A/K are rejected:
// "only 'C' or 'F' order is permitted").
NDArray empty_order(const Shape& shape, DType dtype, Order order, bool zeroed = false);

// The layout NumPy gives a copy of `a` (np.copy / a.copy / astype order=):
// C, F, A (F if `a` is F- but not C-contiguous) or K (keep `a`'s stride order).
Strides copy_layout(const NDArray& a, std::size_t itemsize, Order order);

// Copy of `a` cast to `dtype` with the given order.
NDArray copy_order(const NDArray& a, DType dtype, Order order);

// NumPy reshape(order=): C reads/writes in row-major index order, F in
// column-major. A means F if `a` is F-contiguous and not C, else C.
// Returns a view when possible, otherwise a copy.
NDArray reshape_order(const NDArray& a, const Shape& shape, Order order);

// NumPy ravel(order=) (view when possible) and flatten(order=) (always a copy).
// ravel 'K' reads elements in memory order.
NDArray ravel_order(const NDArray& a, Order order);
NDArray flatten_order(const NDArray& a, Order order);

}  // namespace nativpy
