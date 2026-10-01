#include "layout.hpp"

#include "broadcast.hpp"
#include "error.hpp"
#include "shape.hpp"
#include "shape_ops.hpp"
#include "strides.hpp"

namespace nativpy {

namespace {

// Dense strides with `axes` listed outermost → innermost.
Strides dense_strides(const Shape& shape, std::size_t itemsize, const std::vector<std::size_t>& axes) {
  Strides st(shape.size(), 0);
  if (shape_size(shape) == 0) return st;
  auto acc = static_cast<std::int64_t>(itemsize);
  for (std::size_t i = axes.size(); i-- > 0;) {
    st[axes[i]] = acc;
    acc *= shape[axes[i]];
  }
  return st;
}

// Orders that resolve 'A' against `a`: F only if F- and not C-contiguous.
Order resolve_any(const NDArray& a, Order order) {
  if (order != Order::A) return order;
  return (a.is_f_contiguous() && !a.is_c_contiguous()) ? Order::F : Order::C;
}

// View of `a` with axes sorted by |stride| (outermost first, C order wins
// ties), the element order NumPy's ravel/flatten 'K' reads.
NDArray k_order_view(const NDArray& a) {
  const auto axes = keep_order_axes(a.shape(), {&a});
  return transpose(a, std::vector<std::int64_t>(axes.begin(), axes.end()));
}

}  // namespace

Strides order_strides(const Shape& shape, std::size_t itemsize, Order order) {
  if (order == Order::F) {
    std::vector<std::size_t> axes(shape.size());
    for (std::size_t i = 0; i < axes.size(); ++i) axes[i] = axes.size() - 1 - i;
    return dense_strides(shape, itemsize, axes);
  }
  return allocation_strides(shape, itemsize);
}

NDArray empty_order(const Shape& shape, DType dtype, Order order, bool zeroed) {
  if (order != Order::C && order != Order::F) {
    throw_error(ErrorKind::Value, "only 'C' or 'F' order is permitted");
  }
  for (const auto d : shape) {
    if (d < 0) throw_error(ErrorKind::Value, "negative dimensions are not allowed");
  }
  return NDArray::empty_strided(shape, order_strides(shape, itemsize(dtype), order), dtype, zeroed);
}

Strides copy_layout(const NDArray& a, std::size_t itemsize, Order order) {
  order = resolve_any(a, order);
  if (order == Order::K) {
    if (a.is_c_contiguous()) return allocation_strides(a.shape(), itemsize);
    if (a.is_f_contiguous()) return order_strides(a.shape(), itemsize, Order::F);
    if (a.size() == 0) return allocation_strides(a.shape(), itemsize);
    return dense_strides(a.shape(), itemsize, keep_order_axes(a.shape(), {&a}));
  }
  return order_strides(a.shape(), itemsize, order);
}

NDArray copy_order(const NDArray& a, DType dtype, Order order) {
  NDArray out = NDArray::empty_strided(a.shape(), copy_layout(a, itemsize(dtype), order), dtype);
  if (out.size() > 0) copy_into(out, a);
  return out;
}

NDArray reshape_order(const NDArray& a, const Shape& shape, Order order) {
  if (order == Order::K) throw_error(ErrorKind::Value, "order 'K' is not permitted for reshaping");
  order = resolve_any(a, order);
  if (order == Order::C) return a.reshape(shape);
  // F: reverse both shapes, reshape in C order, reverse back.
  const Shape rev(shape.rbegin(), shape.rend());
  return transpose(transpose(a, {}).reshape(rev), {});
}

NDArray ravel_order(const NDArray& a, Order order) {
  if (order == Order::K) {
    if (a.is_c_contiguous()) return a.reshape({-1});
    if (a.is_f_contiguous()) return reshape_order(a, {-1}, Order::F);
    const NDArray k = k_order_view(a);
    return k.is_c_contiguous() ? k.reshape({-1}) : k.copy().reshape({-1});
  }
  return reshape_order(a, {-1}, order);
}

NDArray flatten_order(const NDArray& a, Order order) {
  if (order == Order::K) return k_order_view(a).copy().reshape({-1});
  order = resolve_any(a, order);
  if (order == Order::F) return reshape_order(copy_order(a, a.dtype(), Order::F), {-1}, Order::F);
  return a.copy().reshape({-1});
}

}  // namespace nativpy
