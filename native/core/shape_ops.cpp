#include "shape_ops.hpp"

#include <algorithm>
#include <string>

#include "error.hpp"

namespace nativpy {

std::vector<std::int64_t> normalize_axes(const std::vector<std::int64_t>& axes, std::int64_t ndim) {
  std::vector<std::int64_t> out;
  out.reserve(axes.size());
  std::vector<bool> seen(static_cast<std::size_t>(ndim), false);
  for (const auto ax : axes) {
    const std::int64_t n = normalize_axis(ax, ndim);
    if (seen[static_cast<std::size_t>(n)]) throw_error(ErrorKind::Value, "repeated axis");
    seen[static_cast<std::size_t>(n)] = true;
    out.push_back(n);
  }
  return out;
}

NDArray transpose(const NDArray& a, const std::vector<std::int64_t>& axes) {
  const auto nd = static_cast<std::int64_t>(a.ndim());
  std::vector<std::int64_t> perm;
  if (axes.empty()) {
    for (std::int64_t i = nd; i-- > 0;) perm.push_back(i);
  } else {
    if (static_cast<std::int64_t>(axes.size()) != nd) {
      throw_error(ErrorKind::Value, "axes don't match array");
    }
    perm = normalize_axes(axes, nd);
  }
  Shape shape;
  Strides strides;
  for (const auto p : perm) {
    shape.push_back(a.shape()[static_cast<std::size_t>(p)]);
    strides.push_back(a.strides()[static_cast<std::size_t>(p)]);
  }
  return a.view(std::move(shape), std::move(strides), a.offset());
}

NDArray squeeze(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axes) {
  const auto nd = static_cast<std::int64_t>(a.ndim());
  std::vector<bool> drop(a.ndim(), false);
  if (axes) {
    for (const auto ax : normalize_axes(*axes, nd)) {
      if (a.shape()[static_cast<std::size_t>(ax)] != 1) {
        throw_error(ErrorKind::Value,
                    "cannot select an axis to squeeze out which has size not equal to one");
      }
      drop[static_cast<std::size_t>(ax)] = true;
    }
  } else {
    for (std::size_t i = 0; i < a.ndim(); ++i) drop[i] = a.shape()[i] == 1;
  }
  Shape shape;
  Strides strides;
  for (std::size_t i = 0; i < a.ndim(); ++i) {
    if (drop[i]) continue;
    shape.push_back(a.shape()[i]);
    strides.push_back(a.strides()[i]);
  }
  return a.view(std::move(shape), std::move(strides), a.offset());
}

NDArray expand_dims(const NDArray& a, const std::vector<std::int64_t>& axes) {
  const auto out_nd = static_cast<std::int64_t>(a.ndim() + axes.size());
  if (static_cast<std::size_t>(out_nd) > kMaxDims) throw_error(ErrorKind::Value, "too many dimensions");
  const auto norm = normalize_axes(axes, out_nd);
  std::vector<bool> is_new(static_cast<std::size_t>(out_nd), false);
  for (const auto ax : norm) is_new[static_cast<std::size_t>(ax)] = true;
  Shape shape;
  std::size_t src = 0;
  for (std::size_t i = 0; i < is_new.size(); ++i) shape.push_back(is_new[i] ? 1 : a.shape()[src++]);
  // NumPy implements expand_dims with reshape; inserting size-1 axes is
  // always a view, and this reproduces NumPy's strides exactly.
  return a.reshape(shape);
}

NDArray swapaxes(const NDArray& a, std::int64_t axis1, std::int64_t axis2) {
  const auto nd = static_cast<std::int64_t>(a.ndim());
  const std::int64_t i = normalize_axis(axis1, nd);
  const std::int64_t j = normalize_axis(axis2, nd);
  std::vector<std::int64_t> perm;
  for (std::int64_t d = 0; d < nd; ++d) perm.push_back(d);
  std::swap(perm[static_cast<std::size_t>(i)], perm[static_cast<std::size_t>(j)]);
  return transpose(a, perm);
}

NDArray moveaxis(const NDArray& a, const std::vector<std::int64_t>& source,
                 const std::vector<std::int64_t>& destination) {
  if (source.size() != destination.size()) {
    throw_error(ErrorKind::Value,
                "`source` and `destination` arguments must have the same number of elements");
  }
  const auto nd = static_cast<std::int64_t>(a.ndim());
  const auto src = normalize_axes(source, nd);
  const auto dst = normalize_axes(destination, nd);
  std::vector<std::int64_t> order;
  std::vector<bool> used(a.ndim(), false);
  for (const auto s : src) used[static_cast<std::size_t>(s)] = true;
  for (std::int64_t d = 0; d < nd; ++d) {
    if (!used[static_cast<std::size_t>(d)]) order.push_back(d);
  }
  // Insert in increasing destination order (NumPy algorithm).
  std::vector<std::size_t> idx(src.size());
  for (std::size_t k = 0; k < idx.size(); ++k) idx[k] = k;
  std::sort(idx.begin(), idx.end(), [&](std::size_t x, std::size_t y) { return dst[x] < dst[y]; });
  for (const auto k : idx) order.insert(order.begin() + dst[k], src[k]);
  return transpose(a, order);
}

NDArray ravel(const NDArray& a) { return a.reshape({-1}); }

NDArray flatten(const NDArray& a) { return a.copy().reshape({-1}); }

}  // namespace nativpy
