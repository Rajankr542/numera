#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ndarray.hpp"

namespace nativpy {

// One component of a NumPy index tuple (PLAN §13, D-015).
struct IndexItem {
  enum class Kind { Integer, Slice, NewAxis, Ellipsis, Array };
  Kind kind = Kind::Slice;
  std::int64_t integer = 0;                  // Kind::Integer
  std::optional<std::int64_t> start, stop;   // Kind::Slice
  std::optional<std::int64_t> step;          // Kind::Slice (nullopt = 1)
  std::optional<NDArray> array;              // Kind::Array (integer or bool dtype)

  static IndexItem integer_(std::int64_t i) {
    IndexItem it;
    it.kind = Kind::Integer;
    it.integer = i;
    return it;
  }
  static IndexItem slice(std::optional<std::int64_t> start, std::optional<std::int64_t> stop,
                         std::optional<std::int64_t> step) {
    IndexItem it;
    it.kind = Kind::Slice;
    it.start = start;
    it.stop = stop;
    it.step = step;
    return it;
  }
  static IndexItem new_axis() {
    IndexItem it;
    it.kind = Kind::NewAxis;
    return it;
  }
  static IndexItem ellipsis() {
    IndexItem it;
    it.kind = Kind::Ellipsis;
    return it;
  }
  static IndexItem array_(NDArray a) {
    IndexItem it;
    it.kind = Kind::Array;
    it.array = std::move(a);
    return it;
  }
};

// Python slice.indices(): normalized (start, stop, step, length).
struct SliceBounds {
  std::int64_t start, stop, step, length;
};
SliceBounds slice_indices(std::optional<std::int64_t> start, std::optional<std::int64_t> stop,
                          std::optional<std::int64_t> step, std::int64_t dim);

// NumPy a[index]. Basic indexing (ints, slices, newaxis, ellipsis) returns a
// view; any integer/bool array component makes it advanced indexing (copy).
NDArray get_index(const NDArray& a, const std::vector<IndexItem>& index);

// NumPy a[index] = value (value broadcast to the selection, cast unsafely).
void set_index(NDArray& a, const std::vector<IndexItem>& index, const NDArray& value);

// Indices of nonzero elements, one int64 array per dimension (np.nonzero).
std::vector<NDArray> nonzero(const NDArray& a);

// np.take(a, indices, axis) (axis nullopt = flattened).
NDArray take(const NDArray& a, const NDArray& indices, std::optional<std::int64_t> axis);

// np.where(cond, x, y) with broadcasting and promoted dtype.
NDArray where(const NDArray& cond, const NDArray& x, const NDArray& y);

}  // namespace nativpy
