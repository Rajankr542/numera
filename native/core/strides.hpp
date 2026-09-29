#pragma once

#include <cstddef>
#include <cstdint>

#include "shape.hpp"

namespace nativpy {

// C-order (row-major) strides in bytes. Matches NumPy: dims of size <= 1 still
// get the running product.
Strides c_contiguous_strides(const Shape& shape, std::size_t itemsize);

// Strides for a freshly allocated C-order array. Same as c_contiguous_strides,
// except any zero-size dimension makes all strides 0 (NumPy, D-011).
Strides allocation_strides(const Shape& shape, std::size_t itemsize);

// NumPy's _attempt_nocopy_reshape (C order). Returns true and fills
// `out_strides` when `new_shape` can be expressed as a view of the given
// layout. Both shapes must have the same number of elements, which must be > 0.
bool attempt_nocopy_reshape(const Shape& old_shape, const Strides& old_strides,
                            const Shape& new_shape, std::size_t itemsize,
                            Strides& out_strides);

bool is_c_contiguous(const Shape& shape, const Strides& strides,
                     std::size_t itemsize);
bool is_f_contiguous(const Shape& shape, const Strides& strides,
                     std::size_t itemsize);

// Min/max byte offsets (relative to `offset`) touched by a view. For empty
// arrays returns {0, 0} with `empty = true`.
struct ByteExtent {
  std::int64_t low;
  std::int64_t high;  // exclusive
  bool empty;
};
ByteExtent byte_extent(const Shape& shape, const Strides& strides,
                       std::int64_t offset, std::size_t itemsize);

// Throws Value/Index error if the view reaches outside [0, buffer_size).
void check_view_bounds(const Shape& shape, const Strides& strides,
                       std::int64_t offset, std::size_t itemsize,
                       std::size_t buffer_size);

}  // namespace nativpy
