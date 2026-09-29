#include "strides.hpp"

#include <string>
#include <utility>

#include "error.hpp"

namespace nativpy {

Strides c_contiguous_strides(const Shape& shape, std::size_t itemsize) {
  Strides strides(shape.size());
  auto running = static_cast<std::int64_t>(itemsize);
  for (std::size_t i = shape.size(); i-- > 0;) {
    strides[i] = running;
    // NumPy uses max(dim, 1) so that zero-size dims don't zero the strides.
    running *= shape[i] > 0 ? shape[i] : 1;
  }
  return strides;
}

namespace {
bool has_zero_dim(const Shape& shape) {
  for (const auto d : shape) {
    if (d == 0) return true;
  }
  return false;
}
}  // namespace

Strides allocation_strides(const Shape& shape, std::size_t itemsize) {
  if (has_zero_dim(shape)) return Strides(shape.size(), 0);
  return c_contiguous_strides(shape, itemsize);
}

bool attempt_nocopy_reshape(const Shape& old_shape, const Strides& old_strides,
                            const Shape& new_shape, std::size_t itemsize,
                            Strides& out_strides) {
  // Drop size-1 dimensions of the old layout; they do not constrain strides.
  Shape odims;
  Strides ostrides;
  for (std::size_t i = 0; i < old_shape.size(); ++i) {
    if (old_shape[i] != 1) {
      odims.push_back(old_shape[i]);
      ostrides.push_back(old_strides[i]);
    }
  }
  const std::size_t oldnd = odims.size();
  const std::size_t newnd = new_shape.size();
  Strides nstrides(newnd, 0);

  std::size_t oi = 0;
  std::size_t oj = 1;
  std::size_t ni = 0;
  std::size_t nj = 1;
  while (ni < newnd && oi < oldnd) {
    std::int64_t np = new_shape[ni];
    std::int64_t op = odims[oi];
    while (np != op) {
      if (np < op) {
        np *= new_shape[nj++];
      } else {
        op *= odims[oj++];
      }
    }
    // The grouped old axes [oi, oj) must be contiguous with each other.
    for (std::size_t ok = oi; ok + 1 < oj; ++ok) {
      if (ostrides[ok] != odims[ok + 1] * ostrides[ok + 1]) return false;
    }
    nstrides[nj - 1] = ostrides[oj - 1];
    for (std::size_t nk = nj - 1; nk > ni; --nk) {
      nstrides[nk - 1] = nstrides[nk] * new_shape[nk];
    }
    ni = nj++;
    oi = oj++;
  }
  // Trailing size-1 dimensions of the new shape.
  const std::int64_t last = ni >= 1 ? nstrides[ni - 1] : static_cast<std::int64_t>(itemsize);
  for (std::size_t nk = ni; nk < newnd; ++nk) nstrides[nk] = last;
  out_strides = std::move(nstrides);
  return true;
}

// NumPy flag semantics: size-1 dims are ignored; empty arrays are contiguous.
bool is_c_contiguous(const Shape& shape, const Strides& strides,
                     std::size_t itemsize) {
  if (has_zero_dim(shape)) return true;
  auto expected = static_cast<std::int64_t>(itemsize);
  for (std::size_t i = shape.size(); i-- > 0;) {
    if (shape[i] == 1) continue;
    if (strides[i] != expected) return false;
    expected *= shape[i];
  }
  return true;
}

bool is_f_contiguous(const Shape& shape, const Strides& strides,
                     std::size_t itemsize) {
  if (has_zero_dim(shape)) return true;
  auto expected = static_cast<std::int64_t>(itemsize);
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 1) continue;
    if (strides[i] != expected) return false;
    expected *= shape[i];
  }
  return true;
}

ByteExtent byte_extent(const Shape& shape, const Strides& strides,
                       std::int64_t offset, std::size_t itemsize) {
  if (has_zero_dim(shape)) return {offset, offset, true};
  std::int64_t low = offset;
  std::int64_t high = offset;
  for (std::size_t i = 0; i < shape.size(); ++i) {
    const std::int64_t span = (shape[i] - 1) * strides[i];
    if (span >= 0) {
      high += span;
    } else {
      low += span;
    }
  }
  return {low, high + static_cast<std::int64_t>(itemsize), false};
}

void check_view_bounds(const Shape& shape, const Strides& strides,
                       std::int64_t offset, std::size_t itemsize,
                       std::size_t buffer_size) {
  if (shape.size() != strides.size()) {
    throw_error(ErrorKind::Value, "shape and strides must have the same length");
  }
  validate_shape(shape);
  const ByteExtent ext = byte_extent(shape, strides, offset, itemsize);
  if (ext.empty) {
    if (offset < 0 || static_cast<std::uint64_t>(offset) > buffer_size) {
      throw_error(ErrorKind::Value, "view offset out of buffer bounds");
    }
    return;
  }
  if (ext.low < 0 || static_cast<std::uint64_t>(ext.high) > buffer_size) {
    throw_error(ErrorKind::Value,
                "view requires bytes [" + std::to_string(ext.low) + ", " +
                    std::to_string(ext.high) + ") but buffer has " +
                    std::to_string(buffer_size) + " bytes");
  }
}

}  // namespace nativpy
