#include "ndarray.hpp"

#include <string>
#include <utility>

#include "cast.hpp"
#include "error.hpp"

namespace nativpy {

namespace {
std::size_t checked_nbytes(const Shape& shape, DType dtype) {
  validate_shape(shape);
  const std::int64_t n = shape_size(shape);
  const auto isz = static_cast<std::int64_t>(itemsize(dtype));
  if (n > std::numeric_limits<std::int64_t>::max() / isz) {
    throw_error(ErrorKind::Memory, "array is too big; " + shape_to_string(shape));
  }
  return static_cast<std::size_t>(n * isz);
}
}  // namespace

NDArray::NDArray(std::shared_ptr<MemoryBuffer> buffer, DType dtype, Shape shape,
                 Strides strides, std::int64_t offset, bool owns_data)
    : buffer_(std::move(buffer)),
      dtype_(dtype),
      shape_(std::move(shape)),
      strides_(std::move(strides)),
      offset_(offset),
      size_(0),
      owns_data_(owns_data) {
  if (!buffer_) throw_error(ErrorKind::Value, "NDArray requires a buffer");
  check_view_bounds(shape_, strides_, offset_, itemsize(), buffer_->size());
  size_ = shape_size(shape_);
}

NDArray::NDArray(std::shared_ptr<MemoryBuffer> buffer, DType dtype, Shape shape,
                 Strides strides, std::int64_t offset)
    : NDArray(std::move(buffer), dtype, std::move(shape), std::move(strides),
              offset, false) {}

NDArray NDArray::empty(const Shape& shape, DType dtype) {
  const std::size_t nbytes = checked_nbytes(shape, dtype);
  auto buf = MemoryBuffer::allocate(nbytes, false);
  return {std::move(buf), dtype, shape, allocation_strides(shape, nativpy::itemsize(dtype)),
          0, true};
}

NDArray NDArray::zeros(const Shape& shape, DType dtype) {
  const std::size_t nbytes = checked_nbytes(shape, dtype);
  auto buf = MemoryBuffer::allocate(nbytes, true);
  return {std::move(buf), dtype, shape, allocation_strides(shape, nativpy::itemsize(dtype)),
          0, true};
}

bool NDArray::is_c_contiguous() const noexcept {
  return nativpy::is_c_contiguous(shape_, strides_, itemsize());
}

bool NDArray::is_f_contiguous() const noexcept {
  return nativpy::is_f_contiguous(shape_, strides_, itemsize());
}

bool NDArray::may_share_memory(const NDArray& other) const noexcept {
  if (!shares_buffer(other)) return false;
  const ByteExtent a = byte_extent(shape_, strides_, offset_, itemsize());
  const ByteExtent b = byte_extent(other.shape_, other.strides_, other.offset_, other.itemsize());
  if (a.empty || b.empty) return false;
  return a.low < b.high && b.low < a.high;
}

NDArray NDArray::view(Shape shape, Strides strides, std::int64_t offset) const {
  return {buffer_, dtype_, std::move(shape), std::move(strides), offset, false};
}

std::int64_t NDArray::byte_offset_of(const std::vector<std::int64_t>& index) const {
  if (index.size() != ndim()) {
    throw_error(ErrorKind::Index, "expected " + std::to_string(ndim()) +
                                      " indices, got " + std::to_string(index.size()));
  }
  std::int64_t off = offset_;
  for (std::size_t d = 0; d < index.size(); ++d) {
    std::int64_t i = index[d];
    if (i < -shape_[d] || i >= shape_[d]) {
      throw_error(ErrorKind::Index, "index " + std::to_string(i) +
                                        " is out of bounds for axis " + std::to_string(d) +
                                        " with size " + std::to_string(shape_[d]));
    }
    if (i < 0) i += shape_[d];
    off += i * strides_[d];
  }
  return off;
}

std::byte* NDArray::element_ptr(std::int64_t flat) const {
  if (flat < 0 || flat >= size_) {
    throw_error(ErrorKind::Index, "flat index " + std::to_string(flat) +
                                      " is out of bounds for size " + std::to_string(size_));
  }
  std::int64_t off = offset_;
  for (std::size_t d = ndim(); d-- > 0;) {
    const std::int64_t i = flat % shape_[d];
    flat /= shape_[d];
    off += i * strides_[d];
  }
  return buffer_->data() + off;
}

namespace {
template <typename Out>
Out read_as(DType dt, const std::byte* p) {
  return dispatch_dtype(dt, [&](auto tag) -> Out {
    using T = dtype_t<decltype(tag)::value>;
    return cast_value<Out>(load<T>(p));
  });
}

template <typename In>
void write_from(DType dt, std::byte* p, In v) {
  dispatch_dtype(dt, [&](auto tag) {
    using T = dtype_t<decltype(tag)::value>;
    store<T>(p, cast_value<T>(v));
  });
}
}  // namespace

double NDArray::get_double(std::int64_t flat) const {
  return read_as<double>(dtype_, element_ptr(flat));
}
std::int64_t NDArray::get_int64(std::int64_t flat) const {
  return read_as<std::int64_t>(dtype_, element_ptr(flat));
}
std::uint64_t NDArray::get_uint64(std::int64_t flat) const {
  return read_as<std::uint64_t>(dtype_, element_ptr(flat));
}
void NDArray::set_double(std::int64_t flat, double value) {
  write_from(dtype_, element_ptr(flat), value);
}
void NDArray::set_int64(std::int64_t flat, std::int64_t value) {
  write_from(dtype_, element_ptr(flat), value);
}
void NDArray::set_uint64(std::int64_t flat, std::uint64_t value) {
  write_from(dtype_, element_ptr(flat), value);
}

NDArray NDArray::copy() const { return astype(dtype_); }

NDArray NDArray::astype(DType target) const {
  NDArray out = NDArray::empty(shape_, target);
  if (size_ == 0) return out;
  if (target == dtype_ && is_c_contiguous()) {
    std::memcpy(out.data(), data(), static_cast<std::size_t>(nbytes()));
    return out;
  }
  dispatch_dtype(dtype_, [&](auto src_tag) {
    using S = dtype_t<decltype(src_tag)::value>;
    dispatch_dtype(target, [&](auto dst_tag) {
      using D = dtype_t<decltype(dst_tag)::value>;
      std::byte* dst = out.data();
      if (is_c_contiguous()) {
        // Contiguous fast path (PLAN §80): a flat loop the compiler can
        // auto-vectorize. Same per-element cast_value as the strided path.
        const std::byte* src = data();
        const auto n = static_cast<std::size_t>(size_);
        for (std::size_t i = 0; i < n; ++i) {
          store<D>(dst + i * sizeof(D), cast_value<D>(load<S>(src + i * sizeof(S))));
        }
        return;
      }
      for_each_element(*this, [&](const std::byte* src) {
        store<D>(dst, cast_value<D>(load<S>(src)));
        dst += sizeof(D);
      });
    });
  });
  return out;
}

NDArray NDArray::reshape(const Shape& target) const {
  const Shape resolved = resolve_reshape(target, size_);
  if (size_ == 0) {
    // NumPy: empty reshapes are views with max(dim,1) C strides.
    return view(resolved, c_contiguous_strides(resolved, itemsize()), offset_);
  }
  Strides new_strides;
  if (attempt_nocopy_reshape(shape_, strides_, resolved, itemsize(), new_strides)) {
    return view(resolved, std::move(new_strides), offset_);
  }
  const NDArray contiguous = copy();
  return {contiguous.buffer_, dtype_, resolved,
          allocation_strides(resolved, itemsize()), 0, true};
}

}  // namespace nativpy
