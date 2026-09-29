#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include "dtype.hpp"
#include "memory.hpp"
#include "shape.hpp"
#include "strides.hpp"

namespace nativpy {

// N-dimensional strided view over a shared MemoryBuffer (PLAN §8, D-003).
// Strides and offset are expressed in bytes.
class NDArray {
 public:
  // Allocates a new C-contiguous array.
  static NDArray empty(const Shape& shape, DType dtype);
  static NDArray zeros(const Shape& shape, DType dtype);

  // Creates a view over an existing buffer. Validates bounds.
  NDArray(std::shared_ptr<MemoryBuffer> buffer, DType dtype, Shape shape,
          Strides strides, std::int64_t offset);

  [[nodiscard]] DType dtype() const noexcept { return dtype_; }
  [[nodiscard]] std::size_t itemsize() const noexcept { return nativpy::itemsize(dtype_); }
  [[nodiscard]] const Shape& shape() const noexcept { return shape_; }
  [[nodiscard]] const Strides& strides() const noexcept { return strides_; }
  [[nodiscard]] std::int64_t offset() const noexcept { return offset_; }
  [[nodiscard]] std::size_t ndim() const noexcept { return shape_.size(); }
  [[nodiscard]] std::int64_t size() const noexcept { return size_; }
  [[nodiscard]] std::int64_t nbytes() const noexcept {
    return size_ * static_cast<std::int64_t>(itemsize());
  }
  [[nodiscard]] const std::shared_ptr<MemoryBuffer>& buffer() const noexcept { return buffer_; }

  [[nodiscard]] std::byte* data() const noexcept { return buffer_->data() + offset_; }

  [[nodiscard]] bool is_c_contiguous() const noexcept;
  [[nodiscard]] bool is_f_contiguous() const noexcept;
  [[nodiscard]] bool owns_data() const noexcept { return owns_data_; }

  // True if both arrays reference the same MemoryBuffer.
  [[nodiscard]] bool shares_buffer(const NDArray& other) const noexcept {
    return buffer_ == other.buffer_;
  }

  // NumPy may_share_memory (bounds check): same buffer, both non-empty and
  // overlapping byte extents (D-011).
  [[nodiscard]] bool may_share_memory(const NDArray& other) const noexcept;

  // New view on the same buffer (bounds-checked). Offset is absolute bytes.
  [[nodiscard]] NDArray view(Shape shape, Strides strides, std::int64_t offset) const;

  // Byte offset (absolute, into buffer) of the element at a multi-index.
  [[nodiscard]] std::int64_t byte_offset_of(const std::vector<std::int64_t>& index) const;

  // Element access by flat C-order position (0 <= flat < size).
  [[nodiscard]] double get_double(std::int64_t flat) const;
  [[nodiscard]] std::int64_t get_int64(std::int64_t flat) const;
  [[nodiscard]] std::uint64_t get_uint64(std::int64_t flat) const;
  void set_double(std::int64_t flat, double value);
  void set_int64(std::int64_t flat, std::int64_t value);
  void set_uint64(std::int64_t flat, std::uint64_t value);

  // C-contiguous deep copy (same dtype).
  [[nodiscard]] NDArray copy() const;
  // C-contiguous converted copy (unsafe casting, like NumPy astype default).
  [[nodiscard]] NDArray astype(DType dtype) const;
  // View if C-contiguous, otherwise copy then view. Supports one -1.
  [[nodiscard]] NDArray reshape(const Shape& shape) const;

 private:
  NDArray(std::shared_ptr<MemoryBuffer> buffer, DType dtype, Shape shape,
          Strides strides, std::int64_t offset, bool owns_data);
  [[nodiscard]] std::byte* element_ptr(std::int64_t flat) const;

  std::shared_ptr<MemoryBuffer> buffer_;
  DType dtype_;
  Shape shape_;
  Strides strides_;
  std::int64_t offset_;
  std::int64_t size_;
  bool owns_data_;
};

// Calls fn(byte_ptr) for every element in C order. Generic strided iteration.
template <typename Fn>
void for_each_element(const NDArray& a, Fn&& fn) {
  const std::int64_t n = a.size();
  if (n == 0) return;
  const std::size_t nd = a.ndim();
  const auto& shape = a.shape();
  const auto& strides = a.strides();
  std::vector<std::int64_t> idx(nd, 0);
  std::byte* ptr = a.data();
  for (std::int64_t i = 0; i < n; ++i) {
    fn(ptr);
    for (std::size_t d = nd; d-- > 0;) {
      if (++idx[d] < shape[d]) {
        ptr += strides[d];
        break;
      }
      ptr -= strides[d] * (shape[d] - 1);
      idx[d] = 0;
    }
  }
}

}  // namespace nativpy
