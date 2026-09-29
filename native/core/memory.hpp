#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace nativpy {

// Owns a single aligned, zero-initializable heap allocation (RAII).
// Non-copyable, non-movable: always shared through std::shared_ptr.
class MemoryBuffer {
 public:
  static constexpr std::size_t kAlignment = 64;

  // Allocates `nbytes` (0 allowed). Throws Error(Memory) on failure.
  static std::shared_ptr<MemoryBuffer> allocate(std::size_t nbytes,
                                                bool zero_fill);

  ~MemoryBuffer();
  MemoryBuffer(const MemoryBuffer&) = delete;
  MemoryBuffer& operator=(const MemoryBuffer&) = delete;
  MemoryBuffer(MemoryBuffer&&) = delete;
  MemoryBuffer& operator=(MemoryBuffer&&) = delete;

  [[nodiscard]] std::byte* data() noexcept { return data_; }
  [[nodiscard]] const std::byte* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return size_; }

  // Development instrumentation (PLAN §33): live buffer count/bytes.
  static std::int64_t live_buffers() noexcept;
  static std::int64_t live_bytes() noexcept;

 private:
  MemoryBuffer(void* raw, std::byte* data, std::size_t size) noexcept;

  void* raw_;  // pointer returned by malloc/calloc (freed in the destructor)
  std::byte* data_;  // raw_ rounded up to kAlignment
  std::size_t size_;
};

}  // namespace nativpy
