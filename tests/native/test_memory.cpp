#include <cstdint>

#include "memory.hpp"
#include "test_harness.hpp"

using nativpy::MemoryBuffer;

TEST_CASE("memory: aligned allocation and zero fill") {
  auto buf = MemoryBuffer::allocate(1000, true);
  CHECK(buf->data() != nullptr);
  CHECK_EQ(buf->size(), std::size_t{1000});
  CHECK_EQ(reinterpret_cast<std::uintptr_t>(buf->data()) % MemoryBuffer::kAlignment,
           std::uintptr_t{0});
  bool all_zero = true;
  for (std::size_t i = 0; i < buf->size(); ++i) {
    all_zero = all_zero && buf->data()[i] == std::byte{0};
  }
  CHECK(all_zero);
}

TEST_CASE("memory: zero-byte allocation is valid and non-null") {
  auto buf = MemoryBuffer::allocate(0, false);
  CHECK(buf->data() != nullptr);
  CHECK_EQ(buf->size(), std::size_t{0});
}

TEST_CASE("memory: live buffer accounting is released on destruction") {
  const auto before = MemoryBuffer::live_buffers();
  const auto before_bytes = MemoryBuffer::live_bytes();
  {
    auto a = MemoryBuffer::allocate(128, false);
    auto b = a;  // shared ownership, not a new buffer
    CHECK_EQ(MemoryBuffer::live_buffers(), before + 1);
    CHECK_EQ(MemoryBuffer::live_bytes(), before_bytes + 128);
  }
  CHECK_EQ(MemoryBuffer::live_buffers(), before);
  CHECK_EQ(MemoryBuffer::live_bytes(), before_bytes);
}
