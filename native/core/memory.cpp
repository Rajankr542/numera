#include "memory.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <new>
#include <string>

#include "error.hpp"

namespace nativpy {

namespace {
std::atomic<std::int64_t> g_live_buffers{0};
std::atomic<std::int64_t> g_live_bytes{0};
}  // namespace

MemoryBuffer::MemoryBuffer(std::byte* data, std::size_t size) noexcept
    : data_(data), size_(size) {
  g_live_buffers.fetch_add(1, std::memory_order_relaxed);
  g_live_bytes.fetch_add(static_cast<std::int64_t>(size),
                         std::memory_order_relaxed);
}

MemoryBuffer::~MemoryBuffer() {
  ::operator delete(data_, std::align_val_t{kAlignment});
  g_live_buffers.fetch_sub(1, std::memory_order_relaxed);
  g_live_bytes.fetch_sub(static_cast<std::int64_t>(size_),
                         std::memory_order_relaxed);
}

std::shared_ptr<MemoryBuffer> MemoryBuffer::allocate(std::size_t nbytes,
                                                     bool zero_fill) {
  // Always allocate at least one alignment unit so data() is never null.
  const std::size_t alloc = nbytes == 0 ? kAlignment : nbytes;
  std::byte* ptr = nullptr;
  try {
    ptr = static_cast<std::byte*>(
        ::operator new(alloc, std::align_val_t{kAlignment}));
  } catch (const std::bad_alloc&) {
    throw_error(ErrorKind::Memory,
                "unable to allocate " + std::to_string(nbytes) + " bytes");
  }
  if (zero_fill) {
    std::memset(ptr, 0, alloc);
  }
  try {
    return std::shared_ptr<MemoryBuffer>(new MemoryBuffer(ptr, nbytes));
  } catch (...) {
    ::operator delete(ptr, std::align_val_t{kAlignment});
    throw;
  }
}

std::int64_t MemoryBuffer::live_buffers() noexcept {
  return g_live_buffers.load(std::memory_order_relaxed);
}

std::int64_t MemoryBuffer::live_bytes() noexcept {
  return g_live_bytes.load(std::memory_order_relaxed);
}

}  // namespace nativpy
