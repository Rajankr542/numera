#include "memory.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <string>

#include "error.hpp"

namespace nativpy {

namespace {
std::atomic<std::int64_t> g_live_buffers{0};
std::atomic<std::int64_t> g_live_bytes{0};
}  // namespace

MemoryBuffer::MemoryBuffer(void* raw, std::byte* data, std::size_t size) noexcept
    : raw_(raw), data_(data), size_(size) {
  g_live_buffers.fetch_add(1, std::memory_order_relaxed);
  g_live_bytes.fetch_add(static_cast<std::int64_t>(size),
                         std::memory_order_relaxed);
}

MemoryBuffer::~MemoryBuffer() {
  std::free(raw_);
  g_live_buffers.fetch_sub(1, std::memory_order_relaxed);
  g_live_bytes.fetch_sub(static_cast<std::int64_t>(size_),
                         std::memory_order_relaxed);
}

std::shared_ptr<MemoryBuffer> MemoryBuffer::allocate(std::size_t nbytes,
                                                     bool zero_fill) {
  // D-013: over-allocate by kAlignment and align by hand, so zero-filled
  // buffers can use calloc (the OS hands out lazily zeroed pages for large
  // blocks instead of memset touching every page). Always at least one
  // alignment unit so data() is never null.
  const std::size_t payload = nbytes == 0 ? kAlignment : nbytes;
  if (payload > std::numeric_limits<std::size_t>::max() - kAlignment) {
    throw_error(ErrorKind::Memory, "unable to allocate " + std::to_string(nbytes) + " bytes");
  }
  const std::size_t total = payload + kAlignment;
  void* raw = zero_fill ? std::calloc(total, 1) : std::malloc(total);
  if (raw == nullptr) {
    throw_error(ErrorKind::Memory,
                "unable to allocate " + std::to_string(nbytes) + " bytes");
  }
  const auto addr = reinterpret_cast<std::uintptr_t>(raw);
  const std::uintptr_t aligned = (addr + kAlignment - 1) & ~(std::uintptr_t{kAlignment} - 1);
  auto* ptr = reinterpret_cast<std::byte*>(aligned);
  try {
    return std::shared_ptr<MemoryBuffer>(new MemoryBuffer(raw, ptr, nbytes));
  } catch (...) {
    std::free(raw);
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
