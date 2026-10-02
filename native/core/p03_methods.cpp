#include "p03_methods.hpp"

#include <algorithm>
#include <cstring>
#include <string>

#include "broadcast.hpp"
#include "error.hpp"
#include "layout.hpp"
#include "strides.hpp"

namespace nativpy {

void fill(const NDArray& a, const NDArray& value) {
  a.check_writeable();
  if (value.size() == 0) throw_error(ErrorKind::Value, "fill value must not be empty");
  if (a.size() == 0) return;
  const NDArray first = NDArray::empty({}, a.dtype());
  const NDArray v0 = value.view({}, {}, value.offset());
  copy_into(first, v0);
  copy_into(a, first);
}

std::vector<std::byte> to_bytes(const NDArray& a, Order order) {
  if (order == Order::K) order = Order::C;
  const NDArray c = copy_order(a, a.dtype(), order);
  std::vector<std::byte> out(static_cast<std::size_t>(c.nbytes()));
  if (!out.empty()) std::memcpy(out.data(), c.data(), out.size());
  return out;
}

NDArray view_as(const NDArray& a, DType dtype) {
  const auto old_size = static_cast<std::int64_t>(a.itemsize());
  const auto new_size = static_cast<std::int64_t>(itemsize(dtype));
  if (old_size == new_size) {
    NDArray v{a.buffer(), dtype, a.shape(), a.strides(), a.offset()};
    return a.writeable() ? v : v.as_readonly();
  }
  if (a.ndim() == 0) {
    throw_error(ErrorKind::Value,
                "Changing the dtype of a 0d array is only supported if the itemsize is unchanged");
  }
  Shape shape = a.shape();
  Strides strides = a.strides();
  const std::size_t last = a.ndim() - 1;
  const bool empty = a.size() == 0;
  if (!empty && shape[last] != 1 && strides[last] != old_size) {
    throw_error(ErrorKind::Value,
                "To change to a dtype of a different size, the last axis must be contiguous");
  }
  const std::int64_t total = shape[last] * old_size;
  if (new_size > old_size && total % new_size != 0) {
    throw_error(ErrorKind::Value,
                "When changing to a larger dtype, its size must be a divisor of the total size "
                "in bytes of the last axis of the array.");
  }
  shape[last] = total / new_size;
  strides[last] = new_size;
  NDArray v{a.buffer(), dtype, shape, strides, a.offset()};
  return a.writeable() ? v : v.as_readonly();
}

namespace {
void swap_element(std::byte* p, std::size_t itemsize, bool complex_type) {
  if (complex_type) {
    const std::size_t half = itemsize / 2;
    std::reverse(p, p + half);
    std::reverse(p + half, p + itemsize);
  } else {
    std::reverse(p, p + itemsize);
  }
}
}  // namespace

NDArray byteswap(const NDArray& a, bool inplace) {
  if (inplace) {
    if (!a.writeable()) throw_error(ErrorKind::Value, "array to be byte-swapped is read-only");
  }
  const NDArray target = inplace ? a : copy_order(a, a.dtype(), Order::K);
  const std::size_t isz = target.itemsize();
  if (isz > 1) {
    const bool cplx = is_complex(target.dtype());
    for_each_element(target, [&](std::byte* p) { swap_element(p, isz, cplx); });
  }
  return target;
}

void flat_assign(const NDArray& a, const NDArray* positions, const NDArray& values) {
  a.check_writeable();
  const std::int64_t n = positions != nullptr ? positions->size() : a.size();
  if (n == 0) return;
  if (values.size() == 0) return;  // NumPy: nothing to repeat, no-op
  // Values cast to a's dtype, read in C order (cyclically).
  const NDArray vals = values.astype(a.dtype());
  const std::size_t isz = a.itemsize();
  std::vector<std::int64_t> pos;
  if (positions != nullptr) {
    const NDArray p = positions->astype(DType::Int64);
    pos.resize(static_cast<std::size_t>(n));
    if (n > 0) std::memcpy(pos.data(), p.data(), pos.size() * sizeof(std::int64_t));
    for (auto& i : pos) {
      if (i < -a.size() || i >= a.size()) {
        throw_error(ErrorKind::Index, "index " + std::to_string(i) + " is out of bounds for size " +
                                          std::to_string(a.size()));
      }
      if (i < 0) i += a.size();
    }
  }
  const auto ptr_of = [&](std::int64_t flat) {
    std::int64_t off = a.offset();
    for (std::size_t d = a.ndim(); d-- > 0;) {
      off += (flat % a.shape()[d]) * a.strides()[d];
      flat /= a.shape()[d];
    }
    return a.buffer()->data() + off;
  };
  const std::int64_t m = vals.size();
  for (std::int64_t k = 0; k < n; ++k) {
    const std::int64_t flat = positions != nullptr ? pos[static_cast<std::size_t>(k)] : k;
    std::memcpy(ptr_of(flat), vals.data() + static_cast<std::size_t>(k % m) * isz, isz);
  }
}

}  // namespace nativpy
