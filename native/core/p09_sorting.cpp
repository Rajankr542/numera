#include "p09_sorting.hpp"

#include <algorithm>
#include <numeric>
#include <string>

#include "cast.hpp"
#include "error.hpp"
#include "layout.hpp"
#include "p09_sort_kernels.hpp"
#include "shape.hpp"

namespace nativpy {

namespace {

using p09::gt;
using p09::lt;

// Byte offsets (relative to a.data()) of the first element of every 1-D lane
// along `axis`, in C order of the remaining axes.
std::vector<std::int64_t> lane_offsets(const NDArray& a, std::int64_t axis) {
  std::vector<std::int64_t> out;
  if (a.size() == 0) return out;
  const auto& shape = a.shape();
  const auto& strides = a.strides();
  const std::size_t nd = a.ndim();
  const auto ax = static_cast<std::size_t>(axis);
  out.reserve(static_cast<std::size_t>(a.size() / shape[ax]));
  std::vector<std::int64_t> idx(nd, 0);
  std::int64_t off = 0;
  for (;;) {
    out.push_back(off);
    std::size_t d = nd;
    for (; d-- > 0;) {
      if (d == ax) continue;
      if (++idx[d] < shape[d]) {
        off += strides[d];
        break;
      }
      off -= strides[d] * (shape[d] - 1);
      idx[d] = 0;
    }
    if (d == static_cast<std::size_t>(-1)) break;
  }
  return out;
}

template <typename T>
void gather(const std::byte* base, std::int64_t stride, std::int64_t n, std::vector<T>& v) {
  v.resize(static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) v[static_cast<std::size_t>(i)] = load<T>(base + i * stride);
}

template <typename T>
void scatter(std::byte* base, std::int64_t stride, const std::vector<T>& v) {
  for (std::size_t i = 0; i < v.size(); ++i) store<T>(base + static_cast<std::int64_t>(i) * stride, v[i]);
}

template <typename T>
void sort_values(std::vector<T>& v, SortKind kind, bool descending) {
  const auto n = static_cast<std::int64_t>(v.size());
  if (kind == SortKind::Stable) {
    if (descending) std::stable_sort(v.begin(), v.end(), [](const T& x, const T& y) { return gt(x, y); });
    else std::stable_sort(v.begin(), v.end(), [](const T& x, const T& y) { return lt(x, y); });
  } else if (descending) {
    p09::quicksort(v.data(), n, [](const T& x, const T& y) { return gt(x, y); });
  } else {
    p09::quicksort(v.data(), n, [](const T& x, const T& y) { return lt(x, y); });
  }
}

template <typename T>
void argsort_values(const std::vector<T>& v, std::vector<std::int64_t>& idx, SortKind kind,
                    bool descending) {
  idx.resize(v.size());
  std::iota(idx.begin(), idx.end(), std::int64_t{0});
  const auto n = static_cast<std::int64_t>(v.size());
  const auto less = [&v](std::int64_t x, std::int64_t y) {
    return lt(v[static_cast<std::size_t>(x)], v[static_cast<std::size_t>(y)]);
  };
  const auto greater = [&v](std::int64_t x, std::int64_t y) {
    return gt(v[static_cast<std::size_t>(x)], v[static_cast<std::size_t>(y)]);
  };
  if (kind == SortKind::Stable) {
    if (descending) std::stable_sort(idx.begin(), idx.end(), greater);
    else std::stable_sort(idx.begin(), idx.end(), less);
  } else if (descending) {
    p09::quicksort(idx.data(), n, greater);
  } else {
    p09::quicksort(idx.data(), n, less);
  }
}

// NumPy's introselect treats types whose storage is integral (bool, ints,
// npy_half) as exact.
template <typename T>
inline constexpr bool select_inexact_v = std::is_floating_point_v<T> || is_complex_v<T>;

template <typename Item, typename Less, bool Inexact>
void select_all(Item* data, std::int64_t n, const std::vector<std::int64_t>& kth, Less less) {
  std::int64_t pivots[p09::kMaxPivotStack];
  std::int64_t npiv = 0;
  for (const auto k : kth) p09::introselect<Inexact>(data, n, k, pivots, &npiv, less);
}

// kth normalized against the axis length and sorted (partition_prep_kth_array).
std::vector<std::int64_t> prep_kth(const std::vector<std::int64_t>& kth, std::int64_t n, std::int64_t size) {
  std::vector<std::int64_t> out = kth;
  for (auto& k : out) {
    if (k < 0) k += n;
    if (size != 0 && (k < 0 || k >= n)) {
      throw_error(ErrorKind::Value, "kth(=" + std::to_string(k) + ") out of bounds (" + std::to_string(n) + ")");
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

NDArray flat_copy(const NDArray& a) { return a.copy().reshape({-1}); }

}  // namespace

void sort_inplace(const NDArray& a, std::int64_t axis, SortKind kind, bool descending) {
  const std::int64_t ax = normalize_axis(axis, static_cast<std::int64_t>(a.ndim()));
  a.check_writeable();
  const std::int64_t n = a.shape()[static_cast<std::size_t>(ax)];
  if (n <= 1 || a.size() == 0) return;
  const std::int64_t stride = a.strides()[static_cast<std::size_t>(ax)];
  const auto lanes = lane_offsets(a, ax);
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    std::vector<T> buf;
    for (const auto off : lanes) {
      std::byte* base = a.data() + off;
      gather(base, stride, n, buf);
      sort_values(buf, kind, descending);
      scatter(base, stride, buf);
    }
  });
}

NDArray sort_copy(const NDArray& a, std::optional<std::int64_t> axis, SortKind kind, bool descending) {
  if (!axis) {
    NDArray out = flat_copy(a);
    sort_inplace(out, 0, kind, descending);
    return out;
  }
  normalize_axis(*axis, static_cast<std::int64_t>(a.ndim()));
  NDArray out = copy_order(a, a.dtype(), Order::K);
  sort_inplace(out, *axis, kind, descending);
  return out;
}

NDArray argsort(const NDArray& a, std::optional<std::int64_t> axis, SortKind kind, bool descending) {
  const NDArray src = axis ? a : a.reshape({-1});
  const std::int64_t ax = normalize_axis(axis.value_or(0), static_cast<std::int64_t>(src.ndim()));
  NDArray out = NDArray::zeros(src.shape(), DType::Int64);
  const std::int64_t n = src.shape()[static_cast<std::size_t>(ax)];
  if (src.size() == 0 || n <= 1) return out;
  const std::int64_t stride = src.strides()[static_cast<std::size_t>(ax)];
  const std::int64_t ostride = out.strides()[static_cast<std::size_t>(ax)];
  const auto lanes = lane_offsets(src, ax);
  const auto olanes = lane_offsets(out, ax);
  dispatch_dtype(src.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    std::vector<T> buf;
    std::vector<std::int64_t> idx;
    for (std::size_t l = 0; l < lanes.size(); ++l) {
      gather(src.data() + lanes[l], stride, n, buf);
      argsort_values(buf, idx, kind, descending);
      scatter(out.data() + olanes[l], ostride, idx);
    }
  });
  return out;
}

void partition_inplace(const NDArray& a, const std::vector<std::int64_t>& kth, std::int64_t axis) {
  const std::int64_t ax = normalize_axis(axis, static_cast<std::int64_t>(a.ndim()));
  a.check_writeable();
  const std::int64_t n = a.shape()[static_cast<std::size_t>(ax)];
  const auto ks = prep_kth(kth, n, a.size());
  if (n <= 1 || a.size() == 0) return;
  const std::int64_t stride = a.strides()[static_cast<std::size_t>(ax)];
  const auto lanes = lane_offsets(a, ax);
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    std::vector<T> buf;
    const auto less = [](const T& x, const T& y) { return lt(x, y); };
    for (const auto off : lanes) {
      std::byte* base = a.data() + off;
      gather(base, stride, n, buf);
      select_all<T, decltype(less), select_inexact_v<T>>(buf.data(), n, ks, less);
      scatter(base, stride, buf);
    }
  });
}

NDArray partition_copy(const NDArray& a, const std::vector<std::int64_t>& kth,
                       std::optional<std::int64_t> axis) {
  if (!axis) {
    NDArray out = flat_copy(a);
    partition_inplace(out, kth, 0);
    return out;
  }
  normalize_axis(*axis, static_cast<std::int64_t>(a.ndim()));
  NDArray out = copy_order(a, a.dtype(), Order::K);
  partition_inplace(out, kth, *axis);
  return out;
}

NDArray argpartition(const NDArray& a, const std::vector<std::int64_t>& kth,
                     std::optional<std::int64_t> axis) {
  const NDArray src = axis ? a : a.reshape({-1});
  const std::int64_t ax = normalize_axis(axis.value_or(0), static_cast<std::int64_t>(src.ndim()));
  const std::int64_t n = src.shape()[static_cast<std::size_t>(ax)];
  const auto ks = prep_kth(kth, n, src.size());
  NDArray out = NDArray::zeros(src.shape(), DType::Int64);
  if (src.size() == 0 || n <= 1) return out;
  const std::int64_t stride = src.strides()[static_cast<std::size_t>(ax)];
  const std::int64_t ostride = out.strides()[static_cast<std::size_t>(ax)];
  const auto lanes = lane_offsets(src, ax);
  const auto olanes = lane_offsets(out, ax);
  dispatch_dtype(src.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    std::vector<T> buf;
    std::vector<std::int64_t> idx(static_cast<std::size_t>(n));
    const auto less = [&buf](std::int64_t x, std::int64_t y) {
      return lt(buf[static_cast<std::size_t>(x)], buf[static_cast<std::size_t>(y)]);
    };
    for (std::size_t l = 0; l < lanes.size(); ++l) {
      gather(src.data() + lanes[l], stride, n, buf);
      std::iota(idx.begin(), idx.end(), std::int64_t{0});
      select_all<std::int64_t, decltype(less), select_inexact_v<T>>(idx.data(), n, ks, less);
      scatter(out.data() + olanes[l], ostride, idx);
    }
  });
  return out;
}

NDArray lexsort(const std::vector<NDArray>& keys, std::int64_t axis) {
  if (keys.empty()) throw_error(ErrorKind::DType, "need sequence of keys with len > 0 in lexsort");
  const Shape& shape = keys[0].shape();
  for (const auto& k : keys) {
    if (k.shape() != shape) throw_error(ErrorKind::Value, "all keys need to be the same shape");
  }
  if (shape.empty()) return NDArray::zeros({}, DType::Int64);
  const std::int64_t ax = normalize_axis(axis, static_cast<std::int64_t>(shape.size()));
  NDArray out = NDArray::zeros(shape, DType::Int64);
  const std::int64_t n = shape[static_cast<std::size_t>(ax)];
  if (out.size() == 0) return out;
  const auto olanes = lane_offsets(out, ax);
  const std::int64_t ostride = out.strides()[static_cast<std::size_t>(ax)];
  std::vector<std::vector<std::int64_t>> klanes;
  klanes.reserve(keys.size());
  for (const auto& k : keys) klanes.push_back(lane_offsets(k, ax));
  std::vector<std::int64_t> idx(static_cast<std::size_t>(n));
  for (std::size_t l = 0; l < olanes.size(); ++l) {
    std::iota(idx.begin(), idx.end(), std::int64_t{0});
    // Successive stable sorts, least significant (first) key first.
    for (std::size_t k = 0; k < keys.size(); ++k) {
      const NDArray& key = keys[k];
      const std::int64_t stride = key.strides()[static_cast<std::size_t>(ax)];
      const std::byte* base = key.data() + klanes[k][l];
      dispatch_dtype(key.dtype(), [&](auto tag) {
        using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
        std::vector<T> buf;
        gather(base, stride, n, buf);
        std::stable_sort(idx.begin(), idx.end(), [&buf](std::int64_t x, std::int64_t y) {
          return lt(buf[static_cast<std::size_t>(x)], buf[static_cast<std::size_t>(y)]);
        });
      });
    }
    scatter(out.data() + olanes[l], ostride, idx);
  }
  return out;
}

NDArray searchsorted(const NDArray& a, const NDArray& v, bool right,
                     const std::optional<NDArray>& sorter) {
  if (a.ndim() != 1) {
    throw_error(ErrorKind::Value, a.ndim() > 1 ? "object too deep for desired array"
                                               : "object of too small depth for desired array");
  }
  if (a.dtype() != v.dtype()) throw_error(ErrorKind::DType, "searchsorted: a and v must have the same dtype");
  const std::int64_t n = a.size();
  if (sorter) {
    if (sorter->ndim() != 1) throw_error(ErrorKind::DType, "could not parse sorter argument");
    if (!is_integer(sorter->dtype())) throw_error(ErrorKind::DType, "sorter must only contain integers");
    if (sorter->size() != n) throw_error(ErrorKind::Value, "sorter.size must equal a.size");
  }
  NDArray out = NDArray::zeros(v.shape(), DType::Int64);
  if (n == 0 || v.size() == 0) return out;
  std::vector<std::int64_t> perm;
  if (sorter) {
    perm.resize(static_cast<std::size_t>(n));
    const NDArray s = sorter->astype(DType::Int64);
    const auto* sp = reinterpret_cast<const std::int64_t*>(s.data());
    std::copy(sp, sp + n, perm.begin());
  }
  auto* dst = reinterpret_cast<std::int64_t*>(out.data());
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    std::vector<T> arr;
    gather(a.data(), a.strides()[0], n, arr);
    const auto at = [&](std::int64_t i) -> const T& {
      if (!sorter) return arr[static_cast<std::size_t>(i)];
      const std::int64_t j = perm[static_cast<std::size_t>(i)];
      if (j < 0 || j >= n) throw_error(ErrorKind::Value, "Sorter index out of range.");
      return arr[static_cast<std::size_t>(j)];
    };
    const auto cmp = [right](const T& x, const T& key) -> std::int64_t {
      return right ? !lt(key, x) : lt(x, key);
    };
    std::int64_t k = 0;
    for_each_element(v, [&](const std::byte* p) {
      const T key = load<T>(p);
      std::int64_t len = n;
      std::int64_t half = len >> 1;
      len -= half;
      std::int64_t base = cmp(at(half), key) * half;
      while (len > 1) {
        half = len >> 1;
        len -= half;
        base += cmp(at(base + half), key) * half;
      }
      base += cmp(at(base), key);
      dst[k++] = base;
    });
  });
  return out;
}

}  // namespace nativpy
