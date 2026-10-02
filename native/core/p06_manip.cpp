#include "p06_manip.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"
#include "indexing.hpp"
#include "layout.hpp"
#include "shape.hpp"
#include "shape_ops.hpp"
#include "strides.hpp"

namespace nativpy {

namespace {

std::string dtype_repr(DType dt) { return "dtype('" + std::string(dtype_name(dt)) + "')"; }

void check_cast(DType from, DType to, Casting casting) {
  if (!can_cast(from, to, casting)) {
    throw_error(ErrorKind::DType, "Cannot cast array data from " + dtype_repr(from) + " to " +
                                      dtype_repr(to) + " according to the rule '" +
                                      std::string(casting_name(casting)) + "'");
  }
}

// Dense strides with `perm` listed outermost -> innermost.
Strides strides_from_perm(const Shape& shape, std::size_t itemsize,
                          const std::vector<std::size_t>& perm) {
  if (shape_size(shape) == 0) return allocation_strides(shape, itemsize);
  Strides st(shape.size(), 0);
  auto acc = static_cast<std::int64_t>(itemsize);
  for (std::size_t i = perm.size(); i-- > 0;) {
    st[perm[i]] = acc;
    acc *= shape[perm[i]];
  }
  return st;
}

}  // namespace

NDArray slice_axis(const NDArray& a, std::size_t axis, std::int64_t start, std::int64_t stop,
                   std::int64_t step) {
  Shape shape = a.shape();
  Strides strides = a.strides();
  std::int64_t len = 0;
  if (step > 0 && stop > start) len = (stop - start + step - 1) / step;
  shape[axis] = len;
  const std::int64_t offset = a.offset() + (len > 0 ? start * strides[axis] : 0);
  strides[axis] *= step;
  return a.view(std::move(shape), std::move(strides), offset);
}

Strides multi_sorted_strides(const Shape& shape, std::size_t itemsize,
                             const std::vector<const NDArray*>& like) {
  // NumPy PyArray_CreateMultiSortedStridePerm: insertion sort of the axes by
  // |stride| (largest first), skipping size-1 axes; ambiguity keeps C order.
  const std::size_t nd = shape.size();
  std::vector<std::size_t> perm(nd);
  for (std::size_t i = 0; i < nd; ++i) perm[i] = i;
  for (std::size_t i0 = 1; i0 < nd; ++i0) {
    std::size_t ipos = i0;
    const std::size_t j0 = perm[i0];
    for (std::size_t i1 = i0; i1-- > 0;) {
      const std::size_t j1 = perm[i1];
      bool ambig = true;
      bool swap = false;
      for (const NDArray* a : like) {
        if (a->shape()[j0] != 1 && a->shape()[j1] != 1) {
          if (std::llabs(a->strides()[j0]) <= std::llabs(a->strides()[j1])) {
            swap = false;
          } else if (ambig) {
            swap = true;
          }
          ambig = false;
        }
      }
      if (!ambig) {
        if (swap) ipos = i1;
        else break;
      }
    }
    if (ipos != i0) {
      for (std::size_t i = i0; i > ipos; --i) perm[i] = perm[i - 1];
      perm[ipos] = j0;
    }
  }
  return strides_from_perm(shape, itemsize, perm);
}

// ---- join / split ----

NDArray concatenate(const std::vector<NDArray>& arrays_in, std::optional<std::int64_t> axis_in,
                    std::optional<DType> dtype, Casting casting, const NDArray* out) {
  if (arrays_in.empty()) throw_error(ErrorKind::Value, "need at least one array to concatenate");
  if (out != nullptr && dtype) {
    throw_error(ErrorKind::DType,
                "concatenate() only takes `out` or `dtype` as an argument, but both were provided.");
  }
  std::vector<NDArray> arrays;
  arrays.reserve(arrays_in.size());
  if (!axis_in) {
    for (const auto& a : arrays_in) arrays.push_back(ravel(a));
  } else {
    arrays = arrays_in;
  }
  const NDArray& first = arrays[0];
  const std::size_t nd = first.ndim();
  if (nd == 0) throw_error(ErrorKind::Value, "zero-dimensional arrays cannot be concatenated");
  const auto axis =
      static_cast<std::size_t>(normalize_axis(axis_in.value_or(0), static_cast<std::int64_t>(nd)));
  Shape shape = first.shape();
  DType res = first.dtype();
  for (std::size_t k = 1; k < arrays.size(); ++k) {
    const NDArray& a = arrays[k];
    if (a.ndim() != nd) {
      throw_error(ErrorKind::Value,
                  "all the input arrays must have same number of dimensions, but the array at "
                  "index 0 has " + std::to_string(nd) + " dimension(s) and the array at index " +
                      std::to_string(k) + " has " + std::to_string(a.ndim()) + " dimension(s)");
    }
    for (std::size_t d = 0; d < nd; ++d) {
      if (d == axis) continue;
      if (a.shape()[d] != shape[d]) {
        throw_error(ErrorKind::Value,
                    "all the input array dimensions except for the concatenation axis must match "
                    "exactly, but along dimension " + std::to_string(d) +
                        ", the array at index 0 has size " + std::to_string(shape[d]) +
                        " and the array at index " + std::to_string(k) + " has size " +
                        std::to_string(a.shape()[d]));
      }
    }
    shape[axis] += a.shape()[axis];
    res = promote_types(res, a.dtype());
  }
  if (dtype) res = *dtype;
  NDArray result = [&]() {
    if (out != nullptr) {
      out->check_writeable();
      if (out->shape() != shape) throw_error(ErrorKind::Value, "Output array is the wrong shape");
      return *out;
    }
    std::vector<const NDArray*> like;
    for (const auto& a : arrays) like.push_back(&a);
    return NDArray::empty_strided(shape, multi_sorted_strides(shape, itemsize(res), like), res);
  }();
  for (const auto& a : arrays) check_cast(a.dtype(), result.dtype(), casting);
  std::int64_t pos = 0;
  for (const auto& a : arrays) {
    const std::int64_t n = a.shape()[axis];
    if (a.size() > 0) copy_into(slice_axis(result, axis, pos, pos + n), a);
    pos += n;
  }
  return result;
}

NDArray stack(const std::vector<NDArray>& arrays, std::int64_t axis, std::optional<DType> dtype,
              Casting casting, const NDArray* out) {
  if (arrays.empty()) throw_error(ErrorKind::Value, "need at least one array to stack");
  for (const auto& a : arrays) {
    if (a.shape() != arrays[0].shape()) {
      throw_error(ErrorKind::Value, "all input arrays must have the same shape");
    }
  }
  const auto ax = normalize_axis(axis, static_cast<std::int64_t>(arrays[0].ndim() + 1));
  std::vector<NDArray> expanded;
  expanded.reserve(arrays.size());
  for (const auto& a : arrays) expanded.push_back(expand_dims(a, {ax}));
  return concatenate(expanded, ax, dtype, casting, out);
}

std::vector<NDArray> split_at(const NDArray& a, const std::vector<std::int64_t>& indices,
                              std::int64_t axis) {
  const auto ax = static_cast<std::size_t>(normalize_axis(axis, static_cast<std::int64_t>(a.ndim())));
  const std::int64_t n = a.shape()[ax];
  std::vector<NDArray> out;
  out.reserve(indices.size() + 1);
  std::optional<std::int64_t> prev = 0;
  for (std::size_t k = 0; k <= indices.size(); ++k) {
    const std::optional<std::int64_t> next =
        k < indices.size() ? std::optional<std::int64_t>(indices[k]) : std::optional<std::int64_t>(n);
    const SliceBounds b = slice_indices(prev, next, std::nullopt, n);
    out.push_back(slice_axis(a, ax, b.start, b.start + b.length));
    prev = next;
  }
  return out;
}

std::vector<NDArray> split_sections(const NDArray& a, std::int64_t sections, std::int64_t axis,
                                    bool equal) {
  const auto ax = static_cast<std::size_t>(normalize_axis(axis, static_cast<std::int64_t>(a.ndim())));
  if (sections <= 0) throw_error(ErrorKind::Value, "number sections must be larger than 0.");
  const std::int64_t n = a.shape()[ax];
  if (equal && n % sections != 0) {
    throw_error(ErrorKind::Value, "array split does not result in an equal division");
  }
  const std::int64_t each = n / sections;
  const std::int64_t extras = n % sections;
  std::vector<NDArray> out;
  out.reserve(static_cast<std::size_t>(sections));
  std::int64_t pos = 0;
  for (std::int64_t k = 0; k < sections; ++k) {
    const std::int64_t len = each + (k < extras ? 1 : 0);
    out.push_back(slice_axis(a, ax, pos, pos + len));
    pos += len;
  }
  return out;
}

std::vector<NDArray> unstack(const NDArray& a, std::int64_t axis) {
  if (a.ndim() == 0) throw_error(ErrorKind::Value, "Input array must be at least 1-d.");
  const auto ax = static_cast<std::size_t>(normalize_axis(axis, static_cast<std::int64_t>(a.ndim())));
  std::vector<NDArray> out;
  for (std::int64_t i = 0; i < a.shape()[ax]; ++i) {
    NDArray s = slice_axis(a, ax, i, i + 1);
    out.push_back(squeeze(s, std::vector<std::int64_t>{static_cast<std::int64_t>(ax)}));
  }
  return out;
}

}  // namespace nativpy

// ---- tile / repeat / resize (D-091, D-093) ----

namespace nativpy {

NDArray tile(const NDArray& a, const std::vector<std::int64_t>& reps_in) {
  for (const auto r : reps_in) {
    if (r < 0) throw_error(ErrorKind::Value, "negative dimensions are not allowed");
  }
  const std::size_t nd = std::max(a.ndim(), reps_in.size());
  if (nd > kMaxDims) throw_error(ErrorKind::Value, "too many dimensions");
  std::vector<std::int64_t> reps(nd - reps_in.size(), 1);
  reps.insert(reps.end(), reps_in.begin(), reps_in.end());
  Shape src(nd - a.ndim(), 1);
  src.insert(src.end(), a.shape().begin(), a.shape().end());
  if (std::all_of(reps.begin(), reps.end(), [](std::int64_t r) { return r == 1; })) {
    return copy_order(a, a.dtype(), Order::K).reshape(src);
  }
  Shape shape(nd);
  for (std::size_t d = 0; d < nd; ++d) shape[d] = src[d] * reps[d];
  NDArray out = NDArray::empty(shape, a.dtype());
  if (out.size() == 0) return out;
  // View out as (r0, s0, r1, s1, ...) and broadcast `a` (1, s0, 1, s1, ...) into it.
  Shape vshape;
  Strides vstrides;
  Shape ashape;
  for (std::size_t d = 0; d < nd; ++d) {
    const std::int64_t st = out.strides()[d];
    vshape.push_back(reps[d]);
    vstrides.push_back(st * src[d]);
    vshape.push_back(src[d]);
    vstrides.push_back(st);
    ashape.push_back(1);
    ashape.push_back(src[d]);
  }
  copy_into(out.view(vshape, vstrides, out.offset()), a.reshape(src).reshape(ashape));
  return out;
}

NDArray repeat(const NDArray& a_in, const std::vector<std::int64_t>& repeats,
               std::optional<std::int64_t> axis_in) {
  const NDArray a = axis_in ? a_in : ravel(a_in);
  if (a.ndim() == 0) throw_error(ErrorKind::Index, "axis 0 is out of bounds for array of dimension 0");
  const auto axis = static_cast<std::size_t>(
      normalize_axis(axis_in.value_or(0), static_cast<std::int64_t>(a.ndim())));
  const std::int64_t n = a.shape()[axis];
  if (repeats.size() != 1 && static_cast<std::int64_t>(repeats.size()) != n) {
    throw_error(ErrorKind::Value, "operands could not be broadcast together with shape (" +
                                      std::to_string(n) + ",) (" +
                                      std::to_string(repeats.size()) + ",)");
  }
  std::int64_t total = 0;
  for (const auto r : repeats) {
    if (r < 0) throw_error(ErrorKind::Value, "repeats may not contain negative values.");
  }
  for (std::int64_t i = 0; i < n; ++i) {
    total += repeats.size() == 1 ? repeats[0] : repeats[static_cast<std::size_t>(i)];
  }
  Shape shape = a.shape();
  shape[axis] = total;
  NDArray out = NDArray::empty(shape, a.dtype());
  if (out.size() == 0) return out;
  const NDArray src = a.is_c_contiguous() ? a : a.copy();
  std::int64_t outer = 1;
  for (std::size_t d = 0; d < axis; ++d) outer *= shape[d];
  std::int64_t chunk = static_cast<std::int64_t>(a.itemsize());
  for (std::size_t d = axis + 1; d < shape.size(); ++d) chunk *= shape[d];
  const auto bytes = static_cast<std::size_t>(chunk);
  const std::byte* sp = src.data();
  std::byte* dp = out.data();
  for (std::int64_t o = 0; o < outer; ++o) {
    for (std::int64_t i = 0; i < n; ++i) {
      const std::int64_t r = repeats.size() == 1 ? repeats[0] : repeats[static_cast<std::size_t>(i)];
      for (std::int64_t k = 0; k < r; ++k) {
        std::memcpy(dp, sp, bytes);
        dp += chunk;
      }
      sp += chunk;
    }
  }
  return out;
}

NDArray resize(const NDArray& a, const Shape& shape) {
  for (const auto d : shape) {
    if (d < 0) throw_error(ErrorKind::Value, "all elements of `new_shape` must be non-negative");
  }
  const std::int64_t n = shape_size(shape);
  if (a.size() == 0 || n == 0) return NDArray::zeros(shape, a.dtype());
  const NDArray flat = flatten(a);
  NDArray out = NDArray::empty(shape, a.dtype());
  const auto total = static_cast<std::size_t>(n) * a.itemsize();
  const auto block = static_cast<std::size_t>(flat.nbytes());
  for (std::size_t pos = 0; pos < total; pos += block) {
    std::memcpy(out.data() + pos, flat.data(), std::min(block, total - pos));
  }
  return out;
}

NDArray resize_inplace_data(const NDArray& a, const Shape& shape) {
  if (!a.is_c_contiguous() && !a.is_f_contiguous()) {
    throw_error(ErrorKind::Value, "resize only works on single-segment arrays");
  }
  if (!a.owns_data()) {
    throw_error(ErrorKind::Value, "cannot resize this array: it does not own its data");
  }
  a.check_writeable();
  const Order order = (a.is_f_contiguous() && !a.is_c_contiguous()) ? Order::F : Order::C;
  for (const auto d : shape) {
    if (d < 0) throw_error(ErrorKind::Value, "negative dimensions not allowed");
  }
  NDArray out = empty_order(shape, a.dtype(), order, true);
  const auto bytes = static_cast<std::size_t>(std::min(a.nbytes(), out.nbytes()));
  if (bytes > 0) std::memcpy(out.data(), a.data(), bytes);
  return out;
}

}  // namespace nativpy

// ---- pad (D-092) ----

namespace nativpy {

namespace {

// Calls f(base, stride, length) for every 1-d lane of `v` along `axis`.
template <typename F>
void for_each_lane(const NDArray& v, std::size_t axis, F&& f) {
  const std::size_t nd = v.ndim();
  Shape outer;
  Strides ostr;
  for (std::size_t d = 0; d < nd; ++d) {
    if (d == axis) continue;
    if (v.shape()[d] == 0) return;
    outer.push_back(v.shape()[d]);
    ostr.push_back(v.strides()[d]);
  }
  const std::int64_t len = v.shape()[axis];
  const std::int64_t stride = v.strides()[axis];
  std::vector<std::int64_t> idx(outer.size(), 0);
  std::byte* p = v.data();
  for (;;) {
    f(p, stride, len);
    std::size_t d = outer.size();
    for (; d-- > 0;) {
      if (++idx[d] < outer[d]) {
        p += ostr[d];
        break;
      }
      p -= ostr[d] * (outer[d] - 1);
      idx[d] = 0;
    }
    if (d == static_cast<std::size_t>(-1)) return;
  }
}

template <typename T>
using work_t = std::conditional_t<is_complex_v<T>, std::complex<double>, double>;

template <typename T>
work_t<T> to_work(T v) {
  if constexpr (is_complex_v<T>) {
    return {static_cast<double>(v.real()), static_cast<double>(v.imag())};
  } else {
    return cast_value<double>(v);
  }
}

template <typename T>
bool is_nan_v(T v) {
  if constexpr (is_complex_v<T>) {
    return std::isnan(v.real()) || std::isnan(v.imag());
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return std::isnan(half_to_double(v));
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(v);
  } else {
    return false;
  }
}

// Ordering used by maximum/minimum/median (complex: lexicographic, as NumPy).
template <typename T>
bool less_v(T a, T b) {
  if constexpr (is_complex_v<T>) {
    return a.real() < b.real() || (a.real() == b.real() && a.imag() < b.imag());
  } else if constexpr (std::is_same_v<T, float16_t>) {
    return half_to_double(a) < half_to_double(b);
  } else {
    return a < b;
  }
}

// 2 * edge - v in the array dtype (integers wrap, bool via nonzero).
template <typename T>
T odd_reflect(T edge, T v) {
  if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, float16_t>) {
    return cast_value<T>(2.0 * cast_value<double>(edge) - cast_value<double>(v));
  } else if constexpr (std::is_integral_v<T>) {
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(static_cast<U>(static_cast<U>(2) * static_cast<U>(edge) -
                                         static_cast<U>(v)));
  } else {
    return static_cast<T>(T(2) * edge - v);
  }
}

template <typename T>
T from_stat(work_t<T> w) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    return cast_value<T>(std::nearbyint(w));  // NumPy _round_if_needed (half to even)
  } else {
    return cast_value<T>(w);
  }
}

template <typename T>
struct Lane {
  std::byte* p;
  std::int64_t s;
  T get(std::int64_t i) const { return load<T>(p + i * s); }
  void set(std::int64_t i, T v) const { store<T>(p + i * s, v); }
  void fill(std::int64_t from, std::int64_t to, T v) const {
    for (std::int64_t i = from; i < to; ++i) set(i, v);
  }
};

template <typename T>
T lane_stat(const Lane<T>& l, std::int64_t from, std::int64_t count, PadMode mode) {
  if (mode == PadMode::Maximum || mode == PadMode::Minimum) {
    T best = l.get(from);
    for (std::int64_t i = from; i < from + count; ++i) {
      const T v = l.get(i);
      if (is_nan_v(v)) return v;
      if (mode == PadMode::Maximum ? less_v(best, v) : less_v(v, best)) best = v;
    }
    return best;
  }
  if (mode == PadMode::Mean) {
    work_t<T> sum{};
    for (std::int64_t i = from; i < from + count; ++i) sum += to_work(l.get(i));
    return from_stat<T>(sum / static_cast<double>(count));
  }
  // Median.
  if (count == 0) return from_stat<T>(work_t<T>(std::nan("")));
  std::vector<T> v;
  v.reserve(static_cast<std::size_t>(count));
  for (std::int64_t i = from; i < from + count; ++i) {
    const T x = l.get(i);
    if (is_nan_v(x)) return x;
    v.push_back(x);
  }
  std::sort(v.begin(), v.end(), [](T a, T b) { return less_v(a, b); });
  const auto h = static_cast<std::size_t>(count / 2);
  const work_t<T> m = count % 2 == 1 ? to_work(v[h]) : (to_work(v[h - 1]) + to_work(v[h])) / 2.0;
  return from_stat<T>(m);
}

template <typename T>
void pad_lane(const Lane<T>& l, std::int64_t len, std::int64_t left, std::int64_t right,
              std::int64_t n, std::size_t axis, const PadOptions& o, const NDArray* vals) {
  const PadMode mode = o.mode;
  switch (mode) {
    case PadMode::Constant: {
      l.fill(0, left, load<T>(vals->data() + static_cast<std::int64_t>(2 * axis) * vals->strides()[0]));
      l.fill(len - right, len,
             load<T>(vals->data() + static_cast<std::int64_t>(2 * axis + 1) * vals->strides()[0]));
      return;
    }
    case PadMode::Edge:
      l.fill(0, left, l.get(left));
      l.fill(len - right, len, l.get(left + n - 1));
      return;
    case PadMode::LinearRamp: {
      using W = work_t<T>;
      const auto ev = [&](std::size_t k) {
        return load<W>(vals->data() + static_cast<std::int64_t>(k) * vals->strides()[0]);
      };
      const auto ramp = [&](W start, W stop, std::int64_t num, auto&& put) {
        const W delta = stop - start;
        const W step = delta / static_cast<double>(num);
        for (std::int64_t k = 0; k < num; ++k) {
          W y = step == W{} ? static_cast<double>(k) / static_cast<double>(num) * delta
                            : static_cast<double>(k) * step;
          y += start;
          if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) y = std::floor(y);
          put(k, cast_value<T>(y));
        }
      };
      const W le = to_work(l.get(left));
      const W re = to_work(l.get(left + n - 1));
      ramp(ev(2 * axis), le, left, [&](std::int64_t k, T v) { l.set(k, v); });
      ramp(ev(2 * axis + 1), re, right, [&](std::int64_t k, T v) { l.set(len - 1 - k, v); });
      return;
    }
    case PadMode::Maximum:
    case PadMode::Minimum:
    case PadMode::Mean:
    case PadMode::Median: {
      auto [ll, rl] = o.stat_length[axis];
      if (ll < 0 || ll > n) ll = n;
      if (rl < 0 || rl > n) rl = n;
      if ((ll == 0 || rl == 0) && (mode == PadMode::Maximum || mode == PadMode::Minimum)) {
        throw_error(ErrorKind::Value, "stat_length of 0 yields no value for padding");
      }
      const T ls = lane_stat(l, left, ll, mode);
      const T rs = (ll == n && rl == n) ? ls : lane_stat(l, left + n - rl, rl, mode);
      l.fill(0, left, ls);
      l.fill(len - right, len, rs);
      return;
    }
    case PadMode::Reflect:
    case PadMode::Symmetric: {
      if (n == 1) {
        l.fill(0, left, l.get(left));
        l.fill(len - right, len, l.get(left));
        return;
      }
      const bool include_edge = mode == PadMode::Symmetric;
      std::vector<T> chunk;
      std::int64_t lp = left;
      std::int64_t rp = right;
      while (lp > 0 || rp > 0) {
        std::int64_t old = len - rp - lp;
        std::int64_t edge_offset = 0;
        if (include_edge) {
          old = old / n * n;
          edge_offset = 1;
        } else {
          old = (old - 1) / (n - 1) * (n - 1) + 1 - 1;
        }
        if (lp > 0) {
          const std::int64_t c = std::min(old, lp);
          const std::int64_t start = lp - edge_offset + c;
          chunk.clear();
          for (std::int64_t k = 0; k < c; ++k) {
            const T v = l.get(start - k);
            chunk.push_back(o.odd ? odd_reflect(l.get(lp), v) : v);
          }
          for (std::int64_t k = 0; k < c; ++k) l.set(lp - c + k, chunk[static_cast<std::size_t>(k)]);
          lp -= c;
        }
        if (rp > 0) {
          const std::int64_t c = std::min(old, rp);
          const std::int64_t start = len - rp + edge_offset - 2;
          chunk.clear();
          for (std::int64_t k = 0; k < c; ++k) {
            const T v = l.get(start - k);
            chunk.push_back(o.odd ? odd_reflect(l.get(len - rp - 1), v) : v);
          }
          for (std::int64_t k = 0; k < c; ++k) l.set(len - rp + k, chunk[static_cast<std::size_t>(k)]);
          rp -= c;
        }
      }
      return;
    }
    case PadMode::Wrap: {
      std::int64_t lp = left;
      std::int64_t rp = right;
      while (lp > 0 || rp > 0) {
        const std::int64_t period = (len - rp - lp) / n * n;
        std::int64_t nl = 0;
        std::int64_t nr = 0;
        if (lp > 0) {
          const std::int64_t c = std::min(period, lp);
          const std::int64_t src = lp + period - c;
          const std::int64_t dst = lp > period ? lp - period : 0;
          if (lp > period) nl = lp - period;
          for (std::int64_t k = 0; k < c; ++k) l.set(dst + k, l.get(src + k));
        }
        if (rp > 0) {
          const std::int64_t c = std::min(period, rp);
          const std::int64_t src = len - rp - period;
          if (rp > period) nr = rp - period;
          for (std::int64_t k = 0; k < c; ++k) l.set(len - rp + k, l.get(src + k));
        }
        lp = nl;
        rp = nr;
      }
      return;
    }
    case PadMode::Empty:
      return;
  }
}

}  // namespace

NDArray pad(const NDArray& a, const PadOptions& o) {
  const std::size_t nd = a.ndim();
  if (o.width.size() != nd) throw_error(ErrorKind::Value, "pad_width must have one pair per axis");
  Shape shape(nd);
  for (std::size_t d = 0; d < nd; ++d) {
    const auto [l, r] = o.width[d];
    if (l < 0 || r < 0) throw_error(ErrorKind::Value, "index can't contain negative values");
    shape[d] = l + a.shape()[d] + r;
  }
  const Order order = (a.is_f_contiguous() && !a.is_c_contiguous()) ? Order::F : Order::C;
  NDArray out = empty_order(shape, a.dtype(), order, true);
  NDArray center = out;
  for (std::size_t d = 0; d < nd; ++d) {
    center = slice_axis(center, d, o.width[d].first, o.width[d].first + a.shape()[d], 1);
  }
  copy_into(center, a);
  if (o.mode == PadMode::Empty) return out;
  if (o.mode != PadMode::Constant && a.size() == 0) {
    for (std::size_t d = 0; d < nd; ++d) {
      if (a.shape()[d] == 0 && (o.width[d].first > 0 || o.width[d].second > 0)) {
        throw_error(ErrorKind::Value, "can't extend empty axis " + std::to_string(d) +
                                          " using modes other than 'constant' or 'empty'");
      }
    }
    return out;
  }
  const bool stat = o.mode == PadMode::Maximum || o.mode == PadMode::Minimum ||
                    o.mode == PadMode::Mean || o.mode == PadMode::Median;
  if (stat && o.stat_length.size() != nd) {
    throw_error(ErrorKind::Value, "stat_length must have one pair per axis");
  }
  std::optional<NDArray> vals;
  if (o.mode == PadMode::Constant || o.mode == PadMode::LinearRamp) {
    const DType vt = o.mode == PadMode::Constant
                         ? a.dtype()
                         : (is_complex(a.dtype()) ? DType::Complex128 : DType::Float64);
    if (o.values) {
      if (o.values->size() != static_cast<std::int64_t>(2 * nd)) {
        throw_error(ErrorKind::Value, "pad values must have one pair per axis");
      }
      vals = o.values->reshape({static_cast<std::int64_t>(2 * nd)}).astype(vt);
    } else {
      vals = NDArray::zeros({static_cast<std::int64_t>(2 * nd)}, vt);
    }
  }
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = dtype_t<decltype(tag)::value>;
    for (std::size_t axis = 0; axis < nd; ++axis) {
      const auto [left, right] = o.width[axis];
      if (left == 0 && right == 0) continue;
      NDArray roi = out;
      for (std::size_t d = axis + 1; d < nd; ++d) {
        roi = slice_axis(roi, d, o.width[d].first, o.width[d].first + a.shape()[d], 1);
      }
      const std::int64_t n = a.shape()[axis];
      for_each_lane(roi, axis, [&](std::byte* p, std::int64_t s, std::int64_t len) {
        pad_lane<T>(Lane<T>{p, s}, len, left, right, n, axis, o, vals ? &*vals : nullptr);
      });
    }
  });
  return out;
}

}  // namespace nativpy
