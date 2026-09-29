#include "indexing.hpp"

#include <algorithm>
#include <cstring>
#include <string>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"

namespace nativpy {

SliceBounds slice_indices(std::optional<std::int64_t> start, std::optional<std::int64_t> stop,
                          std::optional<std::int64_t> step, std::int64_t dim) {
  const std::int64_t st = step.value_or(1);
  if (st == 0) throw_error(ErrorKind::Value, "slice step cannot be zero");
  const std::int64_t lo = st < 0 ? -1 : 0;
  const std::int64_t hi = st < 0 ? dim - 1 : dim;
  auto clamp = [&](std::optional<std::int64_t> v, std::int64_t dflt) {
    if (!v) return dflt;
    std::int64_t x = *v;
    if (x < 0) {
      x += dim;
      if (x < lo) x = lo;
    } else if (x > hi) {
      x = hi;
    }
    return x;
  };
  const std::int64_t s = clamp(start, st < 0 ? hi : lo);
  const std::int64_t e = clamp(stop, st < 0 ? lo : hi);
  std::int64_t len = 0;
  if (st > 0 && s < e) len = (e - s - 1) / st + 1;
  if (st < 0 && e < s) len = (s - e - 1) / (-st) + 1;
  return {s, e, st, len};
}

namespace {

// Copies src (same shape as dst, possibly broadcast/strided) into dst with an
// unsafe cast. Handles overlapping memory by staging through a copy.
void assign(NDArray& dst, const NDArray& src_in) {
  const NDArray bsrc = broadcast_to(src_in, dst.shape());
  const NDArray src = bsrc.may_share_memory(dst) ? bsrc.copy() : bsrc;
  const auto plan = make_plan<2>(dst.shape(), {&dst, &src});
  dispatch_dtype(src.dtype(), [&](auto stag) {
    using S = dtype_t<decltype(stag)::value>;
    dispatch_dtype(dst.dtype(), [&](auto dtag) {
      using D = dtype_t<decltype(dtag)::value>;
      run_plan<2>(plan, {dst.data(), src.data()},
                  [](std::array<std::byte*, 2> p, const std::array<std::int64_t, 2>& s,
                     std::int64_t n) {
                    for (std::int64_t i = 0; i < n; ++i) {
                      store<D>(p[0] + i * s[0], cast_value<D>(load<S>(p[1] + i * s[1])));
                    }
                  });
    });
  });
}

std::int64_t norm_index(std::int64_t i, std::int64_t dim, std::size_t axis) {
  if (i < -dim || i >= dim) {
    throw_error(ErrorKind::Index, "index " + std::to_string(i) + " is out of bounds for axis " +
                                      std::to_string(axis) + " with size " + std::to_string(dim));
  }
  return i < 0 ? i + dim : i;
}

bool is_bool_array(const IndexItem& it) {
  return it.kind == IndexItem::Kind::Array && it.array->dtype() == DType::Bool;
}

// Number of source dims an item consumes.
std::size_t consumed(const IndexItem& it) {
  switch (it.kind) {
    case IndexItem::Kind::Integer:
    case IndexItem::Kind::Slice: return 1;
    case IndexItem::Kind::Array: return is_bool_array(it) ? it.array->ndim() : 1;
    default: return 0;
  }
}

// Expands the ellipsis into full slices; converts bool arrays into nonzero
// integer arrays (NumPy semantics); validates dims.
std::vector<IndexItem> normalize(const NDArray& a, const std::vector<IndexItem>& index) {
  std::size_t used = 0;
  int ellipses = 0;
  for (const auto& it : index) {
    used += consumed(it);
    if (it.kind == IndexItem::Kind::Ellipsis) ++ellipses;
    if (it.kind == IndexItem::Kind::Array) {
      const DType dt = it.array->dtype();
      if (dt != DType::Bool && !is_integer(dt)) {
        throw_error(ErrorKind::Index,
                    "arrays used as indices must be of integer (or boolean) type");
      }
    }
  }
  if (ellipses > 1) throw_error(ErrorKind::Index, "an index can only have a single ellipsis ('...')");
  if (used > a.ndim()) {
    throw_error(ErrorKind::Index, "too many indices for array: array is " +
                                      std::to_string(a.ndim()) + "-dimensional, but " +
                                      std::to_string(used) + " were indexed");
  }
  std::vector<IndexItem> out;
  std::size_t axis = 0;
  bool expanded = false;
  auto push_rest = [&]() {
    for (std::size_t k = 0; k < a.ndim() - used; ++k) {
      out.push_back(IndexItem::slice(std::nullopt, std::nullopt, std::nullopt));
    }
    axis += a.ndim() - used;
    expanded = true;
  };
  for (const auto& it : index) {
    if (it.kind == IndexItem::Kind::Ellipsis) {
      push_rest();
      continue;
    }
    if (is_bool_array(it)) {
      const NDArray& m = *it.array;
      for (std::size_t d = 0; d < m.ndim(); ++d) {
        if (m.shape()[d] != a.shape()[axis + d]) {
          throw_error(ErrorKind::Index,
                      "boolean index did not match indexed array along axis " +
                          std::to_string(axis + d) + "; size of axis is " +
                          std::to_string(a.shape()[axis + d]) +
                          " but size of corresponding boolean axis is " +
                          std::to_string(m.shape()[d]));
        }
      }
      if (m.ndim() == 0) {
        // 0-d bool: adds a new axis of length 1 (true) or 0 (false).
        NDArray idx = NDArray::empty({m.get_int64(0) != 0 ? 1 : 0}, DType::Int64);
        if (idx.size() == 1) idx.set_int64(0, 0);
        out.push_back(IndexItem::new_axis());
        out.back().array = idx;  // marker: newaxis + array over the new axis
        continue;
      }
      for (auto& nz : nonzero(m)) out.push_back(IndexItem::array_(std::move(nz)));
      axis += m.ndim();
      continue;
    }
    out.push_back(it);
    axis += consumed(it);
  }
  if (!expanded) push_rest();
  return out;
}

// Result of applying the basic part of an index: a view plus, for each
// advanced item, the view axis it indexes.
struct Staged {
  NDArray view;
  std::vector<std::size_t> adv_axes;  // axes of `view` indexed by arrays
  std::vector<NDArray> adv_arrays;    // matching int arrays
  bool adjacent = true;               // advanced items are consecutive
};

Staged stage(const NDArray& a, const std::vector<IndexItem>& items) {
  Shape shape;
  Strides strides;
  std::int64_t offset = a.offset();
  std::vector<std::size_t> adv_axes;
  std::vector<NDArray> adv_arrays;
  std::size_t src = 0;
  std::int64_t last_adv = -2;
  bool adjacent = true;
  auto mark_adv = [&](std::size_t k) {
    const auto ki = static_cast<std::int64_t>(k);
    if (last_adv >= 0 && last_adv != ki - 1) adjacent = false;
    last_adv = ki;
  };
  for (std::size_t k = 0; k < items.size(); ++k) {
    const auto& it = items[k];
    switch (it.kind) {
      case IndexItem::Kind::Integer: {
        const std::int64_t i = norm_index(it.integer, a.shape()[src], src);
        offset += i * a.strides()[src];
        ++src;
        break;
      }
      case IndexItem::Kind::Slice: {
        const auto b = slice_indices(it.start, it.stop, it.step, a.shape()[src]);
        if (b.length > 0) offset += b.start * a.strides()[src];
        shape.push_back(b.length);
        strides.push_back(b.step * a.strides()[src]);
        ++src;
        break;
      }
      case IndexItem::Kind::NewAxis:
        shape.push_back(1);
        strides.push_back(0);
        if (it.array) {  // 0-d boolean index over the new axis
          mark_adv(k);
          adv_axes.push_back(shape.size() - 1);
          adv_arrays.push_back(*it.array);
        }
        break;
      case IndexItem::Kind::Array:
        mark_adv(k);
        shape.push_back(a.shape()[src]);
        strides.push_back(a.strides()[src]);
        adv_axes.push_back(shape.size() - 1);
        adv_arrays.push_back(*it.array);
        ++src;
        break;
      case IndexItem::Kind::Ellipsis: break;
    }
  }
  return {a.view(std::move(shape), std::move(strides), offset), std::move(adv_axes),
          std::move(adv_arrays), adjacent};
}

// NumPy: once any array index is present, integer scalars are advanced
// indices too (0-d int arrays broadcast with the others).
std::vector<IndexItem> promote_integers(std::vector<IndexItem> items) {
  const bool any_array = std::any_of(items.begin(), items.end(), [](const IndexItem& it) {
    return it.kind == IndexItem::Kind::Array || (it.kind == IndexItem::Kind::NewAxis && it.array);
  });
  if (!any_array) return items;
  for (auto& it : items) {
    if (it.kind != IndexItem::Kind::Integer) continue;
    NDArray z = NDArray::empty({}, DType::Int64);
    z.set_int64(0, it.integer);
    it = IndexItem::array_(std::move(z));
  }
  return items;
}

// Advanced indexing plan: output shape and, per output element, the
// absolute byte offset into the buffer.
struct Gather {
  Shape out_shape;
  std::vector<std::int64_t> offsets;
};

std::int64_t offset_of(std::int64_t flat, const Shape& sh, const Strides& st) {
  std::int64_t off = 0;
  for (std::size_t d = sh.size(); d-- > 0;) {
    off += (flat % sh[d]) * st[d];
    flat /= sh[d];
  }
  return off;
}

Gather plan_gather(const Staged& s) {
  const NDArray& v = s.view;
  std::vector<Shape> ishapes;
  for (const auto& x : s.adv_arrays) ishapes.push_back(x.shape());
  Shape bshape;
  try {
    bshape = broadcast_shapes(ishapes);
  } catch (const Error& err) {
    if (err.kind() != ErrorKind::Broadcast) throw;
    std::string msg = "shape mismatch: indexing arrays could not be broadcast together with shapes";
    for (const auto& sh : ishapes) msg += " " + shape_to_string(sh);
    throw_error(ErrorKind::Index, msg);
  }
  const std::int64_t bsize = shape_size(bshape);
  std::vector<std::vector<std::int64_t>> idx(s.adv_arrays.size());
  for (std::size_t k = 0; k < s.adv_arrays.size(); ++k) {
    const NDArray b = broadcast_to(s.adv_arrays[k].astype(DType::Int64), bshape);
    const std::size_t ax = s.adv_axes[k];
    const std::int64_t dim = v.shape()[ax];
    idx[k].reserve(static_cast<std::size_t>(bsize));
    for_each_element(b, [&](const std::byte* p) {
      idx[k].push_back(norm_index(load<std::int64_t>(p), dim, ax));
    });
  }
  std::vector<bool> is_adv(v.ndim(), false);
  for (const auto ax : s.adv_axes) is_adv[ax] = true;
  Shape pre, post;
  Strides pre_st, post_st;
  const std::size_t first = s.adv_axes.front();
  for (std::size_t d = 0; d < v.ndim(); ++d) {
    if (is_adv[d]) continue;
    const bool before = s.adjacent && d < first;
    (before ? pre : post).push_back(v.shape()[d]);
    (before ? pre_st : post_st).push_back(v.strides()[d]);
  }
  Gather g;
  g.out_shape = pre;
  g.out_shape.insert(g.out_shape.end(), bshape.begin(), bshape.end());
  g.out_shape.insert(g.out_shape.end(), post.begin(), post.end());
  validate_shape(g.out_shape);
  const std::int64_t npre = shape_size(pre);
  const std::int64_t npost = shape_size(post);
  g.offsets.reserve(static_cast<std::size_t>(shape_size(g.out_shape)));
  std::vector<std::int64_t> post_off(static_cast<std::size_t>(npost));
  for (std::int64_t m = 0; m < npost; ++m) {
    post_off[static_cast<std::size_t>(m)] = offset_of(m, post, post_st);
  }
  for (std::int64_t i = 0; i < npre; ++i) {
    const std::int64_t po = v.offset() + offset_of(i, pre, pre_st);
    for (std::int64_t j = 0; j < bsize; ++j) {
      std::int64_t ao = po;
      for (std::size_t k = 0; k < idx.size(); ++k) {
        ao += idx[k][static_cast<std::size_t>(j)] * v.strides()[s.adv_axes[k]];
      }
      for (const auto m : post_off) g.offsets.push_back(ao + m);
    }
  }
  return g;
}

}  // namespace

NDArray get_index(const NDArray& a, const std::vector<IndexItem>& index) {
  const Staged s = stage(a, promote_integers(normalize(a, index)));
  if (s.adv_arrays.empty()) {
    // NumPy returns a scalar (a copy) for a full integer index such as a[1, 2];
    // only an explicit ellipsis (a[1, 2, ...]) yields a 0-d view (D-015).
    const bool has_ellipsis = std::any_of(index.begin(), index.end(), [](const IndexItem& it) {
      return it.kind == IndexItem::Kind::Ellipsis;
    });
    if (s.view.ndim() == 0 && !has_ellipsis) return s.view.copy();
    return s.view;
  }
  const Gather g = plan_gather(s);
  NDArray out = NDArray::empty(g.out_shape, a.dtype());
  const std::size_t isz = a.itemsize();
  const std::byte* base = a.buffer()->data();
  std::byte* dst = out.data();
  for (std::size_t i = 0; i < g.offsets.size(); ++i) {
    std::memcpy(dst + i * isz, base + g.offsets[i], isz);
  }
  return out;
}

void set_index(NDArray& a, const std::vector<IndexItem>& index, const NDArray& value) {
  a.check_writeable();  // D-016
  const Staged s = stage(a, promote_integers(normalize(a, index)));
  if (s.adv_arrays.empty()) {
    NDArray target = s.view;
    assign(target, value);
    return;
  }
  const Gather g = plan_gather(s);
  NDArray vals = NDArray::empty(g.out_shape, a.dtype());
  assign(vals, value);
  const std::size_t isz = a.itemsize();
  std::byte* base = a.buffer()->data();
  const std::byte* src = vals.data();
  // Sequential writes: with repeated indices the last value wins (NumPy).
  for (std::size_t i = 0; i < g.offsets.size(); ++i) {
    std::memcpy(base + g.offsets[i], src + i * isz, isz);
  }
}

std::vector<NDArray> nonzero(const NDArray& a) {
  if (a.ndim() == 0) {
    throw_error(ErrorKind::Value,
                "Calling nonzero on 0d arrays is not allowed. Use np.atleast_1d(scalar).nonzero() instead.");
  }
  const NDArray m = a.dtype() == DType::Bool ? a : a.astype(DType::Bool);
  std::vector<std::int64_t> flat;
  std::int64_t i = 0;
  for_each_element(m, [&](const std::byte* p) {
    if (load<bool>(p)) flat.push_back(i);
    ++i;
  });
  const auto n = static_cast<std::int64_t>(flat.size());
  std::vector<NDArray> out;
  for (std::size_t d = 0; d < a.ndim(); ++d) out.push_back(NDArray::empty({n}, DType::Int64));
  for (std::int64_t k = 0; k < n; ++k) {
    std::int64_t f = flat[static_cast<std::size_t>(k)];
    for (std::size_t d = a.ndim(); d-- > 0;) {
      store<std::int64_t>(out[d].data() + k * 8, f % a.shape()[d]);
      f /= a.shape()[d];
    }
  }
  return out;
}

NDArray take(const NDArray& a, const NDArray& indices, std::optional<std::int64_t> axis) {
  if (!is_integer(indices.dtype())) throw_error(ErrorKind::DType, "take: indices must be integers");
  if (!axis) return take(a.reshape({-1}), indices, 0);
  const std::int64_t ax = normalize_axis(*axis, static_cast<std::int64_t>(a.ndim()));
  std::vector<IndexItem> idx(static_cast<std::size_t>(ax),
                             IndexItem::slice(std::nullopt, std::nullopt, std::nullopt));
  idx.push_back(IndexItem::array_(indices));
  return get_index(a, idx);
}

NDArray where(const NDArray& cond, const NDArray& x, const NDArray& y) {
  const DType dt = promote_types(x.dtype(), y.dtype());
  const Shape shape = broadcast_shapes({cond.shape(), x.shape(), y.shape()});
  NDArray out = NDArray::empty(shape, dt);
  const NDArray c = broadcast_to(cond.astype(DType::Bool), shape);
  const NDArray xs = broadcast_to(x.astype(dt), shape);
  const NDArray ys = broadcast_to(y.astype(dt), shape);
  const auto plan = make_plan<4>(shape, {&out, &c, &xs, &ys});
  const std::size_t isz = itemsize(dt);
  run_plan<4>(plan, {out.data(), c.data(), xs.data(), ys.data()},
              [isz](std::array<std::byte*, 4> p, const std::array<std::int64_t, 4>& s,
                    std::int64_t n) {
                for (std::int64_t i = 0; i < n; ++i) {
                  const bool t = load<bool>(p[1] + i * s[1]);
                  std::memcpy(p[0] + i * s[0], t ? p[2] + i * s[2] : p[3] + i * s[3], isz);
                }
              });
  return out;
}

}  // namespace nativpy


