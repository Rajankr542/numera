#include "p08_indexing.hpp"

#include <algorithm>
#include <cstring>
#include <string>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"
#include "indexing.hpp"
#include "reduce.hpp"

namespace nativpy {

namespace {

std::string cast_error(DType from, DType to) {
  return "Cannot cast array data from dtype('" + std::string(dtype_name(from)) + "') to dtype('" +
         std::string(dtype_name(to)) + "') according to the rule 'safe'";
}

// Integer (or bool) index array -> int64 values in C order.
std::vector<std::int64_t> index_values(const NDArray& a) {
  if (!is_integer(a.dtype()) && a.dtype() != DType::Bool) {
    throw_error(ErrorKind::DType, cast_error(a.dtype(), DType::Int64));
  }
  std::vector<std::int64_t> out;
  out.reserve(static_cast<std::size_t>(a.size()));
  const NDArray c = a.astype(DType::Int64);
  const std::byte* p = c.data();
  for (std::int64_t i = 0; i < c.size(); ++i) out.push_back(load<std::int64_t>(p + i * 8));
  return out;
}

std::int64_t apply_mode(std::int64_t i, std::int64_t dim, ClipMode mode, std::int64_t axis) {
  switch (mode) {
    case ClipMode::Wrap: {
      const std::int64_t r = i % dim;
      return r < 0 ? r + dim : r;
    }
    case ClipMode::Clip: return i < 0 ? 0 : (i >= dim ? dim - 1 : i);
    case ClipMode::Raise: break;
  }
  if (i < -dim || i >= dim) {
    throw_error(ErrorKind::Index, "index " + std::to_string(i) + " is out of bounds for axis " +
                                      std::to_string(axis) + " with size " + std::to_string(dim));
  }
  return i < 0 ? i + dim : i;
}

// Absolute byte offset (into the buffer) of the element at flat C position.
std::int64_t flat_offset(const NDArray& a, std::int64_t flat) {
  std::int64_t off = a.offset();
  for (std::size_t d = a.ndim(); d-- > 0;) {
    off += (flat % a.shape()[d]) * a.strides()[d];
    flat /= a.shape()[d];
  }
  return off;
}

// Walks `shape` in C order; calls fn(index, offsets) with offsets[k] the byte
// offset of operand k (strides[k]) relative to its data pointer.
template <typename Fn>
void walk(const Shape& shape, const std::vector<Strides>& strides, Fn&& fn) {
  const std::int64_t n = shape_size(shape);
  if (n == 0) return;
  const std::size_t nd = shape.size();
  const std::size_t k = strides.size();
  std::vector<std::int64_t> idx(nd, 0);
  std::vector<std::int64_t> off(k, 0);
  for (std::int64_t i = 0; i < n; ++i) {
    fn(static_cast<const std::vector<std::int64_t>&>(idx),
       static_cast<const std::vector<std::int64_t>&>(off));
    for (std::size_t d = nd; d-- > 0;) {
      if (++idx[d] < shape[d]) {
        for (std::size_t j = 0; j < k; ++j) off[j] += strides[j][d];
        break;
      }
      for (std::size_t j = 0; j < k; ++j) off[j] -= strides[j][d] * (shape[d] - 1);
      idx[d] = 0;
    }
  }
}

NDArray bool_of(const NDArray& a) { return a.dtype() == DType::Bool ? a : a.astype(DType::Bool); }

std::vector<bool> flat_bools(const NDArray& a) {
  const NDArray b = bool_of(a).copy();
  std::vector<bool> out(static_cast<std::size_t>(b.size()));
  for (std::int64_t i = 0; i < b.size(); ++i) out[static_cast<std::size_t>(i)] = load<bool>(b.data() + i);
  return out;
}

NDArray int64_array(const std::vector<std::int64_t>& v) {
  NDArray out = NDArray::empty({static_cast<std::int64_t>(v.size())}, DType::Int64);
  if (!v.empty()) std::memcpy(out.data(), v.data(), v.size() * sizeof(std::int64_t));
  return out;
}

NDArray scalar_bool(bool v) {
  NDArray s = NDArray::empty({}, DType::Bool);
  store<bool>(s.data(), v);
  return s;
}

}  // namespace

ClipMode clip_mode_from_name(std::string_view name) {
  if (name == "raise") return ClipMode::Raise;
  if (name == "wrap") return ClipMode::Wrap;
  if (name == "clip") return ClipMode::Clip;
  throw_error(ErrorKind::Value, "clipmode must be one of 'clip', 'raise', or 'wrap' (got '" +
                                    std::string(name) + "')");
}

// ---- take / take_along_axis / put_along_axis ----

NDArray take_mode(const NDArray& a_in, const NDArray& indices, std::optional<std::int64_t> axis,
                  ClipMode mode) {
  const NDArray a = axis ? a_in : a_in.reshape({-1});
  const std::int64_t ax = normalize_axis(axis.value_or(0), static_cast<std::int64_t>(a.ndim()));
  const auto uax = static_cast<std::size_t>(ax);
  std::vector<std::int64_t> idx = index_values(indices);
  const std::int64_t dim = a.shape()[uax];
  if (dim == 0 && !idx.empty()) {
    throw_error(ErrorKind::Index, "cannot do a non-empty take from an empty axes.");
  }
  for (auto& i : idx) i = apply_mode(i, dim, mode, ax);
  Shape out_shape(a.shape().begin(), a.shape().begin() + ax);
  out_shape.insert(out_shape.end(), indices.shape().begin(), indices.shape().end());
  out_shape.insert(out_shape.end(), a.shape().begin() + ax + 1, a.shape().end());
  validate_shape(out_shape);
  NDArray out = NDArray::empty(out_shape, a.dtype());
  if (out.size() == 0) return out;
  const NDArray src = a.is_c_contiguous() ? a : a.copy();
  std::int64_t outer = 1;
  std::int64_t inner = 1;
  for (std::size_t d = 0; d < uax; ++d) outer *= a.shape()[d];
  for (std::size_t d = uax + 1; d < a.ndim(); ++d) inner *= a.shape()[d];
  const auto block = static_cast<std::size_t>(inner) * a.itemsize();
  const auto ni = static_cast<std::int64_t>(idx.size());
  const std::byte* s = src.data();
  std::byte* o = out.data();
  for (std::int64_t r = 0; r < outer; ++r) {
    for (std::int64_t j = 0; j < ni; ++j) {
      const std::int64_t from = r * dim + idx[static_cast<std::size_t>(j)];
      std::memcpy(o + static_cast<std::size_t>(r * ni + j) * block,
                  s + static_cast<std::size_t>(from) * block, block);
    }
  }
  return out;
}

namespace {

struct AlongPlan {
  Shape shape;      // result / selection shape
  NDArray indices;  // int64, broadcast to shape
  std::int64_t axis;
};

AlongPlan plan_along(const NDArray& a, const NDArray& indices, std::int64_t axis_in) {
  if (!is_integer(indices.dtype())) {
    throw_error(ErrorKind::Index, "`indices` must be an integer array");
  }
  if (indices.ndim() != a.ndim()) {
    throw_error(ErrorKind::Value, "`indices` and `arr` must have the same number of dimensions");
  }
  const std::int64_t ax = normalize_axis(axis_in, static_cast<std::int64_t>(a.ndim()));
  Shape shape(a.ndim());
  for (std::size_t d = 0; d < a.ndim(); ++d) {
    const std::int64_t n = a.shape()[d];
    const std::int64_t m = indices.shape()[d];
    if (static_cast<std::int64_t>(d) == ax) {
      shape[d] = m;
    } else if (n == m || m == 1 || n == 1) {
      shape[d] = m == 1 ? n : m;
    } else {
      throw_error(ErrorKind::Index,
                  "shape mismatch: indexing arrays could not be broadcast together with shapes " +
                      shape_to_string(a.shape()) + " " + shape_to_string(indices.shape()));
    }
  }
  return {shape, broadcast_to(indices.astype(DType::Int64), shape), ax};
}

// Calls fn(out_flat, src_byte_offset) for every element of the selection.
template <typename Fn>
void along(const NDArray& a, const AlongPlan& p, Fn&& fn) {
  const auto ax = static_cast<std::size_t>(p.axis);
  const std::int64_t dim = a.shape()[ax];
  std::int64_t flat = 0;
  walk(p.shape, {p.indices.strides()}, [&](const std::vector<std::int64_t>& idx,
                                           const std::vector<std::int64_t>& off) {
    std::int64_t src = a.offset();
    for (std::size_t d = 0; d < a.ndim(); ++d) {
      std::int64_t c = 0;
      if (d == ax) {
        c = apply_mode(load<std::int64_t>(p.indices.data() + off[0]), dim, ClipMode::Raise, p.axis);
      } else if (a.shape()[d] != 1) {
        c = idx[d];
      }
      src += c * a.strides()[d];
    }
    fn(flat++, src);
  });
}

NDArray along_flat_indices(const NDArray& indices) {
  if (!is_integer(indices.dtype())) {
    throw_error(ErrorKind::Index, "`indices` must be an integer array");
  }
  if (indices.ndim() != 1) {
    throw_error(ErrorKind::Value, "when axis=None, `indices` must have a single dimension.");
  }
  return indices;
}

}  // namespace

NDArray take_along_axis(const NDArray& a, const NDArray& indices, std::optional<std::int64_t> axis) {
  if (!axis) return take_mode(a, along_flat_indices(indices), std::nullopt, ClipMode::Raise);
  const AlongPlan p = plan_along(a, indices, *axis);
  NDArray out = NDArray::empty(p.shape, a.dtype());
  const std::size_t isz = a.itemsize();
  const std::byte* base = a.buffer()->data();
  std::byte* o = out.data();
  along(a, p, [&](std::int64_t i, std::int64_t src) {
    std::memcpy(o + static_cast<std::size_t>(i) * isz, base + src, isz);
  });
  return out;
}

void put_along_axis(const NDArray& a, const NDArray& indices, const NDArray& values,
                    std::optional<std::int64_t> axis) {
  a.check_writeable();
  if (!axis) {
    const NDArray ind = along_flat_indices(indices);
    NDArray vals = NDArray::empty(ind.shape(), a.dtype());
    copy_into(vals, values);
    put(a, ind, vals, ClipMode::Raise);
    return;
  }
  const AlongPlan p = plan_along(a, indices, *axis);
  NDArray vals = NDArray::empty(p.shape, a.dtype());
  copy_into(vals, values);
  const std::size_t isz = a.itemsize();
  std::byte* base = a.buffer()->data();
  const std::byte* v = vals.data();
  along(a, p, [&](std::int64_t i, std::int64_t dst) {
    std::memcpy(base + dst, v + static_cast<std::size_t>(i) * isz, isz);
  });
}

// ---- put / putmask / place ----

void put(const NDArray& a, const NDArray& ind, const NDArray& v, ClipMode mode) {
  a.check_writeable();
  std::vector<std::int64_t> idx = index_values(ind);
  if (idx.empty() || v.size() == 0) return;
  const std::int64_t n = a.size();
  if (n == 0) throw_error(ErrorKind::Index, "cannot replace elements of an empty array");
  for (auto& i : idx) i = apply_mode(i, n, mode, 0);
  const NDArray vals = v.astype(a.dtype());
  const std::size_t isz = a.itemsize();
  const auto nv = static_cast<std::size_t>(vals.size());
  std::byte* base = a.buffer()->data();
  for (std::size_t k = 0; k < idx.size(); ++k) {
    std::memcpy(base + flat_offset(a, idx[k]), vals.data() + (k % nv) * isz, isz);
  }
}

namespace {

void masked_fill(const NDArray& a, const NDArray& mask, const NDArray& values, bool safe,
                 const char* name, bool by_position) {
  a.check_writeable();
  if (mask.size() != a.size()) {
    throw_error(ErrorKind::Value, std::string(name) + ": mask and data must be the same size");
  }
  if (safe && !can_cast(values.dtype(), a.dtype(), Casting::Safe)) {
    throw_error(ErrorKind::DType, cast_error(values.dtype(), a.dtype()));
  }
  const std::vector<bool> m = flat_bools(mask);
  const bool any = std::find(m.begin(), m.end(), true) != m.end();
  if (!any) return;
  if (values.size() == 0) {
    if (by_position) return;  // NumPy putmask: no-op
    throw_error(ErrorKind::Value, "Cannot insert from an empty array!");
  }
  const NDArray vals = values.astype(a.dtype());
  const std::size_t isz = a.itemsize();
  const auto nv = static_cast<std::size_t>(vals.size());
  std::size_t i = 0;
  std::size_t k = 0;
  for_each_element(a, [&](std::byte* p) {
    if (m[i]) {
      const std::size_t src = (by_position ? i : k++) % nv;
      std::memcpy(p, vals.data() + src * isz, isz);
    }
    ++i;
  });
}

}  // namespace

void putmask(const NDArray& a, const NDArray& mask, const NDArray& values, bool safe) {
  masked_fill(a, mask, values, safe, "putmask", true);
}

void place(const NDArray& a, const NDArray& mask, const NDArray& vals, bool safe) {
  masked_fill(a, mask, vals, safe, "place", false);
}

// ---- choose / compress / extract / select ----

NDArray choose(const NDArray& a, const std::vector<NDArray>& choices, ClipMode mode) {
  if (choices.empty()) throw_error(ErrorKind::Value, "0-length sequence.");
  if (!is_integer(a.dtype()) && a.dtype() != DType::Bool) {
    throw_error(ErrorKind::DType, cast_error(a.dtype(), DType::Int64));
  }
  DType dt = choices[0].dtype();
  std::vector<Shape> shapes{a.shape()};
  for (const auto& c : choices) {
    dt = promote_types(dt, c.dtype());
    shapes.push_back(c.shape());
  }
  const Shape shape = broadcast_shapes(shapes);
  NDArray out = NDArray::empty(shape, dt);
  std::vector<NDArray> ops{broadcast_to(a.astype(DType::Int64), shape)};
  for (const auto& c : choices) ops.push_back(broadcast_to(c.dtype() == dt ? c : c.astype(dt), shape));
  std::vector<Strides> st;
  for (const auto& o : ops) st.push_back(o.strides());
  const auto n = static_cast<std::int64_t>(choices.size());
  const std::size_t isz = itemsize(dt);
  std::byte* o = out.data();
  std::size_t flat = 0;
  walk(shape, st, [&](const std::vector<std::int64_t>&, const std::vector<std::int64_t>& off) {
    std::int64_t k = load<std::int64_t>(ops[0].data() + off[0]);
    if (k < 0 || k >= n) {
      if (mode == ClipMode::Raise) throw_error(ErrorKind::Value, "invalid entry in choice array");
      k = apply_mode(k, n, mode, 0);
    }
    const auto ku = static_cast<std::size_t>(k) + 1;
    std::memcpy(o + flat++ * isz, ops[ku].data() + off[ku], isz);
  });
  return out;
}

NDArray compress(const NDArray& condition, const NDArray& a, std::optional<std::int64_t> axis) {
  if (condition.ndim() != 1) throw_error(ErrorKind::Value, "condition must be a 1-d array");
  const std::vector<bool> c = flat_bools(condition);
  std::vector<std::int64_t> idx;
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (c[i]) idx.push_back(static_cast<std::int64_t>(i));
  }
  return take_mode(a, int64_array(idx), axis, ClipMode::Raise);
}

NDArray extract(const NDArray& condition, const NDArray& a) {
  const std::vector<bool> c = flat_bools(condition);
  std::vector<std::int64_t> idx;
  for (std::size_t i = 0; i < c.size(); ++i) {
    if (c[i]) idx.push_back(static_cast<std::int64_t>(i));
  }
  return take_mode(a, int64_array(idx), std::nullopt, ClipMode::Raise);
}

NDArray select(const std::vector<NDArray>& condlist, const std::vector<NDArray>& choicelist,
               const NDArray& default_value) {
  if (condlist.size() != choicelist.size()) {
    throw_error(ErrorKind::Value, "list of cases must be same length as list of conditions");
  }
  if (condlist.empty()) {
    throw_error(ErrorKind::Value, "select with an empty condition list is not possible");
  }
  DType dt = default_value.dtype();
  std::vector<Shape> shapes{default_value.shape()};
  for (std::size_t k = 0; k < condlist.size(); ++k) {
    if (condlist[k].dtype() != DType::Bool) {
      throw_error(ErrorKind::DType,
                  "invalid entry " + std::to_string(k) + " in condlist: should be boolean ndarray");
    }
    dt = promote_types(dt, choicelist[k].dtype());
    shapes.push_back(condlist[k].shape());
    shapes.push_back(choicelist[k].shape());
  }
  const Shape shape = broadcast_shapes(shapes);
  NDArray out = NDArray::empty(shape, dt);
  copy_into(out, default_value);
  for (std::size_t k = condlist.size(); k-- > 0;) masked_copy_into(out, choicelist[k], condlist[k]);
  return out;
}

NDArray none_of(const std::vector<NDArray>& conds) {
  std::vector<Shape> shapes;
  for (const auto& c : conds) shapes.push_back(c.shape());
  NDArray out = NDArray::empty(shapes.empty() ? Shape{} : broadcast_shapes(shapes), DType::Bool);
  copy_into(out, scalar_bool(true));
  const NDArray f = scalar_bool(false);
  for (const auto& c : conds) masked_copy_into(out, f, bool_of(c));
  return out;
}

// ---- argwhere / flatnonzero / count_nonzero ----

NDArray argwhere(const NDArray& a) {
  if (a.ndim() == 0) {
    const bool t = load<bool>(bool_of(a).data());
    return NDArray::empty({t ? 1 : 0, 0}, DType::Int64);
  }
  const std::vector<NDArray> nz = nonzero(a);
  const std::int64_t n = nz[0].size();
  const auto nd = static_cast<std::int64_t>(a.ndim());
  NDArray out = NDArray::empty({n, nd}, DType::Int64);
  std::byte* o = out.data();
  for (std::int64_t i = 0; i < n; ++i) {
    for (std::int64_t d = 0; d < nd; ++d) {
      std::memcpy(o + (i * nd + d) * 8, nz[static_cast<std::size_t>(d)].data() + i * 8, 8);
    }
  }
  return out;
}

NDArray flatnonzero(const NDArray& a) { return nonzero(a.reshape({-1}))[0]; }

NDArray count_nonzero(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis,
                      bool keepdims) {
  ReduceOptions o;
  o.axis = axis;
  o.keepdims = keepdims;
  o.dtype = DType::Int64;
  return reduce(ReduceOp::Sum, bool_of(a), o);
}

// ---- ravel_multi_index / unravel_index ----

namespace {

Strides index_strides(const Shape& dims, Order order) {
  if (order != Order::C && order != Order::F) {
    throw_error(ErrorKind::Value, "only 'C' or 'F' order is permitted");
  }
  for (const auto d : dims) {
    if (d < 0) throw_error(ErrorKind::Value, "dimensions must be non-negative");
  }
  (void)shape_size(dims);  // overflow check
  Strides st(dims.size(), 1);
  std::int64_t acc = 1;
  if (order == Order::C) {
    for (std::size_t d = dims.size(); d-- > 0;) {
      st[d] = acc;
      acc *= dims[d];
    }
  } else {
    for (std::size_t d = 0; d < dims.size(); ++d) {
      st[d] = acc;
      acc *= dims[d];
    }
  }
  return st;
}

void check_int(const NDArray& a) {
  if (!is_integer(a.dtype()) && a.dtype() != DType::Bool) {
    throw_error(ErrorKind::DType, "only int indices permitted");
  }
}

}  // namespace

NDArray ravel_multi_index(const std::vector<NDArray>& multi_index, const Shape& dims,
                          const std::vector<ClipMode>& modes, Order order) {
  const std::size_t nd = dims.size();
  if (multi_index.size() != nd) {
    throw_error(ErrorKind::Value,
                "parameter multi_index must be a sequence of length " + std::to_string(nd));
  }
  if (modes.size() != 1 && modes.size() != nd) {
    throw_error(ErrorKind::Value, "list of clipmodes has wrong length (" +
                                      std::to_string(modes.size()) + " instead of " +
                                      std::to_string(nd) + ")");
  }
  const Strides st = index_strides(dims, order);
  std::vector<Shape> shapes;
  for (const auto& m : multi_index) {
    check_int(m);
    shapes.push_back(m.shape());
  }
  const Shape shape = shapes.empty() ? Shape{} : broadcast_shapes(shapes);
  std::vector<NDArray> ops;
  std::vector<Strides> ost;
  for (const auto& m : multi_index) {
    ops.push_back(broadcast_to(m.astype(DType::Int64), shape));
    ost.push_back(ops.back().strides());
  }
  NDArray out = NDArray::zeros(shape, DType::Int64);
  if (nd == 0) return out;
  std::byte* o = out.data();
  std::size_t flat = 0;
  walk(shape, ost, [&](const std::vector<std::int64_t>&, const std::vector<std::int64_t>& off) {
    std::int64_t r = 0;
    for (std::size_t d = 0; d < nd; ++d) {
      std::int64_t v = load<std::int64_t>(ops[d].data() + off[d]);
      const ClipMode m = modes.size() == 1 ? modes[0] : modes[d];
      if (v < 0 || v >= dims[d]) {
        if (m == ClipMode::Raise || dims[d] == 0) {
          throw_error(ErrorKind::Value, "invalid entry in coordinates array");
        }
        v = apply_mode(v, dims[d], m, static_cast<std::int64_t>(d));
      }
      r += v * st[d];
    }
    store<std::int64_t>(o + flat++ * 8, r);
  });
  return out;
}

std::vector<NDArray> unravel_index(const NDArray& indices, const Shape& dims, Order order) {
  check_int(indices);
  const Strides st = index_strides(dims, order);
  const std::int64_t size = shape_size(dims);
  const std::vector<std::int64_t> idx = index_values(indices);
  std::vector<NDArray> out;
  for (std::size_t d = 0; d < dims.size(); ++d) out.push_back(NDArray::empty(indices.shape(), DType::Int64));
  for (std::size_t k = 0; k < idx.size(); ++k) {
    const std::int64_t v = idx[k];
    if (v < 0 || v >= size) {
      throw_error(ErrorKind::Value, "index " + std::to_string(v) +
                                        " is out of bounds for array with size " +
                                        std::to_string(size));
    }
    for (std::size_t d = 0; d < dims.size(); ++d) {
      store<std::int64_t>(out[d].data() + k * 8, (v / st[d]) % dims[d]);
    }
  }
  return out;
}

// ---- diagonal / trace ----

NDArray diagonal(const NDArray& a, std::int64_t offset, std::int64_t axis1, std::int64_t axis2) {
  if (a.ndim() < 2) throw_error(ErrorKind::Value, "diag requires an array of at least two dimensions");
  const auto nd = static_cast<std::int64_t>(a.ndim());
  const auto a1 = static_cast<std::size_t>(normalize_axis(axis1, nd));
  const auto a2 = static_cast<std::size_t>(normalize_axis(axis2, nd));
  if (a1 == a2) throw_error(ErrorKind::Value, "axis1 and axis2 cannot be the same");
  const std::int64_t n1 = a.shape()[a1];
  const std::int64_t n2 = a.shape()[a2];
  std::int64_t len = offset >= 0 ? std::min(n1, n2 - std::min(offset, n2))
                                 : std::min(n1 - std::min(-offset, n1), n2);
  len = std::max<std::int64_t>(len, 0);
  std::int64_t off = a.offset();
  if (len > 0) off += offset >= 0 ? offset * a.strides()[a2] : -offset * a.strides()[a1];
  Shape shape;
  Strides strides;
  for (std::size_t d = 0; d < a.ndim(); ++d) {
    if (d == a1 || d == a2) continue;
    shape.push_back(a.shape()[d]);
    strides.push_back(a.strides()[d]);
  }
  shape.push_back(len);
  strides.push_back(a.strides()[a1] + a.strides()[a2]);
  return a.view(std::move(shape), std::move(strides), off).as_readonly();
}

NDArray trace(const NDArray& a, std::int64_t offset, std::int64_t axis1, std::int64_t axis2,
              std::optional<DType> dtype) {
  const NDArray d = diagonal(a, offset, axis1, axis2);
  ReduceOptions o;
  o.axis = std::vector<std::int64_t>{-1};
  o.dtype = dtype;
  return reduce(ReduceOp::Sum, d, o);
}

}  // namespace nativpy
