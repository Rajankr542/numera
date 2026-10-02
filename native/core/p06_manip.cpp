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
