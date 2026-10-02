#pragma once

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "dtype.hpp"
#include "ndarray.hpp"

namespace nativpy {

// Array manipulation kernels for parity milestone P6 (D-090..D-095).

// View of `a` restricted to [start, stop) (with `step`) along `axis`. The
// bounds must already be normalized (0 <= start, stop <= dim).
NDArray slice_axis(const NDArray& a, std::size_t axis, std::int64_t start, std::int64_t stop,
                   std::int64_t step = 1);

// Strides NumPy's PyArray_CreateMultiSortedStridePerm gives a new array of
// `shape` laid out like all of `like` (C order wins ties).
Strides multi_sorted_strides(const Shape& shape, std::size_t itemsize,
                             const std::vector<const NDArray*>& like);

// ---- join / split (D-090) ----

// NumPy concatenate. `axis` nullopt flattens every input (axis=None).
// Result dtype: `dtype`, or the promotion of all inputs. Every input must cast
// to the result (or to `out`) under `casting`. With `out`, writes into it and
// returns it.
NDArray concatenate(const std::vector<NDArray>& arrays, std::optional<std::int64_t> axis,
                    std::optional<DType> dtype, Casting casting, const NDArray* out);

// NumPy stack: inputs must share a shape; joined along a new `axis`.
NDArray stack(const std::vector<NDArray>& arrays, std::int64_t axis, std::optional<DType> dtype,
              Casting casting, const NDArray* out);

// NumPy array_split with explicit split points (views).
std::vector<NDArray> split_at(const NDArray& a, const std::vector<std::int64_t>& indices,
                              std::int64_t axis);
// NumPy array_split / split with a number of sections. `equal` requires an
// exact division ("array split does not result in an equal division").
std::vector<NDArray> split_sections(const NDArray& a, std::int64_t sections, std::int64_t axis,
                                    bool equal);

// NumPy unstack: views a[..., i, ...] along `axis`.
std::vector<NDArray> unstack(const NDArray& a, std::int64_t axis);

// ---- tile / repeat / resize (D-091, D-093) ----

// NumPy tile (C-contiguous result; reps shorter than ndim are left-padded with 1).
NDArray tile(const NDArray& a, const std::vector<std::int64_t>& reps);

// NumPy repeat. `repeats` has one entry (broadcast) or one per element along
// the axis. `axis` nullopt repeats the flattened array.
NDArray repeat(const NDArray& a, const std::vector<std::int64_t>& repeats,
               std::optional<std::int64_t> axis);

// NumPy np.resize: cycles the flattened data (zeros if `a` is empty).
NDArray resize(const NDArray& a, const Shape& shape);

// NumPy ndarray.resize data: memory-order bytes truncated or zero-extended to
// `shape`, in C order (F if `a` is F- and not C-contiguous). `a` must own its
// data and be contiguous.
NDArray resize_inplace_data(const NDArray& a, const Shape& shape);

// ---- pad (D-092) ----

enum class PadMode { Constant, Edge, LinearRamp, Maximum, Mean, Median, Minimum, Reflect,
                     Symmetric, Wrap, Empty };

struct PadOptions {
  PadMode mode = PadMode::Constant;
  std::vector<std::pair<std::int64_t, std::int64_t>> width;  // one pair per axis
  // constant_values / end_values: shape (ndim, 2) (any dtype); nullopt = 0.
  std::optional<NDArray> values;
  // stat_length per axis; -1 = whole axis.
  std::vector<std::pair<std::int64_t, std::int64_t>> stat_length;
  bool odd = false;  // reflect_type='odd'
};

NDArray pad(const NDArray& a, const PadOptions& opts);

// ---- edit (D-093) ----

// Builds the result of np.insert: `arr` with `values` (broadcast to arr's
// shape with `positions.size()` along `axis`) placed at the result indices
// `positions` (distinct, in [0, N + positions.size())). Layout F if `arr` is
// F- and not C-contiguous, else C.
NDArray insert_along(const NDArray& arr, std::int64_t axis,
                     const std::vector<std::int64_t>& positions, const NDArray& values);

// np.delete core: keeps the indices with keep[i] along `axis` (same layout rule).
NDArray delete_along(const NDArray& arr, std::int64_t axis, const std::vector<bool>& keep);

// np.trim_zeros: view trimmed to the bounding box of nonzero elements on the
// axes with trim_axis[d] (front and/or back).
NDArray trim_zeros(const NDArray& a, bool front, bool back, const std::vector<bool>& trim_axis);

// ---- reorder (D-090) ----

// Reverses the given axes (all when empty): a view with negated strides.
NDArray flip(const NDArray& a, const std::vector<std::int64_t>& axes, bool all);

// np.roll with one (already summed) shift per axis; result laid out like `a` (order K).
NDArray roll(const NDArray& a, const std::vector<std::int64_t>& shifts);

// ---- conversions (D-094) ----

// np.copyto: src broadcast to dst, every src element cast under `casting`,
// only where `where` (bool, broadcast) is true when given.
void copyto(const NDArray& dst, const NDArray& src, Casting casting, const NDArray* where);

// True if every element is finite (always true for bool/integer dtypes).
bool all_finite(const NDArray& a);

}  // namespace nativpy
