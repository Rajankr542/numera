#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "ndarray.hpp"
#include "ufunc.hpp"

namespace nativpy {

// Indexing extras (parity milestone P8, D-110/D-111).

// NumPy clipmode: out-of-range handling for take/put/choose/ravel_multi_index.
enum class ClipMode : std::uint8_t { Raise, Wrap, Clip };
// "raise"/"wrap"/"clip"; throws ValueError otherwise (NumPy message).
ClipMode clip_mode_from_name(std::string_view name);

// np.take(a, indices, axis, mode=). axis nullopt = flattened. Indices must be
// integer or bool. Result shape a.shape[:axis] + indices.shape + a.shape[axis+1:].
NDArray take_mode(const NDArray& a, const NDArray& indices, std::optional<std::int64_t> axis,
                  ClipMode mode);

// np.take_along_axis / np.put_along_axis (axis nullopt = flattened, indices 1-d).
NDArray take_along_axis(const NDArray& a, const NDArray& indices, std::optional<std::int64_t> axis);
void put_along_axis(const NDArray& a, const NDArray& indices, const NDArray& values,
                    std::optional<std::int64_t> axis);

// np.put(a, ind, v, mode=): a.flat[ind] = v (v repeated cyclically, unsafe cast).
void put(const NDArray& a, const NDArray& ind, const NDArray& v, ClipMode mode);
// np.putmask(a, mask, values): a.flat[i] = values[i % n] where mask.flat[i].
// `safe`: values must cast to a.dtype under "safe" casting (D-111).
void putmask(const NDArray& a, const NDArray& mask, const NDArray& values, bool safe);
// np.place(arr, mask, vals): k-th masked element gets vals[k % n].
void place(const NDArray& a, const NDArray& mask, const NDArray& vals, bool safe);

// np.choose(a, choices, mode=). Choices are cast to their promoted dtype.
NDArray choose(const NDArray& a, const std::vector<NDArray>& choices, ClipMode mode);
// np.compress(condition, a, axis) / np.extract(condition, arr).
NDArray compress(const NDArray& condition, const NDArray& a, std::optional<std::int64_t> axis);
NDArray extract(const NDArray& condition, const NDArray& a);
// np.select(condlist, choicelist, default): first true condition wins.
NDArray select(const std::vector<NDArray>& condlist, const std::vector<NDArray>& choicelist,
               const NDArray& default_value);
// ~any(conds): the "otherwise" mask of np.piecewise (conds broadcast together).
NDArray none_of(const std::vector<NDArray>& conds);

// np.argwhere: int64 (N, ndim). np.flatnonzero: int64 1-d.
NDArray argwhere(const NDArray& a);
NDArray flatnonzero(const NDArray& a);
// np.count_nonzero(a, axis, keepdims) -> int64 (0-d when axis is nullopt).
NDArray count_nonzero(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis,
                      bool keepdims);

// np.ravel_multi_index(multi_index, dims, mode=, order=). `modes` has 1 or
// dims.size() entries. Order must be C or F.
NDArray ravel_multi_index(const std::vector<NDArray>& multi_index, const Shape& dims,
                          const std::vector<ClipMode>& modes, Order order);
// np.unravel_index(indices, shape, order=): one int64 array per dimension.
std::vector<NDArray> unravel_index(const NDArray& indices, const Shape& dims, Order order);

// np.diagonal: read-only view (D-110). np.trace: sum of that view over the
// diagonal axis (D-017 sum dtype rules unless `dtype`).
NDArray diagonal(const NDArray& a, std::int64_t offset, std::int64_t axis1, std::int64_t axis2);
NDArray trace(const NDArray& a, std::int64_t offset, std::int64_t axis1, std::int64_t axis2,
              std::optional<DType> dtype);

}  // namespace nativpy
