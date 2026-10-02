#pragma once

#include <optional>
#include <string_view>

#include "ndarray.hpp"

namespace nativpy {

// numpy.emath (lib/_scimath_impl.py), D-180.
enum class EmathOp { Sqrt, Log, Log2, Log10, Arccos, Arcsin, Arctanh };

std::optional<EmathOp> emath_op_from_name(std::string_view name) noexcept;

// Real input whose domain is violated anywhere (x < 0 for sqrt/log*, |x| > 1
// for arccos/arcsin/arctanh) is converted to complex first (complex64 for
// int8/uint8/int16/uint16/float32, complex128 otherwise, as `_tocomplex`);
// other real input uses the float loop of the matching ufunc. Complex input
// keeps its dtype. Returns a new C-contiguous array.
NDArray emath_unary(EmathOp op, const NDArray& x);

}  // namespace nativpy
