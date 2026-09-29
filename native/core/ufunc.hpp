#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "ndarray.hpp"

namespace nativpy {

// Element-wise arithmetic ufuncs (PLAN §15, M4). Inputs broadcast (M5);
// results are new C-contiguous arrays. Semantics: D-014.

enum class BinaryOp { Add, Subtract, Multiply, Divide, Power, Mod, FloorDivide };
enum class UnaryOp { Abs, Negative, Sqrt, Exp, Log };

std::optional<BinaryOp> binary_op_from_name(std::string_view name) noexcept;
std::optional<UnaryOp> unary_op_from_name(std::string_view name) noexcept;

// Loop/result dtype NumPy's type resolver picks for inputs of dtype `in`.
DType binary_result_dtype(BinaryOp op, DType a, DType b);
DType unary_result_dtype(UnaryOp op, DType in);

NDArray binary(BinaryOp op, const NDArray& a, const NDArray& b);
NDArray unary(UnaryOp op, const NDArray& a);

}  // namespace nativpy
