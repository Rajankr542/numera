#pragma once

#include <cstdint>
#include <optional>
#include <string_view>

#include "ndarray.hpp"

namespace nativpy {

// Element-wise arithmetic ufuncs (PLAN §15, M4). Inputs broadcast (M5);
// results are new C-contiguous arrays. Semantics: D-014.

enum class BinaryOp { Add, Subtract, Multiply, Divide, Power, Mod, FloorDivide };
enum class UnaryOp { Abs, Negative, Sqrt, Exp, Log, Conjugate, Angle };

std::optional<BinaryOp> binary_op_from_name(std::string_view name) noexcept;
std::optional<UnaryOp> unary_op_from_name(std::string_view name) noexcept;

// Loop/result dtype NumPy's type resolver picks for inputs of dtype `in`.
DType binary_result_dtype(BinaryOp op, DType a, DType b);
DType unary_result_dtype(UnaryOp op, DType in);

NDArray binary(BinaryOp op, const NDArray& a, const NDArray& b);
NDArray unary(UnaryOp op, const NDArray& a);

// NumPy `out=` (D-046): writes the result into `out` and returns it (a view of
// the same buffer). `out` must be writeable, have exactly the broadcast shape,
// and accept the loop dtype under same_kind casting. Overlapping inputs behave
// as if they were copied first.
NDArray binary(BinaryOp op, const NDArray& a, const NDArray& b, const NDArray& out);
NDArray unary(UnaryOp op, const NDArray& a, const NDArray& out);

// NumPy a.real / a.imag (D-033). For complex input this is a strided view of
// the real-typed component, sharing a's buffer and writeability. For real
// input, `real` returns a itself (a view), and `imag` returns a new read-only
// zero array of a's dtype.
NDArray complex_part(const NDArray& a, bool imag);

// NumPy iscomplex (want_complex) / isreal: a bool array testing imag != 0.
NDArray is_complex_elementwise(const NDArray& a, bool want_complex);

}  // namespace nativpy
