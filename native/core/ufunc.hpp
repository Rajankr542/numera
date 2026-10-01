#pragma once

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "ndarray.hpp"

namespace nativpy {

// Element-wise arithmetic ufuncs (PLAN §15, M4). Inputs broadcast (M5);
// results get NumPy's layout for `order` (D-050). Semantics: D-014.

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

// NumPy `order=` (D-050): memory layout of a freshly allocated result.
enum class Order : std::uint8_t { C, F, A, K };
// "C"/"F"/"A"/"K" (either case), as NumPy's PyArray_OrderConverter.
std::optional<Order> order_from_name(std::string_view name) noexcept;

// NumPy `dtype=` / `casting=` (D-048). `dtype` picks the loop whose output is
// that dtype; every input must cast to its loop dtype, and the loop output to
// `out`, under `casting`.
// `where` (D-049): a bool mask that broadcasts like an extra input; only true
// positions are written (others keep `out`, or are zero without `out`).
// `order` (D-050): layout of the result when there is no `out` (default K).
struct UfuncParams {
  std::optional<DType> dtype;
  Casting casting = Casting::SameKind;
  std::optional<NDArray> where = std::nullopt;
  Order order = Order::K;
};

// NumPy npyiter_find_best_axis_ordering over `ops` broadcast to `shape`:
// array axes from outermost to innermost (C order wins ties).
std::vector<std::size_t> keep_order_axes(const Shape& shape,
                                         const std::vector<const NDArray*>& ops);

// Byte strides NumPy gives a new ufunc result of `shape` (D-050). `inputs`
// are the operands before any dtype cast; `cast[k]` says whether input k is
// cast to the loop dtype. `where` (if any) votes in 'A'/'K' like an operand.
Strides ufunc_result_strides(const Shape& shape, std::size_t itemsize,
                             const std::vector<const NDArray*>& inputs,
                             const std::vector<bool>& cast, const NDArray* where,
                             Order order);

NDArray binary(BinaryOp op, const NDArray& a, const NDArray& b, const NDArray* out,
               const UfuncParams& params);
NDArray unary(UnaryOp op, const NDArray& a, const NDArray* out, const UfuncParams& params);

// NumPy a.real / a.imag (D-033). For complex input this is a strided view of
// the real-typed component, sharing a's buffer and writeability. For real
// input, `real` returns a itself (a view), and `imag` returns a new read-only
// zero array of a's dtype.
NDArray complex_part(const NDArray& a, bool imag);

// NumPy iscomplex (want_complex) / isreal: a bool array testing imag != 0.
NDArray is_complex_elementwise(const NDArray& a, bool want_complex);

}  // namespace nativpy
