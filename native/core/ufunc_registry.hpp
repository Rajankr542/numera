#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

#include "broadcast.hpp"
#include "dtype.hpp"
#include "ufunc.hpp"

namespace nativpy {

// Table-driven ufunc registry (D-051). One immutable record per ufunc.

// Loop dtypes: inputs are cast to `in`, the loop writes `out`.
struct LoopTypes {
  DType in;
  DType out;
};

// Type-erased strided loops; base[0] is the output.
using BinaryLoopFn = void (*)(const BroadcastPlan<3>&, std::array<std::byte*, 3>);
using UnaryLoopFn = void (*)(const BroadcastPlan<2>&, std::array<std::byte*, 2>);

struct Ufunc {
  const char* name;
  int nin;                         // 1 or 2
  std::optional<double> identity;  // NumPy ufunc.identity (none = nullopt)
  // Default type resolver; `b` is ignored when nin == 1.
  LoopTypes (*resolve)(DType a, DType b);
  // dtype= loop selection (D-048); `a` is input 0's dtype.
  LoopTypes (*resolve_dtype)(DType a, DType d);
  // Optional check on the cast inputs before the loop runs. `mask`/`full`
  // are set for where= (D-049); only true positions are checked.
  void (*check)(const NDArray& a, const NDArray* b, const NDArray* mask, const Shape& full);
  // Indexed by the loop input dtype; null = no loop.
  std::array<BinaryLoopFn, kNumDTypes> binary_loops;
  std::array<UnaryLoopFn, kNumDTypes> unary_loops;
};

// Per-family tables (D-056), each defined in its own file so milestones can
// add ufuncs independently: P4 math (ufunc_math.cpp), P5 comparison/logic/
// bitwise (ufunc_logic.cpp). find_ufunc searches the core table first.
std::span<const Ufunc> math_ufuncs() noexcept;
std::span<const Ufunc> logic_ufuncs() noexcept;

// Null for an unknown name.
const Ufunc* find_ufunc(std::string_view name) noexcept;
const Ufunc& get(BinaryOp op) noexcept;
const Ufunc& get(UnaryOp op) noexcept;

// Generic drivers (out/where/dtype/casting/order, D-046..D-050).
NDArray binary(const Ufunc& u, const NDArray& a, const NDArray& b, const NDArray* out,
               const UfuncParams& params);
NDArray unary(const Ufunc& u, const NDArray& a, const NDArray* out, const UfuncParams& params);

}  // namespace nativpy
