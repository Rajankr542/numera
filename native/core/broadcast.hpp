#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ndarray.hpp"

namespace nativpy {

// NumPy broadcasting of shapes. Throws ErrorKind::Broadcast on mismatch.
Shape broadcast_shapes(const std::vector<Shape>& shapes);

// Zero-stride read-only-by-convention view of `a` with the target shape.
NDArray broadcast_to(const NDArray& a, const Shape& shape);

// Iteration plan for N operands over a common (output) shape (PLAN §14).
// Adjacent dimensions that are contiguous for every operand are coalesced,
// so a fully contiguous op becomes one inner loop.
template <std::size_t N>
struct BroadcastPlan {
  Shape shape;                            // coalesced iteration shape (ndim >= 1)
  std::array<Strides, N> strides;         // byte strides per operand (0 = broadcast)
  std::int64_t size = 0;                  // total element count
};

namespace detail {
// Operand strides aligned to `out` (right-aligned, 0 on broadcast dims).
Strides aligned_strides(const NDArray& a, const Shape& out);
void coalesce(Shape& shape, std::vector<Strides*>& strides);
}  // namespace detail

template <std::size_t N>
BroadcastPlan<N> make_plan(const Shape& out, const std::array<const NDArray*, N>& ops) {
  BroadcastPlan<N> p;
  p.shape = out;
  p.size = shape_size(out);
  std::vector<Strides*> refs;
  for (std::size_t k = 0; k < N; ++k) {
    p.strides[k] = detail::aligned_strides(*ops[k], out);
    refs.push_back(&p.strides[k]);
  }
  detail::coalesce(p.shape, refs);
  return p;
}

// Calls inner(ptrs, inner_strides, count) once per innermost row.
// `base` holds each operand's starting byte pointer.
template <std::size_t N, typename Inner>
void run_plan(const BroadcastPlan<N>& p, std::array<std::byte*, N> base, Inner&& inner) {
  if (p.size == 0) return;
  const std::size_t nd = p.shape.size();
  const std::int64_t n_inner = p.shape[nd - 1];
  std::array<std::int64_t, N> is{};
  for (std::size_t k = 0; k < N; ++k) is[k] = p.strides[k][nd - 1];
  const std::int64_t rows = p.size / n_inner;
  std::vector<std::int64_t> idx(nd, 0);
  std::array<std::byte*, N> ptr = base;
  for (std::int64_t r = 0; r < rows; ++r) {
    inner(ptr, is, n_inner);
    for (std::size_t d = nd - 1; d-- > 0;) {
      if (++idx[d] < p.shape[d]) {
        for (std::size_t k = 0; k < N; ++k) ptr[k] += p.strides[k][d];
        break;
      }
      for (std::size_t k = 0; k < N; ++k) ptr[k] -= p.strides[k][d] * (p.shape[d] - 1);
      idx[d] = 0;
    }
  }
}

}  // namespace nativpy
