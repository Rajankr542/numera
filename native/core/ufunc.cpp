#include "ufunc.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <type_traits>

#include "broadcast.hpp"
#include "cast.hpp"
#include "error.hpp"
#include "ufunc_registry.hpp"

namespace nativpy {

namespace {

DType complex_real_dtype(DType dt) noexcept {
  return dt == DType::Complex64 ? DType::Float32 : DType::Float64;
}

}  // namespace

std::optional<BinaryOp> binary_op_from_name(std::string_view n) noexcept {
  for (int k = 0; k <= static_cast<int>(BinaryOp::FloorDivide); ++k) {
    const auto op = static_cast<BinaryOp>(k);
    if (n == get(op).name) return op;
  }
  return std::nullopt;
}

std::optional<UnaryOp> unary_op_from_name(std::string_view n) noexcept {
  for (int k = 0; k <= static_cast<int>(UnaryOp::Angle); ++k) {
    const auto op = static_cast<UnaryOp>(k);
    if (n == get(op).name) return op;
  }
  return std::nullopt;
}

std::optional<Order> order_from_name(std::string_view n) noexcept {
  if (n.size() != 1) return std::nullopt;
  switch (n[0]) {
    case 'C': case 'c': return Order::C;
    case 'F': case 'f': return Order::F;
    case 'A': case 'a': return Order::A;
    case 'K': case 'k': return Order::K;
    default: return std::nullopt;
  }
}

namespace {

// NumPy try_trivial_single_output_loop (D-050): returns the C/F layout NumPy's
// trivial path picks, or nullopt when it would fall back to the iterator.
std::optional<Order> trivial_order(const std::vector<const NDArray*>& inputs, Order order) {
  // Contiguity flags as bits (1 = C, 2 = F); 0 = not fixed yet.
  int want = order == Order::C ? 1 : order == Order::F ? 2 : 0;
  const Shape* shape = nullptr;
  for (const NDArray* a : inputs) {
    if (a->ndim() == 0) continue;
    if (!shape) shape = &a->shape();
    else if (a->shape() != *shape) return std::nullopt;
    if (a->ndim() < 2) continue;
    // Both flags at once (e.g. one non-unit dim) only matches itself.
    const int flags = (a->is_c_contiguous() ? 1 : 0) | (a->is_f_contiguous() ? 2 : 0);
    if (flags == 0) return std::nullopt;
    if (want == 0) want = flags;
    else if (flags != want) return std::nullopt;
  }
  return want == 2 ? Order::F : Order::C;
}

// Operand strides aligned to `shape`, 0 on every dim the operand has extent 1
// (NumPy npyiter_fill_axisdata).
Strides iter_strides(const NDArray& a, const Shape& shape) {
  Strides s(shape.size(), 0);
  const std::size_t off = shape.size() - a.ndim();
  for (std::size_t d = 0; d < a.ndim(); ++d) {
    s[off + d] = a.shape()[d] == 1 ? 0 : a.strides()[d];
  }
  return s;
}

}  // namespace

// NumPy npyiter_find_best_axis_ordering: array axes, outermost first.
std::vector<std::size_t> keep_order_axes(const Shape& shape,
                                         const std::vector<const NDArray*>& ops) {
  const std::size_t nd = shape.size();
  std::vector<Strides> st;
  st.reserve(ops.size());
  for (const NDArray* a : ops) st.push_back(iter_strides(*a, shape));
  // perm[i] = array axis at iterator position i (i = 0 innermost).
  std::vector<std::size_t> perm(nd);
  for (std::size_t i = 0; i < nd; ++i) perm[i] = nd - 1 - i;
  for (std::size_t i0 = 1; i0 < nd; ++i0) {
    std::size_t ipos = i0;
    const std::size_t j0 = perm[i0];
    for (std::size_t i1 = i0; i1-- > 0;) {
      const std::size_t j1 = perm[i1];
      bool ambig = true;
      bool swap = false;
      for (const Strides& s : st) {
        if (s[j0] == 0 || s[j1] == 0) continue;
        if (std::llabs(s[j1]) <= std::llabs(s[j0])) swap = false;  // C order wins conflicts
        else if (ambig) swap = true;
        ambig = false;
      }
      if (ambig) continue;
      if (!swap) break;
      ipos = i1;
    }
    if (ipos != i0) {
      for (std::size_t i = i0; i > ipos; --i) perm[i] = perm[i - 1];
      perm[ipos] = j0;
    }
  }
  return {perm.rbegin(), perm.rend()};
}

Strides ufunc_result_strides(const Shape& shape, std::size_t itemsize,
                             const std::vector<const NDArray*>& inputs,
                             const std::vector<bool>& cast, const NDArray* where,
                             Order order) {
  const std::size_t nd = shape.size();
  if (shape_size(shape) == 0 || nd == 0) return allocation_strides(shape, itemsize);
  // NumPy check_for_trivial_loop (no `where` only): in input order, a cast
  // 0-d or short 1-d input (<= NPY_BUFSIZE elements) is replaced by a
  // contiguous cast copy; any other cast input disables the trivial path and
  // stops the copying.
  constexpr std::int64_t kNpyBufsize = 8192;
  std::vector<bool> copied(inputs.size(), false);
  bool trivial_ok = where == nullptr;
  for (std::size_t k = 0; k < inputs.size() && trivial_ok; ++k) {
    if (!cast[k]) continue;
    const NDArray& a = *inputs[k];
    if (a.ndim() == 0 || (a.ndim() == 1 && a.shape()[0] <= kNpyBufsize)) copied[k] = true;
    else trivial_ok = false;
  }
  if (trivial_ok) {
    if (const auto t = trivial_order(inputs, order)) order = *t;
  }
  std::vector<const NDArray*> ops = inputs;
  if (where) ops.push_back(where);
  if (order == Order::A) {
    bool all_f = true;
    for (std::size_t k = 0; k < ops.size(); ++k) {
      const bool is_copy = k < copied.size() && copied[k];  // contiguous 0-d/1-d copy
      all_f = all_f && (is_copy || ops[k]->is_f_contiguous());
    }
    order = all_f ? Order::F : Order::C;
  }
  std::vector<std::size_t> axes(nd);
  for (std::size_t i = 0; i < nd; ++i) axes[i] = order == Order::F ? nd - 1 - i : i;
  if (order == Order::K) axes = keep_order_axes(shape, ops);
  Strides out(nd, 0);
  auto acc = static_cast<std::int64_t>(itemsize);
  for (std::size_t i = nd; i-- > 0;) {
    out[axes[i]] = acc;
    acc *= shape[axes[i]];
  }
  return out;
}

// ---- enum API: thin aliases over the registry (D-051) ----

DType binary_result_dtype(BinaryOp op, DType a, DType b) { return get(op).resolve(a, b).out; }
DType unary_result_dtype(UnaryOp op, DType in) { return get(op).resolve(in, in).out; }

NDArray binary(BinaryOp op, const NDArray& a, const NDArray& b, const NDArray* out,
               const UfuncParams& params) {
  return binary(get(op), a, b, out, params);
}

NDArray binary(BinaryOp op, const NDArray& a, const NDArray& b) {
  return binary(get(op), a, b, nullptr, UfuncParams{});
}

NDArray binary(BinaryOp op, const NDArray& a, const NDArray& b, const NDArray& out) {
  return binary(get(op), a, b, &out, UfuncParams{});
}

NDArray unary(UnaryOp op, const NDArray& a, const NDArray* out, const UfuncParams& params) {
  return unary(get(op), a, out, params);
}

NDArray unary(UnaryOp op, const NDArray& a) { return unary(get(op), a, nullptr, UfuncParams{}); }

NDArray unary(UnaryOp op, const NDArray& a, const NDArray& out) {
  return unary(get(op), a, &out, UfuncParams{});
}

NDArray complex_part(const NDArray& a, bool imag) {
  if (is_complex(a.dtype())) {
    const DType rt = complex_real_dtype(a.dtype());
    const std::int64_t off = a.offset() + (imag ? static_cast<std::int64_t>(itemsize(rt)) : 0);
    // Same strides (bytes) and buffer; reinterpret each element's re or im.
    NDArray v{a.buffer(), rt, a.shape(), a.strides(), off};
    return a.writeable() ? v : v.as_readonly();
  }
  if (!imag) return a.view(a.shape(), a.strides(), a.offset());
  return NDArray::zeros(a.shape(), a.dtype()).as_readonly();
}

NDArray is_complex_elementwise(const NDArray& a, bool want_complex) {
  NDArray out = NDArray::empty(a.shape(), DType::Bool);
  if (!is_complex(a.dtype())) {
    for (std::int64_t i = 0; i < out.size(); ++i) out.set_int64(i, want_complex ? 0 : 1);
    return out;
  }
  const NDArray im = complex_part(a, true);
  const auto plan = make_plan<2>(a.shape(), {&out, &im});
  const std::array<std::byte*, 2> base{out.data(), im.data()};
  const auto run = [&](auto zero) {
    using R = decltype(zero);
    run_plan(plan, base, [want_complex](const std::array<std::byte*, 2>& ptr,
                                        const std::array<std::int64_t, 2>& is, std::int64_t n) {
      for (std::int64_t i = 0; i < n; ++i) {
        const bool nonzero = load<R>(ptr[1] + i * is[1]) != R{0};  // NaN counts as nonzero
        store<bool>(ptr[0] + i * is[0], nonzero == want_complex);
      }
    });
  };
  if (im.dtype() == DType::Float32) run(float{});
  else run(double{});
  return out;
}

}  // namespace nativpy
