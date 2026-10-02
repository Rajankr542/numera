#include "p03_iter.hpp"

#include "broadcast.hpp"

namespace nativpy {

IterPlan iter_plan(const std::vector<NDArray>& ops, Order order) {
  std::vector<Shape> shapes;
  shapes.reserve(ops.size());
  for (const NDArray& a : ops) shapes.push_back(a.shape());
  IterPlan p;
  p.shape = broadcast_shapes(shapes);
  const std::size_t nd = p.shape.size();
  p.flipped.assign(nd, false);
  if (order == Order::A) {
    bool all_f = true;
    for (const NDArray& a : ops) all_f = all_f && a.is_f_contiguous();
    order = all_f ? Order::F : Order::C;
  }
  if (order == Order::C) {
    for (std::size_t d = 0; d < nd; ++d) p.axes.push_back(d);
  } else if (order == Order::F) {
    for (std::size_t d = nd; d-- > 0;) p.axes.push_back(d);
  } else {
    std::vector<const NDArray*> ptrs;
    for (const NDArray& a : ops) ptrs.push_back(&a);
    p.axes = keep_order_axes(p.shape, ptrs);
    for (std::size_t d = 0; d < nd; ++d) {
      bool pos = false;
      bool neg = false;
      for (const NDArray& a : ops) {
        const std::size_t off = nd - a.ndim();
        if (d < off || a.shape()[d - off] == 1) continue;
        const std::int64_t s = a.strides()[d - off];
        pos = pos || s > 0;
        neg = neg || s < 0;
      }
      p.flipped[d] = neg && !pos;
    }
  }
  return p;
}

}  // namespace nativpy
