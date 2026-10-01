#include "broadcast.hpp"

#include <algorithm>
#include <string>

#include "cast.hpp"
#include "error.hpp"

namespace nativpy {

Shape broadcast_shapes(const std::vector<Shape>& shapes) {
  std::size_t nd = 0;
  for (const auto& s : shapes) nd = std::max(nd, s.size());
  Shape out(nd, 1);
  for (const auto& s : shapes) {
    const std::size_t off = nd - s.size();
    for (std::size_t i = 0; i < s.size(); ++i) {
      const std::int64_t d = s[i];
      std::int64_t& o = out[off + i];
      if (d == o || d == 1) continue;
      if (o == 1) {
        o = d;
        continue;
      }
      std::string msg = "operands could not be broadcast together with shapes";
      for (const auto& t : shapes) msg += " " + shape_to_string(t);
      throw_error(ErrorKind::Broadcast, msg);
    }
  }
  return out;
}

namespace detail {

Strides aligned_strides(const NDArray& a, const Shape& out) {
  const std::size_t nd = out.size();
  const std::size_t an = a.ndim();
  if (an > nd) throw_error(ErrorKind::Broadcast, "operand has more dimensions than output");
  Strides s(nd, 0);
  for (std::size_t i = 0; i < an; ++i) {
    const std::int64_t d = a.shape()[i];
    const std::size_t o = nd - an + i;
    if (d == out[o]) {
      s[o] = d == 1 ? 0 : a.strides()[i];
    } else if (d != 1) {
      throw_error(ErrorKind::Broadcast, "input operand with shape " + shape_to_string(a.shape()) +
                                            " cannot be broadcast to " + shape_to_string(out));
    }
  }
  return s;
}

void coalesce(Shape& shape, std::vector<Strides*>& strides) {
  // Drop size-1 dims (their strides never matter).
  Shape ns;
  std::vector<Strides> nst(strides.size());
  for (std::size_t d = 0; d < shape.size(); ++d) {
    if (shape[d] == 1) continue;
    ns.push_back(shape[d]);
    for (std::size_t k = 0; k < strides.size(); ++k) nst[k].push_back((*strides[k])[d]);
  }
  if (ns.empty()) {  // scalar-like (or size-0 handled by caller via size)
    ns.push_back(shape_size(shape) == 0 ? 0 : 1);
    for (auto& s : nst) s.push_back(0);
  }
  // Also keep a zero-size dim intact: shape_size is computed from original.
  if (std::find(shape.begin(), shape.end(), 0) != shape.end()) {
    shape = Shape{0};
    for (auto* s : strides) *s = Strides{0};
    return;
  }
  // Merge dim d into d+1 when outer stride == inner stride * inner extent.
  Shape cs{ns.back()};
  std::vector<Strides> cst(strides.size());
  for (std::size_t k = 0; k < strides.size(); ++k) cst[k].push_back(nst[k].back());
  for (std::size_t d = ns.size() - 1; d-- > 0;) {
    bool ok = true;
    for (std::size_t k = 0; k < strides.size() && ok; ++k) {
      ok = nst[k][d] == cst[k].front() * cs.front();
    }
    if (ok) {
      cs.front() *= ns[d];
    } else {
      cs.insert(cs.begin(), ns[d]);
      for (std::size_t k = 0; k < strides.size(); ++k) cst[k].insert(cst[k].begin(), nst[k][d]);
    }
  }
  shape = std::move(cs);
  for (std::size_t k = 0; k < strides.size(); ++k) *strides[k] = std::move(cst[k]);
}

}  // namespace detail

NDArray broadcast_to(const NDArray& a, const Shape& shape) {
  validate_shape(shape);
  if (a.ndim() > shape.size()) {
    throw_error(ErrorKind::Broadcast, "input operand has more dimensions than allowed by the axis remapping");
  }
  Strides s = detail::aligned_strides(a, shape);
  // Size-0 target: any strides are valid; keep 0. Read-only like NumPy (D-016).
  return a.view(shape, std::move(s), a.offset()).as_readonly();
}

void copy_into(const NDArray& dst, const NDArray& src_in) {
  const NDArray bsrc = broadcast_to(src_in, dst.shape());
  const NDArray src = bsrc.may_share_memory(dst) ? bsrc.copy() : bsrc;
  const auto plan = make_plan<2>(dst.shape(), {&dst, &src});
  dispatch_dtype(src.dtype(), [&](auto stag) {
    using S = dtype_t<decltype(stag)::value>;
    dispatch_dtype(dst.dtype(), [&](auto dtag) {
      using D = dtype_t<decltype(dtag)::value>;
      run_plan<2>(plan, {dst.data(), src.data()},
                  [](std::array<std::byte*, 2> p, const std::array<std::int64_t, 2>& s,
                     std::int64_t n) {
                    for (std::int64_t i = 0; i < n; ++i) {
                      store<D>(p[0] + i * s[0], cast_value<D>(load<S>(p[1] + i * s[1])));
                    }
                  });
    });
  });
}

}  // namespace nativpy
