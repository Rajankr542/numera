#pragma once

// Helpers shared by the P10 C++ tests.
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <vector>

#include "creation.hpp"
#include "ndarray.hpp"

namespace p10test {

using nativpy::DType;
using nativpy::NDArray;
using nativpy::Shape;
using D = std::vector<double>;

inline NDArray dbl(std::initializer_list<double> v, DType dt = DType::Float64, Shape shape = {}) {
  NDArray a = NDArray::empty({static_cast<std::int64_t>(v.size())}, DType::Float64);
  std::int64_t i = 0;
  for (auto x : v) a.set_double(i++, x);
  const NDArray c = a.astype(dt);
  return shape.empty() ? c : c.reshape(shape);
}

inline D vals(const NDArray& a) {
  D out;
  for (std::int64_t i = 0; i < a.size(); ++i) out.push_back(a.get_double(i));
  return out;
}

// Element-wise equality treating NaN == NaN, with a relative tolerance.
inline bool close(const D& a, const D& b, double rtol = 1e-12) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (std::isnan(a[i]) && std::isnan(b[i])) continue;
    if (a[i] == b[i]) continue;
    if (!(std::fabs(a[i] - b[i]) <= rtol * std::fabs(b[i]))) return false;
  }
  return true;
}

}  // namespace p10test
