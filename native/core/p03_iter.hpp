#pragma once

#include <cstddef>
#include <vector>

#include "ndarray.hpp"
#include "ufunc.hpp"

namespace nativpy {

// Iteration plan of NumPy nditer (D-061): the broadcast shape of `ops`, the
// order in which its axes are walked (outermost first) and, per axis, whether
// it is walked backwards (K order flips axes on which no operand has a
// positive stride and some operand has a negative one).
struct IterPlan {
  Shape shape;
  std::vector<std::size_t> axes;
  std::vector<bool> flipped;
};

IterPlan iter_plan(const std::vector<NDArray>& ops, Order order);

}  // namespace nativpy
