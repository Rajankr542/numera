#include "shape.hpp"

#include <limits>

#include "error.hpp"

namespace nativpy {

void validate_shape(const Shape& shape) {
  if (shape.size() > kMaxDims) {
    throw_error(ErrorKind::Value, "maximum supported dimension for an ndarray is " +
                                      std::to_string(kMaxDims) + ", found " +
                                      std::to_string(shape.size()));
  }
  for (const auto d : shape) {
    if (d < 0) {
      throw_error(ErrorKind::Value, "negative dimensions are not allowed");
    }
  }
  (void)shape_size(shape);
}

std::int64_t shape_size(const Shape& shape) {
  std::int64_t n = 1;
  bool has_zero = false;
  std::int64_t nonzero_prod = 1;
  for (const auto d : shape) {
    if (d == 0) {
      has_zero = true;
      continue;
    }
    if (d < 0) {
      throw_error(ErrorKind::Value, "negative dimensions are not allowed");
    }
    if (nonzero_prod > std::numeric_limits<std::int64_t>::max() / d) {
      throw_error(ErrorKind::Value, "array is too big; shape " +
                                        shape_to_string(shape) + " overflows int64");
    }
    nonzero_prod *= d;
  }
  n = has_zero ? 0 : nonzero_prod;
  return n;
}

std::int64_t normalize_axis(std::int64_t axis, std::int64_t ndim) {
  if (axis < -ndim || axis >= ndim) {
    throw_error(ErrorKind::Index, "axis " + std::to_string(axis) +
                                      " is out of bounds for array of dimension " +
                                      std::to_string(ndim));
  }
  return axis < 0 ? axis + ndim : axis;
}

Shape resolve_reshape(const Shape& target, std::int64_t size) {
  Shape out = target;
  std::int64_t unknown = -1;
  std::int64_t known = 1;
  for (std::size_t i = 0; i < out.size(); ++i) {
    if (out[i] == -1) {
      if (unknown != -1) {
        throw_error(ErrorKind::Value, "can only specify one unknown dimension");
      }
      unknown = static_cast<std::int64_t>(i);
    } else if (out[i] < 0) {
      throw_error(ErrorKind::Value, "negative dimensions not allowed");
    } else {
      known = shape_size(Shape{known, out[i]});
    }
  }
  const auto fail = [&] {
    throw_error(ErrorKind::Shape, "cannot reshape array of size " +
                                      std::to_string(size) + " into shape " +
                                      shape_to_string(target));
  };
  if (unknown >= 0) {
    if (known == 0 || size % known != 0) fail();
    out[static_cast<std::size_t>(unknown)] = size / known;
  } else if (known != size) {
    fail();
  }
  validate_shape(out);
  return out;
}

std::string shape_to_string(const Shape& shape) {
  std::string s = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i > 0) s += ", ";
    s += std::to_string(shape[i]);
  }
  if (shape.size() == 1) s += ",";
  s += ")";
  return s;
}

}  // namespace nativpy
