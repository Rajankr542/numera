#pragma once

#include <array>
#include <optional>

#include "ndarray.hpp"

namespace nativpy {

// P4 multi-output ufuncs (D-073): NumPy divmod (2 in, 2 out), modf and frexp
// (1 in, 2 out). Inputs broadcast; outputs are new C-contiguous arrays, or
// are written into `out` (each cast from the loop dtype under `casting`).
enum class MultiOp { Divmod, Modf, Frexp };

struct MultiParams {
  std::optional<DType> dtype;  // loop dtype of the first output (dtype=)
  Casting casting = Casting::SameKind;
  const NDArray* out0 = nullptr;
  const NDArray* out1 = nullptr;
};

std::array<NDArray, 2> multi_ufunc(MultiOp op, const NDArray& a, const NDArray* b,
                                   const MultiParams& params);

}  // namespace nativpy
