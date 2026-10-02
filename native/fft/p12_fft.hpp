#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "fft.hpp"
#include "ndarray.hpp"

namespace nativpy::fft::p12 {

// P12 FFT completion (D-150..D-152): `out=`, Hermitian and real N-D
// transforms, and fftshift/ifftshift. Built on the M10 kernels in fft.hpp.

enum class Op { Fft, Ifft, Rfft, Irfft, Hfft, Ihfft };
enum class NdOp { Fftn, Ifftn, Rfftn, Irfftn };

// 1-D transform along `axis`. Without `out` the result is a new array; with
// `out` it is written there (shape / same_kind cast / writeable checks, D-150)
// and `out` is returned.
NDArray transform(Op op, const NDArray& a, std::optional<std::int64_t> n, std::int64_t axis,
                  Norm norm, const std::optional<NDArray>& out);

// N-D transform with numpy.fft._cook_nd_args semantics for `s` / `axes`.
NDArray transform_nd(NdOp op, const NDArray& a, const std::optional<std::vector<std::int64_t>>& s,
                     const std::optional<std::vector<std::int64_t>>& axes, Norm norm,
                     const std::optional<NDArray>& out);

// fftshift (inverse = false) / ifftshift (inverse = true) over `axes`
// (all axes when empty optional). Returns a new C-contiguous array.
NDArray shift(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axes, bool inverse);

// np.roll over several axes (shifts add up for repeated axes).
NDArray roll(const NDArray& a, const std::vector<std::int64_t>& shifts,
             const std::vector<std::int64_t>& axes);

}  // namespace nativpy::fft::p12
