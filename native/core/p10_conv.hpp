#pragma once

// correlate(a, v, mode) and convolve(a, v, mode) — 1-D cross-correlation and
// discrete linear convolution (D-135, P10-6), following NumPy's
// numpy/_core/multiarray.py and the reference C implementation.
//
// correlate:  c[k] = Σ_j  a[j+k] · conj(v[j])
// convolve:   c[k] = Σ_j  a[j]   · v[k-j]   (equivalent to correlate with
//                                              v reversed and un-conjugated)
//
// Modes (identical to NumPy):
//   "full"  — output length M+K-1  (all overlapping positions)
//   "same"  — output length max(M,K), centred on the longer array
//   "valid" — output length |M-K|+1 (no zero-padding at either end);
//             when K > M for 'valid', the behaviour matches NumPy (result is
//             computed as correlate(v.conj(), a.conj(), "valid").conj()).
//
// Both functions accept any numeric dtype; the result dtype is
// promote_types(a.dtype(), v.dtype()).  float16 arithmetic is done in float32.
// Both a and v must be 1-D (or 0-D scalars, treated as length-1).

#include <string>

#include "ndarray.hpp"

namespace nativpy::p10 {

// NumPy correlate(a, v, mode).
NDArray correlate(const NDArray& a, const NDArray& v, const std::string& mode);

// NumPy convolve(a, v, mode).
NDArray convolve(const NDArray& a, const NDArray& v, const std::string& mode);

}  // namespace nativpy::p10
