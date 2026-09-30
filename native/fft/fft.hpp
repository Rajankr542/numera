#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ndarray.hpp"

namespace nativpy::fft {

// NumPy-compatible FFTs backed by pocketfft (PLAN §24, M10). Semantics: D-020.
// All results are new C-contiguous arrays.

enum class Norm { Backward, Ortho, Forward };

// Parses "backward" | "ortho" | "forward"; an empty string means backward.
Norm parse_norm(const std::string& s);

// 1-D transforms along `axis`. An empty `n` selects the default length
// (input length; irfft: 2 * (m - 1)).
NDArray fft(const NDArray& a, std::optional<std::int64_t> n, std::int64_t axis, Norm norm);
NDArray ifft(const NDArray& a, std::optional<std::int64_t> n, std::int64_t axis, Norm norm);
NDArray rfft(const NDArray& a, std::optional<std::int64_t> n, std::int64_t axis, Norm norm);
NDArray irfft(const NDArray& a, std::optional<std::int64_t> n, std::int64_t axis, Norm norm);

// N-D complex transforms. `s` / `axes` follow numpy.fft._cook_nd_args:
// no s and no axes = all axes; s without axes = the last len(s) axes;
// s[i] == -1 = the input length along axes[i].
NDArray fftn(const NDArray& a, const std::optional<std::vector<std::int64_t>>& s,
             const std::optional<std::vector<std::int64_t>>& axes, Norm norm);
NDArray ifftn(const NDArray& a, const std::optional<std::vector<std::int64_t>>& s,
              const std::optional<std::vector<std::int64_t>>& axes, Norm norm);

// Sample frequencies (float64).
NDArray fftfreq(std::int64_t n, double d);
NDArray rfftfreq(std::int64_t n, double d);

}  // namespace nativpy::fft
