#pragma once

// Distributions ported from NumPy 2.x numpy/random/src/distributions
// (BSD-3-Clause), D-019. Sampling order matches NumPy for bit-exactness.

#include <cstdint>

#include "bitgen.hpp"
#include "ndarray.hpp"

namespace nativpy::random {

__extension__ typedef __int128 i128;

// next_float: 24-bit float32 in [0, 1).
float next_float(BitGen& g);
// Ziggurat standard normal (Generator).
double standard_normal(BitGen& g);
float standard_normal_f(BitGen& g);
// Uniform integer in [0, max] (masked rejection, NumPy random_interval).
std::uint64_t random_interval(BitGen& g, std::uint64_t max);
// random_bounded_uint64 (single draw, unbuffered), used by Generator.choice.
std::uint64_t bounded_uint64(BitGen& g, std::uint64_t off, std::uint64_t rng, bool masked);

// Legacy (RandomState) gauss with cached second value.
struct LegacyGauss {
  bool has_gauss = false;
  double gauss = 0.0;
  double next(BitGen& g);
};

// NumPy _rand_{dtype}: integers in [low, high) (or [low, high] if closed),
// filled in C order into a new array. Validates bounds with NumPy messages.
// dtype must be Bool or an integer dtype.
NDArray bounded_integers(BitGen& g, i128 low, i128 high, bool closed, const Shape& shape,
                         DType dtype, bool masked);

// Fill a new float64/float32 array.
NDArray random_doubles(BitGen& g, const Shape& shape, DType dtype);
NDArray uniform(BitGen& g, double low, double high, const Shape& shape);
NDArray normal(BitGen& g, double loc, double scale, const Shape& shape, DType dtype);
NDArray legacy_normal(BitGen& g, LegacyGauss& gs, double loc, double scale, const Shape& shape);

// In-place Fisher–Yates along axis 0 (random_interval, as NumPy). `a` must be
// writeable; used after swapaxes for Generator's axis argument.
void shuffle(BitGen& g, const NDArray& a);

// Generator.choice index generation (no p). Returns int64 indices, shape `shape`.
NDArray generator_choice_indices(BitGen& g, std::int64_t pop_size, const Shape& shape,
                                 bool replace, bool shuffle_result);

}  // namespace nativpy::random
