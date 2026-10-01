#include "ufunc_loops.hpp"
#include "ufunc_registry.hpp"

namespace nativpy {

// P4: trig, exp/log, rounding, arithmetic, float bits, integer ufuncs (D-056).
// Add entries to the table below; find_ufunc picks them up by name.

using namespace ufunc_loops;

namespace {

constexpr std::array<Ufunc, 0> kTable{};

}  // namespace

std::span<const Ufunc> math_ufuncs() noexcept { return kTable; }

}  // namespace nativpy
