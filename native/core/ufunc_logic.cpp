#include "ufunc_loops.hpp"
#include "ufunc_registry.hpp"

namespace nativpy {

// P5: comparison, logical, isnan/isinf/isfinite, bitwise ufuncs (D-056).
// Add entries to the table below; find_ufunc picks them up by name.

using namespace ufunc_loops;

namespace {

constexpr std::array<Ufunc, 0> kTable{};

}  // namespace

std::span<const Ufunc> logic_ufuncs() noexcept { return kTable; }

}  // namespace nativpy
