#include "fp_errors.hpp"

#include <cfenv>

#include "error.hpp"

namespace nativpy {

namespace {
thread_local ErrState g_state;
thread_local int g_depth = 0;
thread_local std::vector<FpWarning> g_warnings;
}  // namespace

ErrState get_errstate() noexcept { return g_state; }
void set_errstate(const ErrState& s) noexcept { g_state = s; }

std::vector<FpWarning> take_fp_warnings() {
  std::vector<FpWarning> w;
  w.swap(g_warnings);
  return w;
}

FpScope::FpScope(const char* name) noexcept : name_(name), outer_(g_depth == 0) {
  ++g_depth;
  if (outer_) {
    std::feclearexcept(FE_ALL_EXCEPT);
    g_warnings.clear();
  }
}

FpScope::~FpScope() { --g_depth; }

void FpScope::check() const {
  if (!outer_) return;
  const int flags = std::fetestexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_UNDERFLOW | FE_INVALID);
  std::feclearexcept(FE_ALL_EXCEPT);
  if (flags == 0) return;
  // NumPy order: divide, over, under, invalid.
  const struct {
    int flag;
    FpMode mode;
    const char* what;
  } kinds[] = {
      {FE_DIVBYZERO, g_state.divide, "divide by zero"},
      {FE_OVERFLOW, g_state.over, "overflow"},
      {FE_UNDERFLOW, g_state.under, "underflow"},
      {FE_INVALID, g_state.invalid, "invalid value"},
  };
  for (const auto& k : kinds) {
    if ((flags & k.flag) == 0 || k.mode == FpMode::Ignore) continue;
    const std::string msg = std::string(k.what) + " encountered in " + name_;
    if (k.mode == FpMode::Raise) throw_error(ErrorKind::FloatingPoint, msg);
    g_warnings.push_back({msg, k.mode == FpMode::Print});
  }
}

void raise_fp_divbyzero() noexcept { std::feraiseexcept(FE_DIVBYZERO); }
void raise_fp_overflow() noexcept { std::feraiseexcept(FE_OVERFLOW); }

}  // namespace nativpy
