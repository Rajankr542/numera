#pragma once

#include <string>
#include <vector>

namespace nativpy {

// NumPy floating-point error handling (np.seterr / np.errstate, D-054).

enum class FpMode { Ignore, Warn, Raise, Print };

struct ErrState {
  FpMode divide = FpMode::Warn;
  FpMode over = FpMode::Warn;
  FpMode under = FpMode::Ignore;
  FpMode invalid = FpMode::Warn;
};

// Current (per-thread) settings.
ErrState get_errstate() noexcept;
void set_errstate(const ErrState& s) noexcept;

// A warning queued by FpScope::check for the binding to emit.
struct FpWarning {
  std::string message;  // e.g. "divide by zero encountered in divide"
  bool print;           // FpMode::Print (stdout) instead of a RuntimeWarning
};
// Removes and returns the queued warnings.
std::vector<FpWarning> take_fp_warnings();

// Clears the FP exception flags on entry (outermost scope only); `check()`
// reads them and applies the error state, throwing ErrorKind::FloatingPoint
// for "raise". Nested scopes defer to the outermost one.
class FpScope {
 public:
  explicit FpScope(const char* name) noexcept;
  ~FpScope();
  FpScope(const FpScope&) = delete;
  FpScope& operator=(const FpScope&) = delete;
  void check() const;

 private:
  const char* name_;
  bool outer_;
};

// RAII guard: saves the current ErrState on construction and restores it on
// destruction (even if an exception unwinds the stack).  Use in tests and any
// code that temporarily changes the error mode.
class ErrStateGuard {
 public:
  ErrStateGuard() noexcept : saved_(get_errstate()) {}
  ~ErrStateGuard() noexcept { set_errstate(saved_); }
  ErrStateGuard(const ErrStateGuard&) = delete;
  ErrStateGuard& operator=(const ErrStateGuard&) = delete;

 private:
  ErrState saved_;
};

// Integer kernels: raise the matching FP flag (NumPy's integer error path).
void raise_fp_divbyzero() noexcept;
void raise_fp_overflow() noexcept;

}  // namespace nativpy
