#pragma once

#include <napi.h>

#include <exception>
#include <type_traits>
#include <utility>

namespace nativpy::bindings {

// Throws a JS Error whose `name` is the nativpy error kind (e.g. "ShapeError")
// and whose `code` is "NATIVPY_<KIND>". Never leaks raw C++ exceptions (D-006).
[[noreturn]] void throw_js_error(Napi::Env env, const char* name, const char* message);

// Converts the currently-caught exception into a JS exception.
[[noreturn]] void rethrow_as_js(Napi::Env env, std::exception_ptr eptr);

// Emits FP warnings queued by native code (np.seterr "warn"/"print", D-054)
// as Node `RuntimeWarning`s (or stdout lines). Cheap when nothing is queued.
void emit_fp_warnings(Napi::Env env);

template <typename Fn>
auto translate_errors(Napi::Env env, Fn&& fn) -> decltype(fn()) {
  try {
    if constexpr (std::is_void_v<decltype(fn())>) {
      fn();
      emit_fp_warnings(env);
    } else {
      auto r = fn();
      emit_fp_warnings(env);
      return r;
    }
  } catch (const Napi::Error&) {
    throw;  // already a JS error
  } catch (...) {
    rethrow_as_js(env, std::current_exception());
  }
}

}  // namespace nativpy::bindings
