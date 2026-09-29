#pragma once

#include <napi.h>

#include <exception>
#include <utility>

namespace nativpy::bindings {

// Throws a JS Error whose `name` is the nativpy error kind (e.g. "ShapeError")
// and whose `code` is "NATIVPY_<KIND>". Never leaks raw C++ exceptions (D-006).
[[noreturn]] void throw_js_error(Napi::Env env, const char* name, const char* message);

// Converts the currently-caught exception into a JS exception.
[[noreturn]] void rethrow_as_js(Napi::Env env, std::exception_ptr eptr);

template <typename Fn>
auto translate_errors(Napi::Env env, Fn&& fn) -> decltype(fn()) {
  try {
    return fn();
  } catch (const Napi::Error&) {
    throw;  // already a JS error
  } catch (...) {
    rethrow_as_js(env, std::current_exception());
  }
}

}  // namespace nativpy::bindings
