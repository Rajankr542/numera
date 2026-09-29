#include "error_binding.hpp"

#include <cctype>
#include <new>
#include <string>

#include "error.hpp"

namespace nativpy::bindings {

void throw_js_error(Napi::Env env, const char* name, const char* message) {
  Napi::Error err = Napi::Error::New(env, message);
  err.Set("name", Napi::String::New(env, name));
  std::string code = "NATIVPY";
  for (const char* p = name; *p != '\0'; ++p) {
    const auto c = static_cast<unsigned char>(*p);
    if (std::isupper(c) != 0) code += '_';
    code += static_cast<char>(std::toupper(c));
  }
  err.Set("code", Napi::String::New(env, code));
  throw err;
}

void rethrow_as_js(Napi::Env env, std::exception_ptr eptr) {
  try {
    std::rethrow_exception(std::move(eptr));
  } catch (const nativpy::Error& e) {
    throw_js_error(env, error_kind_name(e.kind()), e.what());
  } catch (const std::bad_alloc&) {
    throw_js_error(env, "MemoryError", "native allocation failed");
  } catch (const std::exception& e) {
    throw_js_error(env, "NativpyError", e.what());
  } catch (...) {
    throw_js_error(env, "NativpyError", "unknown native error");
  }
}

}  // namespace nativpy::bindings
