#include <napi.h>

#include "dtype_binding.hpp"
#include "linalg_binding.hpp"
#include "ndarray_binding.hpp"
#include "ops_binding.hpp"
#include "random_binding.hpp"

namespace {

Napi::Object init(Napi::Env env, Napi::Object exports) {
  nativpy::bindings::init_dtype_binding(env, exports);
  nativpy::bindings::init_ndarray_binding(env, exports);
  nativpy::bindings::init_ops_binding(env, exports);
  nativpy::bindings::init_linalg_binding(env, exports);
  nativpy::bindings::init_random_binding(env, exports);
  exports.Set("napiVersion", Napi::Number::New(env, NAPI_VERSION));
  return exports;
}

}  // namespace

NODE_API_MODULE(nativpy, init)
