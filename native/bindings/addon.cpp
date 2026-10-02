#include <napi.h>

#include "dtype_binding.hpp"
#include "fft_binding.hpp"
#include "linalg_binding.hpp"
#include "milestone_bindings.hpp"
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
  nativpy::bindings::init_fft_binding(env, exports);
  nativpy::bindings::init_p03_binding(env, exports);
  nativpy::bindings::init_p04_binding(env, exports);
  nativpy::bindings::init_p05_binding(env, exports);
  nativpy::bindings::init_p06_binding(env, exports);
  nativpy::bindings::init_p07_binding(env, exports);
  nativpy::bindings::init_p08_binding(env, exports);
  nativpy::bindings::init_p09_binding(env, exports);
  nativpy::bindings::init_p10_binding(env, exports);
  nativpy::bindings::init_p11_binding(env, exports);
  nativpy::bindings::init_p12_binding(env, exports);
  nativpy::bindings::init_p13_binding(env, exports);
  nativpy::bindings::init_p14_binding(env, exports);
  nativpy::bindings::init_p15_binding(env, exports);
  nativpy::bindings::init_p16e_binding(env, exports);
  exports.Set("napiVersion", Napi::Number::New(env, NAPI_VERSION));
  return exports;
}

}  // namespace

NODE_API_MODULE(nativpy, init)
