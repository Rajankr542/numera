#include "binding_utils.hpp"
#include "milestone_bindings.hpp"

namespace nativpy::bindings {

// Native functions for parity milestone P5 (D-056), exposed as `addon.p05`.
void init_p05_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  // m.Set("name", fn(env, "name", [](Info i, Napi::Env e) { return wrap(e, ...); }));
  exports.Set("p05", m);
}

}  // namespace nativpy::bindings
