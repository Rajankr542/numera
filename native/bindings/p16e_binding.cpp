#include "binding_utils.hpp"
#include "milestone_bindings.hpp"

namespace nativpy::bindings {
// P16E native functions (D-190), exposed as `addon.p16e`.
void init_p16e_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  exports.Set("p16e", m);
}
}  // namespace nativpy::bindings
