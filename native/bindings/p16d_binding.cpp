#include "binding_utils.hpp"
#include "milestone_bindings.hpp"

namespace nativpy::bindings {
// P16D native functions (D-190), exposed as `addon.p16d`.
void init_p16d_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  exports.Set("p16d", m);
}
}  // namespace nativpy::bindings
