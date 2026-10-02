#include "binding_utils.hpp"
#include "milestone_bindings.hpp"

namespace nativpy::bindings {
// P16A native functions (D-190), exposed as `addon.p16a`.
void init_p16a_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  exports.Set("p16a", m);
}
}  // namespace nativpy::bindings
