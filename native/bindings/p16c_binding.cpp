#include "binding_utils.hpp"
#include "milestone_bindings.hpp"

namespace nativpy::bindings {
// P16C native functions (D-190), exposed as `addon.p16c`.
void init_p16c_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  exports.Set("p16c", m);
}
}  // namespace nativpy::bindings
