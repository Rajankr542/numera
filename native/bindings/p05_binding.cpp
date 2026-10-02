#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p05_logic.hpp"

namespace nativpy::bindings {

// Native functions for parity milestone P5 (D-056), exposed as `addon.p05`.
void init_p05_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  // isclose(a, b, rtol, atol, equalNan) (D-083).
  m.Set("isclose", fn(env, "isclose", [](Info i, Napi::Env e) {
          return wrap(e, isclose(arr(i, 0), arr(i, 1), arg_double(i[2], "rtol"),
                                 arg_double(i[3], "atol"), arg_bool(i[4])));
        }));
  // packbits(a, axis|null, little) / unpackbits(a, axis|null, count|null, little) (D-084).
  m.Set("packbits", fn(env, "packbits", [](Info i, Napi::Env e) {
          return wrap(e, packbits(arr(i, 0), opt_int(i[1], "axis"), arg_bool(i[2])));
        }));
  m.Set("unpackbits", fn(env, "unpackbits", [](Info i, Napi::Env e) {
          return wrap(e, unpackbits(arr(i, 0), opt_int(i[1], "axis"), opt_int(i[2], "count"), arg_bool(i[3])));
        }));
  exports.Set("p05", m);
}

}  // namespace nativpy::bindings
