#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p11_linalg.hpp"

namespace nativpy::bindings {

// Native functions for parity milestone P11 (D-056, D-140..D-142), exposed as `addon.p11`.
void init_p11_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  m.Set("cholesky", fn(env, "cholesky", [](Info i, Napi::Env e) {
          return wrap(e, linalg::cholesky(arr(i, 0), arg_bool(i[1])));
        }));
  m.Set("slogdet", fn(env, "slogdet", [](Info i, Napi::Env e) -> Napi::Value {
          const auto r = linalg::slogdet(arr(i, 0));
          Napi::Object o = Napi::Object::New(e);
          o.Set("sign", wrap(e, r.sign));
          o.Set("logabsdet", wrap(e, r.logabsdet));
          return o;
        }));
  m.Set("matrixPower", fn(env, "matrixPower", [](Info i, Napi::Env e) {
          return wrap(e, linalg::matrix_power(arr(i, 0), arg_int(i[1], "exponent")));
        }));
  exports.Set("p11", m);
}

}  // namespace nativpy::bindings
