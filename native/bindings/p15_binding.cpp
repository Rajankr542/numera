#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p15_emath.hpp"

namespace nativpy::bindings {

// Native functions for parity milestone P15 (D-056), exposed as `addon.p15`.
void init_p15_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  // emath(name, x): numpy.emath sqrt/log/log2/log10/arccos/arcsin/arctanh (D-180).
  m.Set("emath", fn(env, "emath", [](Info i, Napi::Env e) {
          const std::string name = arg_string(i[0], "name");
          const auto op = emath_op_from_name(name);
          if (!op) throw_error(ErrorKind::Value, "unknown emath function '" + name + "'");
          return wrap(e, emath_unary(*op, arr(i, 1)));
        }));
  exports.Set("p15", m);
}

}  // namespace nativpy::bindings
