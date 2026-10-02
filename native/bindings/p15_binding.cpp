#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p15_emath.hpp"
#include "p15_polynomial.hpp"

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
  // poly(op, basis, ...): numpy.polynomial series kernels (D-182).
  m.Set("poly", fn(env, "poly", [](Info i, Napi::Env e) -> Napi::Value {
          const std::string op = arg_string(i[0], "op");
          const std::string bname = arg_string(i[1], "basis");
          const auto b = poly::basis_from_name(bname);
          if (!b) throw_error(ErrorKind::Value, "unknown polynomial basis '" + bname + "'");
          if (op == "add") return wrap(e, poly::add(*b, arr(i, 2), arr(i, 3)));
          if (op == "sub") return wrap(e, poly::sub(*b, arr(i, 2), arr(i, 3)));
          if (op == "mul") return wrap(e, poly::mul(*b, arr(i, 2), arr(i, 3)));
          if (op == "mulx") return wrap(e, poly::mulx(*b, arr(i, 2)));
          if (op == "div") {
            auto [q, r] = poly::div(*b, arr(i, 2), arr(i, 3));
            return wrap_all(e, {q, r});
          }
          if (op == "pow") return wrap(e, poly::pow(*b, arr(i, 2), arg_int(i[3], "pow")));
          if (op == "der") return wrap(e, poly::der(*b, arr(i, 2), arg_int(i[3], "m"), arg_double(i[4], "scl")));
          if (op == "int")
            return wrap(e, poly::integ(*b, arr(i, 2), arg_int(i[3], "m"), arr(i, 4), arr(i, 5),
                                       arg_double(i[6], "scl")));
          if (op == "val") return wrap(e, poly::val(*b, arr(i, 2), arr(i, 3)));
          if (op == "vander") return wrap(e, poly::vander(*b, arr(i, 2), arg_int(i[3], "deg")));
          if (op == "companion") return wrap(e, poly::companion(*b, arr(i, 2)));
          if (op == "fromroots") return wrap(e, poly::fromroots(*b, arr(i, 2)));
          if (op == "topower") return wrap(e, poly::to_power(*b, arr(i, 2)));
          if (op == "frompower") return wrap(e, poly::from_power(*b, arr(i, 2)));
          throw_error(ErrorKind::Value, "unknown polynomial op '" + op + "'");
        }));
  exports.Set("p15", m);
}

}  // namespace nativpy::bindings
