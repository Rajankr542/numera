#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p11_linalg.hpp"

namespace nativpy::bindings {

namespace {

linalg::NormOrd norm_ord(const Napi::Value& v) {
  linalg::NormOrd ord;
  if (v.IsString()) {
    ord.kind = v.As<Napi::String>().Utf8Value();
    if (ord.kind != "fro" && ord.kind != "nuc") {
      throw_error(ErrorKind::Value, "Invalid norm order '" + ord.kind + "' for matrices");
    }
  } else if (v.IsNumber()) {
    ord.kind = "p";
    ord.p = v.As<Napi::Number>().DoubleValue();
  } else if (!util::is_nullish(v)) {
    throw_error(ErrorKind::Value, "ord must be null, 'fro', 'nuc' or a number");
  }
  return ord;
}

std::optional<NDArray> opt_arr(const Napi::Value& v) {
  if (util::is_nullish(v)) return std::nullopt;
  return NDArrayWrap::unwrap(v);
}

}  // namespace

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
  m.Set("pinv", fn(env, "pinv", [](Info i, Napi::Env e) {
          return wrap(e, linalg::pinv(arr(i, 0), arr(i, 1), arg_bool(i[2])));
        }));
  m.Set("matrixRank", fn(env, "matrixRank", [](Info i, Napi::Env e) {
          return wrap(e, linalg::matrix_rank(arr(i, 0), opt_arr(i[1]), opt_arr(i[2]), arg_bool(i[3])));
        }));
  m.Set("cond", fn(env, "cond", [](Info i, Napi::Env e) {
          return wrap(e, linalg::cond(arr(i, 0), norm_ord(i[1])));
        }));
  exports.Set("p11", m);
}

}  // namespace nativpy::bindings
