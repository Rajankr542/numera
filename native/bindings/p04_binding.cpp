#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p04_multi.hpp"
#include "p04_special.hpp"

namespace nativpy::bindings {

namespace {

// multi(op, a, b | null, {dtype?, casting?}, out0 | null, out1 | null) -> [r0, r1]
Napi::Value multi(const Napi::CallbackInfo& info, Napi::Env env) {
  using namespace util;
  const std::string name = arg_string(info[0], "op");
  MultiOp op;
  if (name == "divmod") op = MultiOp::Divmod;
  else if (name == "modf") op = MultiOp::Modf;
  else if (name == "frexp") op = MultiOp::Frexp;
  else throw_error(ErrorKind::Value, "unknown multi-output ufunc " + name);
  const NDArray& a = arr(info, 1);
  const NDArray* b = is_nullish(info[2]) ? nullptr : &NDArrayWrap::unwrap(info[2]);
  MultiParams p;
  if (info[3].IsObject()) {
    const auto o = info[3].As<Napi::Object>();
    const Napi::Value dt = o.Get("dtype");
    if (!is_nullish(dt)) p.dtype = parse_dtype(dt);
    const Napi::Value casting = o.Get("casting");
    if (!casting.IsUndefined()) p.casting = parse_casting(casting);
  }
  if (info.Length() > 4 && !is_nullish(info[4])) p.out0 = &NDArrayWrap::unwrap(info[4]);
  if (info.Length() > 5 && !is_nullish(info[5])) p.out1 = &NDArrayWrap::unwrap(info[5]);
  const auto r = multi_ufunc(op, a, b, p);
  Napi::Array res = Napi::Array::New(env, 2);
  res.Set(0u, wrap(env, r[0]));
  res.Set(1u, wrap(env, r[1]));
  return res;
}

std::optional<double> opt_double(const Napi::Value& v, const char* what) {
  if (util::is_nullish(v)) return std::nullopt;
  return util::arg_double(v, what);
}

}  // namespace

// Native functions for parity milestone P4 (D-056), exposed as `addon.p04`.
void init_p04_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  m.Set("multi", fn(env, "multi", multi));
  m.Set("i0", fn(env, "i0", [](Info i, Napi::Env e) { return wrap(e, p04_i0(arr(i, 0))); }));
  m.Set("sinc", fn(env, "sinc", [](Info i, Napi::Env e) { return wrap(e, p04_sinc(arr(i, 0))); }));
  // nanToNum(a, copy, nan, posinf | null, neginf | null)
  m.Set("nanToNum", fn(env, "nanToNum", [](Info i, Napi::Env e) {
          return wrap(e, p04_nan_to_num(arr(i, 0), arg_bool(i[1]), arg_double(i[2], "nan"),
                                        opt_double(i[3], "posinf"), opt_double(i[4], "neginf")));
        }));
  m.Set("imagAllBelow", fn(env, "imagAllBelow", [](Info i, Napi::Env e) {
          return Napi::Boolean::New(e, p04_imag_all_below(arr(i, 0), arg_double(i[1], "tol")));
        }));
  // unwrap(p, period, discont | null, axis, integer, outDtype)
  m.Set("unwrap", fn(env, "unwrap", [](Info i, Napi::Env e) {
          return wrap(e, p04_unwrap(arr(i, 0), arg_double(i[1], "period"), opt_double(i[2], "discont"),
                                    arg_int(i[3], "axis"), arg_bool(i[4]), parse_dtype(i[5])));
        }));
  exports.Set("p04", m);
}

}  // namespace nativpy::bindings
