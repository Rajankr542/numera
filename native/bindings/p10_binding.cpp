#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p10_cumdiff.hpp"
#include "p10_quantile.hpp"

namespace nativpy::bindings {

namespace {

using namespace util;

Napi::Value prop(const Napi::Value& o, const char* key) {
  if (!o.IsObject()) return o.Env().Undefined();
  return o.As<Napi::Object>().Get(key);
}

bool opt_bool(const Napi::Value& o, const char* key) {
  const Napi::Value v = prop(o, key);
  return !is_nullish(v) && arg_bool(v);
}

std::optional<NDArray> opt_arr(const Napi::Value& v) {
  if (is_nullish(v)) return std::nullopt;
  return NDArrayWrap::unwrap(v);
}

std::optional<DType> opt_dtype(const Napi::Value& v) {
  if (is_nullish(v)) return std::nullopt;
  const std::string name = arg_string(v, "dtype");
  const auto dt = dtype_from_name(name);
  if (!dt) throw_error(ErrorKind::DType, "data type '" + name + "' not understood");
  return dt;
}

}  // namespace

// Native functions for parity milestone P10 (D-056, D-130..D-136), exposed as `addon.p10`.
void init_p10_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  // quantile(a, q, {axis: number[] | null, keepdims, method, weights, weakQ, percentile, ignoreNan})
  m.Set("quantile", fn(env, "quantile", [](Info i, Napi::Env e) {
          p10::QuantileOptions o;
          o.axis = opt_ints(prop(i[2], "axis"), "axis");
          o.keepdims = opt_bool(i[2], "keepdims");
          const Napi::Value method = prop(i[2], "method");
          if (!is_nullish(method)) o.method = p10::qmethod_from_name(arg_string(method, "method"));
          o.weights = opt_arr(prop(i[2], "weights"));
          o.weak_q = opt_bool(i[2], "weakQ");
          o.percentile = opt_bool(i[2], "percentile");
          o.ignore_nan = opt_bool(i[2], "ignoreNan");
          return wrap(e, p10::quantile(arr(i, 0), arr(i, 1), o));
        }));
  // median(a, axis: number[] | null, keepdims, ignoreNan)
  m.Set("median", fn(env, "median", [](Info i, Napi::Env e) {
          return wrap(e, p10::median(arr(i, 0), opt_ints(i[1], "axis"), arg_bool(i[2]), arg_bool(i[3])));
        }));
  // cumulative(prod, a, {axis, dtype, includeInitial, skipNan, arrayApi}, out | null)
  m.Set("cumulative", fn(env, "cumulative", [](Info i, Napi::Env e) {
          p10::CumulativeOptions o;
          o.axis = opt_int(prop(i[2], "axis"), "axis");
          o.dtype = opt_dtype(prop(i[2], "dtype"));
          o.include_initial = opt_bool(i[2], "includeInitial");
          o.skip_nan = opt_bool(i[2], "skipNan");
          o.array_api = opt_bool(i[2], "arrayApi");
          const std::optional<NDArray> out = opt_arr(i[3]);
          return wrap(e, p10::cumulative(arg_bool(i[0]), arr(i, 1), o, out ? &*out : nullptr));
        }));
  // diff(a, n, axis, prepend | null, append | null)
  m.Set("diff", fn(env, "diff", [](Info i, Napi::Env e) {
          return wrap(e, p10::diff(arr(i, 0), arg_int(i[1], "n"), arg_int(i[2], "axis"), opt_arr(i[3]),
                                   opt_arr(i[4])));
        }));
  // ptp(a, axis: number[] | null, keepdims)
  m.Set("ptp", fn(env, "ptp", [](Info i, Napi::Env e) {
          return wrap(e, p10::ptp(arr(i, 0), opt_ints(i[1], "axis"), arg_bool(i[2])));
        }));
  exports.Set("p10", m);
}

}  // namespace nativpy::bindings
