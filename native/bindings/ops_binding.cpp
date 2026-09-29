#include "ops_binding.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "creation.hpp"
#include "dtype_binding.hpp"
#include "error.hpp"
#include "error_binding.hpp"
#include "ndarray_binding.hpp"
#include "shape_ops.hpp"

namespace nativpy::bindings {

namespace {

std::int64_t arg_int(const Napi::Value& v, const char* what) {
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  const double d = v.As<Napi::Number>().DoubleValue();
  if (!std::isfinite(d) || std::trunc(d) != d || std::fabs(d) > 9007199254740991.0) {
    throw_error(ErrorKind::Value, std::string(what) + " must be a safe integer");
  }
  return static_cast<std::int64_t>(d);
}

double arg_double(const Napi::Value& v, const char* what) {
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be a number");
  return v.As<Napi::Number>().DoubleValue();
}

std::vector<std::int64_t> arg_ints(const Napi::Value& v, const char* what) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, std::string(what) + " must be an array");
  const auto arr = v.As<Napi::Array>();
  std::vector<std::int64_t> out(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) out[i] = arg_int(arr.Get(i), what);
  return out;
}

template <typename Fn>
Napi::Function fn(Napi::Env env, const char* name, Fn body) {
  return Napi::Function::New(
      env,
      [body](const Napi::CallbackInfo& info) -> Napi::Value {
        Napi::Env e = info.Env();
        return translate_errors(e, [&]() -> Napi::Value { return body(info, e); });
      },
      name);
}

const NDArray& arr(const Napi::CallbackInfo& info, std::size_t i) {
  return NDArrayWrap::unwrap(info[i]);
}

}  // namespace

void init_ops_binding(Napi::Env env, Napi::Object exports) {
  using Info = const Napi::CallbackInfo&;
  // ---- creation (M2) ----
  exports.Set("full", fn(env, "full", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, full(arg_ints(i[0], "shape"), arr(i, 1)));
              }));
  exports.Set("ones", fn(env, "ones", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, ones(arg_ints(i[0], "shape"), parse_dtype(i[1])));
              }));
  exports.Set("arange", fn(env, "arange", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(
                    e, arange(arg_double(i[0], "start"), arg_double(i[1], "stop"),
                              arg_double(i[2], "step"), parse_dtype(i[3])));
              }));
  exports.Set("linspace", fn(env, "linspace", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(
                    e, linspace(arg_double(i[0], "start"), arg_double(i[1], "stop"),
                                arg_int(i[2], "num"), i[3].ToBoolean().Value(), parse_dtype(i[4])));
              }));
  exports.Set("eye", fn(env, "eye", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, eye(arg_int(i[0], "N"), arg_int(i[1], "M"),
                                                  arg_int(i[2], "k"), parse_dtype(i[3])));
              }));
  // ---- shape (M3) ----
  exports.Set("transpose", fn(env, "transpose", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, transpose(arr(i, 0), arg_ints(i[1], "axes")));
              }));
  exports.Set("squeeze", fn(env, "squeeze", [](Info i, Napi::Env e) {
                std::optional<std::vector<std::int64_t>> axes;
                if (!i[1].IsUndefined() && !i[1].IsNull()) axes = arg_ints(i[1], "axis");
                return NDArrayWrap::create(e, squeeze(arr(i, 0), axes));
              }));
  exports.Set("expandDims", fn(env, "expandDims", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, expand_dims(arr(i, 0), arg_ints(i[1], "axis")));
              }));
  exports.Set("swapaxes", fn(env, "swapaxes", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(
                    e, swapaxes(arr(i, 0), arg_int(i[1], "axis1"), arg_int(i[2], "axis2")));
              }));
  exports.Set("moveaxis", fn(env, "moveaxis", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, moveaxis(arr(i, 0), arg_ints(i[1], "source"),
                                                       arg_ints(i[2], "destination")));
              }));
  exports.Set("ravel", fn(env, "ravel", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, ravel(arr(i, 0)));
              }));
  exports.Set("flatten", fn(env, "flatten", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, flatten(arr(i, 0)));
              }));
}

}  // namespace nativpy::bindings
