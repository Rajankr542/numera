#include "dtype_binding.hpp"

#include <string>

#include "error.hpp"
#include "error_binding.hpp"

namespace nativpy::bindings {

DType parse_dtype(const Napi::Value& value) {
  if (!value.IsString()) {
    throw_error(ErrorKind::DType, "dtype must be a string name");
  }
  const std::string name = value.As<Napi::String>().Utf8Value();
  const auto dt = dtype_from_name(name);
  if (!dt) throw_error(ErrorKind::DType, "data type '" + name + "' not understood");
  return *dt;
}

namespace {
Napi::Value promote_types_js(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return translate_errors(env, [&]() -> Napi::Value {
    const DType a = parse_dtype(info[0]);
    const DType b = parse_dtype(info[1]);
    return Napi::String::New(env, std::string(dtype_name(promote_types(a, b))));
  });
}

// canCast(from, to, casting): dtype names and a casting rule name (D-045).
Napi::Value can_cast_js(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return translate_errors(env, [&]() -> Napi::Value {
    const DType from = parse_dtype(info[0]);
    const DType to = parse_dtype(info[1]);
    const std::string name = info[2].IsString() ? info[2].As<Napi::String>().Utf8Value() : "";
    const auto casting = casting_from_name(name);
    if (!casting) {
      throw_error(ErrorKind::Value,
                  "casting must be one of 'no', 'equiv', 'safe', 'same_kind', 'unsafe' (got '" +
                      name + "')");
    }
    return Napi::Boolean::New(env, can_cast(from, to, *casting));
  });
}
}  // namespace

void init_dtype_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object table = Napi::Object::New(env);
  for (int i = 0; i < kNumDTypes; ++i) {
    const DTypeInfo& di = dtype_info(static_cast<DType>(i));
    Napi::Object entry = Napi::Object::New(env);
    entry.Set("itemsize", Napi::Number::New(env, static_cast<double>(di.itemsize)));
    entry.Set("alignment", Napi::Number::New(env, static_cast<double>(di.alignment)));
    entry.Set("kind", Napi::String::New(env, std::string(1, di.kind)));
    table.Set(std::string(di.name), entry);
  }
  exports.Set("dtypes", table);
  exports.Set("promoteTypes", Napi::Function::New(env, promote_types_js, "promoteTypes"));
  exports.Set("canCast", Napi::Function::New(env, can_cast_js, "canCast"));
}

}  // namespace nativpy::bindings
