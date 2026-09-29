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
}

}  // namespace nativpy::bindings
