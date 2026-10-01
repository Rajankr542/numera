#pragma once

// Shared argument helpers for the per-milestone binding files (D-056).
// Header-only so each milestone binding (p03_binding.cpp ... p15_binding.cpp)
// can use them without touching ops_binding.cpp.

#include <napi.h>

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "dtype_binding.hpp"
#include "error.hpp"
#include "error_binding.hpp"
#include "ndarray_binding.hpp"
#include "ufunc.hpp"

namespace nativpy::bindings::util {

using Info = const Napi::CallbackInfo&;

inline std::int64_t arg_int(const Napi::Value& v, const char* what) {
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  const double d = v.As<Napi::Number>().DoubleValue();
  if (!std::isfinite(d) || std::trunc(d) != d || std::fabs(d) > 9007199254740991.0) {
    throw_error(ErrorKind::Value, std::string(what) + " must be a safe integer");
  }
  return static_cast<std::int64_t>(d);
}

inline double arg_double(const Napi::Value& v, const char* what) {
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be a number");
  return v.As<Napi::Number>().DoubleValue();
}

inline bool arg_bool(const Napi::Value& v) { return v.ToBoolean().Value(); }

inline std::string arg_string(const Napi::Value& v, const char* what) {
  if (!v.IsString()) throw_error(ErrorKind::Value, std::string(what) + " must be a string");
  return v.As<Napi::String>().Utf8Value();
}

inline bool is_nullish(const Napi::Value& v) { return v.IsUndefined() || v.IsNull(); }

inline std::vector<std::int64_t> arg_ints(const Napi::Value& v, const char* what) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, std::string(what) + " must be an array");
  const auto a = v.As<Napi::Array>();
  std::vector<std::int64_t> out(a.Length());
  for (std::uint32_t i = 0; i < a.Length(); ++i) out[i] = arg_int(a.Get(i), what);
  return out;
}

inline std::vector<double> arg_doubles(const Napi::Value& v, const char* what) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, std::string(what) + " must be an array");
  const auto a = v.As<Napi::Array>();
  std::vector<double> out(a.Length());
  for (std::uint32_t i = 0; i < a.Length(); ++i) out[i] = arg_double(a.Get(i), what);
  return out;
}

inline std::optional<std::int64_t> opt_int(const Napi::Value& v, const char* what) {
  if (is_nullish(v)) return std::nullopt;
  return arg_int(v, what);
}

inline std::optional<std::vector<std::int64_t>> opt_ints(const Napi::Value& v, const char* what) {
  if (is_nullish(v)) return std::nullopt;
  return arg_ints(v, what);
}

inline const NDArray& arr(Info info, std::size_t i) { return NDArrayWrap::unwrap(info[i]); }

inline std::vector<NDArray> arg_arrays(const Napi::Value& v, const char* what) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, std::string(what) + " must be an array of arrays");
  const auto a = v.As<Napi::Array>();
  std::vector<NDArray> out;
  out.reserve(a.Length());
  for (std::uint32_t i = 0; i < a.Length(); ++i) out.push_back(NDArrayWrap::unwrap(a.Get(i)));
  return out;
}

inline Napi::Value wrap(Napi::Env env, NDArray a) { return NDArrayWrap::create(env, std::move(a)); }

inline Napi::Array wrap_all(Napi::Env env, const std::vector<NDArray>& v) {
  Napi::Array out = Napi::Array::New(env, v.size());
  for (std::uint32_t i = 0; i < v.size(); ++i) out.Set(i, NDArrayWrap::create(env, v[i]));
  return out;
}

inline Napi::Array ints_to_js(Napi::Env env, const std::vector<std::int64_t>& v) {
  Napi::Array out = Napi::Array::New(env, v.size());
  for (std::uint32_t i = 0; i < v.size(); ++i) out.Set(i, Napi::Number::New(env, static_cast<double>(v[i])));
  return out;
}

// Memory order: undefined/null = `def`; else "C"/"F"/"A"/"K" (either case).
inline Order arg_order(const Napi::Value& v, Order def) {
  if (is_nullish(v)) return def;
  if (!v.IsString()) throw_error(ErrorKind::DType, "order must be str, not a non-string value");
  const std::string name = v.As<Napi::String>().Utf8Value();
  const auto ord = order_from_name(name);
  if (!ord) throw_error(ErrorKind::Value, "order must be one of 'C', 'F', 'A', or 'K' (got '" + name + "')");
  return *ord;
}

// A JS function whose body runs under translate_errors (C++ errors -> typed JS errors).
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

}  // namespace nativpy::bindings::util
