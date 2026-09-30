#include "fft_binding.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "error.hpp"
#include "error_binding.hpp"
#include "fft.hpp"
#include "ndarray_binding.hpp"

namespace nativpy::bindings {

namespace {

using Info = const Napi::CallbackInfo&;

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

const NDArray& arr(Info info, std::size_t i) { return NDArrayWrap::unwrap(info[i]); }

Napi::Value wrap(Napi::Env e, const NDArray& a) { return NDArrayWrap::create(e, a); }

std::int64_t to_int(const Napi::Value& v, const char* what) {
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  const double d = v.As<Napi::Number>().DoubleValue();
  if (!std::isfinite(d) || std::trunc(d) != d) {
    throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  }
  return static_cast<std::int64_t>(d);
}

std::optional<std::int64_t> opt_int(const Napi::Value& v, const char* what) {
  if (v.IsUndefined() || v.IsNull()) return std::nullopt;
  return to_int(v, what);
}

std::optional<std::vector<std::int64_t>> opt_ints(const Napi::Value& v, const char* what) {
  if (v.IsUndefined() || v.IsNull()) return std::nullopt;
  if (!v.IsArray()) throw_error(ErrorKind::Value, std::string(what) + " must be an array");
  const auto a = v.As<Napi::Array>();
  std::vector<std::int64_t> out;
  out.reserve(a.Length());
  for (std::uint32_t k = 0; k < a.Length(); ++k) out.push_back(to_int(a.Get(k), what));
  return out;
}

fft::Norm norm_arg(const Napi::Value& v) {
  if (v.IsUndefined() || v.IsNull()) return fft::Norm::Backward;
  if (!v.IsString()) throw_error(ErrorKind::Value, "norm must be a string");
  return fft::parse_norm(v.As<Napi::String>().Utf8Value());
}

using OneD = NDArray (*)(const NDArray&, std::optional<std::int64_t>, std::int64_t, fft::Norm);
using ND = NDArray (*)(const NDArray&, const std::optional<std::vector<std::int64_t>>&,
                       const std::optional<std::vector<std::int64_t>>&, fft::Norm);
using Freq = NDArray (*)(std::int64_t, double);

}  // namespace

void init_fft_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object f = Napi::Object::New(env);
  // name(a, n | null, axis, norm | null)
  const auto one = [&](const char* name, OneD op) {
    f.Set(name, fn(env, name, [op](Info i, Napi::Env e) {
            return wrap(e, op(arr(i, 0), opt_int(i[1], "n"), to_int(i[2], "axis"), norm_arg(i[3])));
          }));
  };
  // name(a, s | null, axes | null, norm | null)
  const auto nd = [&](const char* name, ND op) {
    f.Set(name, fn(env, name, [op](Info i, Napi::Env e) {
            return wrap(e, op(arr(i, 0), opt_ints(i[1], "s"), opt_ints(i[2], "axes"),
                              norm_arg(i[3])));
          }));
  };
  // name(n, d)
  const auto freq = [&](const char* name, Freq op) {
    f.Set(name, fn(env, name, [op](Info i, Napi::Env e) {
            if (!i[1].IsNumber()) throw_error(ErrorKind::Value, "d must be a number");
            return wrap(e, op(to_int(i[0], "n"), i[1].As<Napi::Number>().DoubleValue()));
          }));
  };
  one("fft", &fft::fft);
  one("ifft", &fft::ifft);
  one("rfft", &fft::rfft);
  one("irfft", &fft::irfft);
  nd("fftn", &fft::fftn);
  nd("ifftn", &fft::ifftn);
  freq("fftfreq", &fft::fftfreq);
  freq("rfftfreq", &fft::rfftfreq);
  exports.Set("fft", f);
}

}  // namespace nativpy::bindings
