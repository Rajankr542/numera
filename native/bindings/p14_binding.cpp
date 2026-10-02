#include <cstring>

#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "dtype_binding.hpp"
#include "p14_npy.hpp"
#include "p14_text.hpp"

namespace nativpy::bindings {

namespace {
struct Bytes {
  const std::uint8_t* data;
  std::size_t len;
};

// Any TypedArray / Buffer / DataView viewed as bytes (no copy).
Bytes arg_bytes(const Napi::Value& v, const char* what) {
  if (v.IsTypedArray()) {
    const auto ta = v.As<Napi::TypedArray>();
    const auto* base = static_cast<const std::uint8_t*>(ta.ArrayBuffer().Data());
    return {base == nullptr ? nullptr : base + ta.ByteOffset(), ta.ByteLength()};
  }
  if (v.IsDataView()) {
    const auto dv = v.As<Napi::DataView>();
    const auto* base = static_cast<const std::uint8_t*>(dv.ArrayBuffer().Data());
    return {base == nullptr ? nullptr : base + dv.ByteOffset(), dv.ByteLength()};
  }
  if (v.IsArrayBuffer()) {
    auto ab = v.As<Napi::ArrayBuffer>();
    return {static_cast<const std::uint8_t*>(ab.Data()), ab.ByteLength()};
  }
  throw_error(ErrorKind::Value, std::string(what) + " must be a Buffer, TypedArray or ArrayBuffer");
}

Napi::Value to_buffer(Napi::Env env, const std::vector<std::uint8_t>& bytes) {
  return Napi::Buffer<std::uint8_t>::Copy(env, bytes.data(), bytes.size());
}
}  // namespace

// Native functions for parity milestone P14 (D-056, D-170..), exposed as `addon.p14`.
void init_p14_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  m.Set("npyEncode", fn(env, "npyEncode", [](Info i, Napi::Env e) { return to_buffer(e, p14::npy_encode(arr(i, 0))); }));
  m.Set("npyDecode", fn(env, "npyDecode", [](Info i, Napi::Env e) {
          const Bytes b = arg_bytes(i[0], "data");
          return wrap(e, p14::npy_decode(b.data, b.len));
        }));
  m.Set("crc32", fn(env, "crc32", [](Info i, Napi::Env e) {
          const Bytes b = arg_bytes(i[0], "data");
          const auto prev = is_nullish(i[1]) ? 0 : static_cast<std::uint32_t>(arg_int(i[1], "crc"));
          return Napi::Number::New(e, static_cast<double>(p14::crc32(b.data, b.len, prev)));
        }));
  // loadtxt(text, {delimiter, comments, quote, skiprows, usecols, maxRows, dtype})
  m.Set("loadtxt", fn(env, "loadtxt", [](Info i, Napi::Env e) {
          const std::string text = arg_string(i[0], "text");
          const Napi::Object o = i[1].As<Napi::Object>();
          p14::LoadtxtOptions opts;
          if (!is_nullish(o.Get("delimiter"))) opts.delimiter = arg_string(o.Get("delimiter"), "delimiter");
          if (!is_nullish(o.Get("quote"))) opts.quote = arg_string(o.Get("quote"), "quotechar");
          if (!is_nullish(o.Get("comments"))) {
            const auto c = o.Get("comments").As<Napi::Array>();
            for (std::uint32_t k = 0; k < c.Length(); ++k) opts.comments.push_back(arg_string(c.Get(k), "comments"));
          }
          if (!is_nullish(o.Get("skiprows"))) opts.skiprows = arg_int(o.Get("skiprows"), "skiprows");
          if (!is_nullish(o.Get("maxRows"))) opts.max_rows = arg_int(o.Get("maxRows"), "max_rows");
          opts.usecols = opt_ints(o.Get("usecols"), "usecols");
          opts.dtype = parse_dtype(o.Get("dtype"));
          return wrap(e, p14::loadtxt(text, opts));
        }));
  m.Set("formatRows", fn(env, "formatRows", [](Info i, Napi::Env e) {
          return Napi::String::New(e, p14::format_rows(arr(i, 0), arg_string(i[1], "fmt"), arg_string(i[2], "newline")));
        }));
  m.Set("tofileText", fn(env, "tofileText", [](Info i, Napi::Env e) {
          return Napi::String::New(e, p14::tofile_text(arr(i, 0), arg_string(i[1], "sep"), arg_string(i[2], "format")));
        }));
  m.Set("floatStr", fn(env, "floatStr", [](Info i, Napi::Env e) {
          return Napi::String::New(e, p14::float_str(arg_double(i[0], "x"), parse_dtype(i[1])));
        }));
  exports.Set("p14", m);
}

}  // namespace nativpy::bindings
