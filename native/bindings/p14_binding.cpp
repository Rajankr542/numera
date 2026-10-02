#include <cstring>

#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p14_npy.hpp"

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
  exports.Set("p14", m);
}

}  // namespace nativpy::bindings
