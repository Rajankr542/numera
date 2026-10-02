#include <cstring>

#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p03_methods.hpp"
#include "shape_ops.hpp"

namespace nativpy::bindings {

// Native functions for parity milestone P3 (D-056), exposed as `addon.p03`.
void init_p03_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);

  // ---- P3-2 NDArray methods (D-060) ----
  m.Set("fill", fn(env, "fill", [](Info i, Napi::Env e) {
          fill(arr(i, 0), arr(i, 1));
          return e.Undefined();
        }));
  m.Set("tobytes", fn(env, "tobytes", [](Info i, Napi::Env e) -> Napi::Value {
          const auto bytes = to_bytes(arr(i, 0), arg_order(i[1], Order::C));
          Napi::Uint8Array out = Napi::Uint8Array::New(e, bytes.size());
          if (!bytes.empty()) std::memcpy(out.Data(), bytes.data(), bytes.size());
          return out;
        }));
  m.Set("viewAs", fn(env, "viewAs", [](Info i, Napi::Env e) {
          return wrap(e, view_as(arr(i, 0), parse_dtype(i[1])));
        }));
  m.Set("byteswap", fn(env, "byteswap", [](Info i, Napi::Env e) -> Napi::Value {
          const bool inplace = arg_bool(i[1]);
          NDArray r = byteswap(arr(i, 0), inplace);
          if (inplace) return i[0];
          return wrap(e, std::move(r));
        }));
  m.Set("matrixTranspose", fn(env, "matrixTranspose", [](Info i, Napi::Env e) {
          const NDArray& a = arr(i, 0);
          if (a.ndim() < 2) throw_error(ErrorKind::Value, "matrix transpose with ndim < 2 is undefined");
          return wrap(e, swapaxes(a, -1, -2));
        }));
  // flatAssign(a, positions | null, values)
  m.Set("flatAssign", fn(env, "flatAssign", [](Info i, Napi::Env e) {
          const NDArray& a = arr(i, 0);
          if (is_nullish(i[1])) {
            flat_assign(a, nullptr, arr(i, 2));
          } else {
            flat_assign(a, &arr(i, 1), arr(i, 2));
          }
          return e.Undefined();
        }));

  exports.Set("p03", m);
}

}  // namespace nativpy::bindings
