#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p08_indexing.hpp"

namespace nativpy::bindings {

namespace {

using namespace util;

ClipMode arg_mode(const Napi::Value& v) {
  if (is_nullish(v)) return ClipMode::Raise;
  return clip_mode_from_name(arg_string(v, "mode"));
}

std::vector<ClipMode> arg_modes(const Napi::Value& v) {
  if (!v.IsArray()) return {arg_mode(v)};
  const auto a = v.As<Napi::Array>();
  std::vector<ClipMode> out;
  for (std::uint32_t i = 0; i < a.Length(); ++i) out.push_back(arg_mode(a.Get(i)));
  return out;
}

std::optional<DType> opt_dtype(const Napi::Value& v) {
  if (is_nullish(v)) return std::nullopt;
  const std::string name = arg_string(v, "dtype");
  const auto dt = dtype_from_name(name);
  if (!dt) throw_error(ErrorKind::DType, "data type '" + name + "' not understood");
  return dt;
}

}  // namespace

// Native functions for parity milestone P8 (D-056, D-110), exposed as `addon.p08`.
void init_p08_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  // take(a, indices, axis | null, mode?)
  m.Set("take", fn(env, "take", [](Info i, Napi::Env e) {
          return wrap(e, take_mode(arr(i, 0), arr(i, 1), opt_int(i[2], "axis"), arg_mode(i[3])));
        }));
  m.Set("takeAlongAxis", fn(env, "takeAlongAxis", [](Info i, Napi::Env e) {
          return wrap(e, take_along_axis(arr(i, 0), arr(i, 1), opt_int(i[2], "axis")));
        }));
  m.Set("putAlongAxis", fn(env, "putAlongAxis", [](Info i, Napi::Env e) {
          put_along_axis(arr(i, 0), arr(i, 1), arr(i, 2), opt_int(i[3], "axis"));
          return e.Undefined();
        }));
  m.Set("put", fn(env, "put", [](Info i, Napi::Env e) {
          put(arr(i, 0), arr(i, 1), arr(i, 2), arg_mode(i[3]));
          return e.Undefined();
        }));
  // putmask(a, mask, values, safe) / place(a, mask, vals, safe)
  m.Set("putmask", fn(env, "putmask", [](Info i, Napi::Env e) {
          putmask(arr(i, 0), arr(i, 1), arr(i, 2), arg_bool(i[3]));
          return e.Undefined();
        }));
  m.Set("place", fn(env, "place", [](Info i, Napi::Env e) {
          place(arr(i, 0), arr(i, 1), arr(i, 2), arg_bool(i[3]));
          return e.Undefined();
        }));
  m.Set("choose", fn(env, "choose", [](Info i, Napi::Env e) {
          return wrap(e, choose(arr(i, 0), arg_arrays(i[1], "choices"), arg_mode(i[2])));
        }));
  m.Set("compress", fn(env, "compress", [](Info i, Napi::Env e) {
          return wrap(e, compress(arr(i, 0), arr(i, 1), opt_int(i[2], "axis")));
        }));
  m.Set("extract", fn(env, "extract", [](Info i, Napi::Env e) {
          return wrap(e, extract(arr(i, 0), arr(i, 1)));
        }));
  m.Set("select", fn(env, "select", [](Info i, Napi::Env e) {
          return wrap(e, select(arg_arrays(i[0], "condlist"), arg_arrays(i[1], "choicelist"), arr(i, 2)));
        }));
  m.Set("noneOf", fn(env, "noneOf", [](Info i, Napi::Env e) {
          return wrap(e, none_of(arg_arrays(i[0], "condlist")));
        }));
  m.Set("argwhere", fn(env, "argwhere", [](Info i, Napi::Env e) { return wrap(e, argwhere(arr(i, 0))); }));
  m.Set("flatnonzero", fn(env, "flatnonzero", [](Info i, Napi::Env e) {
          return wrap(e, flatnonzero(arr(i, 0)));
        }));
  // countNonzero(a, axis: number[] | null, keepdims)
  m.Set("countNonzero", fn(env, "countNonzero", [](Info i, Napi::Env e) {
          return wrap(e, count_nonzero(arr(i, 0), opt_ints(i[1], "axis"), arg_bool(i[2])));
        }));
  // ravelMultiIndex(arrays, dims, mode | mode[], order)
  m.Set("ravelMultiIndex", fn(env, "ravelMultiIndex", [](Info i, Napi::Env e) {
          const Shape dims = arg_ints(i[1], "dims");
          const std::vector<ClipMode> modes = arg_modes(i[2]);
          if (i[2].IsArray() && modes.size() != dims.size()) {
            throw_error(ErrorKind::Value, "list of clipmodes has wrong length (" +
                                              std::to_string(modes.size()) + " instead of " +
                                              std::to_string(dims.size()) + ")");
          }
          return wrap(e, ravel_multi_index(arg_arrays(i[0], "multi_index"), dims, modes,
                                           arg_order(i[3], Order::C)));
        }));
  m.Set("unravelIndex", fn(env, "unravelIndex", [](Info i, Napi::Env e) {
          return wrap_all(e, unravel_index(arr(i, 0), arg_ints(i[1], "shape"), arg_order(i[2], Order::C)));
        }));
  m.Set("diagonal", fn(env, "diagonal", [](Info i, Napi::Env e) {
          return wrap(e, diagonal(arr(i, 0), arg_int(i[1], "offset"), arg_int(i[2], "axis1"),
                                  arg_int(i[3], "axis2")));
        }));
  m.Set("trace", fn(env, "trace", [](Info i, Napi::Env e) {
          return wrap(e, trace(arr(i, 0), arg_int(i[1], "offset"), arg_int(i[2], "axis1"),
                               arg_int(i[3], "axis2"), opt_dtype(i[4])));
        }));
  exports.Set("p08", m);
}

}  // namespace nativpy::bindings
