#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p06_manip.hpp"

namespace nativpy::bindings {

namespace {

using util::Info;

std::optional<DType> opt_dtype(const Napi::Value& v) {
  if (util::is_nullish(v)) return std::nullopt;
  return parse_dtype(v);
}

Casting casting_or(const Napi::Value& v, Casting def) {
  return util::is_nullish(v) ? def : parse_casting(v);
}

const NDArray* opt_array(const Napi::Value& v) {
  return util::is_nullish(v) ? nullptr : &NDArrayWrap::unwrap(v);
}

}  // namespace

// Native functions for parity milestone P6 (D-056), exposed as `addon.p06`.
void init_p06_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  // concatenate(arrays, axis|null, dtype|null, casting|null, out|null)
  m.Set("concatenate", fn(env, "concatenate", [](Info i, Napi::Env e) {
          return wrap(e, concatenate(arg_arrays(i[0], "arrays"), opt_int(i[1], "axis"),
                                     opt_dtype(i[2]), casting_or(i[3], Casting::SameKind),
                                     opt_array(i[4])));
        }));
  m.Set("stack", fn(env, "stack", [](Info i, Napi::Env e) {
          return wrap(e, stack(arg_arrays(i[0], "arrays"), arg_int(i[1], "axis"), opt_dtype(i[2]),
                               casting_or(i[3], Casting::SameKind), opt_array(i[4])));
        }));
  m.Set("splitAt", fn(env, "splitAt", [](Info i, Napi::Env e) -> Napi::Value {
          return wrap_all(e, split_at(arr(i, 0), arg_ints(i[1], "indices"), arg_int(i[2], "axis")));
        }));
  m.Set("splitSections", fn(env, "splitSections", [](Info i, Napi::Env e) -> Napi::Value {
          return wrap_all(e, split_sections(arr(i, 0), arg_int(i[1], "sections"),
                                            arg_int(i[2], "axis"), arg_bool(i[3])));
        }));
  m.Set("unstack", fn(env, "unstack", [](Info i, Napi::Env e) -> Napi::Value {
          return wrap_all(e, unstack(arr(i, 0), arg_int(i[1], "axis")));
        }));
  m.Set("tile", fn(env, "tile", [](Info i, Napi::Env e) {
          return wrap(e, tile(arr(i, 0), arg_ints(i[1], "reps")));
        }));
  m.Set("repeat", fn(env, "repeat", [](Info i, Napi::Env e) {
          return wrap(e, repeat(arr(i, 0), arg_ints(i[1], "repeats"), opt_int(i[2], "axis")));
        }));
  m.Set("resize", fn(env, "resize", [](Info i, Napi::Env e) {
          return wrap(e, resize(arr(i, 0), arg_ints(i[1], "shape")));
        }));
  m.Set("resizeInplace", fn(env, "resizeInplace", [](Info i, Napi::Env e) {
          return wrap(e, resize_inplace_data(arr(i, 0), arg_ints(i[1], "shape")));
        }));
  exports.Set("p06", m);
}

}  // namespace nativpy::bindings
