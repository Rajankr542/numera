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

std::vector<std::pair<std::int64_t, std::int64_t>> pairs(const Napi::Value& v, const char* what) {
  const auto flat = util::arg_ints(v, what);
  std::vector<std::pair<std::int64_t, std::int64_t>> out;
  for (std::size_t k = 0; k + 1 < flat.size(); k += 2) out.emplace_back(flat[k], flat[k + 1]);
  return out;
}

PadMode pad_mode(const std::string& m) {
  static const std::pair<const char*, PadMode> modes[] = {
      {"constant", PadMode::Constant}, {"edge", PadMode::Edge},
      {"linear_ramp", PadMode::LinearRamp}, {"maximum", PadMode::Maximum},
      {"mean", PadMode::Mean}, {"median", PadMode::Median}, {"minimum", PadMode::Minimum},
      {"reflect", PadMode::Reflect}, {"symmetric", PadMode::Symmetric}, {"wrap", PadMode::Wrap},
      {"empty", PadMode::Empty}};
  for (const auto& [name, mode] : modes) {
    if (m == name) return mode;
  }
  throw_error(ErrorKind::Value, "mode '" + m + "' is not supported");
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
  // pad(a, mode, width[2n], values|null, statLength[2n]|null, odd)
  m.Set("pad", fn(env, "pad", [](Info i, Napi::Env e) {
          PadOptions o;
          o.mode = pad_mode(arg_string(i[1], "mode"));
          o.width = pairs(i[2], "pad_width");
          if (const NDArray* v = opt_array(i[3])) o.values = *v;
          if (!is_nullish(i[4])) o.stat_length = pairs(i[4], "stat_length");
          o.odd = arg_bool(i[5]);
          return wrap(e, pad(arr(i, 0), o));
        }));
  m.Set("insertAlong", fn(env, "insertAlong", [](Info i, Napi::Env e) {
          return wrap(e, insert_along(arr(i, 0), arg_int(i[1], "axis"), arg_ints(i[2], "positions"),
                                      arr(i, 3)));
        }));
  m.Set("deleteAlong", fn(env, "deleteAlong", [](Info i, Napi::Env e) {
          if (!i[2].IsArray()) throw_error(ErrorKind::Value, "keep must be an array");
          const auto js = i[2].As<Napi::Array>();
          std::vector<bool> keep(js.Length());
          for (std::uint32_t k = 0; k < js.Length(); ++k) keep[k] = js.Get(k).ToBoolean().Value();
          return wrap(e, delete_along(arr(i, 0), arg_int(i[1], "axis"), keep));
        }));
  m.Set("trimZeros", fn(env, "trimZeros", [](Info i, Napi::Env e) {
          const auto js = i[3].As<Napi::Array>();
          std::vector<bool> axes(js.Length());
          for (std::uint32_t k = 0; k < js.Length(); ++k) axes[k] = js.Get(k).ToBoolean().Value();
          return wrap(e, trim_zeros(arr(i, 0), arg_bool(i[1]), arg_bool(i[2]), axes));
        }));
  // Number of live native arrays referencing a's buffer (NDArray.resize refcheck, D-093).
  m.Set("bufferRefs", fn(env, "bufferRefs", [](Info i, Napi::Env e) -> Napi::Value {
          return Napi::Number::New(e, static_cast<double>(arr(i, 0).buffer().use_count()));
        }));
  exports.Set("p06", m);
}

}  // namespace nativpy::bindings
