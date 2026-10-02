#include <complex>

#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p07_creation.hpp"

namespace nativpy::bindings {

namespace {
std::complex<double> arg_complex(const Napi::Value& re, const Napi::Value& im) {
  return {util::arg_double(re, "re"), util::arg_double(im, "im")};
}
std::optional<DType> opt_dtype(const Napi::Value& v) {
  if (util::is_nullish(v)) return std::nullopt;
  return parse_dtype(v);
}
}  // namespace

// Native functions for parity milestone P7 (D-056, D-100, D-101), exposed as `addon.p07`.
void init_p07_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  // logspace(sre, sim, ere, eim, isComplex, num, endpoint, base, dtype|null)
  m.Set("logspace", fn(env, "logspace", [](Info i, Napi::Env e) {
          return wrap(e, p07::logspace(arg_complex(i[0], i[1]), arg_complex(i[2], i[3]), arg_bool(i[4]),
                                       arg_int(i[5], "num"), arg_bool(i[6]), arg_double(i[7], "base"),
                                       opt_dtype(i[8])));
        }));
  // geomspace(sre, sim, ere, eim, isComplex, num, endpoint, dtype)
  m.Set("geomspace", fn(env, "geomspace", [](Info i, Napi::Env e) {
          return wrap(e, p07::geomspace(arg_complex(i[0], i[1]), arg_complex(i[2], i[3]), arg_int(i[5], "num"),
                                        arg_bool(i[6]), arg_bool(i[4]), parse_dtype(i[7])));
        }));
  m.Set("tri", fn(env, "tri", [](Info i, Napi::Env e) {
          return wrap(e, p07::tri(arg_int(i[0], "N"), arg_int(i[1], "M"), arg_int(i[2], "k"), parse_dtype(i[3])));
        }));
  m.Set("tril", fn(env, "tril", [](Info i, Napi::Env e) { return wrap(e, p07::tril(arr(i, 0), arg_int(i[1], "k"))); }));
  m.Set("triu", fn(env, "triu", [](Info i, Napi::Env e) { return wrap(e, p07::triu(arr(i, 0), arg_int(i[1], "k"))); }));
  m.Set("diag", fn(env, "diag", [](Info i, Napi::Env e) { return wrap(e, p07::diag(arr(i, 0), arg_int(i[1], "k"))); }));
  m.Set("vander", fn(env, "vander", [](Info i, Napi::Env e) {
          return wrap(e, p07::vander(arr(i, 0), opt_int(i[1], "N"), arg_bool(i[2])));
        }));
  m.Set("indices", fn(env, "indices", [](Info i, Napi::Env e) {
          return wrap(e, p07::indices(arg_ints(i[0], "dimensions"), parse_dtype(i[1])));
        }));
  m.Set("gridAxis", fn(env, "gridAxis", [](Info i, Napi::Env e) {
          return wrap(e, p07::grid_axis(arg_double(i[0], "start"), arg_double(i[1], "step"), arg_int(i[2], "n"),
                                        parse_dtype(i[3])));
        }));
  m.Set("mgrid", fn(env, "mgrid", [](Info i, Napi::Env e) {
          return wrap(e, p07::mgrid(arg_doubles(i[0], "starts"), arg_doubles(i[1], "steps"), arg_ints(i[2], "sizes"),
                                    parse_dtype(i[3])));
        }));
  m.Set("triIndices", fn(env, "triIndices", [](Info i, Napi::Env e) {
          auto [r, c] = p07::tri_indices(arg_int(i[0], "n"), arg_int(i[1], "m"), arg_int(i[2], "k"), arg_bool(i[3]));
          return wrap_all(e, {r, c});
        }));
  m.Set("fillDiagonal", fn(env, "fillDiagonal", [](Info i, Napi::Env e) -> Napi::Value {
          p07::fill_diagonal(arr(i, 0), arr(i, 1), arg_bool(i[2]));
          return e.Undefined();
        }));
  m.Set("concatenate", fn(env, "concatenate", [](Info i, Napi::Env e) {
          return wrap(e, p07::concatenate(arg_arrays(i[0], "arrays"), arg_int(i[1], "axis")));
        }));
  m.Set("fromstring", fn(env, "fromstring", [](Info i, Napi::Env e) {
          return wrap(e, p07::fromstring(arg_string(i[0], "string"), parse_dtype(i[1]), arg_int(i[2], "count"),
                                         arg_string(i[3], "sep")));
        }));
  exports.Set("p07", m);
}

}  // namespace nativpy::bindings
