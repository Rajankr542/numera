#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p12_fft.hpp"

namespace nativpy::bindings {

namespace {

namespace p12 = nativpy::fft::p12;

fft::Norm p12_norm(const Napi::Value& v) {
  if (util::is_nullish(v)) return fft::Norm::Backward;
  return fft::parse_norm(util::arg_string(v, "norm"));
}

std::optional<NDArray> p12_out(const Napi::Value& v) {
  if (util::is_nullish(v)) return std::nullopt;
  return NDArrayWrap::unwrap(v);
}

}  // namespace

// Native functions for parity milestone P12 (D-056, D-150..D-152), exposed as `addon.p12`.
void init_p12_binding(Napi::Env env, Napi::Object exports) {
  using namespace util;
  Napi::Object m = Napi::Object::New(env);
  // name(a, n | null, axis, norm | null, out | null)
  const auto one = [&](const char* name, p12::Op op) {
    m.Set(name, fn(env, name, [op](Info i, Napi::Env e) {
            return wrap(e, p12::transform(op, arr(i, 0), opt_int(i[1], "n"), arg_int(i[2], "axis"),
                                          p12_norm(i[3]), p12_out(i[4])));
          }));
  };
  // name(a, s | null, axes | null, norm | null, out | null)
  const auto nd = [&](const char* name, p12::NdOp op) {
    m.Set(name, fn(env, name, [op](Info i, Napi::Env e) {
            return wrap(e, p12::transform_nd(op, arr(i, 0), opt_ints(i[1], "s"),
                                             opt_ints(i[2], "axes"), p12_norm(i[3]),
                                             p12_out(i[4])));
          }));
  };
  one("fft", p12::Op::Fft);
  one("ifft", p12::Op::Ifft);
  one("rfft", p12::Op::Rfft);
  one("irfft", p12::Op::Irfft);
  one("hfft", p12::Op::Hfft);
  one("ihfft", p12::Op::Ihfft);
  nd("fftn", p12::NdOp::Fftn);
  nd("ifftn", p12::NdOp::Ifftn);
  nd("rfftn", p12::NdOp::Rfftn);
  nd("irfftn", p12::NdOp::Irfftn);
  // shift(a, axes | null, inverse)
  m.Set("shift", fn(env, "shift", [](Info i, Napi::Env e) {
          return wrap(e, p12::shift(arr(i, 0), opt_ints(i[1], "axes"), arg_bool(i[2])));
        }));
  exports.Set("p12", m);
}

}  // namespace nativpy::bindings
