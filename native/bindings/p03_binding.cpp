#include <cstring>

#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p03_dtypes.hpp"
#include "p03_iter.hpp"
#include "p03_methods.hpp"
#include "p03_print.hpp"
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

  // ---- P3-3 iteration plan (D-061) ----
  // iterPlan(ops[], order) -> { shape, axes, flipped }
  m.Set("iterPlan", fn(env, "iterPlan", [](Info i, Napi::Env e) -> Napi::Value {
          const IterPlan p = iter_plan(arg_arrays(i[0], "operands"), arg_order(i[1], Order::K));
          Napi::Object out = Napi::Object::New(e);
          out.Set("shape", ints_to_js(e, p.shape));
          Napi::Array axes = Napi::Array::New(e, p.axes.size());
          Napi::Array flipped = Napi::Array::New(e, p.flipped.size());
          for (std::uint32_t k = 0; k < p.axes.size(); ++k) {
            axes.Set(k, Napi::Number::New(e, static_cast<double>(p.axes[k])));
            flipped.Set(k, Napi::Boolean::New(e, p.flipped[k]));
          }
          out.Set("axes", axes);
          out.Set("flipped", flipped);
          return out;
        }));

  // ---- P3-4 dtype introspection (D-062) ----
  m.Set("finfo", fn(env, "finfo", [](Info i, Napi::Env e) -> Napi::Value {
          const FloatInfo f = float_info(parse_dtype(i[0]));
          Napi::Object o = Napi::Object::New(e);
          const auto num = [&](const char* k, double v) { o.Set(k, Napi::Number::New(e, v)); };
          o.Set("dtype", Napi::String::New(e, std::string(dtype_name(f.dtype))));
          num("bits", f.bits);
          num("precision", f.precision);
          num("iexp", f.iexp);
          num("nexp", f.nexp);
          num("nmant", f.nmant);
          num("machep", f.machep);
          num("negep", f.negep);
          num("minexp", f.minexp);
          num("maxexp", f.maxexp);
          num("eps", f.eps);
          num("epsneg", f.epsneg);
          num("max", f.max);
          num("min", f.min);
          num("tiny", f.tiny);
          num("smallestSubnormal", f.smallest_subnormal);
          num("resolution", f.resolution);
          return o;
        }));
  m.Set("iinfo", fn(env, "iinfo", [](Info i, Napi::Env e) -> Napi::Value {
          const IntInfo n = int_info(parse_dtype(i[0]));
          Napi::Object o = Napi::Object::New(e);
          o.Set("bits", Napi::Number::New(e, n.bits));
          o.Set("min", Napi::BigInt::New(e, n.min));
          o.Set("max", Napi::BigInt::New(e, n.max));
          return o;
        }));
  m.Set("minScalarType", fn(env, "minScalarType", [](Info i, Napi::Env e) -> Napi::Value {
          return Napi::String::New(e, std::string(dtype_name(min_scalar_type(arr(i, 0)))));
        }));

  // ---- P3-5 printing (D-063) ----
  m.Set("formatFloat", fn(env, "formatFloat", [](Info i, Napi::Env e) -> Napi::Value {
          const NDArray& a = arr(i, 0);
          const Napi::Object o = i[1].As<Napi::Object>();
          const auto num = [&](const char* k) {
            const Napi::Value v = o.Get(k);
            return v.IsNumber() ? v.As<Napi::Number>().Int32Value() : -1;
          };
          const auto flag = [&](const char* k) { return o.Get(k).ToBoolean().Value(); };
          Dragon4Options d;
          d.scientific = flag("scientific");
          d.unique = flag("unique");
          d.fractional = flag("fractional");
          d.sign = flag("sign");
          d.precision = num("precision");
          d.min_digits = num("minDigits");
          d.pad_left = num("padLeft");
          d.pad_right = num("padRight");
          d.exp_digits = num("expDigits");
          const std::string t = o.Get("trim").As<Napi::String>().Utf8Value();
          d.trim = t == "." ? TrimMode::Zeros : t == "0" ? TrimMode::LeaveOneZero : t == "-" ? TrimMode::DptZeros
                                                                                            : TrimMode::None;
          return Napi::String::New(e, dragon4(a.get_double(0), a.dtype(), d));
        }));
  m.Set("leadingTrailing", fn(env, "leadingTrailing", [](Info i, Napi::Env e) {
          return wrap(e, leading_trailing(arr(i, 0), i[1].As<Napi::Number>().Int64Value()));
        }));
  m.Set("formatElements", fn(env, "formatElements", [](Info i, Napi::Env e) -> Napi::Value {
          const Napi::Object o = i[1].As<Napi::Object>();
          PrintOptions po;
          const Napi::Value prec = o.Get("precision");
          po.precision = prec.IsNumber() ? prec.As<Napi::Number>().Int32Value() : -1;
          po.floatmode = o.Get("floatmode").As<Napi::String>().Utf8Value();
          po.suppress = o.Get("suppress").ToBoolean().Value();
          po.sign = o.Get("sign").As<Napi::String>().Utf8Value().at(0);
          po.nanstr = o.Get("nanstr").As<Napi::String>().Utf8Value();
          po.infstr = o.Get("infstr").As<Napi::String>().Utf8Value();
          const std::vector<std::string> v = format_elements(arr(i, 0), po);
          Napi::Array out = Napi::Array::New(e, v.size());
          for (std::size_t k = 0; k < v.size(); ++k) out.Set(static_cast<std::uint32_t>(k), v[k]);
          return out;
        }));
  m.Set("scalarStr", fn(env, "scalarStr", [](Info i, Napi::Env e) -> Napi::Value {
          return Napi::String::New(e, scalar_str(arr(i, 0)));
        }));

  exports.Set("p03", m);
}

}  // namespace nativpy::bindings
