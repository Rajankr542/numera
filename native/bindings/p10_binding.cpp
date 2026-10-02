#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p10_conv.hpp"
#include "p10_cumdiff.hpp"
#include "p10_hist.hpp"
#include "p10_nan.hpp"
#include "p10_quantile.hpp"
#include "p10_reduce.hpp"
#include "p10_stats.hpp"

namespace nativpy::bindings {

namespace {

using namespace util;

Napi::Value prop(const Napi::Value& o, const char* key) {
  if (!o.IsObject()) return o.Env().Undefined();
  return o.As<Napi::Object>().Get(key);
}

bool opt_bool(const Napi::Value& o, const char* key) {
  const Napi::Value v = prop(o, key);
  return !is_nullish(v) && arg_bool(v);
}

std::optional<NDArray> opt_arr(const Napi::Value& v) {
  if (is_nullish(v)) return std::nullopt;
  return NDArrayWrap::unwrap(v);
}

std::optional<DType> opt_dtype(const Napi::Value& v) {
  if (is_nullish(v)) return std::nullopt;
  const std::string name = arg_string(v, "dtype");
  const auto dt = dtype_from_name(name);
  if (!dt) throw_error(ErrorKind::DType, "data type '" + name + "' not understood");
  return dt;
}

ReduceOp reduce_op(const std::string& name) {
  if (name == "sum") return ReduceOp::Sum;
  if (name == "prod") return ReduceOp::Prod;
  if (name == "min") return ReduceOp::Min;
  if (name == "max") return ReduceOp::Max;
  if (name == "mean") return ReduceOp::Mean;
  if (name == "var") return ReduceOp::Var;
  if (name == "std") return ReduceOp::Std;
  throw_error(ErrorKind::Value, "unknown reduction '" + name + "'");
}

// {axis: number[] | null, keepdims, dtype, initial, ddof}
ReduceOptions reduce_opts(const Napi::Value& o) {
  ReduceOptions r;
  r.axis = opt_ints(prop(o, "axis"), "axis");
  r.keepdims = opt_bool(o, "keepdims");
  r.dtype = opt_dtype(prop(o, "dtype"));
  const Napi::Value init = prop(o, "initial");
  if (!is_nullish(init)) r.initial = arg_double(init, "initial");
  const Napi::Value ddof = prop(o, "ddof");
  if (!is_nullish(ddof)) r.ddof = arg_int(ddof, "ddof");
  return r;
}

}  // namespace

// Native functions for parity milestone P10 (D-056, D-130..D-136), exposed as `addon.p10`.
void init_p10_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  // quantile(a, q, {axis: number[] | null, keepdims, method, weights, weakQ, percentile, ignoreNan})
  m.Set("quantile", fn(env, "quantile", [](Info i, Napi::Env e) {
          p10::QuantileOptions o;
          o.axis = opt_ints(prop(i[2], "axis"), "axis");
          o.keepdims = opt_bool(i[2], "keepdims");
          const Napi::Value method = prop(i[2], "method");
          if (!is_nullish(method)) o.method = p10::qmethod_from_name(arg_string(method, "method"));
          o.weights = opt_arr(prop(i[2], "weights"));
          o.weak_q = opt_bool(i[2], "weakQ");
          o.percentile = opt_bool(i[2], "percentile");
          o.ignore_nan = opt_bool(i[2], "ignoreNan");
          return wrap(e, p10::quantile(arr(i, 0), arr(i, 1), o));
        }));
  // median(a, axis: number[] | null, keepdims, ignoreNan)
  m.Set("median", fn(env, "median", [](Info i, Napi::Env e) {
          return wrap(e, p10::median(arr(i, 0), opt_ints(i[1], "axis"), arg_bool(i[2]), arg_bool(i[3])));
        }));
  // cumulative(prod, a, {axis, dtype, includeInitial, skipNan, arrayApi}, out | null)
  m.Set("cumulative", fn(env, "cumulative", [](Info i, Napi::Env e) {
          p10::CumulativeOptions o;
          o.axis = opt_int(prop(i[2], "axis"), "axis");
          o.dtype = opt_dtype(prop(i[2], "dtype"));
          o.include_initial = opt_bool(i[2], "includeInitial");
          o.skip_nan = opt_bool(i[2], "skipNan");
          o.array_api = opt_bool(i[2], "arrayApi");
          const std::optional<NDArray> out = opt_arr(i[3]);
          return wrap(e, p10::cumulative(arg_bool(i[0]), arr(i, 1), o, out ? &*out : nullptr));
        }));
  // diff(a, n, axis, prepend | null, append | null)
  m.Set("diff", fn(env, "diff", [](Info i, Napi::Env e) {
          return wrap(e, p10::diff(arr(i, 0), arg_int(i[1], "n"), arg_int(i[2], "axis"), opt_arr(i[3]),
                                   opt_arr(i[4])));
        }));
  // ptp(a, axis: number[] | null, keepdims)
  m.Set("ptp", fn(env, "ptp", [](Info i, Napi::Env e) {
          return wrap(e, p10::ptp(arr(i, 0), opt_ints(i[1], "axis"), arg_bool(i[2])));
        }));
  // nanReduce(op, a, opts)
  m.Set("nanReduce", fn(env, "nanReduce", [](Info i, Napi::Env e) {
          return wrap(e, p10::nan_reduce(reduce_op(arg_string(i[0], "op")), arr(i, 1), reduce_opts(i[2])));
        }));
  // nanArgReduce(isMax, a, axis | null, keepdims)
  m.Set("nanArgReduce", fn(env, "nanArgReduce", [](Info i, Napi::Env e) {
          return wrap(e, p10::nan_arg_reduce(arg_bool(i[0]), arr(i, 1), opt_int(i[2], "axis"), arg_bool(i[3])));
        }));
  // average(a, axis | null, weights | null, keepdims) -> [avg, sumOfWeights]
  m.Set("average", fn(env, "average", [](Info i, Napi::Env e) -> Napi::Value {
          auto [avg, scl] = p10::average(arr(i, 0), opt_ints(i[1], "axis"), opt_arr(i[2]), arg_bool(i[3]));
          return wrap_all(e, {avg, scl});
        }));
  // cov(m, {y, rowvar, bias, ddof, fweights, aweights, dtype}) -> [c, dofWarning]
  m.Set("cov", fn(env, "cov", [](Info i, Napi::Env e) -> Napi::Value {
          p10::CovOptions o;
          o.y = opt_arr(prop(i[1], "y"));
          const Napi::Value rowvar = prop(i[1], "rowvar");
          o.rowvar = is_nullish(rowvar) || arg_bool(rowvar);
          o.bias = opt_bool(i[1], "bias");
          o.ddof = opt_int(prop(i[1], "ddof"), "ddof");
          o.fweights = opt_arr(prop(i[1], "fweights"));
          o.aweights = opt_arr(prop(i[1], "aweights"));
          o.dtype = opt_dtype(prop(i[1], "dtype"));
          bool warn = false;
          NDArray c = opt_bool(i[1], "corrcoef") ? p10::corrcoef(arr(i, 0), o.y, o.rowvar, o.dtype, &warn)
                                                  : p10::cov(arr(i, 0), o, &warn);
          Napi::Array r = Napi::Array::New(e, 2);
          r.Set(0u, wrap(e, std::move(c)));
          r.Set(1u, Napi::Boolean::New(e, warn));
          return r;
        }));
  // gradient(f, spacing: (number | NDArray)[], axis: number[] | null, edgeOrder) -> NDArray[]
  m.Set("gradient", fn(env, "gradient", [](Info i, Napi::Env e) -> Napi::Value {
          std::vector<p10::GradSpacing> sp;
          const Napi::Array js = i[1].As<Napi::Array>();
          for (std::uint32_t k = 0; k < js.Length(); ++k) {
            const Napi::Value v = js.Get(k);
            p10::GradSpacing g;
            if (v.IsNumber()) g.value = arg_double(v, "spacing");
            else g.array = NDArrayWrap::unwrap(v);
            sp.push_back(std::move(g));
          }
          return wrap_all(e, p10::gradient(arr(i, 0), sp, opt_ints(i[2], "axis"), arg_int(i[3], "edge_order")));
        }));
  // trapezoid(y, x | null, dx, axis)
  m.Set("trapezoid", fn(env, "trapezoid", [](Info i, Napi::Env e) {
          return wrap(e, p10::trapezoid(arr(i, 0), opt_arr(i[1]), arg_double(i[2], "dx"), arg_int(i[3], "axis")));
        }));

  // reduceWhere(name, a, {axis, keepdims, dtype, initial, ddof, where, out})
  // where= and out= for np.sum/prod/min/max/mean/var/std (D-136, P10-7).
  m.Set("reduceWhere", fn(env, "reduceWhere", [](Info i, Napi::Env e) {
          const std::string name = arg_string(i[0], "op");
          p10::ReduceWhereOptions opts;
          opts.axis     = opt_ints(prop(i[2], "axis"), "axis");
          opts.keepdims = opt_bool(i[2], "keepdims");
          opts.dtype    = opt_dtype(prop(i[2], "dtype"));
          const Napi::Value init = prop(i[2], "initial");
          if (!is_nullish(init)) opts.initial = arg_double(init, "initial");
          const Napi::Value ddofv = prop(i[2], "ddof");
          if (!is_nullish(ddofv)) opts.ddof = arg_int(ddofv, "ddof");
          const Napi::Value wherev = prop(i[2], "where");
          if (!is_nullish(wherev)) opts.where = NDArrayWrap::unwrap(wherev);
          const Napi::Value outv = prop(i[2], "out");
          if (!is_nullish(outv)) opts.out = NDArrayWrap::unwrap(outv);
          ReduceOp op;
          if      (name == "sum")  op = ReduceOp::Sum;
          else if (name == "prod") op = ReduceOp::Prod;
          else if (name == "min")  op = ReduceOp::Min;
          else if (name == "max")  op = ReduceOp::Max;
          else if (name == "mean") op = ReduceOp::Mean;
          else if (name == "var")  op = ReduceOp::Var;
          else if (name == "std")  op = ReduceOp::Std;
          else throw_error(ErrorKind::Value, "unknown reduction '" + name + "'");
          return wrap(e, p10::reduce_where(op, arr(i, 1), opts));
        }));

  // correlate(a, v, mode) -> NDArray
  m.Set("correlate", fn(env, "correlate", [](Info i, Napi::Env e) {
          return wrap(e, p10::correlate(arr(i, 0), arr(i, 1), arg_string(i[2], "mode")));
        }));
  // convolve(a, v, mode) -> NDArray
  m.Set("convolve", fn(env, "convolve", [](Info i, Napi::Env e) {
          return wrap(e, p10::convolve(arr(i, 0), arr(i, 1), arg_string(i[2], "mode")));
        }));

  // ---- histogram / histogramdd / bincount / digitize / interp (D-134, D-138) ----

  // decode_bins_spec: number | string | NDArray -> HistBins
  auto decode_bins = [](const Napi::Value& v) -> p10::HistBins {
    p10::HistBins b;
    if (v.IsNumber()) { b.count = static_cast<std::int64_t>(v.As<Napi::Number>().DoubleValue()); }
    else if (v.IsString()) { b.estimator = v.As<Napi::String>().Utf8Value(); }
    else { b.edges = NDArrayWrap::unwrap(v); }
    return b;
  };

  // histogram(a, bins: number|string|NDArray, range: [number,number]|null, density, weights|null, edgesOnly)
  //   -> {hist: NDArray, edges: NDArray, warnings: string[]}
  m.Set("histogram", fn(env, "histogram", [decode_bins](Info i, Napi::Env e) {
          const p10::HistBins bins = decode_bins(i[1]);
          p10::Range range;
          if (!is_nullish(i[2])) {
            const Napi::Array r = i[2].As<Napi::Array>();
            range = std::make_pair(arg_double(r.Get(0u), "range[0]"), arg_double(r.Get(1u), "range[1]"));
          }
          std::vector<std::string> warnings;
          auto res = p10::histogram(arr(i, 0), bins, range, arg_bool(i[3]), opt_arr(i[4]), arg_bool(i[5]),
                                    warnings);
          Napi::Object out = Napi::Object::New(e);
          out.Set("hist", wrap(e, std::move(res.hist)));
          out.Set("edges", wrap(e, std::move(res.edges)));
          Napi::Array wa = Napi::Array::New(e, warnings.size());
          for (std::size_t k = 0; k < warnings.size(); ++k)
            wa.Set(static_cast<uint32_t>(k), Napi::String::New(e, warnings[k]));
          out.Set("warnings", wa);
          return out;
        }));

  // histogramdd(cols: NDArray[], bins: spec | spec[], range: ([number,number]|null)[], density, weights|null)
  //   -> {hist: NDArray, edges: NDArray[]}
  m.Set("histogramdd", fn(env, "histogramdd", [decode_bins](Info i, Napi::Env e) {
          // cols
          std::vector<NDArray> cols;
          const Napi::Array jcols = i[0].As<Napi::Array>();
          for (std::uint32_t k = 0; k < jcols.Length(); ++k)
            cols.push_back(NDArrayWrap::unwrap(jcols.Get(k)));
          // bins: single spec or array of specs
          std::vector<p10::HistBins> bins_vec;
          if (i[1].IsArray()) {
            const Napi::Array jb = i[1].As<Napi::Array>();
            for (std::uint32_t k = 0; k < jb.Length(); ++k)
              bins_vec.push_back(decode_bins(jb.Get(k)));
          } else {
            // broadcast single spec to all dims
            const p10::HistBins single = decode_bins(i[1]);
            bins_vec.assign(cols.size(), single);
          }
          // range
          std::vector<p10::Range> range_vec;
          if (!is_nullish(i[2])) {
            const Napi::Array jr = i[2].As<Napi::Array>();
            for (std::uint32_t k = 0; k < jr.Length(); ++k) {
              const Napi::Value rv = jr.Get(k);
              if (is_nullish(rv)) { range_vec.push_back(std::nullopt); }
              else {
                const Napi::Array r2 = rv.As<Napi::Array>();
                range_vec.push_back(std::make_pair(arg_double(r2.Get(0u), "range[0]"),
                                                   arg_double(r2.Get(1u), "range[1]")));
              }
            }
          }
          if (range_vec.size() < cols.size()) range_vec.resize(cols.size());
          auto res = p10::histogramdd(cols, bins_vec, range_vec, arg_bool(i[3]), opt_arr(i[4]));
          Napi::Object out = Napi::Object::New(e);
          out.Set("hist", wrap(e, std::move(res.hist)));
          Napi::Array ea = Napi::Array::New(e, res.edges.size());
          for (std::size_t k = 0; k < res.edges.size(); ++k)
            ea.Set(static_cast<uint32_t>(k), wrap(e, std::move(res.edges[k])));
          out.Set("edges", ea);
          return out;
        }));

  // bincount(x, weights|null, minlength, fromList) -> NDArray
  m.Set("bincount", fn(env, "bincount", [](Info i, Napi::Env e) {
          return wrap(e, p10::bincount(arr(i, 0), opt_arr(i[1]), arg_int(i[2], "minlength"), arg_bool(i[3])));
        }));

  // digitize(x, bins, right) -> NDArray
  m.Set("digitize", fn(env, "digitize", [](Info i, Napi::Env e) {
          return wrap(e, p10::digitize(arr(i, 0), arr(i, 1), arg_bool(i[2])));
        }));

  // interp(x, xp, fp, left: {re,im}|null, right: {re,im}|null, period: number|null) -> NDArray
  m.Set("interp", fn(env, "interp", [](Info i, Napi::Env e) {
          auto opt_complex = [](const Napi::Value& v) -> std::optional<std::complex<double>> {
            if (is_nullish(v)) return std::nullopt;
            if (v.IsObject()) {
              const Napi::Object o = v.As<Napi::Object>();
              return std::complex<double>(arg_double(o.Get("re"), "re"), arg_double(o.Get("im"), "im"));
            }
            return std::complex<double>(arg_double(v, "left/right"), 0.0);
          };
          std::optional<double> period;
          if (!is_nullish(i[5])) period = arg_double(i[5], "period");
          return wrap(e, p10::interp(arr(i, 0), arr(i, 1), arr(i, 2), opt_complex(i[3]), opt_complex(i[4]),
                                     period));
        }));

  exports.Set("p10", m);
}

}  // namespace nativpy::bindings
