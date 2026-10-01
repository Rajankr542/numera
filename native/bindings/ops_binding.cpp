#include "ops_binding.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "broadcast.hpp"
#include "creation.hpp"
#include "dtype_binding.hpp"
#include "error.hpp"
#include "error_binding.hpp"
#include "indexing.hpp"
#include "ndarray_binding.hpp"
#include "reduce.hpp"
#include "shape_ops.hpp"
#include "ufunc.hpp"
#include "ufunc_methods.hpp"
#include "ufunc_registry.hpp"

namespace nativpy::bindings {

namespace {

std::int64_t arg_int(const Napi::Value& v, const char* what) {
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  const double d = v.As<Napi::Number>().DoubleValue();
  if (!std::isfinite(d) || std::trunc(d) != d || std::fabs(d) > 9007199254740991.0) {
    throw_error(ErrorKind::Value, std::string(what) + " must be a safe integer");
  }
  return static_cast<std::int64_t>(d);
}

double arg_double(const Napi::Value& v, const char* what) {
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be a number");
  return v.As<Napi::Number>().DoubleValue();
}

std::vector<std::int64_t> arg_ints(const Napi::Value& v, const char* what) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, std::string(what) + " must be an array");
  const auto arr = v.As<Napi::Array>();
  std::vector<std::int64_t> out(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) out[i] = arg_int(arr.Get(i), what);
  return out;
}

template <typename Fn>
Napi::Function fn(Napi::Env env, const char* name, Fn body) {
  return Napi::Function::New(
      env,
      [body](const Napi::CallbackInfo& info) -> Napi::Value {
        Napi::Env e = info.Env();
        return translate_errors(e, [&]() -> Napi::Value { return body(info, e); });
      },
      name);
}

const NDArray& arr(const Napi::CallbackInfo& info, std::size_t i) {
  return NDArrayWrap::unwrap(info[i]);
}

// Ufunc params (D-048) from an optional { dtype?: string, casting?: string }.
UfuncParams ufunc_params(const Napi::CallbackInfo& info, std::size_t i) {
  UfuncParams p;
  if (info.Length() <= i || !info[i].IsObject()) return p;
  const auto o = info[i].As<Napi::Object>();
  const Napi::Value dt = o.Get("dtype");
  if (!dt.IsUndefined() && !dt.IsNull()) p.dtype = parse_dtype(dt);
  const Napi::Value casting = o.Get("casting");
  if (!casting.IsUndefined()) p.casting = parse_casting(casting);
  // where= (D-049): a native bool array; TS normalises lists/scalars.
  const Napi::Value where = o.Get("where");
  if (!where.IsUndefined() && !where.IsNull()) p.where = NDArrayWrap::unwrap(where);
  // order= (D-050): validated natively, like NumPy's PyArray_OrderConverter.
  const Napi::Value order = o.Get("order");
  if (!order.IsUndefined() && !order.IsNull()) {
    if (!order.IsString()) throw_error(ErrorKind::DType, "order must be str, not a non-string value");
    const std::string name = order.As<Napi::String>().Utf8Value();
    const auto ord = order_from_name(name);
    if (!ord) {
      throw_error(ErrorKind::Value,
                  "order must be one of 'C', 'F', 'A', or 'K' (got '" + name + "')");
    }
    p.order = *ord;
  }
  return p;
}

std::optional<std::int64_t> opt_int(const Napi::Object& o, const char* key) {
  const Napi::Value v = o.Get(key);
  if (v.IsUndefined() || v.IsNull()) return std::nullopt;
  return arg_int(v, key);
}

// Decodes the JS index encoding produced by packages/numera/src/indexing.ts.
std::vector<IndexItem> arg_index(const Napi::Value& v) {
  if (!v.IsArray()) throw_error(ErrorKind::Index, "index must be an array");
  const auto a = v.As<Napi::Array>();
  std::vector<IndexItem> out;
  out.reserve(a.Length());
  for (std::uint32_t k = 0; k < a.Length(); ++k) {
    const Napi::Value it = a.Get(k);
    if (it.IsNull()) {
      out.push_back(IndexItem::new_axis());
    } else if (it.IsNumber()) {
      out.push_back(IndexItem::integer_(arg_int(it, "index")));
    } else if (it.IsBoolean()) {
      NDArray b = NDArray::empty({}, DType::Bool);
      b.set_int64(0, it.As<Napi::Boolean>().Value() ? 1 : 0);
      out.push_back(IndexItem::array_(std::move(b)));
    } else if (it.IsString() && it.As<Napi::String>().Utf8Value() == "...") {
      out.push_back(IndexItem::ellipsis());
    } else if (NDArrayWrap::is_ndarray(it)) {
      out.push_back(IndexItem::array_(NDArrayWrap::unwrap(it)));
    } else if (it.IsObject()) {
      const auto o = it.As<Napi::Object>();
      out.push_back(IndexItem::slice(opt_int(o, "start"), opt_int(o, "stop"), opt_int(o, "step")));
    } else {
      throw_error(ErrorKind::Index,
                  "only integers, slices, newaxis, ellipsis, and integer or boolean arrays are "
                  "valid indices");
    }
  }
  return out;
}

}  // namespace

void init_ops_binding(Napi::Env env, Napi::Object exports) {
  using Info = const Napi::CallbackInfo&;
  // ---- creation (M2) ----
  exports.Set("full", fn(env, "full", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, full(arg_ints(i[0], "shape"), arr(i, 1)));
              }));
  exports.Set("ones", fn(env, "ones", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, ones(arg_ints(i[0], "shape"), parse_dtype(i[1])));
              }));
  exports.Set("arange", fn(env, "arange", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(
                    e, arange(arg_double(i[0], "start"), arg_double(i[1], "stop"),
                              arg_double(i[2], "step"), parse_dtype(i[3])));
              }));
  exports.Set("linspace", fn(env, "linspace", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(
                    e, linspace(arg_double(i[0], "start"), arg_double(i[1], "stop"),
                                arg_int(i[2], "num"), i[3].ToBoolean().Value(), parse_dtype(i[4])));
              }));
  exports.Set("eye", fn(env, "eye", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, eye(arg_int(i[0], "N"), arg_int(i[1], "M"),
                                                  arg_int(i[2], "k"), parse_dtype(i[3])));
              }));
  // ---- shape (M3) ----
  exports.Set("transpose", fn(env, "transpose", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, transpose(arr(i, 0), arg_ints(i[1], "axes")));
              }));
  exports.Set("squeeze", fn(env, "squeeze", [](Info i, Napi::Env e) {
                std::optional<std::vector<std::int64_t>> axes;
                if (!i[1].IsUndefined() && !i[1].IsNull()) axes = arg_ints(i[1], "axis");
                return NDArrayWrap::create(e, squeeze(arr(i, 0), axes));
              }));
  exports.Set("expandDims", fn(env, "expandDims", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, expand_dims(arr(i, 0), arg_ints(i[1], "axis")));
              }));
  exports.Set("swapaxes", fn(env, "swapaxes", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(
                    e, swapaxes(arr(i, 0), arg_int(i[1], "axis1"), arg_int(i[2], "axis2")));
              }));
  exports.Set("moveaxis", fn(env, "moveaxis", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, moveaxis(arr(i, 0), arg_ints(i[1], "source"),
                                                       arg_ints(i[2], "destination")));
              }));
  exports.Set("ravel", fn(env, "ravel", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, ravel(arr(i, 0)));
              }));
  exports.Set("flatten", fn(env, "flatten", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, flatten(arr(i, 0)));
              }));
  // ---- broadcasting (M5) / ufuncs (M4) ----
  exports.Set("broadcastShapes", fn(env, "broadcastShapes", [](Info i, Napi::Env e) {
                if (!i[0].IsArray()) throw_error(ErrorKind::Value, "shapes must be an array");
                const auto list = i[0].As<Napi::Array>();
                std::vector<Shape> shapes;
                for (std::uint32_t k = 0; k < list.Length(); ++k) {
                  shapes.push_back(arg_ints(list.Get(k), "shape"));
                }
                const Shape out = broadcast_shapes(shapes);
                auto res = Napi::Array::New(e, out.size());
                for (std::uint32_t k = 0; k < out.size(); ++k) {
                  res.Set(k, Napi::Number::New(e, static_cast<double>(out[k])));
                }
                return res;
              }));
  exports.Set("broadcastTo", fn(env, "broadcastTo", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, broadcast_to(arr(i, 0), arg_ints(i[1], "shape")));
              }));
  exports.Set("binary", fn(env, "binary", [](Info i, Napi::Env e) {
                const std::string name = i[0].ToString().Utf8Value();
                const Ufunc* u = find_ufunc(name);
                if (!u || u->nin != 2) throw_error(ErrorKind::Value, "unknown binary ufunc " + name);
                const UfuncParams params = ufunc_params(i, 4);
                // Optional out (D-046/D-047): the caller returns its own JS object.
                if (i.Length() > 3 && !i[3].IsUndefined() && !i[3].IsNull()) {
                  const NDArray& out = arr(i, 3);
                  binary(*u, arr(i, 1), arr(i, 2), &out, params);
                  return e.Undefined();
                }
                return Napi::Value(NDArrayWrap::create(e, binary(*u, arr(i, 1), arr(i, 2), nullptr, params)));
              }));
  exports.Set("unary", fn(env, "unary", [](Info i, Napi::Env e) {
                const std::string name = i[0].ToString().Utf8Value();
                const Ufunc* u = find_ufunc(name);
                if (!u || u->nin != 1) throw_error(ErrorKind::Value, "unknown unary ufunc " + name);
                const UfuncParams params = ufunc_params(i, 3);
                if (i.Length() > 2 && !i[2].IsUndefined() && !i[2].IsNull()) {
                  const NDArray& out = arr(i, 2);
                  unary(*u, arr(i, 1), &out, params);
                  return e.Undefined();
                }
                return Napi::Value(NDArrayWrap::create(e, unary(*u, arr(i, 1), nullptr, params)));
              }));
  // ufuncMethod(method, name, a, out | null, {axis?: number[] | null, dtype?,
  // keepdims?, initial?, where?}) for "reduce" / "accumulate" (D-052).
  // axis: undefined = default, null = all axes.
  exports.Set("ufuncMethod", fn(env, "ufuncMethod", [](Info i, Napi::Env e) {
                const std::string method = i[0].ToString().Utf8Value();
                const std::string name = i[1].ToString().Utf8Value();
                const Ufunc* u = find_ufunc(name);
                if (!u) throw_error(ErrorKind::Value, "unknown ufunc " + name);
                UfuncReduceOptions opts;
                if (i.Length() > 4 && i[4].IsObject()) {
                  const auto o = i[4].As<Napi::Object>();
                  if (o.Has("axis")) {
                    const Napi::Value axis = o.Get("axis");
                    if (axis.IsNull()) opts.all_axes = true;
                    else if (!axis.IsUndefined()) opts.axis = arg_ints(axis, "axis");
                  }
                  const Napi::Value dt = o.Get("dtype");
                  if (!dt.IsUndefined() && !dt.IsNull()) opts.dtype = parse_dtype(dt);
                  opts.keepdims = o.Get("keepdims").ToBoolean().Value();
                  const Napi::Value init = o.Get("initial");
                  if (!init.IsUndefined() && !init.IsNull()) opts.initial = arg_double(init, "initial");
                  const Napi::Value where = o.Get("where");
                  if (!where.IsUndefined() && !where.IsNull()) opts.where = NDArrayWrap::unwrap(where);
                }
                const bool has_out = i.Length() > 3 && !i[3].IsUndefined() && !i[3].IsNull();
                const NDArray* out = has_out ? &arr(i, 3) : nullptr;
                NDArray r = method == "reduce"       ? ufunc_reduce(*u, arr(i, 2), out, opts)
                            : method == "accumulate" ? ufunc_accumulate(*u, arr(i, 2), out, opts)
                                                     : (throw_error(ErrorKind::Value, "unknown ufunc method " + method), arr(i, 2));
                if (has_out) return e.Undefined();
                return Napi::Value(NDArrayWrap::create(e, r));
              }));
  // ---- indexing (M6) ----
  exports.Set("complexPart", fn(env, "complexPart", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, complex_part(arr(i, 0), i[1].ToBoolean().Value()));
              }));
  exports.Set("isComplexElementwise", fn(env, "isComplexElementwise", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(
                    e, is_complex_elementwise(arr(i, 0), i[1].ToBoolean().Value()));
              }));
  exports.Set("getIndex", fn(env, "getIndex", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, get_index(arr(i, 0), arg_index(i[1])));
              }));
  exports.Set("setIndex", fn(env, "setIndex", [](Info i, Napi::Env e) {
                NDArray target = arr(i, 0);  // shares the buffer
                set_index(target, arg_index(i[1]), arr(i, 2));
                return e.Undefined();
              }));
  exports.Set("nonzero", fn(env, "nonzero", [](Info i, Napi::Env e) {
                const auto out = nonzero(arr(i, 0));
                Napi::Array res = Napi::Array::New(e, out.size());
                for (std::uint32_t k = 0; k < out.size(); ++k) res.Set(k, NDArrayWrap::create(e, out[k]));
                return res;
              }));
  exports.Set("take", fn(env, "take", [](Info i, Napi::Env e) {
                std::optional<std::int64_t> axis;
                if (!i[2].IsUndefined() && !i[2].IsNull()) axis = arg_int(i[2], "axis");
                return NDArrayWrap::create(e, take(arr(i, 0), arr(i, 1), axis));
              }));
  exports.Set("where", fn(env, "where", [](Info i, Napi::Env e) {
                return NDArrayWrap::create(e, where(arr(i, 0), arr(i, 1), arr(i, 2)));
              }));
  // ---- reductions (M7, D-017) ----
  // reduce(name, a, {axis?: number[] | null, keepdims, dtype?, initial?, ddof})
  exports.Set("reduce", fn(env, "reduce", [](Info i, Napi::Env e) {
                const std::string name = i[0].ToString().Utf8Value();
                std::optional<ReduceOp> op;
                if (name == "sum") op = ReduceOp::Sum;
                else if (name == "prod") op = ReduceOp::Prod;
                else if (name == "min") op = ReduceOp::Min;
                else if (name == "max") op = ReduceOp::Max;
                else if (name == "mean") op = ReduceOp::Mean;
                else if (name == "var") op = ReduceOp::Var;
                else if (name == "std") op = ReduceOp::Std;
                if (!op) throw_error(ErrorKind::Value, "unknown reduction " + name);
                const auto o = i[2].As<Napi::Object>();
                ReduceOptions opts;
                const Napi::Value axis = o.Get("axis");
                if (!axis.IsUndefined() && !axis.IsNull()) opts.axis = arg_ints(axis, "axis");
                opts.keepdims = o.Get("keepdims").ToBoolean().Value();
                const Napi::Value dt = o.Get("dtype");
                if (!dt.IsUndefined() && !dt.IsNull()) opts.dtype = parse_dtype(dt);
                const Napi::Value init = o.Get("initial");
                if (!init.IsUndefined() && !init.IsNull()) opts.initial = arg_double(init, "initial");
                opts.ddof = opt_int(o, "ddof").value_or(0);
                return NDArrayWrap::create(e, reduce(*op, arr(i, 1), opts));
              }));
  // argReduce(isMax, a, axis | null, keepdims)
  exports.Set("argReduce", fn(env, "argReduce", [](Info i, Napi::Env e) {
                std::optional<std::int64_t> axis;
                if (!i[2].IsUndefined() && !i[2].IsNull()) axis = arg_int(i[2], "axis");
                return NDArrayWrap::create(
                    e, arg_reduce(i[0].ToBoolean().Value(), arr(i, 1), axis, i[3].ToBoolean().Value()));
              }));
}

}  // namespace nativpy::bindings
