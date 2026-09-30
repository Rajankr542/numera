#include "linalg_binding.hpp"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "backend.hpp"
#include "error.hpp"
#include "error_binding.hpp"
#include "linalg.hpp"
#include "ndarray_binding.hpp"

namespace nativpy::bindings {

namespace {

using Info = const Napi::CallbackInfo&;

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

const NDArray& arr(Info info, std::size_t i) { return NDArrayWrap::unwrap(info[i]); }

Napi::Value wrap(Napi::Env e, const NDArray& a) { return NDArrayWrap::create(e, a); }

Napi::Value wrap_opt(Napi::Env e, const std::optional<NDArray>& a) {
  return a ? wrap(e, *a) : e.Null();
}

std::int64_t to_int(const Napi::Value& v, const char* what) {
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  const double d = v.As<Napi::Number>().DoubleValue();
  if (!std::isfinite(d) || std::trunc(d) != d) {
    throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  }
  return static_cast<std::int64_t>(d);
}

using Binary = NDArray (*)(const NDArray&, const NDArray&);
using Unary = NDArray (*)(const NDArray&);

}  // namespace

void init_linalg_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object l = Napi::Object::New(env);
  const auto bin = [&](const char* name, Binary f) {
    l.Set(name, fn(env, name, [f](Info i, Napi::Env e) { return wrap(e, f(arr(i, 0), arr(i, 1))); }));
  };
  const auto un = [&](const char* name, Unary f) {
    l.Set(name, fn(env, name, [f](Info i, Napi::Env e) { return wrap(e, f(arr(i, 0))); }));
  };
  bin("matmul", &linalg::matmul);
  bin("dot", &linalg::dot);
  bin("inner", &linalg::inner);
  bin("outer", &linalg::outer);
  bin("solve", &linalg::solve);
  un("det", &linalg::det);
  un("inv", &linalg::inv);
  const auto eig_obj = [](Napi::Env e, const linalg::EigResult& r) {
    Napi::Object o = Napi::Object::New(e);
    o.Set("eigenvalues", wrap(e, r.eigenvalues));
    o.Set("eigenvectors", wrap(e, r.eigenvectors));
    return o;
  };
  l.Set("eig", fn(env, "eig", [eig_obj](Info i, Napi::Env e) { return eig_obj(e, linalg::eig(arr(i, 0))); }));
  l.Set("eigh", fn(env, "eigh", [eig_obj](Info i, Napi::Env e) { return eig_obj(e, linalg::eigh(arr(i, 0))); }));
  // svd(a, fullMatrices, computeUV) -> {U|null, S, Vh|null}
  l.Set("svd", fn(env, "svd", [](Info i, Napi::Env e) {
          const auto r = linalg::svd(arr(i, 0), i[1].ToBoolean().Value(), i[2].ToBoolean().Value());
          Napi::Object o = Napi::Object::New(e);
          o.Set("U", wrap_opt(e, r.u));
          o.Set("S", wrap(e, r.s));
          o.Set("Vh", wrap_opt(e, r.vh));
          return o;
        }));
  // qr(a, mode: 'reduced'|'complete'|'r') -> {Q|null, R}
  l.Set("qr", fn(env, "qr", [](Info i, Napi::Env e) {
          const std::string mode = i[1].ToString().Utf8Value();
          linalg::QrMode m = linalg::QrMode::Reduced;
          if (mode == "complete") m = linalg::QrMode::Complete;
          else if (mode == "r") m = linalg::QrMode::R;
          else if (mode != "reduced") throw_error(ErrorKind::Value, "Unrecognized mode '" + mode + "'");
          const auto r = linalg::qr(arr(i, 0), m);
          Napi::Object o = Napi::Object::New(e);
          o.Set("Q", wrap_opt(e, r.q));
          o.Set("R", wrap(e, r.r));
          return o;
        }));
  // lstsq(a, b, rcond | null) -> {x, residuals, rank, s}
  l.Set("lstsq", fn(env, "lstsq", [](Info i, Napi::Env e) {
          double rcond = -1.0;
          if (!i[2].IsUndefined() && !i[2].IsNull()) {
            if (!i[2].IsNumber()) throw_error(ErrorKind::Value, "rcond must be a number");
            rcond = i[2].As<Napi::Number>().DoubleValue();
            if (rcond < 0) rcond = -1.0;
          }
          const auto r = linalg::lstsq(arr(i, 0), arr(i, 1), rcond);
          Napi::Object o = Napi::Object::New(e);
          o.Set("x", wrap(e, r.x));
          o.Set("residuals", wrap(e, r.residuals));
          o.Set("rank", Napi::Number::New(e, static_cast<double>(r.rank)));
          o.Set("s", wrap(e, r.s));
          return o;
        }));
  // norm(a, ord: null|'fro'|'nuc'|number, axis: null|number[], keepdims)
  l.Set("norm", fn(env, "norm", [](Info i, Napi::Env e) {
          linalg::NormOrd ord;
          if (i[1].IsString()) {
            ord.kind = i[1].As<Napi::String>().Utf8Value();
            if (ord.kind != "fro" && ord.kind != "nuc") {
              throw_error(ErrorKind::Value, "Invalid norm order '" + ord.kind + "'");
            }
          } else if (i[1].IsNumber()) {
            ord.kind = "p";
            ord.p = i[1].As<Napi::Number>().DoubleValue();
          } else if (!i[1].IsUndefined() && !i[1].IsNull()) {
            throw_error(ErrorKind::Value, "ord must be null, 'fro', 'nuc' or a number");
          }
          std::optional<std::vector<std::int64_t>> axis;
          if (i[2].IsArray()) {
            const auto a = i[2].As<Napi::Array>();
            axis.emplace();
            for (std::uint32_t k = 0; k < a.Length(); ++k) axis->push_back(to_int(a.Get(k), "axis"));
          } else if (!i[2].IsUndefined() && !i[2].IsNull()) {
            throw_error(ErrorKind::Value, "axis must be an array of integers");
          }
          return wrap(e, linalg::norm(arr(i, 0), ord, axis, i[3].ToBoolean().Value()));
        }));
  l.Set("backend", fn(env, "backend", [](Info, Napi::Env e) -> Napi::Value {
          return Napi::String::New(e, std::string(linalg::active_backend().name()));
        }));
  // Internal/testing: "fallback" forces the portable backend, "default" restores.
  l.Set("_setBackend", fn(env, "_setBackend", [](Info i, Napi::Env e) -> Napi::Value {
          const std::string which = i[0].ToString().Utf8Value();
          if (which == "fallback") linalg::set_active_backend(&linalg::fallback_backend());
          else if (which == "default") linalg::set_active_backend(nullptr);
          else throw_error(ErrorKind::Value, "unknown backend '" + which + "'");
          return e.Undefined();
        }));
  exports.Set("linalg", l);
}

}  // namespace nativpy::bindings
