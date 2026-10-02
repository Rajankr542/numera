#include "binding_utils.hpp"
#include "milestone_bindings.hpp"
#include "p09_sets.hpp"
#include "p09_sorting.hpp"

namespace nativpy::bindings {

namespace {

using namespace util;

// TS passes "quick" | "stable" after validating `kind`/`stable` (D-120).
SortKind arg_kind(const Napi::Value& v) {
  return arg_string(v, "kind") == "stable" ? SortKind::Stable : SortKind::Quick;
}

Napi::Value opt_wrap(Napi::Env env, const std::optional<NDArray>& a) {
  return a ? wrap(env, *a) : env.Undefined();
}

Napi::Object unique_obj(Napi::Env env, const UniqueResult& r) {
  Napi::Object o = Napi::Object::New(env);
  o.Set("values", wrap(env, r.values));
  if (r.indices) o.Set("indices", wrap(env, *r.indices));
  if (r.inverse) o.Set("inverse", wrap(env, *r.inverse));
  if (r.counts) o.Set("counts", wrap(env, *r.counts));
  return o;
}

std::optional<NDArray> opt_arr(const Napi::Value& v) {
  if (is_nullish(v)) return std::nullopt;
  return NDArrayWrap::unwrap(v);
}

}  // namespace

// Native functions for parity milestone P9 (D-056), exposed as `addon.p09`.
void init_p09_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  // sort(a, axis|null, kind, descending) -> sorted copy
  m.Set("sort", fn(env, "sort", [](Info i, Napi::Env e) {
          return wrap(e, sort_copy(arr(i, 0), opt_int(i[1], "axis"), arg_kind(i[2]), arg_bool(i[3])));
        }));
  // sortInPlace(a, axis, kind, descending)
  m.Set("sortInPlace", fn(env, "sortInPlace", [](Info i, Napi::Env e) {
          sort_inplace(arr(i, 0), arg_int(i[1], "axis"), arg_kind(i[2]), arg_bool(i[3]));
          return e.Undefined();
        }));
  m.Set("argsort", fn(env, "argsort", [](Info i, Napi::Env e) {
          return wrap(e, argsort(arr(i, 0), opt_int(i[1], "axis"), arg_kind(i[2]), arg_bool(i[3])));
        }));
  // partition(a, kth[], axis|null)
  m.Set("partition", fn(env, "partition", [](Info i, Napi::Env e) {
          return wrap(e, partition_copy(arr(i, 0), arg_ints(i[1], "kth"), opt_int(i[2], "axis")));
        }));
  m.Set("partitionInPlace", fn(env, "partitionInPlace", [](Info i, Napi::Env e) {
          partition_inplace(arr(i, 0), arg_ints(i[1], "kth"), arg_int(i[2], "axis"));
          return e.Undefined();
        }));
  m.Set("argpartition", fn(env, "argpartition", [](Info i, Napi::Env e) {
          return wrap(e, argpartition(arr(i, 0), arg_ints(i[1], "kth"), opt_int(i[2], "axis")));
        }));
  // lexsort(keys[], axis)
  m.Set("lexsort", fn(env, "lexsort", [](Info i, Napi::Env e) {
          return wrap(e, lexsort(arg_arrays(i[0], "keys"), arg_int(i[1], "axis")));
        }));
  // searchsorted(a, v, right, sorter|null)
  m.Set("searchsorted", fn(env, "searchsorted", [](Info i, Napi::Env e) {
          return wrap(e, searchsorted(arr(i, 0), arr(i, 1), arg_bool(i[2]), opt_arr(i[3])));
        }));
  // unique1d(a, returnIndex, returnInverse, returnCounts, equalNan) -> {values, indices?, inverse?, counts?}
  m.Set("unique1d", fn(env, "unique1d", [](Info i, Napi::Env e) {
          return unique_obj(e, unique1d(arr(i, 0), arg_bool(i[1]), arg_bool(i[2]), arg_bool(i[3]),
                                        arg_bool(i[4])));
        }));
  // uniqueRows(a2d, returnIndex, returnInverse, returnCounts)
  m.Set("uniqueRows", fn(env, "uniqueRows", [](Info i, Napi::Env e) {
          return unique_obj(e, unique_rows(arr(i, 0), arg_bool(i[1]), arg_bool(i[2]), arg_bool(i[3])));
        }));
  m.Set("isin", fn(env, "isin", [](Info i, Napi::Env e) {
          return wrap(e, isin(arr(i, 0), arr(i, 1), arg_bool(i[2])));
        }));
  // intersect1d(a, b, assumeUnique, returnIndices) -> {values, indices1?, indices2?}
  m.Set("intersect1d", fn(env, "intersect1d", [](Info i, Napi::Env e) {
          const auto r = intersect1d(arr(i, 0), arr(i, 1), arg_bool(i[2]), arg_bool(i[3]));
          Napi::Object o = Napi::Object::New(e);
          o.Set("values", wrap(e, r.values));
          o.Set("indices1", opt_wrap(e, r.indices1));
          o.Set("indices2", opt_wrap(e, r.indices2));
          return o;
        }));
  m.Set("union1d", fn(env, "union1d", [](Info i, Napi::Env e) { return wrap(e, union1d(arr(i, 0), arr(i, 1))); }));
  m.Set("setxor1d", fn(env, "setxor1d", [](Info i, Napi::Env e) {
          return wrap(e, setxor1d(arr(i, 0), arr(i, 1), arg_bool(i[2])));
        }));
  m.Set("setdiff1d", fn(env, "setdiff1d", [](Info i, Napi::Env e) {
          return wrap(e, setdiff1d(arr(i, 0), arr(i, 1), arg_bool(i[2])));
        }));
  // ediff1d(a, toBegin|null, toEnd|null)
  m.Set("ediff1d", fn(env, "ediff1d", [](Info i, Napi::Env e) {
          return wrap(e, ediff1d(arr(i, 0), opt_arr(i[1]), opt_arr(i[2])));
        }));
  exports.Set("p09", m);
}

}  // namespace nativpy::bindings
