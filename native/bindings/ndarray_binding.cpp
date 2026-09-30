#include "ndarray_binding.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cast.hpp"
#include "dtype_binding.hpp"
#include "error.hpp"
#include "error_binding.hpp"

namespace nativpy::bindings {

namespace {


// ---- JS <-> C++ helpers ---------------------------------------------------

std::int64_t to_int64(const Napi::Value& v, const char* what) {
  if (v.IsBigInt()) {
    bool lossless = false;
    const std::int64_t r = v.As<Napi::BigInt>().Int64Value(&lossless);
    if (!lossless) throw_error(ErrorKind::Value, std::string(what) + " out of int64 range");
    return r;
  }
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  const double d = v.As<Napi::Number>().DoubleValue();
  if (!std::isfinite(d) || std::trunc(d) != d || std::fabs(d) > 9007199254740991.0) {
    throw_error(ErrorKind::Value, std::string(what) + " must be a safe integer");
  }
  return static_cast<std::int64_t>(d);
}

std::vector<std::int64_t> to_int64_vector(const Napi::Value& v, const char* what) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, std::string(what) + " must be an array");
  const auto arr = v.As<Napi::Array>();
  std::vector<std::int64_t> out(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) out[i] = to_int64(arr.Get(i), what);
  return out;
}

Napi::Array int64_vector_to_js(Napi::Env env, const std::vector<std::int64_t>& v) {
  Napi::Array out = Napi::Array::New(env, v.size());
  for (std::uint32_t i = 0; i < v.size(); ++i) {
    out.Set(i, Napi::Number::New(env, static_cast<double>(v[i])));
  }
  return out;
}

// Stores one JS number (as double) into an element of dtype T following
// D-009. Shared by the per-element and bulk (fromFloat64) conversion paths.
template <typename T>
inline void store_number(std::byte* p, double d) {
  if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
    if (std::isnan(d)) throw_error(ErrorKind::Value, "cannot convert float NaN to integer");
    if (std::isinf(d)) throw_error(ErrorKind::Value, "cannot convert float infinity to integer");
    const double t = std::trunc(d);
    // Exact range checks in double: min is a power of two (exact); max+1 too.
    const double lo = static_cast<double>(std::numeric_limits<T>::min());
    const double hi_excl = static_cast<double>(std::numeric_limits<T>::max() / 2 + 1) * 2.0;
    if (t < lo || t >= hi_excl) {
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.17g", t);
      throw_error(ErrorKind::Value, std::string("integer ") + buf + " out of bounds for target dtype");
    }
    store<T>(p, static_cast<T>(t));
  } else {
    store<T>(p, cast_value<T>(d));
  }
}

// A JS complex value is any object with a numeric `re` field (D-033).
bool is_js_complex(const Napi::Value& v) {
  if (!v.IsObject() || v.IsArray()) return false;
  return v.As<Napi::Object>().Get("re").IsNumber();
}

double js_real_part(const Napi::Value& v) {
  if (v.IsBoolean()) return v.As<Napi::Boolean>().Value() ? 1.0 : 0.0;
  if (v.IsNumber()) return v.As<Napi::Number>().DoubleValue();
  if (v.IsBigInt()) {
    bool lossless = false;
    return static_cast<double>(v.As<Napi::BigInt>().Int64Value(&lossless));
  }
  throw_error(ErrorKind::Value, "array elements must be numbers, bigints, booleans or complex values");
}

// Stores one JS scalar into an element of dtype T following D-009 / D-033.
template <typename T>
void store_js_scalar(std::byte* p, const Napi::Value& v) {
  if constexpr (is_complex_v<T>) {
    using R = typename T::value_type;
    if (is_js_complex(v)) {
      const Napi::Object o = v.As<Napi::Object>();
      const Napi::Value im = o.Get("im");
      const double i = im.IsUndefined() ? 0.0 : js_real_part(im);
      store<T>(p, T(static_cast<R>(o.Get("re").As<Napi::Number>().DoubleValue()), static_cast<R>(i)));
      return;
    }
    store<T>(p, T(static_cast<R>(js_real_part(v)), R{0}));
  } else {
    if (is_js_complex(v)) {
      throw_error(ErrorKind::DType, "cannot store a complex value in a real array (D-033)");
    }
    if (v.IsBoolean()) {
      store<T>(p, cast_value<T>(v.As<Napi::Boolean>().Value()));
      return;
    }
    if constexpr (std::is_integral_v<T> && !std::is_same_v<T, bool>) {
      if (v.IsBigInt()) {
        bool lossless = false;
        if constexpr (std::is_signed_v<T>) {
          const std::int64_t x = v.As<Napi::BigInt>().Int64Value(&lossless);
          if (!lossless || x < std::numeric_limits<T>::min() || x > std::numeric_limits<T>::max()) {
            throw_error(ErrorKind::Value, "integer out of bounds for target dtype");
          }
          store<T>(p, static_cast<T>(x));
        } else {
          const std::uint64_t x = v.As<Napi::BigInt>().Uint64Value(&lossless);
          if (!lossless || x > std::numeric_limits<T>::max()) {
            throw_error(ErrorKind::Value, "integer out of bounds for target dtype");
          }
          store<T>(p, static_cast<T>(x));
        }
        return;
      }
      if (!v.IsNumber()) throw_error(ErrorKind::Value, "array elements must be numbers, bigints or booleans");
      store_number<T>(p, v.As<Napi::Number>().DoubleValue());
    } else {
      double d = 0.0;
      if (v.IsBigInt()) {
        bool lossless = false;
        d = static_cast<double>(v.As<Napi::BigInt>().Int64Value(&lossless));
      } else if (v.IsNumber()) {
        d = v.As<Napi::Number>().DoubleValue();
      } else {
        throw_error(ErrorKind::Value, "array elements must be numbers, bigints or booleans");
      }
      store_number<T>(p, d);
    }
  }
}

Napi::Value make_js_complex(Napi::Env env, double re, double im);

// Reads one element as a JS value (D-005: 64-bit ints become numbers).
template <typename T>
Napi::Value load_js_scalar(Napi::Env env, const std::byte* p) {
  if constexpr (is_complex_v<T>) {
    const T z = load<T>(p);
    return make_js_complex(env, static_cast<double>(z.real()), static_cast<double>(z.imag()));
  } else if constexpr (std::is_same_v<T, bool>) {
    return Napi::Boolean::New(env, load<bool>(p));
  } else {
    return Napi::Number::New(env, cast_value<double>(load<T>(p)));
  }
}

Napi::Value load_element(Napi::Env env, DType dt, const std::byte* p) {
  return dispatch_dtype(dt, [&](auto tag) -> Napi::Value {
    return load_js_scalar<dtype_t<decltype(tag)::value>>(env, p);
  });
}

}  // namespace

// ---- Per-environment state (no globals; PLAN §61) --------------------------
struct AddonData {
  Napi::FunctionReference ndarray_ctor;
  // Unique object passed to the constructor by create(). The identity check
  // replaces a per-construction string build, UTF-8 copy and compare.
  Napi::ObjectReference construct_token;
  // np.Complex, registered by the TS layer (D-033). Empty until then.
  Napi::FunctionReference complex_ctor;
};

namespace {
AddonData& addon_data(Napi::Env env) {
  AddonData* d = env.GetInstanceData<AddonData>();
  if (d == nullptr) throw_error(ErrorKind::Value, "nativpy addon not initialized");
  return *d;
}

Napi::Value make_js_complex(Napi::Env env, double re, double im) {
  AddonData& d = addon_data(env);
  const Napi::Value r = Napi::Number::New(env, re);
  const Napi::Value i = Napi::Number::New(env, im);
  if (!d.complex_ctor.IsEmpty()) return d.complex_ctor.New({r, i});
  Napi::Object o = Napi::Object::New(env);
  o.Set("re", r);
  o.Set("im", i);
  return o;
}
}  // namespace

// ---- NDArrayWrap ------------------------------------------------------------

NDArrayWrap::NDArrayWrap(const Napi::CallbackInfo& info) : Napi::ObjectWrap<NDArrayWrap>(info) {
  if (info.Length() != 1 || !info[0].IsObject() ||
      !info[0].StrictEquals(addon_data(info.Env()).construct_token.Value())) {
    throw_js_error(info.Env(), "ValueError",
                   "native NDArray handles cannot be constructed directly; use np.array()");
  }
}

NDArrayWrap::~NDArrayWrap() {
  // Buffer memory is owned by the shared_ptr in array_; it is released when the
  // last view (JS or C++) drops. External memory accounting is advisory only.
  // D-023: this runs synchronously inside GC, so it must stay basic-env safe
  // (AdjustExternalMemory takes a BasicEnv and never throws).
  if (reported_bytes_ != 0) {
    Napi::MemoryManagement::AdjustExternalMemory(Env(), -reported_bytes_);
  }
}

const NDArray& NDArrayWrap::array() const {
  if (!array_) throw_error(ErrorKind::Value, "NDArray handle is uninitialized");
  return *array_;
}

Napi::Object NDArrayWrap::create(Napi::Env env, NDArray array) {
  AddonData& d = addon_data(env);
  Napi::Object obj = d.ndarray_ctor.New({d.construct_token.Value()});
  NDArrayWrap* w = NDArrayWrap::Unwrap(obj);
  // Only the allocating handle reports memory to V8 so GC pressure is correct.
  if (array.owns_data()) {
    w->reported_bytes_ = static_cast<std::int64_t>(array.buffer()->size());
    Napi::MemoryManagement::AdjustExternalMemory(env, w->reported_bytes_);
  }
  w->array_.emplace(std::move(array));
  return obj;
}

const NDArray& NDArrayWrap::unwrap(const Napi::Value& value) {
  if (!value.IsObject()) throw_error(ErrorKind::Value, "expected a native NDArray");
  Napi::Object obj = value.As<Napi::Object>();
  Napi::Env env = value.Env();
  if (!obj.InstanceOf(addon_data(env).ndarray_ctor.Value())) {
    throw_error(ErrorKind::Value, "expected a native NDArray");
  }
  return NDArrayWrap::Unwrap(obj)->array();
}

bool NDArrayWrap::is_ndarray(const Napi::Value& value) {
  return value.IsObject() &&
         value.As<Napi::Object>().InstanceOf(addon_data(value.Env()).ndarray_ctor.Value());
}

#define NATIVPY_METHOD(NAME, BODY)                                  \
  Napi::Value NDArrayWrap::NAME(const Napi::CallbackInfo& info) {   \
    Napi::Env env = info.Env();                                     \
    return translate_errors(env, [&]() -> Napi::Value BODY);        \
  }

NATIVPY_METHOD(shape, { return int64_vector_to_js(env, array().shape()); })
NATIVPY_METHOD(strides, { return int64_vector_to_js(env, array().strides()); })
NATIVPY_METHOD(offset, { return Napi::Number::New(env, static_cast<double>(array().offset())); })
NATIVPY_METHOD(dtype, { return Napi::String::New(env, std::string(dtype_name(array().dtype()))); })
NATIVPY_METHOD(ndim, { return Napi::Number::New(env, static_cast<double>(array().ndim())); })
NATIVPY_METHOD(size, { return Napi::Number::New(env, static_cast<double>(array().size())); })
NATIVPY_METHOD(itemsize, { return Napi::Number::New(env, static_cast<double>(array().itemsize())); })
NATIVPY_METHOD(nbytes, { return Napi::Number::New(env, static_cast<double>(array().nbytes())); })

NATIVPY_METHOD(flags, {
  const NDArray& a = array();
  Napi::Object f = Napi::Object::New(env);
  f.Set("cContiguous", Napi::Boolean::New(env, a.is_c_contiguous()));
  f.Set("fContiguous", Napi::Boolean::New(env, a.is_f_contiguous()));
  f.Set("ownData", Napi::Boolean::New(env, a.owns_data()));
  f.Set("writeable", Napi::Boolean::New(env, a.writeable()));
  return f;
})


namespace {
// Builds nested JS arrays in C order, reading via strides.
Napi::Value build_list(Napi::Env env, const NDArray& a, std::size_t dim, const std::byte* ptr) {
  if (dim == a.ndim()) return load_element(env, a.dtype(), ptr);
  const std::int64_t n = a.shape()[dim];
  Napi::Array out = Napi::Array::New(env, static_cast<std::size_t>(n));
  for (std::int64_t i = 0; i < n; ++i) {
    out.Set(static_cast<std::uint32_t>(i),
            build_list(env, a, dim + 1, ptr + i * a.strides()[dim]));
  }
  return out;
}

template <typename TA>
Napi::Value make_typed(Napi::Env env, const NDArray& c) {
  const auto n = static_cast<std::size_t>(c.size());
  TA arr = TA::New(env, n);
  if (n > 0) std::memcpy(arr.Data(), c.data(), static_cast<std::size_t>(c.nbytes()));
  return arr;
}
}  // namespace

NATIVPY_METHOD(to_list, { return build_list(env, array(), 0, array().data()); })

// Always returns a new C-order copy (D-010). float16 -> Uint16Array raw bits,
// bool -> Uint8Array, complex -> interleaved Float32Array/Float64Array.
NATIVPY_METHOD(to_typed_array, {
  const NDArray c = array().copy();
  switch (c.dtype()) {
    case DType::Bool:
    case DType::UInt8: return make_typed<Napi::Uint8Array>(env, c);
    case DType::Int8: return make_typed<Napi::Int8Array>(env, c);
    case DType::Int16: return make_typed<Napi::Int16Array>(env, c);
    case DType::UInt16:
    case DType::Float16: return make_typed<Napi::Uint16Array>(env, c);
    case DType::Int32: return make_typed<Napi::Int32Array>(env, c);
    case DType::UInt32: return make_typed<Napi::Uint32Array>(env, c);
    case DType::Int64: return make_typed<Napi::BigInt64Array>(env, c);
    case DType::UInt64: return make_typed<Napi::BigUint64Array>(env, c);
    case DType::Float32: return make_typed<Napi::Float32Array>(env, c);
    case DType::Float64: return make_typed<Napi::Float64Array>(env, c);
    case DType::Complex64: {
      Napi::Float32Array arr = Napi::Float32Array::New(env, static_cast<std::size_t>(c.size()) * 2);
      if (c.size() > 0) std::memcpy(arr.Data(), c.data(), static_cast<std::size_t>(c.nbytes()));
      return arr;
    }
    case DType::Complex128: {
      Napi::Float64Array arr = Napi::Float64Array::New(env, static_cast<std::size_t>(c.size()) * 2);
      if (c.size() > 0) std::memcpy(arr.Data(), c.data(), static_cast<std::size_t>(c.nbytes()));
      return arr;
    }
  }
  throw_error(ErrorKind::DType, "unsupported dtype");
})

// view(shape, strides, offset) — offset in bytes relative to the buffer start.
NATIVPY_METHOD(view, {
  const auto shp = to_int64_vector(info[0], "shape");
  const auto str = to_int64_vector(info[1], "strides");
  const std::int64_t off = to_int64(info[2], "offset");
  return create(env, array().view(shp, str, off));
})

NATIVPY_METHOD(reshape, { return create(env, array().reshape(to_int64_vector(info[0], "shape"))); })
NATIVPY_METHOD(copy, { return create(env, array().copy()); })
NATIVPY_METHOD(astype, { return create(env, array().astype(parse_dtype(info[0]))); })
NATIVPY_METHOD(shares_memory, {
  return Napi::Boolean::New(env, array().may_share_memory(NDArrayWrap::unwrap(info[0])));
})

// get_item(indices[]) -> scalar at full integer index (negative allowed).
NATIVPY_METHOD(get_item, {
  const NDArray& a = array();
  const auto idx = to_int64_vector(info[0], "index");
  const std::int64_t off = a.byte_offset_of(idx);
  return load_element(env, a.dtype(), a.buffer()->data() + off);
})

#undef NATIVPY_METHOD

void NDArrayWrap::init(Napi::Env env, Napi::Object exports) {
  Napi::Function ctor = DefineClass(
      env, "NativeNDArray",
      {
          InstanceMethod<&NDArrayWrap::shape>("shape"),
          InstanceMethod<&NDArrayWrap::strides>("strides"),
          InstanceMethod<&NDArrayWrap::offset>("offset"),
          InstanceMethod<&NDArrayWrap::dtype>("dtype"),
          InstanceMethod<&NDArrayWrap::ndim>("ndim"),
          InstanceMethod<&NDArrayWrap::size>("size"),
          InstanceMethod<&NDArrayWrap::itemsize>("itemsize"),
          InstanceMethod<&NDArrayWrap::nbytes>("nbytes"),
          InstanceMethod<&NDArrayWrap::flags>("flags"),
          InstanceMethod<&NDArrayWrap::to_list>("toList"),
          InstanceMethod<&NDArrayWrap::to_typed_array>("toTypedArray"),
          InstanceMethod<&NDArrayWrap::view>("view"),
          InstanceMethod<&NDArrayWrap::reshape>("reshape"),
          InstanceMethod<&NDArrayWrap::copy>("copy"),
          InstanceMethod<&NDArrayWrap::astype>("astype"),
          InstanceMethod<&NDArrayWrap::shares_memory>("sharesMemory"),
          InstanceMethod<&NDArrayWrap::get_item>("getItem"),
      });
  auto data = std::make_unique<AddonData>(
      AddonData{Napi::Persistent(ctor), Napi::Persistent(Napi::Object::New(env)), Napi::FunctionReference()});
  env.SetInstanceData<AddonData>(data.release());  // env owns it; deleted on teardown
  exports.Set("NativeNDArray", ctor);
}

// ---- Module-level functions -------------------------------------------------

namespace {

// Recursively checks nested JS arrays are rectangular; fills `shape`.
void infer_shape(const Napi::Value& v, std::size_t depth, Shape& shape) {
  if (!v.IsArray()) return;
  const auto arr = v.As<Napi::Array>();
  if (depth >= kMaxDims) throw_error(ErrorKind::Value, "nesting too deep");
  shape.push_back(static_cast<std::int64_t>(arr.Length()));
  if (arr.Length() > 0) infer_shape(arr.Get(0u), depth + 1, shape);
}

template <typename T>
void fill_nested(const Napi::Value& v, const Shape& shape, std::size_t dim, std::byte*& out) {
  if (dim == shape.size()) {
    if (v.IsArray()) throw_error(ErrorKind::Value, "inhomogeneous shape: sequence where scalar expected");
    store_js_scalar<T>(out, v);
    out += sizeof(T);
    return;
  }
  if (!v.IsArray()) {
    throw_error(ErrorKind::Value,
                "setting an array element with a sequence. The requested array has an "
                "inhomogeneous shape after " + std::to_string(dim) + " dimensions");
  }
  const auto arr = v.As<Napi::Array>();
  if (static_cast<std::int64_t>(arr.Length()) != shape[dim]) {
    throw_error(ErrorKind::Value,
                "setting an array element with a sequence. The requested array has an "
                "inhomogeneous shape after " + std::to_string(dim) + " dimensions");
  }
  for (std::uint32_t i = 0; i < arr.Length(); ++i) fill_nested<T>(arr.Get(i), shape, dim + 1, out);
}

Napi::Value js_empty(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return translate_errors(env, [&]() -> Napi::Value {
    return NDArrayWrap::create(env, NDArray::empty(to_int64_vector(info[0], "shape"), parse_dtype(info[1])));
  });
}

Napi::Value js_zeros(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return translate_errors(env, [&]() -> Napi::Value {
    return NDArrayWrap::create(env, NDArray::zeros(to_int64_vector(info[0], "shape"), parse_dtype(info[1])));
  });
}

// fromNested(data, dtype): data is a JS scalar or (nested) array.
Napi::Value js_from_nested(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return translate_errors(env, [&]() -> Napi::Value {
    const DType dt = parse_dtype(info[1]);
    Shape shape;
    infer_shape(info[0], 0, shape);
    NDArray a = NDArray::empty(shape, dt);
    dispatch_dtype(dt, [&](auto tag) {
      using T = dtype_t<decltype(tag)::value>;
      std::byte* out = a.data();
      fill_nested<T>(info[0], shape, 0, out);
    });
    return NDArrayWrap::create(env, std::move(a));
  });
}

// fromFloat64(f64, shape, dtype): bulk path for np.array on all-number input
// (PLAN §80). The TS layer flattens the numbers into one Float64Array; every
// element goes through the same store_number<T> rule as fromNested (D-009).
Napi::Value js_from_float64(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return translate_errors(env, [&]() -> Napi::Value {
    if (!info[0].IsTypedArray() || info[0].As<Napi::TypedArray>().TypedArrayType() != napi_float64_array) {
      throw_error(ErrorKind::Value, "expected a Float64Array");
    }
    const auto src = info[0].As<Napi::Float64Array>();
    const Shape shape = to_int64_vector(info[1], "shape");
    const DType dt = parse_dtype(info[2]);
    NDArray a = NDArray::empty(shape, dt);
    if (static_cast<std::size_t>(a.size()) != src.ElementLength()) {
      throw_error(ErrorKind::Shape, "fromFloat64: element count does not match shape");
    }
    const double* in = src.Data();
    const auto n = static_cast<std::size_t>(a.size());
    dispatch_dtype(dt, [&](auto tag) {
      using T = dtype_t<decltype(tag)::value>;
      if constexpr (is_complex_v<T>) {
        using R = typename T::value_type;
        std::byte* out = a.data();
        for (std::size_t i = 0; i < n; ++i) store<T>(out + i * sizeof(T), T(static_cast<R>(in[i]), R{0}));
      } else if constexpr (std::is_same_v<T, double>) {
        if (n > 0) std::memcpy(a.data(), in, n * sizeof(double));
      } else {
        std::byte* out = a.data();
        for (std::size_t i = 0; i < n; ++i) store_number<T>(out + i * sizeof(T), in[i]);
      }
    });
    return NDArrayWrap::create(env, std::move(a));
  });
}

// fromTypedArray(typedArray, shape, dtype): copies the bytes (D-010 / PLAN §31:
// zero-copy import is deferred until lifetime rules are designed).
Napi::Value js_from_typed_array(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return translate_errors(env, [&]() -> Napi::Value {
    if (!info[0].IsTypedArray()) throw_error(ErrorKind::Value, "expected a TypedArray");
    const auto ta = info[0].As<Napi::TypedArray>();
    const Shape shape = to_int64_vector(info[1], "shape");
    const DType dt = parse_dtype(info[2]);
    NDArray a = NDArray::empty(shape, dt);
    if (static_cast<std::size_t>(a.nbytes()) != ta.ByteLength()) {
      throw_error(ErrorKind::Shape, "TypedArray byte length " + std::to_string(ta.ByteLength()) +
                                        " does not match shape " + shape_to_string(shape) +
                                        " with dtype " + std::string(dtype_name(dt)));
    }
    if (a.nbytes() > 0) {
      const auto* src = static_cast<const std::byte*>(ta.ArrayBuffer().Data()) + ta.ByteOffset();
      std::memcpy(a.data(), src, static_cast<std::size_t>(a.nbytes()));
    }
    return NDArrayWrap::create(env, std::move(a));
  });
}

Napi::Value js_live_buffers(const Napi::CallbackInfo& info) {
  Napi::Object o = Napi::Object::New(info.Env());
  o.Set("buffers", Napi::Number::New(info.Env(), static_cast<double>(MemoryBuffer::live_buffers())));
  o.Set("bytes", Napi::Number::New(info.Env(), static_cast<double>(MemoryBuffer::live_bytes())));
  return o;
}

// setComplexClass(ctor): the TS np.Complex class used for complex elements (D-033).
Napi::Value js_set_complex_class(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  return translate_errors(env, [&]() -> Napi::Value {
    if (!info[0].IsFunction()) throw_error(ErrorKind::Value, "setComplexClass expects a constructor");
    addon_data(env).complex_ctor = Napi::Persistent(info[0].As<Napi::Function>());
    return env.Undefined();
  });
}

}  // namespace

void init_ndarray_binding(Napi::Env env, Napi::Object exports) {
  NDArrayWrap::init(env, exports);
  exports.Set("setComplexClass", Napi::Function::New(env, js_set_complex_class, "setComplexClass"));
  exports.Set("empty", Napi::Function::New(env, js_empty, "empty"));
  exports.Set("zeros", Napi::Function::New(env, js_zeros, "zeros"));
  exports.Set("fromNested", Napi::Function::New(env, js_from_nested, "fromNested"));
  exports.Set("fromFloat64", Napi::Function::New(env, js_from_float64, "fromFloat64"));
  exports.Set("fromTypedArray", Napi::Function::New(env, js_from_typed_array, "fromTypedArray"));
  exports.Set("memoryStats", Napi::Function::New(env, js_live_buffers, "memoryStats"));
}

}  // namespace nativpy::bindings


