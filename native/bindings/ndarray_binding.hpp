#pragma once

#include <napi.h>

#include <memory>
#include <optional>
#include <unordered_map>

#include "ndarray.hpp"

namespace nativpy::bindings {

class NDArrayWrap;
// Buffer -> the wrapper that owns it (NumPy `base`, D-060). Shared by every
// owning wrapper so its destructor can unregister without N-API calls (D-023).
using OwnerMap = std::unordered_map<const MemoryBuffer*, NDArrayWrap*>;

// JS wrapper owning one NDArray (i.e. one reference to its MemoryBuffer).
// Instances can only be created from C++ via NDArrayWrap::create.
class NDArrayWrap : public Napi::ObjectWrap<NDArrayWrap> {
 public:
  static void init(Napi::Env env, Napi::Object exports);
  static Napi::Object create(Napi::Env env, NDArray array);
  // Unwraps a JS value; throws ValueError if it is not a native NDArray.
  static const NDArray& unwrap(const Napi::Value& value);
  // True if `value` is a native NDArray handle.
  static bool is_ndarray(const Napi::Value& value);

  explicit NDArrayWrap(const Napi::CallbackInfo& info);
  ~NDArrayWrap() override;
  NDArrayWrap(const NDArrayWrap&) = delete;
  NDArrayWrap& operator=(const NDArrayWrap&) = delete;

  [[nodiscard]] const NDArray& array() const;

 private:
  Napi::Value shape(const Napi::CallbackInfo& info);
  Napi::Value strides(const Napi::CallbackInfo& info);
  Napi::Value offset(const Napi::CallbackInfo& info);
  Napi::Value dtype(const Napi::CallbackInfo& info);
  Napi::Value ndim(const Napi::CallbackInfo& info);
  Napi::Value size(const Napi::CallbackInfo& info);
  Napi::Value itemsize(const Napi::CallbackInfo& info);
  Napi::Value nbytes(const Napi::CallbackInfo& info);
  Napi::Value flags(const Napi::CallbackInfo& info);
  Napi::Value to_list(const Napi::CallbackInfo& info);
  Napi::Value to_typed_array(const Napi::CallbackInfo& info);
  Napi::Value view(const Napi::CallbackInfo& info);
  Napi::Value reshape(const Napi::CallbackInfo& info);
  Napi::Value copy(const Napi::CallbackInfo& info);
  Napi::Value astype(const Napi::CallbackInfo& info);
  Napi::Value shares_memory(const Napi::CallbackInfo& info);
  Napi::Value get_item(const Napi::CallbackInfo& info);
  Napi::Value base(const Napi::CallbackInfo& info);
  Napi::Value set_writeable(const Napi::CallbackInfo& info);

  std::optional<NDArray> array_;
  std::int64_t reported_bytes_ = 0;
  std::shared_ptr<OwnerMap> owners_;  // set when this wrapper owns its buffer
};

void init_ndarray_binding(Napi::Env env, Napi::Object exports);

}  // namespace nativpy::bindings
