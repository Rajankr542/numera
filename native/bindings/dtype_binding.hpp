#pragma once

#include <napi.h>

#include "dtype.hpp"

namespace nativpy::bindings {

// Parses a dtype name string. Throws DTypeError for unknown names.
DType parse_dtype(const Napi::Value& value);

// Exports `dtypes`: { name: { itemsize, alignment, kind } } and `promoteTypes`.
void init_dtype_binding(Napi::Env env, Napi::Object exports);

}  // namespace nativpy::bindings
