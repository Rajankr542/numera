#pragma once

#include <napi.h>

#include "dtype.hpp"

namespace nativpy::bindings {

// Parses a dtype name string. Throws DTypeError for unknown names.
DType parse_dtype(const Napi::Value& value);

// Parses a casting rule name (D-045). Throws ValueError for unknown names.
Casting parse_casting(const Napi::Value& value);

// Exports `dtypes`: { name: { itemsize, alignment, kind } }, `promoteTypes`
// and `canCast`.
void init_dtype_binding(Napi::Env env, Napi::Object exports);

}  // namespace nativpy::bindings
