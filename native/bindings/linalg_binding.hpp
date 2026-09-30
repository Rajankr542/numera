#pragma once

#include <napi.h>

namespace nativpy::bindings {

// Linear algebra functions (M8, D-018), exposed as `exports.linalg`.
void init_linalg_binding(Napi::Env env, Napi::Object exports);

}  // namespace nativpy::bindings
