#pragma once

#include <napi.h>

namespace nativpy::bindings {

// Module-level creation (M2), shape (M3), and later ufunc functions.
void init_ops_binding(Napi::Env env, Napi::Object exports);

}  // namespace nativpy::bindings
