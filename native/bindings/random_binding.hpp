#pragma once

#include <napi.h>

namespace nativpy::bindings {

// Random bit generators and distributions (M9, D-019), exposed as
// `exports.random.BitGenerator`.
void init_random_binding(Napi::Env env, Napi::Object exports);

}  // namespace nativpy::bindings
