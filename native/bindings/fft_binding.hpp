#pragma once

#include <napi.h>

namespace nativpy::bindings {

// FFT functions (M10, D-020), exposed as `exports.fft`.
void init_fft_binding(Napi::Env env, Napi::Object exports);

}  // namespace nativpy::bindings
