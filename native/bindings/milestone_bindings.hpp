#pragma once

#include <napi.h>

namespace nativpy::bindings {

// One init function per parity milestone (D-056). Each sets
// `exports.<module>` to an object of native functions, read from TS via
// `nativeModule<T>("<module>")` (packages/numera/src/addon.ts).
void init_p03_binding(Napi::Env env, Napi::Object exports);  // exports.p03
void init_p04_binding(Napi::Env env, Napi::Object exports);  // exports.p04
void init_p05_binding(Napi::Env env, Napi::Object exports);  // exports.p05
void init_p06_binding(Napi::Env env, Napi::Object exports);  // exports.p06
void init_p07_binding(Napi::Env env, Napi::Object exports);  // exports.p07
void init_p08_binding(Napi::Env env, Napi::Object exports);  // exports.p08
void init_p09_binding(Napi::Env env, Napi::Object exports);  // exports.p09
void init_p10_binding(Napi::Env env, Napi::Object exports);  // exports.p10
void init_p11_binding(Napi::Env env, Napi::Object exports);  // exports.p11
void init_p12_binding(Napi::Env env, Napi::Object exports);  // exports.p12
void init_p13_binding(Napi::Env env, Napi::Object exports);  // exports.p13
void init_p14_binding(Napi::Env env, Napi::Object exports);  // exports.p14
void init_p15_binding(Napi::Env env, Napi::Object exports);  // exports.p15
void init_p16c_binding(Napi::Env env, Napi::Object exports); // exports.p16c

}  // namespace nativpy::bindings
