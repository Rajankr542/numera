cmake_minimum_required(VERSION 3.20)
project(nativpy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(NATIVPY_BUILD_ADDON "Build the Node-API addon (requires cmake-js)" ON)
option(NATIVPY_BUILD_TESTS "Build C++ unit tests" OFF)
option(NATIVPY_SANITIZE "Enable AddressSanitizer + UndefinedBehaviorSanitizer" OFF)
option(NATIVPY_NAPI_EXPERIMENTAL "Build against experimental Node-API: GC-time buffer release (D-023)" ON)

if(MSVC)
  set(NATIVPY_WARNINGS /W4 /WX /permissive-)
else()
  set(NATIVPY_WARNINGS -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror)
endif()

# D-025: match how the NumPy wheels are compiled on each platform. The Linux
# wheels do not contract a*b+c, but GCC defaults to -ffp-contract=fast (FMA on
# aarch64), which broke bit-exact arange/linspace/random vs NumPy on Linux arm64.
# The macOS wheels use Clang's default (-ffp-contract=on), so Apple keeps it.
if(NOT MSVC AND NOT APPLE)
  add_compile_options(-ffp-contract=off)
endif()

if(NATIVPY_SANITIZE AND NOT MSVC)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer
                      -fno-sanitize-recover=undefined)
  add_link_options(-fsanitize=address,undefined)
endif()

# ---- Core library (no Node dependency) ------------------------------------
add_library(nativpy_core STATIC
  native/core/error.cpp
  native/core/memory.cpp
  native/core/dtype.cpp
  native/core/shape.cpp
  native/core/strides.cpp
  native/core/ndarray.cpp
  native/core/creation.cpp
  native/core/shape_ops.cpp
  native/core/broadcast.cpp
  native/core/ufunc.cpp
  native/core/indexing.cpp
  native/core/reduce.cpp
)
target_include_directories(nativpy_core PUBLIC native/core)
target_compile_options(nativpy_core PRIVATE ${NATIVPY_WARNINGS})
# D-021: reductions must round every multiply separately (NumPy bit-exactness);
# clang defaults to -ffp-contract=on, which may fuse d*d + acc into an FMA.
if(NOT MSVC)
  set_source_files_properties(native/core/reduce.cpp PROPERTIES COMPILE_OPTIONS -ffp-contract=off)
endif()

# ---- Linear algebra (D-018) ------------------------------------------------
set(NATIVPY_LINALG_BACKEND "auto" CACHE STRING "Linalg backend: auto|accelerate|fallback")
set_property(CACHE NATIVPY_LINALG_BACKEND PROPERTY STRINGS auto accelerate fallback)
add_library(nativpy_linalg STATIC
  native/linalg/fallback_backend.cpp
  native/linalg/accelerate_backend.cpp
  native/linalg/linalg.cpp
)
target_include_directories(nativpy_linalg PUBLIC native/linalg)
target_link_libraries(nativpy_linalg PUBLIC nativpy_core)
target_compile_options(nativpy_linalg PRIVATE ${NATIVPY_WARNINGS})
set(_nativpy_use_accelerate OFF)
if(APPLE AND NOT NATIVPY_LINALG_BACKEND STREQUAL "fallback")
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    set(_nativpy_use_accelerate ON)
  endif()
endif()
if(NATIVPY_LINALG_BACKEND STREQUAL "accelerate" AND NOT _nativpy_use_accelerate)
  message(FATAL_ERROR "NATIVPY_LINALG_BACKEND=accelerate requested but Accelerate not found")
endif()
if(_nativpy_use_accelerate)
  target_compile_definitions(nativpy_linalg PRIVATE NATIVPY_HAVE_ACCELERATE ACCELERATE_NEW_LAPACK)
  target_link_libraries(nativpy_linalg PUBLIC ${ACCELERATE_FRAMEWORK})
  message(STATUS "nativpy linalg backend: accelerate")
else()
  message(STATUS "nativpy linalg backend: fallback")
endif()

# ---- Random (D-019) ---------------------------------------------------------
add_library(nativpy_random STATIC
  native/random/bitgen.cpp
  native/random/distributions.cpp
)
target_include_directories(nativpy_random PUBLIC native/random)
target_link_libraries(nativpy_random PUBLIC nativpy_core)
target_compile_options(nativpy_random PRIVATE ${NATIVPY_WARNINGS})

# ---- FFT (D-020) ------------------------------------------------------------
add_library(nativpy_fft STATIC
  native/fft/fft.cpp
)
target_include_directories(nativpy_fft PUBLIC native/fft)
# Vendored pocketfft: SYSTEM so our -Werror flags do not apply to its header.
target_include_directories(nativpy_fft SYSTEM PRIVATE third_party/pocketfft)
target_link_libraries(nativpy_fft PUBLIC nativpy_core)
target_compile_options(nativpy_fft PRIVATE ${NATIVPY_WARNINGS})

# ---- Node-API addon --------------------------------------------------------
if(NATIVPY_BUILD_ADDON AND CMAKE_JS_VERSION)
  execute_process(
    COMMAND node -p "require('node-addon-api').include"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE NODE_ADDON_API_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  string(REPLACE "\"" "" NODE_ADDON_API_DIR "${NODE_ADDON_API_DIR}")

  add_library(nativpy_addon SHARED
    native/bindings/addon.cpp
    native/bindings/ndarray_binding.cpp
    native/bindings/dtype_binding.cpp
    native/bindings/error_binding.cpp
    native/bindings/ops_binding.cpp
    native/bindings/linalg_binding.cpp
    native/bindings/random_binding.cpp
    native/bindings/fft_binding.cpp
    ${CMAKE_JS_SRC})
  set_target_properties(nativpy_addon PROPERTIES PREFIX "" SUFFIX ".node"
                        OUTPUT_NAME "nativpy")
  target_include_directories(nativpy_addon PRIVATE
    ${CMAKE_JS_INC} ${NODE_ADDON_API_DIR} native/bindings)
  target_compile_definitions(nativpy_addon PRIVATE
    NODE_ADDON_API_CPP_EXCEPTIONS NODE_ADDON_API_DISABLE_DEPRECATED)
  if(NATIVPY_NAPI_EXPERIMENTAL)
    # D-023: experimental Node-API runs basic finalizers synchronously in GC,
    # so result buffers are freed immediately instead of after the event loop.
    # REQUIRE_BASIC_FINALIZERS makes any non-basic finalizer a compile error.
    target_compile_definitions(nativpy_addon PRIVATE
      NAPI_EXPERIMENTAL NODE_ADDON_API_REQUIRE_BASIC_FINALIZERS)
  else()
    target_compile_definitions(nativpy_addon PRIVATE NAPI_VERSION=8)
  endif()
  # Node/N-API headers are not -Wconversion clean; keep warnings on our code only.
  target_compile_options(nativpy_addon PRIVATE
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Werror>)
  target_link_libraries(nativpy_addon PRIVATE nativpy_fft nativpy_random nativpy_linalg nativpy_core ${CMAKE_JS_LIB})
  if(MSVC AND CMAKE_JS_NODELIB_DEF AND CMAKE_JS_NODELIB_TARGET)
    execute_process(COMMAND ${CMAKE_AR} /def:${CMAKE_JS_NODELIB_DEF}
                    /out:${CMAKE_JS_NODELIB_TARGET} ${CMAKE_STATIC_LINKER_FLAGS})
  endif()
endif()

# ---- C++ unit tests --------------------------------------------------------
if(NATIVPY_BUILD_TESTS)
  enable_testing()
  add_executable(nativpy_core_tests
    tests/native/test_main.cpp
    tests/native/test_memory.cpp
    tests/native/test_dtype.cpp
    tests/native/test_shape_strides.cpp
    tests/native/test_ndarray.cpp
    tests/native/test_creation_shape.cpp
    tests/native/test_ufunc.cpp
    tests/native/test_indexing.cpp
    tests/native/test_reduce.cpp
    tests/native/test_linalg.cpp
    tests/native/test_random.cpp
    tests/native/test_fft.cpp)
  target_link_libraries(nativpy_core_tests PRIVATE nativpy_fft nativpy_random nativpy_linalg nativpy_core)
  target_compile_options(nativpy_core_tests PRIVATE ${NATIVPY_WARNINGS})
  add_test(NAME nativpy_core_tests COMMAND nativpy_core_tests)
endif()
