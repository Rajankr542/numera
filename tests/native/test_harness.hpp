#pragma once

// Minimal dependency-free test harness for nativpy C++ unit tests.
#include <cmath>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace nptest {

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

std::vector<TestCase>& registry();
void record_failure(const char* file, int line, const std::string& msg);

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) {
    registry().push_back({name, std::move(fn)});
  }
};

}  // namespace nptest

#define NP_CONCAT2(a, b) a##b
#define NP_CONCAT(a, b) NP_CONCAT2(a, b)
#define TEST_CASE(name)                                                    \
  static void NP_CONCAT(np_test_fn_, __LINE__)();                          \
  static const nptest::Registrar NP_CONCAT(np_test_reg_, __LINE__)(        \
      name, NP_CONCAT(np_test_fn_, __LINE__));                             \
  static void NP_CONCAT(np_test_fn_, __LINE__)()

#define CHECK(cond)                                                        \
  do {                                                                     \
    if (!(cond)) nptest::record_failure(__FILE__, __LINE__, "CHECK(" #cond ")"); \
  } while (0)

#define CHECK_EQ(a, b)                                                     \
  do {                                                                     \
    const auto np_a_ = (a);                                                \
    const auto np_b_ = (b);                                                \
    if (!(np_a_ == np_b_)) {                                               \
      std::ostringstream np_os_;                                           \
      np_os_ << "CHECK_EQ(" #a ", " #b ") got " << np_a_ << " vs " << np_b_; \
      nptest::record_failure(__FILE__, __LINE__, np_os_.str());            \
    }                                                                      \
  } while (0)

#define CHECK_THROWS_KIND(expr, expected_kind)                             \
  do {                                                                     \
    bool np_threw_ = false;                                                \
    try {                                                                  \
      (void)(expr);                                                        \
    } catch (const nativpy::Error& np_e_) {                                \
      np_threw_ = np_e_.kind() == (expected_kind);                         \
    }                                                                      \
    if (!np_threw_)                                                        \
      nptest::record_failure(__FILE__, __LINE__,                           \
                             "expected nativpy::Error kind from " #expr);  \
  } while (0)
