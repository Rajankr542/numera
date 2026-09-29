#include "test_harness.hpp"

namespace nptest {

namespace {
int g_failures = 0;
}

std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

void record_failure(const char* file, int line, const std::string& msg) {
  ++g_failures;
  std::cerr << "  FAIL " << file << ":" << line << ": " << msg << "\n";
}

}  // namespace nptest

int main() {
  int failed_cases = 0;
  for (const auto& tc : nptest::registry()) {
    const int before = nptest::g_failures;
    try {
      tc.fn();
    } catch (const std::exception& e) {
      nptest::record_failure(__FILE__, __LINE__,
                             std::string("unexpected exception: ") + e.what());
    }
    const bool ok = nptest::g_failures == before;
    if (!ok) ++failed_cases;
    std::cout << (ok ? "[ OK ] " : "[FAIL] ") << tc.name << "\n";
  }
  std::cout << nptest::registry().size() << " test cases, " << failed_cases
            << " failed\n";
  return failed_cases == 0 ? 0 : 1;
}
