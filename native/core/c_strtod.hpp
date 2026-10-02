#pragma once

// Locale-independent decimal-to-double parsing.
//
// libc++ only provides floating-point std::from_chars from macOS 26, so it
// cannot be used with the macOS 13.3 deployment target of the prebuilds.
// strtod_l with the "C" locale is correctly rounded on macOS and glibc.

#include <clocale>
#include <cstdlib>
#include <string>

#if defined(__APPLE__)
#include <xlocale.h>
#else
#include <locale.h>
#endif

namespace nativpy {

// Parses [first, last), which callers have already validated as decimal float
// syntax (sign, digits, '.', exponent; no hex/inf/nan). Overflow yields +-inf
// and underflow yields a subnormal or +-0. Returns false if strtod does not
// consume the whole range.
inline bool parse_decimal_c(const char* first, const char* last, double& out) {
  static const locale_t c_locale = newlocale(LC_ALL_MASK, "C", static_cast<locale_t>(nullptr));
  const std::string buf(first, last);
  char* end = nullptr;
  out = strtod_l(buf.c_str(), &end, c_locale);
  return end == buf.c_str() + buf.size() && !buf.empty();
}

}  // namespace nativpy
