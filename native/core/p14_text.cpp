#include "p14_text.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string_view>

#include "c_strtod.hpp"
#include "error.hpp"

namespace nativpy::p14 {

namespace {

[[noreturn]] void value_error(const std::string& m) { throw_error(ErrorKind::Value, m); }

bool is_ws(char c) { return c == ' ' || c == '\t' || c == '\v' || c == '\f' || c == '\r'; }
bool is_digit(char c) { return c >= '0' && c <= '9'; }

std::string_view strip(std::string_view s) {
  while (!s.empty() && (is_ws(s.front()) || s.front() == '\n')) s.remove_prefix(1);
  while (!s.empty() && (is_ws(s.back()) || s.back() == '\n')) s.remove_suffix(1);
  return s;
}

bool ieq_prefix(std::string_view s, std::size_t i, std::string_view word) {
  if (s.size() - i < word.size()) return false;
  for (std::size_t k = 0; k < word.size(); ++k) {
    char c = s[i + k];
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    if (c != word[k]) return false;
  }
  return true;
}

// Parses a float prefix of `s` at `i` (NumPy's text reader: decimal or
// inf/infinity/nan, no hex, no underscores). Returns false if none.
bool parse_float_prefix(std::string_view s, std::size_t& i, double& out) {
  std::size_t p = i;
  bool neg = false;
  if (p < s.size() && (s[p] == '+' || s[p] == '-')) neg = s[p++] == '-';
  if (ieq_prefix(s, p, "inf")) {
    p += 3;
    if (ieq_prefix(s, p, "inity")) p += 5;
    out = neg ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
    i = p;
    return true;
  }
  if (ieq_prefix(s, p, "nan")) {
    out = std::numeric_limits<double>::quiet_NaN();
    i = p + 3;
    return true;
  }
  const std::size_t start = p;
  bool any = false;
  while (p < s.size() && is_digit(s[p])) {
    ++p;
    any = true;
  }
  if (p < s.size() && s[p] == '.') {
    ++p;
    while (p < s.size() && is_digit(s[p])) {
      ++p;
      any = true;
    }
  }
  if (!any) return false;
  if (p < s.size() && (s[p] == 'e' || s[p] == 'E')) {
    std::size_t q = p + 1;
    if (q < s.size() && (s[q] == '+' || s[q] == '-')) ++q;
    if (q < s.size() && is_digit(s[q])) {
      while (q < s.size() && is_digit(s[q])) ++q;
      p = q;
    }
  }
  double v = 0.0;
  if (!parse_decimal_c(s.data() + start, s.data() + p, v)) return false;
  out = neg ? -v : v;
  i = p;
  return true;
}

bool parse_double(std::string_view field, double& out) {
  const std::string_view s = strip(field);
  std::size_t i = 0;
  return parse_float_prefix(s, i, out) && i == s.size();
}

bool parse_complex(std::string_view field, double& re, double& im) {
  std::string_view s = strip(field);
  if (s.size() >= 2 && s.front() == '(' && s.back() == ')') s = s.substr(1, s.size() - 2);
  std::size_t i = 0;
  double a = 0;
  if (!parse_float_prefix(s, i, a)) return false;
  if (i == s.size()) {
    re = a;
    im = 0;
    return true;
  }
  if ((s[i] == 'j' || s[i] == 'J') && i + 1 == s.size()) {
    re = 0;
    im = a;
    return true;
  }
  if (s[i] != '+' && s[i] != '-') return false;
  double b = 0;
  if (!parse_float_prefix(s, i, b)) return false;
  if (i + 1 != s.size() || (s[i] != 'j' && s[i] != 'J')) return false;
  re = a;
  im = b;
  return true;
}

bool parse_int(std::string_view field, bool allow_neg, std::int64_t& si, std::uint64_t& ui, bool& neg) {
  std::string_view s = strip(field);
  neg = false;
  if (!s.empty() && (s[0] == '+' || s[0] == '-')) {
    neg = s[0] == '-';
    s.remove_prefix(1);
  }
  if (s.empty() || !std::all_of(s.begin(), s.end(), is_digit)) return false;
  if (neg && !allow_neg) return false;
  std::uint64_t v = 0;
  const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
  if (r.ec != std::errc{}) return false;
  ui = v;
  if (neg) {
    if (v > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1) return false;
    si = v == static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) + 1
             ? std::numeric_limits<std::int64_t>::min()
             : -static_cast<std::int64_t>(v);
  } else {
    si = static_cast<std::int64_t>(v);
  }
  return true;
}

// Converts one field into `dst` (an item of dtype `dt`). Returns false on failure.
bool convert_field(std::string_view field, DType dt, std::byte* dst) {
  switch (dt) {
    case DType::Float16:
    case DType::Float32:
    case DType::Float64: {
      double v = 0;
      if (!parse_double(field, v)) return false;
      if (dt == DType::Float64) std::memcpy(dst, &v, 8);
      else if (dt == DType::Float32) {
        const auto f = static_cast<float>(v);
        std::memcpy(dst, &f, 4);
      } else {
        const float16_t h = double_to_half(v);
        std::memcpy(dst, &h.bits, 2);
      }
      return true;
    }
    case DType::Complex64:
    case DType::Complex128: {
      double re = 0, im = 0;
      if (!parse_complex(field, re, im)) return false;
      if (dt == DType::Complex128) {
        std::memcpy(dst, &re, 8);
        std::memcpy(dst + 8, &im, 8);
      } else {
        const auto fr = static_cast<float>(re), fi = static_cast<float>(im);
        std::memcpy(dst, &fr, 4);
        std::memcpy(dst + 4, &fi, 4);
      }
      return true;
    }
    default: break;
  }
  std::int64_t si = 0;
  std::uint64_t ui = 0;
  bool neg = false;
  const bool is_unsigned = dtype_info(dt).kind == 'u';
  if (!parse_int(field, !is_unsigned, si, ui, neg)) return false;
  if (dt == DType::Bool) {
    const std::uint8_t b = ui != 0 ? 1 : 0;
    std::memcpy(dst, &b, 1);
    return true;
  }
  if (is_unsigned) {
    const std::size_t n = itemsize(dt);
    if (n < 8 && ui >= (std::uint64_t{1} << (8 * n))) return false;
    switch (n) {
      case 1: { const auto v = static_cast<std::uint8_t>(ui); std::memcpy(dst, &v, 1); break; }
      case 2: { const auto v = static_cast<std::uint16_t>(ui); std::memcpy(dst, &v, 2); break; }
      case 4: { const auto v = static_cast<std::uint32_t>(ui); std::memcpy(dst, &v, 4); break; }
      default: std::memcpy(dst, &ui, 8);
    }
    return true;
  }
  if (!neg && ui > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) return false;
  const std::size_t n = itemsize(dt);
  if (n < 8) {
    const std::int64_t lim = std::int64_t{1} << (8 * n - 1);
    if (si < -lim || si >= lim) return false;
  }
  switch (n) {
    case 1: { const auto v = static_cast<std::int8_t>(si); std::memcpy(dst, &v, 1); break; }
    case 2: { const auto v = static_cast<std::int16_t>(si); std::memcpy(dst, &v, 2); break; }
    case 4: { const auto v = static_cast<std::int32_t>(si); std::memcpy(dst, &v, 4); break; }
    default: std::memcpy(dst, &si, 8);
  }
  return true;
}

// Splits a line into fields. Returns false for a line that holds no data.
bool split_line(std::string_view line, const LoadtxtOptions& o, std::vector<std::string>& fields) {
  fields.clear();
  const bool ws_mode = !o.delimiter.has_value();
  const std::string_view delim = ws_mode ? std::string_view{} : std::string_view(*o.delimiter);
  const std::string_view quote = o.quote ? std::string_view(*o.quote) : std::string_view{};
  auto comment_at = [&](std::size_t i) {
    return std::any_of(o.comments.begin(), o.comments.end(),
                       [&](const std::string& c) { return line.compare(i, c.size(), c) == 0; });
  };
  auto at = [&](std::size_t i, std::string_view w) { return !w.empty() && line.compare(i, w.size(), w) == 0; };
  std::size_t i = 0;
  if (ws_mode) {
    while (i < line.size() && is_ws(line[i])) ++i;
  }
  if (i >= line.size() || comment_at(i)) return false;
  std::string cur;
  bool started = false, in_quote = false;
  while (i < line.size()) {
    if (in_quote) {
      if (at(i, quote)) {
        if (at(i + quote.size(), quote)) {
          cur += quote;
          i += 2 * quote.size();
        } else {
          in_quote = false;
          i += quote.size();
        }
      } else {
        cur += line[i++];
      }
      continue;
    }
    if (comment_at(i)) break;
    if (at(i, quote)) {
      in_quote = started = true;
      i += quote.size();
      continue;
    }
    if (ws_mode && is_ws(line[i])) {
      fields.push_back(std::move(cur));
      cur.clear();
      started = false;
      while (i < line.size() && is_ws(line[i])) ++i;
      continue;
    }
    if (!ws_mode && at(i, delim)) {
      fields.push_back(std::move(cur));
      cur.clear();
      i += delim.size();
      continue;
    }
    cur += line[i++];
    started = true;
  }
  if (!ws_mode || started) fields.push_back(std::move(cur));
  return true;
}

}  // namespace

NDArray loadtxt(const std::string& text, const LoadtxtOptions& o) {
  if (o.skiprows < 0) value_error("argument must be nonnegative");
  const DType dt = o.dtype;
  const std::size_t isz = itemsize(dt);
  std::vector<std::byte> data;
  std::vector<std::string> fields;
  std::int64_t ncols = o.usecols ? static_cast<std::int64_t>(o.usecols->size()) : -1;
  std::int64_t rows = 0;
  std::int64_t skipped = 0;
  std::size_t pos = 0;
  while (pos <= text.size() && (o.max_rows < 0 || rows < o.max_rows)) {
    std::size_t nl = text.find('\n', pos);
    if (nl == std::string::npos) {
      if (pos == text.size()) break;
      nl = text.size();
    }
    std::string_view line(text.data() + pos, nl - pos);
    pos = nl + 1;
    if (skipped < o.skiprows) {
      ++skipped;
      continue;
    }
    if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
    if (!split_line(line, o, fields)) continue;
    const auto nf = static_cast<std::int64_t>(fields.size());
    std::vector<std::int64_t> cols;
    if (o.usecols) {
      for (std::int64_t c : *o.usecols) {
        const std::int64_t k = c < 0 ? c + nf : c;
        if (k < 0 || k >= nf) {
          value_error("invalid column index " + std::to_string(c) + " at row " + std::to_string(rows + 1) +
                      " with " + std::to_string(nf) + " columns");
        }
        cols.push_back(k);
      }
    } else {
      if (ncols < 0) ncols = nf;
      if (nf != ncols) {
        value_error("the number of columns changed from " + std::to_string(ncols) + " to " + std::to_string(nf) +
                    " at row " + std::to_string(rows + 1) +
                    "; use `usecols` to select a subset and avoid this error");
      }
      for (std::int64_t k = 0; k < nf; ++k) cols.push_back(k);
    }
    const std::size_t base = data.size();
    data.resize(base + cols.size() * isz);
    for (std::size_t j = 0; j < cols.size(); ++j) {
      const std::string& f = fields[static_cast<std::size_t>(cols[j])];
      if (!convert_field(f, dt, data.data() + base + j * isz)) {
        value_error("could not convert string '" + f + "' to " + std::string(dtype_name(dt)) + " at row " +
                    std::to_string(rows) + ", column " + std::to_string(cols[j] + 1) + ".");
      }
    }
    ++rows;
  }
  if (ncols < 0) ncols = 1;
  NDArray out = NDArray::empty({rows, rows == 0 ? ncols : ncols}, dt);
  if (!data.empty()) std::memcpy(out.data(), data.data(), data.size());
  return out;
}

// ---------------- float to text ----------------

namespace {

// Shortest round-trip digits and decimal exponent: value = 0.d1d2... * 10^(exp+1)
// i.e. d1.d2d3... * 10^exp.
void shortest_digits(double v, DType dt, std::string& digits, int& exp10) {
  char buf[64];
  std::to_chars_result r{};
  if (dt == DType::Float32) {
    r = std::to_chars(buf, buf + sizeof buf, static_cast<float>(v), std::chars_format::scientific);
  } else if (dt == DType::Float64) {
    r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific);
  } else {
    const std::uint16_t want = double_to_half(v).bits;
    for (int prec = 0; prec < 17; ++prec) {
      r = std::to_chars(buf, buf + sizeof buf, v, std::chars_format::scientific, prec);
      double back = 0;
      parse_decimal_c(buf, r.ptr, back);
      if (double_to_half(back).bits == want) break;
    }
  }
  const std::string s(buf, r.ptr);
  const std::size_t e = s.find('e');
  digits.clear();
  for (std::size_t k = 0; k < e; ++k) {
    if (is_digit(s[k])) digits += s[k];
  }
  while (digits.size() > 1 && digits.back() == '0') digits.pop_back();
  exp10 = std::stoi(s.substr(e + 1));
}

std::string format_shortest(double v, DType dt, double upper, bool force_point_sci) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
  const std::string sign = std::signbit(v) ? "-" : "";
  const double a = std::fabs(v);
  if (a == 0) return sign + "0.0";
  std::string d;
  int e = 0;
  shortest_digits(a, dt, d, e);
  if (a >= 1e-4 && a < upper) {
    std::string out;
    if (e >= 0) {
      const auto ip = static_cast<std::size_t>(e) + 1;
      if (d.size() <= ip) {
        out = d + std::string(ip - d.size(), '0') + ".0";
      } else {
        out = d.substr(0, ip) + "." + d.substr(ip);
      }
    } else {
      out = "0." + std::string(static_cast<std::size_t>(-e - 1), '0') + d;
    }
    return sign + out;
  }
  std::string m = d.substr(0, 1);
  if (d.size() > 1) m += "." + d.substr(1);
  else if (force_point_sci) m += "";
  const int ae = e < 0 ? -e : e;
  std::string es = std::to_string(ae);
  if (es.size() < 2) es = "0" + es;
  return sign + m + "e" + (e < 0 ? "-" : "+") + es;
}

}  // namespace

std::string float_str(double v, DType dt) {
  const double upper = dt == DType::Float16 ? 1e3 : dt == DType::Float32 ? 1e6 : 1e16;
  return format_shortest(v, dt, upper, false);
}

std::string py_float_repr(double v) { return format_shortest(v, DType::Float64, 1e16, false); }

std::string py_complex_repr(double re, double im) {
  auto part = [](double x) {
    std::string s = py_float_repr(x);
    if (s.size() > 2 && s.compare(s.size() - 2, 2, ".0") == 0) s.resize(s.size() - 2);
    return s;
  };
  std::string ims = part(im);
  if (ims[0] != '-') ims = "+" + ims;
  if (re == 0 && !std::signbit(re)) return ims.substr(ims[0] == '+' ? 1 : 0) + "j";
  return "(" + part(re) + ims + "j)";
}

// ---------------- Python %-formatting ----------------

namespace {

struct Spec {
  std::string flags;
  int width = -1;
  int prec = -1;
  char conv = 's';
};

struct Piece {
  std::string text;  // literal when !is_spec
  bool is_spec = false;
  Spec spec;
};

std::vector<Piece> parse_format(const std::string& f) {
  std::vector<Piece> out;
  std::string lit;
  for (std::size_t i = 0; i < f.size();) {
    if (f[i] != '%') {
      lit += f[i++];
      continue;
    }
    if (i + 1 < f.size() && f[i + 1] == '%') {
      lit += '%';
      i += 2;
      continue;
    }
    ++i;
    Spec s;
    if (i < f.size() && f[i] == '(') value_error("format mapping keys are not supported: " + f);
    while (i < f.size() && std::strchr("-+ #0", f[i]) != nullptr) s.flags += f[i++];
    auto number = [&](int& dst) {
      if (i < f.size() && f[i] == '*') value_error("'*' width/precision is not supported: " + f);
      if (i < f.size() && is_digit(f[i])) {
        dst = 0;
        while (i < f.size() && is_digit(f[i])) dst = dst * 10 + (f[i++] - '0');
      }
    };
    number(s.width);
    if (i < f.size() && f[i] == '.') {
      ++i;
      s.prec = 0;
      number(s.prec);
    }
    while (i < f.size() && (f[i] == 'h' || f[i] == 'l' || f[i] == 'L')) ++i;
    if (i >= f.size()) value_error("incomplete format: " + f);
    s.conv = f[i++];
    if (std::strchr("diuoxXeEfFgGs", s.conv) == nullptr) {
      value_error(std::string("unsupported format character '") + s.conv + "' in " + f);
    }
    if (!lit.empty()) out.push_back({lit, false, {}});
    lit.clear();
    out.push_back({"", true, s});
  }
  if (!lit.empty()) out.push_back({lit, false, {}});
  return out;
}

enum class VK { Bool, Int, UInt, Float };
struct Val {
  VK kind;
  std::int64_t i = 0;
  std::uint64_t u = 0;
  double d = 0;
  DType dt = DType::Float64;
  bool py_scalar = false;  // %s uses Python float repr instead of NumPy str
};

bool has(const Spec& s, char c) { return s.flags.find(c) != std::string::npos; }

std::string pad(std::string sign_prefix, std::string body, const Spec& s, bool zero_ok) {
  const auto len = sign_prefix.size() + body.size();
  if (s.width < 0 || static_cast<std::size_t>(s.width) <= len) return sign_prefix + body;
  const std::size_t fill = static_cast<std::size_t>(s.width) - len;
  if (has(s, '-')) return sign_prefix + body + std::string(fill, ' ');
  if (has(s, '0') && zero_ok) return sign_prefix + std::string(fill, '0') + body;
  return std::string(fill, ' ') + sign_prefix + body;
}

// Exact decimal digits of a non-negative integral double.
std::string big_decimal(double a) {
  if (a < 9.2e18) return std::to_string(static_cast<std::uint64_t>(a));
  int e = 0;
  const double m = std::frexp(a, &e);
  auto mant = static_cast<std::uint64_t>(std::ldexp(m, 53));
  e -= 53;
  std::vector<std::uint32_t> limbs;  // base 1e9, little-endian
  while (mant > 0) {
    limbs.push_back(static_cast<std::uint32_t>(mant % 1000000000U));
    mant /= 1000000000U;
  }
  for (int k = 0; k < e; ++k) {
    std::uint32_t carry = 0;
    for (auto& l : limbs) {
      const std::uint64_t t = static_cast<std::uint64_t>(l) * 2 + carry;
      l = static_cast<std::uint32_t>(t % 1000000000U);
      carry = static_cast<std::uint32_t>(t / 1000000000U);
    }
    if (carry != 0) limbs.push_back(carry);
  }
  std::string out = std::to_string(limbs.back());
  for (std::size_t k = limbs.size() - 1; k-- > 0;) {
    std::string part = std::to_string(limbs[k]);
    out += std::string(9 - part.size(), '0') + part;
  }
  return out;
}

std::string to_base(std::uint64_t v, int base, bool upper) {
  if (v == 0) return "0";
  const char* digs = upper ? "0123456789ABCDEF" : "0123456789abcdef";
  std::string s;
  while (v > 0) {
    s += digs[v % static_cast<std::uint64_t>(base)];
    v /= static_cast<std::uint64_t>(base);
  }
  std::reverse(s.begin(), s.end());
  return s;
}

std::string format_value(const Spec& s, const Val& v, const std::string& fmt_for_errors, const char* dtype_name) {
  auto mismatch = [&]() -> std::string {
    throw_error(ErrorKind::DType, std::string("Mismatch between array dtype ('") + dtype_name +
                                      "') and format specifier ('" + fmt_for_errors + "')");
  };
  const char c = s.conv;
  if (c == 's') {
    std::string str;
    switch (v.kind) {
      case VK::Bool: str = v.u != 0 ? "True" : "False"; break;
      case VK::Int: str = std::to_string(v.i); break;
      case VK::UInt: str = std::to_string(v.u); break;
      case VK::Float: str = v.py_scalar ? py_float_repr(v.d) : float_str(v.d, v.dt); break;
    }
    if (s.prec >= 0 && static_cast<std::size_t>(s.prec) < str.size()) str.resize(static_cast<std::size_t>(s.prec));
    return pad("", str, s, false);
  }
  if (std::strchr("diuoxX", c) != nullptr) {
    bool neg = false;
    std::string digits;
    if (v.kind == VK::Float) {
      if (c == 'o' || c == 'x' || c == 'X') return mismatch();
      if (std::isnan(v.d)) value_error("cannot convert float NaN to integer");
      if (std::isinf(v.d)) throw_error(ErrorKind::Value, "cannot convert float infinity to integer");
      const double t = std::trunc(v.d);
      neg = t < 0;
      digits = big_decimal(std::fabs(t));
    } else {
      if (v.kind == VK::Bool && (c == 'o' || c == 'x' || c == 'X')) return mismatch();
      std::uint64_t mag = 0;
      if (v.kind == VK::Int) {
        neg = v.i < 0;
        mag = neg ? static_cast<std::uint64_t>(-(v.i + 1)) + 1 : static_cast<std::uint64_t>(v.i);
      } else {
        mag = v.u;
      }
      const int base = c == 'o' ? 8 : (c == 'x' || c == 'X') ? 16 : 10;
      digits = to_base(mag, base, c == 'X');
    }
    if (s.prec >= 0 && digits.size() < static_cast<std::size_t>(s.prec)) {
      digits = std::string(static_cast<std::size_t>(s.prec) - digits.size(), '0') + digits;
    }
    std::string prefix = neg ? "-" : has(s, '+') ? "+" : has(s, ' ') ? " " : "";
    if (has(s, '#')) {
      if (c == 'o') prefix += "0o";
      if (c == 'x') prefix += "0x";
      if (c == 'X') prefix += "0X";
    }
    return pad(prefix, digits, s, true);
  }
  // e E f F g G
  double d = 0;
  switch (v.kind) {
    case VK::Bool:
    case VK::UInt: d = static_cast<double>(v.u); break;
    case VK::Int: d = static_cast<double>(v.i); break;
    case VK::Float: d = v.d; break;
  }
  const bool upper = c == 'E' || c == 'F' || c == 'G';
  if (!std::isfinite(d)) {
    std::string body = std::isnan(d) ? "nan" : "inf";
    if (upper) std::transform(body.begin(), body.end(), body.begin(), [](char ch) { return static_cast<char>(ch - 32); });
    const std::string sign = (!std::isnan(d) && d < 0) ? "-" : has(s, '+') ? "+" : has(s, ' ') ? " " : "";
    return pad(sign, body, s, true);
  }
  std::string f = "%" + s.flags;
  if (s.width >= 0) f += std::to_string(s.width);
  f += "." + std::to_string(s.prec >= 0 ? s.prec : 6);
  f += c;
  const int n = std::snprintf(nullptr, 0, f.c_str(), d);
  std::string out(static_cast<std::size_t>(n) + 1, '\0');
  std::snprintf(out.data(), out.size(), f.c_str(), d);
  out.resize(static_cast<std::size_t>(n));
  return out;
}

Val value_at(const NDArray& a, std::int64_t flat, bool imag_part) {
  Val v{VK::Float};
  const DType dt = a.dtype();
  switch (dtype_info(dt).kind) {
    case 'b': v.kind = VK::Bool; v.u = a.get_uint64(flat); break;
    case 'i': v.kind = VK::Int; v.i = a.get_int64(flat); break;
    case 'u': v.kind = VK::UInt; v.u = a.get_uint64(flat); break;
    case 'f': v.d = a.get_double(flat); v.dt = dt; break;
    default: {
      const std::byte* p = a.data();
      std::int64_t off = 0;
      // element byte offset (C order) for complex
      std::int64_t rem = flat;
      for (std::size_t k = a.ndim(); k-- > 0;) {
        off += (rem % a.shape()[k]) * a.strides()[k];
        rem /= a.shape()[k];
      }
      if (dt == DType::Complex64) {
        float parts[2];
        std::memcpy(parts, p + off, 8);
        v.d = parts[imag_part ? 1 : 0];
        v.dt = DType::Float32;
      } else {
        double parts[2];
        std::memcpy(parts, p + off, 16);
        v.d = parts[imag_part ? 1 : 0];
        v.dt = DType::Float64;
      }
    }
  }
  return v;
}

std::string replace_all(std::string s, const std::string& from, const std::string& to) {
  std::size_t p = 0;
  while ((p = s.find(from, p)) != std::string::npos) {
    s.replace(p, from.size(), to);
    p += to.size();
  }
  return s;
}

}  // namespace

std::string format_rows(const NDArray& x, const std::string& row_format, const std::string& newline) {
  if (x.ndim() != 2) value_error("format_rows expects a 2-D array");
  const auto pieces = parse_format(row_format);
  const std::size_t nspec = static_cast<std::size_t>(
      std::count_if(pieces.begin(), pieces.end(), [](const Piece& p) { return p.is_spec; }));
  const bool cplx = is_complex(x.dtype());
  const std::int64_t rows = x.shape()[0], cols = x.shape()[1];
  const std::size_t per_row = static_cast<std::size_t>(cols) * (cplx ? 2 : 1);
  if (rows > 0 && nspec != per_row) {
    throw_error(ErrorKind::DType, nspec < per_row ? "not all arguments converted during string formatting"
                                                  : "not enough arguments for format string");
  }
  const char* dname = dtype_name(x.dtype()).data();
  std::string out;
  for (std::int64_t r = 0; r < rows; ++r) {
    std::string line;
    std::size_t k = 0;
    for (const Piece& p : pieces) {
      if (!p.is_spec) {
        line += p.text;
        continue;
      }
      const std::size_t col = cplx ? k / 2 : k;
      const Val v = value_at(x, r * cols + static_cast<std::int64_t>(col), cplx && k % 2 == 1);
      line += format_value(p.spec, v, row_format, dname);
      ++k;
    }
    line += newline;
    out += cplx ? replace_all(line, "+-", "-") : line;
  }
  return out;
}

std::string tofile_text(const NDArray& a, const std::string& sep, const std::string& format) {
  std::vector<Piece> pieces;
  if (!format.empty()) {
    pieces = parse_format(format);
    const auto n = std::count_if(pieces.begin(), pieces.end(), [](const Piece& p) { return p.is_spec; });
    if (n != 1) value_error("format must contain exactly one % conversion: " + format);
    if (is_complex(a.dtype())) throw_error(ErrorKind::DType, "format is not supported for complex arrays");
  }
  const char* dname = dtype_name(a.dtype()).data();
  std::string out;
  for (std::int64_t k = 0; k < a.size(); ++k) {
    if (k > 0) out += sep;
    if (is_complex(a.dtype())) {
      out += py_complex_repr(value_at(a, k, false).d, value_at(a, k, true).d);
      continue;
    }
    Val v = value_at(a, k, false);
    v.py_scalar = true;
    if (pieces.empty()) {
      Spec s;
      out += format_value(s, v, format, dname);
      continue;
    }
    for (const Piece& p : pieces) out += p.is_spec ? format_value(p.spec, v, format, dname) : p.text;
  }
  return out;
}

}  // namespace nativpy::p14
