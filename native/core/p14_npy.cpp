#include "p14_npy.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstring>
#include <optional>
#include <string_view>

#include "error.hpp"
#include "layout.hpp"

namespace nativpy::p14 {

namespace {

constexpr std::array<std::uint8_t, 6> kMagic{0x93, 'N', 'U', 'M', 'P', 'Y'};
constexpr std::size_t kAlign = 64;
constexpr std::size_t kGrowthDigits = 21;

[[noreturn]] void bad(const std::string& msg) { throw_error(ErrorKind::Value, msg); }

char type_char(DType dt) {
  switch (dtype_info(dt).kind) {
    case 'b': return 'b';
    case 'i': return 'i';
    case 'u': return 'u';
    case 'f': return 'f';
    default: return 'c';
  }
}

// Python repr of a shape tuple: "()", "(3,)", "(2, 3)".
std::string shape_repr(const Shape& s) {
  std::string out = "(";
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (i > 0) out += ", ";
    out += std::to_string(s[i]);
  }
  if (s.size() == 1) out += ",";
  return out + ")";
}

bool use_fortran(const NDArray& a) { return a.ndim() > 1 && a.is_f_contiguous() && !a.is_c_contiguous(); }

// ---- tiny parser for the Python dict literal in the header ----
struct Parser {
  std::string_view s;
  std::size_t i = 0;

  void ws() {
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])) != 0) ++i;
  }
  bool eat(char c) {
    ws();
    if (i < s.size() && s[i] == c) {
      ++i;
      return true;
    }
    return false;
  }
  void expect(char c) {
    if (!eat(c)) fail();
  }
  [[noreturn]] void fail() const {
    bad("Cannot parse header: " + std::string(s));
  }
  std::optional<std::string> str() {
    ws();
    if (i >= s.size() || (s[i] != '\'' && s[i] != '"')) return std::nullopt;
    const char q = s[i++];
    std::string out;
    while (i < s.size() && s[i] != q) {
      if (s[i] == '\\') fail();
      out += s[i++];
    }
    if (i >= s.size()) fail();
    ++i;
    return out;
  }
  std::string word() {
    ws();
    std::string out;
    while (i < s.size() && (std::isalnum(static_cast<unsigned char>(s[i])) != 0 || s[i] == '_')) out += s[i++];
    return out;
  }
  // Skips any value (used only to report unsupported descriptors).
  void skip_value() {
    ws();
    int depth = 0;
    while (i < s.size()) {
      const char c = s[i];
      if (c == '\'' || c == '"') {
        (void)str();
        continue;
      }
      if (c == '[' || c == '(' || c == '{') ++depth;
      if (c == ']' || c == ')' || c == '}') {
        if (depth == 0) return;
        --depth;
      }
      if (c == ',' && depth == 0) return;
      ++i;
    }
  }
};

struct Header {
  std::string descr;
  bool descr_is_str = false;
  bool fortran = false;
  Shape shape;
};

Header parse_header(std::string_view text) {
  Parser p{text};
  Header h;
  bool has_descr = false, has_fortran = false, has_shape = false;
  std::vector<std::string> keys;
  p.expect('{');
  while (!p.eat('}')) {
    const auto key = p.str();
    if (!key) p.fail();
    p.expect(':');
    keys.push_back(*key);
    if (*key == "descr") {
      has_descr = true;
      const auto v = p.str();
      if (v) {
        h.descr = *v;
        h.descr_is_str = true;
      } else {
        p.skip_value();
      }
    } else if (*key == "fortran_order") {
      has_fortran = true;
      const std::string w = p.word();
      if (w == "True") {
        h.fortran = true;
      } else if (w == "False") {
        h.fortran = false;
      } else {
        bad("fortran_order is not a valid bool: " + w);
      }
    } else if (*key == "shape") {
      has_shape = true;
      p.expect('(');
      while (!p.eat(')')) {
        const std::string w = p.word();
        if (w.empty() || !std::all_of(w.begin(), w.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }) ||
            w.size() > 18) {
          bad("shape is not valid: " + std::string(text));
        }
        h.shape.push_back(std::stoll(w));
        if (!p.eat(',')) {
          p.expect(')');
          break;
        }
      }
    } else {
      p.skip_value();
    }
    if (!p.eat(',')) {
      p.expect('}');
      break;
    }
  }
  if (!has_descr || !has_fortran || !has_shape || keys.size() != 3) {
    std::sort(keys.begin(), keys.end());
    std::string list = "[";
    for (std::size_t k = 0; k < keys.size(); ++k) list += (k ? ", '" : "'") + keys[k] + "'";
    bad("Header does not contain the correct keys: " + list + "]");
  }
  return h;
}

struct Descr {
  DType dtype;
  bool swap;
};

Descr parse_descr(const Header& h) {
  if (!h.descr_is_str) {
    throw_error(ErrorKind::DType, "structured (record) dtypes are not supported by load");
  }
  const std::string& d = h.descr;
  if (d == "|O" || d == "O") {
    bad("This file contains pickled (object) data, which numera cannot load");
  }
  if (d.size() < 2) throw_error(ErrorKind::DType, "data type '" + d + "' not understood");
  char order = '=';
  std::string_view rest = d;
  if (d[0] == '<' || d[0] == '>' || d[0] == '|' || d[0] == '=') {
    order = d[0];
    rest = rest.substr(1);
  }
  if (rest.empty()) throw_error(ErrorKind::DType, "data type '" + d + "' not understood");
  const char kind = rest[0];
  std::size_t size = 0;
  const std::string_view digits = rest.substr(1);
  if (digits.empty() || digits.size() > 3 ||
      !std::all_of(digits.begin(), digits.end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; })) {
    throw_error(ErrorKind::DType, "data type '" + d + "' not understood by numera's npy reader");
  }
  size = static_cast<std::size_t>(std::stoul(std::string(digits)));
  for (int k = 0; k < kNumDTypes; ++k) {
    const auto dt = static_cast<DType>(k);
    if (type_char(dt) == kind && itemsize(dt) == size) {
      const bool little = std::endian::native == std::endian::little;
      const bool swap = size > 1 && ((order == '>' && little) || (order == '<' && !little));
      return {dt, swap};
    }
  }
  throw_error(ErrorKind::DType, "data type '" + d + "' is not supported by numera");
}

void byteswap_items(std::byte* p, std::size_t n, DType dt) {
  std::size_t width = itemsize(dt);
  std::size_t count = n;
  if (is_complex(dt)) {
    width /= 2;
    count *= 2;
  }
  for (std::size_t k = 0; k < count; ++k) std::reverse(p + k * width, p + (k + 1) * width);
}

}  // namespace

std::string npy_descr(DType dt) {
  const std::size_t n = itemsize(dt);
  const char order = n == 1 ? '|' : (std::endian::native == std::endian::little ? '<' : '>');
  return std::string(1, order) + type_char(dt) + std::to_string(n);
}

std::string npy_header_dict(const NDArray& a) {
  const bool f = use_fortran(a);
  std::string h = "{'descr': '" + npy_descr(a.dtype()) + "', 'fortran_order': " + (f ? "True" : "False") +
                  ", 'shape': " + shape_repr(a.shape()) + ", }";
  if (a.ndim() > 0) {
    const std::int64_t growth = a.shape()[f ? a.ndim() - 1 : 0];
    const std::size_t len = std::to_string(growth).size();
    if (len < kGrowthDigits) h.append(kGrowthDigits - len, ' ');
  }
  return h;
}

std::vector<std::uint8_t> npy_encode(const NDArray& a) {
  const std::string dict = npy_header_dict(a);
  const std::size_t hlen = dict.size() + 1;
  std::uint8_t major = 1;
  std::size_t prefix = kMagic.size() + 2 + 2;
  std::size_t pad = kAlign - ((prefix + hlen) % kAlign);
  if (hlen + pad > 65535) {
    major = 2;
    prefix = kMagic.size() + 2 + 4;
    pad = kAlign - ((prefix + hlen) % kAlign);
  }
  const std::size_t total_h = hlen + pad;
  const bool f = use_fortran(a);
  const NDArray c = copy_order(a, a.dtype(), f ? Order::F : Order::C);
  const auto nbytes = static_cast<std::size_t>(c.nbytes());
  std::vector<std::uint8_t> out;
  out.reserve(prefix + total_h + nbytes);
  out.insert(out.end(), kMagic.begin(), kMagic.end());
  out.push_back(major);
  out.push_back(0);
  if (major == 1) {
    out.push_back(static_cast<std::uint8_t>(total_h & 0xff));
    out.push_back(static_cast<std::uint8_t>((total_h >> 8) & 0xff));
  } else {
    for (int k = 0; k < 4; ++k) out.push_back(static_cast<std::uint8_t>((total_h >> (8 * k)) & 0xff));
  }
  out.insert(out.end(), dict.begin(), dict.end());
  out.insert(out.end(), pad, static_cast<std::uint8_t>(' '));
  out.push_back('\n');
  if (nbytes > 0) {
    const auto* src = reinterpret_cast<const std::uint8_t*>(c.data());
    out.insert(out.end(), src, src + nbytes);
  }
  return out;
}

NDArray npy_decode(const std::uint8_t* data, std::size_t len) {
  if (len == 0) bad("No data left in file");
  if (len < kMagic.size() + 2 || !std::equal(kMagic.begin(), kMagic.end(), data)) {
    bad("the data is not in the NumPy .npy format (magic string mismatch)");
  }
  const std::uint8_t major = data[6];
  const std::uint8_t minor = data[7];
  if (minor != 0 || major < 1 || major > 3) {
    bad("we only support format version (1,0), (2,0), and (3,0), not (" + std::to_string(major) + ", " +
        std::to_string(minor) + ")");
  }
  const std::size_t lenbytes = major == 1 ? 2 : 4;
  std::size_t pos = 8;
  if (len < pos + lenbytes) {
    bad("EOF: reading array header length, expected " + std::to_string(lenbytes) + " bytes got " +
        std::to_string(len - pos));
  }
  std::size_t hlen = 0;
  for (std::size_t k = 0; k < lenbytes; ++k) hlen |= static_cast<std::size_t>(data[pos + k]) << (8 * k);
  pos += lenbytes;
  if (len - pos < hlen) {
    bad("EOF: reading array header, expected " + std::to_string(hlen) + " bytes got " + std::to_string(len - pos));
  }
  // v3 headers are UTF-8, v1/v2 latin-1; the dict we accept is ASCII either way.
  const std::string text(reinterpret_cast<const char*>(data + pos), hlen);
  pos += hlen;
  const Header h = parse_header(text);
  const Descr d = parse_descr(h);
  NDArray out = empty_order(h.shape, d.dtype, h.fortran ? Order::F : Order::C);
  const auto nbytes = static_cast<std::size_t>(out.nbytes());
  if (len - pos < nbytes) {
    bad("EOF: reading array data, expected " + std::to_string(nbytes) + " bytes got " + std::to_string(len - pos));
  }
  if (nbytes > 0) {
    std::memcpy(out.data(), data + pos, nbytes);
    if (d.swap) byteswap_items(out.data(), static_cast<std::size_t>(out.size()), d.dtype);
    if (d.dtype == DType::Bool) {
      auto* b = reinterpret_cast<std::uint8_t*>(out.data());
      for (std::size_t k = 0; k < nbytes; ++k) b[k] = b[k] != 0 ? 1 : 0;
    }
  }
  return out;
}

std::uint32_t crc32(const std::uint8_t* data, std::size_t len, std::uint32_t crc) {
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t n = 0; n < 256; ++n) {
      std::uint32_t c = n;
      for (int k = 0; k < 8; ++k) c = (c & 1U) != 0 ? 0xEDB88320U ^ (c >> 1) : c >> 1;
      t[n] = c;
    }
    return t;
  }();
  crc = ~crc;
  for (std::size_t k = 0; k < len; ++k) crc = table[(crc ^ data[k]) & 0xffU] ^ (crc >> 8);
  return ~crc;
}

}  // namespace nativpy::p14
