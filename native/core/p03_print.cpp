#include "p03_print.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstring>
#include <limits>

#include "error.hpp"
#include "p03_dtypes.hpp"

namespace nativpy {

namespace {

// ---------------------------------------------------------------------------
// Minimal unsigned big integer (little-endian 32-bit blocks, no leading zero
// blocks) for the Dragon4 port.
class BigInt {
 public:
  BigInt() = default;
  explicit BigInt(std::uint64_t v) { set(v); }

  void set(std::uint64_t v) {
    b_.clear();
    while (v != 0) {
      b_.push_back(static_cast<std::uint32_t>(v));
      v >>= 32;
    }
  }
  [[nodiscard]] bool zero() const noexcept { return b_.empty(); }
  [[nodiscard]] bool even() const noexcept { return b_.empty() || (b_[0] & 1u) == 0; }

  static BigInt pow2(int n) {
    BigInt r(1);
    r.shl(n);
    return r;
  }
  static BigInt pow10(int n) {
    BigInt r(1);
    while (n >= 9) {
      r.mul_small(1000000000u);
      n -= 9;
    }
    static constexpr std::uint32_t p[] = {1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000};
    if (n > 0) r.mul_small(p[n]);
    return r;
  }

  void shl(int n) {
    if (zero() || n == 0) return;
    const auto blocks = static_cast<std::size_t>(n / 32);
    const int bits = n % 32;
    std::vector<std::uint32_t> out(blocks, 0);
    std::uint32_t carry = 0;
    for (const std::uint32_t x : b_) {
      if (bits == 0) {
        out.push_back(x);
      } else {
        out.push_back((x << bits) | carry);
        carry = x >> (32 - bits);
      }
    }
    if (carry != 0) out.push_back(carry);
    b_ = std::move(out);
  }

  void mul_small(std::uint32_t k) {
    std::uint64_t carry = 0;
    for (std::uint32_t& x : b_) {
      const std::uint64_t t = static_cast<std::uint64_t>(x) * k + carry;
      x = static_cast<std::uint32_t>(t);
      carry = t >> 32;
    }
    if (carry != 0) b_.push_back(static_cast<std::uint32_t>(carry));
    trim();
  }

  void mul(const BigInt& o) {
    if (zero() || o.zero()) {
      b_.clear();
      return;
    }
    std::vector<std::uint32_t> out(b_.size() + o.b_.size(), 0);
    for (std::size_t i = 0; i < b_.size(); ++i) {
      std::uint64_t carry = 0;
      for (std::size_t j = 0; j < o.b_.size(); ++j) {
        const std::uint64_t t = static_cast<std::uint64_t>(b_[i]) * o.b_[j] + out[i + j] + carry;
        out[i + j] = static_cast<std::uint32_t>(t);
        carry = t >> 32;
      }
      out[i + o.b_.size()] = static_cast<std::uint32_t>(carry);
    }
    b_ = std::move(out);
    trim();
  }

  [[nodiscard]] BigInt plus(const BigInt& o) const {
    BigInt r;
    const std::size_t n = std::max(b_.size(), o.b_.size());
    r.b_.resize(n + 1, 0);
    std::uint64_t carry = 0;
    for (std::size_t i = 0; i < n; ++i) {
      const std::uint64_t t = static_cast<std::uint64_t>(i < b_.size() ? b_[i] : 0) +
                              (i < o.b_.size() ? o.b_[i] : 0) + carry;
      r.b_[i] = static_cast<std::uint32_t>(t);
      carry = t >> 32;
    }
    r.b_[n] = static_cast<std::uint32_t>(carry);
    r.trim();
    return r;
  }

  // this -= o (requires this >= o)
  void sub(const BigInt& o) {
    std::int64_t borrow = 0;
    for (std::size_t i = 0; i < b_.size(); ++i) {
      std::int64_t t = static_cast<std::int64_t>(b_[i]) - (i < o.b_.size() ? o.b_[i] : 0) - borrow;
      borrow = t < 0 ? 1 : 0;
      if (t < 0) t += std::int64_t{1} << 32;
      b_[i] = static_cast<std::uint32_t>(t);
    }
    trim();
  }

  [[nodiscard]] int cmp(const BigInt& o) const noexcept {
    if (b_.size() != o.b_.size()) return b_.size() < o.b_.size() ? -1 : 1;
    for (std::size_t i = b_.size(); i-- > 0;) {
      if (b_[i] != o.b_[i]) return b_[i] < o.b_[i] ? -1 : 1;
    }
    return 0;
  }

  // Quotient (< 10 by construction) of this / d; this becomes the remainder.
  std::uint32_t divmod_small_quotient(const BigInt& d) {
    std::uint32_t q = 0;
    while (cmp(d) >= 0) {
      sub(d);
      ++q;
    }
    return q;
  }

 private:
  void trim() {
    while (!b_.empty() && b_.back() == 0) b_.pop_back();
  }
  std::vector<std::uint32_t> b_;
};

// NumPy dragon4.c `Dragon4`: decimal digits of mantissa * 2^exponent.
// Returns the digits; `out_exp` is the base-10 exponent of the first digit.
std::string dragon4_digits(const BigInt& mantissa, int exponent, int mantissa_bit, bool unequal_margins,
                           bool unique, bool total_length, int cutoff_max, int cutoff_min, int buffer_size,
                           int& out_exp) {
  std::string out;
  if (mantissa.zero()) {
    out_exp = 0;
    return "0";
  }
  BigInt value = mantissa;
  BigInt scale;
  BigInt margin_low;
  if (unequal_margins) {
    if (exponent > 0) {
      value.shl(exponent + 2);
      scale.set(4);
      margin_low = BigInt::pow2(exponent);
    } else {
      value.shl(2);
      scale = BigInt::pow2(-exponent + 2);
      margin_low.set(1);
    }
  } else {
    if (exponent > 0) {
      value.shl(exponent + 1);
      scale.set(2);
      margin_low = BigInt::pow2(exponent);
    } else {
      value.shl(1);
      scale = BigInt::pow2(-exponent + 1);
      margin_low.set(1);
    }
  }
  const auto margin_high = [&] {
    if (!unequal_margins) return margin_low;
    BigInt h = margin_low;
    h.shl(1);
    return h;
  };

  constexpr double log10_2 = 0.30102999566398119521373889472449;
  int digit_exp = static_cast<int>(std::ceil(static_cast<double>(mantissa_bit + exponent) * log10_2 - 0.69));
  if (cutoff_max >= 0 && !total_length && digit_exp <= -cutoff_max) digit_exp = -cutoff_max + 1;

  if (digit_exp > 0) {
    scale.mul(BigInt::pow10(digit_exp));
  } else if (digit_exp < 0) {
    const BigInt p = BigInt::pow10(-digit_exp);
    value.mul(p);
    margin_low.mul(p);
  }
  if (value.cmp(scale) >= 0) {
    digit_exp += 1;
  } else {
    value.mul_small(10);
    margin_low.mul_small(10);
  }

  int cutoff_max_exp = digit_exp - buffer_size;
  if (cutoff_max >= 0) {
    const int desired = total_length ? digit_exp - cutoff_max : -cutoff_max;
    cutoff_max_exp = std::max(cutoff_max_exp, desired);
  }
  int cutoff_min_exp = digit_exp;
  if (cutoff_min >= 0) {
    const int desired = total_length ? digit_exp - cutoff_min : -cutoff_min;
    cutoff_min_exp = std::min(cutoff_min_exp, desired);
  }
  out_exp = digit_exp - 1;

  const bool is_even = mantissa.even();
  bool low = false;
  bool high = false;
  std::uint32_t digit = 0;
  if (unique) {
    for (;;) {
      digit_exp -= 1;
      digit = value.divmod_small_quotient(scale);
      const BigInt value_high = value.plus(margin_high());
      const int cl = value.cmp(margin_low);
      low = is_even ? cl <= 0 : cl < 0;
      const int ch = value_high.cmp(scale);
      high = is_even ? ch >= 0 : ch > 0;
      if (((low || high) && digit_exp <= cutoff_min_exp) || digit_exp == cutoff_max_exp) break;
      out.push_back(static_cast<char>('0' + digit));
      value.mul_small(10);
      margin_low.mul_small(10);
    }
  } else {
    for (;;) {
      digit_exp -= 1;
      digit = value.divmod_small_quotient(scale);
      if (value.zero() || digit_exp == cutoff_max_exp) break;
      out.push_back(static_cast<char>('0' + digit));
      value.mul_small(10);
    }
  }

  bool round_down = low;
  if (low == high) {
    value.shl(1);
    const int c = value.cmp(scale);
    round_down = c < 0;
    if (c == 0) round_down = (digit & 1u) == 0;
  }
  if (round_down) {
    out.push_back(static_cast<char>('0' + digit));
  } else if (digit == 9) {
    for (;;) {
      if (out.empty()) {
        out.push_back('1');
        out_exp += 1;
        break;
      }
      const char c = out.back();
      out.pop_back();
      if (c != '9') {
        out.push_back(static_cast<char>(c + 1));
        break;
      }
    }
  } else {
    out.push_back(static_cast<char>('0' + digit + 1));
  }
  return out;
}

constexpr int kBufferSize = 16384;

std::string format_positional(const BigInt& mantissa, int exponent, char signbit, int mantissa_bit,
                              bool unequal, const Dragon4Options& o) {
  std::string s;
  int has_sign = 0;
  if (signbit != '\0') {
    s.push_back(signbit);
    has_sign = 1;
  }
  int pe = 0;
  const std::string digits = dragon4_digits(mantissa, exponent, mantissa_bit, unequal, o.unique, !o.fractional,
                                            o.precision, o.min_digits, kBufferSize - has_sign, pe);
  const int num_digits = static_cast<int>(digits.size());
  int whole = 0;
  int frac = 0;
  if (pe >= 0) {
    whole = pe + 1;
    if (num_digits <= whole) {
      s += digits;
      s.append(static_cast<std::size_t>(whole - num_digits), '0');
    } else {
      s += digits.substr(0, static_cast<std::size_t>(whole));
      s.push_back('.');
      s += digits.substr(static_cast<std::size_t>(whole));
      frac = num_digits - whole;
    }
  } else {
    const int zeros = -(pe + 1);
    s += "0.";
    s.append(static_cast<std::size_t>(zeros), '0');
    s += digits;
    frac = zeros + num_digits;
    whole = 1;
  }
  if (o.trim != TrimMode::DptZeros && frac == 0) s.push_back('.');

  const int add_digits = o.unique ? o.min_digits : o.precision;
  int desired = add_digits < 0 ? 0 : add_digits;
  if (!o.fractional) desired = add_digits - whole;

  if (o.trim == TrimMode::LeaveOneZero) {
    if (frac == 0) {
      s.push_back('0');
      ++frac;
    }
  } else if (o.trim == TrimMode::None && desired > frac) {
    s.append(static_cast<std::size_t>(desired - frac), '0');
    frac = desired;
  }
  if (o.trim != TrimMode::None && frac > 0) {
    while (s.back() == '0') {
      s.pop_back();
      --frac;
    }
    if (s.back() == '.') {
      if (o.trim == TrimMode::LeaveOneZero) {
        s.push_back('0');
        ++frac;
      } else if (o.trim == TrimMode::DptZeros) {
        s.pop_back();
      }
    }
  }
  if (o.pad_right >= frac) {
    if (o.trim == TrimMode::DptZeros && frac == 0) s.push_back(' ');
    s.append(static_cast<std::size_t>(o.pad_right - frac), ' ');
  }
  if (o.pad_left > whole + has_sign) s.insert(0, static_cast<std::size_t>(o.pad_left - (whole + has_sign)), ' ');
  return s;
}

std::string format_scientific(const BigInt& mantissa, int exponent, char signbit, int mantissa_bit, bool unequal,
                              const Dragon4Options& o) {
  std::string s;
  const int leftchars = 1 + (signbit != '\0' ? 1 : 0);
  if (o.pad_left > leftchars) s.append(static_cast<std::size_t>(o.pad_left - leftchars), ' ');
  if (signbit != '\0') s.push_back(signbit);
  int pe = 0;
  const std::string digits =
      dragon4_digits(mantissa, exponent, mantissa_bit, unequal, o.unique, true, o.precision < 0 ? -1 : o.precision + 1,
                     o.min_digits < 0 ? -1 : o.min_digits + 1, kBufferSize, pe);
  s.push_back(digits[0]);
  int frac = static_cast<int>(digits.size()) - 1;
  if (frac > 0) {
    s.push_back('.');
    s += digits.substr(1);
  }
  if (o.trim != TrimMode::DptZeros && frac == 0) s.push_back('.');
  int add_digits = o.unique ? o.min_digits : o.precision;
  add_digits = add_digits < 0 ? 0 : add_digits;
  if (o.trim == TrimMode::LeaveOneZero) {
    if (frac == 0) {
      s.push_back('0');
      ++frac;
    }
  } else if (o.trim == TrimMode::None && add_digits > frac) {
    s.append(static_cast<std::size_t>(add_digits - frac), '0');
    frac = add_digits;
  }
  if (o.trim != TrimMode::None && frac > 0) {
    while (s.back() == '0') s.pop_back();
    if (o.trim == TrimMode::LeaveOneZero && s.back() == '.') s.push_back('0');
  }
  int exp_digits = o.exp_digits;
  if (exp_digits > 5) exp_digits = 5;
  if (exp_digits < 0) exp_digits = 2;
  s.push_back('e');
  s.push_back(pe >= 0 ? '+' : '-');
  int e = pe < 0 ? -pe : pe;
  int d[5];
  for (int& x : d) {
    x = e % 10;
    e /= 10;
  }
  int size = 5;
  while (size > exp_digits && d[size - 1] == 0) --size;
  for (int i = size; i > 0; --i) s.push_back(static_cast<char>('0' + d[i - 1]));
  return s;
}

std::string format_float_bits(BigInt mantissa, int exponent, char signbit, int mantissa_bit, bool unequal,
                              const Dragon4Options& o) {
  return o.scientific ? format_scientific(mantissa, exponent, signbit, mantissa_bit, unequal, o)
                      : format_positional(mantissa, exponent, signbit, mantissa_bit, unequal, o);
}

std::string inf_nan(bool is_inf, char signbit) {
  if (!is_inf) return "nan";
  std::string s;
  if (signbit != '\0') s.push_back(signbit);
  return s + "inf";
}

int log2_floor(std::uint64_t v) {
  int r = -1;
  while (v != 0) {
    v >>= 1;
    ++r;
  }
  return r;
}

// Decomposes an IEEE value with `mbits` mantissa bits and `ebits` exponent
// bits (NumPy Dragon4_PrintFloat_IEEE_binary*).
std::string format_ieee(std::uint64_t bits, int mbits, int ebits, const Dragon4Options& o) {
  const std::uint64_t fmant = bits & ((std::uint64_t{1} << mbits) - 1);
  const std::uint64_t fexp = (bits >> mbits) & ((std::uint64_t{1} << ebits) - 1);
  const bool neg = ((bits >> (mbits + ebits)) & 1u) != 0;
  const char signbit = neg ? '-' : (o.sign ? '+' : '\0');
  const std::uint64_t emax = (std::uint64_t{1} << ebits) - 1;
  if (fexp == emax) return inf_nan(fmant == 0, signbit);
  const int bias = static_cast<int>((std::uint64_t{1} << (ebits - 1)) - 1);
  std::uint64_t mantissa = 0;
  int exponent = 0;
  int mantissa_bit = 0;
  bool unequal = false;
  if (fexp != 0) {
    mantissa = (std::uint64_t{1} << mbits) | fmant;
    exponent = static_cast<int>(fexp) - bias - mbits;
    mantissa_bit = mbits;
    unequal = fexp != 1 && fmant == 0;
  } else {
    mantissa = fmant;
    exponent = 1 - bias - mbits;
    mantissa_bit = log2_floor(mantissa);
  }
  return format_float_bits(BigInt(mantissa), exponent, signbit, mantissa_bit, unequal, o);
}

// ---------------------------------------------------------------------------
// Element formatting (NumPy arrayprint FloatingFormat & co.)

Dragon4Options pos_opts(int precision, int min_digits, bool unique, TrimMode trim, bool sign, int pad_left = -1,
                        int pad_right = -1) {
  Dragon4Options o;
  o.precision = precision;
  o.min_digits = min_digits;
  o.unique = unique;
  o.trim = trim;
  o.sign = sign;
  o.pad_left = pad_left;
  o.pad_right = pad_right;
  return o;
}

class FloatFormat {
 public:
  FloatFormat(const std::vector<double>& data, DType dt, const PrintOptions& po, char sign)
      : dt_(dt), sign_(sign), nanstr_(po.nanstr), infstr_(po.infstr), floatmode_(po.floatmode) {
    precision_ = po.floatmode == "unique" ? -1 : po.precision;
    std::vector<double> finite;
    for (const double v : data)
      if (std::isfinite(v)) finite.push_back(v);
    double max_v = 0;
    double min_v = std::numeric_limits<double>::infinity();
    bool any_nz = false;
    for (const double v : finite) {
      if (v == 0) continue;
      any_nz = true;
      max_v = std::max(max_v, std::fabs(v));
      min_v = std::min(min_v, std::fabs(v));
    }
    if (any_nz) {
      const double cutoff = std::pow(10.0, std::min(8, float_info(dt).precision));
      if (max_v >= cutoff || (!po.suppress && (min_v < 0.0001 || max_v / min_v > 1000.))) exp_format_ = true;
    }
    const bool fixed = floatmode_ == "fixed";
    if (finite.empty()) {
      pad_left_ = 0;
      pad_right_ = 0;
      trim_ = TrimMode::Zeros;
      exp_size_ = -1;
      unique_ = true;
      min_digits_ = -1;
    } else if (exp_format_) {
      const TrimMode trim = fixed ? TrimMode::None : TrimMode::Zeros;
      unique_ = !fixed;
      std::size_t exp_len = 0;
      std::size_t frac_len = 0;
      std::size_t int_len = 0;
      for (const double v : finite) {
        Dragon4Options o = pos_opts(precision_, -1, unique_, trim, sign_ == '+');
        o.scientific = true;
        const std::string s = dragon4(v, dt_, o);
        const std::size_t e = s.find('e');
        const std::size_t dot = s.find('.');
        exp_len = std::max(exp_len, s.size() - e - 1);
        int_len = std::max(int_len, dot);
        frac_len = std::max(frac_len, e - dot - 1);
      }
      exp_size_ = static_cast<int>(exp_len) - 1;
      trim_ = TrimMode::None;
      precision_ = static_cast<int>(frac_len);
      min_digits_ = precision_;
      pad_left_ = static_cast<int>(int_len);
      pad_right_ = exp_size_ + 2 + precision_;
    } else {
      const TrimMode trim = fixed ? TrimMode::None : TrimMode::Zeros;
      unique_ = !fixed;
      std::size_t int_len = 0;
      std::size_t frac_len = 0;
      for (const double v : finite) {
        const std::string s = dragon4(v, dt_, pos_opts(precision_, -1, unique_, trim, sign_ == '+'));
        const std::size_t dot = s.find('.');
        int_len = std::max(int_len, dot);
        frac_len = std::max(frac_len, s.size() - dot - 1);
      }
      pad_left_ = static_cast<int>(int_len);
      pad_right_ = static_cast<int>(frac_len);
      exp_size_ = -1;
      if (fixed || floatmode_ == "maxprec_equal") {
        precision_ = min_digits_ = pad_right_;
        trim_ = TrimMode::None;
      } else {
        trim_ = TrimMode::Zeros;
        min_digits_ = 0;
      }
    }
    if (sign_ == ' ' && std::none_of(finite.begin(), finite.end(), [](double v) { return std::signbit(v); })) {
      pad_left_ += 1;
    }
    if (finite.size() != data.size()) {
      const bool neginf =
          sign_ != '-' || std::any_of(data.begin(), data.end(), [](double v) { return std::isinf(v) && v < 0; });
      const int offset = pad_right_ + 1;
      pad_left_ = std::max({pad_left_, static_cast<int>(nanstr_.size()) - offset,
                            static_cast<int>(infstr_.size()) + (neginf ? 1 : 0) - offset});
    }
  }

  [[nodiscard]] std::string operator()(double x) const {
    if (!std::isfinite(x)) {
      std::string r;
      if (std::isnan(x)) {
        r = (sign_ == '+' ? "+" : "") + nanstr_;
      } else {
        r = (x < 0 ? "-" : sign_ == '+' ? "+" : "") + infstr_;
      }
      const int pad = pad_left_ + pad_right_ + 1 - static_cast<int>(r.size());
      return std::string(static_cast<std::size_t>(std::max(0, pad)), ' ') + r;
    }
    Dragon4Options o = pos_opts(precision_, min_digits_, unique_, trim_, sign_ == '+', pad_left_);
    if (exp_format_) {
      o.scientific = true;
      o.exp_digits = exp_size_;
    } else {
      o.pad_right = pad_right_;
    }
    return dragon4(x, dt_, o);
  }

 private:
  DType dt_;
  char sign_;
  std::string nanstr_, infstr_, floatmode_;
  int precision_ = -1;
  int min_digits_ = -1;
  int pad_left_ = 0;
  int pad_right_ = 0;
  int exp_size_ = -1;
  bool exp_format_ = false;
  bool unique_ = true;
  TrimMode trim_ = TrimMode::Zeros;
};

DType component(DType dt) {
  if (dt == DType::Complex64) return DType::Float32;
  if (dt == DType::Complex128) return DType::Float64;
  return dt;
}

std::vector<double> as_doubles(const NDArray& a) {
  const NDArray c = a.astype(DType::Float64);
  std::vector<double> v(static_cast<std::size_t>(c.size()));
  if (!v.empty()) std::memcpy(v.data(), c.data(), v.size() * sizeof(double));
  return v;
}

std::vector<std::complex<double>> as_complex(const NDArray& a) {
  const NDArray c = a.astype(DType::Complex128);
  std::vector<std::complex<double>> v(static_cast<std::size_t>(c.size()));
  if (!v.empty()) std::memcpy(static_cast<void*>(v.data()), c.data(), v.size() * sizeof(std::complex<double>));
  return v;
}

template <typename T>
std::string fmt_int(T v, char sign, std::size_t width) {
  std::string s = std::to_string(v);
  if (v >= 0 && sign == '+') s.insert(0, "+");
  if (v >= 0 && sign == ' ') s.insert(0, " ");
  if (s.size() < width) s.insert(0, width - s.size(), ' ');
  return s;
}

template <typename T>
std::vector<std::string> format_ints(const std::vector<T>& v, char sign) {
  std::size_t width = 0;
  if (!v.empty()) {
    const T mx = *std::max_element(v.begin(), v.end());
    const T mn = *std::min_element(v.begin(), v.end());
    if (sign == ' ' && mn < 0) sign = '-';
    std::size_t mx_len = std::to_string(mx).size();
    if (mx >= 0 && (sign == '+' || sign == ' ')) mx_len += 1;
    width = std::max(mx_len, std::to_string(mn).size());
  }
  std::vector<std::string> out;
  out.reserve(v.size());
  for (const T x : v) out.push_back(fmt_int(x, sign, width));
  return out;
}

// NumPy floattype_str_either: positional within [1e-4, max_positional).
std::string float_either(double v, DType dt, TrimMode trim_pos, TrimMode trim_sci, bool sign) {
  const double max_pos = dt == DType::Float16 ? 1e3 : dt == DType::Float32 ? 1e6 : 1e16;
  const double a = std::fabs(v);
  const bool positional = std::isnan(v) || v == 0 || (a < max_pos && a >= 1e-4);
  Dragon4Options o = pos_opts(-1, -1, true, positional ? trim_pos : trim_sci, sign);
  o.scientific = !positional;
  return dragon4(v, dt, o);
}

}  // namespace

std::string dragon4(double value, DType dt, const Dragon4Options& opt) {
  switch (dt) {
    case DType::Float16:
      return format_ieee(double_to_half(value).bits, 10, 5, opt);
    case DType::Float32: {
      const float f = static_cast<float>(value);
      std::uint32_t b = 0;
      std::memcpy(&b, &f, sizeof b);
      return format_ieee(b, 23, 8, opt);
    }
    case DType::Float64: {
      std::uint64_t b = 0;
      std::memcpy(&b, &value, sizeof b);
      return format_ieee(b, 52, 11, opt);
    }
    default:
      throw_error(ErrorKind::DType, "dragon4 requires a float dtype");
  }
}

NDArray leading_trailing(const NDArray& a, std::int64_t edgeitems) {
  const std::size_t nd = a.ndim();
  std::vector<std::vector<std::int64_t>> keep(nd);
  Shape shape(nd);
  for (std::size_t d = 0; d < nd; ++d) {
    const std::int64_t n = a.shape()[d];
    if (n > 2 * edgeitems) {
      for (std::int64_t i = 0; i < edgeitems; ++i) keep[d].push_back(i);
      for (std::int64_t i = n - edgeitems; i < n; ++i) keep[d].push_back(i);
    } else {
      for (std::int64_t i = 0; i < n; ++i) keep[d].push_back(i);
    }
    shape[d] = static_cast<std::int64_t>(keep[d].size());
  }
  NDArray out = NDArray::empty(shape, a.dtype());
  if (out.size() == 0) return out;
  const std::size_t isz = a.itemsize();
  std::vector<std::size_t> pos(nd, 0);
  std::byte* dst = out.data();
  for (std::int64_t k = 0; k < out.size(); ++k) {
    std::int64_t off = a.offset();
    for (std::size_t d = 0; d < nd; ++d) off += keep[d][pos[d]] * a.strides()[d];
    std::memcpy(dst + static_cast<std::size_t>(k) * isz, a.buffer()->data() + off, isz);
    for (std::size_t d = nd; d-- > 0;) {
      if (++pos[d] < keep[d].size()) break;
      pos[d] = 0;
    }
  }
  return out;
}

std::vector<std::string> format_elements(const NDArray& data, const PrintOptions& opt) {
  const DType dt = data.dtype();
  const char kind = dtype_info(dt).kind;
  std::vector<std::string> out;
  if (kind == 'b') {
    const std::vector<double> v = as_doubles(data);
    const std::string t = data.ndim() != 0 ? " True" : "True";
    for (const double x : v) out.push_back(x != 0 ? t : "False");
    return out;
  }
  if (kind == 'u') {
    const NDArray c = data.astype(DType::UInt64);
    std::vector<std::uint64_t> v(static_cast<std::size_t>(c.size()));
    if (!v.empty()) std::memcpy(v.data(), c.data(), v.size() * sizeof(std::uint64_t));
    return format_ints(v, opt.sign);
  }
  if (kind == 'i') {
    const NDArray c = data.astype(DType::Int64);
    std::vector<std::int64_t> v(static_cast<std::size_t>(c.size()));
    if (!v.empty()) std::memcpy(v.data(), c.data(), v.size() * sizeof(std::int64_t));
    return format_ints(v, opt.sign);
  }
  if (kind == 'f') {
    const std::vector<double> v = as_doubles(data);
    const FloatFormat f(v, dt, opt, opt.sign);
    for (const double x : v) out.push_back(f(x));
    return out;
  }
  const std::vector<std::complex<double>> v = as_complex(data);
  std::vector<double> re;
  std::vector<double> im;
  for (const auto& z : v) {
    re.push_back(z.real());
    im.push_back(z.imag());
  }
  const FloatFormat fr(re, component(dt), opt, opt.sign);
  const FloatFormat fi(im, component(dt), opt, '+');
  for (std::size_t k = 0; k < v.size(); ++k) {
    const std::string r = fr(re[k]);
    std::string i = fi(im[k]);
    const std::size_t sp = i.find_last_not_of(' ') + 1;
    i.insert(sp, "j");
    out.push_back(r + i);
  }
  return out;
}

std::string scalar_str(const NDArray& a) {
  if (a.size() != 1) throw_error(ErrorKind::Value, "scalar_str needs a single element");
  const DType dt = a.dtype();
  const char kind = dtype_info(dt).kind;
  if (kind == 'b') return a.get_double(0) != 0 ? "True" : "False";
  if (kind == 'u') return std::to_string(a.get_uint64(0));
  if (kind == 'i') return std::to_string(a.get_int64(0));
  if (kind == 'f') {
    return float_either(a.get_double(0), dt, TrimMode::LeaveOneZero, TrimMode::DptZeros, false);
  }
  const std::complex<double> z = as_complex(a)[0];
  const DType c = component(dt);
  constexpr TrimMode t = TrimMode::DptZeros;
  if (z.real() == 0.0 && !std::signbit(z.real())) return float_either(z.imag(), c, t, t, false) + "j";
  std::string r;
  if (std::isfinite(z.real())) r = float_either(z.real(), c, t, t, false);
  else if (std::isnan(z.real())) r = "nan";
  else r = z.real() > 0 ? "inf" : "-inf";
  std::string i;
  if (std::isfinite(z.imag())) i = float_either(z.imag(), c, t, t, true);
  else if (std::isnan(z.imag())) i = "+nan";
  else i = z.imag() > 0 ? "+inf" : "-inf";
  return "(" + r + i + "j)";
}

}  // namespace nativpy
