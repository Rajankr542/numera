#include "bitgen.hpp"

namespace nativpy::random {

namespace {

// SeedSequence constants (numpy/random/bit_generator.pyx).
constexpr std::uint32_t kInitA = 0x43b0d7e5U;
constexpr std::uint32_t kMultA = 0x931e8875U;
constexpr std::uint32_t kInitB = 0x8b51f9ddU;
constexpr std::uint32_t kMultB = 0x58f38dedU;
constexpr std::uint32_t kMixMultL = 0xca01f9ddU;
constexpr std::uint32_t kMixMultR = 0x4973f715U;
constexpr unsigned kXShift = 16;

std::uint32_t hashmix(std::uint32_t value, std::uint32_t& hash_const) {
  value ^= hash_const;
  hash_const *= kMultA;
  value *= hash_const;
  value ^= value >> kXShift;
  return value;
}

std::uint32_t mix(std::uint32_t x, std::uint32_t y) {
  std::uint32_t result = kMixMultL * x - kMixMultR * y;
  result ^= result >> kXShift;
  return result;
}

// PCG64 constants (numpy/random/src/pcg64/pcg64.h).
constexpr u128 kPcgMult =
    (static_cast<u128>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;

std::uint64_t rotr64(std::uint64_t v, unsigned rot) {
  return (v >> rot) | (v << ((64U - rot) & 63U));
}

// NumPy uint64_to_double: 53 high bits.
double to_double(std::uint64_t r) {
  return static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
}

constexpr std::size_t kN = 624;
constexpr std::size_t kM = 397;
constexpr std::uint32_t kMatrixA = 0x9908b0dfU;
constexpr std::uint32_t kUpper = 0x80000000U;
constexpr std::uint32_t kLower = 0x7fffffffU;

}  // namespace

SeedSequence::SeedSequence(const std::vector<std::uint32_t>& entropy) {
  std::uint32_t hash_const = kInitA;
  const std::size_t n = pool_.size();
  for (std::size_t i = 0; i < n; ++i) {
    pool_[i] = hashmix(i < entropy.size() ? entropy[i] : 0U, hash_const);
  }
  for (std::size_t s = 0; s < n; ++s) {
    for (std::size_t d = 0; d < n; ++d) {
      if (s != d) pool_[d] = mix(pool_[d], hashmix(pool_[s], hash_const));
    }
  }
  for (std::size_t s = n; s < entropy.size(); ++s) {
    for (std::size_t d = 0; d < n; ++d) {
      pool_[d] = mix(pool_[d], hashmix(entropy[s], hash_const));
    }
  }
}

std::vector<std::uint32_t> SeedSequence::generate_state_u32(std::size_t n_words) const {
  std::vector<std::uint32_t> out(n_words);
  std::uint32_t hash_const = kInitB;
  for (std::size_t i = 0; i < n_words; ++i) {
    std::uint32_t v = pool_[i % pool_.size()];
    v ^= hash_const;
    hash_const *= kMultB;
    v *= hash_const;
    v ^= v >> kXShift;
    out[i] = v;
  }
  return out;
}

std::vector<std::uint64_t> SeedSequence::generate_state_u64(std::size_t n_words) const {
  const auto w = generate_state_u32(n_words * 2);
  std::vector<std::uint64_t> out(n_words);
  for (std::size_t i = 0; i < n_words; ++i) {
    out[i] = static_cast<std::uint64_t>(w[2 * i]) |
             (static_cast<std::uint64_t>(w[2 * i + 1]) << 32);
  }
  return out;
}

PCG64::PCG64(const SeedSequence& seq) {
  const auto v = seq.generate_state_u64(4);
  const u128 initstate = (static_cast<u128>(v[0]) << 64) | v[1];
  const u128 initseq = (static_cast<u128>(v[2]) << 64) | v[3];
  // pcg_setseq_128_srandom_r
  state_ = 0;
  inc_ = (initseq << 1U) | 1U;
  state_ = state_ * kPcgMult + inc_;
  state_ += initstate;
  state_ = state_ * kPcgMult + inc_;
}

std::uint64_t PCG64::next_uint64() {
  state_ = state_ * kPcgMult + inc_;
  const auto hi = static_cast<std::uint64_t>(state_ >> 64);
  const auto lo = static_cast<std::uint64_t>(state_);
  return rotr64(hi ^ lo, static_cast<unsigned>(state_ >> 122));
}

std::uint32_t PCG64::next_uint32() {
  if (has_uint32_) {
    has_uint32_ = false;
    return uinteger_;
  }
  const std::uint64_t next = next_uint64();
  has_uint32_ = true;
  uinteger_ = static_cast<std::uint32_t>(next >> 32);
  return static_cast<std::uint32_t>(next & 0xffffffffU);
}

double PCG64::next_double() { return to_double(next_uint64()); }

// ---- MT19937 (numpy/random/src/mt19937) ----

MT19937::MT19937() { seed(0); }

void MT19937::seed(std::uint32_t s) {
  for (std::size_t pos = 0; pos < kN; ++pos) {
    key_[pos] = s;
    s = 1812433253U * (s ^ (s >> 30)) + static_cast<std::uint32_t>(pos) + 1U;
  }
  pos_ = static_cast<int>(kN);
}

void MT19937::seed_by_array(const std::vector<std::uint32_t>& init_key) {
  auto& mt = key_;
  mt[0] = 19650218U;  // init_genrand(19650218)
  for (std::size_t i = 1; i < kN; ++i) {
    mt[i] = 1812433253U * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<std::uint32_t>(i);
  }
  std::size_t i = 1;
  std::size_t j = 0;
  const std::size_t len = init_key.size();
  for (std::size_t k = (kN > len ? kN : len); k; --k) {
    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525U)) + init_key[j] +
            static_cast<std::uint32_t>(j);
    ++i;
    ++j;
    if (i >= kN) {
      mt[0] = mt[kN - 1];
      i = 1;
    }
    if (j >= len) j = 0;
  }
  for (std::size_t k = kN - 1; k; --k) {
    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) -
            static_cast<std::uint32_t>(i);
    ++i;
    if (i >= kN) {
      mt[0] = mt[kN - 1];
      i = 1;
    }
  }
  mt[0] = 0x80000000U;
  pos_ = static_cast<int>(kN);
}

void MT19937::seed_from_state(const std::vector<std::uint32_t>& state624) {
  key_[0] = 0x80000000U;
  for (std::size_t i = 1; i < kN && i < state624.size(); ++i) key_[i] = state624[i];
  pos_ = static_cast<int>(kN);
}

void MT19937::generate() {
  const auto twist = [](std::uint32_t y) { return (y >> 1) ^ ((0U - (y & 1U)) & kMatrixA); };
  std::size_t i = 0;
  for (; i < kN - kM; ++i) {
    key_[i] = key_[i + kM] ^ twist((key_[i] & kUpper) | (key_[i + 1] & kLower));
  }
  for (; i < kN - 1; ++i) {
    key_[i] = key_[i + kM - kN] ^ twist((key_[i] & kUpper) | (key_[i + 1] & kLower));
  }
  key_[kN - 1] = key_[kM - 1] ^ twist((key_[kN - 1] & kUpper) | (key_[0] & kLower));
  pos_ = 0;
}

std::uint32_t MT19937::next_uint32() {
  if (pos_ == static_cast<int>(kN)) generate();
  std::uint32_t y = key_[static_cast<std::size_t>(pos_++)];
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680U;
  y ^= (y << 15) & 0xefc60000U;
  y ^= (y >> 18);
  return y;
}

std::uint64_t MT19937::next_uint64() {
  const std::uint64_t hi = next_uint32();
  return (hi << 32) | next_uint32();
}

double MT19937::next_double() {
  const std::int32_t a = static_cast<std::int32_t>(next_uint32() >> 5);
  const std::int32_t b = static_cast<std::int32_t>(next_uint32() >> 6);
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

}  // namespace nativpy::random
