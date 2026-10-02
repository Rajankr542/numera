#include "p13_bitgen.hpp"

#include "error.hpp"

namespace nativpy::random::p13 {

namespace {

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

double to_double(std::uint64_t r) {
  return static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);
}

constexpr u128 kPcgMult =
    (static_cast<u128>(2549297995355413924ULL) << 64) | 4865540595714422341ULL;
constexpr std::uint64_t kCheapMult = 0xda942042e4dd58b5ULL;

std::uint64_t rotr64(std::uint64_t v, unsigned rot) {
  return (v >> rot) | (v << ((64U - rot) & 63U));
}

void need(const std::vector<std::uint64_t>& w, std::size_t n) {
  if (w.size() != n) throw_error(ErrorKind::Value, "invalid bit generator state");
}

constexpr std::size_t kN = 624;
constexpr std::size_t kM = 397;

}  // namespace

// ---- SeedSequence ----

SeedSeq::SeedSeq(const std::vector<std::uint32_t>& entropy,
                 const std::vector<std::uint32_t>& spawn_key, std::size_t pool_size)
    : pool_(pool_size, 0U) {
  if (pool_size < 4) throw_error(ErrorKind::Value, "The size of the entropy pool should be at least 4");
  std::vector<std::uint32_t> ent = entropy;
  if (!spawn_key.empty() && ent.size() < pool_size) ent.resize(pool_size, 0U);
  ent.insert(ent.end(), spawn_key.begin(), spawn_key.end());
  std::uint32_t hash_const = kInitA;
  const std::size_t n = pool_.size();
  for (std::size_t i = 0; i < n; ++i) pool_[i] = hashmix(i < ent.size() ? ent[i] : 0U, hash_const);
  for (std::size_t s = 0; s < n; ++s) {
    for (std::size_t d = 0; d < n; ++d) {
      if (s != d) pool_[d] = mix(pool_[d], hashmix(pool_[s], hash_const));
    }
  }
  for (std::size_t s = n; s < ent.size(); ++s) {
    for (std::size_t d = 0; d < n; ++d) pool_[d] = mix(pool_[d], hashmix(ent[s], hash_const));
  }
}

std::vector<std::uint32_t> SeedSeq::generate_u32(std::size_t n_words) const {
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

std::vector<std::uint64_t> SeedSeq::generate_u64(std::size_t n_words) const {
  const auto w = generate_u32(n_words * 2);
  std::vector<std::uint64_t> out(n_words);
  for (std::size_t i = 0; i < n_words; ++i) {
    out[i] = static_cast<std::uint64_t>(w[2 * i]) | (static_cast<std::uint64_t>(w[2 * i + 1]) << 32);
  }
  return out;
}

// ---- MT19937 ----

void MT19937::seed(std::uint32_t s) {
  for (std::size_t pos = 0; pos < kN; ++pos) {
    key_[pos] = s;
    s = 1812433253U * (s ^ (s >> 30)) + static_cast<std::uint32_t>(pos) + 1U;
  }
  pos_ = static_cast<int>(kN);
}

void MT19937::seed_by_array(const std::vector<std::uint32_t>& init_key) {
  auto& mt = key_;
  mt[0] = 19650218U;
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
    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941U)) - static_cast<std::uint32_t>(i);
    ++i;
    if (i >= kN) {
      mt[0] = mt[kN - 1];
      i = 1;
    }
  }
  mt[0] = 0x80000000U;
  pos_ = static_cast<int>(kN);
}

void MT19937::seed_seq(const SeedSeq& seq) {
  const auto v = seq.generate_u32(kN);
  key_[0] = 0x80000000U;
  for (std::size_t i = 1; i < kN; ++i) key_[i] = v[i];
  pos_ = static_cast<int>(kN - 1);  // NumPy leaves pos at the loop variable (623)
}

void MT19937::generate() {
  constexpr std::uint32_t kMatrixA = 0x9908b0dfU;
  constexpr std::uint32_t kUpper = 0x80000000U;
  constexpr std::uint32_t kLower = 0x7fffffffU;
  const auto twist = [](std::uint32_t y) { return (y >> 1) ^ ((0U - (y & 1U)) & kMatrixA); };
  std::size_t i = 0;
  for (; i < kN - kM; ++i) key_[i] = key_[i + kM] ^ twist((key_[i] & kUpper) | (key_[i + 1] & kLower));
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

std::vector<std::uint64_t> MT19937::get_state() const {
  std::vector<std::uint64_t> out(kN + 1);
  for (std::size_t i = 0; i < kN; ++i) out[i] = key_[i];
  out[kN] = static_cast<std::uint64_t>(pos_);
  return out;
}

void MT19937::set_state(const std::vector<std::uint64_t>& w) {
  need(w, kN + 1);
  if (w[kN] > kN) throw_error(ErrorKind::Value, "invalid MT19937 state position");
  for (std::size_t i = 0; i < kN; ++i) key_[i] = static_cast<std::uint32_t>(w[i]);
  pos_ = static_cast<int>(w[kN]);
}

// ---- PCG64 / PCG64DXSM ----

PCG64::PCG64(const SeedSeq& seq, bool dxsm) : dxsm_(dxsm) {
  const auto v = seq.generate_u64(4);
  const u128 initstate = (static_cast<u128>(v[0]) << 64) | v[1];
  const u128 initseq = (static_cast<u128>(v[2]) << 64) | v[3];
  // NumPy seeds both variants with pcg64_set_seed (full 128-bit multiplier).
  const u128 mult = kPcgMult;
  state_ = 0;
  inc_ = (initseq << 1U) | 1U;
  state_ = state_ * mult + inc_;
  state_ += initstate;
  state_ = state_ * mult + inc_;
}

std::uint64_t PCG64::next_uint64() {
  if (!dxsm_) {
    state_ = state_ * kPcgMult + inc_;
    const auto hi = static_cast<std::uint64_t>(state_ >> 64);
    const auto lo = static_cast<std::uint64_t>(state_);
    return rotr64(hi ^ lo, static_cast<unsigned>(state_ >> 122));
  }
  auto hi = static_cast<std::uint64_t>(state_ >> 64);
  auto lo = static_cast<std::uint64_t>(state_);
  lo |= 1;
  hi ^= hi >> 32;
  hi *= kCheapMult;
  hi ^= hi >> 48;
  hi *= lo;
  state_ = state_ * kCheapMult + inc_;
  return hi;
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

std::vector<std::uint64_t> PCG64::get_state() const {
  return {static_cast<std::uint64_t>(state_ >> 64), static_cast<std::uint64_t>(state_),
          static_cast<std::uint64_t>(inc_ >> 64), static_cast<std::uint64_t>(inc_),
          has_uint32_ ? 1U : 0U, uinteger_};
}

void PCG64::set_state(const std::vector<std::uint64_t>& w) {
  need(w, 6);
  state_ = (static_cast<u128>(w[0]) << 64) | w[1];
  inc_ = (static_cast<u128>(w[2]) << 64) | w[3];
  has_uint32_ = w[4] != 0;
  uinteger_ = static_cast<std::uint32_t>(w[5]);
}

// ---- Philox4x64-10 ----

Philox::Philox(const SeedSeq* seq, const std::vector<std::uint64_t>& key,
               const std::vector<std::uint64_t>& counter) {
  if (seq != nullptr) {
    const auto k = seq->generate_u64(2);
    key_ = {k[0], k[1]};
  } else {
    need(key, 2);
    key_ = {key[0], key[1]};
  }
  if (!counter.empty()) {
    need(counter, 4);
    for (std::size_t i = 0; i < 4; ++i) ctr_[i] = counter[i];
  }
}

std::uint64_t Philox::next_uint64() {
  if (buffer_pos_ < 4) return buffer_[static_cast<std::size_t>(buffer_pos_++)];
  if (++ctr_[0] == 0 && ++ctr_[1] == 0 && ++ctr_[2] == 0) ++ctr_[3];
  auto ctr = ctr_;
  auto key = key_;
  for (int r = 0; r < 10; ++r) {
    if (r > 0) {
      key[0] += 0x9E3779B97F4A7C15ULL;
      key[1] += 0xBB67AE8584CAA73BULL;
    }
    const u128 p0 = static_cast<u128>(0xD2E7470EE14C6C93ULL) * ctr[0];
    const u128 p1 = static_cast<u128>(0xCA5A826395121157ULL) * ctr[2];
    const auto hi0 = static_cast<std::uint64_t>(p0 >> 64);
    const auto lo0 = static_cast<std::uint64_t>(p0);
    const auto hi1 = static_cast<std::uint64_t>(p1 >> 64);
    const auto lo1 = static_cast<std::uint64_t>(p1);
    ctr = {hi1 ^ ctr[1] ^ key[0], lo1, hi0 ^ ctr[3] ^ key[1], lo0};
  }
  buffer_ = ctr;
  buffer_pos_ = 1;
  return buffer_[0];
}

std::uint32_t Philox::next_uint32() {
  if (has_uint32_) {
    has_uint32_ = false;
    return uinteger_;
  }
  const std::uint64_t next = next_uint64();
  has_uint32_ = true;
  uinteger_ = static_cast<std::uint32_t>(next >> 32);
  return static_cast<std::uint32_t>(next & 0xffffffffU);
}

double Philox::next_double() { return to_double(next_uint64()); }

std::vector<std::uint64_t> Philox::get_state() const {
  return {ctr_[0], ctr_[1], ctr_[2], ctr_[3], key_[0], key_[1],
          buffer_[0], buffer_[1], buffer_[2], buffer_[3],
          static_cast<std::uint64_t>(buffer_pos_), has_uint32_ ? 1U : 0U, uinteger_};
}

void Philox::set_state(const std::vector<std::uint64_t>& w) {
  need(w, 13);
  if (w[10] > 4) throw_error(ErrorKind::Value, "invalid Philox buffer_pos");
  for (std::size_t i = 0; i < 4; ++i) ctr_[i] = w[i];
  key_ = {w[4], w[5]};
  for (std::size_t i = 0; i < 4; ++i) buffer_[i] = w[6 + i];
  buffer_pos_ = static_cast<int>(w[10]);
  has_uint32_ = w[11] != 0;
  uinteger_ = static_cast<std::uint32_t>(w[12]);
}

// ---- SFC64 ----

namespace {
std::uint64_t sfc64_next(std::array<std::uint64_t, 4>& s) {
  const std::uint64_t tmp = s[0] + s[1] + s[3]++;
  s[0] = s[1] ^ (s[1] >> 11);
  s[1] = s[2] + (s[2] << 3);
  s[2] = ((s[2] << 24) | (s[2] >> 40)) + tmp;
  return tmp;
}
}  // namespace

SFC64::SFC64(const SeedSeq& seq) {
  const auto v = seq.generate_u64(3);
  s_ = {v[0], v[1], v[2], 1};
  for (int i = 0; i < 12; ++i) (void)sfc64_next(s_);
}

std::uint64_t SFC64::next_uint64() { return sfc64_next(s_); }

std::uint32_t SFC64::next_uint32() {
  if (has_uint32_) {
    has_uint32_ = false;
    return uinteger_;
  }
  const std::uint64_t next = sfc64_next(s_);
  has_uint32_ = true;
  uinteger_ = static_cast<std::uint32_t>(next >> 32);
  return static_cast<std::uint32_t>(next & 0xffffffffU);
}

double SFC64::next_double() { return to_double(next_uint64()); }

std::vector<std::uint64_t> SFC64::get_state() const {
  return {s_[0], s_[1], s_[2], s_[3], has_uint32_ ? 1U : 0U, uinteger_};
}

void SFC64::set_state(const std::vector<std::uint64_t>& w) {
  need(w, 6);
  for (std::size_t i = 0; i < 4; ++i) s_[i] = w[i];
  has_uint32_ = w[4] != 0;
  uinteger_ = static_cast<std::uint32_t>(w[5]);
}

}  // namespace nativpy::random::p13
