#pragma once

// Bit generators ported from NumPy 2.x (numpy/random, BSD-3-Clause), D-019.
// Streams are intended to be bit-identical to NumPy's.

#include <array>
#include <cstdint>
#include <vector>

namespace nativpy::random {

// 128-bit unsigned (GCC/Clang). MSVC is not supported for PCG64 yet (D-019).
__extension__ typedef unsigned __int128 u128;

// Abstract source of random bits (NumPy bitgen_t).
class BitGen {
 public:
  virtual ~BitGen() = default;
  virtual std::uint64_t next_uint64() = 0;
  virtual std::uint32_t next_uint32() = 0;
  virtual double next_double() = 0;
};

// numpy.random.SeedSequence (pool_size 4, no spawn key).
class SeedSequence {
 public:
  // `entropy`: the seed as uint32 words, least significant first
  // (NumPy _coerce_to_uint32_array).
  explicit SeedSequence(const std::vector<std::uint32_t>& entropy);
  [[nodiscard]] std::vector<std::uint32_t> generate_state_u32(std::size_t n_words) const;
  [[nodiscard]] std::vector<std::uint64_t> generate_state_u64(std::size_t n_words) const;

 private:
  std::array<std::uint32_t, 4> pool_{};
};

// numpy.random.PCG64 (XSL-RR 128/64).
class PCG64 final : public BitGen {
 public:
  explicit PCG64(const SeedSequence& seq);
  std::uint64_t next_uint64() override;
  std::uint32_t next_uint32() override;
  double next_double() override;

  struct State {
    u128 state;
    u128 inc;
    bool has_uint32;
    std::uint32_t uinteger;
  };
  [[nodiscard]] State state() const noexcept {
    return {state_, inc_, has_uint32_, uinteger_};
  }

 private:
  u128 state_ = 0;
  u128 inc_ = 0;
  bool has_uint32_ = false;
  std::uint32_t uinteger_ = 0;
};

// numpy.random.MT19937 with legacy (RandomState) seeding.
class MT19937 final : public BitGen {
 public:
  MT19937();  // seeded with 0 until seed() is called
  void seed(std::uint32_t seed);                           // mt19937_seed
  void seed_by_array(const std::vector<std::uint32_t>& key);  // init_by_array
  // Entropy seeding used by RandomState(None): key[0]=0x80000000, rest from state.
  void seed_from_state(const std::vector<std::uint32_t>& state624);
  std::uint64_t next_uint64() override;
  std::uint32_t next_uint32() override;
  double next_double() override;

 private:
  void generate();
  std::array<std::uint32_t, 624> key_{};
  int pos_ = 624;
};

}  // namespace nativpy::random
