#pragma once

// P13 bit generators with full state access (D-160), ported from NumPy 2.x
// numpy/random (BSD-3-Clause): general SeedSequence (spawn key, pool size),
// MT19937, PCG64, PCG64DXSM, Philox4x64-10 and SFC64.

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "bitgen.hpp"

namespace nativpy::random::p13 {

// numpy.random.SeedSequence: `entropy` and `spawn_key` already coerced to
// uint32 words (_coerce_to_uint32_array); pool_size >= 4.
class SeedSeq {
 public:
  SeedSeq(const std::vector<std::uint32_t>& entropy, const std::vector<std::uint32_t>& spawn_key,
          std::size_t pool_size);
  [[nodiscard]] std::vector<std::uint32_t> generate_u32(std::size_t n_words) const;
  [[nodiscard]] std::vector<std::uint64_t> generate_u64(std::size_t n_words) const;

 private:
  std::vector<std::uint32_t> pool_;
};

// A bit generator whose raw state can be read and written as uint64 words
// (layout per kind, see p13_bitgen.cpp).
class StatefulBitGen : public BitGen {
 public:
  [[nodiscard]] virtual std::vector<std::uint64_t> get_state() const = 0;
  virtual void set_state(const std::vector<std::uint64_t>& words) = 0;
};

class MT19937 final : public StatefulBitGen {
 public:
  MT19937() { seed(0); }
  void seed(std::uint32_t s);
  void seed_by_array(const std::vector<std::uint32_t>& key);
  void seed_seq(const SeedSeq& seq);
  std::uint64_t next_uint64() override;
  std::uint32_t next_uint32() override;
  double next_double() override;
  // [key0..key623, pos]
  [[nodiscard]] std::vector<std::uint64_t> get_state() const override;
  void set_state(const std::vector<std::uint64_t>& words) override;

 private:
  void generate();
  std::array<std::uint32_t, 624> key_{};
  int pos_ = 624;
};

// PCG64 (XSL-RR) and PCG64DXSM (cheap-multiplier DXSM).
class PCG64 final : public StatefulBitGen {
 public:
  PCG64(const SeedSeq& seq, bool dxsm);
  std::uint64_t next_uint64() override;
  std::uint32_t next_uint32() override;
  double next_double() override;
  // [state_hi, state_lo, inc_hi, inc_lo, has_uint32, uinteger]
  [[nodiscard]] std::vector<std::uint64_t> get_state() const override;
  void set_state(const std::vector<std::uint64_t>& words) override;

 private:
  bool dxsm_;
  u128 state_ = 0;
  u128 inc_ = 0;
  bool has_uint32_ = false;
  std::uint32_t uinteger_ = 0;
};

class Philox final : public StatefulBitGen {
 public:
  // key from seq.generate_u64(2) unless `key` is given; counter defaults to 0.
  Philox(const SeedSeq* seq, const std::vector<std::uint64_t>& key,
         const std::vector<std::uint64_t>& counter);
  std::uint64_t next_uint64() override;
  std::uint32_t next_uint32() override;
  double next_double() override;
  // [ctr0..3, key0..1, buf0..3, buffer_pos, has_uint32, uinteger]
  [[nodiscard]] std::vector<std::uint64_t> get_state() const override;
  void set_state(const std::vector<std::uint64_t>& words) override;

 private:
  std::array<std::uint64_t, 4> ctr_{};
  std::array<std::uint64_t, 2> key_{};
  std::array<std::uint64_t, 4> buffer_{};
  int buffer_pos_ = 4;
  bool has_uint32_ = false;
  std::uint32_t uinteger_ = 0;
};

class SFC64 final : public StatefulBitGen {
 public:
  explicit SFC64(const SeedSeq& seq);
  std::uint64_t next_uint64() override;
  std::uint32_t next_uint32() override;
  double next_double() override;
  // [s0..s3, has_uint32, uinteger]
  [[nodiscard]] std::vector<std::uint64_t> get_state() const override;
  void set_state(const std::vector<std::uint64_t>& words) override;

 private:
  std::array<std::uint64_t, 4> s_{};
  bool has_uint32_ = false;
  std::uint32_t uinteger_ = 0;
};

}  // namespace nativpy::random::p13
