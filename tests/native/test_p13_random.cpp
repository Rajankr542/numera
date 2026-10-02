#include <cstdint>
#include <vector>

#include "error.hpp"
#include "p13_bitgen.hpp"
#include "p13_distributions.hpp"
#include "test_harness.hpp"

using namespace nativpy;
using namespace nativpy::random;

namespace {

p13::SeedSeq seq(std::uint32_t s) { return p13::SeedSeq({s}, {}, 4); }

}  // namespace

// Expected values from NumPy 2.5.3: BitGen(42).random_raw(3) and friends.
TEST_CASE("p13_bitgen_raw_streams") {
  p13::PCG64 pcg(seq(42), false);
  CHECK_EQ(pcg.next_uint64(), 14276969152011380360ULL);
  CHECK_EQ(pcg.next_uint64(), 8095878257575067585ULL);
  CHECK_EQ(pcg.next_uint64(), 15838336090824644132ULL);
  p13::PCG64 dxsm(seq(42), true);
  CHECK_EQ(dxsm.next_uint64(), 12329818062196000797ULL);
  CHECK_EQ(dxsm.next_uint64(), 125530269004142706ULL);
  p13::MT19937 mt;
  mt.seed_seq(seq(42));
  CHECK_EQ(mt.next_uint32(), 2327846034U);
  CHECK_EQ(mt.next_uint32(), 3904886566U);
  const auto s = seq(42);
  p13::Philox ph(&s, {}, {});
  CHECK_EQ(ph.next_uint64(), 1587852024645073290ULL);
  CHECK_EQ(ph.next_uint64(), 2611271723512893552ULL);
  p13::SFC64 sfc(seq(42));
  CHECK_EQ(sfc.next_uint64(), 9775594601838723485ULL);
  CHECK_EQ(sfc.next_uint64(), 6977463094773878866ULL);
}

TEST_CASE("p13_seedseq_spawn_key_pool") {
  const p13::SeedSeq ss({42}, {1, 2}, 8);
  const auto w = ss.generate_u32(4);
  CHECK_EQ(w[0], 3762326729U);
  CHECK_EQ(w[1], 1032689629U);
  CHECK_EQ(w[2], 2623687498U);
  CHECK_EQ(w[3], 1198876664U);
  CHECK_THROWS_KIND(p13::SeedSeq({1}, {}, 3), ErrorKind::Value);
}

TEST_CASE("p13_state_roundtrip") {
  p13::PCG64 a(seq(7), false);
  (void)a.next_uint32();
  const auto st = a.get_state();
  const auto x = a.next_uint64();
  a.set_state(st);
  CHECK_EQ(a.next_uint64(), x);
  CHECK_THROWS_KIND(a.set_state({1, 2}), ErrorKind::Value);
}

TEST_CASE("p13_generator_and_legacy_kernels") {
  p13::PCG64 g(seq(42), false);
  CHECK_EQ(p13::standard_gamma(g, 0.5), 0.8045001461315131);
  CHECK_EQ(p13::beta(g, 2, 3), 0.49601324887830794);
  p13::Binomial b;
  CHECK_EQ(p13::binomial(g, 0.3, 100, b), 25);
  CHECK_EQ(p13::poisson(g, 20), 19);
  CHECK_EQ(p13::hypergeometric(g, 10, 20, 5), 2);

  p13::MT19937 mt;
  mt.seed(42);
  LegacyGauss gs;
  CHECK_EQ(p13::legacy::standard_gamma(mt, gs, 0.5), 0.14028030062619642);
  CHECK_EQ(p13::legacy::beta(mt, gs, 2, 3), 0.30614387670416887);
  p13::Binomial lb;
  CHECK_EQ(p13::legacy::binomial(mt, 0.3, 100, lb), 23);
  CHECK_EQ(p13::poisson(mt, 20), 26);
  CHECK_EQ(p13::legacy::zipf(mt, 2.0), 3);
}
