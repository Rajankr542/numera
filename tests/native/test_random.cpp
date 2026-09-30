// Random module tests (D-019). Reference values from NumPy 2.x.
#include <cstdint>
#include <vector>

#include "bitgen.hpp"
#include "distributions.hpp"
#include "error.hpp"
#include "test_harness.hpp"

using namespace nativpy;
using namespace nativpy::random;

namespace {

std::vector<std::int64_t> ints(const NDArray& a) {
  std::vector<std::int64_t> v;
  for (std::int64_t i = 0; i < a.size(); ++i) v.push_back(a.get_int64(i));
  return v;
}

NDArray arange(std::int64_t n) {
  NDArray a = NDArray::empty({n}, DType::Int64);
  for (std::int64_t i = 0; i < n; ++i) a.set_int64(i, i);
  return a;
}

}  // namespace

TEST_CASE("random: PCG64 raw stream from default_rng(42)") {
  PCG64 p(SeedSequence({42}));
  CHECK_EQ(p.next_uint64(), 14276969152011380360ULL);
  CHECK_EQ(p.next_uint64(), 8095878257575067585ULL);
  CHECK_EQ(p.next_uint64(), 15838336090824644132ULL);
}

TEST_CASE("random: SeedSequence big-int and sequence seeds") {
  PCG64 big(SeedSequence({7, 0, 0, 16}));  // 2**100 + 7
  CHECK_EQ(big.next_double(), 0.0653230385916338);
  PCG64 seq(SeedSequence({1, 2, 3}));
  CHECK_EQ(seq.next_double(), 0.6704722626516632);
}

TEST_CASE("random: Generator random/normal/float32") {
  PCG64 a(SeedSequence({42}));
  CHECK_EQ(a.next_double(), 0.7739560485559633);
  PCG64 b(SeedSequence({42}));
  const NDArray n = normal(b, 0.0, 1.0, {3}, DType::Float64);
  CHECK_EQ(n.get_double(0), 0.30471707975443135);
  CHECK_EQ(n.get_double(1), -1.0399841062404955);
  CHECK_EQ(n.get_double(2), 0.7504511958064572);
  PCG64 c(SeedSequence({42}));
  const NDArray f = random_doubles(c, {2}, DType::Float32);
  CHECK_EQ(static_cast<float>(f.get_double(0)), 0.08925092F);
  CHECK_EQ(static_cast<float>(f.get_double(1)), 0.773956F);
  PCG64 d(SeedSequence({42}));
  const NDArray nf = normal(d, 0.0, 1.0, {2}, DType::Float32);
  CHECK_EQ(static_cast<float>(nf.get_double(0)), 0.14190717F);
  CHECK_EQ(static_cast<float>(nf.get_double(1)), -1.6685079F);
  CHECK_THROWS_KIND(normal(d, 1.0, 2.0, {2}, DType::Float32), ErrorKind::Value);
}

TEST_CASE("random: Generator integers (Lemire, buffered)") {
  PCG64 a(SeedSequence({42}));
  CHECK(ints(bounded_integers(a, 0, 10, false, {8}, DType::Int64, false)) ==
        (std::vector<std::int64_t>{0, 7, 6, 4, 4, 8, 0, 6}));
  PCG64 b(SeedSequence({42}));
  CHECK(ints(bounded_integers(b, -5, 5, false, {6}, DType::Int8, false)) ==
        (std::vector<std::int64_t>{0, -4, 3, -5, 4, -4}));
  PCG64 c(SeedSequence({42}));
  CHECK(ints(bounded_integers(c, 0, 2, false, {6}, DType::Bool, false)) ==
        (std::vector<std::int64_t>{0, 0, 0, 1, 0, 0}));
}

TEST_CASE("random: integers bounds errors") {
  PCG64 a(SeedSequence({1}));
  CHECK_THROWS_KIND(bounded_integers(a, 0, 0, false, {1}, DType::Int64, false), ErrorKind::Value);
  CHECK_THROWS_KIND(bounded_integers(a, 0, 300, false, {1}, DType::UInt8, false), ErrorKind::Value);
  CHECK_THROWS_KIND(bounded_integers(a, -1, 3, false, {1}, DType::UInt8, false), ErrorKind::Value);
  CHECK_THROWS_KIND(bounded_integers(a, 0, 3, false, {1}, DType::Float64, false), ErrorKind::DType);
  CHECK_THROWS_KIND(bounded_integers(a, 0, 3, false, {-1}, DType::Int64, false), ErrorKind::Value);
}

TEST_CASE("random: Generator choice/permutation") {
  PCG64 a(SeedSequence({42}));
  CHECK(ints(generator_choice_indices(a, 10, {4}, false, true)) ==
        (std::vector<std::int64_t>{4, 0, 5, 6}));
  PCG64 b(SeedSequence({42}));
  const NDArray p = arange(6);
  shuffle(b, p);
  CHECK(ints(p) == (std::vector<std::int64_t>{3, 2, 5, 4, 1, 0}));
  PCG64 c(SeedSequence({42}));
  CHECK_THROWS_KIND(generator_choice_indices(c, 3, {4}, false, true), ErrorKind::Value);
}

TEST_CASE("random: legacy MT19937 randn/randint/permutation") {
  MT19937 m;
  m.seed(42);
  LegacyGauss gs;
  const NDArray n = legacy_normal(m, gs, 0.0, 1.0, {3});
  CHECK_EQ(n.get_double(0), 0.4967141530112327);
  CHECK_EQ(n.get_double(1), -0.13826430117118466);
  CHECK_EQ(n.get_double(2), 0.6476885381006925);
  MT19937 r;
  r.seed(42);
  CHECK(ints(bounded_integers(r, 0, 10, false, {8}, DType::Int64, true)) ==
        (std::vector<std::int64_t>{6, 3, 7, 4, 6, 9, 2, 6}));
  MT19937 s;
  s.seed(42);
  const NDArray p = arange(6);
  shuffle(s, p);
  CHECK(ints(p) == (std::vector<std::int64_t>{0, 1, 5, 2, 4, 3}));
}

TEST_CASE("random: shuffle 2-D rows and read-only") {
  PCG64 a(SeedSequence({3}));
  NDArray m = arange(6).reshape({3, 2});
  shuffle(a, m);
  for (std::int64_t r = 0; r < 3; ++r) {
    CHECK_EQ(m.get_int64(2 * r) + 1, m.get_int64(2 * r + 1));
    CHECK_EQ(m.get_int64(2 * r) % 2, 0);
  }
  CHECK_THROWS_KIND(shuffle(a, m.as_readonly()), ErrorKind::Value);
}
