#include <cstring>
#include <string>
#include <vector>

#include "creation.hpp"
#include "error.hpp"
#include "layout.hpp"
#include "p14_npy.hpp"
#include "test_harness.hpp"

using namespace nativpy;

namespace {
std::vector<std::uint8_t> bytes_of(const std::string& s) { return {s.begin(), s.end()}; }
NDArray decode(const std::vector<std::uint8_t>& v) { return p14::npy_decode(v.data(), v.size()); }
}  // namespace

TEST_CASE("p14: npy header matches numpy.lib.format") {
  NDArray a = arange(0, 6, 1, DType::Float64).reshape({2, 3});
  CHECK_EQ(p14::npy_header_dict(a),
           std::string("{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }") + std::string(20, ' '));
  const auto enc = p14::npy_encode(a);
  CHECK_EQ(enc.size(), std::size_t{128 + 48});
  CHECK_EQ(enc[8], std::uint8_t{0x76});
  CHECK_EQ(enc[127], std::uint8_t{'\n'});
  NDArray s = NDArray::zeros({}, DType::Float32);
  CHECK_EQ(p14::npy_header_dict(s), std::string("{'descr': '<f4', 'fortran_order': False, 'shape': (), }"));
  CHECK_EQ(p14::npy_descr(DType::Bool), std::string("|b1"));
  CHECK_EQ(p14::npy_descr(DType::Complex128), std::string("<c16"));
  NDArray f = empty_order({2, 3}, DType::Int16, Order::F, true);
  CHECK(p14::npy_header_dict(f).find("'fortran_order': True, 'shape': (2, 3)") != std::string::npos);
}

TEST_CASE("p14: npy round trip for every dtype and order") {
  for (int k = 0; k < kNumDTypes; ++k) {
    const auto dt = static_cast<DType>(k);
    NDArray a = arange(0, 6, 1, DType::Int64).astype(dt).reshape({3, 2});
    NDArray b = decode(p14::npy_encode(a));
    CHECK(b.dtype() == dt);
    CHECK(b.shape() == a.shape());
    CHECK(std::memcmp(a.data(), b.data(), static_cast<std::size_t>(a.nbytes())) == 0);
  }
  NDArray f = copy_order(arange(0, 6, 1, DType::Int64).reshape({2, 3}), DType::Int64, Order::F);
  NDArray g = decode(p14::npy_encode(f));
  CHECK(g.is_f_contiguous());
  CHECK(!g.is_c_contiguous());
  CHECK_EQ(g.get_int64(4), std::int64_t{4});
  NDArray e = decode(p14::npy_encode(NDArray::zeros({0, 4}, DType::UInt8)));
  CHECK(e.shape() == Shape({0, 4}));
}

TEST_CASE("p14: npy decode big-endian, v2 and errors") {
  std::string h = "{'descr': '>i2', 'fortran_order': False, 'shape': (2,), }";
  std::string file = std::string("\x93NUMPY\x02\x00", 8);
  const std::uint32_t hl = static_cast<std::uint32_t>(h.size() + 1);
  for (int k = 0; k < 4; ++k) file += static_cast<char>((hl >> (8 * k)) & 0xff);
  file += h + "\n";
  file += std::string("\x01\x02\x00\x05", 4);
  NDArray a = decode(bytes_of(file));
  CHECK(a.dtype() == DType::Int16);
  CHECK_EQ(a.get_int64(0), std::int64_t{0x0102});
  CHECK_EQ(a.get_int64(1), std::int64_t{5});
  CHECK_THROWS_KIND(decode(bytes_of("")), ErrorKind::Value);
  CHECK_THROWS_KIND(decode(bytes_of("hello world")), ErrorKind::Value);
  CHECK_THROWS_KIND(decode(bytes_of(std::string("\x93NUMPY\x05\x00", 8))), ErrorKind::Value);
  CHECK_THROWS_KIND(decode(bytes_of(std::string("\x93NUMPY\x01\x00", 8))), ErrorKind::Value);
  auto full = p14::npy_encode(arange(0, 4, 1, DType::Float64));
  full.resize(full.size() - 8);
  CHECK_THROWS_KIND(decode(full), ErrorKind::Value);
  auto mk = [](const std::string& dict) {
    std::string f = std::string("\x93NUMPY\x01\x00", 8);
    const auto n = dict.size() + 1;
    f += static_cast<char>(n & 0xff);
    f += static_cast<char>(n >> 8);
    return bytes_of(f + dict + "\n");
  };
  CHECK_THROWS_KIND(decode(mk("{'descr': '<U3', 'fortran_order': False, 'shape': (), }")), ErrorKind::DType);
  CHECK_THROWS_KIND(decode(mk("{'descr': [('a', '<i4')], 'fortran_order': False, 'shape': (), }")), ErrorKind::DType);
  CHECK_THROWS_KIND(decode(mk("{'descr': '|O', 'fortran_order': False, 'shape': (), }")), ErrorKind::Value);
  CHECK_THROWS_KIND(decode(mk("{'descr': '<f8'}")), ErrorKind::Value);
  CHECK_THROWS_KIND(decode(mk("{'descr': '<f8', 'fortran_order': 1, 'shape': (), }")), ErrorKind::Value);
  CHECK_EQ(decode(mk("{\"descr\": \"<f8\", \"shape\": (0,), \"fortran_order\": False}")).size(), std::int64_t{0});
}

TEST_CASE("p14: crc32") {
  const std::string s = "123456789";
  CHECK_EQ(p14::crc32(reinterpret_cast<const std::uint8_t*>(s.data()), s.size()), std::uint32_t{0xCBF43926});
  CHECK_EQ(p14::crc32(nullptr, 0), std::uint32_t{0});
  const auto* p = reinterpret_cast<const std::uint8_t*>(s.data());
  CHECK_EQ(p14::crc32(p + 4, 5, p14::crc32(p, 4)), std::uint32_t{0xCBF43926});
}
