#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ndarray.hpp"

// P14 NPY format codec and CRC-32 (NUMPY_PARITY P14, D-170).
namespace nativpy::p14 {

// NumPy array-protocol descriptor of a native dtype, e.g. "<f8", "|b1".
std::string npy_descr(DType dt);

// Header dict text exactly as numpy.lib.format writes it (before padding).
std::string npy_header_dict(const NDArray& a);

// Full .npy file contents (magic, version, padded header, data). Fortran
// order is used when `a` is F- but not C-contiguous, as NumPy does.
std::vector<std::uint8_t> npy_encode(const NDArray& a);

// Parses .npy bytes. Accepts versions 1.0/2.0/3.0, little/big/native byte
// order, C and Fortran order. Throws Value (bad/truncated data) or DType
// (descriptor not supported).
NDArray npy_decode(const std::uint8_t* data, std::size_t len);

// zlib-compatible CRC-32 (as zipfile/binascii.crc32), continuing from `crc`.
std::uint32_t crc32(const std::uint8_t* data, std::size_t len, std::uint32_t crc = 0);

}  // namespace nativpy::p14
