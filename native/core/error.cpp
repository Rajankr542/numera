#include "error.hpp"

namespace nativpy {

const char* error_kind_name(ErrorKind kind) noexcept {
  switch (kind) {
    case ErrorKind::Shape:
      return "ShapeError";
    case ErrorKind::DType:
      return "DTypeError";
    case ErrorKind::Index:
      return "IndexError";
    case ErrorKind::Broadcast:
      return "BroadcastError";
    case ErrorKind::Value:
      return "ValueError";
    case ErrorKind::Memory:
      return "MemoryError";
    case ErrorKind::NotImplemented:
      return "NotImplementedError";
    case ErrorKind::LinAlg:
      return "LinAlgError";
  }
  return "NativpyError";
}

}  // namespace nativpy
