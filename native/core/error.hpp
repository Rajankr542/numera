#pragma once

#include <stdexcept>
#include <string>

namespace nativpy {

enum class ErrorKind {
  Shape,
  DType,
  Index,
  Broadcast,
  Value,
  Memory,
  NotImplemented,
};

const char* error_kind_name(ErrorKind kind) noexcept;

class Error : public std::runtime_error {
 public:
  Error(ErrorKind kind, const std::string& message)
      : std::runtime_error(message), kind_(kind) {}
  [[nodiscard]] ErrorKind kind() const noexcept { return kind_; }

 private:
  ErrorKind kind_;
};

[[noreturn]] inline void throw_error(ErrorKind kind, const std::string& msg) {
  throw Error(kind, msg);
}

}  // namespace nativpy
