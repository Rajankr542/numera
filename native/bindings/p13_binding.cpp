#include <napi.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "binding_utils.hpp"
#include "error.hpp"
#include "error_binding.hpp"
#include "milestone_bindings.hpp"
#include "p13_bitgen.hpp"

namespace nativpy::bindings {

namespace {

using namespace util;

// Decode a JS Array of Numbers into a vector of uint32_t words.
std::vector<std::uint32_t> to_u32_words(const Napi::Value& v) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, "expected an array of uint32 words");
  const auto arr = v.As<Napi::Array>();
  std::vector<std::uint32_t> out(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) {
    const double d = arr.Get(i).As<Napi::Number>().DoubleValue();
    if (!(d >= 0 && d <= 4294967295.0) || d != static_cast<double>(static_cast<std::uint32_t>(d))) {
      throw_error(ErrorKind::Value, "seed words must be uint32");
    }
    out[i] = static_cast<std::uint32_t>(d);
  }
  return out;
}

// Decode a JS Array of BigInt / Number into a vector of uint64_t.
std::vector<std::uint64_t> to_u64_words(const Napi::Value& v) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, "expected an array of uint64 words");
  const auto arr = v.As<Napi::Array>();
  std::vector<std::uint64_t> out(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) {
    const Napi::Value elem = arr.Get(i);
    if (elem.IsBigInt()) {
      int sign = 0;
      std::size_t count = 1;
      std::uint64_t word = 0;
      elem.As<Napi::BigInt>().ToWords(&sign, &count, &word);
      if (sign != 0) throw_error(ErrorKind::Value, "state words must be non-negative");
      out[i] = word;
    } else {
      out[i] = static_cast<std::uint64_t>(elem.As<Napi::Number>().DoubleValue());
    }
  }
  return out;
}

// Encode a vector of uint64_t as a JS Array of BigInt.
Napi::Array u64_words_to_js(Napi::Env env, const std::vector<std::uint64_t>& w) {
  Napi::Array out = Napi::Array::New(env, w.size());
  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(w.size()); ++i) {
    out.Set(i, Napi::BigInt::New(env, w[i]));
  }
  return out;
}

// Encode a vector of uint32_t as a JS Array of Number.
Napi::Array u32_words_to_js(Napi::Env env, const std::vector<std::uint32_t>& w) {
  Napi::Array out = Napi::Array::New(env, w.size());
  for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(w.size()); ++i) {
    out.Set(i, Napi::Number::New(env, static_cast<double>(w[i])));
  }
  return out;
}

// ---- SeedSequenceWrap ----

class SeedSeqWrap : public Napi::ObjectWrap<SeedSeqWrap> {
 public:
  static Napi::Function define(Napi::Env env) {
    return DefineClass(env, "SeedSequence",
                       {InstanceMethod<&SeedSeqWrap::generate_state>("generateState"),
                        InstanceMethod<&SeedSeqWrap::generate_state64>("generateState64")});
  }

  // new SeedSequence(entropy: number[], spawnKey: number[], poolSize: number)
  explicit SeedSeqWrap(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<SeedSeqWrap>(info) {
    translate_errors(info.Env(), [&] {
      const auto entropy = to_u32_words(info[0]);
      const auto spawn_key = to_u32_words(info[1]);
      const auto pool_size = static_cast<std::size_t>(arg_int(info[2], "poolSize"));
      seq_ = std::make_unique<random::p13::SeedSeq>(entropy, spawn_key, pool_size);
    });
  }

  [[nodiscard]] const random::p13::SeedSeq& seq() const { return *seq_; }

 private:
  Napi::Value generate_state(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return translate_errors(env, [&]() -> Napi::Value {
      const auto n = static_cast<std::size_t>(arg_int(info[0], "n"));
      return u32_words_to_js(env, seq_->generate_u32(n));
    });
  }

  Napi::Value generate_state64(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return translate_errors(env, [&]() -> Napi::Value {
      const auto n = static_cast<std::size_t>(arg_int(info[0], "n"));
      return u64_words_to_js(env, seq_->generate_u64(n));
    });
  }

  std::unique_ptr<random::p13::SeedSeq> seq_;
};

// ---- Generic bit-generator wrapper (templated) ----

template <typename Gen>
class BitGenWrap : public Napi::ObjectWrap<BitGenWrap<Gen>> {
 public:
  using Base = Napi::ObjectWrap<BitGenWrap<Gen>>;

  static Napi::Function define(Napi::Env env, const char* name) {
    return Base::DefineClass(
        env, name,
        {Base::template InstanceMethod<&BitGenWrap::random>("random"),
         Base::template InstanceMethod<&BitGenWrap::get_state>("getState"),
         Base::template InstanceMethod<&BitGenWrap::set_state>("setState")});
  }

  explicit BitGenWrap(const Napi::CallbackInfo& info) : Base(info) {
    translate_errors(info.Env(), [&] { gen_ = make_gen(info); });
  }

  [[nodiscard]] random::p13::StatefulBitGen& gen() { return *gen_; }

 private:
  std::unique_ptr<random::p13::StatefulBitGen> make_gen(const Napi::CallbackInfo& info);

  Napi::Value random(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return translate_errors(env, [&]() -> Napi::Value {
      return Napi::Number::New(env, gen_->next_double());
    });
  }

  Napi::Value get_state(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return translate_errors(env, [&]() -> Napi::Value {
      return u64_words_to_js(env, gen_->get_state());
    });
  }

  Napi::Value set_state(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return translate_errors(env, [&]() -> Napi::Value {
      gen_->set_state(to_u64_words(info[0]));
      return env.Undefined();
    });
  }

  std::unique_ptr<random::p13::StatefulBitGen> gen_;
};

// ---- Specialised constructors ----

template <>
std::unique_ptr<random::p13::StatefulBitGen> BitGenWrap<random::p13::MT19937>::make_gen(
    const Napi::CallbackInfo& info) {
  auto mt = std::make_unique<random::p13::MT19937>();
  const SeedSeqWrap* ss = SeedSeqWrap::Unwrap(info[0].As<Napi::Object>());
  mt->seed_seq(ss->seq());
  return mt;
}

template <>
std::unique_ptr<random::p13::StatefulBitGen> BitGenWrap<random::p13::PCG64>::make_gen(
    const Napi::CallbackInfo& info) {
  const SeedSeqWrap* ss = SeedSeqWrap::Unwrap(info[0].As<Napi::Object>());
  return std::make_unique<random::p13::PCG64>(ss->seq(), false);
}

// PCG64DXSM uses PCG64 with dxsm=true; tag to distinguish.
struct PCG64DXSMTag {};

template <>
std::unique_ptr<random::p13::StatefulBitGen> BitGenWrap<PCG64DXSMTag>::make_gen(
    const Napi::CallbackInfo& info) {
  const SeedSeqWrap* ss = SeedSeqWrap::Unwrap(info[0].As<Napi::Object>());
  return std::make_unique<random::p13::PCG64>(ss->seq(), true);
}

template <>
std::unique_ptr<random::p13::StatefulBitGen> BitGenWrap<random::p13::Philox>::make_gen(
    const Napi::CallbackInfo& info) {
  const SeedSeqWrap* ss = SeedSeqWrap::Unwrap(info[0].As<Napi::Object>());
  return std::make_unique<random::p13::Philox>(&ss->seq(), std::vector<std::uint64_t>{},
                                               std::vector<std::uint64_t>{});
}

template <>
std::unique_ptr<random::p13::StatefulBitGen> BitGenWrap<random::p13::SFC64>::make_gen(
    const Napi::CallbackInfo& info) {
  const SeedSeqWrap* ss = SeedSeqWrap::Unwrap(info[0].As<Napi::Object>());
  return std::make_unique<random::p13::SFC64>(ss->seq());
}

}  // namespace

void init_p13_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object m = Napi::Object::New(env);
  m.Set("SeedSequence", SeedSeqWrap::define(env));
  m.Set("MT19937", BitGenWrap<random::p13::MT19937>::define(env, "MT19937"));
  m.Set("PCG64", BitGenWrap<random::p13::PCG64>::define(env, "PCG64"));
  m.Set("PCG64DXSM", BitGenWrap<PCG64DXSMTag>::define(env, "PCG64DXSM"));
  m.Set("Philox", BitGenWrap<random::p13::Philox>::define(env, "Philox"));
  m.Set("SFC64", BitGenWrap<random::p13::SFC64>::define(env, "SFC64"));
  exports.Set("p13", m);
}

}  // namespace nativpy::bindings
