#include "random_binding.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "bitgen.hpp"
#include "distributions.hpp"
#include "dtype_binding.hpp"
#include "error.hpp"
#include "error_binding.hpp"
#include "ndarray_binding.hpp"

namespace nativpy::bindings {

namespace {

using random::i128;

std::vector<std::uint32_t> to_words(const Napi::Value& v) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, "seed words must be an array");
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

Shape to_shape(const Napi::Value& v) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, "size must be an array of integers");
  const auto arr = v.As<Napi::Array>();
  Shape out(arr.Length());
  for (std::uint32_t i = 0; i < arr.Length(); ++i) {
    const double d = arr.Get(i).As<Napi::Number>().DoubleValue();
    if (d != static_cast<double>(static_cast<std::int64_t>(d))) {
      throw_error(ErrorKind::Value, "size must contain integers");
    }
    out[i] = static_cast<std::int64_t>(d);
  }
  return out;
}

// Integer bound from a JS BigInt or integral Number, as a 128-bit value.
i128 to_i128(const Napi::Value& v, const char* what) {
  if (v.IsBigInt()) {
    int sign = 0;
    std::size_t count = 2;
    std::uint64_t words[2] = {0, 0};
    v.As<Napi::BigInt>().ToWords(&sign, &count, words);
    if (count > 2 || (words[1] >> 63) != 0) {
      throw_error(ErrorKind::Value, std::string(what) + " is out of bounds for int64");
    }
    const i128 mag = (static_cast<i128>(words[1]) << 64) | static_cast<i128>(words[0]);
    return sign ? -mag : mag;
  }
  if (!v.IsNumber()) throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  const double d = v.As<Napi::Number>().DoubleValue();
  if (!(d > -1.8446744073709552e19 && d < 1.8446744073709552e19) ||
      d != static_cast<double>(static_cast<i128>(d))) {
    throw_error(ErrorKind::Value, std::string(what) + " must be an integer");
  }
  return static_cast<i128>(d);
}

double num(const Napi::Value& v) { return v.As<Napi::Number>().DoubleValue(); }

class BitGeneratorWrap : public Napi::ObjectWrap<BitGeneratorWrap> {
 public:
  static Napi::Function define(Napi::Env env) {
    return DefineClass(env, "BitGenerator",
                       {InstanceMethod<&BitGeneratorWrap::reseed>("reseed"),
                        InstanceMethod<&BitGeneratorWrap::random>("random"),
                        InstanceMethod<&BitGeneratorWrap::uniform>("uniform"),
                        InstanceMethod<&BitGeneratorWrap::normal>("normal"),
                        InstanceMethod<&BitGeneratorWrap::legacy_normal>("legacyNormal"),
                        InstanceMethod<&BitGeneratorWrap::integers>("integers"),
                        InstanceMethod<&BitGeneratorWrap::shuffle>("shuffle"),
                        InstanceMethod<&BitGeneratorWrap::choice_indices>("choiceIndices")});
  }

  // new BitGenerator(kind: 'pcg64' | 'mt19937', mode: 'seedseq' | 'int' | 'array', words)
  explicit BitGeneratorWrap(const Napi::CallbackInfo& info)
      : Napi::ObjectWrap<BitGeneratorWrap>(info) {
    translate_errors(info.Env(), [&] {
      kind_ = info[0].As<Napi::String>().Utf8Value();
      if (kind_ != "pcg64" && kind_ != "mt19937") {
        throw_error(ErrorKind::Value, "unknown bit generator '" + kind_ + "'");
      }
      do_seed(info[1].As<Napi::String>().Utf8Value(), to_words(info[2]));
    });
  }

 private:
  void do_seed(const std::string& mode, const std::vector<std::uint32_t>& words) {
    gauss_ = random::LegacyGauss{};
    if (kind_ == "pcg64") {
      if (mode != "seedseq") throw_error(ErrorKind::Value, "pcg64 requires seedseq mode");
      gen_ = std::make_unique<random::PCG64>(random::SeedSequence(words));
      return;
    }
    auto mt = std::make_unique<random::MT19937>();
    if (mode == "int") {
      if (words.size() != 1) throw_error(ErrorKind::Value, "int seed requires one word");
      mt->seed(words[0]);
    } else if (mode == "array") {
      if (words.empty()) throw_error(ErrorKind::Value, "Seed must be non-empty");
      mt->seed_by_array(words);
    } else {
      throw_error(ErrorKind::Value, "unknown seed mode '" + mode + "'");
    }
    gen_ = std::move(mt);
  }

#define NATIVPY_RANDOM_METHOD(NAME, BODY)                         \
  Napi::Value NAME(const Napi::CallbackInfo& info) {              \
    Napi::Env env = info.Env();                                   \
    return translate_errors(env, [&]() -> Napi::Value BODY);      \
  }

  NATIVPY_RANDOM_METHOD(reseed, {
    do_seed(info[0].As<Napi::String>().Utf8Value(), to_words(info[1]));
    return env.Undefined();
  })
  // random(size, dtype)
  NATIVPY_RANDOM_METHOD(random, {
    return NDArrayWrap::create(env, random::random_doubles(*gen_, to_shape(info[0]), parse_dtype(info[1])));
  })
  // uniform(low, high, size)
  NATIVPY_RANDOM_METHOD(uniform, {
    return NDArrayWrap::create(env, random::uniform(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  // normal(loc, scale, size, dtype)
  NATIVPY_RANDOM_METHOD(normal, {
    return NDArrayWrap::create(env, random::normal(*gen_, num(info[0]), num(info[1]),
                                                   to_shape(info[2]), parse_dtype(info[3])));
  })
  // legacyNormal(loc, scale, size)
  NATIVPY_RANDOM_METHOD(legacy_normal, {
    return NDArrayWrap::create(
        env, random::legacy_normal(*gen_, gauss_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  // integers(low, high, closed, size, dtype, masked)
  NATIVPY_RANDOM_METHOD(integers, {
    return NDArrayWrap::create(
        env, random::bounded_integers(*gen_, to_i128(info[0], "low"), to_i128(info[1], "high"),
                                      info[2].ToBoolean().Value(), to_shape(info[3]),
                                      parse_dtype(info[4]), info[5].ToBoolean().Value()));
  })
  // shuffle(ndarray) in place along axis 0.
  NATIVPY_RANDOM_METHOD(shuffle, {
    random::shuffle(*gen_, NDArrayWrap::unwrap(info[0]));
    return env.Undefined();
  })
  // choiceIndices(popSize, size, replace, shuffle)
  NATIVPY_RANDOM_METHOD(choice_indices, {
    const auto pop = static_cast<std::int64_t>(to_i128(info[0], "a"));
    return NDArrayWrap::create(
        env, random::generator_choice_indices(*gen_, pop, to_shape(info[1]),
                                              info[2].ToBoolean().Value(),
                                              info[3].ToBoolean().Value()));
  })

#undef NATIVPY_RANDOM_METHOD

  std::string kind_;
  std::unique_ptr<random::BitGen> gen_;
  random::LegacyGauss gauss_;
};

}  // namespace

void init_random_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object ns = Napi::Object::New(env);
  ns.Set("BitGenerator", BitGeneratorWrap::define(env));
  exports.Set("random", ns);
}

}  // namespace nativpy::bindings
