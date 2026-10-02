#include "random_binding.hpp"

#include <algorithm>
#include <cstring>
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
#include "p13_bitgen.hpp"
#include "p13_distributions.hpp"

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
std::int64_t i64(const Napi::Value& v) {
  return static_cast<std::int64_t>(v.As<Napi::Number>().DoubleValue());
}

// Decode a JS Array of BigInt into a vector of uint64_t (for setState).
std::vector<std::uint64_t> to_u64_words(const Napi::Value& v) {
  if (!v.IsArray()) throw_error(ErrorKind::Value, "state words must be an array");
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

class BitGeneratorWrap : public Napi::ObjectWrap<BitGeneratorWrap> {
 public:
  static Napi::Function define(Napi::Env env) {
    return DefineClass(
        env, "BitGenerator",
        {
            InstanceMethod<&BitGeneratorWrap::reseed>("reseed"),
            InstanceMethod<&BitGeneratorWrap::random>("random"),
            InstanceMethod<&BitGeneratorWrap::uniform>("uniform"),
            InstanceMethod<&BitGeneratorWrap::normal>("normal"),
            InstanceMethod<&BitGeneratorWrap::legacy_normal>("legacyNormal"),
            InstanceMethod<&BitGeneratorWrap::integers>("integers"),
            InstanceMethod<&BitGeneratorWrap::shuffle>("shuffle"),
            InstanceMethod<&BitGeneratorWrap::choice_indices>("choiceIndices"),
            // P13-3: discrete Generator distributions
            InstanceMethod<&BitGeneratorWrap::gen_binomial>("binomial"),
            InstanceMethod<&BitGeneratorWrap::gen_negative_binomial>("negativeBinomial"),
            InstanceMethod<&BitGeneratorWrap::gen_poisson>("poisson"),
            InstanceMethod<&BitGeneratorWrap::gen_zipf>("zipf"),
            InstanceMethod<&BitGeneratorWrap::gen_geometric>("geometric"),
            InstanceMethod<&BitGeneratorWrap::gen_hypergeometric>("hypergeometric"),
            InstanceMethod<&BitGeneratorWrap::gen_logseries>("logseries"),
            // P13-2: continuous Generator distributions
            InstanceMethod<&BitGeneratorWrap::gen_standard_exponential>("standardExponential"),
            InstanceMethod<&BitGeneratorWrap::gen_standard_gamma>("standardGamma"),
            InstanceMethod<&BitGeneratorWrap::gen_gamma>("gamma"),
            InstanceMethod<&BitGeneratorWrap::gen_beta>("beta"),
            InstanceMethod<&BitGeneratorWrap::gen_chisquare>("chisquare"),
            InstanceMethod<&BitGeneratorWrap::gen_f>("f"),
            InstanceMethod<&BitGeneratorWrap::gen_standard_cauchy>("standardCauchy"),
            InstanceMethod<&BitGeneratorWrap::gen_pareto>("pareto"),
            InstanceMethod<&BitGeneratorWrap::gen_weibull>("weibull"),
            InstanceMethod<&BitGeneratorWrap::gen_power>("power"),
            InstanceMethod<&BitGeneratorWrap::gen_laplace>("laplace"),
            InstanceMethod<&BitGeneratorWrap::gen_gumbel>("gumbel"),
            InstanceMethod<&BitGeneratorWrap::gen_logistic>("logistic"),
            InstanceMethod<&BitGeneratorWrap::gen_lognormal>("lognormal"),
            InstanceMethod<&BitGeneratorWrap::gen_rayleigh>("rayleigh"),
            InstanceMethod<&BitGeneratorWrap::gen_standard_t>("standardT"),
            InstanceMethod<&BitGeneratorWrap::gen_noncentral_chisquare>("noncentralChisquare"),
            InstanceMethod<&BitGeneratorWrap::gen_noncentral_f>("noncentralF"),
            InstanceMethod<&BitGeneratorWrap::gen_wald>("wald"),
            InstanceMethod<&BitGeneratorWrap::gen_vonmises>("vonmises"),
            InstanceMethod<&BitGeneratorWrap::gen_triangular>("triangular"),
            InstanceMethod<&BitGeneratorWrap::gen_exponential>("exponential"),
            // P13-5: RandomState / legacy distributions
            InstanceMethod<&BitGeneratorWrap::get_state>("getState"),
            InstanceMethod<&BitGeneratorWrap::set_state>("setState"),
            InstanceMethod<&BitGeneratorWrap::get_bytes>("bytes"),
            InstanceMethod<&BitGeneratorWrap::legacy_standard_exponential>("legacyStandardExponential"),
            InstanceMethod<&BitGeneratorWrap::legacy_exponential>("legacyExponential"),
            InstanceMethod<&BitGeneratorWrap::legacy_standard_gamma>("legacyStandardGamma"),
            InstanceMethod<&BitGeneratorWrap::legacy_gamma>("legacyGamma"),
            InstanceMethod<&BitGeneratorWrap::legacy_beta>("legacyBeta"),
            InstanceMethod<&BitGeneratorWrap::legacy_chisquare>("legacyChisquare"),
            InstanceMethod<&BitGeneratorWrap::legacy_f>("legacyF"),
            InstanceMethod<&BitGeneratorWrap::legacy_noncentral_chisquare>("legacyNoncentralChisquare"),
            InstanceMethod<&BitGeneratorWrap::legacy_noncentral_f>("legacyNoncentralF"),
            InstanceMethod<&BitGeneratorWrap::legacy_standard_cauchy>("legacyStandardCauchy"),
            InstanceMethod<&BitGeneratorWrap::legacy_standard_t>("legacyStandardT"),
            InstanceMethod<&BitGeneratorWrap::legacy_pareto>("legacyPareto"),
            InstanceMethod<&BitGeneratorWrap::legacy_weibull>("legacyWeibull"),
            InstanceMethod<&BitGeneratorWrap::legacy_power>("legacyPower"),
            InstanceMethod<&BitGeneratorWrap::legacy_lognormal>("legacyLognormal"),
            InstanceMethod<&BitGeneratorWrap::legacy_rayleigh>("legacyRayleigh"),
            InstanceMethod<&BitGeneratorWrap::legacy_wald>("legacyWald"),
            InstanceMethod<&BitGeneratorWrap::legacy_vonmises>("legacyVonmises"),
            InstanceMethod<&BitGeneratorWrap::legacy_negative_binomial>("legacyNegativeBinomial"),
            InstanceMethod<&BitGeneratorWrap::legacy_binomial>("legacyBinomial"),
            InstanceMethod<&BitGeneratorWrap::legacy_hypergeometric>("legacyHypergeometric"),
            InstanceMethod<&BitGeneratorWrap::legacy_zipf>("legacyZipf"),
            InstanceMethod<&BitGeneratorWrap::legacy_geometric>("legacyGeometric"),
            InstanceMethod<&BitGeneratorWrap::legacy_logseries>("legacyLogseries"),
            InstanceMethod<&BitGeneratorWrap::legacy_choice_p>("legacyChoiceP"),
            // P13-4 (D-162): multivariate distributions, choice(p=), permuted
            InstanceMethod<&BitGeneratorWrap::gen_multinomial>("multinomial"),
            InstanceMethod<&BitGeneratorWrap::gen_dirichlet>("dirichlet"),
            InstanceMethod<&BitGeneratorWrap::gen_mvhg_count>("mvhgCount"),
            InstanceMethod<&BitGeneratorWrap::gen_mvhg_marginals>("mvhgMarginals"),
            InstanceMethod<&BitGeneratorWrap::gen_choice_p>("choiceP"),
            InstanceMethod<&BitGeneratorWrap::gen_permuted>("permuted"),
        });
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
    binom_ = random::p13::Binomial{};
    if (kind_ == "pcg64") {
      if (mode != "seedseq") throw_error(ErrorKind::Value, "pcg64 requires seedseq mode");
      gen_ = std::make_unique<random::PCG64>(random::SeedSequence(words));
      mt_gen_ = nullptr;
      return;
    }
    // mt19937
    auto mt = std::make_unique<random::p13::MT19937>();
    if (mode == "int") {
      if (words.size() != 1) throw_error(ErrorKind::Value, "int seed requires one word");
      mt->seed(words[0]);
    } else if (mode == "array") {
      if (words.empty()) throw_error(ErrorKind::Value, "Seed must be non-empty");
      mt->seed_by_array(words);
    } else {
      throw_error(ErrorKind::Value, "unknown seed mode '" + mode + "'");
    }
    mt_gen_ = mt.get();
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

  // ---- P13-3: discrete Generator distributions ----
  // genBinomial(n, p, size[])
  NATIVPY_RANDOM_METHOD(gen_binomial, {
    return NDArrayWrap::create(
        env, random::p13::arr_binomial(*gen_, num(info[1]), i64(info[0]), binom_, to_shape(info[2])));
  })
  // genNegativeBinomial(n, p, size[])
  NATIVPY_RANDOM_METHOD(gen_negative_binomial, {
    return NDArrayWrap::create(
        env, random::p13::arr_negative_binomial(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  // genPoisson(lam, size[])
  NATIVPY_RANDOM_METHOD(gen_poisson, {
    return NDArrayWrap::create(env, random::p13::arr_poisson(*gen_, num(info[0]), to_shape(info[1])));
  })
  // genZipf(a, size[])
  NATIVPY_RANDOM_METHOD(gen_zipf, {
    return NDArrayWrap::create(env, random::p13::arr_zipf(*gen_, num(info[0]), to_shape(info[1])));
  })
  // genGeometric(p, size[])
  NATIVPY_RANDOM_METHOD(gen_geometric, {
    return NDArrayWrap::create(env, random::p13::arr_geometric(*gen_, num(info[0]), to_shape(info[1])));
  })
  // genHypergeometric(good, bad, sample, size[])
  NATIVPY_RANDOM_METHOD(gen_hypergeometric, {
    return NDArrayWrap::create(
        env, random::p13::arr_hypergeometric(*gen_, i64(info[0]), i64(info[1]), i64(info[2]),
                                             to_shape(info[3])));
  })
  // genLogseries(p, size[])
  NATIVPY_RANDOM_METHOD(gen_logseries, {
    return NDArrayWrap::create(env, random::p13::arr_logseries(*gen_, num(info[0]), to_shape(info[1])));
  })

  // ---- P13-2: continuous Generator distributions ----
  NATIVPY_RANDOM_METHOD(gen_standard_exponential, {
    return NDArrayWrap::create(env, random::p13::arr_standard_exponential(*gen_, to_shape(info[0])));
  })
  NATIVPY_RANDOM_METHOD(gen_standard_gamma, {
    return NDArrayWrap::create(env, random::p13::arr_standard_gamma(*gen_, num(info[0]), to_shape(info[1])));
  })
  NATIVPY_RANDOM_METHOD(gen_gamma, {
    return NDArrayWrap::create(env, random::p13::arr_gamma(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_beta, {
    return NDArrayWrap::create(env, random::p13::arr_beta(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_chisquare, {
    return NDArrayWrap::create(env, random::p13::arr_chisquare(*gen_, num(info[0]), to_shape(info[1])));
  })
  NATIVPY_RANDOM_METHOD(gen_f, {
    return NDArrayWrap::create(env, random::p13::arr_f(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_standard_cauchy, {
    return NDArrayWrap::create(env, random::p13::arr_standard_cauchy(*gen_, to_shape(info[0])));
  })
  NATIVPY_RANDOM_METHOD(gen_pareto, {
    return NDArrayWrap::create(env, random::p13::arr_pareto(*gen_, num(info[0]), to_shape(info[1])));
  })
  NATIVPY_RANDOM_METHOD(gen_weibull, {
    return NDArrayWrap::create(env, random::p13::arr_weibull(*gen_, num(info[0]), to_shape(info[1])));
  })
  NATIVPY_RANDOM_METHOD(gen_power, {
    return NDArrayWrap::create(env, random::p13::arr_power(*gen_, num(info[0]), to_shape(info[1])));
  })
  NATIVPY_RANDOM_METHOD(gen_laplace, {
    return NDArrayWrap::create(env, random::p13::arr_laplace(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_gumbel, {
    return NDArrayWrap::create(env, random::p13::arr_gumbel(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_logistic, {
    return NDArrayWrap::create(env, random::p13::arr_logistic(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_lognormal, {
    return NDArrayWrap::create(env, random::p13::arr_lognormal(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_rayleigh, {
    return NDArrayWrap::create(env, random::p13::arr_rayleigh(*gen_, num(info[0]), to_shape(info[1])));
  })
  NATIVPY_RANDOM_METHOD(gen_standard_t, {
    return NDArrayWrap::create(env, random::p13::arr_standard_t(*gen_, num(info[0]), to_shape(info[1])));
  })
  NATIVPY_RANDOM_METHOD(gen_noncentral_chisquare, {
    return NDArrayWrap::create(
        env, random::p13::arr_noncentral_chisquare(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_noncentral_f, {
    return NDArrayWrap::create(
        env, random::p13::arr_noncentral_f(*gen_, num(info[0]), num(info[1]), num(info[2]), to_shape(info[3])));
  })
  NATIVPY_RANDOM_METHOD(gen_wald, {
    return NDArrayWrap::create(env, random::p13::arr_wald(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_vonmises, {
    return NDArrayWrap::create(env, random::p13::arr_vonmises(*gen_, num(info[0]), num(info[1]), to_shape(info[2])));
  })
  NATIVPY_RANDOM_METHOD(gen_triangular, {
    return NDArrayWrap::create(
        env, random::p13::arr_triangular(*gen_, num(info[0]), num(info[1]), num(info[2]), to_shape(info[3])));
  })
  NATIVPY_RANDOM_METHOD(gen_exponential, {
    return NDArrayWrap::create(env, random::p13::arr_exponential(*gen_, num(info[0]), to_shape(info[1])));
  })

  // ---- P13-5: RandomState state + bytes ----

  // getState() -> BigInt[] — only available for mt19937
  NATIVPY_RANDOM_METHOD(get_state, {
    if (mt_gen_ == nullptr) {
      throw_error(ErrorKind::Value, "getState is only supported for mt19937");
    }
    return u64_words_to_js(env, mt_gen_->get_state());
  })

  // setState(words: BigInt[]) — only available for mt19937
  NATIVPY_RANDOM_METHOD(set_state, {
    if (mt_gen_ == nullptr) {
      throw_error(ErrorKind::Value, "setState is only supported for mt19937");
    }
    gauss_ = random::LegacyGauss{};
    binom_ = random::p13::Binomial{};
    mt_gen_->set_state(to_u64_words(info[0]));
    return env.Undefined();
  })

  // bytes(length: number) -> Uint8Array
  NATIVPY_RANDOM_METHOD(get_bytes, {
    const auto n = static_cast<std::size_t>(info[0].As<Napi::Number>().DoubleValue());
    // Allocate output buffer
    auto buf = Napi::Buffer<std::uint8_t>::New(env, n);
    std::uint8_t* out = buf.Data();
    std::size_t i = 0;
    // Generate 4 bytes per MT19937 uint32 draw (legacy uint32 stream)
    while (i + 4 <= n) {
      const std::uint32_t v = gen_->next_uint32();
      // little-endian byte order (matches NumPy RandomState.bytes)
      out[i + 0] = static_cast<std::uint8_t>(v & 0xff);
      out[i + 1] = static_cast<std::uint8_t>((v >> 8) & 0xff);
      out[i + 2] = static_cast<std::uint8_t>((v >> 16) & 0xff);
      out[i + 3] = static_cast<std::uint8_t>((v >> 24) & 0xff);
      i += 4;
    }
    if (i < n) {
      const std::uint32_t v = gen_->next_uint32();
      for (std::size_t b = 0; i < n; ++b, ++i) {
        out[i] = static_cast<std::uint8_t>((v >> (8 * b)) & 0xff);
      }
    }
    return buf;
  })

  // ---- P13-5: RandomState legacy distributions ----
  // All delegate to the p13::legacy:: namespace which uses legacy Gauss cache.

  // legacyStandardExponential(size[])
  NATIVPY_RANDOM_METHOD(legacy_standard_exponential, {
    const auto s = to_shape(info[0]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::standard_exponential(*gen_); }));
  })
  // legacyExponential(scale, size[])
  NATIVPY_RANDOM_METHOD(legacy_exponential, {
    const double scale = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::exponential(*gen_, scale); }));
  })
  // legacyStandardGamma(shape, size[])
  NATIVPY_RANDOM_METHOD(legacy_standard_gamma, {
    const double shape = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::standard_gamma(*gen_, gauss_, shape); }));
  })
  // legacyGamma(shape, scale, size[])
  NATIVPY_RANDOM_METHOD(legacy_gamma, {
    const double shape = num(info[0]);
    const double scale = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::gamma(*gen_, gauss_, shape, scale); }));
  })
  // legacyBeta(a, b, size[])
  NATIVPY_RANDOM_METHOD(legacy_beta, {
    const double a = num(info[0]);
    const double b = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::beta(*gen_, gauss_, a, b); }));
  })
  // legacyChisquare(df, size[])
  NATIVPY_RANDOM_METHOD(legacy_chisquare, {
    const double df = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::chisquare(*gen_, gauss_, df); }));
  })
  // legacyF(dfnum, dfden, size[])
  NATIVPY_RANDOM_METHOD(legacy_f, {
    const double dfnum = num(info[0]);
    const double dfden = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::f(*gen_, gauss_, dfnum, dfden); }));
  })
  // legacyNoncentralChisquare(df, nonc, size[])
  NATIVPY_RANDOM_METHOD(legacy_noncentral_chisquare, {
    const double df = num(info[0]);
    const double nonc = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::noncentral_chisquare(*gen_, gauss_, df, nonc); }));
  })
  // legacyNoncentralF(dfnum, dfden, nonc, size[])
  NATIVPY_RANDOM_METHOD(legacy_noncentral_f, {
    const double dfnum = num(info[0]);
    const double dfden = num(info[1]);
    const double nonc = num(info[2]);
    const auto s = to_shape(info[3]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::noncentral_f(*gen_, gauss_, dfnum, dfden, nonc); }));
  })
  // legacyStandardCauchy(size[])
  NATIVPY_RANDOM_METHOD(legacy_standard_cauchy, {
    const auto s = to_shape(info[0]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::standard_cauchy(*gen_, gauss_); }));
  })
  // legacyStandardT(df, size[])
  NATIVPY_RANDOM_METHOD(legacy_standard_t, {
    const double df = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::standard_t(*gen_, gauss_, df); }));
  })
  // legacyPareto(a, size[])
  NATIVPY_RANDOM_METHOD(legacy_pareto, {
    const double a = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::pareto(*gen_, a); }));
  })
  // legacyWeibull(a, size[])
  NATIVPY_RANDOM_METHOD(legacy_weibull, {
    const double a = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::weibull(*gen_, a); }));
  })
  // legacyPower(a, size[])
  NATIVPY_RANDOM_METHOD(legacy_power, {
    const double a = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::power(*gen_, a); }));
  })
  // legacyLognormal(mean, sigma, size[])
  NATIVPY_RANDOM_METHOD(legacy_lognormal, {
    const double mean = num(info[0]);
    const double sigma = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::lognormal(*gen_, gauss_, mean, sigma); }));
  })
  // legacyRayleigh(mode, size[])
  NATIVPY_RANDOM_METHOD(legacy_rayleigh, {
    const double mode = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::rayleigh(*gen_, mode); }));
  })
  // legacyWald(mean, scale, size[])
  NATIVPY_RANDOM_METHOD(legacy_wald, {
    const double mean = num(info[0]);
    const double scale = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::wald(*gen_, gauss_, mean, scale); }));
  })
  // legacyVonmises(mu, kappa, size[])
  NATIVPY_RANDOM_METHOD(legacy_vonmises, {
    const double mu = num(info[0]);
    const double kappa = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_f64(
        s, [&] { return random::p13::legacy::vonmises(*gen_, mu, kappa); }));
  })
  // legacyNegativeBinomial(n, p, size[])
  NATIVPY_RANDOM_METHOD(legacy_negative_binomial, {
    const double n = num(info[0]);
    const double p = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_i64(
        s, [&] { return random::p13::legacy::negative_binomial(*gen_, gauss_, n, p); }));
  })
  // legacyBinomial(n, p, size[])
  NATIVPY_RANDOM_METHOD(legacy_binomial, {
    const std::int64_t n = i64(info[0]);
    const double p = num(info[1]);
    const auto s = to_shape(info[2]);
    return NDArrayWrap::create(env, random::p13::detail::fill_i64(
        s, [&] { return random::p13::legacy::binomial(*gen_, p, n, binom_); }));
  })
  // legacyHypergeometric(good, bad, sample, size[])
  NATIVPY_RANDOM_METHOD(legacy_hypergeometric, {
    const std::int64_t good = i64(info[0]);
    const std::int64_t bad = i64(info[1]);
    const std::int64_t sample = i64(info[2]);
    const auto s = to_shape(info[3]);
    return NDArrayWrap::create(env, random::p13::detail::fill_i64(
        s, [&] { return random::p13::legacy::hypergeometric(*gen_, good, bad, sample); }));
  })
  // legacyZipf(a, size[])
  NATIVPY_RANDOM_METHOD(legacy_zipf, {
    const double a = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_i64(
        s, [&] { return random::p13::legacy::zipf(*gen_, a); }));
  })
  // legacyGeometric(p, size[])
  NATIVPY_RANDOM_METHOD(legacy_geometric, {
    const double p = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_i64(
        s, [&] { return random::p13::legacy::geometric(*gen_, p); }));
  })
  // legacyLogseries(p, size[])
  NATIVPY_RANDOM_METHOD(legacy_logseries, {
    const double p = num(info[0]);
    const auto s = to_shape(info[1]);
    return NDArrayWrap::create(env, random::p13::detail::fill_i64(
        s, [&] { return random::p13::legacy::logseries(*gen_, p); }));
  })
  // legacyChoiceP(cdf: Float64Array, size[]) -> int64 indices via lower_bound search
  // cdf is a pre-computed inclusive CDF (length = population size); TS fills this from p.
  NATIVPY_RANDOM_METHOD(legacy_choice_p, {
    const NDArray cdf_arr = NDArrayWrap::unwrap(info[0]);
    const auto s = to_shape(info[1]);
    const std::size_t pop = static_cast<std::size_t>(cdf_arr.shape()[0]);
    const auto* cdf = reinterpret_cast<const double*>(cdf_arr.data());
    return NDArrayWrap::create(env, random::p13::detail::fill_i64(s, [&]() -> std::int64_t {
      const double u = gen_->next_double();
      // lower_bound: first index where cdf[i] > u
      const auto* it = std::lower_bound(cdf, cdf + pop, u);
      std::ptrdiff_t idx = it - cdf;
      if (idx >= static_cast<std::ptrdiff_t>(pop)) idx = static_cast<std::ptrdiff_t>(pop) - 1;
      return static_cast<std::int64_t>(idx);
    }));
  })

  // ---- P13-4 (D-162): multivariate distributions, choice(p=), permuted ----

  // genMultinomial(n, pvals: number[], size: number[]) -> int64[*size, d]
  NATIVPY_RANDOM_METHOD(gen_multinomial, {
    const auto n = static_cast<std::int64_t>(to_i128(info[0], "n"));
    if (!info[1].IsArray()) throw_error(ErrorKind::Value, "pvals must be an array");
    const auto parr = info[1].As<Napi::Array>();
    const auto d = static_cast<std::int64_t>(parr.Length());
    std::vector<double> pv(static_cast<std::size_t>(d));
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(d); ++i) pv[i] = num(parr.Get(i));
    const Shape base = to_shape(info[2]);
    Shape out_shape(base.size() + 1);
    for (std::size_t i = 0; i < base.size(); ++i) out_shape[i] = base[i];
    out_shape.back() = d;
    std::int64_t outer = 1;
    for (auto s : base) outer *= s;
    NDArray out = NDArray::zeros(out_shape, DType::Int64);
    auto* dst = reinterpret_cast<std::int64_t*>(out.data());
    random::p13::Binomial b{};
    for (std::int64_t k = 0; k < outer; ++k)
      random::p13::multinomial(*gen_, n, dst + k * d, pv.data(), d, b);
    return NDArrayWrap::create(env, std::move(out));
  })

  // genDirichlet(alpha: number[], size: number[]) -> float64[*size, d]
  // Draws gamma(alpha[i], 1) samples then normalises (Dirichlet-by-gamma method).
  NATIVPY_RANDOM_METHOD(gen_dirichlet, {
    if (!info[0].IsArray()) throw_error(ErrorKind::Value, "alpha must be an array");
    const auto aarr = info[0].As<Napi::Array>();
    const auto d = static_cast<std::int64_t>(aarr.Length());
    if (d == 0) throw_error(ErrorKind::Value, "alpha must be non-empty");
    std::vector<double> alpha(static_cast<std::size_t>(d));
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(d); ++i) alpha[i] = num(aarr.Get(i));
    const Shape base = to_shape(info[1]);
    Shape out_shape(base.size() + 1);
    for (std::size_t i = 0; i < base.size(); ++i) out_shape[i] = base[i];
    out_shape.back() = d;
    std::int64_t outer = 1;
    for (auto s : base) outer *= s;
    NDArray out = NDArray::zeros(out_shape, DType::Float64);
    auto* dst = reinterpret_cast<double*>(out.data());
    for (std::int64_t k = 0; k < outer; ++k) {
      double sum = 0.0;
      for (std::int64_t j = 0; j < d; ++j) {
        const double g = random::p13::gamma(*gen_, alpha[static_cast<std::size_t>(j)], 1.0);
        dst[k * d + j] = g;
        sum += g;
      }
      if (sum > 0.0)
        for (std::int64_t j = 0; j < d; ++j) dst[k * d + j] /= sum;
    }
    return NDArrayWrap::create(env, std::move(out));
  })

  // genMvhgCount(colors: number[], nsample, size: number[]) -> int64[*size, nc]
  NATIVPY_RANDOM_METHOD(gen_mvhg_count, {
    if (!info[0].IsArray()) throw_error(ErrorKind::Value, "colors must be an array");
    const auto carr = info[0].As<Napi::Array>();
    const auto nc = static_cast<std::size_t>(carr.Length());
    std::vector<std::int64_t> colors(nc);
    std::int64_t total = 0;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(nc); ++i) {
      colors[i] = static_cast<std::int64_t>(num(carr.Get(i)));
      total += colors[i];
    }
    const auto ns = static_cast<std::int64_t>(to_i128(info[1], "nsample"));
    const Shape base = to_shape(info[2]);
    Shape out_shape(base.size() + 1);
    for (std::size_t i = 0; i < base.size(); ++i) out_shape[i] = base[i];
    out_shape.back() = static_cast<std::int64_t>(nc);
    std::size_t outer = 1;
    for (auto s : base) outer *= static_cast<std::size_t>(s);
    NDArray out = NDArray::zeros(out_shape, DType::Int64);
    auto* dst = reinterpret_cast<std::int64_t*>(out.data());
    random::p13::mvhg_count(*gen_, total, nc, colors.data(), ns, outer, dst);
    return NDArrayWrap::create(env, std::move(out));
  })

  // genMvhgMarginals(colors: number[], nsample, size: number[]) -> int64[*size, nc]
  NATIVPY_RANDOM_METHOD(gen_mvhg_marginals, {
    if (!info[0].IsArray()) throw_error(ErrorKind::Value, "colors must be an array");
    const auto carr = info[0].As<Napi::Array>();
    const auto nc = static_cast<std::size_t>(carr.Length());
    std::vector<std::int64_t> colors(nc);
    std::int64_t total = 0;
    for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(nc); ++i) {
      colors[i] = static_cast<std::int64_t>(num(carr.Get(i)));
      total += colors[i];
    }
    const auto ns = static_cast<std::int64_t>(to_i128(info[1], "nsample"));
    const Shape base = to_shape(info[2]);
    Shape out_shape(base.size() + 1);
    for (std::size_t i = 0; i < base.size(); ++i) out_shape[i] = base[i];
    out_shape.back() = static_cast<std::int64_t>(nc);
    std::size_t outer = 1;
    for (auto s : base) outer *= static_cast<std::size_t>(s);
    NDArray out = NDArray::zeros(out_shape, DType::Int64);
    auto* dst = reinterpret_cast<std::int64_t*>(out.data());
    random::p13::mvhg_marginals(*gen_, total, nc, colors.data(), ns, outer, dst);
    return NDArrayWrap::create(env, std::move(out));
  })

  // genChoiceP(popSize, size: number[], cdf: number[]) -> int64[*size] indices
  // cdf is a pre-built cumulative sum (validated on TS side).
  // (Not using NATIVPY_RANDOM_METHOD: body contains commas at top level.)
  Napi::Value gen_choice_p(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return translate_errors(env, [&]() -> Napi::Value {
      const auto pop = static_cast<std::size_t>(to_i128(info[0], "popSize"));
      const Shape shape = to_shape(info[1]);
      if (!info[2].IsArray()) throw_error(ErrorKind::Value, "cdf must be an array");
      const auto cdf_arr = info[2].As<Napi::Array>();
      if (cdf_arr.Length() != pop)
        throw_error(ErrorKind::Value, "cdf must have the same length as popSize");
      std::vector<double> cdf(pop);
      for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(pop); ++i)
        cdf[i] = num(cdf_arr.Get(i));
      cdf.back() = 1.0;
      std::int64_t n_out = 1;
      for (auto s : shape) n_out *= s;
      NDArray out = NDArray::zeros(shape, DType::Int64);
      auto* dst = reinterpret_cast<std::int64_t*>(out.data());
      for (std::int64_t k = 0; k < n_out; ++k) {
        const double u = gen_->next_double();
        const auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        dst[k] = static_cast<std::int64_t>(it - cdf.begin());
      }
      return NDArrayWrap::create(env, std::move(out));
    });
  }

  // genPermuted(x: NDArray, axis: number) -> new NDArray with lanes independently shuffled
  // (Not using NATIVPY_RANDOM_METHOD: body contains commas at top level.)
  Napi::Value gen_permuted(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return translate_errors(env, [&]() -> Napi::Value {
      NDArray src = NDArrayWrap::unwrap(info[0]).copy();
      const int axis = info[1].As<Napi::Number>().Int32Value();
      const int ndim = static_cast<int>(src.ndim());
      if (ndim == 0) throw_error(ErrorKind::Value, "x must be at least 1-dimensional");
      const int ax = axis < 0 ? ndim + axis : axis;
      if (ax < 0 || ax >= ndim) throw_error(ErrorKind::Value, "axis out of bounds");
      const auto& sh = src.shape();
      std::int64_t outer = 1;
      std::int64_t inner = 1;
      const std::int64_t n = sh[static_cast<std::size_t>(ax)];
      for (int i = 0; i < ax; ++i) outer *= sh[static_cast<std::size_t>(i)];
      for (int i = ax + 1; i < ndim; ++i) inner *= sh[static_cast<std::size_t>(i)];
      const std::size_t lane = static_cast<std::size_t>(inner) * src.itemsize();
      std::byte* base_ptr = src.data();
      std::vector<std::byte> tmp(lane);
      for (std::int64_t o = 0; o < outer; ++o) {
        std::byte* row = base_ptr + static_cast<std::size_t>(o * n) * lane;
        for (std::int64_t i = n - 1; i > 0; --i) {
          const std::size_t j =
              static_cast<std::size_t>(random_interval(*gen_, static_cast<std::uint64_t>(i)));
          if (static_cast<std::int64_t>(j) != i) {
            std::memcpy(tmp.data(), row + static_cast<std::size_t>(i) * lane, lane);
            std::memcpy(row + static_cast<std::size_t>(i) * lane, row + j * lane, lane);
            std::memcpy(row + j * lane, tmp.data(), lane);
          }
        }
      }
      return NDArrayWrap::create(env, std::move(src));
    });
  }

#undef NATIVPY_RANDOM_METHOD

  std::string kind_;
  std::unique_ptr<random::BitGen> gen_;
  // Non-owning pointer to the MT19937 sub-object when kind_=="mt19937"; null otherwise.
  random::p13::MT19937* mt_gen_ = nullptr;
  random::LegacyGauss gauss_;
  random::p13::Binomial binom_;
};

}  // namespace

void init_random_binding(Napi::Env env, Napi::Object exports) {
  Napi::Object ns = Napi::Object::New(env);
  ns.Set("BitGenerator", BitGeneratorWrap::define(env));
  exports.Set("random", ns);
}

}  // namespace nativpy::bindings
