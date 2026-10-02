#pragma once

// Scalar distribution kernels ported from NumPy 2.5.3 (BSD-3-Clause):
// numpy/random/src/distributions/{distributions,logfactorial,
// random_hypergeometric,random_mvhg_count,random_mvhg_marginals}.c and
// numpy/random/src/legacy/legacy-distributions.c (D-161, D-163).
// Every kernel consumes the bit stream in NumPy's order.

#include <cstddef>
#include <cstdint>

#include "bitgen.hpp"
#include "distributions.hpp"

namespace nativpy::random::p13 {

// NumPy binomial_t: cached per-(n, p) setup shared by the binomial kernels.
struct Binomial {
  bool has_binomial = false;
  double psave = 0.0;
  std::int64_t nsave = 0;
  double r = 0.0, q = 0.0, fm = 0.0;
  std::int64_t m = 0;
  double p1 = 0.0, xm = 0.0, xl = 0.0, xr = 0.0, c = 0.0, laml = 0.0, lamr = 0.0, p2 = 0.0,
         p3 = 0.0, p4 = 0.0;
};

// (int64_t)x as on arm64/aarch64 (saturating, NaN -> 0); avoids UB on overflow.
std::int64_t to_i64(double x) noexcept;

double loggam(double x);
double logfactorial(std::int64_t k);

// ---- Generator (numpy/random/src/distributions/distributions.c) ----
double standard_exponential(BitGen& g);
float standard_exponential_f(BitGen& g);
double standard_exponential_inv(BitGen& g);
float standard_exponential_inv_f(BitGen& g);
double standard_gamma(BitGen& g, double shape);
float standard_gamma_f(BitGen& g, float shape);
double gamma(BitGen& g, double shape, double scale);
double beta(BitGen& g, double a, double b);
double chisquare(BitGen& g, double df);
double f(BitGen& g, double dfnum, double dfden);
double standard_cauchy(BitGen& g);
double pareto(BitGen& g, double a);
double weibull(BitGen& g, double a);
double power(BitGen& g, double a);
double laplace(BitGen& g, double loc, double scale);
double gumbel(BitGen& g, double loc, double scale);
double logistic(BitGen& g, double loc, double scale);
double lognormal(BitGen& g, double mean, double sigma);
double rayleigh(BitGen& g, double mode);
double standard_t(BitGen& g, double df);
double noncentral_chisquare(BitGen& g, double df, double nonc);
double noncentral_f(BitGen& g, double dfnum, double dfden, double nonc);
double wald(BitGen& g, double mean, double scale);
double vonmises(BitGen& g, double mu, double kappa);
double triangular(BitGen& g, double left, double mode, double right);
double exponential(BitGen& g, double scale);
std::int64_t poisson(BitGen& g, double lam);
std::int64_t negative_binomial(BitGen& g, double n, double p);
std::int64_t binomial(BitGen& g, double p, std::int64_t n, Binomial& b);
std::int64_t logseries(BitGen& g, double p);
std::int64_t geometric(BitGen& g, double p);
std::int64_t zipf(BitGen& g, double a);
std::int64_t hypergeometric(BitGen& g, std::int64_t good, std::int64_t bad, std::int64_t sample);
void multinomial(BitGen& g, std::int64_t n, std::int64_t* mnix, const double* pix, std::int64_t d,
                 Binomial& b);
// variates: num_variates * num_colors zero-initialised int64 values.
void mvhg_count(BitGen& g, std::int64_t total, std::size_t num_colors, const std::int64_t* colors,
                std::int64_t nsample, std::size_t num_variates, std::int64_t* variates);
void mvhg_marginals(BitGen& g, std::int64_t total, std::size_t num_colors,
                    const std::int64_t* colors, std::int64_t nsample, std::size_t num_variates,
                    std::int64_t* variates);

// ---- RandomState (numpy/random/src/legacy/legacy-distributions.c) ----
namespace legacy {
double standard_exponential(BitGen& g);
double exponential(BitGen& g, double scale);
double standard_gamma(BitGen& g, LegacyGauss& gs, double shape);
double gamma(BitGen& g, LegacyGauss& gs, double shape, double scale);
double beta(BitGen& g, LegacyGauss& gs, double a, double b);
double chisquare(BitGen& g, LegacyGauss& gs, double df);
double f(BitGen& g, LegacyGauss& gs, double dfnum, double dfden);
double noncentral_chisquare(BitGen& g, LegacyGauss& gs, double df, double nonc);
double noncentral_f(BitGen& g, LegacyGauss& gs, double dfnum, double dfden, double nonc);
double standard_cauchy(BitGen& g, LegacyGauss& gs);
double standard_t(BitGen& g, LegacyGauss& gs, double df);
double pareto(BitGen& g, double a);
double weibull(BitGen& g, double a);
double power(BitGen& g, double a);
double lognormal(BitGen& g, LegacyGauss& gs, double mean, double sigma);
double rayleigh(BitGen& g, double mode);
double wald(BitGen& g, LegacyGauss& gs, double mean, double scale);
double vonmises(BitGen& g, double mu, double kappa);
std::int64_t negative_binomial(BitGen& g, LegacyGauss& gs, double n, double p);
std::int64_t binomial(BitGen& g, double p, std::int64_t n, Binomial& b);
std::int64_t hypergeometric(BitGen& g, std::int64_t good, std::int64_t bad, std::int64_t sample);
std::int64_t zipf(BitGen& g, double a);
std::int64_t geometric(BitGen& g, double p);
std::int64_t logseries(BitGen& g, double p);
void multinomial(BitGen& g, std::int64_t n, std::int64_t* mnix, const double* pix, std::int64_t d,
                 Binomial& b);
}  // namespace legacy

}  // namespace nativpy::random::p13
