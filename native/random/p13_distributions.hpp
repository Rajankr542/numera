#pragma once

// Scalar distribution kernels ported from NumPy 2.5.3 (BSD-3-Clause):
// numpy/random/src/distributions/{distributions,logfactorial,
// random_hypergeometric,random_mvhg_count,random_mvhg_marginals}.c and
// numpy/random/src/legacy/legacy-distributions.c (D-161, D-163).
// Every kernel consumes the bit stream in NumPy's order.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>

#include "bitgen.hpp"
#include "distributions.hpp"
#include "ndarray.hpp"

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

// ---- Array-filling helpers (P13-2, D-161): fill float64/int64 NDArrays ----
// Each `arr_*` function calls the corresponding scalar kernel for every element.

namespace detail {
inline NDArray fill_f64(const Shape& shape, const std::function<double()>& fn) {
  const std::size_t n = static_cast<std::size_t>(
      std::accumulate(shape.begin(), shape.end(), std::int64_t{1},
                      std::multiplies<std::int64_t>{}));
  NDArray out = NDArray::zeros(shape, DType::Float64);
  auto* d = reinterpret_cast<double*>(out.data());
  for (std::size_t i = 0; i < n; ++i) d[i] = fn();
  return out;
}
inline NDArray fill_i64(const Shape& shape, const std::function<std::int64_t()>& fn) {
  const std::size_t n = static_cast<std::size_t>(
      std::accumulate(shape.begin(), shape.end(), std::int64_t{1},
                      std::multiplies<std::int64_t>{}));
  NDArray out = NDArray::zeros(shape, DType::Int64);
  auto* d = reinterpret_cast<std::int64_t*>(out.data());
  for (std::size_t i = 0; i < n; ++i) d[i] = fn();
  return out;
}
}  // namespace detail

inline NDArray arr_standard_exponential(BitGen& g, const Shape& s) {
  return detail::fill_f64(s, [&] { return standard_exponential(g); });
}
inline NDArray arr_standard_gamma(BitGen& g, double sh, const Shape& s) {
  return detail::fill_f64(s, [&] { return standard_gamma(g, sh); });
}
inline NDArray arr_gamma(BitGen& g, double sh, double sc, const Shape& s) {
  return detail::fill_f64(s, [&] { return gamma(g, sh, sc); });
}
inline NDArray arr_beta(BitGen& g, double a, double b, const Shape& s) {
  return detail::fill_f64(s, [&] { return beta(g, a, b); });
}
inline NDArray arr_chisquare(BitGen& g, double df, const Shape& s) {
  return detail::fill_f64(s, [&] { return chisquare(g, df); });
}
inline NDArray arr_f(BitGen& g, double dfn, double dfd, const Shape& s) {
  return detail::fill_f64(s, [&] { return f(g, dfn, dfd); });
}
inline NDArray arr_standard_cauchy(BitGen& g, const Shape& s) {
  return detail::fill_f64(s, [&] { return standard_cauchy(g); });
}
inline NDArray arr_pareto(BitGen& g, double a, const Shape& s) {
  return detail::fill_f64(s, [&] { return pareto(g, a); });
}
inline NDArray arr_weibull(BitGen& g, double a, const Shape& s) {
  return detail::fill_f64(s, [&] { return weibull(g, a); });
}
inline NDArray arr_power(BitGen& g, double a, const Shape& s) {
  return detail::fill_f64(s, [&] { return power(g, a); });
}
inline NDArray arr_laplace(BitGen& g, double loc, double sc, const Shape& s) {
  return detail::fill_f64(s, [&] { return laplace(g, loc, sc); });
}
inline NDArray arr_gumbel(BitGen& g, double loc, double sc, const Shape& s) {
  return detail::fill_f64(s, [&] { return gumbel(g, loc, sc); });
}
inline NDArray arr_logistic(BitGen& g, double loc, double sc, const Shape& s) {
  return detail::fill_f64(s, [&] { return logistic(g, loc, sc); });
}
inline NDArray arr_lognormal(BitGen& g, double mean, double sigma, const Shape& s) {
  return detail::fill_f64(s, [&] { return lognormal(g, mean, sigma); });
}
inline NDArray arr_rayleigh(BitGen& g, double mode, const Shape& s) {
  return detail::fill_f64(s, [&] { return rayleigh(g, mode); });
}
inline NDArray arr_standard_t(BitGen& g, double df, const Shape& s) {
  return detail::fill_f64(s, [&] { return standard_t(g, df); });
}
inline NDArray arr_noncentral_chisquare(BitGen& g, double df, double nonc, const Shape& s) {
  return detail::fill_f64(s, [&] { return noncentral_chisquare(g, df, nonc); });
}
inline NDArray arr_noncentral_f(BitGen& g, double dfn, double dfd, double nonc, const Shape& s) {
  return detail::fill_f64(s, [&] { return noncentral_f(g, dfn, dfd, nonc); });
}
inline NDArray arr_wald(BitGen& g, double mean, double sc, const Shape& s) {
  return detail::fill_f64(s, [&] { return wald(g, mean, sc); });
}
inline NDArray arr_vonmises(BitGen& g, double mu, double kappa, const Shape& s) {
  return detail::fill_f64(s, [&] { return vonmises(g, mu, kappa); });
}
inline NDArray arr_triangular(BitGen& g, double l, double mode, double r, const Shape& s) {
  return detail::fill_f64(s, [&] { return triangular(g, l, mode, r); });
}
inline NDArray arr_exponential(BitGen& g, double sc, const Shape& s) {
  return detail::fill_f64(s, [&] { return exponential(g, sc); });
}
inline NDArray arr_poisson(BitGen& g, double lam, const Shape& s) {
  return detail::fill_i64(s, [&] { return poisson(g, lam); });
}
inline NDArray arr_negative_binomial(BitGen& g, double n, double p, const Shape& s) {
  return detail::fill_i64(s, [&] { return negative_binomial(g, n, p); });
}
inline NDArray arr_binomial(BitGen& g, double p, std::int64_t n, Binomial& b, const Shape& s) {
  return detail::fill_i64(s, [&] { return binomial(g, p, n, b); });
}
inline NDArray arr_logseries(BitGen& g, double p, const Shape& s) {
  return detail::fill_i64(s, [&] { return logseries(g, p); });
}
inline NDArray arr_geometric(BitGen& g, double p, const Shape& s) {
  return detail::fill_i64(s, [&] { return geometric(g, p); });
}
inline NDArray arr_zipf(BitGen& g, double a, const Shape& s) {
  return detail::fill_i64(s, [&] { return zipf(g, a); });
}
inline NDArray arr_hypergeometric(BitGen& g, std::int64_t good, std::int64_t bad,
                                  std::int64_t sample, const Shape& s) {
  return detail::fill_i64(s, [&] { return hypergeometric(g, good, bad, sample); });
}

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
