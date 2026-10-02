#include "p13_distributions.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <vector>

#include "p13_ziggurat_exp.hpp"

namespace nativpy::random::p13 {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr std::int64_t kIntMax = std::numeric_limits<std::int64_t>::max();

double D(std::int64_t v) { return static_cast<double>(v); }

double next_double(BitGen& g) { return g.next_double(); }

}  // namespace

std::int64_t to_i64(double x) noexcept {
  if (std::isnan(x)) return 0;
  if (x >= 9223372036854775807.0) return kIntMax;
  if (x <= -9223372036854775808.0) return std::numeric_limits<std::int64_t>::min();
  return static_cast<std::int64_t>(x);
}

double loggam(double x) {
  static constexpr double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                                   7.936507936507937e-04, -5.952380952380952e-04,
                                   8.417508417508418e-04, -1.917526917526918e-03,
                                   6.410256410256410e-03, -2.955065359477124e-02,
                                   1.796443723688307e-01, -1.39243221690590e+00};
  if (x == 1.0 || x == 2.0) return 0.0;
  std::int64_t n = 0;
  if (x < 7.0) n = to_i64(7 - x);
  double x0 = x + D(n);
  const double x2 = (1.0 / x0) * (1.0 / x0);
  const double lg2pi = 1.8378770664093453e+00;
  double gl0 = a[9];
  for (int k = 8; k >= 0; --k) {
    gl0 *= x2;
    gl0 += a[k];
  }
  double gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * std::log(x0) - x0;
  if (x < 7.0) {
    for (std::int64_t k = 1; k <= n; ++k) {
      gl -= std::log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

double logfactorial(std::int64_t k) {
  static constexpr double logfact[] = {
      0,                  0,                  0.69314718055994529, 1.791759469228055,
      3.1780538303479458, 4.7874917427820458, 6.5792512120101012,  8.5251613610654147,
      10.604602902745251, 12.801827480081469, 15.104412573075516,  17.502307845873887,
      19.987214495661885, 22.552163853123425, 25.19122118273868,   27.89927138384089,
      30.671860106080672, 33.505073450136891, 36.395445208033053,  39.339884187199495,
      42.335616460753485, 45.380138898476908, 48.471181351835227,  51.606675567764377,
      54.784729398112319, 58.003605222980518, 61.261701761002001,  64.557538627006338,
      67.88974313718154,  71.257038967168015, 74.658236348830158,  78.092223553315307,
      81.557959456115043, 85.054467017581516, 88.580827542197682,  92.136175603687093,
      95.719694542143202, 99.330612454787428, 102.96819861451381,  106.63176026064346,
      110.32063971475739, 114.03421178146171, 117.77188139974507,  121.53308151543864,
      125.3172711493569,  129.12393363912722, 132.95257503561632,  136.80272263732635,
      140.67392364823425, 144.5657439463449,  148.47776695177302,  152.40959258449735,
      156.3608363030788,  160.3311282166309,  164.32011226319517,  168.32744544842765,
      172.35279713916279, 176.39584840699735, 180.45629141754378,  184.53382886144948,
      188.6281734236716,  192.7390472878449,  196.86618167289001,  201.00931639928152,
      205.1681994826412,  209.34258675253685, 213.53224149456327,  217.73693411395422,
      221.95644181913033, 226.1905483237276,  230.43904356577696,  234.70172344281826,
      238.97838956183432, 243.26884900298271, 247.57291409618688,  251.89040220972319,
      256.22113555000954, 260.56494097186322, 264.92164979855278,  269.29109765101981,
      273.67312428569369, 278.06757344036612, 282.4742926876304,   286.89313329542699,
      291.32395009427029, 295.76660135076065, 300.22094864701415,  304.68685676566872,
      309.1641935801469,  313.65282994987905, 318.1526396202093,   322.66349912672615,
      327.1852877037752,  331.71788719692847, 336.26118197919845,  340.81505887079902,
      345.37940706226686, 349.95411804077025, 354.53908551944079,  359.1342053695754,
      363.73937555556347, 368.35449607240474, 372.97946888568902,  377.61419787391867,
      382.25858877306001, 386.91254912321756, 391.57598821732961,  396.24881705179155,
      400.93094827891576, 405.6222961611449,  410.32277652693733,  415.03230672824964,
      419.75080559954472, 424.47819341825709, 429.21439186665157,  433.95932399501481,
      438.71291418612117, 443.47508812091894, 448.24577274538461,  453.02489623849613,
      457.81238798127816, 462.60817852687489, 467.4121995716082,   472.22438392698058,
      477.04466549258564, 481.87297922988796};
  constexpr auto n = static_cast<std::int64_t>(sizeof(logfact) / sizeof(logfact[0]));
  const double halfln2pi = 0.9189385332046728;
  if (k < n) return logfact[k];
  const double dk = D(k);
  return (dk + 0.5) * std::log(dk) - dk + (halfln2pi + (1.0 / dk) * (1 / 12.0 - 1 / (360.0 * dk * dk)));
}

// ---------------------------------------------------------------- Generator

double standard_exponential(BitGen& g) {
  using namespace zig;
  for (;;) {
    std::uint64_t ri = g.next_uint64();
    ri >>= 3;
    const auto idx = static_cast<std::size_t>(ri & 0xFFU);
    ri >>= 8;
    const double x = static_cast<double>(ri) * we_double[idx];
    if (ri < ke_double[idx]) return x;
    if (idx == 0) return ziggurat_exp_r - std::log1p(-g.next_double());
    if ((fe_double[idx - 1] - fe_double[idx]) * g.next_double() + fe_double[idx] < std::exp(-x)) {
      return x;
    }
  }
}

float standard_exponential_f(BitGen& g) {
  using namespace zig;
  for (;;) {
    std::uint32_t ri = g.next_uint32();
    ri >>= 1;
    const auto idx = static_cast<std::size_t>(ri & 0xFFU);
    ri >>= 8;
    const float x = static_cast<float>(ri) * we_float[idx];
    if (ri < ke_float[idx]) return x;
    if (idx == 0) return ziggurat_exp_r_f - std::log1p(-next_float(g));
    if ((fe_float[idx - 1] - fe_float[idx]) * next_float(g) + fe_float[idx] < std::exp(-x)) {
      return x;
    }
  }
}

double standard_exponential_inv(BitGen& g) { return -std::log1p(-g.next_double()); }

float standard_exponential_inv_f(BitGen& g) {
  // NumPy calls the double npy_log1p here and narrows the result.
  return static_cast<float>(-std::log1p(-static_cast<double>(next_float(g))));
}

double standard_gamma(BitGen& g, double shape) {
  if (shape == 1.0) return standard_exponential(g);
  if (shape == 0.0) return 0.0;
  if (shape < 1.0) {
    for (;;) {
      const double U = g.next_double();
      const double V = standard_exponential(g);
      if (U <= 1.0 - shape) {
        const double X = std::pow(U, 1. / shape);
        if (X <= V) return X;
      } else {
        const double Y = -std::log((1 - U) / shape);
        const double X = std::pow(1.0 - shape + shape * Y, 1. / shape);
        if (X <= (V + Y)) return X;
      }
    }
  }
  const double b = shape - 1. / 3.;
  const double c = 1. / std::sqrt(9 * b);
  for (;;) {
    double X = 0.0;
    double V = 0.0;
    do {
      X = standard_normal(g);
      V = 1.0 + c * X;
    } while (V <= 0.0);
    V = V * V * V;
    const double U = g.next_double();
    if (U < 1.0 - 0.0331 * (X * X) * (X * X)) return b * V;
    if (std::log(U) < 0.5 * X * X + b * (1. - V + std::log(V))) return b * V;
  }
}

float standard_gamma_f(BitGen& g, float shape) {
  if (shape == 1.0F) return standard_exponential_f(g);
  if (shape == 0.0F) return 0.0F;
  if (shape < 1.0F) {
    for (;;) {
      const float U = next_float(g);
      const float V = standard_exponential_f(g);
      if (U <= 1.0F - shape) {
        const float X = std::pow(U, 1.0F / shape);
        if (X <= V) return X;
      } else {
        const float Y = -std::log((1.0F - U) / shape);
        const float X = std::pow(1.0F - shape + shape * Y, 1.0F / shape);
        if (X <= (V + Y)) return X;
      }
    }
  }
  const float b = shape - 1.0F / 3.0F;
  const float c = 1.0F / std::sqrt(9.0F * b);
  for (;;) {
    float X = 0.0F;
    float V = 0.0F;
    do {
      X = standard_normal_f(g);
      V = 1.0F + c * X;
    } while (V <= 0.0F);
    V = V * V * V;
    const float U = next_float(g);
    if (U < 1.0F - 0.0331F * (X * X) * (X * X)) return b * V;
    if (std::log(U) < 0.5F * X * X + b * (1.0F - V + std::log(V))) return b * V;
  }
}

double gamma(BitGen& g, double shape, double scale) { return scale * standard_gamma(g, shape); }

double beta(BitGen& g, double a, double b) {
  if (a <= 1.0 && b <= 1.0) {
    if (a < 3e-103 && b < 3e-103) {
      const double U = g.next_double();
      return ((a + b) * U < a) ? 1.0 : 0.0;
    }
    for (;;) {
      const double U = g.next_double();
      const double V = g.next_double();
      const double X = std::pow(U, 1.0 / a);
      const double Y = std::pow(V, 1.0 / b);
      const double XpY = X + Y;
      if (XpY <= 1.0 && U + V > 0.0) {
        if (X > 0 && Y > 0) return X / XpY;
        const double logX = std::log(U) / a;
        const double logY = std::log(V) / b;
        const double delta = logX - logY;
        if (delta > 0) return std::exp(-std::log1p(std::exp(-delta)));
        return std::exp(delta - std::log1p(std::exp(delta)));
      }
    }
  }
  const double Ga = standard_gamma(g, a);
  const double Gb = standard_gamma(g, b);
  return Ga / (Ga + Gb);
}

double chisquare(BitGen& g, double df) { return 2.0 * standard_gamma(g, df / 2.0); }

double f(BitGen& g, double dfnum, double dfden) {
  const double s1 = chisquare(g, dfnum) * dfden;
  const double s2 = chisquare(g, dfden) * dfnum;
  return s1 / s2;
}

double standard_cauchy(BitGen& g) {
  const double s1 = standard_normal(g);
  const double s2 = standard_normal(g);
  return s1 / s2;
}

double pareto(BitGen& g, double a) { return std::expm1(standard_exponential(g) / a); }

double weibull(BitGen& g, double a) {
  if (a == 0.0) return 0.0;
  return std::pow(standard_exponential(g), 1. / a);
}

double power(BitGen& g, double a) {
  return std::pow(-std::expm1(-standard_exponential(g)), 1. / a);
}

double laplace(BitGen& g, double loc, double scale) {
  for (;;) {
    const double U = g.next_double();
    if (U >= 0.5) return loc - scale * std::log(2.0 - U - U);
    if (U > 0.0) return loc + scale * std::log(U + U);
  }
}

double gumbel(BitGen& g, double loc, double scale) {
  for (;;) {
    const double U = 1.0 - g.next_double();
    if (U < 1.0) return loc - scale * std::log(-std::log(U));
  }
}

double logistic(BitGen& g, double loc, double scale) {
  for (;;) {
    const double U = g.next_double();
    if (U > 0.0) return loc + scale * std::log(U / (1.0 - U));
  }
}

double lognormal(BitGen& g, double mean, double sigma) {
  return std::exp(mean + sigma * standard_normal(g));
}

double rayleigh(BitGen& g, double mode) { return mode * std::sqrt(2.0 * standard_exponential(g)); }

double standard_t(BitGen& g, double df) {
  const double num = standard_normal(g);
  const double denom = standard_gamma(g, df / 2);
  return std::sqrt(df / 2) * num / std::sqrt(denom);
}

double exponential(BitGen& g, double scale) { return scale * standard_exponential(g); }

namespace {

std::int64_t poisson_mult(BitGen& g, double lam) {
  const double enlam = std::exp(-lam);
  std::int64_t X = 0;
  double prod = 1.0;
  for (;;) {
    prod *= g.next_double();
    if (prod > enlam) {
      X += 1;
    } else {
      return X;
    }
  }
}

std::int64_t poisson_ptrs(BitGen& g, double lam) {
  const double slam = std::sqrt(lam);
  const double loglam = std::log(lam);
  const double b = 0.931 + 2.53 * slam;
  const double a = -0.059 + 0.02483 * b;
  const double invalpha = 1.1239 + 1.1328 / (b - 3.4);
  const double vr = 0.9277 - 3.6224 / (b - 2);
  for (;;) {
    const double U = g.next_double() - 0.5;
    const double V = g.next_double();
    const double us = 0.5 - std::fabs(U);
    const std::int64_t k = to_i64(std::floor((2 * a / us + b) * U + lam + 0.43));
    if (us >= 0.07 && V <= vr) return k;
    if (k < 0 || (us < 0.013 && V > us)) continue;
    if ((std::log(V) + std::log(invalpha) - std::log(a / (us * us) + b)) <=
        (-lam + D(k) * loglam - loggam(D(k) + 1))) {
      return k;
    }
  }
}

}  // namespace

std::int64_t poisson(BitGen& g, double lam) {
  if (lam >= 10) return poisson_ptrs(g, lam);
  if (lam == 0) return 0;
  return poisson_mult(g, lam);
}

std::int64_t negative_binomial(BitGen& g, double n, double p) {
  const double Y = gamma(g, n, (1 - p) / p);
  return poisson(g, Y);
}

namespace {

// Shared BTPE body; `legacy` selects legacy-distributions.c's variant of the
// Step 52 Stirling correction (13680 and all-plus signs).
std::int64_t binomial_btpe(BitGen& g, std::int64_t n, double p, Binomial& bn, bool legacy) {
  double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
  std::int64_t m;
  if (!bn.has_binomial || bn.nsave != n || bn.psave != p) {
    bn.nsave = n;
    bn.psave = p;
    bn.has_binomial = true;
    bn.r = r = std::min(p, 1.0 - p);
    bn.q = q = 1.0 - r;
    bn.fm = fm = D(n) * r + r;
    bn.m = m = to_i64(std::floor(bn.fm));
    bn.p1 = p1 = std::floor(2.195 * std::sqrt(D(n) * r * q) - 4.6 * q) + 0.5;
    bn.xm = xm = D(m) + 0.5;
    bn.xl = xl = xm - p1;
    bn.xr = xr = xm + p1;
    bn.c = c = 0.134 + 20.5 / (15.3 + D(m));
    double a = (fm - xl) / (fm - xl * r);
    bn.laml = laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    bn.lamr = lamr = a * (1.0 + a / 2.0);
    bn.p2 = p2 = p1 * (1.0 + 2.0 * c);
    bn.p3 = p3 = p2 + c / laml;
    bn.p4 = p4 = p3 + c / lamr;
  } else {
    r = bn.r;
    q = bn.q;
    fm = bn.fm;
    m = bn.m;
    p1 = bn.p1;
    xm = bn.xm;
    xl = bn.xl;
    xr = bn.xr;
    c = bn.c;
    laml = bn.laml;
    lamr = bn.lamr;
    p2 = bn.p2;
    p3 = bn.p3;
    p4 = bn.p4;
  }
  const double dn = D(n);
  std::int64_t y = 0;
  for (;;) {  // Step10
    const double nrq = dn * r * q;
    const double u = g.next_double() * p4;
    double v = g.next_double();
    if (u <= p1) {
      y = to_i64(std::floor(xm - p1 * v + u));
      break;  // Step60
    }
    if (u <= p2) {  // Step20
      const double x = xl + (u - p1) / c;
      v = v * c + 1.0 - std::fabs(D(m) - x + 0.5) / p1;
      if (v > 1.0) continue;
      y = to_i64(std::floor(x));
    } else if (u <= p3) {  // Step30
      y = to_i64(std::floor(xl + std::log(v) / laml));
      if (y < 0 || v == 0.0) continue;
      v = v * (u - p2) * laml;
    } else {  // Step40
      y = to_i64(std::floor(xr - std::log(v) / lamr));
      if (y > n || v == 0.0) continue;
      v = v * (u - p3) * lamr;
    }
    // Step50
    const std::int64_t k = std::llabs(y - m);
    if (!(k > 20 && D(k) < (nrq / 2.0 - 1))) {
      const double s = r / q;
      const double a = s * (dn + 1);
      double F = 1.0;
      if (m < y) {
        for (std::int64_t i = m + 1; i <= y; ++i) F *= (a / D(i) - s);
      } else if (m > y) {
        for (std::int64_t i = y + 1; i <= m; ++i) F /= (a / D(i) - s);
      }
      if (v > F) continue;
      break;
    }
    // Step52
    const double dk = D(k);
    const double rho = (dk / nrq) * ((dk * (dk / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
    const double t = D(-k * k) / (2 * nrq);
    const double A = std::log(v);
    if (A < (t - rho)) break;
    if (A > (t + rho)) continue;
    const double x1 = D(y) + 1;
    const double f1 = D(m) + 1;
    const double z = dn + 1 - D(m);
    const double w = dn - D(y) + 1;
    const double x2 = x1 * x1;
    const double f2 = f1 * f1;
    const double z2 = z * z;
    const double w2 = w * w;
    const double base = xm * std::log(f1 / x1) + (D(n - m) + 0.5) * std::log(z / w) +
                        D(y - m) * std::log(w * r / (x1 * q));
    double bound = 0.0;
    if (legacy) {
      bound = base +
              (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 / 166320. +
              (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z / 166320. +
              (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 / 166320. +
              (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w / 166320.;
    } else {
      bound = base +
              (13860. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 / 166320. +
              (13860. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z / 166320. -
              (13860. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 / 166320. -
              (13860. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w / 166320.;
    }
    if (A > bound) continue;
    break;
  }
  // Step60
  if (p > 0.5) y = n - y;
  return y;
}

std::int64_t binomial_inversion(BitGen& g, std::int64_t n, double p, Binomial& bn, bool legacy) {
  double q, qn, np;
  std::int64_t bound;
  if (!bn.has_binomial || bn.nsave != n || bn.psave != p) {
    bn.nsave = n;
    bn.psave = p;
    bn.has_binomial = true;
    bn.q = q = 1.0 - p;
    bn.r = qn = legacy ? std::exp(D(n) * std::log(q)) : std::exp(D(n) * std::log1p(-p));
    bn.c = np = D(n) * p;
    const double lim = np + 10.0 * std::sqrt(np * q + 1);
    bn.m = bound = to_i64(D(n) < lim ? D(n) : lim);
  } else {
    q = bn.q;
    qn = bn.r;
    np = bn.c;
    bound = bn.m;
  }
  (void)np;
  std::int64_t X = 0;
  double px = qn;
  double U = g.next_double();
  while (U > px) {
    X++;
    if (X > bound) {
      X = 0;
      px = qn;
      U = g.next_double();
    } else {
      U -= px;
      px = (D(n - X + 1) * p * px) / (D(X) * q);
    }
  }
  return X;
}

std::int64_t binomial_impl(BitGen& g, double p, std::int64_t n, Binomial& b, bool legacy) {
  if (p <= 0.5) {
    if (p * D(n) <= 30.0) return binomial_inversion(g, n, p, b, legacy);
    return binomial_btpe(g, n, p, b, legacy);
  }
  const double q = 1.0 - p;
  if (q * D(n) <= 30.0) return n - binomial_inversion(g, n, q, b, legacy);
  return n - binomial_btpe(g, n, q, b, legacy);
}

}  // namespace

std::int64_t binomial(BitGen& g, double p, std::int64_t n, Binomial& b) {
  if (n == 0 || p == 0.0) return 0;
  return binomial_impl(g, p, n, b, false);
}

double noncentral_chisquare(BitGen& g, double df, double nonc) {
  if (std::isnan(nonc)) return std::numeric_limits<double>::quiet_NaN();
  if (nonc == 0) return chisquare(g, df);
  if (1 < df) {
    const double Chi2 = chisquare(g, df - 1);
    const double n = standard_normal(g) + std::sqrt(nonc);
    return Chi2 + n * n;
  }
  const std::int64_t i = poisson(g, nonc / 2.0);
  return chisquare(g, df + 2 * D(i));
}

double noncentral_f(BitGen& g, double dfnum, double dfden, double nonc) {
  const double t = noncentral_chisquare(g, dfnum, nonc) * dfden;
  return t / (chisquare(g, dfden) * dfnum);
}

double wald(BitGen& g, double mean, double scale) {
  double Y = standard_normal(g);
  Y = mean * Y * Y;
  const double d = 1 + std::sqrt(1 + 4 * scale / Y);
  const double X = mean * (1 - 2 / d);
  const double U = g.next_double();
  if (U <= mean / (mean + X)) return X;
  return mean * mean / X;
}

namespace {

double vonmises_impl(BitGen& g, double mu, double kappa, bool legacy) {
  if (std::isnan(kappa)) return std::numeric_limits<double>::quiet_NaN();
  if (kappa < 1e-8) return kPi * (2 * g.next_double() - 1);
  double s = 0.0;
  if (kappa < 1e-5) {
    s = (1. / kappa + kappa);
  } else if (legacy || kappa <= 1e6) {
    const double r = 1 + std::sqrt(1 + 4 * kappa * kappa);
    const double rho = (r - std::sqrt(2 * r)) / (2 * kappa);
    s = (1 + rho * rho) / (2 * rho);
  } else {
    double result = mu + std::sqrt(1. / kappa) * standard_normal(g);
    if (result < -kPi) result += 2 * kPi;
    if (result > kPi) result -= 2 * kPi;
    return result;
  }
  double W = 0.0;
  for (;;) {
    const double U = g.next_double();
    const double Z = std::cos(kPi * U);
    W = (1 + s * Z) / (s + Z);
    const double Y = kappa * (s - W);
    const double V = g.next_double();
    if ((Y * (2 - Y) - V >= 0) || (std::log(Y / V) + 1 - Y >= 0)) break;
  }
  const double U = g.next_double();
  double result = std::acos(W);
  if (U < 0.5) result = -result;
  result += mu;
  const bool neg = result < 0;
  double mod = std::fabs(result);
  mod = (std::fmod(mod + kPi, 2 * kPi) - kPi);
  if (neg) mod *= -1;
  return mod;
}

}  // namespace

double vonmises(BitGen& g, double mu, double kappa) { return vonmises_impl(g, mu, kappa, false); }

std::int64_t logseries(BitGen& g, double p) {
  const double r = std::log1p(-p);
  for (;;) {
    const double V = g.next_double();
    if (V >= p) return 1;
    const double U = g.next_double();
    const double q = -std::expm1(r * U);
    if (V <= q * q) {
      const std::int64_t result = to_i64(std::floor(1 + std::log(V) / std::log(q)));
      if (result < 1 || V == 0.0) continue;
      return result;
    }
    if (V >= q) return 1;
    return 2;
  }
}

namespace {

std::int64_t geometric_search(BitGen& g, double p) {
  std::int64_t X = 1;
  double sum = p;
  double prod = p;
  const double q = 1.0 - p;
  const double U = g.next_double();
  while (U > sum) {
    prod *= q;
    sum += prod;
    X++;
  }
  return X;
}

}  // namespace

std::int64_t geometric(BitGen& g, double p) {
  if (p >= 0.333333333333333333333333) return geometric_search(g, p);
  const double z = std::ceil(-standard_exponential(g) / std::log1p(-p));
  if (z >= 9.223372036854776e+18) return kIntMax;
  return to_i64(z);
}

std::int64_t zipf(BitGen& g, double a) {
  if (a >= 1025) return 1;
  const double am1 = a - 1.0;
  const double b = std::pow(2.0, am1);
  const double Umin = std::pow(static_cast<double>(kIntMax), -am1);
  for (;;) {
    const double U01 = g.next_double();
    const double U = U01 * Umin + (1 - U01);
    const double V = g.next_double();
    const double X = std::floor(std::pow(U, -1.0 / am1));
    if (X > static_cast<double>(kIntMax) || X < 1.0) continue;
    const double T = std::pow(1.0 + 1.0 / X, am1);
    if (V * X * (T - 1.0) / (b - 1.0) <= T / b) return to_i64(X);
  }
}

double triangular(BitGen& g, double left, double mode, double right) {
  const double base = right - left;
  const double leftbase = mode - left;
  const double ratio = leftbase / base;
  const double leftprod = leftbase * base;
  const double rightprod = (right - mode) * base;
  const double U = g.next_double();
  if (U <= ratio) return left + std::sqrt(U * leftprod);
  return right - std::sqrt((1.0 - U) * rightprod);
}

namespace {

std::int64_t hypergeometric_sample(BitGen& g, std::int64_t good, std::int64_t bad,
                                   std::int64_t sample) {
  const std::int64_t total = good + bad;
  std::int64_t computed_sample = sample > total / 2 ? total - sample : sample;
  std::int64_t remaining_total = total;
  std::int64_t remaining_good = good;
  while (computed_sample > 0 && remaining_good > 0 && remaining_total > remaining_good) {
    --remaining_total;
    if (static_cast<std::int64_t>(random_interval(g, static_cast<std::uint64_t>(remaining_total))) <
        remaining_good) {
      --remaining_good;
    }
    --computed_sample;
  }
  if (remaining_total == remaining_good) remaining_good -= computed_sample;
  return sample > total / 2 ? remaining_good : good - remaining_good;
}

constexpr double kD1 = 1.7155277699214135;
constexpr double kD2 = 0.8989161620588988;

std::int64_t hypergeometric_hrua(BitGen& g, std::int64_t good, std::int64_t bad,
                                 std::int64_t sample) {
  const std::int64_t popsize = good + bad;
  const std::int64_t computed_sample = std::min(sample, popsize - sample);
  const std::int64_t mingoodbad = std::min(good, bad);
  const std::int64_t maxgoodbad = std::max(good, bad);
  const double p = D(mingoodbad) / D(popsize);
  const double q = D(maxgoodbad) / D(popsize);
  const double mu = D(computed_sample) * p;
  const double a = mu + 0.5;
  const double var = (D(popsize - computed_sample) * D(computed_sample) * p * q / D(popsize - 1));
  const double c = std::sqrt(var + 0.5);
  const double h = kD1 * c + kD2;
  const std::int64_t m =
      to_i64(std::floor(D(computed_sample + 1) * D(mingoodbad + 1) / D(popsize + 2)));
  const double gg = logfactorial(m) + logfactorial(mingoodbad - m) +
                    logfactorial(computed_sample - m) +
                    logfactorial(maxgoodbad - computed_sample + m);
  const double b = std::min(D(std::min(computed_sample, mingoodbad) + 1), std::floor(a + 16 * c));
  std::int64_t K = 0;
  for (;;) {
    const double U = g.next_double();
    const double V = g.next_double();
    const double X = a + h * (V - 0.5) / U;
    if (X < 0.0 || X >= b) continue;
    K = to_i64(std::floor(X));
    const double gp = logfactorial(K) + logfactorial(mingoodbad - K) +
                      logfactorial(computed_sample - K) +
                      logfactorial(maxgoodbad - computed_sample + K);
    const double T = gg - gp;
    if ((U * (4.0 - U) - 3.0) <= T) break;
    if (U * (U - T) >= 1) continue;
    if (2.0 * std::log(U) <= T) break;
  }
  if (good > bad) K = computed_sample - K;
  if (computed_sample < sample) K = good - K;
  return K;
}

}  // namespace

std::int64_t hypergeometric(BitGen& g, std::int64_t good, std::int64_t bad, std::int64_t sample) {
  if (sample >= 10 && sample <= good + bad - 10) return hypergeometric_hrua(g, good, bad, sample);
  return hypergeometric_sample(g, good, bad, sample);
}

void multinomial(BitGen& g, std::int64_t n, std::int64_t* mnix, const double* pix, std::int64_t d,
                 Binomial& b) {
  double remaining_p = 1.0;
  std::int64_t dn = n;
  for (std::int64_t j = 0; j < d - 1; ++j) {
    mnix[j] = binomial(g, pix[j] / remaining_p, dn, b);
    dn = dn - mnix[j];
    if (dn <= 0) break;
    remaining_p -= pix[j];
  }
  if (dn > 0) mnix[d - 1] = dn;
}

void mvhg_count(BitGen& g, std::int64_t total, std::size_t num_colors, const std::int64_t* colors,
                std::int64_t nsample, std::size_t num_variates, std::int64_t* variates) {
  if (total == 0 || nsample == 0 || num_variates == 0) return;
  std::vector<std::size_t> choices(static_cast<std::size_t>(total));
  for (std::size_t i = 0, k = 0; i < num_colors; ++i) {
    for (std::int64_t j = 0; j < colors[i]; ++j) choices[k++] = i;
  }
  const bool more_than_half = nsample > total / 2;
  if (more_than_half) nsample = total - nsample;
  const auto ns = static_cast<std::size_t>(nsample);
  const auto tot = static_cast<std::size_t>(total);
  for (std::size_t i = 0; i < num_variates * num_colors; i += num_colors) {
    for (std::size_t j = 0; j < ns; ++j) {
      const std::size_t k = j + static_cast<std::size_t>(random_interval(g, tot - j - 1));
      std::swap(choices[k], choices[j]);
    }
    for (std::size_t j = 0; j < ns; ++j) variates[i + choices[j]] += 1;
    if (more_than_half) {
      for (std::size_t k = 0; k < num_colors; ++k) variates[i + k] = colors[k] - variates[i + k];
    }
  }
}

void mvhg_marginals(BitGen& g, std::int64_t total, std::size_t num_colors,
                    const std::int64_t* colors, std::int64_t nsample, std::size_t num_variates,
                    std::int64_t* variates) {
  if (total == 0 || nsample == 0 || num_variates == 0) return;
  const bool more_than_half = nsample > total / 2;
  if (more_than_half) nsample = total - nsample;
  for (std::size_t i = 0; i < num_variates * num_colors; i += num_colors) {
    std::int64_t num_to_sample = nsample;
    std::int64_t remaining = total;
    for (std::size_t j = 0; num_to_sample > 0 && j + 1 < num_colors; ++j) {
      remaining -= colors[j];
      const std::int64_t r = hypergeometric(g, colors[j], remaining, num_to_sample);
      variates[i + j] = r;
      num_to_sample -= r;
    }
    if (num_to_sample > 0) variates[i + num_colors - 1] = num_to_sample;
    if (more_than_half) {
      for (std::size_t k = 0; k < num_colors; ++k) variates[i + k] = colors[k] - variates[i + k];
    }
  }
}

// ---------------------------------------------------------------- RandomState

namespace legacy {

double standard_exponential(BitGen& g) { return -std::log(1.0 - next_double(g)); }

double exponential(BitGen& g, double scale) { return scale * standard_exponential(g); }

double standard_gamma(BitGen& g, LegacyGauss& gs, double shape) {
  if (shape == 1.0) return standard_exponential(g);
  if (shape == 0.0) return 0.0;
  if (shape < 1.0) {
    for (;;) {
      const double U = next_double(g);
      const double V = standard_exponential(g);
      if (U <= 1.0 - shape) {
        const double X = std::pow(U, 1. / shape);
        if (X <= V) return X;
      } else {
        const double Y = -std::log((1 - U) / shape);
        const double X = std::pow(1.0 - shape + shape * Y, 1. / shape);
        if (X <= (V + Y)) return X;
      }
    }
  }
  const double b = shape - 1. / 3.;
  const double c = 1. / std::sqrt(9 * b);
  for (;;) {
    double X = 0.0;
    double V = 0.0;
    do {
      X = gs.next(g);
      V = 1.0 + c * X;
    } while (V <= 0.0);
    V = V * V * V;
    const double U = next_double(g);
    if (U < 1.0 - 0.0331 * (X * X) * (X * X)) return b * V;
    if (std::log(U) < 0.5 * X * X + b * (1. - V + std::log(V))) return b * V;
  }
}

double gamma(BitGen& g, LegacyGauss& gs, double shape, double scale) {
  return scale * standard_gamma(g, gs, shape);
}

double pareto(BitGen& g, double a) { return std::exp(standard_exponential(g) / a) - 1; }

double weibull(BitGen& g, double a) {
  if (a == 0.0) return 0.0;
  return std::pow(standard_exponential(g), 1. / a);
}

double power(BitGen& g, double a) {
  return std::pow(1 - std::exp(-standard_exponential(g)), 1. / a);
}

double chisquare(BitGen& g, LegacyGauss& gs, double df) {
  return 2.0 * standard_gamma(g, gs, df / 2.0);
}

double rayleigh(BitGen& g, double mode) {
  return mode * std::sqrt(-2.0 * std::log1p(-next_double(g)));
}

double noncentral_chisquare(BitGen& g, LegacyGauss& gs, double df, double nonc) {
  if (nonc == 0) return chisquare(g, gs, df);
  if (1 < df) {
    const double Chi2 = chisquare(g, gs, df - 1);
    const double n = gs.next(g) + std::sqrt(nonc);
    return Chi2 + n * n;
  }
  const std::int64_t i = p13::poisson(g, nonc / 2.0);
  const double out = chisquare(g, gs, df + 2 * D(i));
  if (std::isnan(nonc)) return std::numeric_limits<double>::quiet_NaN();
  return out;
}

double noncentral_f(BitGen& g, LegacyGauss& gs, double dfnum, double dfden, double nonc) {
  const double t = noncentral_chisquare(g, gs, dfnum, nonc) * dfden;
  return t / (chisquare(g, gs, dfden) * dfnum);
}

double wald(BitGen& g, LegacyGauss& gs, double mean, double scale) {
  const double mu_2l = mean / (2 * scale);
  double Y = gs.next(g);
  Y = mean * Y * Y;
  const double X = mean + mu_2l * (Y - std::sqrt(4 * scale * Y + Y * Y));
  const double U = next_double(g);
  if (U <= mean / (mean + X)) return X;
  return mean * mean / X;
}

double lognormal(BitGen& g, LegacyGauss& gs, double mean, double sigma) {
  return std::exp(mean + sigma * gs.next(g));
}

double standard_t(BitGen& g, LegacyGauss& gs, double df) {
  const double num = gs.next(g);
  const double denom = standard_gamma(g, gs, df / 2);
  return std::sqrt(df / 2) * num / std::sqrt(denom);
}

std::int64_t negative_binomial(BitGen& g, LegacyGauss& gs, double n, double p) {
  const double Y = gamma(g, gs, n, (1 - p) / p);
  return p13::poisson(g, Y);
}

double standard_cauchy(BitGen& g, LegacyGauss& gs) {
  const double a = gs.next(g);
  const double b = gs.next(g);
  return a / b;
}

double beta(BitGen& g, LegacyGauss& gs, double a, double b) {
  if (a <= 1.0 && b <= 1.0) {
    for (;;) {
      const double U = next_double(g);
      const double V = next_double(g);
      const double X = std::pow(U, 1.0 / a);
      const double Y = std::pow(V, 1.0 / b);
      if ((X + Y) <= 1.0) {
        if (X + Y > 0) return X / (X + Y);
        double logX = std::log(U) / a;
        double logY = std::log(V) / b;
        const double logM = logX > logY ? logX : logY;
        logX -= logM;
        logY -= logM;
        return std::exp(logX - std::log(std::exp(logX) + std::exp(logY)));
      }
    }
  }
  const double Ga = standard_gamma(g, gs, a);
  const double Gb = standard_gamma(g, gs, b);
  return Ga / (Ga + Gb);
}

double f(BitGen& g, LegacyGauss& gs, double dfnum, double dfden) {
  const double s1 = chisquare(g, gs, dfnum) * dfden;
  const double s2 = chisquare(g, gs, dfden) * dfnum;
  return s1 / s2;
}

std::int64_t binomial(BitGen& g, double p, std::int64_t n, Binomial& b) {
  return binomial_impl(g, p, n, b, true);
}

namespace {

std::int64_t hyp(BitGen& g, std::int64_t good, std::int64_t bad, std::int64_t sample) {
  const std::int64_t d1 = bad + good - sample;
  const double d2 = D(std::min(bad, good));
  double y = d2;
  std::int64_t k = sample;
  while (y > 0.0) {
    const double u = next_double(g);
    y -= D(to_i64(std::floor(u + y / D(d1 + k))));
    k--;
    if (k == 0) break;
  }
  std::int64_t z = to_i64(d2 - y);
  if (good > bad) z = sample - z;
  return z;
}

std::int64_t hrua(BitGen& g, std::int64_t good, std::int64_t bad, std::int64_t sample) {
  const std::int64_t mingoodbad = std::min(good, bad);
  const std::int64_t popsize = good + bad;
  const std::int64_t maxgoodbad = std::max(good, bad);
  const std::int64_t m = std::min(sample, popsize - sample);
  const double d4 = D(mingoodbad) / D(popsize);
  const double d5 = 1.0 - d4;
  const double d6 = D(m) * d4 + 0.5;
  const double d7 = std::sqrt(D(popsize - m) * D(sample) * d4 * d5 / D(popsize - 1) + 0.5);
  const double d8 = kD1 * d7 + kD2;
  const std::int64_t d9 = to_i64(std::floor(D(m + 1) * D(mingoodbad + 1) / D(popsize + 2)));
  const double d10 = loggam(D(d9 + 1)) + loggam(D(mingoodbad - d9 + 1)) + loggam(D(m - d9 + 1)) +
                     loggam(D(maxgoodbad - m + d9 + 1));
  const double d11 = std::min(D(std::min(m, mingoodbad)) + 1.0, std::floor(d6 + 16 * d7));
  std::int64_t Z = 0;
  for (;;) {
    const double X = next_double(g);
    const double Y = next_double(g);
    const double W = d6 + d8 * (Y - 0.5) / X;
    if (W < 0.0 || W >= d11) continue;
    Z = to_i64(std::floor(W));
    const double T = d10 - (loggam(D(Z + 1)) + loggam(D(mingoodbad - Z + 1)) +
                            loggam(D(m - Z + 1)) + loggam(D(maxgoodbad - m + Z + 1)));
    if ((X * (4.0 - X) - 3.0) <= T) break;
    if (X * (X - T) >= 1) continue;
    if (2.0 * std::log(X) <= T) break;
  }
  if (good > bad) Z = m - Z;
  if (m < sample) Z = good - Z;
  return Z;
}

}  // namespace

std::int64_t hypergeometric(BitGen& g, std::int64_t good, std::int64_t bad, std::int64_t sample) {
  if (sample > 10) return hrua(g, good, bad, sample);
  if (sample > 0) return hyp(g, good, bad, sample);
  return 0;
}

std::int64_t zipf(BitGen& g, double a) {
  const double am1 = a - 1.0;
  const double b = std::pow(2.0, am1);
  for (;;) {
    const double U = 1.0 - next_double(g);
    const double V = next_double(g);
    const double X = std::floor(std::pow(U, -1.0 / am1));
    if (X > static_cast<double>(kIntMax) || X < 1.0) continue;
    const double T = std::pow(1.0 + 1.0 / X, am1);
    if (V * X * (T - 1.0) / (b - 1.0) <= T / b) return to_i64(X);
  }
}

std::int64_t geometric(BitGen& g, double p) {
  if (p >= 0.333333333333333333333333) return geometric_search(g, p);
  return to_i64(std::ceil(std::log1p(-next_double(g)) / std::log(1 - p)));
}

std::int64_t logseries(BitGen& g, double p) {
  const double r = std::log(1.0 - p);
  for (;;) {
    const double V = next_double(g);
    if (V >= p) return 1;
    const double U = next_double(g);
    const double q = 1.0 - std::exp(r * U);
    if (V <= q * q) {
      const std::int64_t result = to_i64(std::floor(1 + std::log(V) / std::log(q)));
      if (result < 1 || V == 0.0) continue;
      return result;
    }
    if (V >= q) return 1;
    return 2;
  }
}

double vonmises(BitGen& g, double mu, double kappa) { return vonmises_impl(g, mu, kappa, true); }

void multinomial(BitGen& g, std::int64_t n, std::int64_t* mnix, const double* pix, std::int64_t d,
                 Binomial& b) {
  double remaining_p = 1.0;
  std::int64_t dn = n;
  for (std::int64_t j = 0; j < d - 1; ++j) {
    mnix[j] = legacy::binomial(g, pix[j] / remaining_p, dn, b);
    dn = dn - mnix[j];
    if (dn <= 0) break;
    remaining_p -= pix[j];
  }
  if (dn > 0) mnix[d - 1] = dn;
}

}  // namespace legacy

}  // namespace nativpy::random::p13
