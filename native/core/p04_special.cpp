#include "p04_special.hpp"

#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <string>
#include <type_traits>
#include <vector>

#include "cast.hpp"
#include "error.hpp"
#include "fp_errors.hpp"
#include "shape.hpp"
#include "ufunc_loops.hpp"

namespace nativpy {

using ufunc_loops::compute_t;
using ufunc_loops::is_inexact;
using ufunc_loops::ld;
using ufunc_loops::st;

namespace {

// numpy.lib._function_base_impl._i0A / _i0B (Cephes i0.c).
constexpr double kI0A[] = {
    -4.41534164647933937950E-18, 3.33079451882223809783E-17, -2.43127984654795469359E-16,
    1.71539128555513303061E-15,  -1.16853328779934516808E-14, 7.67618549860493561688E-14,
    -4.85644678311192946090E-13, 2.95505266312963983461E-12, -1.72682629144155570723E-11,
    9.67580903537323691224E-11,  -5.18979560163526290666E-10, 2.65982372468238665035E-9,
    -1.30002500998624804212E-8,  6.04699502254191894932E-8,  -2.67079385394061173391E-7,
    1.11738753912010371815E-6,   -4.41673835845875056359E-6, 1.64484480707288970893E-5,
    -5.75419501008210370398E-5,  1.88502885095841655729E-4,  -5.76375574538582365885E-4,
    1.63947561694133579842E-3,   -4.32430999505057594430E-3, 1.05464603945949983183E-2,
    -2.37374148058994688156E-2,  4.93052842396707084878E-2,  -9.49010970480476444210E-2,
    1.71620901522208775349E-1,   -3.04682672343198398683E-1, 6.76795274409476084995E-1,
};
constexpr double kI0B[] = {
    -7.23318048787475395456E-18, -4.83050448594418207126E-18, 4.46562142029675999901E-17,
    3.46122286769746109310E-17,  -2.82762398051658348494E-16, -3.42548561967721913462E-16,
    1.77256013305652638360E-15,  3.81168066935262242075E-15,  -9.55484669882830764870E-15,
    -4.15056934728722208663E-14, 1.54008621752140982691E-14,  3.85277838274214270114E-13,
    7.18012445138366623367E-13,  -1.79417853150680611778E-12, -1.32158118404477131188E-11,
    -3.14991652796324136454E-11, 1.18891471078464383424E-11,  4.94060238822496958910E-10,
    3.39623202570838634515E-9,   2.26666899049817806459E-8,   2.04891858946906374183E-7,
    2.89137052083475648297E-6,   6.88975834691682398426E-5,   3.36911647825569408990E-3,
    8.04490411014108831608E-1,
};

template <typename T, std::size_t N>
T chbevl(T x, const double (&vals)[N]) noexcept {
  T b0 = static_cast<T>(vals[0]), b1 = 0, b2 = 0;
  for (std::size_t i = 1; i < N; ++i) {
    b2 = b1;
    b1 = b0;
    b0 = x * b1 - b2 + static_cast<T>(vals[i]);
  }
  return static_cast<T>(0.5) * (b0 - b2);
}

template <typename T>
T i0_value(T x) noexcept {
  x = std::fabs(x);
  if (x <= T{8}) return std::exp(x) * chbevl<T>(x / T{2} - T{2}, kI0A);
  return std::exp(x) * chbevl<T>(T{32} / x - T{2}, kI0B) / std::sqrt(x);
}

// Applies fn(in_ptr, out_ptr) over two same-shape arrays (out is C-contiguous).
template <typename Fn>
void map_into(const NDArray& in, NDArray& out, Fn&& fn) {
  std::byte* o = out.data();
  const auto step = static_cast<std::ptrdiff_t>(out.itemsize());
  for_each_element(in, [&](std::byte* p) {
    fn(p, o);
    o += step;
  });
}

DType real_float_for(DType d) {
  return (d == DType::Float16 || d == DType::Float32) ? d : DType::Float64;
}

template <typename T>
T float_max() noexcept {
  return std::numeric_limits<T>::max();
}

template <typename S>
void nan_to_num_real(std::byte* p, double nan, double pinf, double ninf) {
  using C = compute_t<S>;
  const C v = ld<S>(p);
  if (std::isnan(v)) st<S>(p, static_cast<C>(nan));
  else if (std::isinf(v)) st<S>(p, static_cast<C>(v > 0 ? pinf : ninf));
}

}  // namespace

NDArray p04_i0(const NDArray& x) {
  if (is_complex(x.dtype())) throw_error(ErrorKind::DType, "i0 not supported for complex values");
  const DType od = real_float_for(x.dtype());
  const NDArray in = x.dtype() == od ? x : x.astype(od);
  NDArray out = NDArray::empty(x.shape(), od);
  const FpScope fp("i0");
  dispatch_dtype(od, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (std::is_floating_point_v<compute_t<S>>) {
      map_into(in, out, [](std::byte* p, std::byte* o) { st<S>(o, i0_value(ld<S>(p))); });
    }
  });
  fp.check();
  return out;
}

NDArray p04_sinc(const NDArray& x) {
  const DType xd = x.dtype();
  const DType od = is_complex(xd) ? xd : real_float_for(xd);
  const NDArray in = xd == od ? x : x.astype(od);
  NDArray out = NDArray::empty(x.shape(), od);
  const FpScope fp("sin");
  dispatch_dtype(od, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (is_complex_v<S>) {
      using R = typename S::value_type;
      map_into(in, out, [](std::byte* p, std::byte* o) {
        S y = load<S>(p) * std::numbers::pi_v<R>;
        if (y == S{}) y = S{static_cast<R>(1e-20), 0};
        store<S>(o, std::sin(y) / y);
      });
    } else if constexpr (std::is_floating_point_v<compute_t<S>>) {
      using C = compute_t<S>;
      // NumPy substitutes finfo(dtype).eps for zeros.
      const C eps = std::is_same_v<S, float16_t> ? static_cast<C>(0.0009765625)
                                                 : std::numeric_limits<C>::epsilon();
      map_into(in, out, [eps](std::byte* p, std::byte* o) {
        C y = ld<S>(p) * std::numbers::pi_v<C>;
        if (y == C{0}) y = eps;
        st<S>(o, std::sin(y) / y);
      });
    }
  });
  fp.check();
  return out;
}

NDArray p04_nan_to_num(const NDArray& x, bool copy, double nan, std::optional<double> posinf,
                       std::optional<double> neginf) {
  const DType d = x.dtype();
  if (!is_inexact(d)) return copy ? x.copy() : x;
  NDArray r = copy ? x.copy() : x;
  if (!copy) r.check_writeable();
  dispatch_dtype(d, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (is_complex_v<S>) {
      using R = typename S::value_type;
      const double hi = posinf.value_or(static_cast<double>(float_max<R>()));
      const double lo = neginf.value_or(-static_cast<double>(float_max<R>()));
      for_each_element(r, [&](std::byte* p) {
        nan_to_num_real<R>(p, nan, hi, lo);
        nan_to_num_real<R>(p + sizeof(R), nan, hi, lo);
      });
    } else if constexpr (std::is_floating_point_v<compute_t<S>>) {
      const double fmax = std::is_same_v<S, float16_t> ? 65504.0 : static_cast<double>(float_max<compute_t<S>>());
      const double hi = posinf.value_or(fmax), lo = neginf.value_or(-fmax);
      for_each_element(r, [&](std::byte* p) { nan_to_num_real<S>(p, nan, hi, lo); });
    }
  });
  return r;
}

bool p04_imag_all_below(const NDArray& x, double tol) {
  if (!is_complex(x.dtype())) return false;
  bool ok = true;
  dispatch_dtype(x.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (is_complex_v<S>) {
      for_each_element(x, [&](std::byte* p) {
        if (!(std::fabs(static_cast<double>(load<S>(p).imag())) < tol)) ok = false;
      });
    }
  });
  return ok;
}

namespace {

// NumPy unwrap on contiguous lines: dd = diff, ddmod = mod(dd - low, period)
// + low (floor mod), boundary fix, zero corrections below `discont`, then
// p[1:] += cumsum(correction).
template <typename T>
void unwrap_lines(T* v, std::int64_t outer, std::int64_t n, std::int64_t inner, T period, T high,
                  bool ambiguous, double discont, bool xor_diff = false) {
  const T low = -high;
  for (std::int64_t o = 0; o < outer; ++o) {
    for (std::int64_t i = 0; i < inner; ++i) {
      T* line = v + o * n * inner + i;
      T prev = line[0];
      T acc = 0;
      for (std::int64_t k = 1; k < n; ++k) {
        const T cur = line[k * inner];
        // NumPy diff of a bool array is not_equal.
        const T dd = xor_diff ? static_cast<T>(cur != prev) : static_cast<T>(cur - prev);
        T ddmod;
        if constexpr (std::is_integral_v<T>) {
          T r = (dd - low) % period;
          if (r != 0 && ((r < 0) != (period < 0))) r += period;
          ddmod = r + low;
        } else {
          T r = std::fmod(dd - low, period);
          if (r != 0 && ((r < 0) != (period < 0))) r += period;
          else if (r == 0) r = std::copysign(T{0}, period);
          ddmod = r + low;
        }
        if (ambiguous && ddmod == low && dd > 0) ddmod = high;
        T corr = ddmod - dd;
        const double add = std::is_integral_v<T> ? static_cast<double>(dd < 0 ? -dd : dd)
                                                 : static_cast<double>(std::fabs(static_cast<double>(dd)));
        if (add < discont) corr = 0;
        acc += corr;
        prev = cur;
        line[k * inner] = cur + acc;
      }
    }
  }
}

}  // namespace

NDArray p04_unwrap(const NDArray& p, double period, std::optional<double> discont, std::int64_t axis,
                   bool integer, DType out) {
  if (p.ndim() == 0) throw_error(ErrorKind::Value, "diff requires input that is at least one dimensional");
  const auto nd = static_cast<std::int64_t>(p.ndim());
  const std::int64_t ax = normalize_axis(axis, nd);
  std::int64_t outer = 1, inner = 1;
  for (std::int64_t d = 0; d < ax; ++d) outer *= p.shape()[static_cast<std::size_t>(d)];
  for (std::int64_t d = ax + 1; d < nd; ++d) inner *= p.shape()[static_cast<std::size_t>(d)];
  const std::int64_t n = p.shape()[static_cast<std::size_t>(ax)];
  const double disc = discont.value_or(period / 2);
  if (integer) {
    NDArray w = p.astype(DType::Int64);
    const auto per = static_cast<std::int64_t>(period);
    if (per == 0) throw_error(ErrorKind::Value, "unwrap: integer period must be non-zero");
    // Python divmod(period, 2).
    std::int64_t high = per / 2, rem = per % 2;
    if (rem != 0 && rem < 0) {
      high -= 1;
      rem += 2;
    }
    unwrap_lines<std::int64_t>(reinterpret_cast<std::int64_t*>(w.data()), outer, n, inner, per, high,
                               rem == 0, disc, p.dtype() == DType::Bool);
    return w.dtype() == out ? w : w.astype(out);
  }
  const DType cd = out == DType::Float64 ? DType::Float64 : DType::Float32;
  NDArray w = p.astype(cd);
  if (cd == DType::Float64) {
    unwrap_lines<double>(reinterpret_cast<double*>(w.data()), outer, n, inner, period, period / 2, true, disc,
                         p.dtype() == DType::Bool);
  } else {
    const auto per = static_cast<float>(period);
    unwrap_lines<float>(reinterpret_cast<float*>(w.data()), outer, n, inner, per, per / 2, true, disc,
                        p.dtype() == DType::Bool);
  }
  return w.dtype() == out ? w : w.astype(out);
}

}  // namespace nativpy
