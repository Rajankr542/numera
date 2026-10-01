#include "reduce.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "cast.hpp"
#include "complex_kernels.hpp"
#include "error.hpp"
#include "shape.hpp"
#include "shape_ops.hpp"
#include "ufunc_kernels.hpp"

namespace nativpy {

namespace {

// float16 is computed as float (like the ufunc kernels, D-014).
template <typename S>
using compute_t = std::conditional_t<std::is_same_v<S, float16_t>, float, S>;

template <typename S>
compute_t<S> ld(const std::byte* p) noexcept {
  return cast_value<compute_t<S>>(load<S>(p));
}

template <typename T>
bool is_nan(T v) noexcept {
  if constexpr (std::is_floating_point_v<T>) return std::isnan(v);
  else return false;
}

// NumPy min/max treat -0.0 < +0.0 (order-independent result).
template <typename T>
bool sign_bit(T v) noexcept {
  if constexpr (std::is_floating_point_v<T>) return std::signbit(v);
  else return false;
}

// Input rearranged so that reduced axes are last and C-contiguous, or (cols)
// the untransposed input when the reduced axes are the leading axes (D-021).
struct Work {
  NDArray data;       // rows: kept dims + reduced dims; cols: n × rows; C-contiguous
  Shape out_shape;    // kept dims (keepdims applied)
  std::int64_t rows;  // number of output elements
  std::int64_t n;     // elements reduced per output
  bool trailing;      // reduced axes were already the trailing axes (D-021)
  bool cols;          // data is laid out n × rows (reduce down columns)
};

Work prepare(const NDArray& a, const std::optional<std::vector<std::int64_t>>& axis,
             bool keepdims, DType work_dtype, bool allow_cols = false) {
  const auto nd = static_cast<std::int64_t>(a.ndim());
  std::vector<bool> red(static_cast<std::size_t>(nd), !axis.has_value());
  if (axis) {
    for (const auto ax : normalize_axes(*axis, nd)) red[static_cast<std::size_t>(ax)] = true;
  }
  std::vector<std::int64_t> perm;
  Shape out_shape;
  std::int64_t rows = 1;
  std::int64_t n = 1;
  for (std::int64_t d = 0; d < nd; ++d) {
    const auto dim = a.shape()[static_cast<std::size_t>(d)];
    if (red[static_cast<std::size_t>(d)]) {
      n *= dim;
      if (keepdims) out_shape.push_back(1);
    } else {
      perm.push_back(d);
      out_shape.push_back(dim);
      rows *= dim;
    }
  }
  for (std::int64_t d = 0; d < nd; ++d) {
    if (red[static_cast<std::size_t>(d)]) perm.push_back(d);
  }
  // NumPy drops size-1 axes before choosing the loop order, so "trailing" (inner
  // pairwise loop) only compares the order of the non-unit axes (verified bit-exact).
  bool trailing = true;
  std::int64_t prev = -1;
  for (const auto d : perm) {
    if (a.shape()[static_cast<std::size_t>(d)] == 1) continue;
    trailing = trailing && d > prev;
    prev = d;
  }
  // Reduced axes all leading (e.g. axis=0 of a matrix): the untransposed input
  // is already n × rows, so callers that support it reduce down columns
  // instead of materializing a transposed copy.
  if (allow_cols && !trailing && nd > 0) {
    // Leading iff red = [true × k, false × (nd - k)] for some k.
    const auto kept = static_cast<std::int64_t>(std::count(red.begin(), red.end(), false));
    bool leading = true;
    for (std::int64_t d = 0; d < nd; ++d) {
      leading = leading && red[static_cast<std::size_t>(d)] == (d < nd - kept);
    }
    if (leading && rows > 1 && n > 1) {
      const NDArray c = (a.dtype() == work_dtype && a.is_c_contiguous()) ? a : a.astype(work_dtype);
      return {c, std::move(out_shape), rows, n, false, true};
    }
  }
  const NDArray t = nd == 0 ? a : transpose(a, perm);
  // D-021: read in place when the rearranged view is already C-contiguous
  // (reduced axes trailing) and no cast is needed. NDArray shares the buffer.
  if (t.dtype() == work_dtype && t.is_c_contiguous()) {
    return {t, std::move(out_shape), rows, n, trailing, false};
  }
  return {t.astype(work_dtype), std::move(out_shape), rows, n, trailing, false};
}

// NumPy pairwise summation (loops_utils.h.src, D-021) of get(lo..lo+n-1).
template <typename C, typename F>
C pairwise_sum(const F& get, std::int64_t lo, std::int64_t n) {
  if (n < 8) {
    C s = C(-0.0);  // NumPy's seed; callers add 0 so an all -0.0 sum is +0.0
    for (std::int64_t i = 0; i < n; ++i) s += get(lo + i);
    return s;
  }
  if (n <= 128) {
    C r[8];
    for (int j = 0; j < 8; ++j) r[j] = get(lo + j);
    std::int64_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (int j = 0; j < 8; ++j) r[j] += get(lo + i + j);
    }
    C res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) res += get(lo + i);
    return res;
  }
  std::int64_t n2 = n / 2;
  n2 -= n2 % 8;
  return pairwise_sum<C>(get, lo, n2) + pairwise_sum<C>(get, lo + n2, n - n2);
}

// NumPy's complex pairwise sum (pairwise_sum_C*, loops_utils.h.src) over n
// interleaved re/im scalars: 8 scalar lanes = 4 complex accumulators.
template <typename T>
std::complex<T> pairwise_csum(const T* x, std::int64_t n) {
  if (n < 8) {
    T rr = T(-0.0);  // NumPy's seed; callers add 0 so an all -0.0 sum is +0.0
    T ri = T(-0.0);
    for (std::int64_t i = 0; i < n; i += 2) {
      rr += x[i];
      ri += x[i + 1];
    }
    return {rr, ri};
  }
  if (n <= 128) {
    T r[8];
    for (int j = 0; j < 8; ++j) r[j] = x[j];
    std::int64_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (int j = 0; j < 8; ++j) r[j] += x[i + j];
    }
    T rr = (r[0] + r[2]) + (r[4] + r[6]);
    T ri = (r[1] + r[3]) + (r[5] + r[7]);
    for (; i < n; i += 2) {
      rr += x[i];
      ri += x[i + 1];
    }
    return {rr, ri};
  }
  std::int64_t n2 = n / 2;
  n2 -= n2 % 8;
  const std::complex<T> a = pairwise_csum(x, n2);
  const std::complex<T> b = pairwise_csum(x + n2, n - n2);
  return {a.real() + b.real(), a.imag() + b.imag()};
}

// Sum/prod steps; complex uses NumPy's loop formulas (D-033), not std::complex
// operator* (C99 Annex G NaN recovery differs from NumPy).
template <typename C>
C radd(C a, C b) noexcept {
  if constexpr (is_complex_v<C>) return kernels::cadd(a, b);
  else return kernels::add<C>(a, b);
}
template <typename C>
C rmul(C a, C b) noexcept {
  if constexpr (is_complex_v<C>) return kernels::cmul(a, b);
  else return kernels::mul<C>(a, b);
}

// min/max of a contiguous, non-empty row with D-017 NaN/signed-zero rules.
// Lanes use a plain compare-select; NaN is detected by accumulating `v - v`
// (NaN for NaN or ±inf input, so hits are confirmed by an exact rescan). A bool
// flag in the loop blocks vectorization (probe: 428 µs vs 87 µs at 1e6 f64).
// The D-017 signed-zero rule is applied by a rescan only when the result is 0.
template <typename T, bool IsMax>
T minmax_row(const T* x, std::int64_t n) {
  constexpr auto pick = [](T v, T acc) { return (IsMax ? v > acc : v < acc) ? v : acc; };
  constexpr int L = 16;
  T acc;
  T poison = T(0);
  std::int64_t i;
  if (n < 2 * L) {
    acc = x[0];
    for (i = 0; i < n; ++i) {
      if constexpr (std::is_floating_point_v<T>) poison += x[i] - x[i];
      acc = pick(x[i], acc);
    }
  } else {
    T lane[L];
    T pz[L] = {};
    for (int j = 0; j < L; ++j) lane[j] = x[j];
    for (i = L; i + L <= n; i += L) {
      for (int j = 0; j < L; ++j) {
        const T v = x[i + j];
        if constexpr (std::is_floating_point_v<T>) pz[j] += v - v;
        lane[j] = pick(v, lane[j]);
      }
    }
    acc = lane[0];
    for (int j = 1; j < L; ++j) acc = pick(lane[j], acc);
    if constexpr (std::is_floating_point_v<T>) {
      for (int j = 0; j < L; ++j) poison += pz[j] + (lane[j] - lane[j]);
    }
    // Scalar tail kept out of the lane loop so the lane loop stays vectorizable.
    for (; i < n; ++i) {
      if constexpr (std::is_floating_point_v<T>) poison += x[i] - x[i];
      acc = pick(x[i], acc);
    }
  }
  if constexpr (std::is_floating_point_v<T>) {
    if (poison != poison) {
      for (std::int64_t k = 0; k < n; ++k) {
        if (x[k] != x[k]) return x[k];  // first NaN (keeps its payload)
      }
    }
    if (acc == T(0)) {
      // max prefers +0.0, min prefers -0.0 (NumPy, D-017).
      for (std::int64_t k = 0; k < n; ++k) {
        if (x[k] == T(0) && std::signbit(x[k]) != IsMax) return x[k];
      }
    }
  }
  return acc;
}

void reject_complex(DType dt) {
  if (is_complex(dt)) {
    throw_error(ErrorKind::NotImplemented, "reductions on complex arrays are not implemented");
  }
}

const char* op_name(ReduceOp op) {
  switch (op) {
    case ReduceOp::Sum: return "add";
    case ReduceOp::Prod: return "multiply";
    case ReduceOp::Min: return "minimum";
    case ReduceOp::Max: return "maximum";
    default: return "mean";
  }
}

// Sum/prod/min/max over each row of `w` (dtype W), written as dtype W.
template <typename W>
void fold_rows(ReduceOp op, const Work& w, const std::optional<double>& initial, NDArray& out) {
  using C = compute_t<W>;
  const std::byte* src = w.data.data();
  std::byte* dst = out.data();
  const std::size_t isz = sizeof(W);
  if (w.n == 0 && !initial && (op == ReduceOp::Min || op == ReduceOp::Max)) {
    throw_error(ErrorKind::Value, std::string("zero-size array to reduction operation ") +
                                      op_name(op) + " which has no identity");
  }
  const auto step = [op](C acc, C v) {
    switch (op) {
      case ReduceOp::Sum: return radd<C>(acc, v);
      case ReduceOp::Prod: return rmul<C>(acc, v);
      default: break;
    }
    if constexpr (!is_complex_v<C>) {  // complex min/max: P1-3c
      if (op == ReduceOp::Min) {
        return (!is_nan(acc) && (is_nan(v) || v < acc || (v == acc && sign_bit(v)))) ? v : acc;
      }
      if (op == ReduceOp::Max) {
        return (!is_nan(acc) && (is_nan(v) || v > acc || (v == acc && !sign_bit(v)))) ? v : acc;
      }
    }
    return acc;
  };
  if (w.cols) {
    // n × rows input: stream whole rows into per-output accumulators. Same
    // per-output order as the row path (sequential), without a transpose copy.
    // unique_ptr<C[]> rather than std::vector<C>: avoids the vector<bool> proxy.
    const auto acc = std::make_unique<C[]>(static_cast<std::size_t>(w.rows));
    std::int64_t start = 0;
    if (initial || op == ReduceOp::Sum || op == ReduceOp::Prod) {
      const C init = initial ? cast_value<C>(*initial) : (op == ReduceOp::Prod ? C{1} : C{0});
      std::fill(acc.get(), acc.get() + w.rows, init);
    } else {
      for (std::int64_t r = 0; r < w.rows; ++r) acc[static_cast<std::size_t>(r)] = ld<W>(src + static_cast<std::size_t>(r) * isz);
      start = 1;
    }
    for (std::int64_t i = start; i < w.n; ++i) {
      const std::byte* line = src + static_cast<std::size_t>(i * w.rows) * isz;
      // Op dispatched outside the inner loop so it can vectorize.
      const auto sweep = [&](auto f) {
        for (std::int64_t r = 0; r < w.rows; ++r) {
          auto& a = acc[static_cast<std::size_t>(r)];
          a = f(a, ld<W>(line + static_cast<std::size_t>(r) * isz));
        }
      };
      switch (op) {
        case ReduceOp::Sum: sweep([](C a, C v) { return radd<C>(a, v); }); break;
        case ReduceOp::Prod: sweep([](C a, C v) { return rmul<C>(a, v); }); break;
        default: sweep(step); break;
      }
    }
    for (std::int64_t r = 0; r < w.rows; ++r) {
      store<W>(dst + static_cast<std::size_t>(r) * isz, cast_value<W>(acc[static_cast<std::size_t>(r)]));
    }
    return;
  }
  for (std::int64_t r = 0; r < w.rows; ++r) {
    const std::byte* row = src + static_cast<std::size_t>(r * w.n) * isz;
    // D-021 fast paths (no `initial`): pairwise float sum, lane min/max.
    if constexpr (std::is_arithmetic_v<W> && !std::is_same_v<W, bool>) {
      const auto* x = reinterpret_cast<const W*>(row);
      // NumPy sums pairwise only along the inner loop; reductions over leading
      // axes accumulate sequentially (verified bit-exact), so keep that order.
      // `if constexpr`: only instantiate for floats (small ints promote to int
      // inside pairwise_sum and would narrow; GCC -Wconversion).
      if constexpr (std::is_floating_point_v<W>) {
        if (!initial && w.trailing && op == ReduceOp::Sum) {
          const C s = pairwise_sum<C>([x](std::int64_t i) { return static_cast<C>(x[i]); }, 0, w.n);
          store<W>(dst + static_cast<std::size_t>(r) * isz, cast_value<W>(C{0} + s));
          continue;
        }
      }
      if (!initial && w.n > 0 && (op == ReduceOp::Min || op == ReduceOp::Max)) {
        std::byte* o = dst + static_cast<std::size_t>(r) * isz;
        store<W>(o, op == ReduceOp::Max ? minmax_row<W, true>(x, w.n) : minmax_row<W, false>(x, w.n));
        continue;
      }
    }
    if constexpr (is_complex_v<W>) {
      // NumPy complex sum: initial (or 0) + pairwise over interleaved scalars.
      if (w.trailing && op == ReduceOp::Sum) {
        using T = typename W::value_type;
        const W s = pairwise_csum(reinterpret_cast<const T*>(row), 2 * w.n);
        const W init = initial ? cast_value<W>(*initial) : W{};
        store<W>(dst + static_cast<std::size_t>(r) * isz, radd<W>(init, s));
        continue;
      }
    }
    C acc{};
    std::int64_t start = 0;
    if (initial) {
      acc = cast_value<C>(*initial);
    } else if (op == ReduceOp::Sum) {
      acc = C{0};
    } else if (op == ReduceOp::Prod) {
      acc = C{1};
    } else {
      acc = ld<W>(row);
      start = 1;
    }
    // Op dispatched outside the element loop (a switch inside it defeats
    // unswitching/vectorization; measured 3.6× slower on int sums).
    const auto run = [&](auto f) {
      for (std::int64_t i = start; i < w.n; ++i) acc = f(acc, ld<W>(row + static_cast<std::size_t>(i) * isz));
    };
    switch (op) {
      case ReduceOp::Sum: run([](C a, C v) { return radd<C>(a, v); }); break;
      case ReduceOp::Prod: run([](C a, C v) { return rmul<C>(a, v); }); break;
      default: run(step); break;
    }
    store<W>(dst + static_cast<std::size_t>(r) * isz, cast_value<W>(acc));
  }
}

// Complex mean (NumPy _mean): the complex sum in W, then `sum / count` with an
// integer count, which promotes to complex128 and uses its Smith divide before
// casting back (so inf+1j -> inf+nanj and an empty mean is nan+nanj).
template <typename W>
void complex_mean_rows(const Work& w, NDArray& out) {
  fold_rows<W>(ReduceOp::Sum, w, std::nullopt, out);
  using Z = std::complex<double>;
  const Z count{static_cast<double>(w.n), 0.0};
  std::byte* dst = out.data();
  for (std::int64_t r = 0; r < w.rows; ++r) {
    std::byte* o = dst + static_cast<std::size_t>(r) * sizeof(W);
    store<W>(o, cast_value<W>(kernels::cdiv(cast_value<Z>(load<W>(o)), count)));
  }
}

// Mean/var/std in accumulator dtype W (float32 or float64).
template <typename W>
void moment_rows(ReduceOp op, const Work& w, std::int64_t ddof, NDArray& out) {
  const auto* src = reinterpret_cast<const W*>(w.data.data());
  std::byte* dst = out.data();
  const W count = static_cast<W>(w.n);
  if (w.cols) {
    // n × rows input (leading-axis reduction): NumPy's sequential order.
    std::vector<W> s(static_cast<std::size_t>(w.rows), W{0});
    for (std::int64_t i = 0; i < w.n; ++i) {
      const W* line = src + i * w.rows;
      for (std::int64_t r = 0; r < w.rows; ++r) s[static_cast<std::size_t>(r)] += line[r];
    }
    for (auto& v : s) v /= count;
    if (op != ReduceOp::Mean) {
      std::vector<W> ss(static_cast<std::size_t>(w.rows), W{0});
      for (std::int64_t i = 0; i < w.n; ++i) {
        const W* line = src + i * w.rows;
        for (std::int64_t r = 0; r < w.rows; ++r) {
          const W d = line[r] - s[static_cast<std::size_t>(r)];
          const W sq = d * d;  // separate statement: no FMA contraction (NumPy rounds d*d)
          ss[static_cast<std::size_t>(r)] += sq;
        }
      }
      const W denom = static_cast<W>(std::max<std::int64_t>(w.n - ddof, 0));
      for (std::int64_t r = 0; r < w.rows; ++r) {
        W v = ss[static_cast<std::size_t>(r)] / denom;
        if (op == ReduceOp::Std) v = std::sqrt(v);
        s[static_cast<std::size_t>(r)] = v;
      }
    }
    for (std::int64_t r = 0; r < w.rows; ++r) {
      store<W>(dst + static_cast<std::size_t>(r) * sizeof(W), s[static_cast<std::size_t>(r)]);
    }
    return;
  }
  for (std::int64_t r = 0; r < w.rows; ++r) {
    const W* row = src + r * w.n;
    const auto sum_of = [&w](const auto& get) {
      if (w.trailing) return W{0} + pairwise_sum<W>(get, 0, w.n);
      W s = 0;  // leading-axis reductions: NumPy's sequential order
      for (std::int64_t i = 0; i < w.n; ++i) s += get(i);
      return s;
    };
    const W s = sum_of([row](std::int64_t i) { return row[i]; });
    W res = s / count;  // empty -> NaN (D-017)
    if (op != ReduceOp::Mean) {
      const W mean = res;
      const W ss = sum_of([row, mean](std::int64_t i) {
        const W d = row[i] - mean;
        return d * d;
      });
      const W denom = static_cast<W>(std::max<std::int64_t>(w.n - ddof, 0));
      res = ss / denom;
      if (op == ReduceOp::Std) res = std::sqrt(res);
    }
    store<W>(dst + static_cast<std::size_t>(r) * sizeof(W), res);
  }
}

}  // namespace

DType reduce_result_dtype(ReduceOp op, DType in) {
  const bool complex_ok = op == ReduceOp::Sum || op == ReduceOp::Prod || op == ReduceOp::Mean;
  if (!complex_ok) reject_complex(in);  // complex var/std/min/max: P1-3c..3d
  const bool intlike = in == DType::Bool || is_integer(in);
  switch (op) {
    case ReduceOp::Sum:
    case ReduceOp::Prod:
      if (!intlike) return in;
      return dtype_info(in).kind == 'u' ? DType::UInt64 : DType::Int64;
    case ReduceOp::Min:
    case ReduceOp::Max: return in;
    default: return intlike ? DType::Float64 : in;
  }
}

NDArray reduce(ReduceOp op, const NDArray& a, const ReduceOptions& opts) {
  const bool sum_prod = op == ReduceOp::Sum || op == ReduceOp::Prod;
  const bool complex_ok = sum_prod || op == ReduceOp::Mean;
  if (!complex_ok) reject_complex(a.dtype());
  const DType rdt = opts.dtype.value_or(reduce_result_dtype(op, a.dtype()));
  if (!complex_ok) reject_complex(rdt);
  if (op == ReduceOp::Min || op == ReduceOp::Max || sum_prod) {
    const Work w = prepare(a, opts.axis, opts.keepdims, rdt, /*allow_cols=*/true);
    NDArray out = NDArray::empty(w.out_shape, rdt);
    dispatch_dtype(rdt, [&](auto tag) {
      using W = dtype_t<decltype(tag)::value>;
      fold_rows<W>(op, w, opts.initial, out);
    });
    return out;
  }
  if (is_complex(rdt)) {  // complex mean (P1-3b). Complex input + real dtype drops imag (NumPy cast).
    const Work w = prepare(a, opts.axis, opts.keepdims, rdt, /*allow_cols=*/true);
    NDArray out = NDArray::empty(w.out_shape, rdt);
    if (rdt == DType::Complex64) complex_mean_rows<std::complex<float>>(w, out);
    else complex_mean_rows<std::complex<double>>(w, out);
    return out;
  }
  // mean/var/std: accumulate in float64, or float32 for float16/float32
  // results (NumPy keeps float32 and upcasts float16 to float32).
  const DType acc = (rdt == DType::Float16 || rdt == DType::Float32) ? DType::Float32
                                                                      : DType::Float64;
  const Work w = prepare(a, opts.axis, opts.keepdims, acc, /*allow_cols=*/true);
  NDArray tmp = NDArray::empty(w.out_shape, acc);
  if (acc == DType::Float32) moment_rows<float>(op, w, opts.ddof, tmp);
  else moment_rows<double>(op, w, opts.ddof, tmp);
  return tmp.dtype() == rdt ? tmp : tmp.astype(rdt);
}

NDArray arg_reduce(bool is_max, const NDArray& a, std::optional<std::int64_t> axis,
                   bool keepdims) {
  reject_complex(a.dtype());
  std::optional<std::vector<std::int64_t>> axes;
  if (axis) axes = std::vector<std::int64_t>{*axis};
  const Work w = prepare(a, axes, false, a.dtype());
  const char* name = is_max ? "argmax" : "argmin";
  if (w.n == 0) {
    throw_error(ErrorKind::Value, std::string("attempt to get ") + name + " of an empty sequence");
  }
  Shape out_shape = w.out_shape;
  if (keepdims) {
    if (axis) {
      out_shape.insert(out_shape.begin() + normalize_axis(*axis, static_cast<std::int64_t>(a.ndim())), 1);
    } else {
      out_shape.assign(a.ndim(), 1);
    }
  }
  NDArray out = NDArray::empty(out_shape, DType::Int64);
  auto* dst = reinterpret_cast<std::int64_t*>(out.data());
  dispatch_dtype(a.dtype(), [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    if constexpr (!is_complex_v<S>) {
      using C = compute_t<S>;
      const std::byte* src = w.data.data();
      for (std::int64_t r = 0; r < w.rows; ++r) {
        const std::byte* row = src + static_cast<std::size_t>(r * w.n) * sizeof(S);
        if constexpr (std::is_arithmetic_v<S> && !std::is_same_v<S, bool>) {
          // D-021: vectorized extreme value, then the first index holding it.
          // NaN: minmax_row returns a NaN, so find the first NaN. Otherwise
          // `==` treats -0.0 and +0.0 as equal, so the first zero of either
          // sign wins, exactly like NumPy's first-strictly-better scan.
          const auto* x = reinterpret_cast<const S*>(row);
          const S m = is_max ? minmax_row<S, true>(x, w.n) : minmax_row<S, false>(x, w.n);
          std::int64_t k = 0;
          if (is_nan(m)) {
            while (!is_nan(x[k])) ++k;
          } else {
            while (x[k] != m) ++k;
          }
          dst[r] = k;
          continue;
        }
        C best = ld<S>(row);
        std::int64_t bi = 0;
        for (std::int64_t i = 1; i < w.n && !is_nan(best); ++i) {
          const C v = ld<S>(row + static_cast<std::size_t>(i) * sizeof(S));
          if (is_nan(v) || (is_max ? v > best : v < best)) {
            best = v;
            bi = i;
          }
        }
        dst[r] = bi;
      }
    }
  });
  return out;
}

}  // namespace nativpy
