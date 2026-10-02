#include "p10_conv.hpp"

#include <complex>
#include <cstdint>
#include <string>

#include "cast.hpp"
#include "error.hpp"
#include "ndarray.hpp"
#include "p10_common.hpp"

namespace nativpy::p10 {

namespace {

// Parse mode string → 0=full, 1=same, 2=valid.
int parse_mode(const std::string& mode) {
  if (mode == "full") return 0;
  if (mode == "same") return 1;
  if (mode == "valid") return 2;
  throw_error(ErrorKind::Value, "mode must be 'full', 'same', or 'valid'");
}

// Output length for each mode given signal length M and filter length K.
std::int64_t output_len(int mode, std::int64_t M, std::int64_t K) {
  switch (mode) {
    case 0: return M + K - 1;
    case 1: return (M >= K) ? M : K;   // max(M,K)
    case 2: return (M >= K ? M - K : K - M) + 1;  // |M-K|+1
    default: return 0;
  }
}

// Starting index into 'full' output for 'same' trimming.
// NumPy: trim_start = (K-1)//2  where K = min(M,K_orig) (the shorter array).
// We always pass the original K (filter length before any swap).
std::int64_t same_start(std::int64_t K) { return (K - 1) / 2; }

// Core cross-correlation kernel.
//
// Computes c[out_k] = Σ_j  signal[signal_start + out_k + j] · conj(filter[j])
// for out_k in [0, out_len) where the sum over j covers exactly the
// overlapping range.
//
// `conj_filter` — conjugate the filter element (true for correlate, false for
//                 convolve which has reversed v without conjugation).
// `rev_filter`  — traverse filter in reverse order (true for convolve).
//
// Both signal and filter must be C-contiguous flat arrays with the same DType.
template <typename S, typename C = compute_t<S>>
void correlate_kernel(const NDArray& signal, std::int64_t M, const NDArray& filter, std::int64_t K,
                      NDArray& out, std::int64_t out_len, std::int64_t full_start, bool conj_filter,
                      bool rev_filter) {
  for (std::int64_t ok = 0; ok < out_len; ++ok) {
    // In the 'full' sequence this corresponds to offset `full_start + ok`
    // relative to the start of the signal.
    const std::int64_t off = full_start + ok;  // = lag, can be negative

    C acc{};
    for (std::int64_t j = 0; j < K; ++j) {
      const std::int64_t si = off + j;  // index into signal
      if (si < 0 || si >= M) continue;

      const std::int64_t fi = rev_filter ? (K - 1 - j) : j;
      C sv = at<C, S>(signal, si);
      C fv = at<C, S>(filter, fi);
      if constexpr (is_complex_v<C>) {
        if (conj_filter) fv = std::conj(fv);
      }
      acc = static_cast<C>(acc + sv * fv);
    }
    put<S>(out, ok, acc);
  }
}

// Dispatch over all numeric dtypes and run the correlation / convolution.
NDArray run_conv(const NDArray& a_in, const NDArray& v_in, int mode, bool is_convolve) {
  const DType dt = promote_types(a_in.dtype(), v_in.dtype());
  const NDArray a = a_in.astype(dt);
  const NDArray v = v_in.astype(dt);

  const std::int64_t M = a.size();
  const std::int64_t K = v.size();

  // For 'valid' when K > M NumPy effectively swaps signal and filter and
  // conjugates.  For convolve reversing v is symmetric so no swap is needed.
  // We handle this via the `swapped` flag.
  bool swapped = false;
  const NDArray* sig = &a;
  const NDArray* fil = &v;
  std::int64_t sigLen = M;
  std::int64_t filLen = K;

  if (mode == 2 && K > M && !is_convolve) {
    // correlate(a, v, 'valid') with K > M:
    // NumPy returns correlate(v.conj(), a.conj(), 'valid').conj()
    // We implement this by swapping and setting the swap flag for post-conj.
    std::swap(sig, fil);
    sigLen = K;
    filLen = M;
    swapped = true;
  }

  const std::int64_t out_len = output_len(mode, sigLen, filLen);
  NDArray out = NDArray::empty({out_len}, dt);

  // Starting offset in the 'full' sequence: for 'full' it is -(filLen-1),
  // for 'valid' it is 0, for 'same' it is same_start trimmed from full_start.
  std::int64_t full_start = 0;
  if (mode == 0) full_start = -(filLen - 1);
  else if (mode == 1) full_start = -(filLen - 1) + same_start(filLen);
  // mode == 2: full_start = 0

  // For correlate: conj_filter=true, rev_filter=false.
  // For convolve:  conj_filter=false, rev_filter=true.
  // When swapped (valid, K>M): we compute correlate(v, a, 'valid') normally,
  // so conj_filter=true (conjugate the new filter = original a).
  const bool conj_filter = !is_convolve;
  const bool rev_filter  = is_convolve;

  dispatch_dtype(dt, [&](auto tag) {
    using S = dtype_t<decltype(tag)::value>;
    correlate_kernel<S>(*sig, sigLen, *fil, filLen, out, out_len, full_start, conj_filter, rev_filter);
  });

  // When we swapped (correlate valid, K>M), the result must be:
  //   1. Reversed (the swap reverses the lag order).
  //   2. Element-wise conjugated for complex dtypes.
  // This matches NumPy: correlate(a,v,'valid') = conj(correlate(v,a,'valid')[::-1])
  if (swapped) {
    // Reverse in-place.
    dispatch_dtype(dt, [&](auto tag) {
      using S = dtype_t<decltype(tag)::value>;
      for (std::int64_t i = 0, j = out_len - 1; i < j; ++i, --j) {
        const auto tmp = at<compute_t<S>, S>(out, i);
        put<S>(out, i, at<compute_t<S>, S>(out, j));
        put<S>(out, j, tmp);
      }
    });
    // Conjugate complex results.
    if (dt == DType::Complex64 || dt == DType::Complex128) {
      dispatch_dtype(dt, [&](auto tag) {
        using S = dtype_t<decltype(tag)::value>;
        using C = compute_t<S>;
        if constexpr (is_complex_v<C>) {
          for (std::int64_t i = 0; i < out_len; ++i) {
            C val = at<C, S>(out, i);
            put<S>(out, i, std::conj(val));
          }
        }
      });
    }
  }

  return out;
}

// Flatten a 1-D (or 0-D) array to a guaranteed 1-D C-contiguous array.
NDArray flatten1d(const NDArray& a, const char* name) {
  if (a.ndim() > 1)
    throw_error(ErrorKind::Value, std::string(name) + " must be 1-D");
  if (a.size() == 0)
    throw_error(ErrorKind::Value, std::string(name) + " must not be empty");
  // reshape to (n,) and get a C-contiguous copy
  return a.reshape({a.size()}).astype(a.dtype());
}

}  // namespace

NDArray correlate(const NDArray& a, const NDArray& v, const std::string& mode) {
  const int m = parse_mode(mode);
  const NDArray af = flatten1d(a, "a");
  const NDArray vf = flatten1d(v, "v");
  return run_conv(af, vf, m, /*is_convolve=*/false);
}

NDArray convolve(const NDArray& a, const NDArray& v, const std::string& mode) {
  const int m = parse_mode(mode);
  const NDArray af = flatten1d(a, "a");
  const NDArray vf = flatten1d(v, "v");
  return run_conv(af, vf, m, /*is_convolve=*/true);
}

}  // namespace nativpy::p10
