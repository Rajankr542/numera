#pragma once

// Sorting/selection kernels for P9 (D-120, D-121): ports of NumPy's scalar
// introsort (npysort/quicksort.cpp), heapsort (npysort/heapsort.h) and
// introselect (npysort/selection.cpp). They are written over generic "items"
// with a less-than functor: for a value sort the items are the values, for an
// arg sort the items are indices and `less` compares the values they point to.
// Both give exactly the comparisons of NumPy's quicksort_/aquicksort_ pair.

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>

#include "cast.hpp"
#include "dtype.hpp"

namespace nativpy::p09 {

// Sort buffers hold bool as uint8_t (std::vector<bool> has no contiguous data).
template <typename T>
using storage_t = std::conditional_t<std::is_same_v<T, bool>, std::uint8_t, T>;

// ---- element order (npy::*_tag::less / greater) ----

inline bool half_isnan(float16_t h) noexcept {
  return ((h.bits & 0x7c00u) == 0x7c00u) && ((h.bits & 0x03ffu) != 0u);
}
inline bool half_lt_nonan(std::uint16_t a, std::uint16_t b) noexcept {
  if (a & 0x8000u) {
    if (b & 0x8000u) return (a & 0x7fffu) > (b & 0x7fffu);
    return (a != 0x8000u) || (b != 0x0000u);  // signed zeros compare equal
  }
  if (b & 0x8000u) return false;
  return (a & 0x7fffu) < (b & 0x7fffu);
}

template <typename T>
bool is_nan_v(const T& v) noexcept {
  if constexpr (std::is_same_v<T, float16_t>) {
    return half_isnan(v);
  } else if constexpr (is_complex_v<T>) {
    return std::isnan(v.real()) || std::isnan(v.imag());
  } else if constexpr (std::is_floating_point_v<T>) {
    return std::isnan(v);
  } else {
    return false;
  }
}

template <typename T>
bool lt(const T& a, const T& b) noexcept {
  if constexpr (std::is_same_v<T, float16_t>) {
    if (half_isnan(b)) return !half_isnan(a);
    return !half_isnan(a) && half_lt_nonan(a.bits, b.bits);
  } else if constexpr (is_complex_v<T>) {
    const auto ra = a.real(), rb = b.real(), ia = a.imag(), ib = b.imag();
    if (ra < rb) return ia == ia || ib != ib;
    if (ra > rb) return ib != ib && ia == ia;
    if (ra == rb || (ra != ra && rb != rb)) return ia < ib || (ib != ib && ia == ia);
    return rb != rb;
  } else if constexpr (std::is_floating_point_v<T>) {
    return a < b || (b != b && a == a);
  } else {
    return a < b;
  }
}

// NaN sorts to the end in reverse too.
template <typename T>
bool gt(const T& a, const T& b) noexcept {
  if constexpr (std::is_same_v<T, float16_t>) {
    if (half_isnan(b)) return !half_isnan(a);
    return !half_isnan(a) && half_lt_nonan(b.bits, a.bits);
  } else if constexpr (is_complex_v<T>) {
    const auto ra = a.real(), rb = b.real(), ia = a.imag(), ib = b.imag();
    if (ra > rb) return ia == ia || ib != ib;
    if (ra < rb) return ib != ib && ia == ia;
    if (ra == rb || (ra != ra && rb != rb)) return ia > ib || (ib != ib && ia == ia);
    return rb != rb;
  } else if constexpr (std::is_floating_point_v<T>) {
    return a > b || (b != b && a == a);
  } else {
    return a > b;
  }
}

// NumPy `==` for the element type (NaN != NaN, -0 == +0).
template <typename T>
bool eq(const T& a, const T& b) noexcept {
  if constexpr (std::is_same_v<T, float16_t>) {
    if (half_isnan(a) || half_isnan(b)) return false;
    return a.bits == b.bits || ((a.bits | b.bits) & 0x7fffu) == 0;
  } else {
    return a == b;
  }
}

template <typename T>
inline constexpr bool inexact_v = std::is_floating_point_v<T> || is_complex_v<T> ||
                                  std::is_same_v<T, float16_t>;

inline int get_msb(std::int64_t n) noexcept {
  int depth = 0;
  auto u = static_cast<std::uint64_t>(n);
  while (u >>= 1) ++depth;
  return depth;
}

// ---- heapsort (npysort/heapsort.h) ----
template <typename Item, typename Less>
void heapsort(Item* start, std::int64_t n, Less less) {
  Item tmp;
  Item* a = start - 1;
  std::int64_t i, j, l;
  for (l = n >> 1; l > 0; --l) {
    tmp = a[l];
    for (i = l, j = l << 1; j <= n;) {
      if (j < n && less(a[j], a[j + 1])) j += 1;
      if (less(tmp, a[j])) {
        a[i] = a[j];
        i = j;
        j += j;
      } else {
        break;
      }
    }
    a[i] = tmp;
  }
  for (; n > 1;) {
    tmp = a[n];
    a[n] = a[1];
    n -= 1;
    for (i = 1, j = 2; j <= n;) {
      if (j < n && less(a[j], a[j + 1])) j++;
      if (less(tmp, a[j])) {
        a[i] = a[j];
        i = j;
        j += j;
      } else {
        break;
      }
    }
    a[i] = tmp;
  }
}

// ---- introsort (npysort/quicksort.cpp, quicksort_ / aquicksort_) ----
template <typename Item, typename Less>
void quicksort(Item* start, std::int64_t num, Less less) {
  if (num <= 1) return;
  constexpr int kStack = 128;  // PYA_QS_STACK
  constexpr std::int64_t kSmall = 15;  // SMALL_QUICKSORT
  Item vp;
  Item* pl = start;
  Item* pr = pl + num - 1;
  std::array<Item*, kStack> stack{};
  Item** sptr = stack.data();
  std::array<int, kStack> depth{};
  int* psdepth = depth.data();
  int cdepth = get_msb(num) * 2;
  Item *pm, *pi, *pj, *pk;

  for (;;) {
    if (cdepth < 0) {
      heapsort(pl, pr - pl + 1, less);
      goto stack_pop;
    }
    while ((pr - pl) > kSmall) {
      pm = pl + ((pr - pl) >> 1);
      if (less(*pm, *pl)) std::swap(*pm, *pl);
      if (less(*pr, *pm)) std::swap(*pr, *pm);
      if (less(*pm, *pl)) std::swap(*pm, *pl);
      vp = *pm;
      pi = pl;
      pj = pr - 1;
      std::swap(*pm, *pj);
      for (;;) {
        do {
          ++pi;
        } while (less(*pi, vp));
        do {
          --pj;
        } while (less(vp, *pj));
        if (pi >= pj) break;
        std::swap(*pi, *pj);
      }
      pk = pr - 1;
      std::swap(*pi, *pk);
      if (pi - pl < pr - pi) {
        *sptr++ = pi + 1;
        *sptr++ = pr;
        pr = pi - 1;
      } else {
        *sptr++ = pl;
        *sptr++ = pi - 1;
        pl = pi + 1;
      }
      *psdepth++ = --cdepth;
    }
    for (pi = pl + 1; pi <= pr; ++pi) {
      vp = *pi;
      pj = pi;
      pk = pi - 1;
      while (pj > pl && less(vp, *pk)) *pj-- = *pk--;
      *pj = vp;
    }
  stack_pop:
    if (sptr == stack.data()) break;
    pr = *(--sptr);
    pl = *(--sptr);
    cdepth = *(--psdepth);
  }
}

// ---- introselect (npysort/selection.cpp) ----
inline constexpr std::int64_t kMaxPivotStack = 50;  // NPY_MAX_PIVOT_STACK

inline void store_pivot(std::int64_t pivot, std::int64_t kth, std::int64_t* pivots,
                        std::int64_t* npiv) {
  if (pivots == nullptr) return;
  if (pivot == kth && *npiv == kMaxPivotStack) {
    pivots[*npiv - 1] = pivot;
  } else if (pivot >= kth && *npiv < kMaxPivotStack) {
    pivots[*npiv] = pivot;
    *npiv += 1;
  }
}

template <bool Inexact, typename Item, typename Less>
void introselect(Item* v, std::int64_t num, std::int64_t kth, std::int64_t* pivots,
                 std::int64_t* npiv, Less less);

template <typename Item, typename Less>
void median3_swap(Item* v, std::int64_t low, std::int64_t mid, std::int64_t high, Less less) {
  if (less(v[high], v[mid])) std::swap(v[high], v[mid]);
  if (less(v[high], v[low])) std::swap(v[high], v[low]);
  if (less(v[low], v[mid])) std::swap(v[low], v[mid]);
  std::swap(v[mid], v[low + 1]);
}

template <typename Item, typename Less>
std::int64_t median5(Item* v, Less less) {
  if (less(v[1], v[0])) std::swap(v[1], v[0]);
  if (less(v[4], v[3])) std::swap(v[4], v[3]);
  if (less(v[3], v[0])) std::swap(v[3], v[0]);
  if (less(v[4], v[1])) std::swap(v[4], v[1]);
  if (less(v[2], v[1])) std::swap(v[2], v[1]);
  if (less(v[3], v[2])) return less(v[3], v[1]) ? 1 : 3;
  return 2;
}

template <typename Item, typename Less>
void unguarded_partition(Item* v, const Item pivot, std::int64_t* ll, std::int64_t* hh, Less less) {
  for (;;) {
    do {
      (*ll)++;
    } while (less(v[*ll], pivot));
    do {
      (*hh)--;
    } while (less(pivot, v[*hh]));
    if (*hh < *ll) break;
    std::swap(v[*ll], v[*hh]);
  }
}

template <bool Inexact, typename Item, typename Less>
std::int64_t median_of_median5(Item* v, const std::int64_t num, std::int64_t* pivots,
                               std::int64_t* npiv, Less less) {
  const std::int64_t right = num - 1;
  const std::int64_t nmed = (right + 1) / 5;
  std::int64_t i, subleft;
  for (i = 0, subleft = 0; i < nmed; i++, subleft += 5) {
    const std::int64_t m = median5(v + subleft, less);
    std::swap(v[subleft + m], v[i]);
  }
  if (nmed > 2) introselect<Inexact>(v, nmed, nmed / 2, pivots, npiv, less);
  return nmed / 2;
}

template <typename Item, typename Less>
void dumb_select(Item* v, std::int64_t num, std::int64_t kth, Less less) {
  for (std::int64_t i = 0; i <= kth; i++) {
    std::int64_t minidx = i;
    Item minval = v[i];
    for (std::int64_t k = i + 1; k < num; k++) {
      if (less(v[k], minval)) {
        minidx = k;
        minval = v[k];
      }
    }
    std::swap(v[i], v[minidx]);
  }
}

template <bool Inexact, typename Item, typename Less>
void introselect(Item* v, std::int64_t num, std::int64_t kth, std::int64_t* pivots,
                 std::int64_t* npiv, Less less) {
  std::int64_t low = 0;
  std::int64_t high = num - 1;
  if (npiv == nullptr) pivots = nullptr;

  while (pivots != nullptr && *npiv > 0) {
    if (pivots[*npiv - 1] > kth) {
      high = pivots[*npiv - 1] - 1;
      break;
    } else if (pivots[*npiv - 1] == kth) {
      return;
    }
    low = pivots[*npiv - 1] + 1;
    *npiv -= 1;
  }

  if (kth - low < 3) {
    dumb_select(v + low, high - low + 1, kth - low, less);
    store_pivot(kth, kth, pivots, npiv);
    return;
  } else if (Inexact && kth == num - 1) {
    std::int64_t maxidx = low;
    Item maxval = v[low];
    for (std::int64_t k = low + 1; k < num; k++) {
      if (!less(v[k], maxval)) {
        maxidx = k;
        maxval = v[k];
      }
    }
    std::swap(v[kth], v[maxidx]);
    return;
  }

  int depth_limit = get_msb(num) * 2;
  for (; low + 1 < high;) {
    std::int64_t ll = low + 1;
    std::int64_t hh = high;
    if (depth_limit > 0 || hh - ll < 5) {
      const std::int64_t mid = low + (high - low) / 2;
      median3_swap(v, low, mid, high, less);
    } else {
      const std::int64_t mid =
          ll + median_of_median5<Inexact>(v + ll, hh - ll, nullptr, nullptr, less);
      std::swap(v[mid], v[low]);
      ll--;
      hh++;
    }
    depth_limit--;
    unguarded_partition(v, v[low], &ll, &hh, less);
    std::swap(v[low], v[hh]);
    if (hh != kth) store_pivot(hh, kth, pivots, npiv);
    if (hh >= kth) high = hh - 1;
    if (hh <= kth) low = ll;
  }
  if (high == low + 1) {
    if (less(v[high], v[low])) std::swap(v[high], v[low]);
  }
  store_pivot(kth, kth, pivots, npiv);
}

}  // namespace nativpy::p09
