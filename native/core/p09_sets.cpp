#include "p09_sets.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "cast.hpp"
#include "error.hpp"
#include "p09_sort_kernels.hpp"

namespace nativpy {

namespace {

using p09::eq;
using p09::is_nan_v;
using p09::lt;
using Idx = std::vector<std::int64_t>;

template <typename T>
std::vector<T> flat(const NDArray& a) {
  std::vector<T> v;
  v.reserve(static_cast<std::size_t>(a.size()));
  for_each_element(a, [&](const std::byte* p) { v.push_back(load<T>(p)); });
  return v;
}

template <typename T>
NDArray from_vec(const std::vector<T>& v, DType dt) {
  NDArray out = NDArray::empty({static_cast<std::int64_t>(v.size())}, dt);
  if (!v.empty()) std::memcpy(out.data(), v.data(), v.size() * sizeof(T));
  return out;
}

NDArray from_idx(const Idx& v) { return from_vec(v, DType::Int64); }

template <typename T>
Idx stable_argsort(const std::vector<T>& v) {
  Idx perm(v.size());
  std::iota(perm.begin(), perm.end(), std::int64_t{0});
  std::stable_sort(perm.begin(), perm.end(), [&v](std::int64_t x, std::int64_t y) {
    return lt(v[static_cast<std::size_t>(x)], v[static_cast<std::size_t>(y)]);
  });
  return perm;
}

template <typename T>
std::vector<T> permute(const std::vector<T>& v, const Idx& perm) {
  std::vector<T> out(perm.size());
  for (std::size_t i = 0; i < perm.size(); ++i) out[i] = v[static_cast<std::size_t>(perm[i])];
  return out;
}

// Builds the UniqueResult from a sorted permutation and the group-start mask.
template <typename T>
UniqueResult finish(const std::vector<T>& aux, const Idx& perm, const std::vector<bool>& mask,
                    DType dt, bool ri, bool rinv, bool rc) {
  std::vector<T> vals;
  Idx ind, starts;
  for (std::size_t i = 0; i < aux.size(); ++i) {
    if (!mask[i]) continue;
    vals.push_back(aux[i]);
    ind.push_back(perm[i]);
    starts.push_back(static_cast<std::int64_t>(i));
  }
  UniqueResult r{from_vec(vals, dt), std::nullopt, std::nullopt, std::nullopt};
  if (ri) r.indices = from_idx(ind);
  if (rinv) {
    Idx inv(aux.size());
    std::int64_t g = -1;
    for (std::size_t i = 0; i < aux.size(); ++i) {
      if (mask[i]) ++g;
      inv[static_cast<std::size_t>(perm[i])] = g;
    }
    r.inverse = from_idx(inv);
  }
  if (rc) {
    Idx counts(starts.size());
    for (std::size_t g = 0; g < starts.size(); ++g) {
      const std::int64_t end = g + 1 < starts.size() ? starts[g + 1] : static_cast<std::int64_t>(aux.size());
      counts[g] = end - starts[g];
    }
    r.counts = from_idx(counts);
  }
  return r;
}

template <typename T>
UniqueResult unique_impl(const std::vector<T>& v, DType dt, bool ri, bool rinv, bool rc, bool equal_nan) {
  const Idx perm = stable_argsort(v);
  const std::vector<T> aux = permute(v, perm);
  const std::size_t n = aux.size();
  std::vector<bool> mask(n, false);
  if (n > 0) mask[0] = true;
  std::size_t first_nan = n;
  if (equal_nan && n > 0 && is_nan_v(aux[n - 1])) {
    first_nan = 0;
    while (!is_nan_v(aux[first_nan])) ++first_nan;
  }
  for (std::size_t i = 1; i < n; ++i) {
    if (i < first_nan) mask[i] = !eq(aux[i], aux[i - 1]);
    else mask[i] = i == first_nan;
  }
  if (first_nan < n) mask[first_nan] = true;
  return finish(aux, perm, mask, dt, ri, rinv, rc);
}

template <typename T>
std::vector<T> unique_values(const std::vector<T>& v) {
  std::vector<T> s = v;
  std::stable_sort(s.begin(), s.end(), [](const T& x, const T& y) { return lt(x, y); });
  std::vector<T> out;
  const std::size_t n = s.size();
  std::size_t first_nan = n;
  if (n > 0 && is_nan_v(s[n - 1])) {
    first_nan = 0;
    while (!is_nan_v(s[first_nan])) ++first_nan;
  }
  for (std::size_t i = 0; i < n && i <= first_nan; ++i) {
    if (i == 0 || i == first_nan || !eq(s[i], s[i - 1])) out.push_back(s[i]);
  }
  return out;
}

template <typename T>
std::vector<T> concat(const std::vector<T>& a, const std::vector<T>& b) {
  std::vector<T> c = a;
  c.insert(c.end(), b.begin(), b.end());
  return c;
}

template <typename T>
std::vector<bool> member(const std::vector<T>& elem, const std::vector<T>& test) {
  std::vector<T> s = test;
  std::sort(s.begin(), s.end(), [](const T& x, const T& y) { return lt(x, y); });
  std::vector<bool> out(elem.size());
  for (std::size_t i = 0; i < elem.size(); ++i) {
    const T& x = elem[i];
    auto it = std::lower_bound(s.begin(), s.end(), x, [](const T& p, const T& q) { return lt(p, q); });
    bool found = false;
    for (; it != s.end() && !lt(x, *it); ++it) {
      if (eq(*it, x)) {
        found = true;
        break;
      }
    }
    out[i] = found;
  }
  return out;
}

void require_same(const NDArray& a, const NDArray& b) {
  if (a.dtype() != b.dtype()) throw_error(ErrorKind::DType, "set operands must have the same dtype");
}

}  // namespace

UniqueResult unique1d(const NDArray& a, bool ri, bool rinv, bool rc, bool equal_nan) {
  return dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    return unique_impl(flat<T>(a), a.dtype(), ri, rinv, rc, equal_nan);
  });
}

UniqueResult unique_rows(const NDArray& a, bool ri, bool rinv, bool rc) {
  if (a.ndim() != 2) throw_error(ErrorKind::Shape, "unique_rows expects a 2-D array");
  const std::int64_t n = a.shape()[0];
  const std::int64_t m = a.shape()[1];
  return dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    const std::vector<T> v = flat<T>(a);
    const auto row = [&](std::int64_t r) { return v.data() + r * m; };
    const auto row_lt = [&](std::int64_t x, std::int64_t y) {
      const T* p = row(x);
      const T* q = row(y);
      for (std::int64_t j = 0; j < m; ++j) {
        if (lt(p[j], q[j])) return true;
        if (lt(q[j], p[j])) return false;
      }
      return false;
    };
    const auto row_eq = [&](std::int64_t x, std::int64_t y) {
      const T* p = row(x);
      const T* q = row(y);
      for (std::int64_t j = 0; j < m; ++j) {
        if (!eq(p[j], q[j])) return false;
      }
      return true;
    };
    Idx perm(static_cast<std::size_t>(n));
    std::iota(perm.begin(), perm.end(), std::int64_t{0});
    std::stable_sort(perm.begin(), perm.end(), row_lt);
    std::vector<bool> mask(perm.size(), false);
    Idx keep;
    for (std::size_t i = 0; i < perm.size(); ++i) {
      mask[i] = i == 0 || !row_eq(perm[i], perm[i - 1]);
      if (mask[i]) keep.push_back(perm[i]);
    }
    // finish() over row ids, then expand the kept rows into values.
    UniqueResult r = finish(perm, perm, mask, DType::Int64, ri, rinv, rc);
    std::vector<T> vals;
    vals.reserve(keep.size() * static_cast<std::size_t>(m));
    for (const auto k : keep) vals.insert(vals.end(), row(k), row(k) + m);
    NDArray out = NDArray::empty({static_cast<std::int64_t>(keep.size()), m}, a.dtype());
    if (!vals.empty()) std::memcpy(out.data(), vals.data(), vals.size() * sizeof(T));
    r.values = out;
    return r;
  });
}

NDArray isin(const NDArray& elem, const NDArray& test, bool invert) {
  require_same(elem, test);
  NDArray out = NDArray::empty(elem.shape(), DType::Bool);
  dispatch_dtype(elem.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    const auto m = member(flat<T>(elem), flat<T>(test));
    auto* dst = reinterpret_cast<bool*>(out.data());
    for (std::size_t i = 0; i < m.size(); ++i) dst[i] = m[i] != invert;
  });
  return out;
}

IntersectResult intersect1d(const NDArray& a, const NDArray& b, bool assume_unique, bool return_indices) {
  require_same(a, b);
  return dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    std::vector<T> x = flat<T>(a);
    std::vector<T> y = flat<T>(b);
    Idx ind1, ind2;
    if (!assume_unique) {
      UniqueResult ux = unique_impl(x, a.dtype(), return_indices, false, false, true);
      UniqueResult uy = unique_impl(y, a.dtype(), return_indices, false, false, true);
      x = flat<T>(ux.values);
      y = flat<T>(uy.values);
      if (return_indices) {
        ind1 = flat<std::int64_t>(*ux.indices);
        ind2 = flat<std::int64_t>(*uy.indices);
      }
    }
    const std::vector<T> c = concat(x, y);
    const Idx perm = stable_argsort(c);
    const std::vector<T> aux = permute(c, perm);
    std::vector<T> vals;
    Idx i1, i2;
    const auto nx = static_cast<std::int64_t>(x.size());
    for (std::size_t i = 0; i + 1 < aux.size(); ++i) {
      if (!eq(aux[i + 1], aux[i])) continue;
      vals.push_back(aux[i]);
      std::int64_t p = perm[i];
      std::int64_t q = perm[i + 1] - nx;
      if (!assume_unique && return_indices) {
        p = ind1[static_cast<std::size_t>(p)];
        q = ind2[static_cast<std::size_t>(q)];
      }
      i1.push_back(p);
      i2.push_back(q);
    }
    IntersectResult r{from_vec(vals, a.dtype()), std::nullopt, std::nullopt};
    if (return_indices) {
      r.indices1 = from_idx(i1);
      r.indices2 = from_idx(i2);
    }
    return r;
  });
}

NDArray union1d(const NDArray& a, const NDArray& b) {
  require_same(a, b);
  return dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    return from_vec(unique_values(concat(flat<T>(a), flat<T>(b))), a.dtype());
  });
}

NDArray setxor1d(const NDArray& a, const NDArray& b, bool assume_unique) {
  require_same(a, b);
  return dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    std::vector<T> x = flat<T>(a);
    std::vector<T> y = flat<T>(b);
    if (!assume_unique) {
      x = unique_values(x);
      y = unique_values(y);
    }
    std::vector<T> aux = concat(x, y);
    std::stable_sort(aux.begin(), aux.end(), [](const T& p, const T& q) { return lt(p, q); });
    std::vector<T> out;
    const std::size_t n = aux.size();
    for (std::size_t i = 0; i < n; ++i) {
      const bool left = i == 0 || !eq(aux[i], aux[i - 1]);
      const bool right = i + 1 == n || !eq(aux[i + 1], aux[i]);
      if (left && right) out.push_back(aux[i]);
    }
    return from_vec(out, a.dtype());
  });
}

NDArray setdiff1d(const NDArray& a, const NDArray& b, bool assume_unique) {
  require_same(a, b);
  return dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    std::vector<T> x = flat<T>(a);
    std::vector<T> y = flat<T>(b);
    if (!assume_unique) {
      x = unique_values(x);
      y = unique_values(y);
    }
    const auto m = member(x, y);
    std::vector<T> out;
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (!m[i]) out.push_back(x[i]);
    }
    return from_vec(out, a.dtype());
  });
}

NDArray ediff1d(const NDArray& a, const std::optional<NDArray>& to_begin,
                const std::optional<NDArray>& to_end) {
  if (a.dtype() == DType::Bool) {
    throw_error(ErrorKind::DType,
                "numpy boolean subtract, the `-` operator, is not supported, use the bitwise_xor, "
                "the `^` operator, or the logical_xor function instead.");
  }
  if ((to_begin && to_begin->dtype() != a.dtype()) || (to_end && to_end->dtype() != a.dtype())) {
    throw_error(ErrorKind::DType, "ediff1d: to_begin/to_end must have the input dtype");
  }
  return dispatch_dtype(a.dtype(), [&](auto tag) {
    using T = p09::storage_t<dtype_t<decltype(tag)::value>>;
    const std::vector<T> x = flat<T>(a);
    std::vector<T> out;
    if (to_begin) {
      const auto b = flat<T>(*to_begin);
      out.insert(out.end(), b.begin(), b.end());
    }
    for (std::size_t i = 1; i < x.size(); ++i) {
      if constexpr (std::is_same_v<T, float16_t>) {
        out.push_back(double_to_half(half_to_double(x[i]) - half_to_double(x[i - 1])));
      } else if constexpr (std::is_integral_v<T>) {
        using U = std::make_unsigned_t<T>;
        out.push_back(static_cast<T>(static_cast<U>(static_cast<U>(x[i]) - static_cast<U>(x[i - 1]))));
      } else {
        out.push_back(x[i] - x[i - 1]);
      }
    }
    if (to_end) {
      const auto e = flat<T>(*to_end);
      out.insert(out.end(), e.begin(), e.end());
    }
    return from_vec(out, a.dtype());
  });
}

}  // namespace nativpy
