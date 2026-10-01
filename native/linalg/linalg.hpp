#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "ndarray.hpp"

namespace nativpy::linalg {

// NDArray-level linear algebra (PLAN §19/§20, M8). Semantics: D-018.
// All results are new C-contiguous arrays.

NDArray matmul(const NDArray& a, const NDArray& b);
NDArray dot(const NDArray& a, const NDArray& b);
NDArray inner(const NDArray& a, const NDArray& b);
NDArray outer(const NDArray& a, const NDArray& b);

NDArray det(const NDArray& a);
NDArray inv(const NDArray& a);
NDArray solve(const NDArray& a, const NDArray& b);

struct EigResult {
  NDArray eigenvalues;
  NDArray eigenvectors;
};
EigResult eig(const NDArray& a);     // complex OK (D-043)
NDArray eigvals(const NDArray& a);   // values only (JOBVR='N', D-043)
EigResult eigh(const NDArray& a);  // lower triangle (UPLO='L'); complex OK (D-042)
NDArray eigvalsh(const NDArray& a);  // values only (JOBZ='N', D-042)

struct SvdResult {
  std::optional<NDArray> u;
  NDArray s;
  std::optional<NDArray> vh;
};
SvdResult svd(const NDArray& a, bool full_matrices, bool compute_uv);

enum class QrMode { Reduced, Complete, R };
struct QrResult {
  std::optional<NDArray> q;
  NDArray r;
};
QrResult qr(const NDArray& a, QrMode mode);

struct LstsqResult {
  NDArray x;
  NDArray residuals;
  std::int64_t rank;
  NDArray s;
};
// rcond < 0 selects the NumPy default (eps * max(M, N)).
LstsqResult lstsq(const NDArray& a, const NDArray& b, double rcond);

// Norm order: `kind` is "default", "fro", "nuc" or "p" (use `p`, may be ±inf).
struct NormOrd {
  std::string kind = "default";
  double p = 2.0;
};
NDArray norm(const NDArray& a, const NormOrd& ord,
             const std::optional<std::vector<std::int64_t>>& axis, bool keepdims);

}  // namespace nativpy::linalg
