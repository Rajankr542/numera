"""Result positions that depend on an out-of-range float -> integer cast.

C leaves these casts undefined, and NumPy's values differ between platforms and
even between its SIMD and scalar loops (x86 `cvtt*` gives INT_MIN or wraps,
arm64 saturates). numera uses one fixed rule (D-009). The ufunc differential
cases list these positions so the tests skip them instead of comparing
against one platform's NumPy (D-231).
"""
from __future__ import annotations

import math
import warnings

import numpy as np


def decode(v):
    """Inverse of generate_cases.enc for operand data."""
    if isinstance(v, list):
        return [decode(x) for x in v]
    if isinstance(v, dict):
        if "re" in v:
            return complex(decode(v["re"]), decode(v["im"]))
        if "float" in v:
            return float(v["float"])
        return int(v["bigint"])
    return v


def _out_of_range(v, dt) -> np.ndarray:
    info = np.iinfo(dt)
    a = np.real(np.asarray(v))
    bad = np.zeros(a.shape, dtype=bool)
    if a.dtype.kind != "f":
        return bad
    for idx, x in np.ndenumerate(a):
        x = float(x)
        bad[idx] = not math.isfinite(x) or not info.min <= math.trunc(x) <= info.max
    return bad


def ufunc_cast_ub(fn, xs, opts, shape, out_dtype=None, where=None) -> tuple[list[int], str | None]:
    """(flat indices of undefined result positions, loop dtype name or None)."""
    kw = {"dtype": opts["dtype"]} if opts.get("dtype") else {}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            loop = np.asarray(fn(*xs, casting="unsafe", **kw))
        except (TypeError, ValueError, OverflowError):
            return [], None
    ld = loop.dtype
    bad = np.zeros(shape, dtype=bool)
    if ld.kind in "iu":  # float/complex operands cast into an integer loop
        for x in xs:
            a = np.asarray(x)
            if a.dtype.kind in "fc":
                bad |= np.broadcast_to(_out_of_range(a, ld), shape)
    if out_dtype is not None and np.dtype(out_dtype).kind in "iu" and ld.kind in "fc":
        bad |= np.broadcast_to(_out_of_range(loop, out_dtype), shape)
    if where is not None:
        bad &= np.broadcast_to(np.asarray(where).astype(bool), shape)
    return np.flatnonzero(bad).tolist(), ld.name
