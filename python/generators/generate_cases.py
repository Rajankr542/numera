#!/usr/bin/env python3
"""Generate NumPy differential test cases for nativpy (PLAN §63).

Writes JSON files to tests/differential/cases/. Each case records the input
and the expected NumPy result (or expected error class). Non-finite floats,
-0 and integers beyond 2**53 are encoded as tagged objects so JSON is exact.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tests" / "differential" / "cases"

REAL_DTYPES = [
    "bool", "int8", "uint8", "int16", "uint16", "int32", "uint32",
    "int64", "uint64", "float16", "float32", "float64",
]
SMALL_INTS = ["int8", "uint8", "int16", "uint16", "int32", "uint32"]


def enc(v):
    """Encode a scalar or nested list into exact JSON."""
    if isinstance(v, list):
        return [enc(x) for x in v]
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v if abs(v) <= 2**53 else {"bigint": str(v)}
    if isinstance(v, float):
        if math.isnan(v):
            return {"float": "nan"}
        if math.isinf(v):
            return {"float": "inf" if v > 0 else "-inf"}
        if v == 0 and math.copysign(1.0, v) < 0:
            return {"float": "-0"}
        return v
    raise TypeError(type(v))


def describe(a: np.ndarray, values: bool = True) -> dict:
    d = {
        "dtype": str(a.dtype),
        "shape": list(a.shape),
        "strides": list(a.strides),
        "c_contiguous": bool(a.flags.c_contiguous),
        "f_contiguous": bool(a.flags.f_contiguous),
    }
    if values:
        d["values"] = enc(a.tolist())
    return d


def creation_cases() -> list[dict]:
    # Only inputs whose NumPy inference is JS-representable (see D-004).
    inputs = [
        [1, 2, 3], [1, 2.5], [True, False], [True, 2], [],
        [[1, 2], [3, 4]], [[[1], [2]], [[3], [4]]], [[], []], 5, 2.5, True,
        [0.1, -3.5, 1e300], [-7, 0, 7],
    ]
    cases = []
    for data in inputs:
        cases.append({"op": "array", "data": enc(data), "expected": describe(np.array(data))})
        for dt in REAL_DTYPES:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # 1e300 -> inf for float16/float32
                    a = np.array(data, dtype=dt)
            except (OverflowError, ValueError) as e:
                cases.append({"op": "array", "data": enc(data), "dtype": dt,
                              "error": "ValueError", "numpy_error": type(e).__name__})
                continue
            cases.append({"op": "array", "data": enc(data), "dtype": dt, "expected": describe(a)})
    # Out-of-range integer inputs for small integer dtypes (D-009).
    for data in [[300], [-1], [128], [-129], [65536], [2**31], [-(2**31) - 1], [2**32]]:
        for dt in SMALL_INTS:
            try:
                a = np.array(data, dtype=dt)
                cases.append({"op": "array", "data": data, "dtype": dt, "expected": describe(a)})
            except OverflowError as e:
                cases.append({"op": "array", "data": data, "dtype": dt,
                              "error": "ValueError", "numpy_error": type(e).__name__})
    for shape in [[0], [3], [2, 3], [2, 0, 4], [], [1, 1, 1, 1]]:
        for dt in REAL_DTYPES + ["complex64", "complex128"]:
            z = np.zeros(shape, dtype=dt)
            cases.append({"op": "zeros", "shape": shape, "dtype": dt,
                          "expected": describe(z, values=not dt.startswith("complex"))})
    return cases


def _in_range(v: float, dt: str) -> bool:
    """float->int casts outside the target range are platform-dependent in
    NumPy (C UB); only in-range values are compared (D-009)."""
    if math.isnan(v) or math.isinf(v):
        return False
    info = np.iinfo(dt)
    return info.min <= math.trunc(v) <= info.max


def astype_cases() -> list[dict]:
    sources = {
        "float64": [0.0, -0.0, 1.5, -1.5, 2.5, -2.7, 127.9, 255.0, 300.0, 65504.0,
                    65520.0, 1e-8, 3.4e38, 1e39, float("nan"), float("inf"), -float("inf")],
        "int64": [0, 1, -1, 127, 128, -128, -129, 255, 256, 32767, 32768, 65535,
                  65536, 2**31 - 1, 2**31, -(2**31), 2**53, 2**62, -(2**62)],
        "bool": [True, False],
        "uint8": [0, 1, 127, 128, 255],
        "float16": [0.0, 1.5, -2.0, 65504.0, 6.1e-5],
    }
    cases = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for src_dt, values in sources.items():
            for dt in REAL_DTYPES:
                vals = values
                if src_dt.startswith("float") and dt != "bool" and not dt.startswith("float"):
                    vals = [v for v in values if _in_range(v, dt)]
                src = np.array(vals, dtype=src_dt)
                cases.append({"op": "astype", "src_dtype": src_dt, "data": enc(src.tolist()),
                              "dtype": dt, "expected": describe(src.astype(dt))})
    return cases


def view_cases() -> list[dict]:
    base = np.arange(24, dtype=np.int64)
    specs = [
        ((3, 4), (32, 8), 0), ((3, 2), (32, 8), 8), ((4, 3), (8, 32), 0),
        ((24,), (8,), 0), ((12,), (16,), 0), ((5,), (-8,), 32), ((2, 3), (0, 8), 8),
        ((2, 3, 4), (96, 32, 8), 0), ((4, 3, 2), (8, 32, 96), 0), ((0, 3), (24, 8), 0),
        ((1, 5), (0, 8), 0), ((3, 3), (64, -8), 16),
    ]
    cases = []
    for shape, strides, offset in specs:
        v = np.lib.stride_tricks.as_strided(base[offset // 8:], shape=shape, strides=strides)
        flat = v.reshape(-1)
        cases.append({"op": "view", "n": 24, "shape": list(shape), "strides": list(strides),
                      "offset": offset, "expected": describe(v),
                      "view_shares": bool(np.may_share_memory(v, base)),
                      "reshape_flat": enc(flat.tolist()),
                      "reshape_shares": bool(np.may_share_memory(flat, base)),
                      "copy_strides": list(v.copy().strides)})
    # Out-of-bounds views: NumPy's as_strided would read garbage; nativpy must raise.
    for shape, strides, offset in [((25,), (8,), 0), ((5,), (-8,), 24), ((2,), (8,), -8),
                                   ((4, 4), (48, 16), 0)]:
        cases.append({"op": "view", "n": 24, "shape": list(shape), "strides": list(strides),
                      "offset": offset, "error": "ValueError"})
    return cases


def reshape_cases() -> list[dict]:
    a = np.arange(24)
    cases = []
    for shape in [[24], [2, 12], [4, -1], [-1, 2, 3], [2, 3, 4], [1, 24, 1], [3, 8], [-1]]:
        cases.append({"op": "reshape", "n": 24, "shape": shape,
                      "expected": describe(a.reshape(shape))})
    for shape in [[5, -1], [-1, -1], [25], [7, 3], [0, -1]]:
        try:
            a.reshape(shape)
        except ValueError as e:
            # nativpy splits NumPy's ValueError: bad -1 usage stays ValueError,
            # size mismatch is ShapeError (D-006).
            kind = "ValueError" if "unknown dimension" in str(e) else "ShapeError"
            cases.append({"op": "reshape", "n": 24, "shape": shape, "error": kind})
            continue
        raise AssertionError(f"expected NumPy to reject reshape {shape}")
    return cases


def strided_reshape_cases() -> list[dict]:
    """Reshape of non-contiguous views: NumPy returns a view when possible."""
    base = np.arange(24, dtype=np.int64)
    st = np.lib.stride_tricks.as_strided
    specs = [  # (shape, strides, byte offset, new shape)
        ((12,), (16,), 0, [3, 4]), ((12,), (16,), 0, [2, 2, 3]), ((6, 4), (8, 48), 0, [6, 2, 2]),
        ((6, 4), (8, 48), 0, [24]), ((2, 2, 4), (96, 64, 8), 0, [2, 2, 2, 2]),
        ((2, 2, 4), (96, 64, 8), 0, [2, 8]), ((2, 3), (0, 8), 0, [2, 3, 1]),
        ((2, 3), (0, 8), 0, [6]), ((4, 3), (8, 32), 0, [4, 3, 1]), ((4, 3), (8, 32), 0, [1, 4, 3]),
        ((4, 3), (8, 32), 0, [2, 2, 3]), ((3, 1, 2), (64, 8, 16), 0, [3, 2]),
        ((3, 1, 2), (64, 8, 16), 0, [3, 1, 1, 2]), ((5,), (-8,), 32, [5, 1]),
        ((0, 3), (24, 8), 0, [3, 0]),
    ]
    cases = []
    for shape, strides, offset, new_shape in specs:
        v = st(base[offset // 8:], shape=shape, strides=strides)
        r = v.reshape(new_shape)
        cases.append({"op": "strided_reshape", "n": 24, "shape": list(shape),
                      "strides": list(strides), "offset": offset, "new_shape": new_shape,
                      "expected": describe(r),
                      "shares": bool(np.may_share_memory(r, base))})
    return cases


def promotion_cases() -> list[dict]:
    all_dt = REAL_DTYPES + ["complex64", "complex128"]
    return [{"op": "promote", "a": a, "b": b, "expected": str(np.promote_types(a, b))}
            for a in all_dt for b in all_dt]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {
        "creation": creation_cases(),
        "astype": astype_cases(),
        "views": view_cases(),
        "reshape": reshape_cases(),
        "strided_reshape": strided_reshape_cases(),
        "promotion": promotion_cases(),
    }
    for name, cases in groups.items():
        payload = {"numpy_version": np.__version__, "group": name, "cases": cases}
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {len(cases):4d} cases -> tests/differential/cases/{name}.json")


if __name__ == "__main__":
    main()

