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

from ufunc_out_cases import ufunc_out_cases
from ufunc_dtype_casting_cases import ufunc_dtype_casting_cases
from ufunc_where_cases import ufunc_where_cases
from ufunc_order_cases import ufunc_order_cases

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
    if isinstance(v, complex):
        return {"re": enc(v.real), "im": enc(v.imag)}
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


def range_cases() -> list[dict]:
    """ones/full/arange/linspace/eye across dtypes (M2)."""
    cases = []
    for dt in REAL_DTYPES + ["complex64", "complex128"]:
        is_cx = dt.startswith("complex")
        cases.append({"op": "ones", "shape": [2, 3], "dtype": dt,
                      "expected": describe(np.ones((2, 3), dtype=dt), values=not is_cx)})
        if not is_cx:
            cases.append({"op": "eye", "args": [3, 4, -1], "dtype": dt,
                          "expected": describe(np.eye(3, 4, k=-1, dtype=dt))})
    for dt in REAL_DTYPES:
        for fill in [0, 1, 7, 2.5, -1.9, -1, 300]:
            if dt.startswith("uint") and isinstance(fill, float) and fill < 0:
                # D-009/D-012 divergence: NumPy casts negative floats into
                # unsigned types without checking (platform-dependent values);
                # nativpy raises.
                cases.append({"op": "full", "shape": [2, 2], "fill": fill, "dtype": dt,
                              "error": "ValueError", "divergence": "D-009"})
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    a = np.full((2, 2), fill, dtype=dt)
            except OverflowError:
                cases.append({"op": "full", "shape": [2, 2], "fill": fill, "dtype": dt,
                              "error": "ValueError"})
                continue
            cases.append({"op": "full", "shape": [2, 2], "fill": fill, "dtype": dt,
                          "expected": describe(a)})
    for fill in [5, 2.5, True]:
        cases.append({"op": "full", "shape": [3], "fill": fill,
                      "expected": describe(np.full(3, fill))})
    aranges = [[5], [2, 9], [10, 0, -3], [0, 1, 0.1], [0.5, 4], [-3, 3, 0.75], [5, 0],
               [0, 10, 3], [1e-3, 1, 0.1], [0, -1, -0.2]]
    for args in aranges:
        cases.append({"op": "arange", "args": args, "expected": describe(np.arange(*args))})
        for dt in ["int8", "int32", "float16", "float32", "float64"]:
            a = np.arange(*args, dtype=dt)
            cases.append({"op": "arange", "args": args, "dtype": dt, "expected": describe(a)})
    cases.append({"op": "arange", "args": [0, 5, 0], "dtype": "float64", "error": "ValueError"})
    cases.append({"op": "arange", "args": [0, 3, 1], "dtype": "bool", "error": "ValueError"})
    cases.append({"op": "arange", "args": [0, 2, 1], "dtype": "bool",
                  "expected": describe(np.arange(0, 2, 1, dtype=bool))})
    lin = [([0, 1, 5], True), ([0, 1, 5], False), ([-1, 1, 5], True), ([2, 3, 1], True),
           ([2, 3, 0], True), ([0, 10, 7], True), ([1, -1, 9], False), ([0, 1e-10, 3], True),
           ([3, 3, 4], True)]
    for args, endpoint in lin:
        for dt in ["float64", "float32", "int32", "int64"]:
            a = np.linspace(*args, endpoint=endpoint, dtype=dt)
            cases.append({"op": "linspace", "args": args, "endpoint": endpoint, "dtype": dt,
                          "expected": describe(a)})
    cases.append({"op": "linspace", "args": [0, 1, -1], "endpoint": True, "dtype": "float64",
                  "error": "ValueError"})
    for args in [[1, 1, 0], [2, 5, 3], [4, 2, -1], [3, 3, 5], [0, 3, 0], [3, 0, 0]]:
        cases.append({"op": "eye", "args": args, "dtype": "float64",
                      "expected": describe(np.eye(args[0], args[1], k=args[2]))})
    return cases


def _apply_shape_op(src: np.ndarray, op: str, arg):
    if op == "transpose":
        return src.transpose(arg) if arg else src.transpose()
    if op == "squeeze":
        return np.squeeze(src, axis=None if arg is None else tuple(arg))
    if op == "expand_dims":
        return np.expand_dims(src, tuple(arg))
    if op == "swapaxes":
        return np.swapaxes(src, *arg)
    if op == "moveaxis":
        return np.moveaxis(src, *arg)
    return src.ravel() if op == "ravel" else src.flatten()


def shape_op_cases() -> list[dict]:
    """transpose/squeeze/expand_dims/swapaxes/moveaxis/ravel/flatten (M3).

    Each spec runs on a C-contiguous arange and on its transpose (`t`).
    """
    specs = [
        ([2, 3, 4], "transpose", []), ([2, 3, 4], "transpose", [1, 0, 2]),
        ([2, 3, 4], "transpose", [-1, 0, 1]), ([6], "transpose", []), ([], "transpose", []),
        ([2, 3, 4], "transpose", [0, 0, 1]), ([2, 3], "transpose", [0]),
        ([2, 3], "transpose", [0, 2]),
        ([1, 3, 1], "squeeze", None), ([1, 3, 1], "squeeze", [0]),
        ([1, 3, 1], "squeeze", [-1, 0]), ([1, 3, 1], "squeeze", [1]), ([1], "squeeze", None),
        ([2, 3], "squeeze", None), ([1, 3, 1], "squeeze", [5]),
        ([2, 3], "expand_dims", [0]), ([2, 3], "expand_dims", [-1]),
        ([2, 3], "expand_dims", [1]), ([2, 3], "expand_dims", [0, 3]),
        ([2, 3], "expand_dims", [0, 0]), ([2, 3], "expand_dims", [4]), ([], "expand_dims", [0]),
        ([2, 3, 4], "swapaxes", [0, 2]), ([2, 3, 4], "swapaxes", [-1, 1]),
        ([2, 3, 4], "swapaxes", [0, 3]),
        ([2, 3, 4], "moveaxis", [[0], [-1]]), ([2, 3, 4], "moveaxis", [[0, 1], [-1, -2]]),
        ([2, 3, 4], "moveaxis", [[2], [0]]), ([2, 3, 4], "moveaxis", [[0, 0], [1, 2]]),
        ([2, 3, 4], "ravel", None), ([2, 3, 4], "flatten", None), ([1, 4], "ravel", None),
    ]
    cases = []
    for shape, op, arg in specs:
        a = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
        for transposed in [False, True]:
            base = {"op": op, "shape": shape, "t": transposed, "arg": arg}
            try:
                r = _apply_shape_op(a.T if transposed else a, op, arg)
            except np.exceptions.AxisError:
                cases.append({**base, "error": "IndexError"})
                continue
            except ValueError:
                cases.append({**base, "error": "ValueError"})
                continue
            cases.append({**base, "expected": describe(r),
                          "shares": bool(np.may_share_memory(r, a))})
    return cases


UFUNC_BINARY = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply,
                "divide": np.true_divide, "power": np.power, "mod": np.mod,
                "floorDivide": np.floor_divide}
UFUNC_UNARY = {"abs": np.abs, "negative": np.negative, "sqrt": np.sqrt,
               "exp": np.exp, "log": np.log}
APPROX = {"power", "sqrt", "exp", "log"}  # libm vs NumPy SIMD (D-014)


def _operand_values(dt: str, which: int) -> list:
    if dt == "bool":
        return [True, False, True, True, False, True]
    kind = np.dtype(dt).kind
    if kind == "u":
        return [0, 1, 2, 7, 200 if dt != "uint8" else 250, 3] if which == 0 else [1, 0, 3, 2, 5, 255]
    if kind == "i":
        return [-7, 0, 5, 127, -128, 3] if which == 0 else [2, 0, -3, 1, -1, 4]
    return [-7.5, 0.0, 2.25, float("nan"), float("inf"), -0.0] if which == 0 else \
        [2.0, 0.0, -3.0, 0.5, float("-inf"), 4.0]


def _run(fn, *args):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return fn(*args), None
        except (TypeError, ValueError) as e:
            return None, type(e).__name__
        except OverflowError:
            return None, "ValueError"  # D-009: nativpy raises ValueError


def ufunc_cases() -> list[dict]:
    cases = []
    pairs = [(d, d) for d in REAL_DTYPES] + [
        ("int8", "uint8"), ("int64", "uint64"), ("int32", "float32"),
        ("uint8", "float16"), ("bool", "int16"), ("int16", "float64")]
    for name, fn in UFUNC_BINARY.items():
        for da, db in pairs:
            av, bv = _operand_values(da, 0), _operand_values(db, 1)
            if name == "power" and np.dtype(db).kind in "iu":
                bv = [abs(x) % 5 for x in bv]  # non-negative exponents
            a = np.array(av, dtype=da)
            b = np.array(bv, dtype=db)
            r, err = _run(fn, a, b)
            base = {"op": name, "a_data": enc(av), "a_dtype": da, "b_data": enc(bv),
                    "b_dtype": db, "approx": name in APPROX}
            cases.append({**base, "error": err} if err else {**base, "expected": describe(r)})
    # Broadcasting shapes (float64 and int32).
    shapes = [((3, 1), (4,)), ((2, 3), ()), ((), (5,)), ((2, 1, 3), (4, 1)),
              ((0, 3), (1, 3)), ((1,), (0,)), ((3,), (4,)), ((2, 3), (3, 2))]
    for sa, sb in shapes:
        for dt in ("float64", "int32"):
            na, nb = int(np.prod(sa)), int(np.prod(sb))
            a = np.arange(na, dtype=dt).reshape(sa)
            b = (np.arange(nb, dtype=dt) + 1).reshape(sb)
            r, err = _run(np.add, a, b)
            base = {"op": "add_bcast", "a_shape": list(sa), "b_shape": list(sb), "dtype": dt}
            cases.append({**base, "error": err or "BroadcastError"} if r is None
                         else {**base, "expected": describe(r)})
    # Non-contiguous operand (transposed view), values only.
    a = np.arange(12.0).reshape(3, 4).T
    cases.append({"op": "sub_transposed", "expected": describe(a - np.arange(3.0))})
    for name, fn in UFUNC_UNARY.items():
        for dt in REAL_DTYPES:
            av = _operand_values(dt, 0)
            r, err = _run(fn, np.array(av, dtype=dt))
            base = {"op": name, "a_data": enc(av), "a_dtype": dt, "approx": name in APPROX}
            cases.append({**base, "error": err} if err else {**base, "expected": describe(r)})
    # NEP 50 weak scalars.
    for dt in ("bool", "int8", "uint8", "int32", "float16", "float32", "float64"):
        for s in (2, 2.5, 300, -1):
            a = np.array(_operand_values(dt, 0)[:3], dtype=dt)
            r, err = _run(np.add, a, s)
            base = {"op": "add_scalar", "a_data": enc(a.tolist()), "a_dtype": dt, "scalar": s}
            cases.append({**base, "error": err} if err else {**base, "expected": describe(r)})
    return cases


def promotion_cases() -> list[dict]:
    all_dt = REAL_DTYPES + ["complex64", "complex128"]
    return [{"op": "promote", "a": a, "b": b, "expected": str(np.promote_types(a, b))}
            for a in all_dt for b in all_dt]


def casting_cases() -> list[dict]:
    """np.can_cast over every dtype pair and casting rule (D-045)."""
    all_dt = REAL_DTYPES + ["complex64", "complex128"]
    rules = ["no", "equiv", "safe", "same_kind", "unsafe"]
    cases = [{"op": "can_cast", "a": a, "b": b, "arg": r, "expected": bool(np.can_cast(a, b, r))}
             for r in rules for a in all_dt for b in all_dt]
    # Default rule is "safe".
    cases += [{"op": "can_cast", "a": a, "b": b, "expected": bool(np.can_cast(a, b))}
              for a, b in (("float64", "float32"), ("int8", "int16"), ("uint64", "int64"))]
    # Array argument: its dtype is used, never its values (NumPy 2).
    cases.append({"op": "can_cast_array", "a": "int64", "data": 1000, "b": "int8",
                  "expected": bool(np.can_cast(np.array(1000), np.int8))})
    cases.append({"op": "can_cast_array", "a": "int8", "data": [1, 2], "b": "int16",
                  "expected": bool(np.can_cast(np.array([1, 2], np.int8), np.int16))})
    for bad in ("bogus", "SAFE"):
        try:
            np.can_cast("int8", "int16", bad)
            err = None
        except (TypeError, ValueError) as e:
            err = type(e).__name__
        cases.append({"op": "can_cast", "a": "int8", "b": "int16", "arg": bad, "error": err})
    return cases


# ---- indexing (M6) ----
# Index items are JSON-encoded: {"i": n} integer, {"s": [start, stop, step]}
# slice (null = None), "newaxis", "...", {"b": bool} 0-d bool, and
# {"arr": nested, "dtype": dt} integer/boolean index arrays.
def _dec_index(items: list) -> tuple:
    out = []
    for it in items:
        if it == "newaxis":
            out.append(np.newaxis)
        elif it == "...":
            out.append(Ellipsis)
        elif "i" in it:
            out.append(it["i"])
        elif "s" in it:
            out.append(slice(*it["s"]))
        elif "b" in it:
            out.append(it["b"])
        else:
            out.append(np.array(it["arr"], dtype=it["dtype"]))
    return tuple(out)


def _S(start=None, stop=None, step=None):
    return {"s": [start, stop, step]}


def _A(v, dt="int64"):
    return {"arr": v, "dtype": dt}


INDEX_CASES: list[tuple[list[int], list]] = [
    ([10], [{"i": 3}]), ([10], [{"i": -1}]), ([10], [{"i": 10}]), ([10], [{"i": -11}]),
    ([10], [_S(2, 7)]), ([10], [_S(None, None, -1)]), ([10], [_S(8, 1, -3)]),
    ([10], [_S(-3)]), ([10], [_S(100, 200)]), ([10], [_S(None, None, 4)]),
    ([10], [_S(-100, 100, 3)]), ([10], [_S(5, 2)]), ([10], [_S(None, None, 0)]),
    ([3, 4], [{"i": 1}, {"i": 2}]), ([3, 4], [_S(), _S(1, 3)]), ([3, 4], [{"i": -1}, _S(None, None, -2)]),
    ([3, 4], [_S(None, None, -1), _S(None, None, -1)]), ([3, 4], [{"i": 0}, {"i": 0}, {"i": 0}]),
    ([2, 3, 4], ["...", {"i": 1}]), ([2, 3, 4], [{"i": 1}, "..."]), ([2, 3, 4], ["..."]),
    ([2, 3, 4], [_S(), "...", _S(1, 2)]), ([2, 3, 4], ["newaxis", "...", "newaxis"]),
    ([2, 3, 4], [_S(), "newaxis", {"i": 1}]), ([2, 3, 4], ["...", "..."]),
    ([2, 3, 4], [{"i": 0}, {"i": 1}, {"i": 2}, "newaxis"]), ([2, 3, 4], [_S(3, 5), {"i": 0}]),
    ([], ["..."]), ([], ["newaxis"]), ([], [{"i": 0}]),
    ([10], [_A([0, 9, -1, 3])]), ([10], [_A([[1, 2], [3, 4]])]), ([10], [_A([10])]),
    ([10], [_A([], "int64")]), ([10], [_A([1, 2], "int8")]), ([10], [_A([1, 2], "uint32")]),
    ([3, 4], [_A([2, 0])]), ([3, 4], [_S(), _A([3, 0, 3])]), ([3, 4], [_A([0, 2]), _A([1, 3])]),
    ([3, 4], [_A([[0], [2]]), _A([1, 3])]), ([3, 4], [_A([0, 1]), _A([0, 1, 2])]),
    ([3, 4], [{"i": 1}, _A([0, 3])]), ([3, 4], [_A([1]), {"i": -1}]),
    ([2, 3, 4], [_A([0, 1]), _S(), _A([1, 2])]), ([2, 3, 4], [_S(), _A([0, 2]), _A([1, 3])]),
    ([2, 3, 4], [{"i": 0}, _S(), _A([1, 2])]), ([2, 3, 4], [_A([1, 0]), "...", _A([3])]),
    ([2, 3, 4], ["newaxis", _A([1]), _S(None, None, -1), _A([0, 1])]),
    ([2, 3, 4], [_S(1, 2), _A([[0, 1], [2, 0]])]), ([2, 3, 4], [_A([0]), "newaxis", _A([0])]),
    ([10], [_A([True, False] * 5, "bool")]), ([10], [_A([True] * 3, "bool")]),
    ([3, 4], [_A([True, False, True], "bool")]), ([3, 4], [_S(), _A([False] * 4, "bool")]),
    ([3, 4], [_A([[True, False, True, False], [False] * 4, [True, True, False, True]], "bool")]),
    ([2, 3, 4], [_A([[True, False, True], [False, True, True]], "bool"), _S(1, 3)]),
    ([2, 3, 4], [{"i": 1}, _A([True, False, True], "bool"), _A([0, 3])]),
    ([3, 4], [{"b": True}]), ([3, 4], [{"b": False}]), ([3, 4], [_S(), {"b": True}, {"i": 1}]),
    ([], [{"b": True}]), ([], [_A(True, "bool")]), ([3, 4], [_A([1.0], "float64")]),
]


def _run_index(fn):
    try:
        return fn(), None
    except IndexError:
        return None, "IndexError"
    except (TypeError, ValueError) as e:
        return None, type(e).__name__


def index_cases() -> list[dict]:
    cases = []
    for shape, index in INDEX_CASES:
        src = np.arange(int(np.prod(shape)), dtype="int64").reshape(shape)
        r, err = _run_index(lambda: src[_dec_index(index)])
        base = {"op": "getitem", "shape": shape, "index": index}
        if err:
            cases.append({**base, "error": err})
            continue
        r = np.asarray(r)
        cases.append({**base, "expected": describe(r), "shares": bool(np.shares_memory(src, r))})
    # Assignment: result is the whole array after a[index] = value.
    for dt in ("int32", "float64", "uint8", "bool"):
        for shape, index, value in [
            ([3, 4], [_S(), {"i": 1}], 7.9), ([3, 4], [{"i": 2}], [1, 2, 3, 4]),
            ([3, 4], [_S(None, None, -1), _S(1, 3)], [[1, 2]]), ([10], [_A([0, 0, 5])], [1, 2, 3]),
            ([3, 4], [_A([[True, False, True, False], [False] * 4, [True] * 4], "bool")], 5),
            ([2, 3, 4], [{"i": 0}, _S(), _A([1, 2])], [[9, 8]] * 3), ([10], [{"b": True}], 1),
            ([3, 4], [_S(), {"i": 1}], [1, 2]), ([10], [_A([10])], 0), ([3, 4], ["..."], -1),
        ]:
            a = np.arange(int(np.prod(shape))).reshape(shape).astype(dt)
            idx = _dec_index(index)

            def do(a=a, idx=idx, value=value):
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    a[idx] = np.array(value).astype(dt) if dt == "uint8" else value
                return a
            r, err = _run_index(do)
            base = {"op": "setitem", "shape": shape, "dtype": dt, "index": index, "arg": value}
            cases.append({**base, "error": err} if err else {**base, "expected": describe(r)})
    return cases


# D-016: flags.writeable after a chain of ops starting from arange(6).
# Each step name is replayed by the JS runner.
_WRITEABLE_STEPS = {
    "broadcast": lambda a: np.broadcast_to(a, (2,) + a.shape),
    "broadcast_same": lambda a: np.broadcast_to(a, a.shape),
    "row": lambda a: a[0],
    "slice": lambda a: a[..., 1:],
    "ellipsis0d": lambda a: a[(0,) * a.ndim + (Ellipsis,)],
    "fancy": lambda a: a[np.array([0])],
    "T": lambda a: a.T,
    "expand": lambda a: np.expand_dims(a, 0),
    "squeeze": lambda a: np.squeeze(a),
    "swap": lambda a: np.swapaxes(a, 0, -1),
    "moveaxis": lambda a: np.moveaxis(a, 0, -1),
    "reshape_flat": lambda a: a.reshape(-1),
    "reshape_23": lambda a: a.reshape(2, 3),
    "ravel": lambda a: np.ravel(a),
    "flatten": lambda a: a.flatten(),
    "copy": lambda a: a.copy(),
    "astype": lambda a: a.astype("float64"),
    "add": lambda a: a + 1,
    "strided": lambda a: np.lib.stride_tricks.as_strided(a, (2,), (a.itemsize * 2,)),
}


def writeable_cases() -> list[dict]:
    chains = [
        [], ["broadcast"], ["broadcast_same"], ["broadcast", "row"], ["broadcast", "slice"],
        ["broadcast", "ellipsis0d"], ["broadcast", "fancy"], ["broadcast", "T"],
        ["broadcast", "expand"], ["broadcast", "expand", "squeeze"], ["broadcast", "swap"],
        ["broadcast", "moveaxis"], ["broadcast", "reshape_flat"], ["broadcast_same", "reshape_23"],
        ["broadcast", "ravel"], ["broadcast", "flatten"], ["broadcast", "copy"],
        ["broadcast", "astype"], ["broadcast", "add"], ["broadcast", "strided"],
        ["broadcast", "copy", "row"], ["row_of_2d"], ["slice"], ["T"], ["strided"],
        ["reshape_23", "row"], ["broadcast", "row", "copy", "slice"],
    ]
    steps = {**_WRITEABLE_STEPS, "row_of_2d": lambda a: a.reshape(2, 3)[1]}
    cases = []
    for chain in chains:
        a = np.arange(6)
        for s in chain:
            a = steps[s](a)
        case = {"op": "writeable", "chain": chain, "expected": describe(a),
                "writeable": bool(a.flags.writeable)}
        try:
            a[(0,) * a.ndim] = 99
            case["set_error"] = None
        except ValueError as e:
            case["set_error"] = str(e)
        cases.append(case)
    return cases


# ---- reductions (M7, D-017) ----
REDUCE_FNS = {
    "sum": np.sum, "prod": np.prod, "min": np.min, "max": np.max, "mean": np.mean,
    "var": np.var, "std": np.std, "argmin": np.argmin, "argmax": np.argmax,
}
# Float results of these depend on summation order, so the generic cases keep a
# relative tolerance (D-017). Since D-021 nativpy matches NumPy's order on
# contiguous input; d021_cases() checks that exactly.
REDUCE_APPROX = {"sum", "prod", "mean", "var", "std"}


def _reduce_values(dt: str, n: int) -> list:
    if dt == "bool":
        return [i % 3 == 0 for i in range(n)]
    kind = np.dtype(dt).kind
    if kind == "u":
        return [(i * 7) % 11 for i in range(n)]
    if kind == "i":
        return [(i * 7) % 11 - 5 for i in range(n)]
    return [((i * 7) % 11 - 5) * 0.5 for i in range(n)]


def _run_reduce(fn, a, kw):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return fn(a, **kw), None
        except np.exceptions.AxisError:
            return None, "IndexError"  # D-012
        except ValueError as e:
            return None, type(e).__name__


def _reduce_case(name, data, dt, shape, kw, transpose=False) -> dict:
    a = np.array(data, dtype=dt).reshape(shape)
    if transpose:
        a = a.T
    call_kw = {**kw, "axis": tuple(kw["axis"])} if isinstance(kw.get("axis"), list) else kw
    r, err = _run_reduce(REDUCE_FNS[name], a, call_kw)
    base = {"op": "reduce", "fn": name, "a_data": enc(list(data)), "a_dtype": dt,
            "a_shape": list(shape), "kw": kw, "t": transpose}
    if err:
        return {**base, "error": err}
    r = np.asarray(r)
    base["approx"] = name in REDUCE_APPROX and r.dtype.kind == "f"
    # Complex prod uses complex multiply, whose rounding depends on the build's
    # FMA contraction (NumPy's arm64 complex64 loop fuses one product); compare
    # it with a tolerance, as complex_ufunc_cases() does for multiply.
    if name == "prod" and r.dtype.kind == "c":
        base["approx"] = True
    # D-025: NumPy's min/max sign of zero depends on its SIMD path (x86 baseline
    # vs AVX2/NEON disagree when both +0 and -0 are present). Such cases check
    # the D-017 rule (min -> -0.0, max -> +0.0) instead of the platform result.
    if (name in ("min", "max") and r.ndim == 0 and r.dtype.kind == "f" and r == 0
            and not np.dtype(dt).kind == "c"
            and any(v == 0 and math.copysign(1.0, v) < 0 for v in data)
            and any(v == 0 and math.copysign(1.0, v) > 0 for v in data)):
        base["zero_rule"] = True
    return {**base, "expected": describe(r)}


def reduce_cases() -> list[dict]:
    cases = []
    shape = (2, 3, 4)
    axis_kws = [{}, {"axis": 0}, {"axis": 1}, {"axis": -1}, {"axis": [0, 2]},
                {"axis": 1, "keepdims": True}, {"keepdims": True}]
    for dt in REAL_DTYPES:
        data = _reduce_values(dt, 24)
        for name in REDUCE_FNS:
            for kw in axis_kws:
                if name.startswith("arg") and isinstance(kw.get("axis"), list):
                    continue
                cases.append(_reduce_case(name, data, dt, shape, kw))
        # Non-contiguous (transposed) input.
        for name in ("sum", "max", "argmin", "mean"):
            cases.append(_reduce_case(name, data, dt, shape, {"axis": 1}, transpose=True))
    f = float("nan")
    specials = [
        ("max", [1.0, f, 3.0], "float64", (3,), {}),
        ("min", [1.0, f, 3.0], "float32", (3,), {}),
        ("argmax", [1.0, 5.0, f, f], "float64", (4,), {}),
        ("argmin", [2.0, f, 1.0, 0.5], "float64", (2, 2), {"axis": 0}),
        ("sum", [1.0, f], "float64", (2,), {}),
        ("min", [0.0, -0.0], "float64", (2,), {}),
        ("max", [-0.0, 0.0], "float64", (2,), {}),
        ("argmax", [3, 1, 3], "int64", (3,), {}),
        ("max", [], "float64", (0,), {}),
        ("max", [], "float64", (0,), {"initial": -1}),
        ("max", [], "float64", (0, 3), {"axis": 1}),
        ("max", [], "float64", (0, 3), {"axis": 0}),
        ("sum", [], "int32", (0, 3), {"axis": 0}),
        ("prod", [], "float64", (0,), {}),
        ("mean", [], "float64", (0,), {}),
        ("var", [], "float64", (3, 0), {"axis": 1}),
        ("argmin", [], "float64", (0,), {}),
        ("argmin", [], "float64", (0, 3), {"axis": 1}),
        ("sum", [1, 2, 3], "int64", (3,), {"axis": 1}),
        ("sum", [1, 2, 3, 4], "int64", (2, 2), {"axis": [0, -2]}),
        ("argmax", [1, 2], "int64", (2,), {"axis": 2}),
        ("sum", [5], "int64", (), {}),
        ("argmax", [5], "int64", (), {}),
        ("var", [1.0, 2.0, 4.0], "float64", (3,), {"ddof": 1}),
        ("std", [1.0, 2.0, 4.0], "float64", (3,), {"ddof": 3}),
        ("var", [1, 2, 4, 8], "int16", (2, 2), {"axis": 0, "ddof": 1}),
        ("sum", [100, 100, 100], "int8", (3,), {"dtype": "int8"}),
        ("sum", [1, 2], "int32", (2,), {"dtype": "float32"}),
        ("prod", [10, 30], "int64", (2,), {"dtype": "uint8"}),
        ("mean", [1, 2], "int32", (2,), {"dtype": "float32"}),
        ("sum", [1, 2, 3], "int64", (3,), {"initial": 10}),
        ("prod", [1, 2, 3], "float64", (3,), {"initial": 0.5}),
        ("min", [4, 5], "uint8", (2,), {"initial": 2}),
    ]
    for name, data, dt, shp, kw in specials:
        cases.append(_reduce_case(name, data, dt, shp, kw))
    cases.extend(d021_cases())
    return cases


def d021_cases() -> list[dict]:
    """D-021: contiguous float sum/mean/var/std match NumPy bit-for-bit
    (pairwise along the trailing axes, sequential down leading axes), and the
    lane min/max keeps NaN/signed-zero semantics. Compared exactly."""
    rng = np.random.default_rng(21)
    cases = []

    def add(name, arr, kw):
        arr = np.asarray(arr)
        c = _reduce_case(name, [float(v) for v in arr.ravel()], str(arr.dtype), arr.shape, kw)
        c["approx"] = False
        cases.append(c)

    for n in (7, 8, 9, 127, 128, 129, 1000, 4099):
        x = rng.standard_normal(n) * 100
        for name in ("sum", "mean", "var", "std", "min", "max"):
            add(name, x, {})
    add("sum", rng.standard_normal(1000).astype(np.float32), {})
    m = rng.standard_normal((300, 37))
    for name in ("sum", "mean", "std", "max"):
        add(name, m, {"axis": 0})
        add(name, m, {"axis": 1})
    add("sum", rng.standard_normal((4, 50, 3)), {"axis": [0, 1]})
    add("sum", rng.standard_normal((4, 50, 3)), {"axis": [1, 2]})
    for name in ("min", "max"):
        y = rng.standard_normal(1000)
        y[0] = np.nan
        add(name, y, {})
        y = rng.standard_normal(1000)
        y[5], y[900] = np.inf, np.nan
        add(name, y, {})
        add(name, np.array([np.inf, -np.inf] * 100), {})
        add(name, np.array([-0.0] * 100 + [0.0] + [-0.0] * 99), {})
        add(name, np.array([0.0] * 100 + [-0.0] + [0.0] * 99), {})
        add(name, np.array([0.0, -1.0] * 100 + [-0.0]), {})
        z = rng.standard_normal((20, 50))
        z[3, 7] = z[11, 40] = np.nan
        add(name, z, {"axis": 1})
        add(name, z, {"axis": 0})
    add("sum", np.array([-0.0] * 3), {})
    add("sum", np.array([-0.0] * 300), {})
    # argmin/argmax: vectorized value search + first-index scan (D-021).
    for name in ("argmin", "argmax"):
        add(name, rng.standard_normal(1000), {})
        add(name, rng.integers(0, 5, 1000).astype(np.float64), {})  # many ties
        add(name, rng.integers(-9, 9, 777).astype(np.int32), {})
        y = rng.standard_normal(1000)
        y[640], y[900] = np.nan, np.nan
        add(name, y, {})
        add(name, np.array([0.0, -0.0] * 200), {})
        add(name, np.array([-0.0, 0.0] * 200), {})
        y = rng.standard_normal(300)
        y[250] = np.inf
        y[260] = -np.inf
        add(name, y, {})
        z = rng.standard_normal((20, 50))
        z[3, 7] = np.nan
        add(name, z, {"axis": 1})
        add(name, z, {"axis": 0})
    return cases


def _linalg_run(fn):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return fn(), None
        except np.linalg.LinAlgError:
            return None, "LinAlgError"
        except np.exceptions.AxisError:
            return None, "IndexError"
        except (ValueError, TypeError) as e:
            return None, type(e).__name__


def _mat(dt, shape, seed):
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape)) if shape else 1
    if dt == "bool":
        return (rng.integers(0, 2, n).astype(bool)).reshape(shape)
    kind = np.dtype(dt).kind
    if kind in "iu":
        lo = 0 if kind == "u" else -4
        return rng.integers(lo, 5, n).astype(dt).reshape(shape)
    return (rng.integers(-8, 9, n) * 0.25).astype(dt).reshape(shape)


def _lin_case(fn, inputs, kw, call):
    r, err = _linalg_run(call)
    base = {"op": "linalg", "fn": fn,
            "kw": {k: enc(v) if isinstance(v, float) else v for k, v in kw.items()},
            "inputs": [{"data": enc(x.ravel().tolist()), "dtype": str(x.dtype),
                        "shape": list(x.shape)} for x in inputs]}
    if err:
        return {**base, "error": err}
    if isinstance(r, tuple):
        return {**base, "expected": [int(v) if fn == "lstsq" and i == 2
                                     else describe_c(np.asarray(v))
                                     for i, v in enumerate(r)]}
    return {**base, "expected": describe_c(np.asarray(r))}


def describe_c(a: np.ndarray) -> dict:
    """describe() that splits complex arrays into real/imag value lists."""
    if a.dtype.kind == "c":
        d = describe(a.real, values=True)
        d["dtype"] = str(a.dtype)
        d["imag"] = enc(a.imag.tolist())
        return d
    return describe(a)


def linalg_product_cases() -> list[dict]:
    cases = []
    prod_fns = {"matmul": np.matmul, "dot": np.dot, "inner": np.inner, "outer": np.outer}
    shapes = [((3, 4), (4, 2)), ((4,), (4,)), ((4,), (4, 3)), ((2, 3), (3,)),
              ((2, 2, 3), (3, 2)), ((2, 1, 2, 3), (3, 3, 2)), ((0, 3), (3, 2)),
              ((2, 0), (0, 3)), ((3, 4), (3, 2)), ((2, 3, 4), (5, 4, 2))]
    for i, (sa, sb) in enumerate(shapes):
        for dt in ("float64", "float32", "int32", "int8", "bool", "uint16"):
            a, b = _mat(dt, sa, i), _mat(dt, sb, i + 100)
            for fn, f in prod_fns.items():
                if fn == "inner":
                    b = _mat(dt, sb[:-1] + sa[-1:], i + 7)
                cases.append(_lin_case(fn, [a, b], {}, lambda f=f, a=a, b=b: f(a, b)))
    a, b = _mat("int16", (2, 3), 1), _mat("float32", (3, 2), 2)
    cases.append(_lin_case("matmul", [a, b], {}, lambda: np.matmul(a, b)))
    c, d = _mat("int64", (2, 3), 1), _mat("float32", (3, 2), 2)
    cases.append(_lin_case("matmul", [c, d], {}, lambda: np.matmul(c, d)))
    w = np.full((2, 3), 100, np.int8)
    wt = w.T.copy()
    cases.append(_lin_case("matmul", [w, wt], {}, lambda: np.matmul(w, wt)))
    h = _mat("float16", (2, 3), 3)
    ht = h.T.copy()
    cases.append(_lin_case("matmul", [h, ht], {}, lambda: np.matmul(h, ht)))
    s = np.array(3.0)
    cases.append(_lin_case("dot", [s, a], {}, lambda: np.dot(s, a)))
    return cases


def linalg_decomp_cases() -> list[dict]:
    cases = []
    well = np.array([[4, 1, 2], [1, 5, 1], [2, 1, 6]], np.float64)
    squares = [well, well.astype(np.float32), well.astype(np.int64), _mat("float64", (2, 3, 3), 3),
               np.array([[1, 2], [2, 4]], np.float64), np.zeros((0, 0)), np.eye(4) * 2,
               _mat("float64", (4, 4), 9), np.array([[0, 1], [-1, 0]], np.float64),
               well.astype(bool)]
    for s in squares:
        cases.append(_lin_case("det", [s], {}, lambda s=s: np.linalg.det(s)))
        cases.append(_lin_case("inv", [s], {}, lambda s=s: np.linalg.inv(s)))
        rhs = _mat("float64", s.shape[:-1], 5)
        cases.append(_lin_case("solve", [s, rhs], {}, lambda s=s, r=rhs: np.linalg.solve(s, r)))
        rhs2 = _mat("float64", s.shape[:-1] + (2,), 6)
        cases.append(_lin_case("solve", [s, rhs2], {}, lambda s=s, r=rhs2: np.linalg.solve(s, r)))
        sym = s + np.swapaxes(s, -1, -2)
        cases.append(_lin_case("eigh", [sym], {}, lambda m=sym: np.linalg.eigh(m)))
        cases.append(_lin_case("eig", [s], {}, lambda s=s: np.linalg.eig(s)))
    rect = _mat("float64", (2, 3), 4)
    vec = np.arange(3.0)
    h = np.eye(2, dtype=np.float16)
    nanm = np.array([[np.nan, 1], [0, 1]])
    for fn, f in (("det", np.linalg.det), ("inv", np.linalg.inv), ("eig", np.linalg.eig),
                  ("eigh", np.linalg.eigh)):
        for bad in (rect, vec, h):
            cases.append(_lin_case(fn, [bad], {}, lambda f=f, x=bad: f(x)))
    cases.append(_lin_case("eig", [nanm], {}, lambda: np.linalg.eig(nanm)))
    cases.append(_lin_case("solve", [well, vec[:2]], {}, lambda: np.linalg.solve(well, vec[:2])))
    return cases


def linalg_svd_qr_cases() -> list[dict]:
    cases = []
    well = np.array([[4, 1, 2], [1, 5, 1], [2, 1, 6]], np.float64)
    rects = [_mat("float64", (4, 3), 11), _mat("float64", (3, 5), 12), well,
             _mat("float32", (5, 2), 13), _mat("int32", (3, 3), 14), _mat("float64", (2, 4, 3), 15),
             np.array([[1, 2], [2, 4], [3, 6]], np.float64), np.zeros((0, 3))]
    for r in rects:
        for full in (True, False):
            cases.append(_lin_case("svd", [r], {"fullMatrices": full},
                                   lambda r=r, full=full: np.linalg.svd(r, full_matrices=full)))
        cases.append(_lin_case("svd", [r], {"computeUV": False},
                               lambda r=r: np.linalg.svd(r, compute_uv=False)))
        for mode in ("reduced", "complete", "r"):
            cases.append(_lin_case("qr", [r], {"mode": mode},
                                   lambda r=r, mode=mode: np.linalg.qr(r, mode=mode)))
        if r.ndim == 2 and r.size > 0:
            b = _mat("float64", (r.shape[0],), 16)
            cases.append(_lin_case("lstsq", [r, b], {}, lambda r=r, b=b: np.linalg.lstsq(r, b)))
            b2 = _mat("float64", (r.shape[0], 2), 17)
            cases.append(_lin_case("lstsq", [r, b2], {}, lambda r=r, b=b2: np.linalg.lstsq(r, b)))
    return cases


def linalg_norm_cases() -> list[dict]:
    cases = []
    inf = float("inf")
    vec = _mat("float64", (5,), 20)
    m2 = _mat("float64", (3, 4), 21)
    m3 = _mat("float32", (2, 3, 4), 22)
    for ord_ in (None, 1, 2, inf, -inf, 0, 3, -1, 0.5):
        cases.append(_lin_case("norm", [vec], {"ord": ord_},
                               lambda o=ord_: np.linalg.norm(vec, ord=o)))
    for ord_ in (None, "fro", "nuc", 1, -1, 2, -2, inf, -inf, 3):
        cases.append(_lin_case("norm", [m2], {"ord": ord_},
                               lambda o=ord_: np.linalg.norm(m2, ord=o)))
    for kw in ({"axis": 0}, {"axis": -1, "keepdims": True}, {"axis": [1, 2]},
               {"axis": [2, 0], "ord": 1}, {"keepdims": True}, {"axis": [0, 1], "ord": "nuc"},
               {"axis": 3}):
        ax = kw.get("axis")
        call = {**kw, "axis": tuple(ax) if isinstance(ax, list) else ax}
        cases.append(_lin_case("norm", [m3], kw, lambda c=call: np.linalg.norm(m3, **c)))
    ints = _mat("int32", (3, 3), 23)
    cases.append(_lin_case("norm", [ints], {}, lambda: np.linalg.norm(ints)))
    cases.append(_lin_case("norm", [vec], {"ord": "fro"}, lambda: np.linalg.norm(vec, ord="fro")))
    cases.append(_lin_case("norm", [m3], {"ord": 1}, lambda: np.linalg.norm(m3, ord=1)))
    return cases


def _rand_result(v) -> dict:
    if isinstance(v, np.ndarray):
        return {"dtype": str(v.dtype), "shape": list(v.shape), "values": enc(v.tolist())}
    if isinstance(v, np.generic):
        v = v.item()
    return {"scalar": enc(v)}


def _rand_call(rng, m: str, args: list, kw: dict) -> dict:
    """Runs one call; `shuffle` operates on arange(prod(shape)).reshape(shape)."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if m == "shuffle":
                x = np.arange(int(np.prod(args[0]))).reshape(args[0])
                rng.shuffle(x, **kw)
                r = x
            else:
                r = getattr(rng, m)(*args, **kw)
        res = _rand_result(r)
    except (TypeError, ValueError) as e:
        res = {"error": type(e).__name__}
    return {"m": m, "args": enc(list(args)), "kw": kw, "result": res}


def _rand_case(api: str, seed, calls: list) -> dict:
    s = seed
    if isinstance(seed, dict):  # {"bigint": "..."} tag for large ints
        s = int(seed["bigint"])
    rng = np.random.default_rng(s) if api == "gen" else np.random.RandomState(s)
    return {"op": "random", "api": api, "seed": enc(s) if not isinstance(s, list) else enc(s),
            "calls": [_rand_call(rng, m, a, k) for m, a, k in calls]}


def _fft_run(fn):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            return fn(), None
        except np.exceptions.AxisError:
            return None, "IndexError"
        except (ValueError, TypeError, IndexError, ZeroDivisionError) as e:
            return None, type(e).__name__


def _flat_interleaved(a: np.ndarray) -> list:
    """Row-major flat values; complex arrays as interleaved re/im."""
    a = np.ascontiguousarray(a)
    if a.dtype.kind == "c":
        return enc(a.view(a.real.dtype).ravel().astype(np.float64).tolist())
    return enc(a.ravel().astype(np.float64).tolist())


def _fft_input(dt, shape, seed):
    if dt.startswith("complex"):
        re = _mat("float64", shape, seed)
        im = _mat("float64", shape, seed + 50)
        return (re + 1j * im).astype(dt)
    return _mat(dt, shape, seed)


def _fft_case(fn, x, kw, call):
    r, err = _fft_run(call)
    base = {"op": "fft", "fn": fn, "kw": kw,
            "input": {"data": _flat_interleaved(x), "dtype": str(x.dtype), "shape": list(x.shape)}}
    if err:
        return {**base, "error": err}
    return {**base, "expected": {"dtype": str(r.dtype), "shape": list(r.shape),
                                 "values": _flat_interleaved(r)}}


def fft_cases() -> list[dict]:
    cases = []
    one_d = {"fft": np.fft.fft, "ifft": np.fft.ifft, "rfft": np.fft.rfft, "irfft": np.fft.irfft}
    dtypes = ("float64", "float32", "float16", "int32", "uint8", "bool", "complex128", "complex64")
    for fn, f in one_d.items():
        # dtype x length sweep (odd, even, power of two, prime, length one)
        for i, dt in enumerate(dtypes):
            for shape in ((7,), (8,), (1,), (13,), (3, 6)):
                x = _fft_input(dt, shape, 30 + i)
                cases.append(_fft_case(fn, x, {}, lambda f=f, x=x: f(x)))
        x = _fft_input("float64", (3, 5), 40)
        c = _fft_input("complex128", (3, 5), 41)
        src = c if fn in ("ifft", "irfft") else x
        for n in (None, 1, 2, 4, 5, 9, 16, 0, -1):
            for axis in (-1, 0, 1, 2, -3):
                kw = {"n": n, "axis": axis}
                cases.append(_fft_case(fn, src, kw, lambda f=f, s=src, n=n, a=axis: f(s, n=n, axis=a)))
        for norm in ("backward", "ortho", "forward", "bogus"):
            for n in (None, 6, 11):
                kw = {"n": n, "norm": norm}
                cases.append(_fft_case(fn, src, kw, lambda f=f, s=src, n=n, m=norm: f(s, n=n, norm=m)))
        # float16 computes fct in half precision (NumPy real_dtype rule)
        h16 = _fft_input("float16", (2, 7), 46)
        for norm in ("backward", "ortho", "forward"):
            for n in (None, 5, 11):
                cases.append(_fft_case(fn, h16, {"n": n, "norm": norm},
                                       lambda f=f, s=h16, n=n, m=norm: f(s, n=n, norm=m)))
        # zero-length transform axis, zero-size other axis, 0-d
        for shape, axis, n in (((0,), -1, None), ((0,), -1, 4), ((4, 0), 0, None),
                               ((0, 3), 1, None), ((2, 1), -1, None)):
            z = np.zeros(shape)
            cases.append(_fft_case(fn, z, {"n": n, "axis": axis},
                                   lambda f=f, z=z, n=n, a=axis: f(z, n=n, axis=a)))
        s0 = np.array(2.0)
        cases.append(_fft_case(fn, s0, {}, lambda f=f, s0=s0: f(s0)))
    # rfft rejects complex input
    cx = _fft_input("complex128", (4,), 42)
    cases.append(_fft_case("rfft", cx, {}, lambda: np.fft.rfft(cx)))
    nd = {"fftn": np.fft.fftn, "ifftn": np.fft.ifftn, "fft2": np.fft.fft2, "ifft2": np.fft.ifft2}
    x3 = _fft_input("float64", (2, 3, 4), 43)
    c3 = _fft_input("complex64", (3, 4, 5), 44)
    i2 = _fft_input("int16", (4, 6), 45)
    combos = [(None, None), ([3], None), ([2, 5], None), ([-1, 6], None), (None, [0]),
              (None, [1, 0]), (None, [-1, -1]), ([4, 4], [0, 2]), ([2], [0, 1]), (None, []),
              ([3, 3, 3, 3], None), (None, [5]), ([0], [0]), ([-2], [0])]
    for fn, f in nd.items():
        for src in (x3, c3, i2):
            for s, axes in combos:
                kw = {"s": s, "axes": axes}
                if fn in ("fft2", "ifft2") and axes is None:
                    kw = {"s": s}
                    call = (lambda f=f, x=src, s=s: f(x, s=s))
                else:
                    call = (lambda f=f, x=src, s=s, a=axes: f(x, s=s, axes=a))
                cases.append(_fft_case(fn, src, kw, call))
            for norm in ("ortho", "forward"):
                cases.append(_fft_case(fn, src, {"norm": norm},
                                       lambda f=f, x=src, m=norm: f(x, norm=m)))
    for n in (1, 2, 5, 8, 0, -1, -3):
        for d in (1.0, 0.1, 2.5, 0.0):
            dummy = np.zeros(0)
            for fn, f in (("fftfreq", np.fft.fftfreq), ("rfftfreq", np.fft.rfftfreq)):
                cases.append(_fft_case(fn, dummy, {"n": n, "d": d},
                                       lambda f=f, n=n, d=d: f(n, d)))
    return cases


def random_cases() -> list[dict]:
    cases = []
    gen_calls = [
        ("random", [], {}), ("random", [5], {}), ("random", [[2, 3]], {"dtype": "float32"}),
        ("standard_normal", [7], {}), ("standard_normal", [4], {"dtype": "float32"}),
        ("normal", [3.0, 2.5, [2, 2]], {}), ("uniform", [-1.0, 4.0, 6], {}),
        ("integers", [10], {}), ("integers", [0, 10, 12], {}),
        ("integers", [-5, 5, 9], {"dtype": "int8"}), ("integers", [0, 2, 20], {"dtype": "bool"}),
        ("integers", [0, 255, 10], {"dtype": "uint8", "endpoint": True}),
        ("integers", [-1000, 1000, 10], {"dtype": "int16"}),
        ("integers", [0, 2**31, 10], {"dtype": "uint32"}),
        ("integers", [-(2**40), 2**40, 8], {}),
        ("integers", [0, 2**53, 6], {"dtype": "uint64"}),
        ("integers", [5, 5], {}), ("integers", [0, 300, 3], {"dtype": "uint8"}),
        ("choice", [10, 5], {}), ("choice", [10, 5], {"replace": False}),
        ("choice", [100000, 4], {"replace": False}),
        ("choice", [10, [2, 3]], {"replace": False, "shuffle": False}),
        ("choice", [[5, 6, 7, 8], 3], {}), ("choice", [3, 5], {"replace": False}),
        ("permutation", [10], {}), ("permutation", [[1, 2, 3, 4, 5]], {}),
        ("shuffle", [[6]], {}), ("shuffle", [[3, 4]], {}), ("shuffle", [[3, 4]], {"axis": 1}),
        ("normal", [0.0, -1.0], {}),
    ]
    for seed in (0, 42, 12345, 2**32 + 5, {"bigint": str(2**100 + 7)}, [1, 2, 3]):
        cases.append(_rand_case("gen", seed, gen_calls))
    legacy_calls = [
        ("rand", [], {}), ("rand", [3, 2], {}), ("randn", [5], {}), ("randn", [], {}),
        ("random_sample", [4], {}), ("standard_normal", [3], {}),
        ("normal", [1.0, 3.0, 4], {}), ("uniform", [2.0, 5.0, 3], {}),
        ("randint", [10], {}), ("randint", [0, 10, 12], {}),
        ("randint", [-5, 5, 9], {"dtype": "int8"}), ("randint", [0, 2, 10], {"dtype": "bool"}),
        ("randint", [0, 2**31, 10], {"dtype": "uint32"}),
        ("randint", [-(2**40), 2**40, 8], {}),
        ("choice", [10, 5], {}), ("choice", [10, 5], {"replace": False}),
        ("choice", [[5, 6, 7, 8], 3], {}),
        ("permutation", [10], {}), ("shuffle", [[3, 4]], {}), ("shuffle", [[7]], {}),
        ("randint", [5, 5], {}),
    ]
    for seed in (0, 42, 2**32 - 1, [1, 2, 3]):
        cases.append(_rand_case("legacy", seed, legacy_calls))
    return cases


def complex_cases() -> list[dict]:
    """P1 / D-033: complex conversion from/to JS and astype to/from complex."""
    cases = []
    inputs = [
        [1 + 2j, 3, -1.5 - 0j], [[1j, 2], [3, 4 - 1j]], [complex(float("nan"), 1), complex(1, float("inf"))],
        [complex(-0.0, -0.0), 0j], [True, 1j], [1e300 + 1e-300j],
    ]
    for data in inputs:
        cases.append({"op": "array", "data": enc(data), "expected": describe(np.array(data))})
        for dt in ["complex64", "complex128"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = np.array(data, dtype=dt)
            cases.append({"op": "array", "data": enc(data), "dtype": dt, "expected": describe(a)})
    for data in [[1, 2.5, -3], [True, False], [[0.5], [-0.0]]]:
        for dt in ["complex64", "complex128"]:
            cases.append({"op": "array", "data": enc(data), "dtype": dt,
                          "expected": describe(np.array(data, dtype=dt))})
    # Real -> complex and complex -> complex/real astype (complex -> real drops imag).
    src = [1.5 - 2j, -0.0 + 0j, 3 + 4j, complex(float("inf"), -1)]
    for s in ["complex64", "complex128"]:
        for dt in ["float16", "float32", "float64", "complex64", "complex128"]:
            # Integer casts of inf are platform-dependent, so only float targets.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = np.array(src, dtype=s).astype(dt)
            cases.append({"op": "astype", "data": enc(src), "src_dtype": s, "dtype": dt,
                          "expected": describe(r)})
    for s in ["int8", "uint64", "float16", "float32", "bool"]:
        for dt in ["complex64", "complex128"]:
            data = [0, 1, 5] if s != "bool" else [True, False]
            r = np.array(data, dtype=s).astype(dt)
            cases.append({"op": "astype", "data": enc(data), "src_dtype": s, "dtype": dt,
                          "expected": describe(r)})
    cases.append({"op": "zeros", "shape": [2, 2], "dtype": "complex64",
                  "expected": describe(np.zeros((2, 2), dtype="complex64"))})
    return cases


def complex_ufunc_cases() -> list[dict]:
    """P1 step 2 / D-033: complex ufuncs and real/imag/conj/angle/iscomplex."""
    inf, nan = float("inf"), float("nan")
    A = [1 + 2j, -3 + 0.5j, 0j, complex(-0.0, 0.0), 1e300 + 1e300j, complex(inf, 1),
         complex(nan, 1), 2 + 0j, -1 + 0j, 1e-310 + 1e-310j, 0.3 - 0.7j, complex(-inf, 0)]
    B = [2 - 1j, 0j, 1 + 1j, 1e300 - 1e300j, complex(3, -0.0), complex(1, inf),
         1 + 1j, -2 + 0j, 0.5 + 0j, 1e-310 + 0j, 3 + 0j, 2 + 0j]
    exact = {"add", "subtract", "negative", "conjugate", "real", "imag", "iscomplex", "isreal"}
    cases = []

    def add(op, args, r, kw=None):
        c = {"op": op, "args": args, "approx": op not in exact, "expected": describe(np.asarray(r))}
        if kw:
            c["kw"] = kw
        cases.append(c)

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for dt in ["complex64", "complex128"]:
            a, b = np.array(A, dtype=dt), np.array(B, dtype=dt)
            ea = {"data": enc(a.tolist()), "dtype": dt}
            eb = {"data": enc(b.tolist()), "dtype": dt}
            for op in ["add", "subtract", "multiply", "divide", "power"]:
                add(op, [ea, eb], getattr(np, op)(a, b))
            for op in ["negative", "conjugate", "abs", "sqrt", "exp", "log", "angle",
                       "real", "imag", "iscomplex", "isreal"]:
                add(op, [ea], getattr(np, op)(a))
            add("angle", [ea], np.angle(a, deg=True), {"deg": True})
            for p in [-3, 0, 1, 2, 3, 7, 0.5]:
                add("power", [ea, {"data": enc(p)}], np.power(a, p))
        # Mixed real/complex promotion and broadcasting.
        z = np.array([[1 + 1j], [2 - 3j]])
        for rdt in ["int8", "float32", "float64", "bool"]:
            r = np.array([1, 0, 1], dtype=rdt)
            er = {"data": enc(r.tolist()), "dtype": rdt}
            ez = {"data": enc(z.tolist()), "dtype": "complex128"}
            for op in ["add", "multiply", "divide"]:
                add(op, [ez, er], getattr(np, op)(z, r))
            add("real", [er], np.real(r))
            add("iscomplex", [er], np.iscomplex(r))
            add("angle", [er], np.angle(r))
            add("conjugate", [er], np.conjugate(r))
        c64 = np.array([1 + 2j, 3 - 4j], dtype="complex64")
        e64 = {"data": enc(c64.tolist()), "dtype": "complex64"}
        add("add", [e64, {"data": enc(1.5)}], c64 + 1.5)
        add("multiply", [e64, {"data": enc(2j)}], c64 * 2j)
        f32 = np.array([1, 2], dtype="float32")
        add("add", [{"data": enc(f32.tolist()), "dtype": "float32"}, {"data": enc(1j)}], f32 + 1j)
    return cases


def complex_reduction_cases() -> list[dict]:
    """P1 step 3 / D-034: complex sum/prod/mean/min/max/arg*/var/std."""
    inf, nan = float("inf"), float("nan")
    cases = []
    shape = (2, 3, 4)
    axis_kws = [{}, {"axis": 0}, {"axis": 1}, {"axis": -1}, {"axis": [0, 2]},
                {"axis": 1, "keepdims": True}, {"keepdims": True}]
    # Quarter/half-step values: sums are exact, so complex results compare exactly.
    data = [complex(((i * 7) % 11 - 5) * 0.5, ((i * 5) % 7 - 3) * 0.25) for i in range(24)]
    for dt in ["complex64", "complex128"]:
        for name in REDUCE_FNS:
            for kw in axis_kws:
                if name.startswith("arg") and isinstance(kw.get("axis"), list):
                    continue
                cases.append(_reduce_case(name, data, dt, shape, kw))
        for name in ("sum", "max", "argmin", "mean", "var"):
            cases.append(_reduce_case(name, data, dt, shape, {"axis": 1}, transpose=True))
    specials = [
        # Lexicographic order, ties keep the first index.
        ("max", [1 + 5j, 2 + 0j, 2 - 1j, 1 + 9j], (4,), {}),
        ("min", [1 + 5j, 2 + 0j, 2 - 1j, 1 + 9j], (4,), {}),
        ("argmax", [3 + 1j, 3 + 1j, 1 + 0j], (3,), {}),
        ("argmin", [3 + 1j, 1 + 0j, 1 + 0j], (3,), {}),
        # NaN in either part propagates; the first NaN wins.
        ("max", [1 + 1j, complex(0, nan), complex(nan, 0)], (3,), {}),
        ("min", [1 + 1j, complex(nan, 0), complex(0, nan)], (3,), {}),
        ("argmax", [1 + 1j, 5 + 0j, complex(0, nan), complex(nan, 0)], (4,), {}),
        ("argmin", [2 + 0j, complex(nan, 1), 1 + 0j, 0.5 + 0j], (2, 2), {"axis": 0}),
        ("sum", [1 + 1j, complex(nan, 0)], (2,), {}),
        ("sum", [complex(inf, 1), complex(-inf, 1)], (2,), {}),
        ("prod", [complex(inf, 0), complex(inf, 0)], (2,), {}),
        ("prod", [1 + 2j, 3 - 1j, 0.5j], (3,), {}),
        ("mean", [complex(inf, 0), 1 + 0j], (2,), {}),
        ("var", [complex(inf, 0), 1 + 0j], (2,), {}),
        ("std", [complex(0, nan), 1 + 0j], (2,), {}),
        # Signed zeros.
        ("sum", [complex(-0.0, -0.0), complex(-0.0, -0.0)], (2,), {}),
        ("max", [complex(-0.0, -0.0), complex(0.0, -0.0)], (2,), {}),
        ("min", [complex(0.0, 0.0), complex(-0.0, 0.0)], (2,), {}),
        # Empty inputs.
        ("sum", [], (0,), {}),
        ("prod", [], (0,), {}),
        ("mean", [], (0,), {}),
        ("var", [], (3, 0), {"axis": 1}),
        ("sum", [], (0, 3), {"axis": 0}),
        ("max", [], (0,), {}),
        ("max", [], (0,), {"initial": -1}),
        ("argmin", [], (0,), {}),
        ("argmax", [], (0, 3), {"axis": 1}),
        # initial, ddof, 0-d and axis errors.
        ("sum", [1 + 2j, 3 - 1j], (2,), {"initial": 10}),
        ("prod", [1 + 2j, 3 - 1j], (2,), {"initial": 0.5}),
        ("max", [1 + 1j], (1,), {"initial": 5}),
        ("min", [4 + 0j, 5 + 0j], (2,), {"initial": 2}),
        ("var", [1 + 1j, 2 + 0j, 4 - 1j], (3,), {"ddof": 1}),
        ("std", [1 + 1j, 2 + 0j, 4 - 1j], (3,), {"ddof": 3}),
        ("var", [1 + 1j, 2 + 0j, 3 + 0j, 4j], (2, 2), {"axis": 0, "ddof": 1}),
        ("sum", [5 + 1j], (), {}),
        ("argmax", [5 + 1j], (), {}),
        ("sum", [1j, 2j], (2,), {"axis": 1}),
        ("argmax", [1j, 2j], (2,), {"axis": 2}),
    ]
    for dt in ["complex64", "complex128"]:
        for name, vals, shp, kw in specials:
            cases.append(_reduce_case(name, vals, dt, shp, kw))
    # Real input reduced into a complex dtype (sum/prod/mean only, D-034).
    for name in ("sum", "prod", "mean"):
        cases.append(_reduce_case(name, [1, 2, 3, 4], "float32", (2, 2), {"axis": 0, "dtype": "complex64"}))
        cases.append(_reduce_case(name, [1, 2, 3, 4], "int64", (4,), {"dtype": "complex128"}))
    # D-021-style exactness on larger random input: pairwise (trailing axis) and
    # sequential (leading axis) summation orders, compared bit-for-bit.
    rng = np.random.default_rng(1234)
    for dt in ["complex64", "complex128"]:
        for shp, kw in [((1000,), {}), ((129,), {}), ((4, 300), {"axis": 1}), ((300, 4), {"axis": 0}),
                        ((7, 300), {})]:
            vals = (rng.standard_normal(shp) * 1e3 + 1j * rng.standard_normal(shp)).astype(dt)
            vals = vals.astype(complex).ravel().tolist()
            for name in ("sum", "mean", "var", "std"):
                c = _reduce_case(name, vals, dt, shp, kw)
                c["approx"] = False
                cases.append(c)
    return cases


def _cmat(dt, shape, seed):
    """Quarter-step complex values in [-2, 2]: products/sums are exact in complex64."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape)) if shape else 1
    re = rng.integers(-8, 9, n) * 0.25
    im = rng.integers(-8, 9, n) * 0.25
    return (re + 1j * im).astype(dt).reshape(shape)


def _noblas_matmul(a, b, dtype):
    """NumPy's non-BLAS complex matmul loop (D-035), written out: each output
    starts at +0 and adds (ar*br - ai*bi, ar*bi + ai*br) in increasing k order,
    in the loop's real precision. Strided views do not reliably avoid BLAS
    (OpenBLAS takes any element stride), and BLAS kernels differ per CPU."""
    real = np.float32 if np.dtype(dtype) == np.complex64 else np.float64
    A = np.asarray(a).astype(dtype)
    B = np.asarray(b).astype(dtype)
    A2 = A if A.ndim == 2 else A[None, :]
    B2 = B if B.ndim == 2 else B[:, None]
    m, k = A2.shape
    n = B2.shape[1]
    out = np.zeros((m, n), dtype=dtype)
    for i in range(m):
        for j in range(n):
            cr = ci = real(0.0)
            for p in range(k):
                ar, ai = real(A2[i, p].real), real(A2[i, p].imag)
                br, bi = real(B2[p, j].real), real(B2[p, j].imag)
                cr = real(cr + real(real(ar * br) - real(ai * bi)))
                ci = real(ci + real(real(ar * bi) + real(ai * br)))
            out[i, j] = complex(cr, ci)
    return out


def complex_matmul_cases() -> list[dict]:
    """P1-4d / D-035/D-036/D-037: complex matmul/dot/inner/outer (no conjugation)."""
    cases = []
    prod_fns = {"matmul": np.matmul, "dot": np.dot, "inner": np.inner, "outer": np.outer}

    def add(fn, a, b, approx=False):
        c = _lin_case(fn, [a, b], {}, lambda: prod_fns[fn](a, b))
        if "expected" in c:
            with np.errstate(all="ignore"):
                r = np.asarray(prod_fns[fn](a, b))
                c["expected"] = describe(r)
                # D-037: with non-finite input NumPy's BLAS path and its own non-BLAS
                # loop can disagree. Record the latter for the fallback backend.
                # Recorded for every exact 1-D/2-D product: BLAS kernels differ per
                # CPU (signed zeros, NaN vs inf), while the fallback is NumPy's loop.
                if (fn != "outer" and not approx and 1 <= a.ndim <= 2 and 1 <= b.ndim <= 2
                        and (fn != "dot" or b.ndim == 1 or a.shape[-1] == b.shape[0])):
                    bm = np.swapaxes(b, -1, -2) if fn == "inner" and b.ndim == 2 else b
                    nb = _noblas_matmul(a, bm, r.dtype).reshape(r.shape)
                    if enc(nb.tolist()) != enc(r.tolist()):
                        c["expected_noblas"] = enc(nb.tolist())
        c["approx"] = approx
        cases.append(c)

    # (a, b) shapes; inner swaps b's last axis to match a's.
    shapes = [((3, 4), (4, 2)), ((4,), (4,)), ((4,), (4, 3)), ((2, 3), (3,)),
              ((1, 1), (1, 1)), ((1, 5), (5, 1)), ((3, 1), (1, 4)), ((1, 1), (1, 3)),
              ((2, 2, 3), (3, 2)), ((2, 1, 2, 3), (3, 3, 2)), ((3, 2), (4, 2, 5)),
              ((2, 3, 4), (5, 4, 2)), ((0, 3), (3, 2)), ((2, 0), (0, 3)), ((0,), (0,)),
              ((3, 4), (3, 2)), ((2, 2, 3), (3, 3, 2))]
    for i, (sa, sb) in enumerate(shapes):
        for dt in ("complex64", "complex128"):
            a = _cmat(dt, sa, i)
            for fn in prod_fns:
                b = _cmat(dt, sb[:-1] + sa[-1:] if fn == "inner" else sb, i + 100)
                add(fn, a, b)
    # Mixed real/complex promotion, both operand orders.
    for j, rdt in enumerate(["bool", "int8", "uint8", "int16", "int32", "int64", "uint64",
                             "float16", "float32", "float64"]):
        for cdt in ("complex64", "complex128"):
            r = _mat(rdt, (2, 3), j)
            c = _cmat(cdt, (3, 2), j + 50)
            add("matmul", r, c)
            add("matmul", c.T.copy(), r.T.copy())
            add("dot", _mat(rdt, (3,), j), _cmat(cdt, (3,), j + 60))
            add("inner", _cmat(cdt, (2, 3), j + 70), _mat(rdt, (4, 3), j))
            add("outer", _mat(rdt, (3,), j), _cmat(cdt, (2,), j + 80))
    add("matmul", _cmat("complex64", (2, 2, 3), 1), _cmat("complex128", (3, 2), 2))
    s = np.array(2 - 1j)
    add("dot", s, _cmat("complex128", (2, 3), 3))
    add("dot", _cmat("complex64", (3,), 4), np.array(1.5, np.float32))
    # Larger random operands exercise the gemm/gemv kernels with rounding (D-018).
    rng = np.random.default_rng(4321)
    for dt in ("complex64", "complex128"):
        for sa, sb in [((17, 33), (33, 9)), ((1, 40), (40, 7)), ((6, 40), (40, 1)),
                       ((64,), (64,)), ((3, 8, 16), (16, 5))]:
            a = (rng.standard_normal(sa) + 1j * rng.standard_normal(sa)).astype(dt)
            b = (rng.standard_normal(sb) + 1j * rng.standard_normal(sb)).astype(dt)
            add("matmul", a, b, approx=True)
    # Non-finite values on the vector paths (dotu / gemv / outer), from NumPy itself.
    inf, nan = float("inf"), float("nan")
    for dt in ("complex64", "complex128"):
        v = np.array([complex(inf, 0), 1 + 1j], dt)
        w = np.array([1 + 0j, 2 - 1j], dt)
        add("dot", v, w)
        add("matmul", v.reshape(1, 2), np.array([[1 + 0j, 0j], [1j, 1 + 0j]], dt))
        add("outer", np.array([complex(inf, inf), complex(nan, 0)], dt), np.array([1j, 2 + 0j], dt))
        add("inner", np.array([[1 + 0j, complex(0, nan)]], dt), w)
    # Errors: core mismatch (ValueError) and batch broadcast mismatch.
    add("matmul", _cmat("complex128", (2, 3), 1), _cmat("complex128", (4, 2), 2))
    add("matmul", _cmat("complex64", (2, 2, 3), 1), _cmat("complex64", (3, 3, 2), 2))
    add("dot", _cmat("complex128", (3,), 1), _cmat("complex128", (4,), 2))
    add("inner", _cmat("complex128", (2, 3), 1), _cmat("complex128", (2, 4), 2))
    add("matmul", np.array(1j), _cmat("complex128", (2,), 3))
    return cases


def _hermitian(x):
    return (x + np.conj(np.swapaxes(x, -1, -2))) / 2


def complex_linalg_cases() -> list[dict]:
    """P1-5e.4 / D-038–D-044: complex decompositions. Vectors are checked by
    reconstruction in the runner (sign/phase are backend dependent)."""
    cases = []
    L = np.linalg

    def add(fn, inputs, call, kw=None):
        cases.append(_lin_case(fn, inputs, kw or {}, call))

    for dt in ("complex64", "complex128"):
        sq = [_cmat(dt, (3, 3), 1), _cmat(dt, (2, 4, 4), 2), _cmat(dt, (1, 1), 3),
              np.zeros((0, 0), dt), (np.eye(3) * (1 + 1j)).astype(dt)]
        for s in sq:
            add("det", [s], lambda s=s: L.det(s))
            add("inv", [s], lambda s=s: L.inv(s))
            rhs = _cmat(dt, s.shape[:-1] + (2,), 4)
            add("solve", [s, rhs], lambda s=s, r=rhs: L.solve(s, r))
            add("eig", [s], lambda s=s: L.eig(s))
            add("eigvals", [s], lambda s=s: L.eigvals(s))
            h = _hermitian(s)
            add("eigh", [h], lambda h=h: L.eigh(h))
            add("eigvalsh", [h], lambda h=h: L.eigvalsh(h))
        rects = [_cmat(dt, (4, 3), 5), _cmat(dt, (3, 5), 6), _cmat(dt, (2, 3, 2), 7),
                 np.array([[1, 2j], [2, 4j], [3, 6j]], dt), np.zeros((0, 3), dt)]
        for r in rects:
            for full in (True, False):
                add("svd", [r], lambda r=r, f=full: L.svd(r, full_matrices=f),
                    {"fullMatrices": full})
            add("svd", [r], lambda r=r: L.svd(r, compute_uv=False), {"computeUV": False})
            for mode in ("reduced", "complete", "r"):
                add("qr", [r], lambda r=r, m=mode: L.qr(r, mode=m), {"mode": mode})
            if r.ndim == 2:
                b = _cmat(dt, (r.shape[0],), 8)
                add("lstsq", [r, b], lambda r=r, b=b: L.lstsq(r, b))
                b2 = _cmat(dt, (r.shape[0], 2), 9)
                add("lstsq", [r, b2], lambda r=r, b=b2: L.lstsq(r, b))
        inf = float("inf")
        v, m2, m3 = _cmat(dt, (5,), 10), _cmat(dt, (3, 4), 11), _cmat(dt, (2, 3, 4), 12)
        for o in (None, 1, 2, inf, -inf, 0, 3, -1, 0.5):
            add("norm", [v], lambda o=o: L.norm(v, ord=o), {"ord": o})
        for o in (None, "fro", "nuc", 1, -1, 2, -2, inf, -inf):
            add("norm", [m2], lambda o=o: L.norm(m2, ord=o), {"ord": o})
        add("norm", [m3], lambda: L.norm(m3, axis=(1, 2)), {"axis": [1, 2]})
        add("norm", [m3], lambda: L.norm(m3, axis=-1, keepdims=True), {"axis": -1, "keepdims": True})
    # Mixed real/complex promotion.
    for rdt in ("float32", "float64", "int32"):
        for cdt in ("complex64", "complex128"):
            a, b = _mat(rdt, (3, 3), 13), _cmat(cdt, (3, 2), 14)
            add("solve", [a, b], lambda a=a, b=b: L.solve(a, b))
            add("lstsq", [a, b], lambda a=a, b=b: L.lstsq(a, b))
            ca, rb = _cmat(cdt, (4, 2), 15), _mat(rdt, (4,), 16)
            add("lstsq", [ca, rb], lambda a=ca, b=rb: L.lstsq(a, b))
    # Errors.
    c = _cmat("complex128", (2, 3), 17)
    for fn in ("det", "inv", "eig", "eigh"):
        add(fn, [c], lambda f=getattr(L, fn): f(c))
    sing = np.array([[1 + 1j, 2 + 2j], [1 + 1j, 2 + 2j]])
    add("inv", [sing], lambda: L.inv(sing))
    nanm = np.array([[complex(np.nan, 0), 1], [0, 1j]])
    add("eig", [nanm], lambda: L.eig(nanm))
    add("norm", [v], lambda: L.norm(v, ord="fro"), {"ord": "fro"})
    return cases


def _view(x, v):
    """The same strided view the JS runner builds (D-008 views, not copies)."""
    if v == "T":
        return x.T
    if v == "rev":
        return x[..., ::-1]
    if v == "step":
        return x[..., ::2]
    return x


def complex_fft_cases() -> list[dict]:
    """P1-5e.5: FFT on np.Complex input end to end. Inputs are nested lists of
    complex scalars (np.array(list) in JS, not fromTypedArray), optionally
    turned into a strided view. Results are compared through toArray()."""
    cases = []

    def add(fn, data, dt, kw, call, view=None, plain=False):
        x = np.array(data, dtype=dt) if dt else np.array(data)
        r, err = _fft_run(lambda: call(_view(x, view)))
        c = {"op": "complex_fft", "fn": fn, "data": enc(data), "kw": kw}
        if dt:
            c["dtype"] = dt
        if view:
            c["view"] = view
        if plain:
            c["plain"] = True
        if err:
            c["error"] = err
        else:
            c["expected"] = {"dtype": str(r.dtype), "shape": list(r.shape), "values": enc(r.tolist())}
        cases.append(c)

    F = np.fft
    for dt in (None, "complex64", "complex128"):
        v1 = _cmat("complex128", (7,), 60).tolist()
        v8 = _cmat("complex128", (8,), 61).tolist()
        m = _cmat("complex128", (3, 4), 62).tolist()
        t = _cmat("complex128", (2, 3, 4), 63).tolist()
        for fn, f in (("fft", F.fft), ("ifft", F.ifft)):
            for data in (v1, v8, [1 + 1j], m):
                add(fn, data, dt, {}, lambda x, f=f: f(x))
            for n in (4, 10):
                add(fn, v1, dt, {"n": n}, lambda x, f=f, n=n: f(x, n=n))
            for norm in ("ortho", "forward"):
                add(fn, m, dt, {"axis": 0, "norm": norm}, lambda x, f=f, m_=norm: f(x, axis=0, norm=m_))
            for view in ("T", "rev", "step"):
                add(fn, m, dt, {}, lambda x, f=f: f(x), view=view)
        add("irfft", v1, dt, {}, lambda x: F.irfft(x))
        add("irfft", m, dt, {"n": 6, "axis": 0}, lambda x: F.irfft(x, n=6, axis=0))
        add("irfft", m, dt, {}, lambda x: F.irfft(x), view="T")
        for fn, f in (("fft2", F.fft2), ("ifft2", F.ifft2)):
            add(fn, m, dt, {}, lambda x, f=f: f(x))
            add(fn, t, dt, {}, lambda x, f=f: f(x), view="rev")
        for fn, f in (("fftn", F.fftn), ("ifftn", F.ifftn)):
            add(fn, t, dt, {}, lambda x, f=f: f(x))
            add(fn, t, dt, {"s": [3, 2], "axes": [0, 2]}, lambda x, f=f: f(x, s=[3, 2], axes=[0, 2]))
            add(fn, m, dt, {"norm": "ortho"}, lambda x, f=f: f(x, norm="ortho"), view="T")
        add("rfft", v1, dt, {}, lambda x: F.rfft(x))  # TypeError (D-020)
    # Mixed real/complex/bool lists infer complex128 (D-004/D-033); plain {re, im} objects.
    for data in ([1, 1j, 2, 3], [True, 1j], [[1j, 2], [3, 4 - 1j]], [0.5, -2 + 0.25j, 3, 1j, -1]):
        add("fft", data, None, {}, lambda x: F.fft(x))
        add("fft", data, None, {}, lambda x: F.fft(x), plain=True)
    add("ifftn", [[1j, 2], [3, 4 - 1j]], None, {}, lambda x: F.ifftn(x), plain=True)
    # Non-finite values propagate as in pocketfft.
    nonfinite = [complex(float("nan"), 0), 1 + 0j, 2j, 3 + 0j]
    add("fft", nonfinite, None, {}, lambda x: F.fft(x))
    # 0-d complex scalar and empty complex input are rejected.
    add("fft", 1 + 2j, None, {}, lambda x: F.fft(x))
    add("fft", [], "complex128", {}, lambda x: F.fft(x))
    return cases


FLOAT_DTYPES = ["float16", "float32", "float64"]
COMPLEX_DTYPES = ["complex64", "complex128"]
TRIG_DTYPES = FLOAT_DTYPES + COMPLEX_DTYPES
INT_DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]


def _math_float_vals(dt: str) -> list:
    """Representative float values safe for trig/exp/log families."""
    if dt in COMPLEX_DTYPES:
        # Avoid real-axis inputs (im=0) to sidestep signed-zero divergences in cos/sin.
        # Non-finite complex edge cases are covered by complex_ufunc_cases.
        return [0.5 + 0.3j, -1.2 + 0.7j, 0.1 + 0.1j, 2.0 - 1.5j, 1.0 + 0.2j, -0.5 + 0.5j]
    return [0.0, 0.5, -0.5, 1.0, -1.0, float("nan"), float("inf"), -float("inf"), -0.0]


def _math_float_vals_small(dt: str) -> list:
    """Small-domain float values for arcsin/arccos/arctanh etc."""
    if dt in COMPLEX_DTYPES:
        return [0.5 + 0.3j, -0.5 + 0.1j, 0.1 + 0.1j, 0.9 - 0.4j, 0.2 + 0.2j]
    return [0.0, 0.5, -0.5, 0.9, -0.9, float("nan"), -0.0]


def _math_real_all(dt: str) -> list:
    """Values usable for rounding ufuncs across all real dtypes."""
    if dt == "bool":
        return [True, False, True, True, False, True]
    kind = np.dtype(dt).kind
    if kind == "u":
        return [0, 1, 2, 5, 10, 3]
    if kind == "i":
        return [-3, 0, 2, 5, -7, 1]
    return [-1.7, -0.5, 0.0, 0.5, 1.5, float("nan")]


def math_ufunc_cases() -> list[dict]:
    """P4 math ufuncs differential test cases (V-P4).

    Covers: trig, exp/log, rounding, arithmetic, float-bits, special.
    Group name: "math_ufuncs" (separate from the existing "ufuncs" group).
    """
    cases = []

    def add(op, vals, dt, approx=True, expected=None, error=None, **extra):
        a = np.array(vals, dtype=dt)
        if error is None and expected is None:
            r, err = _run(getattr(np, op), a)
            if err:
                cases.append({"op": op, "a_data": enc(vals), "a_dtype": dt,
                               "approx": approx, "error": err, **extra})
            else:
                cases.append({"op": op, "a_data": enc(vals), "a_dtype": dt,
                               "approx": approx, "expected": describe(r), **extra})
        elif error is not None:
            cases.append({"op": op, "a_data": enc(vals), "a_dtype": dt,
                           "approx": approx, "error": error, **extra})
        else:
            cases.append({"op": op, "a_data": enc(vals), "a_dtype": dt,
                           "approx": approx, "expected": expected, **extra})

    def add2(op, av, ad, bv, bd, approx=True, **extra):
        a = np.array(av, dtype=ad)
        b = np.array(bv, dtype=bd)
        r, err = _run(getattr(np, op), a, b)
        base = {"op": op, "a_data": enc(av), "a_dtype": ad,
                "b_data": enc(bv), "b_dtype": bd, "approx": approx}
        if err:
            cases.append({**base, "error": err, **extra})
        else:
            cases.append({**base, "expected": describe(r), **extra})

    # ---- trig unary (float + complex) ----
    for dt in TRIG_DTYPES:
        fv = _math_float_vals(dt)
        sv = _math_float_vals_small(dt)
        for op in ("sin", "cos", "tan"):
            add(op, fv, dt)
        for op in ("arcsin", "arccos"):
            add(op, sv, dt)
        add("arctan", fv, dt)
        for op in ("sinh", "cosh", "tanh"):
            add(op, fv, dt)
        add("arcsinh", fv, dt)
        add("arccosh", fv, dt)  # <1 -> NaN for float; for complex it's fine
        add("arctanh", sv, dt)

    # ---- trig binary ----
    for dt in FLOAT_DTYPES:
        fv = [0.0, 1.0, -1.0, 0.5, float("nan"), float("inf")]
        add2("arctan2", fv, dt, [1.0, 0.0, 1.0, -1.0, 0.5, 1.0], dt)
        add2("hypot", fv, dt, [1.0, 2.0, -3.0, 4.0, 5.0, 6.0], dt)

    # ---- angle conversion (float only) ----
    for dt in FLOAT_DTYPES:
        av = [0.0, 1.5708, 3.14159, -1.5708, float("nan")]
        add("deg2rad", av, dt)
        add("rad2deg", av, dt)
        add("radians", av, dt)
        add("degrees", av, dt)

    # ---- exp/log family ----
    for dt in TRIG_DTYPES:
        fv = _math_float_vals(dt)
        add("exp2", fv, dt)
        add("expm1", fv, dt)
        add("log2", fv, dt)
        add("log10", fv, dt)
        add("log1p", fv, dt)

    for dt in FLOAT_DTYPES:
        fv = [0.0, 0.5, -0.5, 1.0, 2.0, float("nan"), float("inf")]
        add2("logaddexp", fv, dt, [1.0, 0.5, -0.5, 2.0, 1.0, 1.0, 0.0], dt)
        add2("logaddexp2", fv, dt, [1.0, 0.5, -0.5, 2.0, 1.0, 1.0, 0.0], dt)
        add("cbrt", fv + [-float("inf")], dt)
        add("square", fv, dt)
        add("reciprocal", [1.0, 2.0, -0.5, 0.0, float("nan"), float("inf")], dt)

    # ---- rounding (all real dtypes) ----
    for dt in REAL_DTYPES:
        rv = _math_real_all(dt)
        for op in ("floor", "ceil", "trunc", "rint"):
            add(op, rv, dt, approx=False)
        add("positive", rv, dt, approx=False)

    # ---- arithmetic: fmod, floatPower, sign, heaviside, maximum, minimum, fmax, fmin ----
    for dt in FLOAT_DTYPES:
        fv = [1.5, -2.5, 0.0, 3.0, float("nan"), float("inf")]
        bv = [0.5, 1.5, 1.0, -1.0, 2.0, 3.0]
        add2("fmod", fv, dt, bv, dt)
        add2("float_power", [1.0, 2.0, 0.5, 4.0, float("nan")], dt,
             [2.0, 3.0, 0.5, -1.0, 1.0], dt)
        add("sign", fv + [-0.0], dt, approx=False)
        add("fabs", fv, dt, approx=False)
        add2("maximum", fv, dt, bv, dt)
        add2("minimum", fv, dt, bv, dt)
        add2("fmax", fv, dt, bv, dt)
        add2("fmin", fv, dt, bv, dt)

        # heaviside: second arg is same-dtype scalar value
        hv = [-1.5, 0.0, 1.5, float("nan"), float("inf"), -0.0]
        for h2 in [0.0, 0.5, 1.0, float("nan")]:
            add2("heaviside", hv, dt, [h2] * len(hv), dt)

    for dt in REAL_DTYPES:
        sv = _math_real_all(dt)
        add("sign", sv, dt, approx=False)
        add("fabs", sv if np.dtype(dt).kind == "f" else [abs(x) if isinstance(x, (int, bool)) else x for x in sv], dt, approx=False)

    # ---- gcd / lcm (integer only) ----
    for dt in INT_DTYPES:
        kind = np.dtype(dt).kind
        if kind == "u":
            av = [0, 6, 12, 15, 100]
            bv = [0, 4, 8, 10, 60]
        else:
            av = [-6, 0, 12, 15, -100]
            bv = [4, 0, -8, 10, 60]
        # uint8/int8 gcd/lcm: clamp to safe range
        if dt in ("int8", "uint8"):
            av = [0, 6, 12, 4, 8]
            bv = [6, 0, 4, 4, 2]
        add2("gcd", av, dt, bv, dt, approx=False)
        add2("lcm", av, dt, bv, dt, approx=False)

    # ---- float-bits: signbit, spacing, nextafter, copysign, ldexp ----
    for dt in FLOAT_DTYPES:
        fv = [0.0, 1.0, -1.0, float("nan"), float("inf"), -float("inf"), -0.0]
        add("signbit", fv, dt, approx=False)
        add("spacing", [0.5, 1.0, 2.0, 4.0, float("inf")], dt, approx=False)
        add2("nextafter", fv, dt, [1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0], dt)
        add2("copysign", fv, dt, [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0], dt, approx=False)

    # ldexp: float x int32
    for ft in FLOAT_DTYPES:
        fv = [1.0, 0.5, -2.0, 0.0, float("nan"), float("inf")]
        add2("ldexp", fv, ft, [0, 1, -1, 5, 0, 0], "int32", approx=False)

    # ---- multi-output ufuncs: divmod, modf, frexp ----
    # These use a special case format with op="divmod", "modf", "frexp"
    # and expected as a list of two describe() dicts.
    for dt in FLOAT_DTYPES:
        fv_dm = [5.0, -7.0, 2.5, 0.0, float("nan"), float("inf")]
        bv_dm = [2.0, 3.0, 1.5, 1.0, 1.0, 2.0]
        a_dm = np.array(fv_dm, dtype=dt)
        b_dm = np.array(bv_dm, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            q, r = np.divmod(a_dm, b_dm)
        cases.append({
            "op": "divmod", "a_data": enc(fv_dm), "a_dtype": dt,
            "b_data": enc(bv_dm), "b_dtype": dt, "approx": True,
            "expected": [describe(q), describe(r)],
        })

        fv_m = [3.7, -2.3, 0.0, float("nan"), float("inf"), -0.0]
        a_m = np.array(fv_m, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            frac, intg = np.modf(a_m)
        cases.append({
            "op": "modf", "a_data": enc(fv_m), "a_dtype": dt, "approx": True,
            "expected": [describe(frac), describe(intg)],
        })

        fv_f = [1.5, -0.75, 0.0, float("nan"), float("inf"), -float("inf")]
        a_f = np.array(fv_f, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            mant, exp_ = np.frexp(a_f)
        cases.append({
            "op": "frexp", "a_data": enc(fv_f), "a_dtype": dt, "approx": True,
            "expected": [describe(mant), describe(exp_)],
        })

    # ---- clip (3-arg function) ----
    for dt in FLOAT_DTYPES + ["int32", "int64"]:
        cv = [-2.0, -0.5, 0.5, 1.5, float("nan")] if dt.startswith("f") else [-3, -1, 0, 1, 2]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            ac = np.array(cv, dtype=dt)
            r = np.clip(ac, -1 if dt.startswith("f") else -1, 1)
        cases.append({
            "op": "clip", "a_data": enc(cv), "a_dtype": dt,
            "clip_min": -1.0, "clip_max": 1.0, "approx": False,
            "expected": describe(r),
        })

    # ---- nanToNum ----
    nan, inf = float("nan"), float("inf")
    for dt in FLOAT_DTYPES:
        fv = [nan, inf, -inf, 0.0, 1.0, -0.0]
        a_n = np.array(fv, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = np.nan_to_num(a_n)
        cases.append({
            "op": "nanToNum", "a_data": enc(fv), "a_dtype": dt, "approx": False,
            "expected": describe(r),
        })

    # ---- sinc ----
    for dt in ["float32", "float64"]:  # float16 has too much accumulated error for sinc
        sv = [0.0, 0.5, 1.0, -1.0, 2.0, float("nan")]
        add("sinc", sv, dt)

    # ---- i0 ----
    for dt in FLOAT_DTYPES:
        iv = [0.0, 1.0, 2.0, -2.0, 3.5, float("nan")]
        add("i0", iv, dt)

    # ---- unwrap ----
    for dt in FLOAT_DTYPES:
        uv = [0.0, 0.5, 1.0, 5.5, 6.0, 6.5, 1.0]  # jumps > pi will be unwrapped
        a_u = np.array(uv, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = np.unwrap(a_u)
        cases.append({
            "op": "unwrap", "a_data": enc(uv), "a_dtype": dt, "approx": True,
            "expected": describe(r),
        })

    # ---- realIfClose ----
    for dt in COMPLEX_DTYPES:
        rv = [1 + 0j, 2 + 1e-10j, 3 + 1j, 0 + 0j]
        a_r = np.array(rv, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = np.real_if_close(a_r)
        cases.append({
            "op": "realIfClose", "a_data": enc(rv), "a_dtype": dt, "approx": True,
            "expected": describe(r),
        })

    return cases


def logic_cases() -> list[dict]:
    """P5 logic/bitwise/comparison differential test cases (V-P5).

    Covers: comparisons, logic ufuncs, all/any, classify, bitwise,
    isclose/allclose/arrayEqual/arrayEquiv, packbits/unpackbits.
    Group name: "logic_cases".
    """
    cases = []
    nan = float("nan")
    inf = float("inf")

    INT_DTYPES = ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
    FLOAT_DTYPES = ["float16", "float32", "float64"]

    def _cmp_vals(dt: str, which: int) -> list:
        """Values for comparison tests."""
        kind = np.dtype(dt).kind
        if dt == "bool":
            return [True, False, True, False] if which == 0 else [False, True, True, False]
        if kind == "u":
            return [0, 1, 10, 255 if dt == "uint8" else 1000] if which == 0 else [1, 0, 10, 200 if dt == "uint8" else 999]
        if kind == "i":
            return [0, -1, 5, 127 if dt == "int8" else 100] if which == 0 else [0, 1, 5, -128 if dt == "int8" else -100]
        # float
        return [0.0, -1.0, nan, inf] if which == 0 else [0.0, 1.0, nan, -inf]

    # ---- 1. Comparison ufuncs (no complex) ----
    CMP_OPS = [
        ("equal",        np.equal),
        ("notEqual",     np.not_equal),
        ("less",         np.less),
        ("lessEqual",    np.less_equal),
        ("greater",      np.greater),
        ("greaterEqual", np.greater_equal),
    ]
    for ts_name, fn in CMP_OPS:
        for dt in REAL_DTYPES:
            av = _cmp_vals(dt, 0)
            bv = _cmp_vals(dt, 1)
            a = np.array(av, dtype=dt)
            b = np.array(bv, dtype=dt)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                r = fn(a, b)
            cases.append({
                "op": ts_name,
                "a_data": enc(av), "a_dtype": dt,
                "b_data": enc(bv), "b_dtype": dt,
                "approx": False,
                "expected": describe(r),
            })

    # ---- 2. Logic ufuncs (all real dtypes) ----
    LOGIC_BINARY = [
        ("logicalAnd", np.logical_and),
        ("logicalOr",  np.logical_or),
        ("logicalXor", np.logical_xor),
    ]
    LOGIC_UNARY = [
        ("logicalNot", np.logical_not),
    ]

    def _logic_vals(dt: str, which: int) -> list:
        kind = np.dtype(dt).kind
        if dt == "bool":
            return [True, False, True, False, True, True] if which == 0 else [False, False, True, True, False, True]
        if kind == "u":
            return [0, 1, 0, 5, 10, 0] if which == 0 else [1, 0, 0, 3, 0, 7]
        if kind == "i":
            return [0, -1, 0, 5, -3, 0] if which == 0 else [1, 0, 0, -2, 0, 4]
        # float
        return [0.0, 1.5, 0.0, nan, -1.0, inf] if which == 0 else [1.0, 0.0, 0.0, 2.0, 0.0, nan]

    for ts_name, fn in LOGIC_BINARY:
        for dt in REAL_DTYPES:
            av = _logic_vals(dt, 0)
            bv = _logic_vals(dt, 1)
            a = np.array(av, dtype=dt)
            b = np.array(bv, dtype=dt)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                r = fn(a, b)
            cases.append({
                "op": ts_name,
                "a_data": enc(av), "a_dtype": dt,
                "b_data": enc(bv), "b_dtype": dt,
                "approx": False,
                "expected": describe(r),
            })

    for ts_name, fn in LOGIC_UNARY:
        for dt in REAL_DTYPES:
            av = _logic_vals(dt, 0)
            a = np.array(av, dtype=dt)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                r = fn(a)
            cases.append({
                "op": ts_name,
                "a_data": enc(av), "a_dtype": dt,
                "approx": False,
                "expected": describe(r),
            })

    # ---- 3. all / any ----
    ALL_ANY_DATA = [
        ([True, False, True, False, True, True], (2, 3), "bool"),
        ([1, 0, 1, 0, 1, 1], (2, 3), "int32"),
        ([1, 2, 3, 4, 5, 6], (2, 3), "int32"),
        ([0, 0, 0, 0, 0, 0], (2, 3), "int32"),
        ([1.0, 0.0, nan, 1.0], (2, 2), "float64"),
    ]
    for data, shape, dt in ALL_ANY_DATA:
        a = np.array(data, dtype=dt).reshape(shape)
        for fn_name, fn in [("all", np.all), ("any", np.any)]:
            for kw_axis in [None, 0, 1]:
                kw = {"axis": kw_axis}
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    if kw_axis is None:
                        r = fn(a)
                    else:
                        r = fn(a, axis=kw_axis)
                r_arr = np.asarray(r)
                cases.append({
                    "op": fn_name,
                    "a_data": enc(data), "a_dtype": dt, "a_shape": list(shape),
                    "kw": kw,
                    "approx": False,
                    "expected": describe(r_arr),
                })
        for fn_name, fn in [("all", np.all), ("any", np.any)]:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                r = fn(a, axis=0, keepdims=True)
            r_arr = np.asarray(r)
            cases.append({
                "op": fn_name,
                "a_data": enc(data), "a_dtype": dt, "a_shape": list(shape),
                "kw": {"axis": 0, "keepdims": True},
                "approx": False,
                "expected": describe(r_arr),
            })

    # ---- 4. Classify ufuncs ----
    CLASSIFY_FLOAT_VALS = [0.0, 1.5, -1.5, nan, inf, -inf, -0.0]
    CLASSIFY_UNARY = [
        ("isnan",    np.isnan),
        ("isinf",    np.isinf),
        ("isfinite", np.isfinite),
        ("isposinf", np.isposinf),
        ("isneginf", np.isneginf),
    ]
    for ts_name, fn in CLASSIFY_UNARY:
        for dt in FLOAT_DTYPES:
            av = CLASSIFY_FLOAT_VALS
            a = np.array(av, dtype=dt)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                r = fn(a)
            cases.append({
                "op": ts_name,
                "a_data": enc(av), "a_dtype": dt,
                "approx": False,
                "expected": describe(r),
            })

    COMPLEX_CLASSIFY = [("isnan", np.isnan), ("isinf", np.isinf), ("isfinite", np.isfinite)]
    COMPLEX_DTYPES = ["complex64", "complex128"]
    cx_vals = [complex(0, 0), complex(1, nan), complex(inf, 0), complex(nan, inf),
               complex(-inf, 1), complex(1, 1)]
    for ts_name, fn in COMPLEX_CLASSIFY:
        for dt in COMPLEX_DTYPES:
            a = np.array(cx_vals, dtype=dt)
            av_enc = enc(a.tolist())
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                r = fn(a)
            cases.append({
                "op": ts_name,
                "a_data": av_enc, "a_dtype": dt,
                "approx": False,
                "expected": describe(r),
            })

    # ---- 5. Bitwise ufuncs (integer dtypes only) ----
    def _bit_vals(dt: str, which: int) -> list:
        kind = np.dtype(dt).kind
        if kind == "u":
            if which == 0:
                return [0, 1, 0b1010, 0b1111, 100, 255 if dt == "uint8" else 1000]
            else:
                return [1, 0, 0b0110, 0b1001, 200, 127 if dt == "uint8" else 500]
        else:
            if which == 0:
                return [0, 1, -1, 0b1010, -0b1010, 100]
            else:
                return [1, 0, -1, 0b0110, 0b0011, -100]

    def _shift_vals(dt: str, which: int) -> list:
        itemsize = np.dtype(dt).itemsize
        max_shift = itemsize * 8 - 1
        if which == 0:
            kind = np.dtype(dt).kind
            if kind == "u":
                return [0, 1, 0b0010, 0b1100, 0xFF if dt == "uint8" else 0x0F00, 0b11111111 if dt in ("uint8", "int8") else 0x7FFF]
            else:
                return [0, 1, -1, 0b0110, -0b0110, 0x7F if dt == "int8" else 0x7FFF]
        else:
            return [0, 1, 2, 3, min(4, max_shift - 1), min(7, max_shift - 1)]

    BITWISE_BINARY = [
        ("bitwiseAnd", np.bitwise_and),
        ("bitwiseOr",  np.bitwise_or),
        ("bitwiseXor", np.bitwise_xor),
        ("leftShift",  np.left_shift),
        ("rightShift", np.right_shift),
    ]
    BITWISE_UNARY = [
        ("invert",       np.invert),
        ("bitwiseCount", np.bitwise_count),
    ]
    for ts_name, fn in BITWISE_BINARY:
        for dt in INT_DTYPES:
            if ts_name in ("leftShift", "rightShift"):
                av = _shift_vals(dt, 0)
                bv = _shift_vals(dt, 1)
                if np.dtype(dt).kind == "u":
                    av = [0, 1, 0b1010, 0b11001100, 255 if dt == "uint8" else 0xFFFF, 0b10101010]
            else:
                av = _bit_vals(dt, 0)
                bv = _bit_vals(dt, 1)
            a = np.array(av, dtype=dt)
            b = np.array(bv, dtype=dt)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                try:
                    r = fn(a, b)
                except (TypeError, ValueError, OverflowError):
                    continue
            cases.append({
                "op": ts_name,
                "a_data": enc(av), "a_dtype": dt,
                "b_data": enc(bv), "b_dtype": dt,
                "approx": False,
                "expected": describe(r),
            })

    for ts_name, fn in BITWISE_UNARY:
        for dt in INT_DTYPES:
            av = _bit_vals(dt, 0)
            a = np.array(av, dtype=dt)
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                try:
                    r = fn(a)
                except (TypeError, ValueError):
                    continue
            cases.append({
                "op": ts_name,
                "a_data": enc(av), "a_dtype": dt,
                "approx": False,
                "expected": describe(r),
            })

    # ---- 6. isclose / allclose / arrayEqual / arrayEquiv ----
    ISCLOSE_PAIRS = [
        ([1.0, 2.0, 3.0], [1.0, 2.0 + 1e-9, 3.0 + 1e-7], "float64", {}),
        ([1.0, nan, inf], [1.0, nan, inf], "float64", {}),
        ([1.0, nan, inf], [1.0, nan, inf], "float64", {"equalNan": True}),
        ([1.0, 2.0, 3.0], [1.0, 2.001, 3.001], "float32", {}),
        ([0.0, 1.0, -1.0], [0.0, 1.0, -1.0], "float64", {}),
    ]
    for av, bv, dt, kw in ISCLOSE_PAIRS:
        a = np.array(av, dtype=dt)
        b = np.array(bv, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = np.isclose(a, b, equal_nan=bool(kw.get("equalNan")))
        cases.append({
            "op": "isclose",
            "a_data": enc(av), "a_dtype": dt,
            "b_data": enc(bv), "b_dtype": dt,
            "kw": kw,
            "approx": False,
            "expected": describe(r),
        })

    ALLCLOSE_PAIRS = [
        ([1.0, 2.0, 3.0], [1.0, 2.0 + 1e-9, 3.0 + 1e-9], "float64", True),
        ([1.0, 2.0, 3.0], [1.0, 3.0, 3.0], "float64", False),
        ([nan, 1.0], [nan, 1.0], "float64", False),
        ([0.0, 1.0], [0.0, 1.0], "float64", True),
    ]
    for av, bv, dt, expected in ALLCLOSE_PAIRS:
        a = np.array(av, dtype=dt)
        b = np.array(bv, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = bool(np.allclose(a, b))
        cases.append({
            "op": "allclose",
            "a_data": enc(av), "a_dtype": dt,
            "b_data": enc(bv), "b_dtype": dt,
            "approx": False,
            "expected": r,
        })

    ARRAY_EQUAL_PAIRS = [
        ([1, 2, 3], [1, 2, 3], "int32", True),
        ([1, 2, 3], [1, 2, 4], "int32", False),
        ([1, 2], [1, 2, 3], "int32", False),
        ([float("nan"), 1.0], [float("nan"), 1.0], "float64", False),
    ]
    for av, bv, dt, expected in ARRAY_EQUAL_PAIRS:
        a = np.array(av, dtype=dt)
        b = np.array(bv, dtype=dt)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = bool(np.array_equal(a, b))
        cases.append({
            "op": "arrayEqual",
            "a_data": enc(av), "a_dtype": dt,
            "b_data": enc(bv), "b_dtype": dt,
            "approx": False,
            "expected": r,
        })

    ARRAY_EQUIV_PAIRS = [
        ([1, 2, 3], [1, 2, 3], "int32", "int32"),
        ([1, 2, 3], [[1, 2, 3], [1, 2, 3]], "int32", "int32"),
        ([1, 2, 3], [1, 2, 4], "int32", "int32"),
    ]
    for av, bv, da, db in ARRAY_EQUIV_PAIRS:
        a = np.array(av, dtype=da)
        b = np.array(bv, dtype=db)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = bool(np.array_equiv(a, b))
        cases.append({
            "op": "arrayEquiv",
            "a_data": enc(av), "a_dtype": da,
            "b_data": enc(bv), "b_dtype": db,
            "approx": False,
            "expected": r,
        })

    # ---- 7. packbits / unpackbits ----
    PACKBITS_CASES = [
        ([1, 0, 1, 1, 0, 1, 0, 0], "uint8", {}),
        ([1, 0, 1, 1, 0, 1, 0, 0], "uint8", {"bitorder": "little"}),
        ([[1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 1, 1]], "uint8", {}),
        ([[1, 0, 1, 1, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0, 1, 1]], "uint8", {"axis": 1}),
        ([True, False, True, True, False, True, False, False], "bool", {}),
        ([1, 0, 1, 0], "uint8", {}),
    ]
    for data, a_dtype, kw in PACKBITS_CASES:
        a = np.array(data, dtype=a_dtype)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            if "bitorder" in kw:
                r = np.packbits(a, bitorder=kw["bitorder"])
            elif "axis" in kw:
                r = np.packbits(a, axis=kw["axis"])
            else:
                r = np.packbits(a)
        cases.append({
            "op": "packbits",
            "a_data": enc(data), "a_dtype": a_dtype,
            "kw": kw,
            "approx": False,
            "expected": describe(r),
        })

    UNPACKBITS_CASES = [
        ([0b10110100], {}),
        ([0b10110100], {"bitorder": "little"}),
        ([0xFF, 0x00, 0xAA], {}),
        ([[0b10110100, 0b01001011], [0xFF, 0x00]], {"axis": 1}),
        ([0xFF, 0x00, 0xAA], {"count": 16}),
        ([0xFF], {"count": 4}),
    ]
    for data, kw in UNPACKBITS_CASES:
        a = np.array(data, dtype="uint8")
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            r = np.unpackbits(a, **{k: v for k, v in kw.items() if k in ("axis", "count", "bitorder")})
        cases.append({
            "op": "unpackbits",
            "a_data": enc(data), "a_dtype": "uint8",
            "kw": kw,
            "approx": False,
            "expected": describe(r),
        })

    return cases


def manip_cases() -> list[dict]:
    import numpy as np
    cases = []

    # ---- concatenate ----
    a1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    a2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)

    r = np.concatenate([a1, a2], axis=0)
    cases.append({"op": "concatenate", "arrays": [enc(a1.tolist()), enc(a2.tolist())],
                  "dtype": "float64", "axis": 0, "expected": describe(r)})

    r = np.concatenate([a1, a2], axis=1)
    cases.append({"op": "concatenate", "arrays": [enc(a1.tolist()), enc(a2.tolist())],
                  "dtype": "float64", "axis": 1, "expected": describe(r)})

    r = np.concatenate([a1, a2], axis=None)
    cases.append({"op": "concatenate", "arrays": [enc(a1.tolist()), enc(a2.tolist())],
                  "dtype": "float64", "axis": None, "expected": describe(r)})

    i1 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    i2 = np.array([[5, 6]], dtype=np.int32)
    r = np.concatenate([i1, i2], axis=0)
    cases.append({"op": "concatenate", "arrays": [enc(i1.tolist()), enc(i2.tolist())],
                  "dtype": "int32", "axis": 0, "expected": describe(r)})

    # 1D concatenate
    v1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    v2 = np.array([4.0, 5.0], dtype=np.float64)
    r = np.concatenate([v1, v2])
    cases.append({"op": "concatenate", "arrays": [enc(v1.tolist()), enc(v2.tolist())],
                  "dtype": "float64", "axis": 0, "expected": describe(r)})

    # ---- stack ----
    s1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    s2 = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    r = np.stack([s1, s2], axis=0)
    cases.append({"op": "stack", "arrays": [enc(s1.tolist()), enc(s2.tolist())],
                  "dtype": "float64", "axis": 0, "expected": describe(r)})

    r = np.stack([s1, s2], axis=1)
    cases.append({"op": "stack", "arrays": [enc(s1.tolist()), enc(s2.tolist())],
                  "dtype": "float64", "axis": 1, "expected": describe(r)})

    # ---- vstack ----
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    r = np.vstack([m1, m2])
    cases.append({"op": "vstack", "arrays": [enc(m1.tolist()), enc(m2.tolist())],
                  "dtype": "float64", "expected": describe(r)})

    # ---- hstack ----
    r = np.hstack([m1, m2])
    cases.append({"op": "hstack", "arrays": [enc(m1.tolist()), enc(m2.tolist())],
                  "dtype": "float64", "expected": describe(r)})

    # 1D hstack
    r = np.hstack([s1, s2])
    cases.append({"op": "hstack", "arrays": [enc(s1.tolist()), enc(s2.tolist())],
                  "dtype": "float64", "expected": describe(r)})

    # ---- split ----
    sp = np.arange(12.0, dtype=np.float64).reshape(4, 3)
    parts = np.split(sp, 2, axis=0)
    cases.append({"op": "split", "data": enc(sp.tolist()), "shape": [4, 3],
                  "dtype": "float64", "sections": 2, "axis": 0,
                  "expected": [describe(p) for p in parts]})

    parts = np.split(sp, 3, axis=1)
    cases.append({"op": "split", "data": enc(sp.tolist()), "shape": [4, 3],
                  "dtype": "float64", "sections": 3, "axis": 1,
                  "expected": [describe(p) for p in parts]})

    # split with indices
    v = np.arange(5.0, dtype=np.float64)
    parts = np.split(v, [2])
    cases.append({"op": "split", "data": enc(v.tolist()), "shape": [5],
                  "dtype": "float64", "indices": [2], "axis": 0,
                  "expected": [describe(p) for p in parts]})

    # ---- arraySplit (non-even) ----
    v5 = np.arange(5.0, dtype=np.float64)
    parts = np.array_split(v5, 2)
    cases.append({"op": "arraySplit", "data": enc(v5.tolist()), "shape": [5],
                  "dtype": "float64", "sections": 2, "axis": 0,
                  "expected": [describe(p) for p in parts]})

    v7 = np.arange(7.0, dtype=np.float64)
    parts = np.array_split(v7, 3)
    cases.append({"op": "arraySplit", "data": enc(v7.tolist()), "shape": [7],
                  "dtype": "float64", "sections": 3, "axis": 0,
                  "expected": [describe(p) for p in parts]})

    # ---- atleast1d ----
    r = np.atleast_1d(5.0)
    cases.append({"op": "atleast1d", "data": enc(5.0), "dtype": "float64",
                  "expected": describe(r)})

    r = np.atleast_1d(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    cases.append({"op": "atleast1d", "data": enc([1.0, 2.0, 3.0]), "shape": [3],
                  "dtype": "float64", "expected": describe(r)})

    r = np.atleast_1d(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    cases.append({"op": "atleast1d", "data": enc([[1.0, 2.0], [3.0, 4.0]]), "shape": [2, 2],
                  "dtype": "float64", "expected": describe(r)})

    # atleast2d
    r = np.atleast_2d(np.array([1.0, 2.0, 3.0], dtype=np.float64))
    cases.append({"op": "atleast2d", "data": enc([1.0, 2.0, 3.0]), "shape": [3],
                  "dtype": "float64", "expected": describe(r)})

    r = np.atleast_2d(5.0)
    cases.append({"op": "atleast2d", "data": enc(5.0), "dtype": "float64",
                  "expected": describe(r)})

    # atleast3d
    r = np.atleast_3d(np.array([1.0, 2.0], dtype=np.float64))
    cases.append({"op": "atleast3d", "data": enc([1.0, 2.0]), "shape": [2],
                  "dtype": "float64", "expected": describe(r)})

    r = np.atleast_3d(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    cases.append({"op": "atleast3d", "data": enc([[1.0, 2.0], [3.0, 4.0]]), "shape": [2, 2],
                  "dtype": "float64", "expected": describe(r)})

    # ---- repeat ----
    rep1 = np.array([1, 2, 3], dtype=np.int32)
    r = np.repeat(rep1, 2)
    cases.append({"op": "repeat", "data": enc(rep1.tolist()), "shape": [3],
                  "dtype": "int32", "repeats": 2, "axis": None, "expected": describe(r)})

    rep2 = np.array([[1, 2], [3, 4]], dtype=np.int32)
    r = np.repeat(rep2, 2, axis=1)
    cases.append({"op": "repeat", "data": enc(rep2.tolist()), "shape": [2, 2],
                  "dtype": "int32", "repeats": 2, "axis": 1, "expected": describe(r)})

    rep3 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    r = np.repeat(rep3, 3, axis=0)
    cases.append({"op": "repeat", "data": enc(rep3.tolist()), "shape": [2, 2],
                  "dtype": "float64", "repeats": 3, "axis": 0, "expected": describe(r)})

    # repeat with per-element counts
    rep4 = np.array([1, 2, 3], dtype=np.int32)
    r = np.repeat(rep4, [1, 2, 3])
    cases.append({"op": "repeat", "data": enc(rep4.tolist()), "shape": [3],
                  "dtype": "int32", "repeats": [1, 2, 3], "axis": None, "expected": describe(r)})

    # ---- flip ----
    fl = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    r = np.flip(fl)
    cases.append({"op": "flip", "data": enc(fl.tolist()), "shape": [2, 3],
                  "dtype": "float64", "axis": None, "expected": describe(r)})

    r = np.flip(fl, axis=0)
    cases.append({"op": "flip", "data": enc(fl.tolist()), "shape": [2, 3],
                  "dtype": "float64", "axis": 0, "expected": describe(r)})

    r = np.flip(fl, axis=1)
    cases.append({"op": "flip", "data": enc(fl.tolist()), "shape": [2, 3],
                  "dtype": "float64", "axis": 1, "expected": describe(r)})

    # fliplr
    r = np.fliplr(fl)
    cases.append({"op": "fliplr", "data": enc(fl.tolist()), "shape": [2, 3],
                  "dtype": "float64", "expected": describe(r)})

    # flipud
    r = np.flipud(fl)
    cases.append({"op": "flipud", "data": enc(fl.tolist()), "shape": [2, 3],
                  "dtype": "float64", "expected": describe(r)})

    # flip 1D
    fv = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    r = np.flip(fv)
    cases.append({"op": "flip", "data": enc(fv.tolist()), "shape": [4],
                  "dtype": "float64", "axis": None, "expected": describe(r)})

    # ---- roll ----
    rv = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    r = np.roll(rv, 2)
    cases.append({"op": "roll", "data": enc(rv.tolist()), "shape": [5],
                  "dtype": "float64", "shift": 2, "axis": None, "expected": describe(r)})

    r = np.roll(rv, -1)
    cases.append({"op": "roll", "data": enc(rv.tolist()), "shape": [5],
                  "dtype": "float64", "shift": -1, "axis": None, "expected": describe(r)})

    rm = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    r = np.roll(rm, 1, axis=0)
    cases.append({"op": "roll", "data": enc(rm.tolist()), "shape": [2, 3],
                  "dtype": "float64", "shift": 1, "axis": 0, "expected": describe(r)})

    r = np.roll(rm, 2, axis=1)
    cases.append({"op": "roll", "data": enc(rm.tolist()), "shape": [2, 3],
                  "dtype": "float64", "shift": 2, "axis": 1, "expected": describe(r)})

    # ---- rot90 ----
    rot = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)
    for k in [1, 2, 3]:
        r = np.rot90(rot, k=k)
        cases.append({"op": "rot90", "data": enc(rot.tolist()), "shape": [3, 2],
                      "dtype": "float64", "k": k, "expected": describe(r)})

    # rot90 k=1 default
    rot2 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    r = np.rot90(rot2)
    cases.append({"op": "rot90", "data": enc(rot2.tolist()), "shape": [2, 3],
                  "dtype": "float64", "k": 1, "expected": describe(r)})

    # ---- tile ----
    ta = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    r = np.tile(ta, 2)
    cases.append({"op": "tile", "data": enc(ta.tolist()), "shape": [3],
                  "dtype": "float64", "reps": 2, "expected": describe(r)})

    r = np.tile(ta, (2, 3))
    cases.append({"op": "tile", "data": enc(ta.tolist()), "shape": [3],
                  "dtype": "float64", "reps": [2, 3], "expected": describe(r)})

    tm = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    r = np.tile(tm, (2, 2))
    cases.append({"op": "tile", "data": enc(tm.tolist()), "shape": [2, 2],
                  "dtype": "float64", "reps": [2, 2], "expected": describe(r)})

    r = np.tile(tm, 3)
    cases.append({"op": "tile", "data": enc(tm.tolist()), "shape": [2, 2],
                  "dtype": "float64", "reps": 3, "expected": describe(r)})

    # ---- insert ----
    ins = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    r = np.insert(ins, 2, 99)
    cases.append({"op": "insert", "data": enc(ins.tolist()), "shape": [5],
                  "dtype": "int32", "obj": 2, "values": enc(99), "axis": None,
                  "expected": describe(r)})

    r = np.insert(ins, [1, 3], [10, 20])
    cases.append({"op": "insert", "data": enc(ins.tolist()), "shape": [5],
                  "dtype": "int32", "obj": [1, 3], "values": enc([10, 20]), "axis": None,
                  "expected": describe(r)})

    # insert with axis
    inm = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    r = np.insert(inm, 1, [10, 20, 30], axis=0)
    cases.append({"op": "insert", "data": enc(inm.tolist()), "shape": [2, 3],
                  "dtype": "int32", "obj": 1, "values": enc([10, 20, 30]), "axis": 0,
                  "expected": describe(r)})

    # ---- delete ----
    d = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    r = np.delete(d, 2)
    cases.append({"op": "delete", "data": enc(d.tolist()), "shape": [5],
                  "dtype": "int32", "obj": 2, "axis": None, "expected": describe(r)})

    r = np.delete(d, [0, 3])
    cases.append({"op": "delete", "data": enc(d.tolist()), "shape": [5],
                  "dtype": "int32", "obj": [0, 3], "axis": None, "expected": describe(r)})

    dm = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32)
    r = np.delete(dm, 1, axis=0)
    cases.append({"op": "delete", "data": enc(dm.tolist()), "shape": [3, 3],
                  "dtype": "int32", "obj": 1, "axis": 0, "expected": describe(r)})

    # ---- append ----
    ap = np.array([1, 2, 3], dtype=np.int32)
    r = np.append(ap, np.array([4, 5], dtype=np.int32))
    cases.append({"op": "append", "data": enc(ap.tolist()), "shape": [3],
                  "dtype": "int32", "values": enc([4, 5]), "axis": None,
                  "expected": describe(r)})

    apm = np.array([[1, 2], [3, 4]], dtype=np.int32)
    r = np.append(apm, np.array([[5, 6]], dtype=np.int32), axis=0)
    cases.append({"op": "append", "data": enc(apm.tolist()), "shape": [2, 2],
                  "dtype": "int32", "values": enc([[5, 6]]), "axis": 0,
                  "expected": describe(r)})

    # ---- resize ----
    rs = np.array([1, 2, 3], dtype=np.int32)
    r = np.resize(rs, (2, 3))
    cases.append({"op": "resize", "data": enc(rs.tolist()), "shape": [3],
                  "dtype": "int32", "new_shape": [2, 3], "expected": describe(r)})

    r = np.resize(rs, (6,))
    cases.append({"op": "resize", "data": enc(rs.tolist()), "shape": [3],
                  "dtype": "int32", "new_shape": [6], "expected": describe(r)})

    r = np.resize(rs, (2,))
    cases.append({"op": "resize", "data": enc(rs.tolist()), "shape": [3],
                  "dtype": "int32", "new_shape": [2], "expected": describe(r)})

    # ---- trimZeros ----
    tz1 = np.array([0, 0, 1, 2, 3, 0], dtype=np.int32)
    r = np.trim_zeros(tz1)
    cases.append({"op": "trimZeros", "data": enc(tz1.tolist()), "shape": [6],
                  "dtype": "int32", "trim": "fb", "expected": describe(r)})

    r = np.trim_zeros(tz1, "f")
    cases.append({"op": "trimZeros", "data": enc(tz1.tolist()), "shape": [6],
                  "dtype": "int32", "trim": "f", "expected": describe(r)})

    r = np.trim_zeros(tz1, "b")
    cases.append({"op": "trimZeros", "data": enc(tz1.tolist()), "shape": [6],
                  "dtype": "int32", "trim": "b", "expected": describe(r)})

    # ---- pad ----
    pv = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    r = np.pad(pv, 2, mode="constant")
    cases.append({"op": "pad", "data": enc(pv.tolist()), "shape": [3],
                  "dtype": "float64", "pad_width": 2, "mode": "constant",
                  "constant_values": 0, "expected": describe(r)})

    r = np.pad(pv, (1, 3), mode="constant")
    cases.append({"op": "pad", "data": enc(pv.tolist()), "shape": [3],
                  "dtype": "float64", "pad_width": [1, 3], "mode": "constant",
                  "constant_values": 0, "expected": describe(r)})

    r = np.pad(pv, 1, mode="constant", constant_values=5.0)
    cases.append({"op": "pad", "data": enc(pv.tolist()), "shape": [3],
                  "dtype": "float64", "pad_width": 1, "mode": "constant",
                  "constant_values": 5.0, "expected": describe(r)})

    pm = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    r = np.pad(pm, 1, mode="constant")
    cases.append({"op": "pad", "data": enc(pm.tolist()), "shape": [2, 2],
                  "dtype": "float64", "pad_width": 1, "mode": "constant",
                  "constant_values": 0, "expected": describe(r)})

    r = np.pad(pm, ((1, 2), (0, 1)), mode="constant", constant_values=0)
    cases.append({"op": "pad", "data": enc(pm.tolist()), "shape": [2, 2],
                  "dtype": "float64", "pad_width": [[1, 2], [0, 1]], "mode": "constant",
                  "constant_values": 0, "expected": describe(r)})

    # ---- broadcastArrays ----
    ba1 = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)   # shape (3,1)
    ba2 = np.array([[10.0, 20.0, 30.0, 40.0]], dtype=np.float64)  # shape (1,4)
    bparts = np.broadcast_arrays(ba1, ba2)
    cases.append({"op": "broadcastArrays",
                  "arrays": [
                      {"data": enc(ba1.tolist()), "shape": [3, 1], "dtype": "float64"},
                      {"data": enc(ba2.tolist()), "shape": [1, 4], "dtype": "float64"},
                  ],
                  "expected": [describe(p) for p in bparts]})

    # simple 1D broadcast
    bc1 = np.array([1.0, 2.0, 3.0], dtype=np.float64)   # shape (3,)
    bc2 = np.array([[1.0], [2.0]], dtype=np.float64)     # shape (2,1)
    bparts2 = np.broadcast_arrays(bc1, bc2)
    cases.append({"op": "broadcastArrays",
                  "arrays": [
                      {"data": enc(bc1.tolist()), "shape": [3], "dtype": "float64"},
                      {"data": enc(bc2.tolist()), "shape": [2, 1], "dtype": "float64"},
                  ],
                  "expected": [describe(p) for p in bparts2]})

    # ---- shape / size / ndim ----
    meta1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    cases.append({"op": "shape", "data": enc(meta1.tolist()), "shape": [2, 3],
                  "dtype": "float64", "expected": list(meta1.shape)})

    cases.append({"op": "size", "data": enc(meta1.tolist()), "shape": [2, 3],
                  "dtype": "float64", "axis": None, "expected": int(meta1.size)})

    cases.append({"op": "size", "data": enc(meta1.tolist()), "shape": [2, 3],
                  "dtype": "float64", "axis": 0, "expected": int(meta1.shape[0])})

    cases.append({"op": "size", "data": enc(meta1.tolist()), "shape": [2, 3],
                  "dtype": "float64", "axis": 1, "expected": int(meta1.shape[1])})

    cases.append({"op": "ndim", "data": enc(meta1.tolist()), "shape": [2, 3],
                  "dtype": "float64", "expected": int(meta1.ndim)})

    meta2 = np.array([1.0, 2.0], dtype=np.float64)
    cases.append({"op": "ndim", "data": enc(meta2.tolist()), "shape": [2],
                  "dtype": "float64", "expected": int(meta2.ndim)})

    # ---- isfortran ----
    fc = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64, order="C")
    cases.append({"op": "isfortran", "data": enc(fc.tolist()), "shape": [2, 2],
                  "dtype": "float64", "order": "C", "expected": bool(np.isfortran(fc))})

    ff = np.asfortranarray(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64))
    cases.append({"op": "isfortran", "data": enc(ff.tolist()), "shape": [2, 2],
                  "dtype": "float64", "order": "F", "expected": bool(np.isfortran(ff))})

    # ---- copyto ----
    dst = np.zeros((3,), dtype=np.float64)
    src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    np.copyto(dst, src)
    cases.append({"op": "copyto", "dst_shape": [3], "dst_dtype": "float64",
                  "src": enc(src.tolist()), "src_dtype": "float64",
                  "expected": describe(dst)})

    dst2 = np.zeros((2, 2), dtype=np.float64)
    src2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    np.copyto(dst2, src2)
    cases.append({"op": "copyto", "dst_shape": [2, 2], "dst_dtype": "float64",
                  "src": enc(src2.tolist()), "src_dtype": "float64",
                  "expected": describe(dst2)})

    # ---- asanyarray ----
    aa = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    r = np.asanyarray(aa)
    cases.append({"op": "asanyarray", "data": enc(aa.tolist()), "shape": [3],
                  "dtype": "float64", "expected": describe(r)})

    r = np.asanyarray([1, 2, 3], dtype=np.float64)
    cases.append({"op": "asanyarray", "data": enc([1, 2, 3]), "dtype": "float64",
                  "expected": describe(r)})

    # ---- asarrayChkfinite ----
    ac = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    import numpy.lib
    r = np.lib.array_utils.normalize_axis_index  # just check import
    r = np.asarray(ac)  # asarray_chkfinite is the same for finite values
    cases.append({"op": "asarrayChkfinite", "data": enc(ac.tolist()), "shape": [3],
                  "dtype": "float64", "expected": describe(r)})

    ac2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    cases.append({"op": "asarrayChkfinite", "data": enc(ac2.tolist()), "shape": [2, 2],
                  "dtype": "float64", "expected": describe(ac2)})

    # ---- require ----
    req = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    r = np.require(req, dtype=np.float64, requirements=["C_CONTIGUOUS"])
    cases.append({"op": "require", "data": enc(req.tolist()), "shape": [2, 2],
                  "dtype": "float64", "requirements": ["C_CONTIGUOUS"],
                  "expected": describe(r)})

    r = np.require(req, dtype=np.float32, requirements=["C_CONTIGUOUS"])
    cases.append({"op": "require", "data": enc(req.tolist()), "shape": [2, 2],
                  "src_dtype": "float64", "dtype": "float32",
                  "requirements": ["C_CONTIGUOUS"],
                  "expected": describe(r)})

    return cases


def creation2_cases() -> list[dict]:
    import math
    import numpy as np

    def enc(v):
        if isinstance(v, list):
            return [enc(x) for x in v]
        if isinstance(v, bool):
            return v
        if isinstance(v, complex):
            return {"re": enc(v.real), "im": enc(v.imag)}
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

    def describe(a: np.ndarray) -> dict:
        return {
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "strides": list(a.strides),
            "c_contiguous": bool(a.flags.c_contiguous),
            "f_contiguous": bool(a.flags.f_contiguous),
            "values": enc(a.tolist()),
        }

    cases = []
    mat3x3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

    # ---------- logspace ----------
    cases.append({
        "op": "logspace", "start": 0, "stop": 2, "num": 5, "kw": {},
        "expected": describe(np.logspace(0, 2, 5)),
    })
    cases.append({
        "op": "logspace", "start": 0, "stop": 3, "num": 5, "kw": {"base": 2},
        "expected": describe(np.logspace(0, 3, 5, base=2)),
    })
    cases.append({
        "op": "logspace", "start": 0, "stop": 4, "num": 5, "kw": {"endpoint": False},
        "expected": describe(np.logspace(0, 4, 5, endpoint=False)),
    })
    cases.append({
        "op": "logspace", "start": 0, "stop": 2, "num": 5, "kw": {"dtype": "float32"},
        "expected": describe(np.logspace(0, 2, 5, dtype="float32")),
    })
    cases.append({
        "op": "logspace", "start": 0, "stop": 2, "num": 4, "kw": {"base": 10},
        "expected": describe(np.logspace(0, 2, 4, base=10)),
    })

    # ---------- geomspace ----------
    cases.append({
        "op": "geomspace", "start": 1, "stop": 1000, "num": 5, "kw": {},
        "expected": describe(np.geomspace(1, 1000, 5)),
    })
    cases.append({
        "op": "geomspace", "start": 2, "stop": 32, "num": 5, "kw": {},
        "expected": describe(np.geomspace(2, 32, 5)),
    })
    cases.append({
        "op": "geomspace", "start": 1, "stop": 16, "num": 5, "kw": {"endpoint": False},
        "expected": describe(np.geomspace(1, 16, 5, endpoint=False)),
    })
    cases.append({
        "op": "geomspace", "start": 1, "stop": 100, "num": 5, "kw": {"dtype": "float64"},
        "expected": describe(np.geomspace(1, 100, 5, dtype="float64")),
    })

    # ---------- tri ----------
    cases.append({
        "op": "tri", "n": 4, "kw": {},
        "expected": describe(np.tri(4)),
    })
    cases.append({
        "op": "tri", "n": 3, "kw": {"m": 4},
        "expected": describe(np.tri(3, 4)),
    })
    cases.append({
        "op": "tri", "n": 4, "kw": {"k": 1},
        "expected": describe(np.tri(4, k=1)),
    })
    cases.append({
        "op": "tri", "n": 4, "kw": {"k": -1},
        "expected": describe(np.tri(4, k=-1)),
    })
    cases.append({
        "op": "tri", "n": 3, "kw": {"dtype": "float64"},
        "expected": describe(np.tri(3, dtype="float64")),
    })
    cases.append({
        "op": "tri", "n": 3, "kw": {"dtype": "int32"},
        "expected": describe(np.tri(3, dtype="int32")),
    })

    # ---------- tril ----------
    cases.append({
        "op": "tril", "data": mat3x3, "kw": {},
        "expected": describe(np.tril(mat3x3)),
    })
    cases.append({
        "op": "tril", "data": mat3x3, "kw": {"k": 1},
        "expected": describe(np.tril(mat3x3, k=1)),
    })
    cases.append({
        "op": "tril", "data": mat3x3, "kw": {"k": -1},
        "expected": describe(np.tril(mat3x3, k=-1)),
    })

    # ---------- triu ----------
    cases.append({
        "op": "triu", "data": mat3x3, "kw": {},
        "expected": describe(np.triu(mat3x3)),
    })
    cases.append({
        "op": "triu", "data": mat3x3, "kw": {"k": 1},
        "expected": describe(np.triu(mat3x3, k=1)),
    })
    cases.append({
        "op": "triu", "data": mat3x3, "kw": {"k": -1},
        "expected": describe(np.triu(mat3x3, k=-1)),
    })

    # ---------- diag ----------
    cases.append({
        "op": "diag", "data": mat3x3, "kw": {},
        "expected": describe(np.diag(np.array(mat3x3))),
    })
    cases.append({
        "op": "diag", "data": mat3x3, "kw": {"k": 1},
        "expected": describe(np.diag(np.array(mat3x3), k=1)),
    })
    cases.append({
        "op": "diag", "data": mat3x3, "kw": {"k": -1},
        "expected": describe(np.diag(np.array(mat3x3), k=-1)),
    })
    cases.append({
        "op": "diag", "data": [1, 2, 3], "kw": {},
        "expected": describe(np.diag([1, 2, 3])),
    })
    cases.append({
        "op": "diag", "data": [1, 2, 3], "kw": {"k": 1},
        "expected": describe(np.diag([1, 2, 3], k=1)),
    })

    # ---------- diagflat ----------
    cases.append({
        "op": "diagflat", "data": [1, 2, 3], "kw": {},
        "expected": describe(np.diagflat([1, 2, 3])),
    })
    cases.append({
        "op": "diagflat", "data": [1, 2, 3], "kw": {"k": 1},
        "expected": describe(np.diagflat([1, 2, 3], k=1)),
    })

    # ---------- vander ----------
    cases.append({
        "op": "vander", "data": [1, 2, 3, 4], "kw": {},
        "expected": describe(np.vander([1, 2, 3, 4])),
    })
    cases.append({
        "op": "vander", "data": [1, 2, 3, 4], "kw": {"n": 3},
        "expected": describe(np.vander([1, 2, 3, 4], 3)),
    })
    cases.append({
        "op": "vander", "data": [1, 2, 3, 4], "kw": {"n": 3, "increasing": True},
        "expected": describe(np.vander([1, 2, 3, 4], 3, increasing=True)),
    })
    cases.append({
        "op": "vander", "data": [1, 2, 3, 4], "kw": {"increasing": False},
        "expected": describe(np.vander([1, 2, 3, 4], increasing=False)),
    })

    # ---------- trilIndices / triuIndices ----------
    r = np.tril_indices(4)
    cases.append({
        "op": "trilIndices", "n": 4, "kw": {},
        "expected": [describe(r[0]), describe(r[1])],
    })
    r = np.tril_indices(4, k=-1)
    cases.append({
        "op": "trilIndices", "n": 4, "kw": {"k": -1},
        "expected": [describe(r[0]), describe(r[1])],
    })
    r = np.tril_indices(4, m=5)
    cases.append({
        "op": "trilIndices", "n": 4, "kw": {"m": 5},
        "expected": [describe(r[0]), describe(r[1])],
    })
    r = np.triu_indices(4)
    cases.append({
        "op": "triuIndices", "n": 4, "kw": {},
        "expected": [describe(r[0]), describe(r[1])],
    })
    r = np.triu_indices(4, k=1)
    cases.append({
        "op": "triuIndices", "n": 4, "kw": {"k": 1},
        "expected": [describe(r[0]), describe(r[1])],
    })

    # ---------- diagIndices ----------
    r = np.diag_indices(3)
    cases.append({
        "op": "diagIndices", "n": 3, "kw": {},
        "expected": [describe(r[0]), describe(r[1])],
    })
    r = np.diag_indices(3, ndim=3)
    cases.append({
        "op": "diagIndices", "n": 3, "kw": {"ndim": 3},
        "expected": [describe(r[0]), describe(r[1]), describe(r[2])],
    })
    r = np.diag_indices(4)
    cases.append({
        "op": "diagIndices", "n": 4, "kw": {},
        "expected": [describe(r[0]), describe(r[1])],
    })

    # ---------- fromfunction ----------
    r = np.fromfunction(lambda i, j: i + j, (3, 3), dtype=float)
    cases.append({
        "op": "fromfunction", "shape": [3, 3], "kw": {"dtype": "float64"},
        "expected": describe(r),
    })
    r = np.fromfunction(lambda i, j: i * j, (2, 4), dtype=float)
    cases.append({
        "op": "fromfunction", "shape": [2, 4], "kw": {"dtype": "float64"},
        "fn": "multiply",
        "expected": describe(r),
    })

    # ---------- fromiter ----------
    r = np.fromiter(range(5), dtype=float)
    cases.append({
        "op": "fromiter", "iter": list(range(5)), "kw": {"dtype": "float64"},
        "expected": describe(r),
    })
    r = np.fromiter(range(6), dtype="int32")
    cases.append({
        "op": "fromiter", "iter": list(range(6)), "kw": {"dtype": "int32"},
        "expected": describe(r),
    })

    # ---------- frombuffer ----------
    buf = bytes([0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64])
    r = np.frombuffer(buf, dtype="float64")
    cases.append({
        "op": "frombuffer", "hex": buf.hex(), "kw": {"dtype": "float64"},
        "expected": describe(r),
    })
    buf2 = bytes([1, 2, 3, 4, 5, 6])
    r = np.frombuffer(buf2, dtype="int8")
    cases.append({
        "op": "frombuffer", "hex": buf2.hex(), "kw": {"dtype": "int8"},
        "expected": describe(r),
    })

    # ---------- fromstring ----------
    r = np.fromstring("1 2 3 4", sep=" ", dtype=float)
    cases.append({
        "op": "fromstring", "text": "1 2 3 4", "kw": {"sep": " ", "dtype": "float64"},
        "expected": describe(r),
    })
    r = np.fromstring("1,2,3", sep=",", dtype="int32")
    cases.append({
        "op": "fromstring", "text": "1,2,3", "kw": {"sep": ",", "dtype": "int32"},
        "expected": describe(r),
    })

    # ---------- indices ----------
    r = np.indices([3, 4])
    cases.append({
        "op": "indices", "shape": [3, 4], "kw": {},
        "expected": describe(r),
    })
    r = np.indices([2, 3], dtype="int32")
    cases.append({
        "op": "indices", "shape": [2, 3], "kw": {"dtype": "int32"},
        "expected": describe(r),
    })

    # ---------- meshgrid ----------
    x = [1.0, 2.0, 3.0]
    y = [4.0, 5.0]
    rx, ry = np.meshgrid(x, y, indexing="xy")
    cases.append({
        "op": "meshgrid", "arrays": [x, y], "kw": {"indexing": "xy"},
        "expected": [describe(rx), describe(ry)],
    })
    rx, ry = np.meshgrid(x, y, indexing="ij")
    cases.append({
        "op": "meshgrid", "arrays": [x, y], "kw": {"indexing": "ij"},
        "expected": [describe(rx), describe(ry)],
    })

    # ---------- astype ----------
    src_f64 = np.array([1.7, -2.2])
    cases.append({
        "op": "astype", "data": enc(src_f64.tolist()), "src_dtype": "float64",
        "kw": {"dtype": "int32"},
        "expected": describe(src_f64.astype("int32")),
    })
    cases.append({
        "op": "astype", "data": enc(src_f64.tolist()), "src_dtype": "float64",
        "kw": {"dtype": "float32"},
        "expected": describe(src_f64.astype("float32")),
    })
    src_i32 = np.array([1, 2, 3, 4], dtype="int32")
    cases.append({
        "op": "astype", "data": enc(src_i32.tolist()), "src_dtype": "int32",
        "kw": {"dtype": "float64"},
        "expected": describe(src_i32.astype("float64")),
    })
    cases.append({
        "op": "astype", "data": enc(src_f64.tolist()), "src_dtype": "float64",
        "kw": {"dtype": "float64", "copy": True},
        "expected": describe(src_f64.astype("float64", copy=True)),
    })

    return cases


def indexing2_cases() -> list[dict]:
    """V-P8 differential test cases: P8 indexing extras (D-248).

    Covers take, takeAlongAxis, putAlongAxis, put, putmask, place, choose,
    compress, extract, select, piecewise, argwhere, flatnonzero, countNonzero,
    ravelMultiIndex, unravelIndex, diagonal, trace.
    """
    import warnings
    import numpy as np

    cases = []

    def _run(fn, *args, **kw):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                return fn(*args, **kw), None
            except (IndexError,) as e:
                return None, "IndexError"
            except (TypeError,) as e:
                return None, "TypeError"
            except (ValueError,) as e:
                return None, "ValueError"

    # -----------------------------------------------------------------------
    # take
    # -----------------------------------------------------------------------
    a1d_f = np.array([1.5, 2.5, 3.5, 4.5, 5.5], dtype="float64")
    a1d_i = np.array([10, 20, 30, 40, 50], dtype="int32")
    a2d = np.arange(12, dtype="float64").reshape(3, 4)

    for mode in ("raise", "wrap", "clip"):
        r, _ = _run(np.take, a1d_f, [0, 2], mode=mode)
        cases.append({
            "op": "take",
            "a_data": a1d_f.tolist(),
            "a_dtype": "float64",
            "indices": [0, 2],
            "kw": {"mode": mode},
            "expected": describe(r),
        })

    for mode in ("raise", "wrap", "clip"):
        r, _ = _run(np.take, a1d_i, [0, 4], mode=mode)
        cases.append({
            "op": "take",
            "a_data": a1d_i.tolist(),
            "a_dtype": "int32",
            "indices": [0, 4],
            "kw": {"mode": mode},
            "expected": describe(r),
        })

    # 2D along axis=0
    r, _ = _run(np.take, a2d, [0, 2], axis=0)
    cases.append({
        "op": "take",
        "a_data": a2d.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 4],
        "indices": [0, 2],
        "kw": {"axis": 0},
        "expected": describe(r),
    })

    # 2D along axis=1
    r, _ = _run(np.take, a2d, [1, 3], axis=1)
    cases.append({
        "op": "take",
        "a_data": a2d.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 4],
        "indices": [1, 3],
        "kw": {"axis": 1},
        "expected": describe(r),
    })

    # wrap mode with out-of-range indices
    r, _ = _run(np.take, a1d_f, [-1, 6], mode="wrap")
    cases.append({
        "op": "take",
        "a_data": a1d_f.tolist(),
        "a_dtype": "float64",
        "indices": [-1, 6],
        "kw": {"mode": "wrap"},
        "expected": describe(r),
    })

    # clip mode
    r, _ = _run(np.take, a1d_f, [-9, 99], mode="clip")
    cases.append({
        "op": "take",
        "a_data": a1d_f.tolist(),
        "a_dtype": "float64",
        "indices": [-9, 99],
        "kw": {"mode": "clip"},
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # takeAlongAxis
    # -----------------------------------------------------------------------
    arr2d = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]], dtype="float64")
    # use argmax indices
    argmax_idx = np.argmax(arr2d, axis=1).reshape(2, 1)  # shape (2,1)
    r, _ = _run(np.take_along_axis, arr2d, argmax_idx, 1)
    cases.append({
        "op": "takeAlongAxis",
        "a_data": arr2d.tolist(),
        "a_dtype": "float64",
        "a_shape": [2, 3],
        "indices_data": argmax_idx.tolist(),
        "axis": 1,
        "expected": describe(r),
    })

    # axis=0 case
    argmin_idx = np.argmin(arr2d, axis=0).reshape(1, 3)
    r, _ = _run(np.take_along_axis, arr2d, argmin_idx, 0)
    cases.append({
        "op": "takeAlongAxis",
        "a_data": arr2d.tolist(),
        "a_dtype": "float64",
        "a_shape": [2, 3],
        "indices_data": argmin_idx.tolist(),
        "axis": 0,
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # putAlongAxis  (void — record modified copy)
    # -----------------------------------------------------------------------
    base2d = np.arange(6, dtype="int64").reshape(2, 3)
    b = base2d.copy()
    np.put_along_axis(b, np.array([[2], [0]]), 99, axis=1)
    cases.append({
        "op": "putAlongAxis",
        "a_data": base2d.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 3],
        "indices_data": [[2], [0]],
        "values": 99,
        "axis": 1,
        "expected": describe(b),
    })

    # -----------------------------------------------------------------------
    # put  (void)
    # -----------------------------------------------------------------------
    x_put = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64")
    b = x_put.copy()
    np.put(b, [0, 3], [9.0, 7.0])
    cases.append({
        "op": "put",
        "a_data": x_put.tolist(),
        "a_dtype": "float64",
        "ind_data": [0, 3],
        "v_data": [9.0, 7.0],
        "kw": {},
        "expected": describe(b),
    })

    # wrap mode
    b = x_put.copy()
    np.put(b, [0, 7], [9.0, 8.0], mode="wrap")
    cases.append({
        "op": "put",
        "a_data": x_put.tolist(),
        "a_dtype": "float64",
        "ind_data": [0, 7],
        "v_data": [9.0, 8.0],
        "kw": {"mode": "wrap"},
        "expected": describe(b),
    })

    # clip mode
    b = x_put.copy()
    np.put(b, [-9], [5.0], mode="clip")
    cases.append({
        "op": "put",
        "a_data": x_put.tolist(),
        "a_dtype": "float64",
        "ind_data": [-9],
        "v_data": [5.0],
        "kw": {"mode": "clip"},
        "expected": describe(b),
    })

    # -----------------------------------------------------------------------
    # putmask  (void)
    # -----------------------------------------------------------------------
    x_pm = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64")
    b = x_pm.copy()
    np.putmask(b, [False, False, True, True, True], [10.0, 20.0])
    cases.append({
        "op": "putmask",
        "a_data": x_pm.tolist(),
        "a_dtype": "float64",
        "mask": [False, False, True, True, True],
        "values_data": [10.0, 20.0],
        "expected": describe(b),
    })

    # -----------------------------------------------------------------------
    # place  (void)
    # -----------------------------------------------------------------------
    x_pl = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64")
    b = x_pl.copy()
    np.place(b, [0, 0, 1, 1, 1], [10.0, 20.0])
    cases.append({
        "op": "place",
        "a_data": x_pl.tolist(),
        "a_dtype": "float64",
        "mask": [0, 0, 1, 1, 1],
        "vals_data": [10.0, 20.0],
        "expected": describe(b),
    })

    # -----------------------------------------------------------------------
    # choose
    # -----------------------------------------------------------------------
    ch = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    r, _ = _run(np.choose, [0, 1, 2], ch)
    cases.append({
        "op": "choose",
        "a_data": [0, 1, 2],
        "choices": ch,
        "kw": {},
        "expected": describe(r),
    })

    # mode=clip
    r, _ = _run(np.choose, [0, 5], [[1.0, 2.0], [3.0, 4.0]], mode="clip")
    cases.append({
        "op": "choose",
        "a_data": [0, 5],
        "choices": [[1.0, 2.0], [3.0, 4.0]],
        "kw": {"mode": "clip"},
        "expected": describe(r),
    })

    # mode=wrap
    r, _ = _run(np.choose, [-1, 2], [[1.0, 2.0], [3.0, 4.0]], mode="wrap")
    cases.append({
        "op": "choose",
        "a_data": [-1, 2],
        "choices": [[1.0, 2.0], [3.0, 4.0]],
        "kw": {"mode": "wrap"},
        "expected": describe(r),
    })

    # 2D index broadcast
    r, _ = _run(np.choose, [[0], [1]], [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cases.append({
        "op": "choose",
        "a_data": [[0], [1]],
        "choices": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "kw": {},
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # compress
    # -----------------------------------------------------------------------
    a_comp = np.arange(6, dtype="int64")
    r, _ = _run(np.compress, [True, False, True, True, False, False], a_comp)
    cases.append({
        "op": "compress",
        "condition": [True, False, True, True, False, False],
        "a_data": a_comp.tolist(),
        "a_dtype": "int64",
        "kw": {},
        "expected": describe(r),
    })

    a_comp2d = np.arange(6, dtype="int64").reshape(2, 3)
    r, _ = _run(np.compress, [True, False], a_comp2d, axis=0)
    cases.append({
        "op": "compress",
        "condition": [True, False],
        "a_data": a_comp2d.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 3],
        "kw": {"axis": 0},
        "expected": describe(r),
    })

    r, _ = _run(np.compress, [True, False, True], a_comp2d, axis=1)
    cases.append({
        "op": "compress",
        "condition": [True, False, True],
        "a_data": a_comp2d.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 3],
        "kw": {"axis": 1},
        "expected": describe(r),
    })

    # 1D with 4 condition elements (flattened)
    r, _ = _run(np.compress, [True, False, True, True], a_comp2d)
    cases.append({
        "op": "compress",
        "condition": [True, False, True, True],
        "a_data": a_comp2d.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 3],
        "kw": {},
        "expected": describe(r),
    })

    # float64 array
    a_comp_f = np.array([1.0, 0.0, 3.0, 0.0, 5.0], dtype="float64")
    r, _ = _run(np.compress, [True, False, True, False, True], a_comp_f)
    cases.append({
        "op": "compress",
        "condition": [True, False, True, False, True],
        "a_data": a_comp_f.tolist(),
        "a_dtype": "float64",
        "kw": {},
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # extract
    # -----------------------------------------------------------------------
    a_ext = np.array([[1, 2, 3], [4, 5, 6]], dtype="int64")
    cond_ext = np.array([[False, False, False], [True, True, True]])
    r, _ = _run(np.extract, cond_ext, a_ext)
    cases.append({
        "op": "extract",
        "condition": cond_ext.tolist(),
        "a_data": a_ext.tolist(),
        "a_dtype": "int64",
        "expected": describe(r),
    })

    # 1D extract
    r, _ = _run(np.extract, [True, False, True], [1, 2, 3, 4])
    cases.append({
        "op": "extract",
        "condition": [True, False, True],
        "a_data": [1, 2, 3, 4],
        "a_dtype": "int64",
        "expected": describe(r),
    })

    # nonzero condition extract
    a_ext2 = np.array([0.0, 1.0, 0.0, 2.0, 3.0], dtype="float64")
    cond2 = a_ext2 > 0
    r, _ = _run(np.extract, cond2, a_ext2)
    cases.append({
        "op": "extract",
        "condition": cond2.tolist(),
        "a_data": a_ext2.tolist(),
        "a_dtype": "float64",
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # select
    # -----------------------------------------------------------------------
    x_sel = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype="float64")
    cond1 = x_sel < 2
    cond2 = x_sel >= 3
    ch1 = x_sel * 10
    ch2 = x_sel * -1
    r, _ = _run(np.select, [cond1, cond2], [ch1, ch2], default=0.0)
    cases.append({
        "op": "select",
        "condlist": [cond1.tolist(), cond2.tolist()],
        "choicelist": [ch1.tolist(), ch2.tolist()],
        "default": 0.0,
        "dtype": "float64",
        "expected": describe(r),
    })

    # integer select
    x_isel = np.array([10, 20, 30, 40, 50], dtype="int32")
    r, _ = _run(np.select,
                [x_isel > 25, x_isel < 15],
                [x_isel + 100, x_isel - 5])
    cases.append({
        "op": "select",
        "condlist": [(x_isel > 25).tolist(), (x_isel < 15).tolist()],
        "choicelist": [(x_isel + 100).tolist(), (x_isel - 5).tolist()],
        "default": 0,
        "dtype": "int32",
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # piecewise
    # -----------------------------------------------------------------------
    x_pw = np.array([-1.0, 0.0, 1.5], dtype="float64")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        r_pw = np.piecewise(
            x_pw,
            [[True, False, False], [False, False, True]],
            [lambda v: np.negative(v), 5.0, 7.0],
        )
    cases.append({
        "op": "piecewise",
        "x_data": x_pw.tolist(),
        "x_dtype": "float64",
        "condlist": [[True, False, False], [False, False, True]],
        "funclist_scalar": [None, 5.0, 7.0],
        "expected": describe(r_pw),
    })

    x_pw2 = np.arange(4, dtype="float64")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        r_pw2 = np.piecewise(
            x_pw2,
            [x_pw2 >= 2],
            [lambda v: v * 10.0, -1.0],
        )
    cases.append({
        "op": "piecewise",
        "x_data": x_pw2.tolist(),
        "x_dtype": "float64",
        "condlist_derived": ">=2",
        "expected": describe(r_pw2),
    })

    # -----------------------------------------------------------------------
    # argwhere
    # -----------------------------------------------------------------------
    aw = np.array([[0, 3], [4, 0]], dtype="int64")
    r, _ = _run(np.argwhere, aw)
    cases.append({
        "op": "argwhere",
        "a_data": aw.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 2],
        "expected": describe(r),
    })

    aw2 = np.array([0.0, 1.0, 0.0, 2.0, 0.0], dtype="float64")
    r, _ = _run(np.argwhere, aw2)
    cases.append({
        "op": "argwhere",
        "a_data": aw2.tolist(),
        "a_dtype": "float64",
        "a_shape": [5],
        "expected": describe(r),
    })

    # all zeros
    aw3 = np.zeros((2, 3), dtype="float64")
    r, _ = _run(np.argwhere, aw3)
    cases.append({
        "op": "argwhere",
        "a_data": aw3.tolist(),
        "a_dtype": "float64",
        "a_shape": [2, 3],
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # flatnonzero
    # -----------------------------------------------------------------------
    fn_arr = np.array([[0, 1], [2, 0]], dtype="int64")
    r, _ = _run(np.flatnonzero, fn_arr)
    cases.append({
        "op": "flatnonzero",
        "a_data": fn_arr.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 2],
        "expected": describe(r),
    })

    fn_arr2 = np.array([1.0, 0.0, -1.0, 0.0, 2.5], dtype="float64")
    r, _ = _run(np.flatnonzero, fn_arr2)
    cases.append({
        "op": "flatnonzero",
        "a_data": fn_arr2.tolist(),
        "a_dtype": "float64",
        "expected": describe(r),
    })

    fn_arr3 = np.zeros(5, dtype="int64")
    r, _ = _run(np.flatnonzero, fn_arr3)
    cases.append({
        "op": "flatnonzero",
        "a_data": fn_arr3.tolist(),
        "a_dtype": "int64",
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # countNonzero
    # -----------------------------------------------------------------------
    cn_arr = np.arange(6, dtype="int64").reshape(2, 3)
    r, _ = _run(np.count_nonzero, cn_arr)
    cases.append({
        "op": "countNonzero",
        "a_data": cn_arr.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 3],
        "kw": {},
        "expected": describe(np.asarray(r)),
    })

    r, _ = _run(np.count_nonzero, cn_arr, axis=0)
    cases.append({
        "op": "countNonzero",
        "a_data": cn_arr.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 3],
        "kw": {"axis": 0},
        "expected": describe(np.asarray(r)),
    })

    r, _ = _run(np.count_nonzero, cn_arr, axis=1)
    cases.append({
        "op": "countNonzero",
        "a_data": cn_arr.tolist(),
        "a_dtype": "int64",
        "a_shape": [2, 3],
        "kw": {"axis": 1},
        "expected": describe(np.asarray(r)),
    })

    # 1D float array
    cn_arr2 = np.array([0.0, 1.0, 0.0, 2.0, 3.0], dtype="float64")
    r, _ = _run(np.count_nonzero, cn_arr2)
    cases.append({
        "op": "countNonzero",
        "a_data": cn_arr2.tolist(),
        "a_dtype": "float64",
        "a_shape": [5],
        "kw": {},
        "expected": describe(np.asarray(r)),
    })

    # -----------------------------------------------------------------------
    # ravelMultiIndex
    # -----------------------------------------------------------------------
    rows = [1, 2]
    cols = [3, 1]
    dims = [4, 5]
    r, _ = _run(np.ravel_multi_index, [rows, cols], dims)
    cases.append({
        "op": "ravelMultiIndex",
        "multi_index": [rows, cols],
        "dims": dims,
        "kw": {},
        "expected": describe(r),
    })

    r, _ = _run(np.ravel_multi_index, [rows, cols], dims, order="F")
    cases.append({
        "op": "ravelMultiIndex",
        "multi_index": [rows, cols],
        "dims": dims,
        "kw": {"order": "F"},
        "expected": describe(r),
    })

    r, _ = _run(np.ravel_multi_index, [[-1, 4], [3, 1]], dims, mode="wrap")
    cases.append({
        "op": "ravelMultiIndex",
        "multi_index": [[-1, 4], [3, 1]],
        "dims": dims,
        "kw": {"mode": "wrap"},
        "expected": describe(r),
    })

    r, _ = _run(np.ravel_multi_index, [[-1, 5], [3, 1]], dims, mode="clip")
    cases.append({
        "op": "ravelMultiIndex",
        "multi_index": [[-1, 5], [3, 1]],
        "dims": dims,
        "kw": {"mode": "clip"},
        "expected": describe(r),
    })

    # mixed mode per axis
    r, _ = _run(np.ravel_multi_index, [[1, 2], [3, 9]], [3, 4],
                mode=["raise", "clip"])
    cases.append({
        "op": "ravelMultiIndex",
        "multi_index": [[1, 2], [3, 9]],
        "dims": [3, 4],
        "kw": {"mode": ["raise", "clip"]},
        "expected": describe(r),
    })

    # 2D multi-index
    r, _ = _run(np.ravel_multi_index, [[[0, 1]], [[2], [3]]], [2, 4])
    cases.append({
        "op": "ravelMultiIndex",
        "multi_index": [[[0, 1]], [[2], [3]]],
        "dims": [2, 4],
        "kw": {},
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # unravelIndex
    # -----------------------------------------------------------------------
    r = np.unravel_index([5, 7], [3, 4])
    cases.append({
        "op": "unravelIndex",
        "flat_indices": [5, 7],
        "shape": [3, 4],
        "kw": {},
        "expected": [describe(x) for x in r],
    })

    r = np.unravel_index(5, [3, 4], order="F")
    cases.append({
        "op": "unravelIndex",
        "flat_indices": 5,
        "shape": [3, 4],
        "kw": {"order": "F"},
        "expected": [describe(x) for x in r],
    })

    r = np.unravel_index([[1, 2]], [3, 4])
    cases.append({
        "op": "unravelIndex",
        "flat_indices": [[1, 2]],
        "shape": [3, 4],
        "kw": {},
        "expected": [describe(x) for x in r],
    })

    r = np.unravel_index(np.array([], dtype=np.intp), [3, 4])
    cases.append({
        "op": "unravelIndex",
        "flat_indices": [],
        "shape": [3, 4],
        "kw": {},
        "expected": [describe(x) for x in r],
    })

    # -----------------------------------------------------------------------
    # diagonal
    # -----------------------------------------------------------------------
    m3x3 = np.arange(9, dtype="float64").reshape(3, 3)
    r, _ = _run(np.diagonal, m3x3)
    cases.append({
        "op": "diagonal",
        "a_data": m3x3.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 3],
        "kw": {},
        "expected": describe(r),
    })

    # offset=1
    r, _ = _run(np.diagonal, m3x3, offset=1)
    cases.append({
        "op": "diagonal",
        "a_data": m3x3.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 3],
        "kw": {"offset": 1},
        "expected": describe(r),
    })

    # offset=-1
    r, _ = _run(np.diagonal, m3x3, offset=-1)
    cases.append({
        "op": "diagonal",
        "a_data": m3x3.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 3],
        "kw": {"offset": -1},
        "expected": describe(r),
    })

    # float32 matrix
    m3x3_f32 = np.arange(9, dtype="float32").reshape(3, 3)
    r, _ = _run(np.diagonal, m3x3_f32)
    cases.append({
        "op": "diagonal",
        "a_data": m3x3_f32.tolist(),
        "a_dtype": "float32",
        "a_shape": [3, 3],
        "kw": {},
        "expected": describe(r),
    })

    # 3D array with axis1=0, axis2=1
    a3d = np.arange(24, dtype="float64").reshape(2, 3, 4)
    r, _ = _run(np.diagonal, a3d, offset=0, axis1=0, axis2=1)
    cases.append({
        "op": "diagonal",
        "a_data": a3d.tolist(),
        "a_dtype": "float64",
        "a_shape": [2, 3, 4],
        "kw": {"offset": 0, "axis1": 0, "axis2": 1},
        "expected": describe(r),
    })

    # 3D with offset=1
    r, _ = _run(np.diagonal, a3d, offset=1, axis1=0, axis2=2)
    cases.append({
        "op": "diagonal",
        "a_data": a3d.tolist(),
        "a_dtype": "float64",
        "a_shape": [2, 3, 4],
        "kw": {"offset": 1, "axis1": 0, "axis2": 2},
        "expected": describe(r),
    })

    # integer matrix
    m4x4_i = np.arange(16, dtype="int32").reshape(4, 4)
    r, _ = _run(np.diagonal, m4x4_i)
    cases.append({
        "op": "diagonal",
        "a_data": m4x4_i.tolist(),
        "a_dtype": "int32",
        "a_shape": [4, 4],
        "kw": {},
        "expected": describe(r),
    })

    # -----------------------------------------------------------------------
    # trace
    # -----------------------------------------------------------------------
    r, _ = _run(np.trace, m3x3)
    cases.append({
        "op": "trace",
        "a_data": m3x3.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 3],
        "kw": {},
        "expected": describe(np.asarray(r)),
    })

    r, _ = _run(np.trace, m3x3, offset=1)
    cases.append({
        "op": "trace",
        "a_data": m3x3.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 3],
        "kw": {"offset": 1},
        "expected": describe(np.asarray(r)),
    })

    r, _ = _run(np.trace, m3x3, dtype="float64")
    cases.append({
        "op": "trace",
        "a_data": m3x3.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 3],
        "kw": {"dtype": "float64"},
        "expected": describe(np.asarray(r)),
    })

    # int32 trace (should return int64)
    r, _ = _run(np.trace, m4x4_i)
    cases.append({
        "op": "trace",
        "a_data": m4x4_i.tolist(),
        "a_dtype": "int32",
        "a_shape": [4, 4],
        "kw": {},
        "expected": describe(np.asarray(r)),
    })

    # 3D trace
    r, _ = _run(np.trace, a3d)
    cases.append({
        "op": "trace",
        "a_data": a3d.tolist(),
        "a_dtype": "float64",
        "a_shape": [2, 3, 4],
        "kw": {},
        "expected": describe(np.asarray(r)),
    })

    # 3D trace with axis1=0, axis2=2, offset=1
    r, _ = _run(np.trace, a3d, offset=1, axis1=0, axis2=2)
    cases.append({
        "op": "trace",
        "a_data": a3d.tolist(),
        "a_dtype": "float64",
        "a_shape": [2, 3, 4],
        "kw": {"offset": 1, "axis1": 0, "axis2": 2},
        "expected": describe(np.asarray(r)),
    })

    # eye matrix trace
    eye3 = np.eye(3, dtype="float64")
    r, _ = _run(np.trace, eye3)
    cases.append({
        "op": "trace",
        "a_data": eye3.tolist(),
        "a_dtype": "float64",
        "a_shape": [3, 3],
        "kw": {},
        "expected": describe(np.asarray(r)),
    })

    return cases


def stats_cases() -> list[dict]:
    """Differential test cases for P10: statistics and NaN reduction functions (V-P10, D-250)."""
    import math
    import numpy as np

    cases = []

    # ---- helpers (local copies of module-level enc/describe) ----
    def enc(v):
        if isinstance(v, list):
            return [enc(x) for x in v]
        if isinstance(v, bool):
            return v
        if isinstance(v, complex):
            return {"re": enc(v.real), "im": enc(v.imag)}
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

    # ================================================================
    # 1. quantile / percentile
    # ================================================================
    a1d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=np.float64)
    for q in [0.25, 0.5, 0.75]:
        cases.append({
            "op": "quantile", "a_data": enc(a1d.tolist()), "a_dtype": "float64",
            "a_shape": [8], "q": q, "kw": {},
            "approx": True,
            "expected": describe(np.quantile(a1d, q)),
        })
    cases.append({
        "op": "quantile", "a_data": enc(a1d.tolist()), "a_dtype": "float64",
        "a_shape": [8], "q": [0.25, 0.5, 0.75], "kw": {},
        "approx": True,
        "expected": describe(np.quantile(a1d, [0.25, 0.5, 0.75])),
    })
    # 2D quantile along axis 0 and 1
    a2d = np.arange(12, dtype=np.float64).reshape(3, 4)
    for axis in [0, 1]:
        cases.append({
            "op": "quantile", "a_data": enc(a2d.tolist()), "a_dtype": "float64",
            "a_shape": [3, 4], "q": 0.5, "kw": {"axis": axis},
            "approx": True,
            "expected": describe(np.quantile(a2d, 0.5, axis=axis)),
        })
    cases.append({
        "op": "quantile", "a_data": enc(a2d.tolist()), "a_dtype": "float64",
        "a_shape": [3, 4], "q": 0.5, "kw": {"axis": 0, "keepdims": True},
        "approx": True,
        "expected": describe(np.quantile(a2d, 0.5, axis=0, keepdims=True)),
    })
    # percentile (same data, q as percentage)
    cases.append({
        "op": "percentile", "a_data": enc(a1d.tolist()), "a_dtype": "float64",
        "a_shape": [8], "q": [25.0, 50.0, 75.0], "kw": {},
        "approx": True,
        "expected": describe(np.percentile(a1d, [25.0, 50.0, 75.0])),
    })

    # ================================================================
    # 2. median
    # ================================================================
    cases.append({
        "op": "median", "a_data": enc(a1d.tolist()), "a_dtype": "float64",
        "a_shape": [8], "kw": {},
        "approx": True,
        "expected": describe(np.median(a1d)),
    })
    cases.append({
        "op": "median", "a_data": enc(a2d.tolist()), "a_dtype": "float64",
        "a_shape": [3, 4], "kw": {"axis": 0},
        "approx": True,
        "expected": describe(np.median(a2d, axis=0)),
    })
    cases.append({
        "op": "median", "a_data": enc(a2d.tolist()), "a_dtype": "float64",
        "a_shape": [3, 4], "kw": {"axis": 1, "keepdims": True},
        "approx": True,
        "expected": describe(np.median(a2d, axis=1, keepdims=True)),
    })

    # ================================================================
    # 3. nanquantile / nanmedian  (arrays with NaN)
    # ================================================================
    anan = np.array([1.0, float("nan"), 3.0, float("nan"), 5.0], dtype=np.float64)
    cases.append({
        "op": "nanquantile", "a_data": enc(anan.tolist()), "a_dtype": "float64",
        "a_shape": [5], "q": 0.5, "kw": {},
        "approx": True,
        "expected": describe(np.nanquantile(anan, 0.5)),
    })
    cases.append({
        "op": "nanquantile", "a_data": enc(anan.tolist()), "a_dtype": "float64",
        "a_shape": [5], "q": [0.25, 0.75], "kw": {},
        "approx": True,
        "expected": describe(np.nanquantile(anan, [0.25, 0.75])),
    })
    cases.append({
        "op": "nanpercentile", "a_data": enc(anan.tolist()), "a_dtype": "float64",
        "a_shape": [5], "q": 50.0, "kw": {},
        "approx": True,
        "expected": describe(np.nanpercentile(anan, 50.0)),
    })
    anan2d = np.array([[1.0, float("nan"), 3.0], [float("nan"), 5.0, 6.0]], dtype=np.float64)
    cases.append({
        "op": "nanmedian", "a_data": enc(anan2d.tolist()), "a_dtype": "float64",
        "a_shape": [2, 3], "kw": {"axis": 1},
        "approx": True,
        "expected": describe(np.nanmedian(anan2d, axis=1)),
    })
    cases.append({
        "op": "nanmedian", "a_data": enc(anan.tolist()), "a_dtype": "float64",
        "a_shape": [5], "kw": {},
        "approx": True,
        "expected": describe(np.nanmedian(anan)),
    })

    # ================================================================
    # 4. cumsum / cumprod
    # ================================================================
    ai1d = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    cases.append({
        "op": "cumsum", "a_data": enc(ai1d.tolist()), "a_dtype": "int64",
        "a_shape": [5], "kw": {},
        "approx": False,
        "expected": describe(np.cumsum(ai1d)),
    })
    cases.append({
        "op": "cumprod", "a_data": enc(ai1d.tolist()), "a_dtype": "int64",
        "a_shape": [5], "kw": {},
        "approx": False,
        "expected": describe(np.cumprod(ai1d)),
    })
    a2d_i = np.arange(6, dtype=np.int64).reshape(2, 3) + 1
    for axis in [0, 1]:
        cases.append({
            "op": "cumsum", "a_data": enc(a2d_i.tolist()), "a_dtype": "int64",
            "a_shape": [2, 3], "kw": {"axis": axis},
            "approx": False,
            "expected": describe(np.cumsum(a2d_i, axis=axis)),
        })
        cases.append({
            "op": "cumprod", "a_data": enc(a2d_i.tolist()), "a_dtype": "int64",
            "a_shape": [2, 3], "kw": {"axis": axis},
            "approx": False,
            "expected": describe(np.cumprod(a2d_i, axis=axis)),
        })

    # ================================================================
    # 5. diff
    # ================================================================
    adiff = np.array([1.0, 4.0, 9.0, 16.0, 25.0], dtype=np.float64)
    cases.append({
        "op": "diff", "a_data": enc(adiff.tolist()), "a_dtype": "float64",
        "a_shape": [5], "n": 1, "kw": {},
        "approx": False,
        "expected": describe(np.diff(adiff, 1)),
    })
    cases.append({
        "op": "diff", "a_data": enc(adiff.tolist()), "a_dtype": "float64",
        "a_shape": [5], "n": 2, "kw": {},
        "approx": False,
        "expected": describe(np.diff(adiff, 2)),
    })
    adiff2d = np.array([[1.0, 3.0, 6.0], [10.0, 15.0, 21.0]], dtype=np.float64)
    for axis in [0, 1]:
        cases.append({
            "op": "diff", "a_data": enc(adiff2d.tolist()), "a_dtype": "float64",
            "a_shape": [2, 3], "n": 1, "kw": {"axis": axis},
            "approx": False,
            "expected": describe(np.diff(adiff2d, 1, axis=axis)),
        })

    # ================================================================
    # 6. ptp
    # ================================================================
    cases.append({
        "op": "ptp", "a_data": enc(a1d.tolist()), "a_dtype": "float64",
        "a_shape": [8], "kw": {},
        "approx": False,
        "expected": describe(np.ptp(a1d)),
    })
    cases.append({
        "op": "ptp", "a_data": enc(a2d.tolist()), "a_dtype": "float64",
        "a_shape": [3, 4], "kw": {"axis": 0},
        "approx": False,
        "expected": describe(np.ptp(a2d, axis=0)),
    })
    cases.append({
        "op": "ptp", "a_data": enc(a2d.tolist()), "a_dtype": "float64",
        "a_shape": [3, 4], "kw": {"axis": 1, "keepdims": True},
        "approx": False,
        "expected": describe(np.ptp(a2d, axis=1, keepdims=True)),
    })

    # ================================================================
    # 7. nansum / nanprod / nanmean / nanvar / nanstd / nanmin / nanmax / nanargmin / nanargmax
    # ================================================================
    anan3 = np.array([1.0, float("nan"), 3.0, float("nan"), 5.0, 2.0], dtype=np.float64)
    anan2x3 = np.array([[1.0, float("nan"), 3.0], [float("nan"), float("nan"), 6.0]], dtype=np.float64)

    for fn_name, np_fn in [
        ("nansum", np.nansum), ("nanprod", np.nanprod), ("nanmean", np.nanmean),
    ]:
        cases.append({
            "op": fn_name, "a_data": enc(anan3.tolist()), "a_dtype": "float64",
            "a_shape": [6], "kw": {},
            "approx": True,
            "expected": describe(np_fn(anan3)),
        })
        cases.append({
            "op": fn_name, "a_data": enc(anan2x3.tolist()), "a_dtype": "float64",
            "a_shape": [2, 3], "kw": {"axis": 1},
            "approx": True,
            "expected": describe(np_fn(anan2x3, axis=1)),
        })

    for fn_name, np_fn in [
        ("nanvar", np.nanvar), ("nanstd", np.nanstd),
    ]:
        cases.append({
            "op": fn_name, "a_data": enc(anan3.tolist()), "a_dtype": "float64",
            "a_shape": [6], "kw": {},
            "approx": True,
            "expected": describe(np_fn(anan3)),
        })
        cases.append({
            "op": fn_name, "a_data": enc(anan3.tolist()), "a_dtype": "float64",
            "a_shape": [6], "kw": {"ddof": 1},
            "approx": True,
            "expected": describe(np_fn(anan3, ddof=1)),
        })

    for fn_name, np_fn in [
        ("nanmin", np.nanmin), ("nanmax", np.nanmax),
    ]:
        cases.append({
            "op": fn_name, "a_data": enc(anan3.tolist()), "a_dtype": "float64",
            "a_shape": [6], "kw": {},
            "approx": False,
            "expected": describe(np_fn(anan3)),
        })
        cases.append({
            "op": fn_name, "a_data": enc(anan2x3.tolist()), "a_dtype": "float64",
            "a_shape": [2, 3], "kw": {"axis": 0},
            "approx": False,
            "expected": describe(np_fn(anan2x3, axis=0)),
        })

    for fn_name, np_fn in [
        ("nanargmin", np.nanargmin), ("nanargmax", np.nanargmax),
    ]:
        cases.append({
            "op": fn_name, "a_data": enc(anan3.tolist()), "a_dtype": "float64",
            "a_shape": [6], "kw": {},
            "approx": False,
            "expected": describe(np_fn(anan3)),
        })

    # ================================================================
    # 8. average
    # ================================================================
    aavg = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    cases.append({
        "op": "average", "a_data": enc(aavg.tolist()), "a_dtype": "float64",
        "a_shape": [5], "kw": {},
        "approx": True,
        "expected": describe(np.average(aavg)),
    })
    # With weights
    weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    avg_w = np.average(aavg, weights=weights)
    cases.append({
        "op": "average", "a_data": enc(aavg.tolist()), "a_dtype": "float64",
        "a_shape": [5], "kw": {"weights": enc(weights.tolist())},
        "approx": True,
        "expected": describe(avg_w),
    })
    # 2D with axis
    a2d_avg = np.arange(6, dtype=np.float64).reshape(2, 3)
    cases.append({
        "op": "average", "a_data": enc(a2d_avg.tolist()), "a_dtype": "float64",
        "a_shape": [2, 3], "kw": {"axis": 1},
        "approx": True,
        "expected": describe(np.average(a2d_avg, axis=1)),
    })
    # returned=True → [avg, sum_of_weights]
    avg_ret, wsum_ret = np.average(aavg, weights=weights, returned=True)
    cases.append({
        "op": "average_returned",
        "a_data": enc(aavg.tolist()), "a_dtype": "float64",
        "a_shape": [5], "weights": enc(weights.tolist()),
        "approx": True,
        "expected": [describe(avg_ret), describe(wsum_ret)],
    })

    # ================================================================
    # 9. cov
    # ================================================================
    acov = np.array([[1.0, 2.0, 3.0, 4.0],
                     [5.0, 3.0, 1.0, 2.0]], dtype=np.float64)
    cases.append({
        "op": "cov", "a_data": enc(acov.tolist()), "a_dtype": "float64",
        "a_shape": [2, 4], "kw": {"rowvar": True},
        "approx": True,
        "expected": describe(np.cov(acov, rowvar=True)),
    })
    cases.append({
        "op": "cov", "a_data": enc(acov.T.copy().tolist()), "a_dtype": "float64",
        "a_shape": [4, 2], "kw": {"rowvar": False},
        "approx": True,
        "expected": describe(np.cov(acov.T, rowvar=False)),
    })
    cases.append({
        "op": "cov", "a_data": enc(acov.tolist()), "a_dtype": "float64",
        "a_shape": [2, 4], "kw": {"rowvar": True, "bias": True},
        "approx": True,
        "expected": describe(np.cov(acov, rowvar=True, bias=True)),
    })
    cases.append({
        "op": "cov", "a_data": enc(acov.tolist()), "a_dtype": "float64",
        "a_shape": [2, 4], "kw": {"rowvar": True, "ddof": 2},
        "approx": True,
        "expected": describe(np.cov(acov, rowvar=True, ddof=2)),
    })

    # ================================================================
    # 10. corrcoef
    # ================================================================
    cases.append({
        "op": "corrcoef", "a_data": enc(acov.tolist()), "a_dtype": "float64",
        "a_shape": [2, 4], "kw": {"rowvar": True},
        "approx": True,
        "expected": describe(np.corrcoef(acov, rowvar=True)),
    })
    cases.append({
        "op": "corrcoef", "a_data": enc(acov.T.copy().tolist()), "a_dtype": "float64",
        "a_shape": [4, 2], "kw": {"rowvar": False},
        "approx": True,
        "expected": describe(np.corrcoef(acov.T, rowvar=False)),
    })

    # ================================================================
    # 11. gradient
    # ================================================================
    agrads = np.array([1.0, 2.0, 4.0, 7.0, 11.0], dtype=np.float64)
    g_default = np.gradient(agrads)
    cases.append({
        "op": "gradient", "a_data": enc(agrads.tolist()), "a_dtype": "float64",
        "a_shape": [5], "spacing": None, "kw": {},
        "approx": True,
        "expected": describe(g_default),
    })
    g_scalar = np.gradient(agrads, 2.0)
    cases.append({
        "op": "gradient_scalar", "a_data": enc(agrads.tolist()), "a_dtype": "float64",
        "a_shape": [5], "spacing": 2.0, "kw": {},
        "approx": True,
        "expected": describe(g_scalar),
    })
    xcoords = np.array([0.0, 1.0, 3.0, 4.0, 7.0], dtype=np.float64)
    g_coords = np.gradient(agrads, xcoords)
    cases.append({
        "op": "gradient_coords", "a_data": enc(agrads.tolist()), "a_dtype": "float64",
        "a_shape": [5], "xcoords": enc(xcoords.tolist()), "kw": {},
        "approx": True,
        "expected": describe(g_coords),
    })

    # ================================================================
    # 12. trapezoid
    # ================================================================
    atrap = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    cases.append({
        "op": "trapezoid", "a_data": enc(atrap.tolist()), "a_dtype": "float64",
        "a_shape": [5], "kw": {},
        "approx": True,
        "expected": describe(np.trapezoid(atrap)),
    })
    xtrap = np.array([0.0, 1.0, 3.0, 5.0, 6.0], dtype=np.float64)
    cases.append({
        "op": "trapezoid", "a_data": enc(atrap.tolist()), "a_dtype": "float64",
        "a_shape": [5], "kw": {"x": enc(xtrap.tolist())},
        "approx": True,
        "expected": describe(np.trapezoid(atrap, x=xtrap)),
    })
    cases.append({
        "op": "trapezoid", "a_data": enc(atrap.tolist()), "a_dtype": "float64",
        "a_shape": [5], "kw": {"dx": 2.0},
        "approx": True,
        "expected": describe(np.trapezoid(atrap, dx=2.0)),
    })
    atrap2d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    cases.append({
        "op": "trapezoid", "a_data": enc(atrap2d.tolist()), "a_dtype": "float64",
        "a_shape": [2, 3], "kw": {"axis": 0},
        "approx": True,
        "expected": describe(np.trapezoid(atrap2d, axis=0)),
    })

    # ================================================================
    # 13. histogram
    # ================================================================
    ahist = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 3.5, 4.0, 0.5], dtype=np.float64)
    hist_r, edges_r = np.histogram(ahist, bins=4)
    cases.append({
        "op": "histogram", "a_data": enc(ahist.tolist()), "a_dtype": "float64",
        "a_shape": [8], "kw": {"bins": 4},
        "approx": False,
        "expected": {"hist": describe(hist_r), "binEdges": describe(edges_r)},
    })
    # Explicit bin edges
    bin_edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    hist_e, edges_e = np.histogram(ahist, bins=bin_edges)
    cases.append({
        "op": "histogram", "a_data": enc(ahist.tolist()), "a_dtype": "float64",
        "a_shape": [8], "kw": {"bins": enc(bin_edges.tolist())},
        "approx": False,
        "expected": {"hist": describe(hist_e), "binEdges": describe(edges_e)},
    })
    # density=True
    hist_d, edges_d = np.histogram(ahist, bins=4, density=True)
    cases.append({
        "op": "histogram", "a_data": enc(ahist.tolist()), "a_dtype": "float64",
        "a_shape": [8], "kw": {"bins": 4, "density": True},
        "approx": True,
        "expected": {"hist": describe(hist_d), "binEdges": describe(edges_d)},
    })
    # range clipping
    hist_range, edges_range = np.histogram(ahist, bins=3, range=(1.0, 3.0))
    cases.append({
        "op": "histogram", "a_data": enc(ahist.tolist()), "a_dtype": "float64",
        "a_shape": [8], "kw": {"bins": 3, "range": [1.0, 3.0]},
        "approx": False,
        "expected": {"hist": describe(hist_range), "binEdges": describe(edges_range)},
    })

    # ================================================================
    # 14. bincount
    # ================================================================
    abc = np.array([0, 1, 2, 1, 0, 3, 2, 1], dtype=np.int64)
    cases.append({
        "op": "bincount", "a_data": enc(abc.tolist()), "a_dtype": "int64",
        "a_shape": [8], "kw": {},
        "approx": False,
        "expected": describe(np.bincount(abc)),
    })
    cases.append({
        "op": "bincount", "a_data": enc(abc.tolist()), "a_dtype": "int64",
        "a_shape": [8], "kw": {"minlength": 6},
        "approx": False,
        "expected": describe(np.bincount(abc, minlength=6)),
    })
    wbc = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], dtype=np.float64)
    cases.append({
        "op": "bincount", "a_data": enc(abc.tolist()), "a_dtype": "int64",
        "a_shape": [8], "kw": {"weights": enc(wbc.tolist())},
        "approx": True,
        "expected": describe(np.bincount(abc, weights=wbc)),
    })

    # ================================================================
    # 15. digitize
    # ================================================================
    xdig = np.array([0.2, 6.4, 3.0, 1.6, 2.5], dtype=np.float64)
    bins_dig = np.array([0.0, 1.0, 2.5, 4.0, 10.0], dtype=np.float64)
    cases.append({
        "op": "digitize", "a_data": enc(xdig.tolist()), "a_dtype": "float64",
        "a_shape": [5], "bins": enc(bins_dig.tolist()), "kw": {"right": False},
        "approx": False,
        "expected": describe(np.digitize(xdig, bins_dig, right=False)),
    })
    cases.append({
        "op": "digitize", "a_data": enc(xdig.tolist()), "a_dtype": "float64",
        "a_shape": [5], "bins": enc(bins_dig.tolist()), "kw": {"right": True},
        "approx": False,
        "expected": describe(np.digitize(xdig, bins_dig, right=True)),
    })

    # ================================================================
    # 16. interp
    # ================================================================
    xinterp = np.array([0.0, 0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    xp = np.array([0.0, 1.0, 2.0], dtype=np.float64)
    fp = np.array([0.0, 3.0, 1.0], dtype=np.float64)
    cases.append({
        "op": "interp",
        "x_data": enc(xinterp.tolist()), "xp_data": enc(xp.tolist()), "fp_data": enc(fp.tolist()),
        "kw": {},
        "approx": True,
        "expected": describe(np.interp(xinterp, xp, fp)),
    })
    cases.append({
        "op": "interp",
        "x_data": enc(xinterp.tolist()), "xp_data": enc(xp.tolist()), "fp_data": enc(fp.tolist()),
        "kw": {"left": -99.0, "right": 99.0},
        "approx": True,
        "expected": describe(np.interp(xinterp, xp, fp, left=-99.0, right=99.0)),
    })

    # ================================================================
    # 17. correlate — mode full / valid / same
    # ================================================================
    acorr = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    vcorr = np.array([1.0, 0.5, 0.5], dtype=np.float64)
    for mode in ["full", "valid", "same"]:
        cases.append({
            "op": "correlate",
            "a_data": enc(acorr.tolist()), "v_data": enc(vcorr.tolist()),
            "a_dtype": "float64", "v_dtype": "float64",
            "kw": {"mode": mode},
            "approx": True,
            "expected": describe(np.correlate(acorr, vcorr, mode=mode)),
        })

    # ================================================================
    # 18. convolve — mode full / valid / same
    # ================================================================
    aconv = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    vconv = np.array([1.0, 0.5], dtype=np.float64)
    for mode in ["full", "valid", "same"]:
        cases.append({
            "op": "convolve",
            "a_data": enc(aconv.tolist()), "v_data": enc(vconv.tolist()),
            "a_dtype": "float64", "v_dtype": "float64",
            "kw": {"mode": mode},
            "approx": True,
            "expected": describe(np.convolve(aconv, vconv, mode=mode)),
        })

    return cases



def sort_cases() -> list[dict]:
    import math
    import warnings
    import numpy as np

    def enc(v):
        """Encode a scalar or nested list into exact JSON."""
        if isinstance(v, list):
            return [enc(x) for x in v]
        if isinstance(v, bool):
            return v
        if isinstance(v, complex):
            return {"re": enc(v.real), "im": enc(v.imag)}
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

    def describe(a: np.ndarray) -> dict:
        return {
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "strides": list(a.strides),
            "c_contiguous": bool(a.flags.c_contiguous),
            "f_contiguous": bool(a.flags.f_contiguous),
            "values": enc(a.tolist()),
        }

    cases = []

    # ------------------------------------------------------------------ sort
    # 1D float64 with NaN and negatives
    a_f64 = np.array([3.0, float("nan"), -1.5, 2.0, float("nan"), 0.0])
    cases.append({
        "op": "sort",
        "a_data": enc(a_f64.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_f64.shape),
        "kw": {},
        "expected": describe(np.sort(a_f64)),
    })

    # 1D float64 descending (no NaN — NaN placement in descending sort is implementation-defined)
    a_f64_nonan = np.array([3.0, -1.5, 2.0, 0.0, 4.5, -0.5])
    cases.append({
        "op": "sort",
        "a_data": enc(a_f64_nonan.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_f64_nonan.shape),
        "kw": {"descending": True},
        "expected": describe(np.sort(a_f64_nonan)[::-1].copy()),
    })

    # 1D int32
    a_i32 = np.array([5, -3, 2, 8, 1, -7], dtype=np.int32)
    cases.append({
        "op": "sort",
        "a_data": enc(a_i32.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_i32.shape),
        "kw": {},
        "expected": describe(np.sort(a_i32)),
    })

    # 2D float64 axis=0
    a_2d = np.array([[3.0, 1.0, 4.0], [1.0, 5.0, 0.0]], dtype=np.float64)
    cases.append({
        "op": "sort",
        "a_data": enc(a_2d.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_2d.shape),
        "kw": {"axis": 0},
        "expected": describe(np.sort(a_2d, axis=0)),
    })

    # 2D float64 axis=1
    cases.append({
        "op": "sort",
        "a_data": enc(a_2d.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_2d.shape),
        "kw": {"axis": 1},
        "expected": describe(np.sort(a_2d, axis=1)),
    })

    # ---------------------------------------------------------------- argsort
    # 1D float64 with NaN
    cases.append({
        "op": "argsort",
        "a_data": enc(a_f64.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_f64.shape),
        "kw": {},
        "expected": describe(np.argsort(a_f64)),
    })

    # 1D float64 descending (no NaN — NaN placement in descending argsort is implementation-defined)
    cases.append({
        "op": "argsort",
        "a_data": enc(a_f64_nonan.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_f64_nonan.shape),
        "kw": {"descending": True},
        "expected": describe(np.argsort(a_f64_nonan)[::-1].copy()),
    })

    # 1D int32
    cases.append({
        "op": "argsort",
        "a_data": enc(a_i32.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_i32.shape),
        "kw": {},
        "expected": describe(np.argsort(a_i32)),
    })

    # 2D float64 axis=0
    cases.append({
        "op": "argsort",
        "a_data": enc(a_2d.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_2d.shape),
        "kw": {"axis": 0},
        "expected": describe(np.argsort(a_2d, axis=0)),
    })

    # ---------------------------------------------------------------- sortComplex
    # complex64 input
    a_cx = np.array([3+2j, 1-1j, 2+0j, 1+1j], dtype=np.complex64)
    cases.append({
        "op": "sortComplex",
        "a_data": enc(a_cx.tolist()),
        "a_dtype": "complex64",
        "a_shape": list(a_cx.shape),
        "kw": {},
        "expected": describe(np.sort(a_cx)),
    })

    # ---------------------------------------------------------------- partition
    # kth=2 on 1D float64
    a_part = np.array([7.0, 2.0, 5.0, 1.0, 9.0, 3.0, 8.0], dtype=np.float64)
    r_part2 = np.partition(a_part, 2)
    cases.append({
        "op": "partition",
        "a_data": enc(a_part.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_part.shape),
        "kw": {"kth": 2},
        "expected_kth": 2,
        "expected": describe(r_part2),
    })

    # kth=[1, 3] on 1D float64
    r_part13 = np.partition(a_part, [1, 3])
    cases.append({
        "op": "partition",
        "a_data": enc(a_part.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_part.shape),
        "kw": {"kth": [1, 3]},
        "expected_kth": [1, 3],
        "expected": describe(r_part13),
    })

    # ---------------------------------------------------------------- argpartition
    # kth=2 on 1D float64
    cases.append({
        "op": "argpartition",
        "a_data": enc(a_part.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_part.shape),
        "kw": {"kth": 2},
        "expected_kth": 2,
        "expected": describe(np.argpartition(a_part, 2)),
    })

    # kth=[1, 3] on 1D float64
    cases.append({
        "op": "argpartition",
        "a_data": enc(a_part.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_part.shape),
        "kw": {"kth": [1, 3]},
        "expected_kth": [1, 3],
        "expected": describe(np.argpartition(a_part, [1, 3])),
    })

    # ---------------------------------------------------------------- lexsort
    # Sort by last-name then first-name (NumPy convention: last key is primary)
    surnames  = np.array([3, 1, 2, 1], dtype=np.int64)
    firstnames = np.array([0, 2, 1, 1], dtype=np.int64)
    r_lex = np.lexsort([firstnames, surnames])
    cases.append({
        "op": "lexsort",
        "keys": [enc(firstnames.tolist()), enc(surnames.tolist())],
        "key_dtypes": ["int64", "int64"],
        "kw": {},
        "expected": describe(r_lex),
    })

    # ---------------------------------------------------------------- searchsorted
    a_sorted = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    v_vals   = np.array([0.5, 2.0, 3.5, 5.5], dtype=np.float64)

    cases.append({
        "op": "searchsorted",
        "a_data": enc(a_sorted.tolist()),
        "a_dtype": "float64",
        "v_data": enc(v_vals.tolist()),
        "v_dtype": "float64",
        "kw": {"side": "left"},
        "expected": describe(np.searchsorted(a_sorted, v_vals, side="left")),
    })

    cases.append({
        "op": "searchsorted",
        "a_data": enc(a_sorted.tolist()),
        "a_dtype": "float64",
        "v_data": enc(v_vals.tolist()),
        "v_dtype": "float64",
        "kw": {"side": "right"},
        "expected": describe(np.searchsorted(a_sorted, v_vals, side="right")),
    })

    # searchsorted with duplicates
    a_dup = np.array([1.0, 2.0, 2.0, 3.0], dtype=np.float64)
    v_dup = np.array([2.0], dtype=np.float64)
    cases.append({
        "op": "searchsorted",
        "a_data": enc(a_dup.tolist()),
        "a_dtype": "float64",
        "v_data": enc(v_dup.tolist()),
        "v_dtype": "float64",
        "kw": {"side": "left"},
        "expected": describe(np.searchsorted(a_dup, v_dup, side="left")),
    })
    cases.append({
        "op": "searchsorted",
        "a_data": enc(a_dup.tolist()),
        "a_dtype": "float64",
        "v_data": enc(v_dup.tolist()),
        "v_dtype": "float64",
        "kw": {"side": "right"},
        "expected": describe(np.searchsorted(a_dup, v_dup, side="right")),
    })

    # ---------------------------------------------------------------- unique
    a_uniq = np.array([3, 1, 4, 1, 5, 9, 2, 6, 5, 3], dtype=np.int32)

    # basic
    cases.append({
        "op": "unique",
        "a_data": enc(a_uniq.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_uniq.shape),
        "kw": {},
        "expected_values": describe(np.unique(a_uniq)),
    })

    # returnIndex=True
    uv, ui = np.unique(a_uniq, return_index=True)
    cases.append({
        "op": "unique",
        "a_data": enc(a_uniq.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_uniq.shape),
        "kw": {"returnIndex": True},
        "expected_values": describe(uv),
        "expected_indices": describe(ui),
    })

    # returnInverse=True
    uv2, uinv = np.unique(a_uniq, return_inverse=True)
    cases.append({
        "op": "unique",
        "a_data": enc(a_uniq.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_uniq.shape),
        "kw": {"returnInverse": True},
        "expected_values": describe(uv2),
        "expected_inverse": describe(uinv),
    })

    # returnCounts=True
    uv3, uc = np.unique(a_uniq, return_counts=True)
    cases.append({
        "op": "unique",
        "a_data": enc(a_uniq.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_uniq.shape),
        "kw": {"returnCounts": True},
        "expected_values": describe(uv3),
        "expected_counts": describe(uc),
    })

    # equalNan=True: NaN values treated as identical
    a_nan = np.array([1.0, float("nan"), 2.0, float("nan")], dtype=np.float64)
    uv_nan = np.unique(a_nan, equal_nan=True)
    cases.append({
        "op": "unique",
        "a_data": enc(a_nan.tolist()),
        "a_dtype": "float64",
        "a_shape": list(a_nan.shape),
        "kw": {"equalNan": True},
        "expected_values": describe(uv_nan),
    })

    # ---------------------------------------------------------------- uniqueAll
    a_ua = np.array([2, 1, 2, 3, 1], dtype=np.int32)
    uv_a, ui_a, uinv_a, uc_a = np.unique(a_ua, return_index=True, return_inverse=True, return_counts=True)
    cases.append({
        "op": "uniqueAll",
        "a_data": enc(a_ua.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_ua.shape),
        "kw": {},
        "expected_values": describe(uv_a),
        "expected_indices": describe(ui_a),
        "expected_inverse_indices": describe(uinv_a),
        "expected_counts": describe(uc_a),
    })

    # ---------------------------------------------------------------- intersect1d
    a_set = np.array([1, 3, 4, 3, 2], dtype=np.int32)
    b_set = np.array([3, 1, 2, 5, 1], dtype=np.int32)

    iv = np.intersect1d(a_set, b_set)
    cases.append({
        "op": "intersect1d",
        "a_data": enc(a_set.tolist()),
        "a_dtype": "int32",
        "b_data": enc(b_set.tolist()),
        "b_dtype": "int32",
        "kw": {},
        "expected": describe(iv),
    })

    # returnIndices=True
    iv2, ia, ib = np.intersect1d(a_set, b_set, return_indices=True)
    cases.append({
        "op": "intersect1d",
        "a_data": enc(a_set.tolist()),
        "a_dtype": "int32",
        "b_data": enc(b_set.tolist()),
        "b_dtype": "int32",
        "kw": {"returnIndices": True},
        "expected": describe(iv2),
        "expected_ia": describe(ia),
        "expected_ib": describe(ib),
    })

    # ---------------------------------------------------------------- union1d
    cases.append({
        "op": "union1d",
        "a_data": enc(a_set.tolist()),
        "a_dtype": "int32",
        "b_data": enc(b_set.tolist()),
        "b_dtype": "int32",
        "kw": {},
        "expected": describe(np.union1d(a_set, b_set)),
    })

    # ---------------------------------------------------------------- setdiff1d
    cases.append({
        "op": "setdiff1d",
        "a_data": enc(a_set.tolist()),
        "a_dtype": "int32",
        "b_data": enc(b_set.tolist()),
        "b_dtype": "int32",
        "kw": {},
        "expected": describe(np.setdiff1d(a_set, b_set)),
    })

    # ---------------------------------------------------------------- setxor1d
    cases.append({
        "op": "setxor1d",
        "a_data": enc(a_set.tolist()),
        "a_dtype": "int32",
        "b_data": enc(b_set.tolist()),
        "b_dtype": "int32",
        "kw": {},
        "expected": describe(np.setxor1d(a_set, b_set)),
    })

    # ---------------------------------------------------------------- isin
    a_isin  = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    b_isin  = np.array([2, 4], dtype=np.int32)
    cases.append({
        "op": "isin",
        "a_data": enc(a_isin.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_isin.shape),
        "b_data": enc(b_isin.tolist()),
        "b_dtype": "int32",
        "kw": {},
        "expected": describe(np.isin(a_isin, b_isin)),
    })

    # invert=True
    cases.append({
        "op": "isin",
        "a_data": enc(a_isin.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_isin.shape),
        "b_data": enc(b_isin.tolist()),
        "b_dtype": "int32",
        "kw": {"invert": True},
        "expected": describe(np.isin(a_isin, b_isin, invert=True)),
    })

    # 2D element array
    a_isin2d = np.array([[1, 5], [3, 4]], dtype=np.int32)
    cases.append({
        "op": "isin",
        "a_data": enc(a_isin2d.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_isin2d.shape),
        "b_data": enc(b_isin.tolist()),
        "b_dtype": "int32",
        "kw": {},
        "expected": describe(np.isin(a_isin2d, b_isin)),
    })

    # ---------------------------------------------------------------- ediff1d
    a_ed = np.array([1, 3, 6, 10, 15], dtype=np.int32)

    # basic
    cases.append({
        "op": "ediff1d",
        "a_data": enc(a_ed.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_ed.shape),
        "kw": {},
        "expected": describe(np.ediff1d(a_ed)),
    })

    # toEnd
    cases.append({
        "op": "ediff1d",
        "a_data": enc(a_ed.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_ed.shape),
        "kw": {"toEnd": [100, 200]},
        "expected": describe(np.ediff1d(a_ed, to_end=np.array([100, 200], dtype=np.int32))),
    })

    # toBegin
    cases.append({
        "op": "ediff1d",
        "a_data": enc(a_ed.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_ed.shape),
        "kw": {"toBegin": [-1]},
        "expected": describe(np.ediff1d(a_ed, to_begin=np.array([-1], dtype=np.int32))),
    })

    # toEnd and toBegin
    cases.append({
        "op": "ediff1d",
        "a_data": enc(a_ed.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_ed.shape),
        "kw": {"toBegin": [-9], "toEnd": [99]},
        "expected": describe(np.ediff1d(
            a_ed,
            to_begin=np.array([-9], dtype=np.int32),
            to_end=np.array([99], dtype=np.int32),
        )),
    })

    # 2D input (flattened)
    a_ed2d = np.array([[1, 2], [4, 7]], dtype=np.int32)
    cases.append({
        "op": "ediff1d",
        "a_data": enc(a_ed2d.tolist()),
        "a_dtype": "int32",
        "a_shape": list(a_ed2d.shape),
        "kw": {},
        "expected": describe(np.ediff1d(a_ed2d)),
    })

    return cases



def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {
        "creation": creation_cases(),
        "astype": astype_cases(),
        "views": view_cases(),
        "reshape": reshape_cases(),
        "strided_reshape": strided_reshape_cases(),
        "ranges": range_cases(),
        "shape_ops": shape_op_cases(),
        "ufuncs": ufunc_cases(),
        "math_ufuncs": math_ufunc_cases(),
        "promotion": promotion_cases(),
        "casting": casting_cases(),
        "ufunc_out": ufunc_out_cases(enc, describe),
        "ufunc_dtype_casting": ufunc_dtype_casting_cases(enc, describe),
        "ufunc_where": ufunc_where_cases(enc, describe),
        "ufunc_order": ufunc_order_cases(enc, describe),
        "indexing": index_cases(),
        "writeable": writeable_cases(),
        "reduce": reduce_cases(),
        "linalg": linalg_product_cases() + linalg_decomp_cases() + linalg_svd_qr_cases()
        + linalg_norm_cases(),
        "random": random_cases(),
        "fft": fft_cases(),
        "complex": complex_cases(),
        "complex_ufuncs": complex_ufunc_cases(),
        "complex_reductions": complex_reduction_cases(),
        "complex_matmul": complex_matmul_cases(),
        "complex_linalg": complex_linalg_cases(),
        "complex_fft": complex_fft_cases(),
        "logic_cases": logic_cases(),
        "manip_cases": manip_cases(),
        "creation2": creation2_cases(),
        "indexing2_cases": indexing2_cases(),
        "stats_cases": stats_cases(),
        "sort_cases": sort_cases(),
    }
    for name, cases in groups.items():
        payload = {"numpy_version": np.__version__, "group": name, "cases": cases}
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {len(cases):4d} cases -> tests/differential/cases/{name}.json")


if __name__ == "__main__":
    main()

