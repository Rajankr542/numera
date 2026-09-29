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
        "promotion": promotion_cases(),
        "indexing": index_cases(),
        "writeable": writeable_cases(),
    }
    for name, cases in groups.items():
        payload = {"numpy_version": np.__version__, "group": name, "cases": cases}
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {len(cases):4d} cases -> tests/differential/cases/{name}.json")


if __name__ == "__main__":
    main()

