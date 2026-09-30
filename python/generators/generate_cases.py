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


# ---- reductions (M7, D-017) ----
REDUCE_FNS = {
    "sum": np.sum, "prod": np.prod, "min": np.min, "max": np.max, "mean": np.mean,
    "var": np.var, "std": np.std, "argmin": np.argmin, "argmax": np.argmax,
}
# Float results of these depend on summation order (NumPy is pairwise, nativpy
# sequential; D-017) so they are compared with a relative tolerance.
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
        "reduce": reduce_cases(),
        "linalg": linalg_product_cases() + linalg_decomp_cases() + linalg_svd_qr_cases()
        + linalg_norm_cases(),
        "random": random_cases(),
        "fft": fft_cases(),
    }
    for name, cases in groups.items():
        payload = {"numpy_version": np.__version__, "group": name, "cases": cases}
        (OUT / f"{name}.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"wrote {len(cases):4d} cases -> tests/differential/cases/{name}.json")


if __name__ == "__main__":
    main()

