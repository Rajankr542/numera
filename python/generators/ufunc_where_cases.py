"""Differential cases for ufunc `where=` (P2-5, DECISIONS D-049).

Each case gives the operands (array data + dtype, or a JS-like scalar), the
mask (bool data, a non-bool array, a nested list or a scalar), options
(`casting`, `dtype`, and optionally an `out` spec with an initial fill) and
NumPy's result or error class.

Without `out`, NumPy leaves masked-out elements uninitialized, while numera
zeroes them (D-049). Those cases run NumPy with `out=np.zeros(...)` of the
result dtype and shape, which is exactly numera's contract.
"""
from __future__ import annotations

import warnings

import numpy as np

from cast_ub import ufunc_cast_ub

BINARY = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply,
          "divide": np.true_divide, "power": np.power, "mod": np.mod,
          "floorDivide": np.floor_divide}
UNARY = {"abs": np.abs, "negative": np.negative, "sqrt": np.sqrt, "exp": np.exp, "log": np.log}
APPROX = {"power", "sqrt", "exp", "log", "divide"}  # libm vs NumPy SIMD (D-014)
IN_DTYPES = ["bool", "int8", "uint8", "int64", "float32", "float64", "complex128"]
MASKS = [[True, False, True], [False, False, False], [True, True, True]]


def _values(dt: str, which: int) -> list:
    kind = np.dtype(dt).kind
    if kind == "b":
        return [True, False, True]
    if kind == "u":
        return [3, 1, 200] if which == 0 else [2, 1, 3]
    if kind == "i":
        return [-7, 1, 100] if which == 0 else [2, 1, 3]
    if kind == "c":
        return [complex(3, 4), complex(1, -1), complex(0.5, 0)] if which == 0 else \
               [complex(2, 0), complex(0, 1), complex(1, 1)]
    return [-7.5, 1.0, 2.25] if which == 0 else [2.0, 0.5, -3.0]


def _err(e: Exception) -> str:
    if isinstance(e, TypeError):
        return "DTypeError"
    return "BroadcastError" if "broadcast" in str(e) else "ValueError"


def _np_where(w):
    """The NumPy `where=` value for a case's mask spec."""
    if "scalar" in w:
        return w["scalar"]
    if "list" in w:
        return w["list"]
    return np.array(w["data"], dtype=w.get("dtype", "bool"))


def _np_operand(x):
    return x["scalar"] if "scalar" in x else np.array(x["data"], dtype=x["dtype"])


def ufunc_where_cases(enc, describe) -> list[dict]:
    cases: list[dict] = []

    def add(op, args, where, opts=None, out=None):
        """args: raw [{data, dtype} | {scalar}]; where: mask spec; out: {dtype, shape[, readonly]}."""
        opts = dict(opts or {})
        fn = BINARY.get(op) or UNARY[op]
        xs = [_np_operand(x) for x in args]
        case = {"op": op,
                "args": [x if "scalar" in x else {"data": enc(x["data"]), "dtype": x["dtype"]} for x in args],
                "where": {**where, "data": enc(where["data"])} if "data" in where else where,
                "opts": opts}
        if out is not None:
            case["out"] = out
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                if out is not None:
                    r = np.full(out["shape"], 7, dtype=out["dtype"])
                    if out.get("readonly"):
                        r.flags.writeable = False
                    fn(*xs, out=r, where=_np_where(where), **opts)
                else:
                    r0 = fn(*xs, where=_np_where(where), **opts)
                    # Unmasked positions are zero in numera (D-049).
                    r = np.zeros(r0.shape, dtype=r0.dtype)
                    fn(*xs, out=r, where=_np_where(where), **opts)
            except (TypeError, ValueError, OverflowError) as e:  # OverflowError -> ValueError (D-009)
                case["error"] = _err(e)
                cases.append(case)
                return
        case["approx"] = op in APPROX
        r = np.asarray(r)
        case["expected"] = describe(r)
        ub, loop = ufunc_cast_ub(fn, xs, opts, r.shape, out["dtype"] if out else None,
                                 _np_where(where))
        if ub:
            case["cast_ub"] = ub  # D-231: undefined float->int casts, not compared
        if loop is not None:
            case["loop"] = loop
        cases.append(case)

    def arr(v, dt):
        return {"data": v, "dtype": dt}

    def m(v, dt="bool"):
        return {"data": v, "dtype": dt}

    # 1) Every ufunc x input dtype x mask pattern, with and without out.
    for name in list(BINARY) + list(UNARY):
        for in_dt in IN_DTYPES:
            args = [arr(_values(in_dt, 0), in_dt)]
            if name in BINARY:
                args.append(arr(_values(in_dt, 1), in_dt))
            for mask in MASKS:
                add(name, args, m(mask))
                add(name, args, m(mask), {"casting": "unsafe"}, {"dtype": "float64", "shape": [3]})
                add(name, args, m(mask), {"casting": "unsafe"}, {"dtype": "int16", "shape": [3]})

    f3 = arr([1.5, -2.0, 3.25], "float64")
    i3 = arr([4, -5, 6], "int64")
    b3 = arr([True, False, True], "bool")
    o3 = {"dtype": "float64", "shape": [3]}
    # 2) Broadcasting: the mask expands the result; out never broadcasts.
    add("add", [f3, {"scalar": 1.0}], m([[True], [False]]))
    add("add", [f3, {"scalar": 1.0}], m([[True], [False]]), out={"dtype": "float64", "shape": [2, 3]})
    add("add", [{"scalar": 1.5}, {"scalar": 2.25}], m([[True, False, True], [False, True, False]]))
    add("negative", [i3], m([[True], [False]]))
    add("add", [arr([[1.0], [2.0]], "float64"), f3], m([True, False, True]))
    add("add", [f3, f3], m(True))
    add("add", [f3, f3], m(False), out=o3)
    add("add", [f3, f3], m([True, False]))                                   # incompatible
    add("sqrt", [f3], m([True, False]))
    add("add", [f3, f3], m([[True], [False]]), out=o3)                       # mask > out
    add("add", [f3, f3], m([True, False, True]), out={"dtype": "float64", "shape": [4]})
    # 3) Mask kinds: non-bool arrays refused; lists / scalars truthy-converted.
    for dt in ["int64", "uint8", "float64", "int8"]:
        add("add", [f3, f3], m([1, 0, 1], dt), out=o3)
    for lst in [[1.5, 0, 0.0], [1, 0, 2], [True, False, True], [[True], [False]]]:
        add("add", [f3, f3], {"list": lst})
        add("add", [f3, f3], {"list": lst},
            out={"dtype": "float64", "shape": [2, 3] if isinstance(lst[0], list) else [3]})
    for s in [True, False, 0, 1, 2.5, 0.0]:
        add("add", [f3, f3], {"scalar": s}, out=o3)
    # 4) Error order: read-only out, mask dtype, loop, input/out cast, shapes, values.
    ro = {**o3, "readonly": True}
    add("add", [f3, f3], m([1, 0, 1], "int64"), out=ro)
    add("add", [f3, f3], m([True, False]), out=ro)
    add("divide", [i3, i3], m([1, 0, 1], "int64"), {"dtype": "int8"})
    add("divide", [i3, i3], m([True, False, True]), {"dtype": "int8"})
    add("add", [f3, f3], m([1, 0, 1], "int64"), out={"dtype": "int8", "shape": [3]})
    add("add", [i3, {"scalar": 1.5}], m([True, False]), {"casting": "no"})
    add("add", [f3, f3], m([True, False]), out={"dtype": "int8", "shape": [3]})
    add("subtract", [b3, b3], m([1, 0, 1], "int64"))
    add("subtract", [b3, b3], m([True, False, True]))
    # 5) Integer power: negative exponents only matter where the mask is true.
    e = arr([1, -1, 2], "int64")
    for mask in MASKS:
        add("power", [i3, e], m(mask))
        add("power", [i3, e], m(mask), out={"dtype": "int64", "shape": [3]})
    add("power", [i3, e], m([[True, False, True], [False, False, False]]))
    # 6) dtype= / casting= together with where=.
    i8 = arr([100, 2, -3], "int8")
    for loop in ["int16", "float32", "complex128"]:
        for casting in ["same_kind", "unsafe"]:
            add("add", [i8, i8], m([True, False, True]), {"dtype": loop, "casting": casting})
            add("abs", [arr(_values("complex128", 0), "complex128")], m([False, True, True]),
                {"dtype": "float64", "casting": casting}, {"dtype": "float32", "shape": [3]})
    return cases

