"""Differential cases for ufunc `out=` (P2-3, DECISIONS D-046/D-047).

Each case describes the operands, the `out` array (dtype, shape, an optional
strided/overlapping view of a base buffer) and NumPy's result: either the
final contents of the whole base buffer, or the expected error class.
"""
from __future__ import annotations

import warnings

import numpy as np


BINARY = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply,
          "divide": np.true_divide, "power": np.power, "mod": np.mod,
          "floorDivide": np.floor_divide}
UNARY = {"abs": np.abs, "negative": np.negative, "sqrt": np.sqrt, "exp": np.exp, "log": np.log}
APPROX = {"power", "sqrt", "exp", "log"}  # libm vs NumPy SIMD (D-014)
OUT_DTYPES = ["bool", "int8", "uint8", "int16", "int32", "int64", "uint64",
              "float16", "float32", "float64", "complex128"]


def _err(e: Exception) -> str:
    if isinstance(e, TypeError):
        return "DTypeError"
    msg = str(e)
    if "broadcast" in msg:  # D-014/D-046: shape errors are BroadcastError
        return "BroadcastError"
    return "ValueError"


def _values(dt: str, which: int) -> list:
    kind = np.dtype(dt).kind
    if kind == "b":
        return [True, False, True, True]
    if kind == "u":
        return [3, 0, 7, 200] if which == 0 else [2, 1, 3, 4]
    if kind == "i":
        return [-7, 0, 5, 100] if which == 0 else [2, 1, 3, 2]
    return [-7.5, 0.0, 2.25, float("nan")] if which == 0 else [2.0, 0.5, -3.0, 4.0]


def _case(enc, describe, op, args, out_spec, run):
    """args: list of {data, dtype} or {scalar}; out_spec: dict for the TS side."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            base = run()
        except (TypeError, ValueError) as e:
            return {"op": op, "args": args, "out": out_spec, "error": _err(e)}
    case = {"op": op, "args": args, "out": out_spec, "approx": op in APPROX,
            "expected": describe(base)}
    if all("dtype" in a or "scalar" in a for a in args):
        # Loop dtype from one-element operands (shape-independent).
        xs = [a["scalar"] if "scalar" in a else np.ones(1, dtype=a["dtype"]) for a in args]
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            fn = BINARY.get(op) or UNARY[op]
            case["loop"] = np.asarray(fn(*xs)).dtype.name
    return case


def ufunc_out_cases(enc, describe) -> list[dict]:
    cases = []

    def arr(data, dt):
        return {"data": enc(data), "dtype": dt}

    # 1) Loop dtype x out dtype (same_kind output casting), same shape.
    for in_dt in ["bool", "int8", "uint8", "int64", "float32", "float64"]:
        for out_dt in OUT_DTYPES:
            for name in ["add", "divide", "abs", "sqrt"]:
                av = _values(in_dt, 0)
                a = np.array(av, dtype=in_dt)
                spec = {"dtype": out_dt, "n": 4, "init": "zeros", "view": {"shape": [4], "strides": [1], "offset": 0}}
                if name in UNARY:
                    def run(fn=UNARY[name], a=a, out_dt=out_dt):
                        out = np.zeros(4, dtype=out_dt)
                        fn(a, out=out)
                        return out
                    cases.append(_case(enc, describe, name, [arr(av, in_dt)], spec, run))
                else:
                    bv = _values(in_dt, 1)
                    b = np.array(bv, dtype=in_dt)

                    def run(fn=BINARY[name], a=a, b=b, out_dt=out_dt):
                        out = np.zeros(4, dtype=out_dt)
                        fn(a, b, out=out)
                        return out
                    cases.append(_case(enc, describe, name, [arr(av, in_dt), arr(bv, in_dt)], spec, run))
    # 2) Every op into a float64 and a narrower out.
    for name, fn in {**BINARY, **UNARY}.items():
        for in_dt, out_dt in [("int16", "float64"), ("float64", "float32"), ("int32", "int64"),
                              ("int8", "int16")]:
            av, bv = _values(in_dt, 0), _values(in_dt, 1)
            a, b = np.array(av, dtype=in_dt), np.array(bv, dtype=in_dt)
            spec = {"dtype": out_dt, "n": 4, "init": "zeros", "view": {"shape": [4], "strides": [1], "offset": 0}}
            if name in UNARY:
                def run(fn=fn, a=a, out_dt=out_dt):
                    out = np.zeros(4, dtype=out_dt)
                    fn(a, out=out)
                    return out
                cases.append(_case(enc, describe, name, [arr(av, in_dt)], spec, run))
            else:
                def run(fn=fn, a=a, b=b, out_dt=out_dt):
                    out = np.zeros(4, dtype=out_dt)
                    fn(a, b, out=out)
                    return out
                cases.append(_case(enc, describe, name, [arr(av, in_dt), arr(bv, in_dt)], spec, run))
    cases += _layout_cases(enc, describe)
    return cases


def _view(base: np.ndarray, v: dict) -> np.ndarray:
    """A view of `base` from an element-unit {shape, strides, offset} spec."""
    isz = base.itemsize
    return np.lib.stride_tricks.as_strided(
        base[v["offset"]:], shape=v["shape"], strides=[s * isz for s in v["strides"]])


def _layout_cases(enc, describe) -> list[dict]:
    """Broadcast, scalar, strided, overlapping, read-only and shape-error cases.

    `out` spec: {dtype, n, init: "zeros"|"arange", view, readonly?}. The base is
    a 1-D array of n elements; `view` selects `out` from it. An arg {"view": v}
    is another view of the same base (overlap). Expected: the whole base.
    """
    cases = []

    def add(op, args, out):
        fn = BINARY.get(op) or UNARY[op]

        def make(a):
            return (np.arange(int(np.prod(a["shape"]))) % 3 + 1).astype(a["dtype"]).reshape(a["shape"])

        def run():
            base = (np.arange(out["n"]) if out["init"] == "arange" else np.zeros(out["n"])).astype(out["dtype"])
            o = _view(base, out["view"])
            if out.get("readonly"):
                o = np.broadcast_to(base[:1], out["view"]["shape"])
            xs = []
            for a in args:
                if "view" in a:
                    xs.append(_view(base, a["view"]))
                elif "scalar" in a:
                    xs.append(a["scalar"])
                else:
                    xs.append(make(a))
            fn(*xs, out=o)
            return base
        enc_args = [{**a, "data": enc(make(a).tolist())} if "shape" in a else a for a in args]
        cases.append(_case(enc, describe, op, enc_args, out, run))

    def lin(shape):  # C-contiguous element strides
        st, acc = [], 1
        for d in reversed(shape):
            st.insert(0, acc)
            acc *= d
        return st

    def plain(n, shape, dtype="float64", init="zeros", strides=None, offset=0, readonly=False):
        o = {"dtype": dtype, "n": n, "init": init,
             "view": {"shape": shape, "strides": strides or lin(shape), "offset": offset}}
        if readonly:
            o["readonly"] = True
        return o

    def gen(shape, dtype="float64"):
        return {"shape": shape, "dtype": dtype}

    # Broadcast inputs into a larger out; out never broadcasts.
    add("add", [gen([3, 1]), gen([4])], plain(12, [3, 4]))
    add("add", [gen([4]), gen([])], plain(12, [3, 4]))
    add("multiply", [gen([1]), gen([1])], plain(5, [5]))
    add("negative", [gen([3])], plain(6, [2, 3]))
    add("add", [gen([2, 3]), gen([3])], plain(3, [3]))           # out smaller
    add("add", [gen([3]), gen([])], plain(4, [4]))               # incompatible
    add("negative", [gen([2, 3])], plain(3, [3]))
    add("add", [gen([]), gen([])], plain(1, []))                 # 0-d out
    add("add", [gen([0, 3]), gen([3])], plain(0, [0, 3]))        # empty
    # JS scalars stay weak relative to the array operand (NEP 50).
    add("add", [gen([3], "int8"), {"scalar": 2}], plain(3, [3], "int16"))
    add("add", [gen([3], "float32"), {"scalar": 0.1}], plain(3, [3], "float64"))
    add("add", [gen([3], "int8"), {"scalar": 2.5}], plain(3, [3], "int64"))
    add("multiply", [{"scalar": 3}, gen([3], "uint8")], plain(3, [3], "float32"))
    # Strided / reversed / transposed / zero-stride out.
    add("add", [gen([3]), {"scalar": 2.0}], plain(6, [3], strides=[2]))
    add("add", [gen([3]), {"scalar": 2.0}], plain(6, [3], strides=[-2], offset=5))
    add("subtract", [gen([3, 2]), gen([2])], plain(6, [3, 2], strides=[1, 3]))
    add("sqrt", [gen([2, 2], "int32")], plain(8, [2, 2], "float32", strides=[4, 1], offset=1))
    add("add", [gen([3]), {"scalar": 1.0}], plain(1, [3], strides=[0]))
    # Overlap between inputs and out (results as if inputs were copied).
    a6 = lambda s, st, off: {"view": {"shape": s, "strides": st, "offset": off}}  # noqa: E731
    add("add", [a6([6], [1], 0), a6([6], [1], 0)], plain(6, [6], init="arange"))  # in-place
    add("add", [a6([5], [1], 0), a6([5], [1], 1)], plain(6, [5], init="arange", offset=1))
    add("add", [a6([5], [1], 1), a6([5], [1], 0)], plain(6, [5], init="arange"))
    add("negative", [a6([6], [-1], 5)], plain(6, [6], init="arange"))
    add("multiply", [a6([3], [2], 0), a6([3], [2], 1)], plain(6, [3], init="arange"))
    add("add", [a6([1], [1], 0), a6([4], [1], 0)], plain(4, [4], init="arange"))
    add("power", [a6([4], [1], 0), {"scalar": 2}], plain(4, [4], "int64", init="arange"))
    add("floorDivide", [a6([3, 2], [1, 3], 0), {"scalar": 2}], plain(6, [3, 2], "int64", init="arange"))
    add("add", [a6([3], [1], 0), {"scalar": 0.5}], plain(6, [3], "int64", init="arange", offset=1))
    # Read-only out, and error precedence (read-only before cast/shape).
    add("add", [gen([3]), {"scalar": 1.0}], plain(3, [3], readonly=True))
    add("add", [gen([2, 3]), {"scalar": 1.5}], plain(3, [3], "int64", readonly=True))
    add("add", [gen([2, 3]), {"scalar": 1.5}], plain(3, [3], "int64"))  # cast before shape
    add("sqrt", [gen([3])], plain(3, [3], readonly=True))
    add("power", [gen([3], "int64"), {"scalar": -1}], plain(3, [3], "bool"))  # cast before value
    add("power", [gen([3], "int64"), {"scalar": -1}], plain(3, [3], "float64"))
    add("subtract", [gen([2], "bool"), gen([2], "bool")], plain(2, [2], "int64"))
    add("negative", [gen([2], "bool")], plain(2, [2], "int8"))
    return cases
