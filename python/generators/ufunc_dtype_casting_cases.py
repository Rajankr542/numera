"""Differential cases for ufunc `dtype=` / `casting=` (P2-4, DECISIONS D-048).

Each case gives the operands (array data + dtype, or a JS-like scalar), the
options (`dtype`, `casting`, optionally an `out` dtype) and NumPy's result:
either the result array or the expected error class.
"""
from __future__ import annotations

import warnings

import numpy as np

BINARY = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply,
          "divide": np.true_divide, "power": np.power, "mod": np.mod,
          "floorDivide": np.floor_divide}
UNARY = {"abs": np.abs, "negative": np.negative, "sqrt": np.sqrt, "exp": np.exp, "log": np.log}
APPROX = {"power", "sqrt", "exp", "log", "divide"}  # libm vs NumPy SIMD (D-014)
CASTINGS = ["no", "equiv", "safe", "same_kind", "unsafe"]
LOOP_DTYPES = ["bool", "int8", "uint8", "int16", "int64", "uint64",
               "float16", "float32", "float64", "complex64", "complex128"]
IN_DTYPES = ["bool", "int8", "uint8", "int64", "float32", "float64", "complex128"]


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
    return "DTypeError" if isinstance(e, TypeError) else "ValueError"


def ufunc_dtype_casting_cases(enc, describe) -> list[dict]:
    cases: list[dict] = []

    def add(op, args, opts, run):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                r = run()
            except (TypeError, ValueError, OverflowError) as e:  # OverflowError -> ValueError (D-009)
                cases.append({"op": op, "args": args, "opts": opts, "error": _err(e)})
                return
        cases.append({"op": op, "args": args, "opts": opts, "approx": op in APPROX,
                      "expected": describe(np.asarray(r))})

    def arr(v, dt):
        return {"data": enc(v), "dtype": dt}

    # 1) dtype= over every loop dtype, input dtype and ufunc (unsafe input casts
    #    so most loops run; same_kind to exercise the input-cast check).
    for name in list(BINARY) + list(UNARY):
        for in_dt in IN_DTYPES:
            for loop in LOOP_DTYPES:
                for casting in ["same_kind", "unsafe"]:
                    av = _values(in_dt, 0)
                    a = np.array(av, dtype=in_dt)
                    opts = {"dtype": loop, "casting": casting}
                    if name in UNARY:
                        add(name, [arr(av, in_dt)], opts,
                            lambda f=UNARY[name], a=a, d=loop, c=casting: f(a, dtype=d, casting=c))
                    else:
                        bv = _values(in_dt, 1)
                        b = np.array(bv, dtype=in_dt)
                        add(name, [arr(av, in_dt), arr(bv, in_dt)], opts,
                            lambda f=BINARY[name], a=a, b=b, d=loop, c=casting:
                            f(a, b, dtype=d, casting=c))

    # 2) casting= alone over mixed input dtypes (input-cast check of D-014 loops).
    pairs = [("int8", "int8"), ("int8", "int16"), ("uint8", "int8"), ("int64", "float64"),
             ("float32", "float64"), ("bool", "bool"), ("int64", "complex128")]
    for name in ["add", "divide", "power"]:
        for x, y in pairs:
            for casting in CASTINGS:
                av, bv = _values(x, 0), _values(y, 1)
                a, b = np.array(av, dtype=x), np.array(bv, dtype=y)
                add(name, [arr(av, x), arr(bv, y)], {"casting": casting},
                    lambda f=BINARY[name], a=a, b=b, c=casting: f(a, b, casting=c))
    for name in ["sqrt", "abs", "negative"]:
        for x in IN_DTYPES:
            for casting in CASTINGS:
                av = _values(x, 0)
                a = np.array(av, dtype=x)
                add(name, [arr(av, x)], {"casting": casting},
                    lambda f=UNARY[name], a=a, c=casting: f(a, casting=c))

    # 3) casting= with out (output-cast check) and dtype= + out.
    for loop_dt in ["int8", "int64", "float64"]:
        for out_dt in ["int8", "int64", "float32", "float64", "complex128"]:
            for casting in CASTINGS:
                for with_dtype in [False, True]:
                    av, bv = _values("int8", 0), _values("int8", 1)
                    a, b = np.array(av, dtype="int8"), np.array(bv, dtype="int8")
                    opts = {"casting": casting, "out": out_dt}
                    if with_dtype:
                        opts["dtype"] = loop_dt
                    elif loop_dt != "int8":
                        continue

                    def run(a=a, b=b, o=out_dt, c=casting, d=opts.get("dtype")):
                        out = np.zeros(3, dtype=o)
                        np.add(a, b, out=out, casting=c, dtype=d)
                        return out
                    add("add", [arr(av, "int8"), arr(bv, "int8")], opts, run)

    # 4) JS-like scalars are weak relative to dtype=.
    for scalar in [1000, 1.5, True, -3]:
        for loop in ["int8", "int16", "uint8", "float32", "float64"]:
            for casting in ["same_kind", "unsafe"]:
                av = _values("int8", 0)
                a = np.array(av, dtype="int8")
                add("add", [arr(av, "int8"), {"scalar": scalar}], {"dtype": loop, "casting": casting},
                    lambda a=a, s=scalar, d=loop, c=casting: np.add(a, s, dtype=d, casting=c))
    return cases
