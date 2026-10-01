"""Differential cases for ufunc `order=` and result layout (P2-6, DECISIONS D-050).

Each array operand is a strided view {shape, strides (elements), offset} of
a fresh `arange(n)` base of its dtype, so TS rebuilds it exactly with
`asStrided`. A `where` mask is the same kind of view of `arange(n) % 3 != 0`.
The expected result carries NumPy's strides and contiguity flags.
"""
from __future__ import annotations

import random
import warnings

import numpy as np

BINARY = {"add": np.add, "subtract": np.subtract, "multiply": np.multiply,
          "divide": np.true_divide, "power": np.power}
# dot/inner with a 0-d operand reuse multiply (D-050 linalg note); no order=.
LINALG = {"dot": np.dot, "inner": np.inner}
UNARY = {"negative": np.negative, "abs": np.abs, "sqrt": np.sqrt, "exp": np.exp}
APPROX = {"divide", "power", "sqrt", "exp"}  # libm vs NumPy SIMD (D-014)
ORDERS = ["C", "F", "A", "K"]


def _spec(rng: random.Random) -> dict:
    """A random layout: C/F/transposed/reversed/strided/size-1/0-d/zero-size."""
    nd = rng.randint(0, 3)
    shape = [rng.choice([1, 2, 3, 4]) for _ in range(nd)]
    if nd and rng.random() < 0.05:
        shape[rng.randrange(nd)] = 0
    step = [rng.choice([1, 1, 1, 2]) for _ in range(nd)]
    # Dense layout over the stepped extents, then permute the axes.
    perm = list(range(nd))
    if rng.random() < 0.6:
        rng.shuffle(perm)
    strides = [0] * nd
    acc = 1
    for ax in reversed(perm):
        strides[ax] = acc * step[ax]
        acc *= max(shape[ax], 1) * step[ax]
    offset = 0
    for ax in range(nd):
        if rng.random() < 0.25 and shape[ax] > 1:  # reverse the axis
            offset += (shape[ax] - 1) * strides[ax]
            strides[ax] = -strides[ax]
    return {"n": max(acc, 1), "shape": shape, "strides": strides, "offset": offset}


def _make(spec: dict, dtype: str, mask: bool = False) -> np.ndarray:
    base = np.arange(spec["n"])
    base = (base % 3 != 0) if mask else base.astype(dtype)
    it = base.itemsize
    return np.lib.stride_tricks.as_strided(
        base[spec["offset"]:], spec["shape"], [s * it for s in spec["strides"]])


def ufunc_order_cases(enc, describe) -> list[dict]:
    cases: list[dict] = []

    def add(op, args, opts, where=None):
        """args: [{dtype, spec} | {scalar}]; where: spec or None."""
        fn = BINARY.get(op) or LINALG.get(op) or UNARY[op]
        xs = [a["scalar"] if "scalar" in a else _make(a["spec"], a["dtype"]) for a in args]
        case = {"op": op, "args": args, "opts": dict(opts)}
        kw = dict(opts)
        if where is not None:
            case["where"] = where
            kw["where"] = _make(where, "bool", mask=True)
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            try:
                r = fn(*xs, **kw)
                if where is not None:  # numera zeroes masked-out positions (D-049)
                    if np.ndim(r) == 0:  # NumPy returns a scalar for 0-d results
                        r = np.array(r if np.asarray(kw["where"]) else 0, dtype=np.asarray(r).dtype)
                    else:
                        np.copyto(r, 0, where=~np.asarray(kw["where"]))
            except (TypeError, ValueError) as e:
                case["error"] = "DTypeError" if isinstance(e, TypeError) else (
                    "BroadcastError" if "broadcast" in str(e) else "ValueError")
                cases.append(case)
                return
        case["approx"] = op in APPROX
        case["expected"] = describe(np.asarray(r))
        cases.append(case)

    def fits(*specs):
        try:
            np.broadcast_shapes(*[tuple(s["shape"]) for s in specs])
            return True
        except ValueError:
            return False

    rng = random.Random(2026)
    for _ in range(160):  # binary, random layouts (some with a dtype cast)
        dt = rng.choice(["float64", "int32", "float32"])
        a, b = _spec(rng), _spec(rng)
        if not fits(a, b):
            continue
        op = rng.choice(list(BINARY))
        args = [{"dtype": dt, "spec": a}, {"dtype": rng.choice([dt, dt, "int16"]), "spec": b}]
        for o in ORDERS:
            add(op, args, {"order": o})
    for _ in range(60):  # unary
        a = _spec(rng)
        op = rng.choice(list(UNARY))
        for o in ORDERS:
            add(op, [{"dtype": rng.choice(["float64", "int8"]), "spec": a}], {"order": o})
    for _ in range(30):  # dtype= casts every input
        a, b = _spec(rng), _spec(rng)
        if not fits(a, b):
            continue
        for o in ORDERS:
            add("add", [{"dtype": "int16", "spec": a}, {"dtype": "float32", "spec": b}],
                {"order": o, "dtype": "float64"})
    for _ in range(40):  # where= votes in A/K and disables the trivial path
        a, w = _spec(rng), _spec(rng)
        if not fits(a, w):
            continue
        for o in ORDERS:
            add("multiply", [{"dtype": "float64", "spec": a}, {"scalar": 2.0}], {"order": o}, where=w)
    # Fixed cases: F/C inputs, mixed, scalars, broadcasting, both-flags, empty.
    F = {"n": 6, "shape": [2, 3], "strides": [1, 2], "offset": 0}
    C = {"n": 6, "shape": [2, 3], "strides": [3, 1], "offset": 0}
    row = {"n": 3, "shape": [3], "strides": [1], "offset": 0}
    col = {"n": 6, "shape": [6, 1], "strides": [1, 1], "offset": 0}
    empty = {"n": 1, "shape": [2, 0], "strides": [0, 0], "offset": 0}
    f64 = lambda s: {"dtype": "float64", "spec": s}  # noqa: E731
    for o in ORDERS + ["k", "f"]:
        add("add", [f64(F), {"scalar": 1.0}], {"order": o})
        add("add", [f64(C), f64(F)], {"order": o})
        add("add", [f64(F), f64(row)], {"order": o})
        add("add", [f64(col), f64(row)], {"order": o})
        add("negative", [f64(empty)], {"order": o})
        add("add", [{"scalar": 1.5}, {"scalar": 2.5}], {"order": o})
    for bad in ["X", "", "KK", "Fortran"]:
        add("add", [f64(C), {"scalar": 1.0}], {"order": bad})
        add("negative", [f64(C)], {"order": bad})
    # dot/inner with a scalar: C for BLAS dtypes with ndim <= 2, else K.
    for _ in range(40):
        a = _spec(rng)
        if not a["shape"]:
            continue
        fn = rng.choice(["dot", "inner"])
        s = {"scalar": rng.choice([2.5, 3])}
        x = {"dtype": rng.choice(["float64", "float32", "int32", "int8"]), "spec": a}
        add(fn, [x, s] if rng.random() < 0.5 else [s, x], {})
    return cases
