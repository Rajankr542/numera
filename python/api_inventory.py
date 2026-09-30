#!/usr/bin/env python3
"""Writes api/numpy-api.json: NumPy's public callable surface (D-032, PLAN §37).

Run: python3 python/api_inventory.py   (re-run only when the pinned NumPy changes)
"""
import inspect
import json
import types
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "api" / "numpy-api.json"


def kind_of(o) -> str:
    if isinstance(o, types.ModuleType):
        return "module"
    if isinstance(o, np.ufunc):
        return "ufunc"
    if inspect.isclass(o):
        return "class"
    if callable(o):
        return "function"
    return "constant"


def module_surface(mod) -> dict:
    out = {}
    for name in sorted(dir(mod)):
        if name.startswith("_"):
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                obj = getattr(mod, name)
        except Exception:
            continue
        k = kind_of(obj)
        if k == "module":
            continue
        out[name] = k
    return out


def class_surface(cls) -> dict:
    out = {}
    for name in sorted(dir(cls)):
        if name.startswith("_"):
            continue
        attr = inspect.getattr_static(cls, name)
        out[name] = "method" if callable(attr) or isinstance(attr, (staticmethod, classmethod)) else "attribute"
    return out


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        surfaces = {
            "np": module_surface(np),
            "linalg": module_surface(np.linalg),
            "fft": module_surface(np.fft),
            "random": module_surface(np.random),
            "Generator": class_surface(np.random.Generator),
            "RandomState": class_surface(np.random.RandomState),
            "ndarray": class_surface(np.ndarray),
            "ma": module_surface(np.ma),
            "polynomial": module_surface(np.polynomial),
            "strings": module_surface(np.strings),
            "char": module_surface(np.char),
            "rec": module_surface(np.rec),
            "emath": module_surface(np.emath),
            "testing": module_surface(np.testing),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = {"numpy_version": np.__version__, "surfaces": surfaces}
    OUT.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    total = sum(len(v) for v in surfaces.values())
    print(f"wrote {OUT.relative_to(ROOT)}: numpy {np.__version__}, {total} names")


if __name__ == "__main__":
    main()
