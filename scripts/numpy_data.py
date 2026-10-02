"""Extracts NumPy one-line summaries and documentation URLs for every name in
the numera API inventory (api/numpy-api.json of the main repository).

    python3 scripts/numpy_data.py <numera checkout> > data/numpy-2.5.3.json

Summaries come from the installed NumPy's docstrings (first paragraph). URLs
come from NumPy's Sphinx inventory (objects.inv) for the matching minor
version, so every link points at an existing page. NumPy documentation is
BSD-3-Clause licensed; the site footer credits it.
"""

import inspect
import json
import re
import sys
import urllib.request
import zlib

import numpy as np
import numpy.char

import numpy.fft
import numpy.linalg
import numpy.ma
import numpy.polynomial
import numpy.random
import numpy.rec
import numpy.strings
import numpy.testing

repo = sys.argv[1]
inventory = json.load(open(f"{repo}/api/numpy-api.json"))
version = inventory["numpy_version"]
minor = ".".join(version.split(".")[:2])
base = f"https://numpy.org/doc/{minor}/"

modules = {
    "np": ("numpy", np),
    "linalg": ("numpy.linalg", np.linalg),
    "fft": ("numpy.fft", np.fft),
    "random": ("numpy.random", np.random),
    "Generator": ("numpy.random.Generator", np.random.Generator),
    "RandomState": ("numpy.random.RandomState", np.random.RandomState),
    "ndarray": ("numpy.ndarray", np.ndarray),
    "ma": ("numpy.ma", np.ma),
    "polynomial": ("numpy.polynomial", np.polynomial),
    "strings": ("numpy.strings", np.strings),
    "char": ("numpy.char", np.char),
    "rec": ("numpy.rec", np.rec),
    "emath": ("numpy.emath", np.emath),
    "testing": ("numpy.testing", np.testing),
}


def load_inventory():
    req = urllib.request.Request(base + "objects.inv", headers={"User-Agent": "Mozilla/5.0 (numera-docs)"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    lines = raw.split(b"\n", 4)
    body = zlib.decompress(lines[4]).decode("utf8")
    urls = {}
    for line in body.splitlines():
        m = re.match(r"(\S+)\s+(py:\w+)\s+-?\d+\s+(\S+)\s+(.*)", line)
        if not m:
            continue
        name, role, uri, _ = m.groups()
        if uri.endswith("$"):
            uri = uri[:-1] + name
        # Prefer the object's own page over module-level mentions.
        if name not in urls or role in ("py:function", "py:method", "py:class", "py:data", "py:attribute"):
            urls.setdefault(name, base + uri)
    return urls


def summary(obj):
    doc = getattr(obj, "__doc__", None)
    if not isinstance(doc, str) or not doc.strip():
        return ""
    doc = inspect.cleandoc(doc)
    for para in re.split(r"\n\s*\n", doc):
        text = " ".join(para.split())
        if not text:
            continue
        # Skip signature lines such as "add(x1, x2, /, out=None, ...)".
        if re.match(r"^[\w.]+\(.*\)(\s*->.*)?$", text):
            continue
        if text.startswith("--"):
            continue
        if re.match(r"^(Parameters|Returns|See Also|Notes|Examples)\s*-{3,}", text):
            break
        return text[:400]
    return ""


urls = load_inventory()
out = {"numpy_version": version, "docs": base, "surfaces": {}}
for surface, names in inventory["surfaces"].items():
    qual, mod = modules[surface]
    entries = {}
    for name in names:
        obj = getattr(mod, name, None)
        full = f"{qual}.{name}"
        url = urls.get(full)
        if url is None:
            # Constants, aliases, re-exports and classes documented elsewhere.
            target = getattr(obj, "__name__", name) if not isinstance(obj, (int, float, bool, str)) else name
            mod_of = getattr(obj, "__module__", "") or ""
            for cand in (f"numpy.{target}", f"{mod_of}.{target}", f"numpy.{name}",
                         f"numpy.polynomial.{name.lower()}.{name}", f"numpy.polynomial.hermite_e.{name}"):
                if cand in urls:
                    url = urls[cand]
                    break
        entries[name] = {"summary": summary(obj) if obj is not None and not isinstance(obj, (int, float, bool, str, tuple, dict)) else "",
                         "url": url or ""}
    out["surfaces"][surface] = entries
json.dump(out, sys.stdout, indent=1, ensure_ascii=False)
sys.stdout.write("\n")
