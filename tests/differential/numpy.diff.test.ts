/**
 * NumPy differential tests (PLAN §63). Expected values are produced by
 * python/generators/generate_cases.py from the installed NumPy.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import np, {
  type IndexSpec,
  type NDArray,
  type NestedArray,
} from "../../packages/nativpy/src/index.js";

const casesDir = join(dirname(fileURLToPath(import.meta.url)), "cases");

type Encoded = number | boolean | { float: string } | { bigint: string } | Encoded[];

interface Described {
  dtype: string;
  shape: number[];
  strides: number[];
  c_contiguous: boolean;
  f_contiguous: boolean;
  values?: Encoded;
}

interface Case {
  op: string;
  data?: Encoded;
  dtype?: string;
  src_dtype?: string;
  shape?: number[];
  strides?: number[];
  offset?: number;
  n?: number;
  a?: string;
  b?: string;
  error?: string;
  expected?: Described | string;
  reshape_flat?: Encoded;
  reshape_shares?: boolean;
  view_shares?: boolean;
  new_shape?: number[];
  shares?: boolean;
  copy_strides?: number[];
  args?: number[];
  endpoint?: boolean;
  fill?: number | boolean;
  t?: boolean;
  arg?: unknown;
  a_data?: Encoded;
  b_data?: Encoded;
  a_dtype?: string;
  b_dtype?: string;
  a_shape?: number[];
  b_shape?: number[];
  approx?: boolean;
  scalar?: number;
  index?: unknown[];
  chain?: string[];
  writeable?: boolean;
  set_error?: string | null;
  fn?: string;
  kw?: Record<string, unknown>;
}

function load(group: string): { numpy_version: string; cases: Case[] } {
  const path = join(casesDir, `${group}.json`);
  if (!existsSync(path)) {
    throw new Error(`missing ${path}; run python/generators/generate_cases.py`);
  }
  return JSON.parse(readFileSync(path, "utf8")) as { numpy_version: string; cases: Case[] };
}

/** Decodes an input value for np.array (bigint tags become bigint). */
function decodeInput(v: Encoded): NestedArray {
  if (Array.isArray(v)) return v.map(decodeInput);
  if (typeof v === "object") {
    if ("bigint" in v) return BigInt(v.bigint);
    return decodeFloat(v.float);
  }
  return v;
}

function decodeFloat(tag: string): number {
  switch (tag) {
    case "nan":
      return NaN;
    case "inf":
      return Infinity;
    case "-inf":
      return -Infinity;
    case "-0":
      return -0;
    default:
      throw new Error(`bad float tag ${tag}`);
  }
}

/** Decodes an expected value as it appears through toArray() (D-005: numbers). */
function decodeExpected(v: Encoded): unknown {
  if (Array.isArray(v)) return v.map(decodeExpected);
  if (typeof v === "object") {
    if ("bigint" in v) return Number(BigInt(v.bigint));
    return decodeFloat(v.float);
  }
  return v;
}

function checkArray(actual: NDArray, exp: Described): void {
  expect(actual.dtype.name).toBe(exp.dtype);
  expect(actual.shape).toEqual(exp.shape);
  expect(actual.strides).toEqual(exp.strides);
  expect(actual.flags.cContiguous).toBe(exp.c_contiguous);
  expect(actual.flags.fContiguous).toBe(exp.f_contiguous);
  // toEqual distinguishes -0/+0 and treats NaN as equal to NaN.
  if (exp.values !== undefined) expect(actual.toArray()).toEqual(decodeExpected(exp.values));
}

function label(c: Case): string {
  return JSON.stringify({ ...c, expected: undefined, reshape_flat: undefined });
}

function arange(n: number): NDArray {
  return np.array(Array.from({ length: n }, (_, i) => i));
}

describe("differential: creation", () => {
  const { numpy_version, cases } = load("creation");
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const make = (): NDArray =>
      c.op === "zeros"
        ? np.zeros(c.shape!, { dtype: c.dtype! })
        : np.array(decodeInput(c.data!), c.dtype === undefined ? {} : { dtype: c.dtype });
    if (c.error !== undefined) {
      expect(make).toThrow(np.ValueError);
    } else {
      checkArray(make(), c.expected as Described);
    }
  });
});

describe("differential: astype", () => {
  const { numpy_version, cases } = load("astype");
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const src = np.array(decodeInput(c.data!), { dtype: c.src_dtype! });
    checkArray(src.astype(c.dtype!), c.expected as Described);
  });
});

describe("differential: views", () => {
  const { numpy_version, cases } = load("views");
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const base = arange(c.n!);
    const make = () => np.lib.stride_tricks.asStrided(base, c.shape!, c.strides!, c.offset!);
    if (c.error !== undefined) {
      expect(make).toThrow(np.ValueError);
      return;
    }
    const v = make();
    checkArray(v, c.expected as Described);
    expect(np.mayShareMemory(v, base)).toBe(c.view_shares);
    const flat = v.reshape(-1);
    expect(flat.toArray()).toEqual(decodeExpected(c.reshape_flat!));
    expect(np.mayShareMemory(flat, base)).toBe(c.reshape_shares);
    expect(v.copy().strides).toEqual(c.copy_strides);
  });
});

describe("differential: reshape", () => {
  const { numpy_version, cases } = load("reshape");
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const a = arange(c.n!);
    if (c.error !== undefined) {
      expect(() => a.reshape(c.shape!)).toThrow(c.error === "ValueError" ? np.ValueError : np.ShapeError);
      return;
    }
    const r = a.reshape(c.shape!);
    checkArray(r, c.expected as Described);
    expect(np.mayShareMemory(a, r)).toBe(true);
  });
});

describe("differential: reshape of strided views", () => {
  const { numpy_version, cases } = load("strided_reshape");
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const base = arange(c.n!);
    const v = np.lib.stride_tricks.asStrided(base, c.shape!, c.strides!, c.offset!);
    const r = v.reshape(c.new_shape!);
    checkArray(r, c.expected as Described);
    expect(np.mayShareMemory(r, base)).toBe(c.shares);
  });
});

describe("differential: ranges (ones/full/arange/linspace/eye)", () => {
  const { numpy_version, cases } = load("ranges");
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const opts = c.dtype === undefined ? {} : { dtype: c.dtype };
    const a = c.args ?? [];
    const make = (): NDArray => {
      switch (c.op) {
        case "ones":
          return np.ones(c.shape!, opts);
        case "full":
          return np.full(c.shape!, c.fill!, opts);
        case "arange":
          return np.arange(a[0]!, a[1], a[2], opts);
        case "linspace":
          return np.linspace(a[0]!, a[1]!, a[2]!, { ...opts, endpoint: c.endpoint! });
        case "eye":
          return np.eye(a[0]!, a[1], { ...opts, k: a[2]! });
        default:
          throw new Error(`unknown op ${c.op}`);
      }
    };
    if (c.error !== undefined) {
      expect(make).toThrow(np.ValueError);
      return;
    }
    checkArray(make(), c.expected as Described);
  });
});

describe("differential: shape ops", () => {
  const { numpy_version, cases } = load("shape_ops");
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const size = c.shape!.reduce((p, d) => p * d, 1);
    const base = arange(size).reshape(c.shape!);
    const src = c.t! ? base.T : base;
    const arg = c.arg as number[] | [number[], number[]] | null;
    const make = (): NDArray => {
      switch (c.op) {
        case "transpose":
          return np.transpose(src, arg as number[]);
        case "squeeze":
          return np.squeeze(src, arg === null ? undefined : (arg as number[]));
        case "expand_dims":
          return np.expandDims(src, arg as number[]);
        case "swapaxes":
          return np.swapAxes(src, (arg as number[])[0]!, (arg as number[])[1]!);
        case "moveaxis":
          return np.moveAxis(src, (arg as number[][])[0]!, (arg as number[][])[1]!);
        case "ravel":
          return src.ravel();
        case "flatten":
          return src.flatten();
        default:
          throw new Error(`unknown op ${c.op}`);
      }
    };
    if (c.error !== undefined) {
      expect(make).toThrow(c.error === "IndexError" ? np.IndexError : np.ValueError);
      return;
    }
    const r = make();
    checkArray(r, c.expected as Described);
    expect(np.mayShareMemory(r, base)).toBe(c.shares);
  });
});

describe("differential: ufuncs + broadcasting", () => {
  const { numpy_version, cases } = load("ufuncs");
  type Fn = (...xs: (NDArray | number)[]) => NDArray;
  const fns = np as unknown as Record<string, Fn>;
  const errorClass = (e: string): typeof np.ValueError =>
    e === "TypeError" ? np.DTypeError : np.ValueError;
  // Elementwise closeness for libm-based functions (D-014): rtol 4 ulp-ish.
  const close = (got: unknown, exp: unknown, rtol: number): void => {
    if (Array.isArray(exp)) {
      expect(Array.isArray(got)).toBe(true);
      (exp as unknown[]).forEach((e, i) => close((got as unknown[])[i], e, rtol));
      return;
    }
    const g = got as number;
    const e = exp as number;
    if (Number.isNaN(e) || !Number.isFinite(e) || Number.isInteger(e) || e === 0) {
      expect(g).toEqual(e);
    } else {
      expect(Math.abs(g - e)).toBeLessThanOrEqual(rtol * Math.abs(e));
    }
  };
  const rtolFor = (dt: string): number => (dt === "float16" ? 1e-3 : dt === "float32" ? 1e-6 : 1e-14);

  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    let make: () => NDArray;
    if (c.op === "add_bcast") {
      const na = c.a_shape!.reduce((p, d) => p * d, 1);
      const nb = c.b_shape!.reduce((p, d) => p * d, 1);
      const a = np.arange(0, na, 1, { dtype: c.dtype! }).reshape(c.a_shape!);
      const b = np.add(np.arange(0, nb, 1, { dtype: c.dtype! }), 1).reshape(c.b_shape!);
      make = () => np.add(a, b);
    } else if (c.op === "sub_transposed") {
      make = () => np.subtract(np.arange(12).astype("float64").reshape([3, 4]).T, np.arange(3.0));
    } else if (c.op === "add_scalar") {
      const a = np.array(decodeInput(c.a_data!) as NestedArray, { dtype: c.a_dtype! });
      make = () => np.add(a, c.scalar!);
    } else {
      const a = np.array(decodeInput(c.a_data!) as NestedArray, { dtype: c.a_dtype! });
      if (c.b_data === undefined) {
        make = () => fns[c.op]!(a);
      } else {
        const b = np.array(decodeInput(c.b_data) as NestedArray, { dtype: c.b_dtype! });
        make = () => fns[c.op]!(a, b);
      }
    }
    if (c.error !== undefined) {
      const expected = c.op === "add_bcast" ? np.BroadcastError : errorClass(c.error);
      expect(make).toThrow(expected);
      return;
    }
    const r = make();
    const exp = c.expected as Described;
    if (c.op === "sub_transposed") {
      // D-014: results are always C-contiguous; NumPy order='K' keeps F order.
      expect(r.dtype.name).toBe(exp.dtype);
      expect(r.shape).toEqual(exp.shape);
      expect(r.flags.cContiguous).toBe(true);
      expect(r.toArray()).toEqual(decodeExpected(exp.values!));
    } else if (c.approx) {
      expect(r.dtype.name).toBe(exp.dtype);
      expect(r.shape).toEqual(exp.shape);
      expect(r.strides).toEqual(exp.strides);
      close(r.toArray(), decodeExpected(exp.values!), rtolFor(exp.dtype));
    } else {
      checkArray(r, exp);
    }
  });
});

describe("differential: promote_types", () => {
  const { numpy_version, cases } = load("promotion");
  it.each(cases.map((c) => [`${c.a} + ${c.b}`, c] as const))(
    `numpy ${numpy_version}: %s`,
    (_l, c) => {
      expect(np.promoteTypes(c.a!, c.b!).name).toBe(c.expected);
    },
  );
});

describe("differential: indexing (M6)", () => {
  const { numpy_version, cases } = load("indexing");
  type Item =
    | "newaxis"
    | "..."
    | { i: number }
    | { s: (number | null)[] }
    | { b: boolean }
    | { arr: NestedArray; dtype: string };
  const decode = (items: Item[]): IndexSpec[] =>
    items.map((it): IndexSpec => {
      if (it === "newaxis") return np.newaxis;
      if (it === "...") return np.ellipsis;
      if ("i" in it) return it.i;
      if ("s" in it) return it.s as [number | null, number | null, number | null];
      if ("b" in it) return it.b;
      return np.array(it.arr, { dtype: it.dtype });
    });
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const n = c.shape!.reduce((p, d) => p * d, 1);
    const idx = decode(c.index as Item[]);
    let make: () => NDArray;
    let src: NDArray;
    if (c.op === "getitem") {
      src = np.arange(n).reshape(c.shape!);
      make = () => src.get(...idx);
    } else {
      src = np.arange(n).reshape(c.shape!).astype(c.dtype!);
      const v =
        c.dtype === "uint8"
          ? np.array(c.arg as NestedArray).astype("uint8")
          : (c.arg as NestedArray);
      make = () => {
        src.set(idx, v);
        return src;
      };
    }
    if (c.error !== undefined) {
      // NumPy raises ValueError for setitem broadcast mismatches; nativpy
      // raises BroadcastError (D-014).
      const cls =
        c.error === "IndexError"
          ? np.IndexError
          : c.op === "setitem"
            ? np.BroadcastError
            : np.ValueError;
      expect(make).toThrow(cls);
      return;
    }
    const r = make();
    const exp = c.expected as Described;
    expect(r.dtype.name).toBe(exp.dtype);
    expect(r.shape).toEqual(exp.shape);
    expect(r.toArray()).toEqual(decodeExpected(exp.values!));
    if (c.op === "getitem") {
      expect(np.mayShareMemory(src, r)).toBe(c.shares!);
      // D-015: basic-index views match NumPy strides exactly; advanced
      // results are fresh C-contiguous copies.
      if (c.shares) expect(r.strides).toEqual(exp.strides);
      else expect(r.flags.cContiguous).toBe(true);
    }
  });
});


describe("differential: writeable flag (D-016)", () => {
  const { numpy_version, cases } = load("writeable");
  const steps: Record<string, (a: NDArray) => NDArray> = {
    broadcast: (a) => np.broadcastTo(a, [2, ...a.shape]),
    broadcast_same: (a) => np.broadcastTo(a, a.shape),
    row: (a) => a.get(0),
    slice: (a) => a.get(np.ellipsis, [1, null, null]),
    ellipsis0d: (a) => a.get(...new Array<IndexSpec>(a.ndim).fill(0), np.ellipsis),
    fancy: (a) => a.get(np.array([0])),
    T: (a) => a.T,
    expand: (a) => np.expandDims(a, 0),
    squeeze: (a) => a.squeeze(),
    swap: (a) => a.swapAxes(0, -1),
    moveaxis: (a) => np.moveAxis(a, 0, -1),
    reshape_flat: (a) => a.reshape(-1),
    reshape_23: (a) => a.reshape(2, 3),
    ravel: (a) => a.ravel(),
    flatten: (a) => a.flatten(),
    copy: (a) => a.copy(),
    astype: (a) => a.astype("float64"),
    add: (a) => np.add(a, 1),
    strided: (a) => np.lib.stride_tricks.asStrided(a, [2], [a.itemSize * 2]),
    row_of_2d: (a) => a.reshape(2, 3).get(1),
  };
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const src = np.arange(6);
    let a = src;
    for (const s of c.chain!) a = steps[s]!(a);
    const exp = c.expected as Described;
    if (c.chain!.includes("astype")) {
      // astype on non-C inputs: NumPy order='K' keeps layout; nativpy is
      // always C-contiguous (documented in COMPATIBILITY.md).
      expect(a.shape).toEqual(exp.shape);
      expect(a.toArray()).toEqual(decodeExpected(exp.values!));
      expect(a.flags.cContiguous).toBe(true);
    } else {
      checkArray(a, exp);
    }
    expect(a.flags.writeable).toBe(c.writeable);
    const before = src.toArray();
    const write = () => a.set(new Array<IndexSpec>(a.ndim).fill(0), 99);
    if (c.set_error) {
      expect(write).toThrow(np.ValueError);
      expect(write).toThrow(c.set_error);
      expect(src.toArray()).toEqual(before);
    } else {
      write();
      expect(a.item(...new Array<number>(a.ndim).fill(0))).toBe(99);
    }
  });
});

describe("differential: reductions (M7, D-017)", () => {
  const { numpy_version, cases } = load("reduce");
  type RFn = (a: NDArray, opts: Record<string, unknown>) => NDArray;
  const fns = np as unknown as Record<string, RFn>;
  // Float sum/prod/mean/var/std: NumPy uses pairwise summation, nativpy a
  // sequential loop (D-017), so compare with a relative tolerance.
  const rtolFor = (dt: string): number => (dt === "float16" ? 2e-3 : dt === "float32" ? 1e-5 : 1e-12);
  const close = (got: unknown, exp: unknown, rtol: number): void => {
    if (Array.isArray(exp)) {
      expect(Array.isArray(got)).toBe(true);
      (exp as unknown[]).forEach((e, i) => close((got as unknown[])[i], e, rtol));
      return;
    }
    const g = got as number;
    const e = exp as number;
    if (Number.isNaN(e) || !Number.isFinite(e)) expect(g).toEqual(e);
    else expect(Math.abs(g - e)).toBeLessThanOrEqual(rtol * Math.max(Math.abs(e), 1e-300) + 1e-300);
  };

  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    let a = np
      .array(decodeInput(c.a_data!) as NestedArray, { dtype: c.a_dtype! })
      .reshape(c.a_shape!);
    if (c.t) a = a.T;
    const make = (): NDArray => fns[c.fn!]!(a, c.kw ?? {});
    if (c.error !== undefined) {
      expect(make).toThrow(c.error === "IndexError" ? np.IndexError : np.ValueError);
      return;
    }
    const r = make();
    const exp = c.expected as Described;
    if (c.t) {
      // D-017 (like D-014): results are always C-contiguous; NumPy keeps the
      // input's memory order for transposed inputs.
      expect(r.dtype.name).toBe(exp.dtype);
      expect(r.shape).toEqual(exp.shape);
      expect(r.flags.cContiguous).toBe(true);
      if (c.approx) close(r.toArray(), decodeExpected(exp.values!), rtolFor(exp.dtype));
      else expect(r.toArray()).toEqual(decodeExpected(exp.values!));
    } else if (c.approx) {
      expect(r.dtype.name).toBe(exp.dtype);
      expect(r.shape).toEqual(exp.shape);
      expect(r.strides).toEqual(exp.strides);
      close(r.toArray(), decodeExpected(exp.values!), rtolFor(exp.dtype));
    } else {
      checkArray(r, exp);
    }
  });
});

