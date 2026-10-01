/**
 * NumPy differential tests (PLAN §63). Expected values are produced by
 * python/generators/generate_cases.py from the installed NumPy.
 */
import { existsSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import np, {
  type Casting,
  Complex,
  type IndexSpec,
  type NDArray,
  type NestedArray,
} from "../../packages/numera/src/index.js";

const casesDir = join(dirname(fileURLToPath(import.meta.url)), "cases");

type Encoded =
  | number
  | boolean
  | { float: string }
  | { bigint: string }
  | { re: Encoded; im: Encoded }
  | Encoded[];

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
  zero_rule?: boolean;
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
    if ("re" in v) return np.complex(decodeExpected(v.re) as number, decodeExpected(v.im) as number);
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
    if ("re" in v) return np.complex(decodeExpected(v.re) as number, decodeExpected(v.im) as number);
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
  const s = JSON.stringify({ ...c, expected: undefined, reshape_flat: undefined });
  if (s.length <= 400) return s;
  // Large generated inputs (e.g. D-021 cases): keep test names readable but
  // unique (FNV-1a hash of the full label).
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 0x01000193) >>> 0;
  return `${s.slice(0, 360)}…#${h.toString(16)}`;
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

describe("differential: can_cast (D-045)", () => {
  const { numpy_version, cases } = load("casting");
  const label = (c: Case): string =>
    `${c.op} ${c.a} -> ${c.b}${c.arg === undefined ? "" : ` (${String(c.arg)})`}`;
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    const from = c.op === "can_cast_array" ? np.array(decodeInput(c.data!), { dtype: c.a! }) : c.a!;
    const run = (): boolean =>
      c.arg === undefined ? np.canCast(from, c.b!) : np.canCast(from, c.b!, c.arg as Casting);
    if (c.error !== undefined) {
      expect(run).toThrow(c.error === "ValueError" ? np.ValueError : np.DTypeError);
      return;
    }
    expect(run()).toBe(c.expected);
  });
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

describe.each([
  ["reduce", "reductions (M7, D-017)"],
  ["complex_reductions", "complex reductions (P1 step 3, D-034)"],
])("differential: %s", (group, _title) => {
  const { numpy_version, cases } = load(group);
  type RFn = (a: NDArray, opts: Record<string, unknown>) => NDArray;
  const fns = np as unknown as Record<string, RFn>;
  // Float sum/prod/mean/var/std on the small generic cases are compared with a
  // relative tolerance (D-017); D-021 cases (approx=false) are checked exactly.
  const rtolFor = (dt: string): number =>
    dt === "float16" ? 2e-3 : dt === "float32" || dt === "complex64" ? 1e-5 : 1e-12;
  const close = (got: unknown, exp: unknown, rtol: number): void => {
    if (Array.isArray(exp)) {
      expect(Array.isArray(got)).toBe(true);
      (exp as unknown[]).forEach((e, i) => close((got as unknown[])[i], e, rtol));
      return;
    }
    if (exp instanceof Complex) {
      expect(got).toBeInstanceOf(Complex);
      close((got as Complex).re, exp.re, rtol);
      close((got as Complex).im, exp.im, rtol);
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
    if (c.zero_rule) {
      // D-025: NumPy's sign here is SIMD-path dependent; assert the D-017 rule.
      expect(r.dtype.name).toBe(exp.dtype);
      expect(r.shape).toEqual(exp.shape);
      expect(r.toArray()).toEqual(c.fn === "min" ? -0 : 0);
      return;
    }
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


describe("differential: linalg (M8, D-018)", () => {
  const { numpy_version, cases } = load("linalg");
  interface LCase extends Case {
    inputs: { data: Encoded; dtype: string; shape: number[] }[];
  }
  type Ex = Described & { imag?: Encoded };
  const flat = (v: unknown): number[] =>
    Array.isArray(v) ? v.flatMap(flat) : [typeof v === "boolean" ? Number(v) : (v as number)];
  const tolFor = (dt: string): number =>
    dt.includes("16") ? 3e-3 : dt === "float32" || dt === "complex64" ? 2e-5 : 1e-10;
  /** Elementwise |got - exp| <= tol * max(1, max|exp|). */
  const closeFlat = (got: number[], exp: number[], tol: number): void => {
    expect(got.length).toBe(exp.length);
    const scale = Math.max(1, ...exp.map((x) => (Number.isFinite(x) ? Math.abs(x) : 0)));
    exp.forEach((e, i) => {
      if (!Number.isFinite(e)) expect(got[i]).toEqual(e);
      else expect(Math.abs(got[i]! - e)).toBeLessThanOrEqual(tol * scale);
    });
  };
  const meta = (r: NDArray, exp: Ex): void => {
    expect(r.dtype.name).toBe(exp.dtype);
    expect(r.shape).toEqual(exp.shape);
    expect(r.flags.cContiguous).toBe(true);
  };
  const approx = (r: NDArray, exp: Ex): void => {
    meta(r, exp);
    closeFlat(flat(r.toArray()), flat(decodeExpected(exp.values!)), tolFor(exp.dtype));
  };
  const exact = (r: NDArray, exp: Ex): void => {
    meta(r, exp);
    expect(r.toArray()).toEqual(decodeExpected(exp.values!));
  };
  const isFloat = (dt: string): boolean => dt.startsWith("float") || dt.startsWith("complex");
  const lastTwo = (s: number[]): [number, number] => [s[s.length - 2]!, s[s.length - 1]!];
  /** Row-major batch of matrices -> number[][][] */
  const mats = (a: NDArray): number[][][] => {
    const [m, n] = lastTwo(a.shape);
    const f = flat(a.toArray());
    const out: number[][][] = [];
    for (let t = 0; t * m * n < f.length || (out.length === 0 && f.length === 0); t++) {
      if (m * n === 0) break;
      out.push(Array.from({ length: m }, (_, i) => f.slice(t * m * n + i * n, t * m * n + i * n + n)));
    }
    return out;
  };
  const mm = (a: number[][], b: number[][]): number[][] =>
    a.map((row) => b[0]!.map((_, j) => row.reduce((s, x, k) => s + x * b[k]![j]!, 0)));
  const T = (a: number[][]): number[][] => a[0]!.map((_, j) => a.map((row) => row[j]!));
  const closeMat = (got: number[][], exp: number[][], tol: number): void =>
    closeFlat(got.flat(), exp.flat(), tol);
  const eye = (n: number): number[][] =>
    Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

  const call = (c: LCase, xs: NDArray[]): unknown => {
    const L = np.linalg;
    const kw = c.kw ?? {};
    switch (c.fn) {
      case "matmul": return np.matmul(xs[0]!, xs[1]!);
      case "dot": return np.dot(xs[0]!, xs[1]!);
      case "inner": return np.inner(xs[0]!, xs[1]!);
      case "outer": return np.outer(xs[0]!, xs[1]!);
      case "det": return L.det(xs[0]!);
      case "inv": return L.inv(xs[0]!);
      case "solve": return L.solve(xs[0]!, xs[1]!);
      case "eig": return L.eig(xs[0]!);
      case "eigh": return L.eigh(xs[0]!);
      case "svd": return L.svd(xs[0]!, kw as { fullMatrices?: boolean; computeUV?: boolean });
      case "qr": return L.qr(xs[0]!, (kw.mode as "reduced" | "complete" | "r") ?? "reduced");
      case "lstsq": return L.lstsq(xs[0]!, xs[1]!);
      case "norm": {
        const ord = kw.ord == null ? null : (decodeExpected(kw.ord as Encoded) as never);
        return L.norm(xs[0]!, { ord, axis: kw.axis as never, keepdims: kw.keepdims as boolean });
      }
      default: throw new Error(`unknown linalg fn ${c.fn}`);
    }
  };
  const errorsFor = (c: LCase): (typeof np.ValueError)[] => {
    if (c.error === "LinAlgError") return [np.LinAlgError];
    if (c.error === "TypeError") return [np.DTypeError]; // D-018: float16
    if (c.error === "IndexError") return [np.IndexError]; // D-012
    // D-018: core-dimension mismatches raise ShapeError; batch-dimension
    // mismatches raise BroadcastError (D-014). NumPy raises ValueError for both.
    if (["matmul", "dot", "inner", "solve"].includes(c.fn!)) return [np.ShapeError, np.BroadcastError];
    return [np.ValueError];
  };

  const complexParts = (r: NDArray): [number[], number[]] => {
    const t = Array.from(r.toTypedArray() as Float64Array);
    return [t.filter((_, i) => i % 2 === 0), t.filter((_, i) => i % 2 === 1)];
  };
  const byMatrix = (arr: number[], m: number, n: number): number[][][] =>
    Array.from({ length: m * n === 0 ? 0 : arr.length / (m * n) }, (_, t) =>
      Array.from({ length: m }, (_, i) => arr.slice(t * m * n + i * n, t * m * n + i * n + n)));

  const check = (c: LCase, xs: NDArray[], r: unknown): void => {
    const A = xs[0]!;
    const tol = (dt: string): number => tolFor(dt) * 100;
    switch (c.fn) {
      case "eigh": {
        const [w, V] = c.expected as unknown as Ex[];
        const res = r as { eigenvalues: NDArray; eigenvectors: NDArray };
        approx(res.eigenvalues, w!);
        meta(res.eigenvectors, V!);
        const ws = mats(res.eigenvalues.reshape([...res.eigenvalues.shape, 1]));
        mats(A).forEach((a, t) => {
          const v = mats(res.eigenvectors)[t]!;
          const vd = v.map((row) => row.map((x, j) => x * ws[t]![j]![0]!));
          closeMat(mm(a, v), vd, tol(V!.dtype));
          closeMat(mm(T(v), v), eye(v.length), tol(V!.dtype));
        });
        return;
      }
      case "eig": {
        const [w, V] = c.expected as unknown as Ex[];
        const res = r as { eigenvalues: NDArray; eigenvectors: NDArray };
        meta(res.eigenvalues, w!);
        meta(res.eigenvectors, V!);
        const [wr, wi] = complexParts(res.eigenvalues);
        // Conjugate pairs share a real part that differs only by rounding
        // (platform/compiler dependent), so compare real parts with a tolerance
        // before ordering by the imaginary part; otherwise pairs can swap.
        const keyTol = tol(w!.dtype) * Math.max(1, ...flat(decodeExpected(w!.values!)).map(Math.abs));
        const key = (re: number[], im: number[]): number[][] =>
          re.map((x, i) => [x, im[i]!]).sort((p, q) =>
            Math.abs(p[0]! - q[0]!) > keyTol ? p[0]! - q[0]! : p[1]! - q[1]!);
        const n = A.shape[A.shape.length - 1]!;
        for (let t = 0; t * n < wr.length; t++) {
          const g = key(wr.slice(t * n, t * n + n), wi.slice(t * n, t * n + n));
          const e = key(flat(decodeExpected(w!.values!)).slice(t * n, t * n + n),
            flat(decodeExpected(w!.imag!)).slice(t * n, t * n + n));
          closeFlat(g.flat(), e.flat(), tol(w!.dtype));
        }
        // A v_j = w_j v_j  (complex)
        const [vr, vi] = complexParts(res.eigenvectors);
        const VR = byMatrix(vr, n, n);
        const VI = byMatrix(vi, n, n);
        mats(A).forEach((a, t) => {
          const lr = mm(a, VR[t]!);
          const li = mm(a, VI[t]!);
          const rr = VR[t]!.map((row, i) => row.map((x, j) => x * wr[t * n + j]! - VI[t]![i]![j]! * wi[t * n + j]!));
          const ri = VR[t]!.map((row, i) => row.map((x, j) => x * wi[t * n + j]! + VI[t]![i]![j]! * wr[t * n + j]!));
          closeMat(lr, rr, tol(w!.dtype));
          closeMat(li, ri, tol(w!.dtype));
        });
        return;
      }
      case "svd": {
        const exp = c.expected as Ex | Ex[];
        const kw = c.kw ?? {};
        const res = r as { U: NDArray | null; S: NDArray; Vh: NDArray | null };
        if (kw.computeUV === false) {
          expect(res.U).toBeNull();
          approx(res.S, exp as Ex);
          return;
        }
        const [U, S, Vh] = exp as Ex[];
        meta(res.U!, U!);
        approx(res.S, S!);
        meta(res.Vh!, Vh!);
        const [m, n] = lastTwo(A.shape);
        const k = Math.min(m, n);
        const us = mats(res.U!);
        const vs = mats(res.Vh!);
        const ss = flat(res.S.toArray());
        mats(A).forEach((a, t) => {
          const u = us[t]!;
          const v = vs[t]!;
          // U[:, :k] diag(S) Vh[:k, :] == A
          const usk = u.map((row) => row.slice(0, k).map((x, j) => x * ss[t * k + j]!));
          closeMat(mm(usk, v.slice(0, k)), a, tol(S!.dtype));
          closeMat(mm(T(u), u), eye(u[0]!.length), tol(S!.dtype));
          closeMat(mm(v, T(v)), eye(v.length), tol(S!.dtype));
        });
        return;
      }
      case "qr": {
        const exp = c.expected as Ex | Ex[];
        const res = r as { Q: NDArray | null; R: NDArray };
        if (c.kw?.mode === "r") {
          expect(res.Q).toBeNull();
          meta(res.R, exp as Ex);
          // R is unique up to row signs.
          const g = flat(res.R.toArray()).map(Math.abs);
          closeFlat(g, flat(decodeExpected((exp as Ex).values!)).map(Math.abs), tol((exp as Ex).dtype));
          return;
        }
        const [Q, R] = exp as Ex[];
        meta(res.Q!, Q!);
        meta(res.R, R!);
        const rs = mats(res.R);
        mats(res.Q!).forEach((q, t) => {
          closeMat(mm(q, rs[t]!), mats(A)[t]!, tol(R!.dtype));
          closeMat(mm(T(q), q), eye(q[0]!.length), tol(R!.dtype));
          rs[t]!.forEach((row, i) => row.forEach((x, j) => { if (j < i) expect(x).toBe(0); }));
        });
        return;
      }
      case "lstsq": {
        const [x, res0, rank, s] = c.expected as unknown as [Ex, Ex, number, Ex];
        const res = r as { x: NDArray; residuals: NDArray; rank: number; s: NDArray };
        approx(res.x, x);
        approx(res.residuals, res0);
        expect(res.rank).toBe(rank);
        approx(res.s, s);
        return;
      }
      default: {
        const exp = c.expected as Ex;
        if (isFloat(exp.dtype)) approx(r as NDArray, exp);
        else exact(r as NDArray, exp);
      }
    }
  };

  const run = (_l: string, c: LCase): void => {
    const xs = c.inputs.map((i) =>
      np.array(decodeInput(i.data) as NestedArray, { dtype: i.dtype }).reshape(i.shape));
    if (c.error !== undefined) {
      let thrown: unknown;
      try {
        call(c, xs);
      } catch (e) {
        thrown = e;
      }
      expect(errorsFor(c).some((k) => thrown instanceof k), String(thrown)).toBe(true);
      return;
    }
    check(c, xs, call(c, xs));
  };
  const table = (cases as LCase[]).map((c) => [label(c), c] as const);
  describe(`default backend (${np.linalg.backend()})`, () => {
    it.each(table)(`numpy ${numpy_version}: %s`, run);
  });
  describe("fallback backend", () => {
    beforeAll(() => np.linalg._setBackend("fallback"));
    afterAll(() => np.linalg._setBackend("default"));
    it("is active", () => expect(np.linalg.backend()).toBe("fallback"));
    it.each(table)(`numpy ${numpy_version}: %s`, run);
  });
});


describe("differential: complex_matmul (P1-4d, D-035/D-036)", () => {
  const { numpy_version, cases } = load("complex_matmul");
  type CCase = Case & {
    inputs: { data: Encoded; dtype: string; shape: number[] }[];
    expected_noblas?: Encoded;
  };
  type PFn = (a: NDArray, b: NDArray) => NDArray;
  const fns = np as unknown as Record<string, PFn>;
  const parts = (v: unknown): number[] =>
    Array.isArray(v) ? v.flatMap(parts) : v instanceof Complex ? [v.re, v.im] : [v as number];
  // D-018: rounded results are compared with a tolerance scaled by max|expected|.
  const tolFor = (dt: string): number => (dt === "complex64" ? 2e-5 : 1e-12);

  const run = (_l: string, c: CCase): void => {
    const [a, b] = c.inputs.map((i) =>
      np.array(decodeInput(i.data) as NestedArray, { dtype: i.dtype }).reshape(i.shape));
    const make = (): NDArray => fns[c.fn!]!(a!, b!);
    if (c.error !== undefined) {
      // D-018: core mismatches raise ShapeError, batch mismatches BroadcastError,
      // 0-d matmul operands ValueError; NumPy raises ValueError for all three.
      let thrown: unknown;
      try {
        make();
      } catch (e) {
        thrown = e;
      }
      expect([np.ShapeError, np.BroadcastError, np.ValueError].some((k) => thrown instanceof k),
        String(thrown)).toBe(true);
      return;
    }
    const r = make();
    const exp = c.expected as Described;
    expect(r.dtype.name).toBe(exp.dtype);
    expect(r.shape).toEqual(exp.shape);
    expect(r.flags.cContiguous).toBe(true);
    // NumPy reports (0, 0) strides for some zero-size products; ours are C strides.
    if (r.size > 0) expect(r.strides).toEqual(exp.strides);
    if (!c.approx) {
      // D-037: with non-finite input, NumPy's BLAS path and its own non-BLAS loop
      // can differ; the fallback backend matches the non-BLAS loop.
      const want = np.linalg.backend() === "fallback" && c.expected_noblas !== undefined
        ? c.expected_noblas : exp.values!;
      expect(r.toArray()).toEqual(decodeExpected(want));
      return;
    }
    const got = parts(r.toArray());
    const want = parts(decodeExpected(exp.values!));
    expect(got.length).toBe(want.length);
    const scale = Math.max(1, ...want.map(Math.abs));
    const tol = tolFor(exp.dtype) * scale;
    want.forEach((e, i) => expect(Math.abs(got[i]! - e), `element ${i}`).toBeLessThanOrEqual(tol));
  };
  const table = (cases as CCase[]).map((c) => [label(c), c] as const);
  describe(`default backend (${np.linalg.backend()})`, () => {
    it.each(table)(`numpy ${numpy_version}: %s`, run);
  });
  describe("fallback backend", () => {
    beforeAll(() => np.linalg._setBackend("fallback"));
    afterAll(() => np.linalg._setBackend("default"));
    it("is active", () => expect(np.linalg.backend()).toBe("fallback"));
    it.each(table)(`numpy ${numpy_version}: %s`, run);
  });
});
describe("differential: complex_linalg (P1-5e.4, D-038–D-044)", () => {
  const { numpy_version, cases } = load("complex_linalg");
  type Ex = Described & { imag?: Encoded };
  type LCase = Case & { inputs: { data: Encoded; dtype: string; shape: number[] }[] };
  /** Complex matrix as separate real/imag row-major parts. */
  interface CM { re: number[][]; im: number[][] }
  const flat = (v: unknown): number[] =>
    Array.isArray(v) ? v.flatMap(flat) : [typeof v === "boolean" ? Number(v) : (v as number)];
  const isC = (dt: string): boolean => dt.startsWith("complex");
  // D-018: values within a tolerance scaled by max|expected|; vectors are
  // compared by reconstruction (sign/phase are backend dependent).
  const tolFor = (dt: string): number => (dt === "float32" || dt === "complex64" ? 2e-4 : 1e-10);
  /** [re, im] flat parts of a real or complex result. */
  const parts = (r: NDArray): [number[], number[]] => {
    if (!isC(r.dtype.name)) {
      const re = flat(r.toArray());
      return [re, re.map(() => 0)];
    }
    const t = Array.from(r.toTypedArray() as Float64Array);
    return [t.filter((_, i) => i % 2 === 0), t.filter((_, i) => i % 2 === 1)];
  };
  const expParts = (e: Ex): [number[], number[]] => {
    const re = flat(decodeExpected(e.values!));
    return [re, e.imag === undefined ? re.map(() => 0) : flat(decodeExpected(e.imag))];
  };
  const close = (got: number[], exp: number[], tol: number, scaleOf: number[] = exp): void => {
    expect(got.length).toBe(exp.length);
    const scale = Math.max(1, ...scaleOf.map((x) => (Number.isFinite(x) ? Math.abs(x) : 0)));
    exp.forEach((e, i) => {
      if (!Number.isFinite(e)) expect(got[i]).toEqual(e);
      else expect(Math.abs(got[i]! - e), `element ${i}`).toBeLessThanOrEqual(tol * scale);
    });
  };
  const meta = (r: NDArray, e: Ex): void => {
    expect(r.dtype.name).toBe(e.dtype);
    expect(r.shape).toEqual(e.shape);
    expect(r.flags.cContiguous).toBe(true);
  };
  /** dtype/shape exact; real and imaginary parts within tolerance. */
  const approx = (r: NDArray, e: Ex): void => {
    meta(r, e);
    const [gr, gi] = parts(r);
    const [er, ei] = expParts(e);
    close(gr, er, tolFor(e.dtype), [...er, ...ei]);
    close(gi, ei, tolFor(e.dtype), [...er, ...ei]);
  };
  /** Batch of complex matrices (last two dims) from an NDArray. */
  const cmats = (a: NDArray): CM[] => {
    const [m, n] = [a.shape[a.shape.length - 2]!, a.shape[a.shape.length - 1]!];
    if (m * n === 0) return [];
    const [re, im] = parts(a);
    const rows = (f: number[], t: number): number[][] =>
      Array.from({ length: m }, (_, i) => f.slice(t * m * n + i * n, t * m * n + i * n + n));
    return Array.from({ length: re.length / (m * n) }, (_, t) => ({ re: rows(re, t), im: rows(im, t) }));
  };
  const cmm = (a: CM, b: CM): CM => {
    const k = b.re.length;
    const p = b.re[0]!.length;
    const re: number[][] = [];
    const im: number[][] = [];
    for (let i = 0; i < a.re.length; i++) {
      re.push(new Array<number>(p).fill(0));
      im.push(new Array<number>(p).fill(0));
      for (let j = 0; j < p; j++) {
        for (let q = 0; q < k; q++) {
          const [ar, ai, br, bi] = [a.re[i]![q]!, a.im[i]![q]!, b.re[q]![j]!, b.im[q]![j]!];
          re[i]![j]! += ar * br - ai * bi;
          im[i]![j]! += ar * bi + ai * br;
        }
      }
    }
    return { re, im };
  };
  /** Conjugate transpose. */
  const H = (a: CM): CM => ({
    re: a.re[0]!.map((_, j) => a.re.map((row) => row[j]!)),
    im: a.im[0]!.map((_, j) => a.im.map((row) => -row[j]!)),
  });
  /** Scale column j by (wr[j] + i·wi[j]). */
  const scaleCols = (a: CM, wr: number[], wi: number[]): CM => ({
    re: a.re.map((row, i) => row.map((x, j) => x * wr[j]! - a.im[i]![j]! * wi[j]!)),
    im: a.re.map((row, i) => row.map((x, j) => x * wi[j]! + a.im[i]![j]! * wr[j]!)),
  });
  const cclose = (got: CM, exp: CM, tol: number): void => {
    const all = [...exp.re.flat(), ...exp.im.flat()];
    close(got.re.flat(), exp.re.flat(), tol, all);
    close(got.im.flat(), exp.im.flat(), tol, all);
  };
  const ceye = (n: number): CM => ({
    re: Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))),
    im: Array.from({ length: n }, () => new Array<number>(n).fill(0)),
  });
  /** Q^H Q = I for every matrix in the batch. */
  const unitary = (q: CM[], tol: number): void =>
    q.forEach((m) => cclose(cmm(H(m), m), ceye(m.re[0]!.length), tol));
  const call = (c: LCase, xs: NDArray[]): unknown => {
    const L = np.linalg;
    const kw = c.kw ?? {};
    const [a, b] = [xs[0]!, xs[1]!];
    switch (c.fn) {
      case "det": return L.det(a);
      case "inv": return L.inv(a);
      case "solve": return L.solve(a, b);
      case "eig": return L.eig(a);
      case "eigvals": return L.eigvals(a);
      case "eigh": return L.eigh(a);
      case "eigvalsh": return L.eigvalsh(a);
      case "svd": return L.svd(a, kw as { fullMatrices?: boolean; computeUV?: boolean });
      case "qr": return L.qr(a, (kw.mode as "reduced" | "complete" | "r") ?? "reduced");
      case "lstsq": return L.lstsq(a, b);
      case "norm": {
        const ord = kw.ord == null ? null : (decodeExpected(kw.ord as Encoded) as never);
        return L.norm(a, { ord, axis: kw.axis as never, keepdims: kw.keepdims as boolean });
      }
      default: throw new Error(`unknown fn ${c.fn}`);
    }
  };
  /** Eigenvalues as sorted [re, im] pairs per matrix; ties in re within tol. */
  const sortedPairs = (re: number[], im: number[], n: number, tol: number): number[] => {
    const out: number[] = [];
    for (let t = 0; t * n < re.length; t++) {
      const ps = re.slice(t * n, t * n + n).map((x, i) => [x, im[t * n + i]!] as const);
      ps.sort((p, q) => (Math.abs(p[0] - q[0]) > tol ? p[0] - q[0] : p[1] - q[1]));
      out.push(...ps.flat());
    }
    return out;
  };
  const eigvalsCheck = (w: NDArray, e: Ex, n: number): void => {
    meta(w, e);
    const [gr, gi] = parts(w);
    const [er, ei] = expParts(e);
    const tol = tolFor(e.dtype);
    const key = tol * 100 * Math.max(1, ...er.map(Math.abs), ...ei.map(Math.abs));
    close(sortedPairs(gr, gi, n, key), sortedPairs(er, ei, n, key), tol, [...er, ...ei]);
  };
  const check = (c: LCase, xs: NDArray[], r: unknown): void => {
    const A = cmats(xs[0]!);
    const tv = (dt: string): number => tolFor(dt) * 100; // vector reconstructions
    const n = xs[0]!.shape[xs[0]!.shape.length - 1]!;
    type Pair = { eigenvalues: NDArray; eigenvectors: NDArray };
    switch (c.fn) {
      case "eig":
      case "eigh": {
        const [w, V] = c.expected as unknown as Ex[];
        const res = r as Pair;
        eigvalsCheck(res.eigenvalues, w!, n);
        meta(res.eigenvectors, V!);
        const [wr, wi] = parts(res.eigenvalues);
        cmats(res.eigenvectors).forEach((v, t) => {
          // A V = V diag(w)
          const ws = [wr.slice(t * n, t * n + n), wi.slice(t * n, t * n + n)] as const;
          cclose(cmm(A[t]!, v), scaleCols(v, ws[0], ws[1]), tv(V!.dtype));
        });
        if (c.fn === "eigh") unitary(cmats(res.eigenvectors), tv(V!.dtype));
        return;
      }
      case "eigvals":
      case "eigvalsh":
        eigvalsCheck(r as NDArray, c.expected as Ex, n);
        return;
      case "svd": {
        const res = r as { U: NDArray | null; S: NDArray; Vh: NDArray | null };
        if (c.kw?.computeUV === false) {
          expect(res.U).toBeNull();
          approx(res.S, c.expected as Ex);
          return;
        }
        const [U, S, Vh] = c.expected as unknown as Ex[];
        meta(res.U!, U!);
        approx(res.S, S!);
        meta(res.Vh!, Vh!);
        const k = Math.min(xs[0]!.shape.at(-2)!, n);
        const s = parts(res.S)[0];
        const us = cmats(res.U!);
        const vs = cmats(res.Vh!);
        us.forEach((u, t) => {
          const uk = { re: u.re.map((row) => row.slice(0, k)), im: u.im.map((row) => row.slice(0, k)) };
          const vk = { re: vs[t]!.re.slice(0, k), im: vs[t]!.im.slice(0, k) };
          const sk = s.slice(t * k, t * k + k);
          cclose(cmm(scaleCols(uk, sk, sk.map(() => 0)), vk), A[t]!, tv(S!.dtype));
          cclose(cmm(vs[t]!, H(vs[t]!)), ceye(vs[t]!.re.length), tv(S!.dtype));
        });
        unitary(us, tv(S!.dtype));
        return;
      }
      case "qr": {
        const res = r as { Q: NDArray | null; R: NDArray };
        if (c.kw?.mode === "r") {
          // R is unique up to a unit phase per row: compare |R|.
          const e = c.expected as Ex;
          expect(res.Q).toBeNull();
          meta(res.R, e);
          const [gr, gi] = parts(res.R);
          const [er, ei] = expParts(e);
          close(gr.map((x, i) => Math.hypot(x, gi[i]!)), er.map((x, i) => Math.hypot(x, ei[i]!)),
            tv(e.dtype));
          return;
        }
        const [Q, R] = c.expected as unknown as Ex[];
        meta(res.Q!, Q!);
        meta(res.R, R!);
        const rs = cmats(res.R);
        cmats(res.Q!).forEach((q, t) => {
          cclose(cmm(q, rs[t]!), A[t]!, tv(R!.dtype));
          rs[t]!.re.forEach((row, i) => row.forEach((x, j) => {
            if (j < i) expect([x, rs[t]!.im[i]![j]]).toEqual([0, 0]);
          }));
        });
        unitary(cmats(res.Q!), tv(R!.dtype));
        return;
      }
      case "lstsq": {
        const [x, res0, rank, s] = c.expected as unknown as [Ex, Ex, number, Ex];
        const res = r as { x: NDArray; residuals: NDArray; rank: number; s: NDArray };
        approx(res.x, x);
        approx(res.residuals, res0);
        expect(res.rank).toBe(rank);
        approx(res.s, s);
        return;
      }
      default:
        approx(r as NDArray, c.expected as Ex);
    }
  };
  const run = (_l: string, c: LCase): void => {
    const xs = c.inputs.map((i) =>
      np.array(decodeInput(i.data) as NestedArray, { dtype: i.dtype }).reshape(i.shape));
    if (c.error !== undefined) {
      let thrown: unknown;
      try {
        call(c, xs);
      } catch (e) {
        thrown = e;
      }
      const want = c.error === "LinAlgError" ? np.LinAlgError : np.ValueError;
      expect(thrown instanceof want, String(thrown)).toBe(true);
      return;
    }
    check(c, xs, call(c, xs));
  };
  const table = (cases as LCase[]).map((c) => [label(c), c] as const);
  describe(`default backend (${np.linalg.backend()})`, () => {
    it.each(table)(`numpy ${numpy_version}: %s`, run);
  });
  describe("fallback backend", () => {
    beforeAll(() => np.linalg._setBackend("fallback"));
    afterAll(() => np.linalg._setBackend("default"));
    it("is active", () => expect(np.linalg.backend()).toBe("fallback"));
    it.each(table)(`numpy ${numpy_version}: %s`, run);
  });
});






describe("differential: random (M9, D-019) — bit-exact streams", () => {
  const { cases } = load("random");
  type Res = { scalar?: Encoded; error?: string; dtype?: string; shape?: number[]; values?: Encoded };
  type Call = { m: string; args: Encoded[]; kw: Record<string, unknown>; result: Res };
  type RC = Case & { api: string; seed: Encoded; calls: Call[] };
  const R = np.random;

  const call = (rng: InstanceType<typeof R.Generator> | InstanceType<typeof R.RandomState>, c: Call): unknown => {
    const a = c.args.map((v) => decodeInput(v)) as unknown as [never, never, never];
    const kw = c.kw as Record<string, never>;
    const gen = rng instanceof R.Generator;
    switch (c.m) {
      case "random":
        return gen ? rng.random({ size: a[0], dtype: kw.dtype }) : rng.random(a[0]);
      case "standard_normal":
        return gen ? rng.standardNormal({ size: a[0], dtype: kw.dtype }) : rng.standardNormal(a[0]);
      case "normal":
        return rng.normal(a[0], a[1], a[2]);
      case "uniform":
        return rng.uniform(a[0], a[1], a[2]);
      case "integers":
        return (rng as InstanceType<typeof R.Generator>).integers(
          a[0], a[1] ?? null, a[2], kw.dtype ?? "int64", kw.endpoint ?? false);
      case "randint":
        return (rng as InstanceType<typeof R.RandomState>).randint(a[0], a[1] ?? null, a[2], kw.dtype ?? "int64");
      case "choice":
        return gen
          ? rng.choice(a[0], { size: a[1], replace: kw.replace ?? true, shuffle: kw.shuffle ?? true })
          : rng.choice(a[0], { size: a[1], replace: kw.replace ?? true });
      case "permutation":
        return rng.permutation(a[0]);
      case "shuffle": {
        const shape = a[0] as unknown as number[];
        const x = np.arange(shape.reduce((p, d) => p * d, 1)).reshape(shape);
        if (gen) rng.shuffle(x, kw.axis ?? 0);
        else rng.shuffle(x);
        return x;
      }
      case "rand":
        return (rng as InstanceType<typeof R.RandomState>).rand(...(a as number[]));
      case "randn":
        return (rng as InstanceType<typeof R.RandomState>).randn(...(a as number[]));
      case "random_sample":
        return (rng as InstanceType<typeof R.RandomState>).randomSample(a[0]);
      default:
        throw new Error(`unknown random method ${c.m}`);
    }
  };

  for (const rc of cases as RC[]) {
    it(`${rc.api} seed=${JSON.stringify(rc.seed)}`, () => {
      const seed = decodeInput(rc.seed) as never;
      const rng = rc.api === "gen" ? R.defaultRng(seed) : new R.RandomState(seed);
      for (const c of rc.calls) {
        const where = `${c.m}(${JSON.stringify(c.args)}, ${JSON.stringify(c.kw)})`;
        if (c.result.error !== undefined) {
          expect(() => call(rng, c), where).toThrow(np.ValueError);
          continue;
        }
        const r = call(rng, c);
        if (c.result.scalar !== undefined) {
          expect(r, where).toEqual(decodeExpected(c.result.scalar));
          continue;
        }
        const arr = r as NDArray;
        expect(arr.dtype.name, where).toBe(c.result.dtype);
        expect(arr.shape, where).toEqual(c.result.shape);
        // Exact equality: the streams must be bit-identical to NumPy.
        expect(arr.toArray(), where).toEqual(decodeExpected(c.result.values!));
      }
    });
  }
});


describe("differential: fft (M10, D-020)", () => {
  const { numpy_version, cases } = load("fft");
  type FIn = { data: Encoded[]; dtype: string; shape: number[] };
  type FCase = Case & { input: FIn; expected?: { dtype: string; shape: number[]; values: Encoded[] } };
  const F = np.fft;
  const num = (v: Encoded): number => decodeInput(v) as number;

  /** Rebuilds the input; complex data arrives as interleaved re/im. */
  const build = (i: FIn): NDArray => {
    const vals = i.data.map(num);
    if (i.dtype.startsWith("complex")) {
      const src = i.dtype === "complex64" ? Float32Array.from(vals) : Float64Array.from(vals);
      return np.fromTypedArray(src, i.shape, { dtype: i.dtype });
    }
    return np.array(vals, { dtype: i.dtype }).reshape(i.shape);
  };
  const call = (c: FCase, x: NDArray): NDArray => {
    const kw = c.kw ?? {};
    const n = kw.n as number | null | undefined;
    const axis = (kw.axis as number | undefined) ?? -1;
    const norm = kw.norm as never;
    const s = kw.s as number[] | null | undefined;
    switch (c.fn) {
      case "fft": return F.fft(x, n, axis, norm);
      case "ifft": return F.ifft(x, n, axis, norm);
      case "rfft": return F.rfft(x, n, axis, norm);
      case "irfft": return F.irfft(x, n, axis, norm);
      case "fftn": return F.fftn(x, s, kw.axes as never, norm);
      case "ifftn": return F.ifftn(x, s, kw.axes as never, norm);
      // Options-object form, with axes defaulted when absent (NumPy [-2, -1]).
      case "fft2": return "axes" in kw ? F.fft2(x, s, kw.axes as never, norm) : F.fft2(x, { s, norm });
      case "ifft2": return "axes" in kw ? F.ifft2(x, s, kw.axes as never, norm) : F.ifft2(x, { s, norm });
      case "fftfreq": return F.fftfreq(n!, kw.d as number);
      case "rfftfreq": return F.rfftfreq(n!, kw.d as number);
      default: throw new Error(`unknown fft fn ${c.fn}`);
    }
  };
  const errorFor = (e: string): typeof np.ValueError => {
    if (e === "IndexError") return np.IndexError; // AxisError, D-012
    if (e === "TypeError") return np.DTypeError; // rfft of complex, D-020
    return np.ValueError; // ValueError; ZeroDivisionError in fftfreq (D-020)
  };
  /** pocketfft is shared with NumPy, but summation order differs slightly on some paths. */
  const tolFor = (dt: string): number =>
    dt === "float16" ? 2e-3 : dt === "float32" || dt === "complex64" ? 2e-6 : 1e-12;

  const run = (_l: string, c: FCase): void => {
    const x = build(c.input);
    if (c.error !== undefined) {
      expect(() => call(c, x)).toThrow(errorFor(c.error));
      return;
    }
    const r = call(c, x);
    const exp = c.expected!;
    expect(r.dtype.name).toBe(exp.dtype);
    expect(r.shape).toEqual(exp.shape);
    expect(r.flags.cContiguous).toBe(true);
    const got = exp.dtype.startsWith("complex")
      ? Array.from(r.toTypedArray() as Float64Array)
      : (r.astype("float64").toArray() as NestedArray[]).flat(Infinity as 1) as number[];
    const want = exp.values.map(num);
    expect(got.length).toBe(want.length);
    const scale = Math.max(1, ...want.map(Math.abs));
    const tol = tolFor(exp.dtype) * scale;
    want.forEach((w, k) => expect(Math.abs(got[k]! - w), `index ${k}`).toBeLessThanOrEqual(tol));
  };
  const label = (c: FCase): string =>
    `${c.fn}(${c.input.dtype}${JSON.stringify(c.input.shape)}, ${JSON.stringify(c.kw)})`;
  const table = (cases as FCase[]).map((c) => [label(c), c] as const);
  it.each(table)(`numpy ${numpy_version}: %s`, run);
});

describe("differential: complex_fft (P1-5e.5, D-020/D-033)", () => {
  // Input goes through np.array(nested Complex list) and comes back through
  // toArray(), so the whole JS <-> complex path is covered, not only the kernel.
  interface CFCase {
    fn: string;
    data: Encoded;
    dtype?: string;
    view?: "T" | "rev" | "step";
    plain?: boolean;
    kw: { n?: number; axis?: number; norm?: string; s?: number[]; axes?: number[] };
    error?: string;
    expected?: { dtype: string; shape: number[]; values: Encoded };
  }
  const { numpy_version, cases } = load("complex_fft") as unknown as { numpy_version: string; cases: CFCase[] };
  const F = np.fft;
  /** Complex -> plain `{ re, im }` objects, the other accepted ComplexLike form (D-033). */
  const toPlain = (v: NestedArray): NestedArray =>
    Array.isArray(v) ? v.map(toPlain) : v instanceof Complex ? { re: v.re, im: v.im } : v;
  const build = (c: CFCase): NDArray => {
    const data = decodeInput(c.data);
    const x = np.array(c.plain ? toPlain(data) : data, c.dtype === undefined ? {} : { dtype: c.dtype });
    if (c.view === "T") return x.T;
    if (c.view === "rev") return x.get(np.ellipsis, [null, null, -1]);
    if (c.view === "step") return x.get(np.ellipsis, [null, null, 2]);
    return x;
  };
  const call = (c: CFCase, x: NDArray): NDArray => {
    const { n, axis, norm, s, axes } = c.kw;
    const nm = norm as never;
    switch (c.fn) {
      case "fft": return F.fft(x, { n, axis, norm: nm });
      case "ifft": return F.ifft(x, { n, axis, norm: nm });
      case "rfft": return F.rfft(x, { n, axis, norm: nm });
      case "irfft": return F.irfft(x, { n, axis, norm: nm });
      case "fft2": return F.fft2(x, { s, norm: nm });
      case "ifft2": return F.ifft2(x, { s, norm: nm });
      case "fftn": return F.fftn(x, { s, axes, norm: nm });
      case "ifftn": return F.ifftn(x, { s, axes, norm: nm });
      default: throw new Error(`unknown fft fn ${c.fn}`);
    }
  };
  const errorFor = (e: string): typeof np.ValueError =>
    e === "IndexError" ? np.IndexError : e === "TypeError" ? np.DTypeError : np.ValueError;
  /** [re, im] pairs (real results get im = 0) from toArray() or decoded values. */
  const pairs = (v: unknown): [number, number][] =>
    Array.isArray(v) ? v.flatMap(pairs) : v instanceof Complex ? [[v.re, v.im]] : [[v as number, 0]];
  const tolFor = (dt: string): number => (dt === "float32" || dt === "complex64" ? 2e-6 : 1e-12);

  const run = (_l: string, c: CFCase): void => {
    if (c.error !== undefined) {
      expect(() => call(c, build(c))).toThrow(errorFor(c.error));
      return;
    }
    const x = build(c);
    if (c.view !== undefined) expect(x.flags.cContiguous).toBe(false);
    const r = call(c, x);
    const exp = c.expected!;
    expect(r.dtype.name).toBe(exp.dtype);
    expect(r.shape).toEqual(exp.shape);
    const out = r.toArray();
    if (exp.dtype.startsWith("complex")) {
      const flat = (v: unknown): unknown[] => (Array.isArray(v) ? v.flatMap(flat) : [v]);
      flat(out).forEach((z) => expect(z).toBeInstanceOf(Complex));
    }
    const got = pairs(out);
    const want = pairs(decodeExpected(exp.values));
    expect(got.length).toBe(want.length);
    const finite = want.flat().filter(Number.isFinite);
    const tol = tolFor(exp.dtype) * Math.max(1, ...finite.map(Math.abs));
    want.forEach((w, k) => {
      for (let j = 0; j < 2; j++) {
        const g = got[k]![j]!;
        const e = w[j]!;
        if (Number.isFinite(e)) expect(Math.abs(g - e), `index ${k}.${j}`).toBeLessThanOrEqual(tol);
        else expect(g, `index ${k}.${j}`).toBe(e); // NaN/±inf must match exactly
      }
    });
  };
  const label = (c: CFCase): string =>
    `${c.fn}(${c.dtype ?? "inferred"}${c.view ? `.${c.view}` : ""}${c.plain ? " {re,im}" : ""}, ` +
    `${JSON.stringify(c.kw)}) <- ${JSON.stringify(c.data).slice(0, 40)}`;
  it("covers every complex FFT entry point", () => {
    expect(new Set(cases.map((c) => c.fn))).toEqual(
      new Set(["fft", "ifft", "rfft", "irfft", "fft2", "ifft2", "fftn", "ifftn"]),
    );
  });
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, run);
});


describe("differential: complex conversion (P1, D-033)", () => {
  const { numpy_version, cases } = load("complex");
  it.each(cases.map((c) => [label(c), c] as const))(`numpy ${numpy_version}: %s`, (_l, c) => {
    if (c.op === "zeros") {
      checkArray(np.zeros(c.shape!, { dtype: c.dtype! }), c.expected as Described);
    } else if (c.op === "astype") {
      const src = np.array(decodeInput(c.data!), { dtype: c.src_dtype! });
      checkArray(src.astype(c.dtype!), c.expected as Described);
    } else {
      const opts = c.dtype === undefined ? {} : { dtype: c.dtype };
      checkArray(np.array(decodeInput(c.data!), opts), c.expected as Described);
    }
  });
});

describe("differential: complex ufuncs (P1 step 2, D-033)", () => {
  interface CUArg { data: Encoded; dtype?: string }
  interface CUCase { op: string; args: CUArg[]; kw?: { deg?: boolean }; approx: boolean; expected: Described }
  const { numpy_version, cases } = load("complex_ufuncs") as unknown as { numpy_version: string; cases: CUCase[] };
  // Scalars stay JS scalars (weak NEP 50 operands); arrays keep NumPy's dtype.
  const operand = (x: CUArg): NDArray | number | Complex => {
    const v = decodeInput(x.data);
    return x.dtype === undefined ? (v as number | Complex) : np.array(v as NestedArray, { dtype: x.dtype });
  };
  // Libm-based results (sqrt/exp/log/pow/abs/angle, D-014/D-033): relative
  // closeness per component; inf/nan/signed-zero categories must still match.
  const rtol = (dt: string): number => (dt === "complex64" || dt === "float32" ? 1e-6 : 1e-14);
  const near = (g: number, e: number, tol: number): boolean => {
    if (Number.isNaN(e) || !Number.isFinite(e) || e === 0) return Object.is(g, e) || (e === 0 && g === 0 && Math.abs(g) <= tol);
    return Math.abs(g - e) <= tol * Math.abs(e);
  };
  const flat = (v: unknown): unknown[] => (Array.isArray(v) ? v.flatMap(flat) : [v]);
  const run = (_l: string, c: CUCase): void => {
    const [x, y] = c.args.map(operand);
    const ufuncs = np as unknown as Record<string, (...a: unknown[]) => NDArray>;
    const r = c.op === "angle" ? np.angle(x as NDArray, c.kw?.deg ?? false) : y === undefined ? ufuncs[c.op]!(x) : ufuncs[c.op]!(x, y);
    const exp = c.expected;
    expect(r.dtype.name).toBe(exp.dtype);
    expect(r.shape).toEqual(exp.shape);
    const want = decodeExpected(exp.values!);
    if (!c.approx) {
      expect(r.toArray()).toEqual(want);
      return;
    }
    const tol = rtol(exp.dtype);
    const got = flat(r.toArray());
    flat(want).forEach((e, i) => {
      const g = got[i];
      const pairs: [number, number][] = e instanceof Complex
        ? [[(g as Complex).re, e.re], [(g as Complex).im, e.im]]
        : [[g as number, e as number]];
      for (const [gv, ev] of pairs) {
        expect(near(gv, ev, tol), `[${i}] got ${String(g)} want ${String(e)}`).toBe(true);
      }
    });
  };
  it.each(cases.map((c) => [JSON.stringify({ op: c.op, args: c.args, kw: c.kw }).slice(0, 300), c] as const))(
    `numpy ${numpy_version}: %s`,
    run,
  );
});
