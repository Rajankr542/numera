// P7 grids and index helpers (D-100): indices, meshgrid, mgrid, ogrid, ix_,
// r_, c_, s_, indexExp. JS has no `__getitem__`, so the NumPy index-trick
// objects are plain functions taking slice tuples or slice strings.
import { isComplexLike, type ComplexLike } from "./complex.js";
import { arange, array, asarray, linspace } from "./creation.js";
import { dtype as toDType, promoteTypes, type DType, type DTypeLike } from "./dtype.js";
import { NotImplementedError, ValueError } from "./errors.js";
import { nonzero } from "./indexing.js";
import { ellipsis, NDArray, type IndexSpec, type NestedArray } from "./ndarray.js";
import { broadcastShapes, broadcastTo } from "./ufunc.js";
import { wrapP07 } from "./p07_native.js";

type ArrayInput = NDArray | NestedArray;

/** Shape with `n` at axis `k` and 1 elsewhere (`nd` axes). */
const axisShape = (nd: number, k: number, n: number): number[] =>
  Array.from({ length: nd }, (_, i) => (i === k ? n : 1));

export interface IndicesOptions {
  /** Default int64. */
  dtype?: DTypeLike;
  /** Return one broadcastable array per axis instead of the dense grid. */
  sparse?: boolean;
}

/** np.indices(dimensions, {dtype, sparse}): the dense `(N, ...dims)` grid, or N sparse arrays. */
export function indices(dimensions: readonly number[], options: IndicesOptions & { sparse: true }): NDArray[];
export function indices(dimensions: readonly number[], options?: IndicesOptions): NDArray;
export function indices(dimensions: readonly number[], options: IndicesOptions = {}): NDArray | NDArray[] {
  const dt = toDType(options.dtype ?? "int64").name;
  const dims = [...dimensions];
  if (options.sparse) {
    return dims.map((n, k) =>
      wrapP07((x) => x.gridAxis(0, 1, n, "int64")).astype(dt).reshape(axisShape(dims.length, k, n)),
    );
  }
  return wrapP07((x) => x.indices(dims, dt));
}

export interface MeshgridOptions {
  /** "xy" (default, Cartesian) or "ij" (matrix). */
  indexing?: "xy" | "ij";
  sparse?: boolean;
  /** Default true. `false` returns views (dense results are read-only broadcast views, D-100). */
  copy?: boolean;
}

const isOptions = (x: unknown): x is MeshgridOptions =>
  typeof x === "object" && x !== null && !Array.isArray(x) && !(x instanceof NDArray) && !isComplexLike(x);

/** np.meshgrid(...xi, [options]): coordinate matrices from coordinate vectors. */
export function meshgrid(...args: (ArrayInput | MeshgridOptions)[]): NDArray[] {
  const opts: MeshgridOptions = args.length > 0 && isOptions(args[args.length - 1]) ? (args.pop() as MeshgridOptions) : {};
  const indexing = opts.indexing ?? "xy";
  if (indexing !== "xy" && indexing !== "ij") {
    throw new ValueError("Valid values for `indexing` are 'xy' and 'ij'.");
  }
  const xs = args as ArrayInput[];
  const nd = xs.length;
  const out = xs.map((x, i) => asarray(x).reshape(axisShape(nd, i, -1)));
  if (indexing === "xy" && nd > 1) {
    out[0] = out[0]!.reshape(axisShape(nd, 1, -1));
    out[1] = out[1]!.reshape(axisShape(nd, 0, -1));
  }
  let res = out;
  if (!opts.sparse) {
    const shape = broadcastShapes(...out.map((a) => a.shape));
    res = out.map((a) => broadcastTo(a, shape));
  }
  return opts.copy ?? true ? res.map((a) => a.copy()) : res;
}

/** A grid slice `[start, stop, step?]`; a complex step `np.complex(0, n)` means n points, stop inclusive (D-100). */
export type GridSlice =
  | readonly [number | null, number]
  | readonly [number | null, number, number | ComplexLike | null];

interface GridAxis {
  start: number;
  stop: number;
  /** Numeric step (null = 1) or the complex point count. */
  step: number | null;
  count: number | null;
}

function gridAxis(s: GridSlice): GridAxis {
  if (!Array.isArray(s) || s.length < 2 || s.length > 3) {
    throw new ValueError("grid slices are [start, stop, step?] tuples");
  }
  const [start, stop, step] = s as readonly [number | null, number | null, number | ComplexLike | null | undefined];
  if (typeof stop !== "number") throw new ValueError("grid slices need a numeric stop");
  if (start !== null && typeof start !== "number") throw new ValueError("grid slice start must be a number or null");
  if (step !== undefined && step !== null && isComplexLike(step)) {
    return { start: start ?? 0, stop, step: null, count: Math.trunc(Math.hypot(step.re, step.im ?? 0)) };
  }
  if (step !== undefined && step !== null && typeof step !== "number") {
    throw new ValueError("grid slice step must be a number, complex or null");
  }
  if (step === 0) throw new ValueError("slice step cannot be zero");
  return { start: start ?? 0, stop, step: step ?? null, count: null };
}

const isInt = (v: number): boolean => Number.isSafeInteger(v) && !Object.is(v, -0);

function nativeGrid(slices: readonly GridSlice[], sparse: boolean): NDArray | NDArray[] {
  if (slices.length === 0) throw new ValueError("mgrid/ogrid need at least one slice");
  const axes = slices.map(gridAxis);
  if (axes.length === 1 && !sparse) return grid1d(axes[0]!);
  if (axes.length === 1) return grid1d(axes[0]!);
  const allInt = axes.every((a) => a.count === null && isInt(a.start) && isInt(a.stop) && isInt(a.step ?? 1));
  const dt = allInt ? "int64" : "float64";
  const sizes = axes.map((a) => (a.count !== null ? a.count : Math.ceil((a.stop - a.start) / (a.step ?? 1))));
  const steps = axes.map((a, k) => {
    if (a.count === null) return a.step ?? 1;
    return a.count !== 1 ? (a.stop - a.start) / (sizes[k]! - 1) : 1;
  });
  const starts = axes.map((a) => a.start);
  if (sizes.some((n) => !Number.isFinite(n))) throw new ValueError("grid: cannot compute length");
  if (sparse) {
    if (sizes.some((n) => n < 0)) throw new ValueError("negative dimensions are not allowed");
    return sizes.map((n, k) =>
      wrapP07((x) => x.gridAxis(starts[k]!, steps[k]!, n, dt)).reshape(axisShape(sizes.length, k, n)),
    );
  }
  return wrapP07((x) => x.mgrid(starts, steps, sizes, dt));
}

function grid1d(a: GridAxis): NDArray {
  if (a.count === null) return arange(a.start, a.stop, a.step ?? 1);
  const step = a.count !== 1 ? (a.stop - a.start) / (a.count - 1) : 1;
  return wrapP07((x) => x.gridAxis(a.start, step, Math.max(a.count!, 0), "float64"));
}

/**
 * np.mgrid[...]: `np.mgrid([0, 3], [0, 1, np.complex(0, 5)])` is NumPy's
 * `np.mgrid[0:3, 0:1:5j]` (D-100). One slice returns a 1-D array.
 */
export function mgrid(...slices: GridSlice[]): NDArray {
  return nativeGrid(slices, false) as NDArray;
}

/** np.ogrid[...]: like `mgrid`, but returns one broadcastable (sparse) array per slice. */
export function ogrid(slice: GridSlice): NDArray;
export function ogrid(...slices: GridSlice[]): NDArray[];
export function ogrid(...slices: GridSlice[]): NDArray | NDArray[] {
  return nativeGrid(slices, true);
}

/** np.ix_(...seqs): open mesh from 1-D index sequences (boolean sequences become their nonzero indices). */
export function ix_(...seqs: ArrayInput[]): NDArray[] {
  const nd = seqs.length;
  return seqs.map((seq, k) => {
    let a = asarray(seq);
    if (!(seq instanceof NDArray) && a.size === 0) a = a.astype("int64");
    if (a.ndim !== 1) throw new ValueError("Cross index must be 1 dimensional");
    if (a.dtype.kind === "b") a = nonzero(a)[0]!;
    return a.reshape(axisShape(nd, k, a.size));
  });
}

// ---- r_ / c_ (AxisConcatenator) ----

interface SliceSpec {
  start: number | null;
  stop: number | null;
  step: number | null;
  /** Complex step `Nj`: number of points. */
  count: number | null;
}

/** Parses a Python slice string "start:stop:step" (step may be "5j" when `complexStep`). */
function parseSlice(s: string, complexStep: boolean): SliceSpec | undefined {
  if (!s.includes(":")) return undefined;
  const parts = s.split(":").map((p) => p.trim());
  if (parts.length > 3) throw new ValueError(`invalid slice '${s}'`);
  const num = (p: string | undefined): number | null => {
    if (p === undefined || p === "") return null;
    const v = Number(p);
    if (!Number.isFinite(v)) throw new ValueError(`invalid slice '${s}'`);
    return v;
  };
  const stepText = parts[2] ?? "";
  if (/[jJ]$/.test(stepText)) {
    if (!complexStep) throw new ValueError(`invalid slice '${s}'`);
    const n = num(stepText.slice(0, -1));
    return { start: num(parts[0]), stop: num(parts[1]), step: null, count: Math.trunc(Math.abs(n ?? 0)) };
  }
  return { start: num(parts[0]), stop: num(parts[1]), step: num(stepText), count: null };
}

type Scalar = number | boolean | bigint | ComplexLike;
const isScalar = (x: unknown): x is Scalar =>
  typeof x === "number" || typeof x === "boolean" || typeof x === "bigint" || isComplexLike(x);

/** NumPy result_type of array dtypes and weak (NEP 50) JS scalars. */
function resultType(dtypes: DType[], scalars: Scalar[]): DType {
  const kindOf = (v: Scalar): "b" | "i" | "f" | "c" =>
    isComplexLike(v) ? "c" : typeof v === "boolean" ? "b" : typeof v === "bigint" || isInt(v) ? "i" : "f";
  const order = { b: 0, i: 1, u: 1, f: 2, c: 3 } as const;
  if (dtypes.length === 0) {
    const k = scalars.map(kindOf).reduce((a, b) => (order[b] > order[a] ? b : a), "b" as "b" | "i" | "f" | "c");
    return toDType({ b: "bool", i: "int64", f: "float64", c: "complex128" }[k]);
  }
  let dt = dtypes.reduce((a, b) => promoteTypes(a, b));
  for (const s of scalars) {
    const k = kindOf(s);
    if (order[k] <= order[dt.kind]) continue;
    if (k === "i") dt = toDType("int64");
    else if (k === "f") dt = toDType("float64");
    else dt = toDType(["float16", "float32"].includes(dt.name) ? "complex64" : "complex128");
  }
  return dt;
}

const withNdmin = (a: NDArray, ndmin: number): NDArray =>
  a.ndim >= ndmin ? a : a.reshape([...Array<number>(ndmin - a.ndim).fill(1), ...a.shape]);

/** A part of `np.r_(...)` / `np.c_(...)`: data, a JS scalar, or a string (slice like "1:4" / "0:1:5j", or a leading directive). */
export type ConcatItem = NDArray | NestedArray | string;

function axisConcat(items: readonly ConcatItem[], axis0: number, ndmin0: number, trans0: number): NDArray {
  let axis = axis0, ndmin = ndmin0, trans1d = trans0;
  const objs: NDArray[] = [];
  const isScalarObj: boolean[] = [];
  const dtypes: DType[] = [];
  const scalars: Scalar[] = [];
  items.forEach((item, k) => {
    if (typeof item === "string") {
      const sl = parseSlice(item, true);
      if (sl !== undefined) {
        const start = sl.start ?? 0;
        let obj: NDArray;
        if (sl.count !== null) {
          if (sl.stop === null) throw new ValueError(`invalid slice '${item}'`);
          obj = linspace(start, sl.stop, sl.count);
        } else {
          if (sl.stop === null) throw new ValueError(`slice '${item}' needs a stop`);
          obj = arange(start, sl.stop, sl.step ?? 1);
        }
        if (ndmin > 1) {
          obj = withNdmin(obj, ndmin);
          if (trans1d !== -1) obj = obj.swapAxes(-1, trans1d);
        }
        objs.push(obj);
        isScalarObj.push(false);
        dtypes.push(obj.dtype);
        return;
      }
      if (k !== 0) throw new ValueError("special directives must be the first entry.");
      if (item === "r" || item === "c") {
        throw new NotImplementedError("matrix directives 'r'/'c' are not supported (no matrix class, D-100)");
      }
      if (item.includes(",")) {
        const vec = item.split(",");
        const nums = vec.slice(0, 3).map((v) => Number(v.trim()));
        if (vec.length > 3 || vec.length < 2 || nums.some((v) => !Number.isInteger(v))) {
          throw new ValueError(`unknown special directive '${item}'`);
        }
        [axis, ndmin] = [nums[0]!, nums[1]!];
        if (vec.length === 3) trans1d = nums[2]!;
        return;
      }
      const ax = Number(item.trim());
      if (item.trim() === "" || !Number.isInteger(ax)) throw new ValueError("unknown special directive");
      axis = ax;
      return;
    }
    if (isScalar(item)) {
      objs.push(array(item));
      isScalarObj.push(true);
      scalars.push(item);
      return;
    }
    const a = asarray(item);
    const itemNdim = a.ndim;
    let obj = withNdmin(a, ndmin);
    if (trans1d !== -1 && itemNdim < ndmin) {
      const k2 = ndmin - itemNdim;
      let k1 = trans1d;
      if (k1 < 0) k1 += k2 + 1;
      const def = Array.from({ length: ndmin }, (_, i) => i);
      obj = obj.transpose([...def.slice(0, k1), ...def.slice(k2), ...def.slice(k1, k2)]);
    }
    objs.push(obj);
    isScalarObj.push(false);
    dtypes.push(obj.dtype);
  });
  if (objs.length === 0) throw new ValueError("need at least one array to concatenate");
  const dt = resultType(dtypes, scalars);
  const cast = objs.map((o, i) => {
    const v = isScalarObj[i] ? array(scalars[isScalarObj.slice(0, i).filter(Boolean).length]!, { dtype: dt }) : o;
    return withNdmin(v.dtype === dt ? v : v.astype(dt), ndmin);
  });
  return wrapP07((n) => n.concatenate(cast.map((c) => c._native), axis));
}

/**
 * np.r_[...]: concatenates along the first axis. Slices are written as
 * strings ("1:4", "0:1:5j"); a leading string like "0,2" or "-1" is a NumPy
 * directive (D-100). `np.r_("1:4", 0, [7, 8])` is NumPy `np.r_[1:4, 0, [7, 8]]`.
 */
export function r_(...items: ConcatItem[]): NDArray {
  return axisConcat(items, 0, 1, -1);
}

/** np.c_[...]: like `r_` but stacks 1-D inputs as columns (directive "-1,2,0"). */
export function c_(...items: ConcatItem[]): NDArray {
  return axisConcat(items, -1, 2, 0);
}

function toSpec(spec: IndexSpec | string): IndexSpec {
  if (typeof spec !== "string") return spec;
  if (spec === ellipsis) return ellipsis;
  if (spec === "newaxis") return spec;
  const sl = parseSlice(spec, false);
  if (sl === undefined) throw new ValueError(`invalid index string '${spec}' (use slices like "1:3" or "...")`);
  return [sl.start, sl.stop, sl.step];
}

/**
 * np.s_[...]: builds index specs for `a.get(...)` (D-015, D-100); slice
 * strings like "1:3" become `[1, 3, null]`. One spec returns the spec itself.
 */
export function s_(spec: IndexSpec | string): IndexSpec;
export function s_(...specs: (IndexSpec | string)[]): IndexSpec[];
export function s_(...specs: (IndexSpec | string)[]): IndexSpec | IndexSpec[] {
  const out = specs.map(toSpec);
  return out.length === 1 ? out[0]! : out;
}

/** np.index_exp[...]: like `s_`, but always returns an array of specs. */
export function indexExp(...specs: (IndexSpec | string)[]): IndexSpec[] {
  return specs.map(toSpec);
}
