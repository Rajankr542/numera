// P3-3: ndindex, ndenumerate, nditer (D-061).
import { asarray } from "./creation.js";
import { IndexError, NotImplementedError, ValueError, wrapNative } from "./errors.js";
import { NDArray, type NestedArray, type ScalarValue } from "./ndarray.js";
import { p03native } from "./p03_native.js";
import { broadcastTo } from "./ufunc.js";

type Operand = NDArray | NestedArray;

function shapeArgs(args: (number | readonly number[])[]): number[] {
  const shape = args.length === 1 && Array.isArray(args[0]) ? [...(args[0] as readonly number[])] : (args as number[]);
  for (const d of shape) {
    if (typeof d !== "number" || !Number.isInteger(d)) throw new ValueError("shape dimensions must be integers");
    if (d < 0) throw new ValueError("negative dimensions are not allowed");
  }
  return shape;
}

function* cOrder(shape: readonly number[]): Generator<number[]> {
  if (shape.some((d) => d === 0)) return;
  const idx = new Array<number>(shape.length).fill(0);
  for (;;) {
    yield [...idx];
    let d = shape.length - 1;
    while (d >= 0) {
      if (++idx[d]! < shape[d]!) break;
      idx[d] = 0;
      d--;
    }
    if (d < 0) return;
  }
}

/**
 * NumPy `ndindex`: every index tuple of `shape` in C order. Takes the
 * dimensions as arguments or one shape array; `ndindex()` yields `[]` once.
 */
export function ndindex(...shape: (number | readonly number[])[]): Generator<number[]> {
  return cOrder(shapeArgs(shape));
}

/** NumPy `ndenumerate`: `[index, value]` pairs in C order (values are JS scalars, D-005). */
export function* ndenumerate(a: Operand): Generator<[number[], ScalarValue]> {
  const x = a instanceof NDArray ? a : asarray(a);
  for (const idx of cOrder(x.shape)) yield [idx, x.item(...idx)];
}

const SUPPORTED_FLAGS: Record<string, string> = {
  multi_index: "multi_index",
  multiindex: "multi_index",
  c_index: "c_index",
  cindex: "c_index",
  f_index: "f_index",
  findex: "f_index",
  zerosize_ok: "zerosize_ok",
  zerosizeok: "zerosize_ok",
};
const UNSUPPORTED_FLAGS = new Set([
  "buffered",
  "external_loop",
  "reduce_ok",
  "refs_ok",
  "ranged",
  "delay_bufalloc",
  "grow_inner",
  "copy_if_overlap",
  "common_dtype",
]);

export interface NDIterOptions {
  /** Global flags: `multi_index`, `c_index`, `f_index`, `zerosize_ok` (camelCase also accepted). */
  flags?: readonly string[];
  /** Iteration order "C", "F", "A" or "K" (default, memory order). */
  order?: "C" | "F" | "A" | "K";
  /** Per-operand flags; only `readonly` is supported. */
  opFlags?: readonly (readonly string[])[];
}

/**
 * NumPy `nditer` (read-only, D-061). Iterating yields a 0-d read-only view per
 * step: one `NDArray` for a single operand, otherwise an array of them.
 */
export class NDIter implements Iterable<NDArray | NDArray[]> {
  readonly operands: NDArray[];
  readonly shape: number[];
  private readonly views: NDArray[];
  private readonly axes: number[];
  private readonly flipped: boolean[];
  private readonly single: boolean;
  private readonly flags: Set<string>;
  private pos = 0;
  // NumPy: `next()` returns the current element, then later calls advance first.
  private started = false;

  /** @internal */
  constructor(op: Operand | readonly Operand[], opts: NDIterOptions = {}) {
    this.single = op instanceof NDArray || !Array.isArray(op);
    const list = this.single ? [op as Operand] : (op as readonly Operand[]);
    this.operands = list.map((o) => (o instanceof NDArray ? o : asarray(o)));
    this.flags = new Set<string>();
    for (const f of opts.flags ?? []) {
      const key = f.replace(/([a-z])([A-Z])/g, "$1_$2").toLowerCase();
      const known = SUPPORTED_FLAGS[key] ?? SUPPORTED_FLAGS[key.replace(/_/g, "")];
      if (known !== undefined) this.flags.add(known);
      else if (UNSUPPORTED_FLAGS.has(key)) throw new NotImplementedError(`nditer flag "${f}" is not supported`);
      else throw new ValueError(`Unexpected iterator global flag "${f}"`);
    }
    for (const fl of opts.opFlags ?? []) {
      for (const f of fl) {
        if (f !== "readonly") throw new NotImplementedError(`nditer op flag "${f}" is not supported (read-only only)`);
      }
    }
    if (this.flags.has("c_index") && this.flags.has("f_index")) {
      throw new ValueError("Iterator flags C_INDEX and F_INDEX cannot both be specified");
    }
    const plan = wrapNative(() =>
      p03native.iterPlan(
        this.operands.map((o) => o._native),
        opts.order,
      ),
    );
    this.shape = plan.shape;
    this.axes = plan.axes;
    this.flipped = plan.flipped;
    if (this.itersize === 0 && !this.flags.has("zerosize_ok")) {
      throw new ValueError("Iteration of zero-sized operands is not enabled");
    }
    this.views = this.operands.map((o) => broadcastTo(o, this.shape));
  }

  get ndim(): number {
    return this.shape.length;
  }
  get nop(): number {
    return this.operands.length;
  }
  get itersize(): number {
    return this.shape.reduce((p, d) => p * d, 1);
  }
  get finished(): boolean {
    return this.pos >= this.itersize;
  }

  /** Position in the iteration order (settable). */
  get iterindex(): number {
    return this.pos;
  }
  set iterindex(i: number) {
    if (!Number.isInteger(i) || i < 0 || i >= this.itersize) {
      throw new IndexError("Iterator GotoIterIndex called with an iterindex outside the iteration range.");
    }
    this.pos = i;
    this.started = false;
  }

  /** Coordinates of the current element (requires the `multi_index` flag; settable). */
  get multiIndex(): number[] {
    if (!this.flags.has("multi_index")) throw new ValueError("Iterator is not tracking a multi-index");
    this.checkActive();
    return this.coords(this.pos);
  }
  set multiIndex(idx: readonly number[]) {
    if (!this.flags.has("multi_index")) throw new ValueError("Iterator is not tracking a multi-index");
    if (idx.length !== this.ndim || idx.some((v, d) => !Number.isInteger(v) || v < 0 || v >= this.shape[d]!)) {
      throw new IndexError("Iterator GotoMultiIndex called with an out-of-bounds multi-index");
    }
    let p = 0;
    for (const ax of this.axes) {
      const n = this.shape[ax]!;
      const k = this.flipped[ax] ? n - 1 - idx[ax]! : idx[ax]!;
      p = p * n + k;
    }
    this.pos = p;
    this.started = false;
  }

  /** C- or F-order flat index of the current element (requires `c_index`/`f_index`). */
  get index(): number {
    const c = this.flags.has("c_index");
    if (!c && !this.flags.has("f_index")) throw new ValueError("Iterator does not have an index");
    this.checkActive();
    const co = this.coords(this.pos);
    let r = 0;
    if (c) for (let d = 0; d < co.length; d++) r = r * this.shape[d]! + co[d]!;
    else for (let d = co.length - 1; d >= 0; d--) r = r * this.shape[d]! + co[d]!;
    return r;
  }

  /** The current element(s) as 0-d read-only views. */
  get value(): NDArray | NDArray[] {
    this.checkActive();
    const co = this.coords(this.pos);
    const vals = this.views.map((v) => {
      const x = v.get(...co);
      x.setflags({ write: false });
      return x;
    });
    return this.single ? vals[0]! : vals;
  }

  /** Advances one step; returns false when finished (NumPy `iternext`). */
  iternext(): boolean {
    if (this.pos < this.itersize) this.pos++;
    this.started = false;
    return this.pos < this.itersize;
  }

  reset(): void {
    this.pos = 0;
    this.started = false;
  }

  next(): IteratorResult<NDArray | NDArray[]> {
    if (this.started) this.iternext();
    this.started = true;
    if (this.finished) return { value: undefined, done: true };
    return { value: this.value, done: false };
  }

  [Symbol.iterator](): Iterator<NDArray | NDArray[]> {
    return this;
  }

  private checkActive(): void {
    if (this.finished) throw new ValueError("Iterator is past the end");
  }

  private coords(p: number): number[] {
    const out = new Array<number>(this.ndim).fill(0);
    for (let k = this.axes.length - 1; k >= 0; k--) {
      const ax = this.axes[k]!;
      const n = this.shape[ax]!;
      const i = p % n;
      p = Math.floor(p / n);
      out[ax] = this.flipped[ax] ? n - 1 - i : i;
    }
    return out;
  }
}

/**
 * NumPy `nditer` (read-only subset, D-061). `op` is one array, or a JS list of
 * operands (as in NumPy, `nditer([1, 2])` has two 0-d operands).
 */
export function nditer(op: Operand | readonly Operand[], opts: NDIterOptions = {}): NDIter {
  return new NDIter(op, opts);
}
