import { randomFillSync } from "node:crypto";
import { addon } from "./addon.js";
import type { NativeBitGenerator, SeedMode } from "./addon.js";
import { array } from "./creation.js";
import { dtype as toDType } from "./dtype.js";
import type { DTypeLike } from "./dtype.js";
import { NotImplementedError, ValueError, wrapNative } from "./errors.js";
import { take } from "./indexing.js";
import { NDArray } from "./ndarray.js";
import type { NestedArray } from "./ndarray.js";
import type { Complex } from "./complex.js";
import { BitGenerator, MT19937, PCG64, PCG64DXSM, Philox, SFC64, SeedSequence } from "./p13_bitgen.js";

/**
 * Random sampling (PLAN M9, DECISIONS D-019). Bit-exact with NumPy for the
 * implemented methods: `defaultRng` (Generator/PCG64/SeedSequence) and the
 * legacy `RandomState` (MT19937) behind the global `np.random.*` functions.
 */

export type Seed = number | bigint | readonly (number | bigint)[];
export type Size = number | readonly number[];

const MASK32 = 0xffffffffn;

function intToWords(v: number | bigint): number[] {
  if (typeof v === "number" && !Number.isInteger(v)) {
    throw new TypeError("SeedSequence expects int or sequence of ints for entropy not " + v);
  }
  let n = BigInt(v);
  if (n < 0n) throw new ValueError("expected non-negative integer");
  if (n === 0n) return [0];
  const out: number[] = [];
  while (n > 0n) {
    out.push(Number(n & MASK32));
    n >>= 32n;
  }
  return out;
}

/** NumPy `_coerce_to_uint32_array` for int / sequence-of-int entropy. */
function seedSeqWords(seed: Seed): number[] {
  if (Array.isArray(seed)) return (seed as (number | bigint)[]).flatMap(intToWords);
  return intToWords(seed as number | bigint);
}

// `node:crypto` rather than `globalThis.crypto`: the Web Crypto global is only
// present from Node 19 (flagged in 18), and package.json supports Node >= 18.
function osEntropy(nWords: number): number[] {
  const buf = new Uint32Array(nWords);
  randomFillSync(buf);
  return Array.from(buf);
}

function checkU32(v: number | bigint): number {
  const n = typeof v === "bigint" ? v : Number.isInteger(v) ? BigInt(v) : null;
  if (n === null) throw new TypeError("Cannot cast seed to integer");
  if (n < 0n || n > MASK32) throw new ValueError("Seed must be between 0 and 2**32 - 1");
  return Number(n);
}

/** Legacy MT19937 seeding mode + words (NumPy `_legacy_seeding`). */
function legacySeed(seed: Seed | null | undefined): [SeedMode, number[]] {
  if (seed === null || seed === undefined) {
    // NumPy fills the 624-word key from SeedSequence entropy; we use array
    // seeding from OS entropy (non-reproducible either way).
    return ["array", osEntropy(624)];
  }
  if (Array.isArray(seed)) {
    if (seed.length === 0) throw new ValueError("Seed must be non-empty");
    return ["array", (seed as (number | bigint)[]).map(checkU32)];
  }
  return ["int", [checkU32(seed as number | bigint)]];
}

function toShape(size: Size | null | undefined): number[] | null {
  if (size === null || size === undefined) return null;
  const s = typeof size === "number" ? [size] : [...size];
  for (const d of s) {
    if (!Number.isInteger(d)) throw new TypeError("size must contain integers");
  }
  return s;
}

function scalarParam(v: unknown, name: string): number {
  if (typeof v === "number") return v;
  if (typeof v === "bigint") return Number(v);
  if (typeof v === "boolean") return v ? 1 : 0;
  throw new NotImplementedError(
    `array-valued '${name}' is not supported yet; pass a scalar (D-019)`,
  );
}

function nonNegative(v: number, name: string): void {
  if (!Number.isNaN(v) && (v < 0 || Object.is(v, -0))) throw new ValueError(`${name} < 0`);
}

function isOptions(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v) && !(v instanceof NDArray);
}

/** Returns a JS scalar when `size` was omitted (D-005), else the array. */
function finish(out: NDArray, scalar: boolean): Out {
  return scalar ? out.item() : out;
}

type Out = NDArray | number | boolean | Complex;
type Population = number | bigint | NDArray | NestedArray;

function populationOf(a: Population): { pop: number; arr: NDArray | null } {
  if (typeof a === "number" || typeof a === "bigint") {
    if (typeof a === "number" && !Number.isInteger(a)) {
      throw new ValueError("a must be a sequence or an integer, not <class 'float'>");
    }
    return { pop: Number(a), arr: null };
  }
  const arr = a instanceof NDArray ? a : array(a);
  if (arr.ndim === 0) return { pop: Number(arr.item()), arr: null };
  return { pop: arr.shape[0] ?? 0, arr };
}

function sizeProduct(shape: number[] | null): number {
  return shape === null ? 1 : shape.reduce((p, d) => p * d, 1);
}

/** `numpy.random.Generator` backed by PCG64 (D-019). */
export class Generator {
  /** @internal */
  readonly _bg: NativeBitGenerator;

  /**
   * @internal Use `defaultRng()`.
   * Accepts either a legacy `NativeBitGenerator` or a P13 `BitGenerator` instance (D-160).
   * Full distribution delegation for P13 bit generators is wired in P13-2.
   */
  constructor(bg: NativeBitGenerator | BitGenerator) {
    if (bg instanceof BitGenerator) {
      // P13-1 shim: expose only scalar .random(); all other methods throw.
      // The full distribution delegation is implemented via the legacy BitGenerator
      // wrapper (addon.random.BitGenerator), not via P13 BitGenerator instances directly.
      const p13bg = bg;
      const _ni = (name: string) => () => { throw new NotImplementedError(`Generator with P13 bit generator: ${name} not supported via P13 BitGenerator constructor — use np.random.defaultRng() instead`); };
      this._bg = {
        reseed: _ni("reseed"),
        random(size: number[], dtype: string) {
          if (size.length > 0) throw new NotImplementedError("Generator with P13 bit generator: array random not supported — use np.random.defaultRng()");
          void dtype;
          const val = p13bg.random();
          return addon.fromFloat64(new Float64Array([val]), [], "float64") as ReturnType<NativeBitGenerator["random"]>;
        },
        uniform: _ni("uniform"),
        normal: _ni("normal"),
        legacyNormal: _ni("legacyNormal"),
        integers: _ni("integers"),
        shuffle: _ni("shuffle"),
        choiceIndices: _ni("choiceIndices"),
        binomial: _ni("binomial"),
        negativeBinomial: _ni("negativeBinomial"),
        poisson: _ni("poisson"),
        zipf: _ni("zipf"),
        geometric: _ni("geometric"),
        hypergeometric: _ni("hypergeometric"),
        logseries: _ni("logseries"),
        standardExponential: _ni("standardExponential"),
        exponential: _ni("exponential"),
        standardGamma: _ni("standardGamma"),
        gamma: _ni("gamma"),
        beta: _ni("beta"),
        chisquare: _ni("chisquare"),
        f: _ni("f"),
        standardCauchy: _ni("standardCauchy"),
        pareto: _ni("pareto"),
        weibull: _ni("weibull"),
        power: _ni("power"),
        laplace: _ni("laplace"),
        gumbel: _ni("gumbel"),
        logistic: _ni("logistic"),
        lognormal: _ni("lognormal"),
        rayleigh: _ni("rayleigh"),
        standardT: _ni("standardT"),
        noncentralChisquare: _ni("noncentralChisquare"),
        noncentralF: _ni("noncentralF"),
        wald: _ni("wald"),
        vonmises: _ni("vonmises"),
        triangular: _ni("triangular"),
        multinomial: _ni("multinomial"),
        dirichlet: _ni("dirichlet"),
        mvhgCount: _ni("mvhgCount"),
        mvhgMarginals: _ni("mvhgMarginals"),
        choiceP: _ni("choiceP"),
        permuted: _ni("permuted"),
        getState: _ni("getState"),
        setState: _ni("setState"),
        bytes: _ni("bytes"),
        legacyStandardExponential: _ni("legacyStandardExponential"),
        legacyExponential: _ni("legacyExponential"),
        legacyStandardGamma: _ni("legacyStandardGamma"),
        legacyGamma: _ni("legacyGamma"),
        legacyBeta: _ni("legacyBeta"),
        legacyChisquare: _ni("legacyChisquare"),
        legacyF: _ni("legacyF"),
        legacyNoncentralChisquare: _ni("legacyNoncentralChisquare"),
        legacyNoncentralF: _ni("legacyNoncentralF"),
        legacyStandardCauchy: _ni("legacyStandardCauchy"),
        legacyStandardT: _ni("legacyStandardT"),
        legacyPareto: _ni("legacyPareto"),
        legacyWeibull: _ni("legacyWeibull"),
        legacyPower: _ni("legacyPower"),
        legacyLognormal: _ni("legacyLognormal"),
        legacyRayleigh: _ni("legacyRayleigh"),
        legacyWald: _ni("legacyWald"),
        legacyVonmises: _ni("legacyVonmises"),
        legacyNegativeBinomial: _ni("legacyNegativeBinomial"),
        legacyBinomial: _ni("legacyBinomial"),
        legacyHypergeometric: _ni("legacyHypergeometric"),
        legacyZipf: _ni("legacyZipf"),
        legacyGeometric: _ni("legacyGeometric"),
        legacyLogseries: _ni("legacyLogseries"),
        legacyChoiceP: _ni("legacyChoiceP"),
      } satisfies NativeBitGenerator;
      return;
    }
    this._bg = bg;
  }

  /** Floats in [0, 1): `random(size?, dtype?)` or `random({size, dtype})`. */
  random(size?: Size | { size?: Size; dtype?: DTypeLike }, dt: DTypeLike = "float64"): Out {
    const o = isOptions(size) ? size : { size: size as Size | undefined, dtype: dt };
    const shape = toShape(o.size);
    const name = toDType(o.dtype ?? "float64").name;
    return wrapNative(() => finish(NDArray._wrap(this._bg.random(shape ?? [], name)), shape === null));
  }

  uniform(
    low: number | { low?: number; high?: number; size?: Size } = 0,
    high = 1,
    size?: Size,
  ): Out {
    const o = isOptions(low) ? low : { low, high, size };
    const lo = scalarParam(o.low ?? 0, "low");
    const hi = scalarParam(o.high ?? 1, "high");
    const rng = hi - lo;
    if (!Number.isFinite(rng)) throw new RangeError("high - low range exceeds valid bounds");
    nonNegative(rng, "high - low");
    const shape = toShape(o.size);
    return wrapNative(() => finish(NDArray._wrap(this._bg.uniform(lo, hi, shape ?? [])), shape === null));
  }

  standardNormal(size?: Size | { size?: Size; dtype?: DTypeLike }, dt: DTypeLike = "float64"): Out {
    const o = isOptions(size) ? size : { size: size as Size | undefined, dtype: dt };
    const shape = toShape(o.size);
    const name = toDType(o.dtype ?? "float64").name;
    return wrapNative(() =>
      finish(NDArray._wrap(this._bg.normal(0, 1, shape ?? [], name)), shape === null),
    );
  }

  normal(
    loc: number | { loc?: number; scale?: number; size?: Size } = 0,
    scale = 1,
    size?: Size,
  ): Out {
    const o = isOptions(loc) ? loc : { loc, scale, size };
    const l = scalarParam(o.loc ?? 0, "loc");
    const s = scalarParam(o.scale ?? 1, "scale");
    nonNegative(s, "scale");
    const shape = toShape(o.size);
    return wrapNative(() =>
      finish(NDArray._wrap(this._bg.normal(l, s, shape ?? [], "float64")), shape === null),
    );
  }

  /** `integers(low, high?, size?, dtype?, endpoint?)` or options object. */
  integers(
    low:
      | number
      | bigint
      | { low: number | bigint; high?: number | bigint | null; size?: Size; dtype?: DTypeLike; endpoint?: boolean },
    high?: number | bigint | null,
    size?: Size,
    dt: DTypeLike = "int64",
    endpoint = false,
  ): Out {
    const o = isOptions(low) ? low : { low, high, size, dtype: dt, endpoint };
    let lo = o.low;
    let hi = o.high ?? null;
    if (hi === null) {
      hi = lo;
      lo = 0;
    }
    const shape = toShape(o.size);
    const name = toDType(o.dtype ?? "int64").name;
    return wrapNative(() =>
      finish(
        NDArray._wrap(this._bg.integers(lo, hi, o.endpoint ?? false, shape ?? [], name, false)),
        shape === null,
      ),
    );
  }

  /** `choice(a, size?, replace?, p?, axis?, shuffle?)` — `p` not yet supported. */
  choice(
    a: Population,
    size?: Size | null | { size?: Size; replace?: boolean; p?: unknown; axis?: number; shuffle?: boolean },
    replace = true,
    p: unknown = null,
    axis = 0,
    shuffle = true,
  ): Out {
    const o = isOptions(size) ? size : { size: size ?? undefined, replace, p, axis, shuffle };
    if (o.p !== null && o.p !== undefined) {
      throw new NotImplementedError("choice with p is not supported yet (D-019)");
    }
    const { pop, arr } = populationOf(a);
    const ax = o.axis ?? 0;
    const popSize = arr === null ? pop : (arr.shape[ax < 0 ? arr.ndim + ax : ax] ?? 0);
    const shape = toShape(o.size);
    const n = sizeProduct(shape);
    if (arr === null && popSize <= 0 && n !== 0) {
      throw new ValueError("a must be a positive integer unless no samples are taken");
    }
    if (arr !== null && popSize === 0 && n !== 0) {
      throw new ValueError("a cannot be empty unless no samples are taken");
    }
    return wrapNative(() => {
      const idx = NDArray._wrap(
        this._bg.choiceIndices(popSize, shape ?? [], o.replace ?? true, o.shuffle ?? true),
      );
      if (arr === null) return finish(idx, shape === null);
      if (shape === null) {
        const r = take(arr, idx.reshape([]), ax);
        return r.ndim === 0 ? r.item() : r;
      }
      return take(arr, idx, ax);
    });
  }

  /** In-place shuffle along `axis` of a writeable NDArray. */
  shuffle(x: NDArray, axis = 0): void {
    if (!(x instanceof NDArray)) throw new TypeError("shuffle requires an NDArray");
    if (!x.flags.writeable) throw new ValueError("array is read-only");
    wrapNative(() => {
      const view = axis === 0 ? x : x.swapAxes(0, axis);
      this._bg.shuffle(view._native);
    });
  }

  /** Permuted copy (or `arange(n)` permutation for an integer). */
  permutation(x: number | bigint | NDArray | NestedArray, axis = 0): NDArray {
    if (typeof x === "number" || typeof x === "bigint") {
      const arr = wrapNative(() => NDArray._wrap(addon.arange(0, Number(x), 1, "int64")));
      this.shuffle(arr);
      return arr;
    }
    const src = x instanceof NDArray ? x : array(x);
    if (src.ndim === 0) throw new ValueError("x must be an integer or at least 1-dimensional");
    const out = src.copy();
    this.shuffle(out, axis);
    return out;
  }
}

/** `np.random.defaultRng(seed?)`: Generator(PCG64(SeedSequence(seed))). */
export function defaultRng(seed?: Seed | Generator | null): Generator {
  if (seed instanceof Generator) return seed;
  const words = seed === null || seed === undefined ? osEntropy(4) : seedSeqWords(seed);
  return wrapNative(() => new Generator(new addon.random.BitGenerator("pcg64", "seedseq", words)));
}

/** Legacy `numpy.random.RandomState` (MT19937, D-019). */
export class RandomState {
  /** @internal */
  readonly _bg: NativeBitGenerator;

  constructor(seed?: Seed | null) {
    const [mode, words] = legacySeed(seed);
    this._bg = wrapNative(() => new addon.random.BitGenerator("mt19937", mode, words));
  }

  seed(seed?: Seed | null): void {
    const [mode, words] = legacySeed(seed);
    wrapNative(() => this._bg.reseed(mode, words));
  }

  randomSample(size?: Size | null): Out {
    const shape = toShape(size);
    return wrapNative(() => finish(NDArray._wrap(this._bg.random(shape ?? [], "float64")), shape === null));
  }

  random(size?: Size | null): Out {
    return this.randomSample(size);
  }

  rand(...dims: number[]): Out {
    return this.randomSample(dims.length === 0 ? null : dims);
  }

  randn(...dims: number[]): Out {
    return this.standardNormal(dims.length === 0 ? null : dims);
  }

  standardNormal(size?: Size | null): Out {
    return this.normal(0, 1, size ?? undefined);
  }

  normal(
    loc: number | { loc?: number; scale?: number; size?: Size } = 0,
    scale = 1,
    size?: Size,
  ): Out {
    const o = isOptions(loc) ? loc : { loc, scale, size };
    const l = scalarParam(o.loc ?? 0, "loc");
    const s = scalarParam(o.scale ?? 1, "scale");
    nonNegative(s, "scale");
    const shape = toShape(o.size);
    return wrapNative(() =>
      finish(NDArray._wrap(this._bg.legacyNormal(l, s, shape ?? [])), shape === null),
    );
  }

  uniform(
    low: number | { low?: number; high?: number; size?: Size } = 0,
    high = 1,
    size?: Size,
  ): Out {
    const o = isOptions(low) ? low : { low, high, size };
    const lo = scalarParam(o.low ?? 0, "low");
    const hi = scalarParam(o.high ?? 1, "high");
    if (!Number.isFinite(hi - lo)) throw new RangeError("Range exceeds valid bounds");
    const shape = toShape(o.size);
    return wrapNative(() => finish(NDArray._wrap(this._bg.uniform(lo, hi, shape ?? [])), shape === null));
  }

  /** `randint(low, high?, size?, dtype?)` — masked rejection, default int64. */
  randint(
    low: number | bigint | { low: number | bigint; high?: number | bigint | null; size?: Size; dtype?: DTypeLike },
    high?: number | bigint | null,
    size?: Size,
    dt: DTypeLike = "int64",
  ): Out {
    const o = isOptions(low) ? low : { low, high, size, dtype: dt };
    let lo = o.low;
    let hi = o.high ?? null;
    if (hi === null) {
      hi = lo;
      lo = 0;
    }
    const shape = toShape(o.size);
    const name = toDType(o.dtype ?? "int64").name;
    return wrapNative(() =>
      finish(NDArray._wrap(this._bg.integers(lo, hi, false, shape ?? [], name, true)), shape === null),
    );
  }

  /** `choice(a, size?, replace?, p?)` — `p` not yet supported. */
  choice(
    a: Population,
    size?: Size | null | { size?: Size; replace?: boolean; p?: unknown },
    replace = true,
    p: unknown = null,
  ): Out {
    const o = isOptions(size) ? size : { size: size ?? undefined, replace, p };
    if (o.p !== null && o.p !== undefined) {
      throw new NotImplementedError("choice with p is not supported yet (D-019)");
    }
    const { pop, arr } = populationOf(a);
    if (arr !== null && arr.ndim !== 1) throw new ValueError("a must be 1-dimensional");
    const shape = toShape(o.size);
    const n = sizeProduct(shape);
    if (arr === null && pop <= 0 && n !== 0) {
      throw new ValueError("a must be greater than 0 unless no samples are taken");
    }
    if (arr !== null && pop === 0 && n !== 0) {
      throw new ValueError("'a' cannot be empty unless no samples are taken");
    }
    let idx: NDArray;
    if (o.replace ?? true) {
      idx = this.randint(0, pop, shape ?? []) as NDArray;
    } else {
      if (n > pop) {
        throw new ValueError("Cannot take a larger sample than population when 'replace=False'");
      }
      if (n < 0) throw new ValueError("Negative dimensions are not allowed");
      const perm = this.permutation(pop);
      idx = perm.slice([[0, n, 1]]).reshape(shape ?? []);
    }
    if (arr === null) return shape === null ? idx.item() : idx;
    const r = take(arr, idx, 0);
    return shape === null ? r.item() : r;
  }

  /** In-place shuffle along axis 0. */
  shuffle(x: NDArray): void {
    if (!(x instanceof NDArray)) throw new TypeError("shuffle requires an NDArray");
    if (!x.flags.writeable) throw new ValueError("array is read-only");
    wrapNative(() => this._bg.shuffle(x._native));
  }

  permutation(x: number | bigint | NDArray | NestedArray): NDArray {
    if (typeof x === "number" || typeof x === "bigint") {
      const arr = wrapNative(() => NDArray._wrap(addon.arange(0, Number(x), 1, "int64")));
      this.shuffle(arr);
      return arr;
    }
    const src = x instanceof NDArray ? x : array(x);
    if (src.ndim === 0) throw new ValueError("x must be an integer or at least 1-dimensional");
    const out = src.copy();
    this.shuffle(out);
    return out;
  }
}

// Legacy global state, as NumPy's `mtrand._rand` (seeded from OS entropy).
let globalState: RandomState | null = null;
const g = (): RandomState => (globalState ??= new RandomState());

/** The `np.random` namespace (legacy functions + modern API). */
export const random = {
  defaultRng,
  Generator,
  RandomState,
  BitGenerator,
  MT19937,
  PCG64,
  PCG64DXSM,
  Philox,
  SFC64,
  SeedSequence,
  seed: (seed?: Seed | null): void => g().seed(seed),
  rand: (...dims: number[]): Out => g().rand(...dims),
  randn: (...dims: number[]): Out => g().randn(...dims),
  random: (size?: Size | null): Out => g().random(size),
  randomSample: (size?: Size | null): Out => g().randomSample(size),
  standardNormal: (size?: Size | null): Out => g().standardNormal(size),
  normal: (...args: Parameters<RandomState["normal"]>): Out => g().normal(...args),
  uniform: (...args: Parameters<RandomState["uniform"]>): Out => g().uniform(...args),
  randint: (...args: Parameters<RandomState["randint"]>): Out => g().randint(...args),
  choice: (...args: Parameters<RandomState["choice"]>): Out => g().choice(...args),
  shuffle: (x: NDArray): void => g().shuffle(x),
  permutation: (x: number | bigint | NDArray | NestedArray): NDArray => g().permutation(x),
  // ── flat np.random.* wrappers (delegate to global RandomState) ──────────────
  // continuous
  beta: (...a: Parameters<RandomState["beta"]>): Out => g().beta(...a),
  chisquare: (...a: Parameters<RandomState["chisquare"]>): Out => g().chisquare(...a),
  exponential: (...a: Parameters<RandomState["exponential"]>): Out => g().exponential(...a),
  f: (...a: Parameters<RandomState["f"]>): Out => g().f(...a),
  gamma: (...a: Parameters<RandomState["gamma"]>): Out => g().gamma(...a),
  gumbel: (...a: Parameters<RandomState["gumbel"]>): Out => g().gumbel(...a),
  laplace: (...a: Parameters<RandomState["laplace"]>): Out => g().laplace(...a),
  logistic: (...a: Parameters<RandomState["logistic"]>): Out => g().logistic(...a),
  lognormal: (...a: Parameters<RandomState["lognormal"]>): Out => g().lognormal(...a),
  noncentral_chisquare: (...a: Parameters<RandomState["noncentralChisquare"]>): Out => g().noncentralChisquare(...a),
  noncentral_f: (...a: Parameters<RandomState["noncentralF"]>): Out => g().noncentralF(...a),
  pareto: (...a: Parameters<RandomState["pareto"]>): Out => g().pareto(...a),
  power: (...a: Parameters<RandomState["power"]>): Out => g().power(...a),
  rayleigh: (...a: Parameters<RandomState["rayleigh"]>): Out => g().rayleigh(...a),
  standard_cauchy: (...a: Parameters<RandomState["standardCauchy"]>): Out => g().standardCauchy(...a),
  standard_exponential: (...a: Parameters<RandomState["standardExponential"]>): Out => g().standardExponential(...a),
  standard_gamma: (...a: Parameters<RandomState["standardGamma"]>): Out => g().standardGamma(...a),
  standard_t: (...a: Parameters<RandomState["standardT"]>): Out => g().standardT(...a),
  triangular: (...a: Parameters<RandomState["triangular"]>): Out => g().triangular(...a),
  vonmises: (...a: Parameters<RandomState["vonmises"]>): Out => g().vonmises(...a),
  wald: (...a: Parameters<RandomState["wald"]>): Out => g().wald(...a),
  weibull: (...a: Parameters<RandomState["weibull"]>): Out => g().weibull(...a),
  // discrete
  binomial: (...a: Parameters<RandomState["binomial"]>): Out => g().binomial(...a),
  geometric: (...a: Parameters<RandomState["geometric"]>): Out => g().geometric(...a),
  hypergeometric: (...a: Parameters<RandomState["hypergeometric"]>): Out => g().hypergeometric(...a),
  logseries: (...a: Parameters<RandomState["logseries"]>): Out => g().logseries(...a),
  negative_binomial: (...a: Parameters<RandomState["negativeBinomial"]>): Out => g().negativeBinomial(...a),
  poisson: (...a: Parameters<RandomState["poisson"]>): Out => g().poisson(...a),
  zipf: (...a: Parameters<RandomState["zipf"]>): Out => g().zipf(...a),
  // multivariate
  multinomial: (...a: Parameters<RandomState["multinomial"]>): NDArray => g().multinomial(...a),
  dirichlet: (...a: Parameters<RandomState["dirichlet"]>): NDArray => g().dirichlet(...a),
  // state
  bytes: (length: number): Uint8Array => g().bytes(length),
  get_state: (): bigint[] => g().getState(),
  set_state: (words: bigint[]): void => g().setState(words),
  // aliases / legacy
  ranf: (size?: Size | null): Out => g().random(size),
  sample: (size?: Size | null): Out => g().random(size),
  random_integers: (...a: Parameters<RandomState["random_integers"]>): Out => g().random_integers(...a),
} as const;
