// P13 bit generators: SeedSequence, MT19937, PCG64, PCG64DXSM, Philox, SFC64
// Wraps the native objects exposed at addon.p13 (native/bindings/p13_binding.cpp).
// API: matching numpy.random bit generator interface (D-160).

import { randomFillSync } from "node:crypto";
import { nativeModule } from "./addon.js";
import { wrapNative, ValueError } from "./errors.js";

const MASK32 = 0xffffffffn;

function intToU32Words(v: number | bigint): number[] {
  let n = BigInt(v);
  if (n < 0n) throw new ValueError("expected non-negative integer for entropy");
  if (n === 0n) return [0];
  const out: number[] = [];
  while (n > 0n) {
    out.push(Number(n & MASK32));
    n >>= 32n;
  }
  return out;
}

function coerceToU32Words(entropy: SeedLike): number[] {
  if (Array.isArray(entropy)) {
    return (entropy as (number | bigint)[]).flatMap(intToU32Words);
  }
  return intToU32Words(entropy as number | bigint);
}

function osEntropy(nWords: number): number[] {
  const buf = new Uint32Array(nWords);
  randomFillSync(buf);
  return Array.from(buf);
}

/** Seed types accepted by SeedSequence and bit-generator constructors. */
export type SeedLike = number | bigint | readonly (number | bigint)[];

// ---- native module shape ----

interface NativeSeedSeq {
  generateState(n: number): number[];
  generateState64(n: number): bigint[];
}

interface NativeBitGen {
  random(): number;
  getState(): bigint[];
  setState(words: bigint[]): void;
}

interface P13Native {
  SeedSequence: new (entropy: number[], spawnKey: number[], poolSize: number) => NativeSeedSeq;
  MT19937: new (ss: NativeSeedSeq) => NativeBitGen;
  PCG64: new (ss: NativeSeedSeq) => NativeBitGen;
  PCG64DXSM: new (ss: NativeSeedSeq) => NativeBitGen;
  Philox: new (ss: NativeSeedSeq) => NativeBitGen;
  SFC64: new (ss: NativeSeedSeq) => NativeBitGen;
}

let _native: P13Native | null = null;
function native(): P13Native {
  return (_native ??= nativeModule<P13Native>("p13"));
}

// ---- SeedSequence ----

export interface SeedSequenceOptions {
  spawnKey?: number[];
  poolSize?: number;
}

/**
 * `numpy.random.SeedSequence` (D-160). Mixes an integer (or array of integers)
 * entropy source into an initialised pool that can seed any bit generator.
 */
export class SeedSequence {
  /** @internal */ readonly _native: NativeSeedSeq;
  readonly entropy: SeedLike | null;
  readonly spawnKey: number[];
  readonly poolSize: number;
  /** @internal */ private _nSpawned = 0;

  constructor(entropy?: SeedLike | null, options?: SeedSequenceOptions) {
    const poolSize = options?.poolSize ?? 4;
    if (poolSize < 4) throw new ValueError("poolSize must be >= 4");
    const spawnKey = options?.spawnKey ?? [];
    const entropyWords =
      entropy == null ? osEntropy(poolSize) : coerceToU32Words(entropy);
    this.entropy = entropy ?? null;
    this.spawnKey = spawnKey;
    this.poolSize = poolSize;
    this._native = wrapNative(() => new (native().SeedSequence)(entropyWords, spawnKey, poolSize));
  }

  /** Generate `n` uint32 state words (matching `numpy.random.SeedSequence.generate_state`). */
  generateState(nWords = 1): number[] {
    return wrapNative(() => this._native.generateState(nWords));
  }

  /** Generate `n` uint64 state words as BigInts. */
  generateState64(nWords = 1): bigint[] {
    return wrapNative(() => this._native.generateState64(nWords));
  }

  /**
   * Spawn `n` independent child `SeedSequence` objects.
   * Matches `numpy.random.SeedSequence.spawn(n)`.
   */
  spawn(n: number): SeedSequence[] {
    const base = this._nSpawned;
    this._nSpawned += n;
    return Array.from({ length: n }, (_, i) => {
      return new SeedSequence(this.entropy, {
        spawnKey: [...this.spawnKey, base + i],
        poolSize: this.poolSize,
      });
    });
  }
}

// ---- BitGenerator base ----

/**
 * Abstract base class for all P13 bit generators.
 * Subclasses: `MT19937`, `PCG64`, `PCG64DXSM`, `Philox`, `SFC64`.
 */
export abstract class BitGenerator {
  /** @internal */ abstract readonly _native: NativeBitGen;

  /** Generate a uniformly distributed double in [0, 1). */
  random(): number {
    return wrapNative(() => this._native.random());
  }

  /** Raw generator state as an array of BigInt words (layout is generator-specific). */
  get state(): bigint[] {
    return wrapNative(() => this._native.getState());
  }

  set state(words: bigint[]) {
    wrapNative(() => this._native.setState(words));
  }
}

// ---- Concrete bit generators ----

function makeSeedSeqNative(seed?: SeedLike | SeedSequence | null): NativeSeedSeq {
  if (seed instanceof SeedSequence) return seed._native;
  const ss = new SeedSequence(seed ?? undefined);
  return ss._native;
}

/** `numpy.random.MT19937` bit generator (D-160). */
export class MT19937 extends BitGenerator {
  /** @internal */ readonly _native: NativeBitGen;

  constructor(seed?: SeedLike | SeedSequence | null) {
    super();
    const ss = makeSeedSeqNative(seed);
    this._native = wrapNative(() => new (native().MT19937)(ss));
  }
}

/** `numpy.random.PCG64` bit generator (D-160). */
export class PCG64 extends BitGenerator {
  /** @internal */ readonly _native: NativeBitGen;

  constructor(seed?: SeedLike | SeedSequence | null) {
    super();
    const ss = makeSeedSeqNative(seed);
    this._native = wrapNative(() => new (native().PCG64)(ss));
  }
}

/** `numpy.random.PCG64DXSM` bit generator (D-160). */
export class PCG64DXSM extends BitGenerator {
  /** @internal */ readonly _native: NativeBitGen;

  constructor(seed?: SeedLike | SeedSequence | null) {
    super();
    const ss = makeSeedSeqNative(seed);
    this._native = wrapNative(() => new (native().PCG64DXSM)(ss));
  }
}

/** `numpy.random.Philox` bit generator (D-160). */
export class Philox extends BitGenerator {
  /** @internal */ readonly _native: NativeBitGen;

  constructor(seed?: SeedLike | SeedSequence | null) {
    super();
    const ss = makeSeedSeqNative(seed);
    this._native = wrapNative(() => new (native().Philox)(ss));
  }
}

/** `numpy.random.SFC64` bit generator (D-160). */
export class SFC64 extends BitGenerator {
  /** @internal */ readonly _native: NativeBitGen;

  constructor(seed?: SeedLike | SeedSequence | null) {
    super();
    const ss = makeSeedSeqNative(seed);
    this._native = wrapNative(() => new (native().SFC64)(ss));
  }
}
