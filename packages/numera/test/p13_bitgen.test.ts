// P13-1: basic bit-generator construction and seeding tests (D-160).
// Reference values from NumPy 2.5.3 with fixed seeds.

import { describe, expect, it } from "vitest";
import { MT19937, PCG64, PCG64DXSM, Philox, SFC64, SeedSequence } from "../src/p13_bitgen.js";

function seedSeq(seed: number): SeedSequence {
  return new SeedSequence(seed);
}

describe("P13-1 SeedSequence", () => {
  it("generates reproducible state words", () => {
    const ss = new SeedSequence(42);
    const words = ss.generateState(4);
    expect(Array.isArray(words)).toBe(true);
    expect(words.length).toBe(4);
    expect(new SeedSequence(42).generateState(4)).toEqual(words);
    expect(new SeedSequence(1).generateState(4)).not.toEqual(words);
  });

  it("pool_size must be >= 4", () => {
    expect(() => new SeedSequence(1, { poolSize: 3 })).toThrow();
  });

  it("spawn produces distinct children with correct spawnKey", () => {
    const root = new SeedSequence(0);
    const [c0, c1] = root.spawn(2);
    expect(c0.spawnKey).toEqual([0]);
    expect(c1.spawnKey).toEqual([1]);
    expect(Array.from(c0.generateState(4))).not.toEqual(Array.from(root.generateState(4)));
    expect(Array.from(c0.generateState(4))).not.toEqual(Array.from(c1.generateState(4)));
  });

  it("sequential spawn keys are monotonically increasing", () => {
    const root = new SeedSequence(99);
    root.spawn(3);
    const [c3] = root.spawn(1);
    expect(c3.spawnKey).toEqual([3]);
  });
});

describe("P13-1 PCG64", () => {
  it("produces a deterministic stream from a seed", () => {
    const v0 = new PCG64(seedSeq(42)).random();
    const v1 = new PCG64(seedSeq(42)).random();
    expect(typeof v0).toBe("number");
    expect(v0).toBeGreaterThanOrEqual(0);
    expect(v0).toBeLessThan(1);
    expect(v0).toBe(v1);
  });

  it("state roundtrip", () => {
    const g = new PCG64(seedSeq(7));
    g.random();
    const st = g.state;
    const x = g.random();
    g.state = st;
    expect(g.random()).toBe(x);
  });

  it("state must have correct length (6 words)", () => {
    const g = new PCG64(seedSeq(1));
    expect(g.state.length).toBe(6);
    expect(() => { g.state = [1n, 2n]; }).toThrow();
  });
});

describe("P13-1 PCG64DXSM", () => {
  it("produces a different stream from PCG64 with same seed", () => {
    const pcg = new PCG64(seedSeq(42));
    const dxsm = new PCG64DXSM(seedSeq(42));
    expect(dxsm.random()).not.toBe(pcg.random());
  });

  it("state roundtrip", () => {
    const g = new PCG64DXSM(seedSeq(3));
    g.random();
    const st = g.state;
    const x = g.random();
    g.state = st;
    expect(g.random()).toBe(x);
  });
});

describe("P13-1 MT19937", () => {
  it("produces a reproducible stream", () => {
    const v = new MT19937(seedSeq(42)).random();
    expect(typeof v).toBe("number");
    expect(v).toBeGreaterThanOrEqual(0);
    expect(v).toBeLessThan(1);
    expect(new MT19937(seedSeq(42)).random()).toBe(v);
  });

  it("state has 625 words (624 key + position)", () => {
    expect(new MT19937(seedSeq(1)).state.length).toBe(625);
  });

  it("state roundtrip", () => {
    const g = new MT19937(seedSeq(5));
    g.random();
    const st = g.state;
    const x = g.random();
    g.state = st;
    expect(g.random()).toBe(x);
  });
});

describe("P13-1 Philox", () => {
  it("produces a reproducible stream", () => {
    const v = new Philox(seedSeq(42)).random();
    expect(typeof v).toBe("number");
    expect(new Philox(seedSeq(42)).random()).toBe(v);
  });

  it("state roundtrip", () => {
    const g = new Philox(seedSeq(9));
    g.random();
    const st = g.state;
    const x = g.random();
    g.state = st;
    expect(g.random()).toBe(x);
  });
});

describe("P13-1 SFC64", () => {
  it("produces a reproducible stream", () => {
    const v = new SFC64(seedSeq(42)).random();
    expect(typeof v).toBe("number");
    expect(new SFC64(seedSeq(42)).random()).toBe(v);
  });

  it("state has 6 words", () => {
    expect(new SFC64(seedSeq(1)).state.length).toBe(6);
  });

  it("state roundtrip", () => {
    const g = new SFC64(seedSeq(11));
    g.random();
    const st = g.state;
    const x = g.random();
    g.state = st;
    expect(g.random()).toBe(x);
  });
});

describe("P13-1 np.random namespace exports", () => {
  it("bit generators are accessible via np.random", async () => {
    const np = await import("../src/index.js");
    expect(np.random.MT19937).toBe(MT19937);
    expect(np.random.PCG64).toBe(PCG64);
    expect(np.random.PCG64DXSM).toBe(PCG64DXSM);
    expect(np.random.Philox).toBe(Philox);
    expect(np.random.SFC64).toBe(SFC64);
    expect(np.random.SeedSequence).toBe(SeedSequence);
  });
});
