import { describe, expect, it } from "vitest";
import np, { BroadcastError, IndexError, NDArray, NotImplementedError, ValueError } from "../src/index.js";

const b = () => np.array([[0, 1, 2], [3, 4, 5]]);
const items = (it: Iterable<NDArray | NDArray[]>) =>
  [...it].map((v) => (Array.isArray(v) ? v.map((x) => x.item()) : v.item()));

describe("P3-3 ndindex / ndenumerate (D-061)", () => {
  it("ndindex", () => {
    expect([...np.ndindex(2, 2)]).toEqual([[0, 0], [0, 1], [1, 0], [1, 1]]);
    expect([...np.ndindex([2, 1])]).toEqual([[0, 0], [1, 0]]);
    expect([...np.ndindex()]).toEqual([[]]);
    expect([...np.ndindex(0, 2)]).toEqual([]);
    expect(() => [...np.ndindex(-1)]).toThrow(/negative dimensions/);
  });
  it("ndenumerate", () => {
    expect([...np.ndenumerate(np.array([[1, 2], [3, 4]]).T)]).toEqual([
      [[0, 0], 1],
      [[0, 1], 3],
      [[1, 0], 2],
      [[1, 1], 4],
    ]);
    expect([...np.ndenumerate(np.array(5))]).toEqual([[[], 5]]);
    expect([...np.ndenumerate([[true], [false]])]).toEqual([[[0, 0], true], [[1, 0], false]]);
  });
});

describe("P3-3 nditer (D-061)", () => {
  it("K order follows memory, C/F/A orders", () => {
    expect(items(np.nditer(b().T))).toEqual([0, 1, 2, 3, 4, 5]);
    expect(items(np.nditer(b().T, { order: "C" }))).toEqual([0, 3, 1, 4, 2, 5]);
    expect(items(np.nditer(b(), { order: "F" }))).toEqual([0, 3, 1, 4, 2, 5]);
    const f = np.asfortranarray(b());
    expect(items(np.nditer([f, f], { order: "A" })).map((p) => (p as number[])[0])).toEqual([0, 3, 1, 4, 2, 5]);
    expect(items(np.nditer([f, b()], { order: "A" })).map((p) => (p as number[])[0])).toEqual([0, 1, 2, 3, 4, 5]);
    const rev = b().slice([null, [null, null, -1]]);
    expect(items(np.nditer(rev))).toEqual([0, 1, 2, 3, 4, 5]);
    expect(items(np.nditer(rev, { order: "C" }))).toEqual([2, 1, 0, 5, 4, 3]);
    expect(() => np.nditer(b(), { order: "X" as never })).toThrow(/order must be one of/);
  });

  it("multi-operand broadcasting", () => {
    const rev = b().slice([null, [null, null, -1]]);
    expect(items(np.nditer([rev, b()]))).toEqual([[2, 0], [1, 1], [0, 2], [5, 3], [4, 4], [3, 5]]);
    expect(items(np.nditer([np.array([[1], [2]]), np.array([10, 20])]))).toEqual([
      [1, 10], [1, 20], [2, 10], [2, 20],
    ]);
    expect(() => np.nditer([np.zeros([2]), np.zeros([3])])).toThrow(BroadcastError);
    expect(items(np.nditer([1, 2]))).toEqual([[1, 2]]);
  });

  it("multiIndex / index tracking", () => {
    const it1 = np.nditer(b().T, { flags: ["multi_index", "c_index"] });
    const r: unknown[] = [];
    for (const _ of it1) r.push([it1.multiIndex, it1.index]);
    expect(r).toEqual([[[0, 0], 0], [[1, 0], 2], [[2, 0], 4], [[0, 1], 1], [[1, 1], 3], [[2, 1], 5]]);
    const it2 = np.nditer(b().T, { flags: ["fIndex"] });
    const r2: number[] = [];
    for (const _ of it2) r2.push(it2.index);
    expect(r2).toEqual([0, 1, 2, 3, 4, 5]);
    const it3 = np.nditer(b().slice([null, [null, null, -1]]), { flags: ["multiIndex"] });
    const r3: number[][] = [];
    for (const _ of it3) r3.push(it3.multiIndex);
    expect(r3).toEqual([[0, 2], [0, 1], [0, 0], [1, 2], [1, 1], [1, 0]]);
    const it4 = np.nditer(np.array(5), { flags: ["multi_index"] });
    expect([...it4].length).toBe(1);
  });

  it("members and positioning", () => {
    const it = np.nditer(b(), { order: "F" });
    expect([it.shape, it.ndim, it.nop, it.itersize, it.iterindex]).toEqual([[2, 3], 2, 1, 6, 0]);
    expect(it.operands[0]!.shape).toEqual([2, 3]);
    const v = it.value as NDArray;
    expect(v.ndim).toBe(0);
    expect(v.flags.writeable).toBe(false);
    const a = np.arange(6);
    const j = np.nditer(a);
    j.iterindex = 4;
    expect((j.next().value as NDArray).item()).toBe(4);
    j.reset();
    expect((j.next().value as NDArray).item()).toBe(0);
    expect(j.iternext()).toBe(true);
    expect((j.next().value as NDArray).item()).toBe(1);
    const m = np.nditer(b(), { flags: ["multi_index"] });
    m.multiIndex = [1, 0];
    expect([(m.value as NDArray).item(), m.iterindex]).toEqual([3, 3]);
    expect(() => (m.multiIndex = [5, 0])).toThrow(IndexError);
    expect(() => (m.iterindex = 6)).toThrow(IndexError);
    const k = np.nditer(np.arange(1));
    expect(k.iternext()).toBe(false);
    expect(k.finished).toBe(true);
  });

  it("errors", () => {
    expect(() => np.nditer(np.zeros([0, 3]))).toThrow(/zero-sized operands/);
    const z = np.nditer(np.zeros([0, 3]), { flags: ["zerosize_ok"] });
    expect([z.itersize, z.finished, [...z].length]).toEqual([0, true, 0]);
    expect(() => np.nditer(b(), { flags: ["c_index", "f_index"] })).toThrow(/cannot both/);
    expect(() => np.nditer(b()).multiIndex).toThrow(/not tracking a multi-index/);
    expect(() => np.nditer(b()).index).toThrow(/does not have an index/);
    expect(() => np.nditer(b(), { flags: ["bogus"] })).toThrow(ValueError);
    expect(() => np.nditer(b(), { flags: ["buffered"] })).toThrow(NotImplementedError);
    expect(() => np.nditer(b(), { opFlags: [["readwrite"]] })).toThrow(NotImplementedError);
  });
});
