import { describe, expect, it } from "vitest";
import np, { BroadcastError, DTypeError, type NDArray, ValueError } from "../src/index.js";

describe("ufuncs (M4) + broadcasting (M5)", () => {
  it("broadcasts per PLAN §14/§54", () => {
    const r = np.add(np.ones([1000, 3]), np.array([1, 2, 3]));
    expect(r.shape).toEqual([1000, 3]);
    expect((r.toArray() as number[][])[999]).toEqual([2, 3, 4]);
    expect(np.broadcastShapes([3, 1], [4], [])).toEqual([3, 4]);
    expect(() => np.broadcastShapes([3], [4])).toThrow(BroadcastError);
    expect(() => np.add(np.ones([2, 3]), np.ones([3, 2]))).toThrow(BroadcastError);
  });

  it("broadcastTo returns a zero-stride view", () => {
    const a = np.array([1, 2, 3]);
    const b = np.broadcastTo(a, [2, 3]);
    expect(b.strides).toEqual([0, 8]);
    expect(np.mayShareMemory(a, b)).toBe(true);
    expect(b.toArray()).toEqual([[1, 2, 3], [1, 2, 3]]);
  });

  it("accepts nested lists and NEP 50 weak scalars", () => {
    expect(np.multiply([1, 2], [3, 4]).toArray()).toEqual([3, 8]);
    expect(np.add(np.ones(2, { dtype: "int8" }), 3).dtype).toBe(np.int8);
    expect(np.add(2, np.ones(2, { dtype: "float32" })).dtype).toBe(np.float32);
    expect(np.add(np.ones(2, { dtype: "int32" }), 0.5).dtype).toBe(np.float64);
    expect(np.add(np.array([true]), 1).dtype).toBe(np.int64);
    expect(() => np.add(np.ones(2, { dtype: "uint8" }), -1)).toThrow(ValueError);
    expect(np.subtract(5, 2).toArray()).toBe(3);
  });

  it("reports unsupported loops with typed errors", () => {
    expect(() => np.subtract([true], [false])).toThrow(DTypeError);
    expect(() => np.negative([true])).toThrow(DTypeError);
    expect(() => np.power([2], [-1])).toThrow(ValueError);
    // P1 (D-033): complex arithmetic is supported; complex mod has no loop.
    expect(() => np.mod(np.zeros(2, { dtype: "complex128" }), 1)).toThrow(DTypeError);
    expect(() => np.floorDivide(np.zeros(2, { dtype: "complex64" }), 1)).toThrow(DTypeError);
  });

  it("works on non-contiguous inputs and returns C-contiguous results", () => {
    const t = np.arange(6).reshape([2, 3]).T;
    const r = np.multiply(t, 2);
    expect(r.flags.cContiguous).toBe(true);
    expect(r.toArray()).toEqual([[0, 6], [2, 8], [4, 10]]);
    expect(np.sqrt(np.array([4, 9], { dtype: "uint8" })).dtype).toBe(np.float16);
    expect(np.abs([-3, 3]).toArray()).toEqual([3, 3]);
    expect(np.floorDivide([-7], [2]).toArray()).toEqual([-4]);
    expect(np.mod([-7], [2]).toArray()).toEqual([1]);
    expect(np.divide([1], [0]).toArray()).toEqual([Infinity]);
  });
});

describe("ufunc out= (P2-3, D-046/D-047)", () => {
  it("writes into out and returns the same object", () => {
    const out = np.zeros([2, 3]);
    const r = np.add(np.ones([2, 3]), [1, 2, 3], { out });
    expect(r).toBe(out);
    expect(out.toArray()).toEqual([[2, 3, 4], [2, 3, 4]]);
    const u = np.zeros(2);
    expect(np.sqrt([4, 9], { out: u })).toBe(u);
    expect(u.toArray()).toEqual([2, 3]);
    // null / undefined allocate, like NumPy out=None.
    const a = np.ones(2);
    expect(np.negative(a, { out: null })).not.toBe(a);
    expect(np.add(a, 1, { out: undefined }).toArray()).toEqual([2, 2]);
  });

  it("casts the loop result to out.dtype under same_kind", () => {
    const i16 = np.zeros(2, { dtype: "int16" });
    np.add(np.array([100, 100], { dtype: "int8" }), np.array([100, 100], { dtype: "int8" }), { out: i16 });
    expect(i16.toArray()).toEqual([-56, -56]); // computed in int8, as NumPy
    const f32 = np.zeros(2, { dtype: "float32" });
    np.add([1.5, 2], 1, { out: f32 });
    expect(f32.toArray()).toEqual([2.5, 3]);
    expect(() => np.add([1.5], 1, { out: np.zeros(1, { dtype: "int64" }) })).toThrow(DTypeError);
    expect(() => np.divide([1], [2], { out: np.zeros(1, { dtype: "int64" }) })).toThrow(DTypeError);
    expect(() => np.abs(np.array([-1], { dtype: "int8" }), { out: np.zeros(1, { dtype: "uint8" }) })).toThrow(
      DTypeError,
    );
  });

  it("handles strided, in-place and overlapping out like NumPy", () => {
    const a = np.zeros(6);
    np.add(np.ones(3), 2, { out: a.slice([[0, null, 2]]) });
    expect(a.toArray()).toEqual([3, 0, 3, 0, 3, 0]);
    const b = np.arange(6).astype("float64");
    np.add(b, b, { out: b });
    expect(b.toArray()).toEqual([0, 2, 4, 6, 8, 10]);
    const c = np.arange(6).astype("float64");
    np.add(c.slice([[null, -1]]), c.slice([[1, null]]), { out: c.slice([[1, null]]) });
    expect(c.toArray()).toEqual([0, 1, 3, 5, 7, 9]);
    const d = np.arange(6).astype("float64");
    np.negative(d.slice([[null, null, -1]]), { out: d });
    expect(d.toArray()).toEqual([-5, -4, -3, -2, -1, -0]);
  });

  it("raises NumPy's errors for bad out", () => {
    expect(() => np.add(np.ones([2, 3]), 1, { out: np.zeros(3) })).toThrow(BroadcastError);
    expect(() => np.add(np.ones(3), 1, { out: np.zeros(4) })).toThrow(BroadcastError);
    expect(() => np.add(np.ones(3), 1, { out: np.broadcastTo(np.zeros(1), [3]) })).toThrow(ValueError);
    const bad = [0, 0] as unknown as NDArray;
    expect(() => np.add([1, 2], 1, { out: bad })).toThrow(DTypeError);
    expect(() => np.exp([1], { out: 3 as unknown as NDArray })).toThrow(DTypeError);
  });
});


// Expected values from NumPy 2.5.3 (D-048).
describe("ufunc dtype= / casting= (P2-4, D-048)", () => {
  const i8 = () => np.array([100, 2], { dtype: "int8" });
  const c = () => np.array([{ re: 3, im: 4 }, { re: 1, im: -1 }]);

  it("dtype= picks the loop and casts the inputs", () => {
    const r = np.add(i8(), i8(), { dtype: "int16" });
    expect(r.dtype.name).toBe("int16");
    expect(r.toArray()).toEqual([200, 4]);
    expect(np.divide([1, 2], 4, { dtype: "float32" }).dtype.name).toBe("float32");
    expect(np.sqrt(np.array([4], { dtype: "int64" }), { dtype: "float32" }).toArray()).toEqual([2]);
    expect(np.add([1], [2], { dtype: null }).dtype.name).toBe("int64");
  });

  it("makes JS scalars weak relative to dtype", () => {
    expect(np.add(i8(), 1000, { dtype: "int16" }).toArray()).toEqual([1100, 1002]);
    expect(np.add(i8(), 1.5, { dtype: "float32" }).toArray()).toEqual([101.5, 3.5]);
    expect(() => np.add(i8(), 1.5, { dtype: "int8" })).toThrow(DTypeError);
    expect(np.add(2.5, i8(), { dtype: "int16", casting: "unsafe" }).toArray()).toEqual([102, 4]);
  });

  it("abs of complex with a real dtype", () => {
    expect(np.abs(c(), { dtype: "float64" }).toArray()).toEqual([5, Math.SQRT2]);
    expect(() => np.abs(c(), { dtype: "float32" })).toThrow(DTypeError);
    expect(np.abs(c(), { dtype: "float32", casting: "unsafe" }).toArray()).toEqual([3, 1]);
    expect(() => np.abs(c(), { dtype: "complex128" })).toThrow(/No loop matching/);
  });

  it("casting= applies to inputs and out", () => {
    expect(np.add(i8(), 1, { casting: "no" }).dtype.name).toBe("int8");
    expect(() => np.add(i8(), 1.5, { casting: "no" })).toThrow(
      "Cannot cast ufunc 'add' input 0 from int8 to float64 with casting rule 'no'",
    );
    expect(() => np.sqrt(i8(), { casting: "no" })).toThrow(/input from int8 to float16/);
    const out = np.zeros(2, { dtype: "int8" });
    expect(() => np.add([1.5, 2], 1, { out })).toThrow(/output from float64 to int8/);
    expect(np.add([1.5, 2], 1, { out, casting: "unsafe" })).toBe(out);
    expect(out.toArray()).toEqual([2, 3]);
    expect(() => np.add(np.array([1]), 1, { out, casting: "safe" })).toThrow(/rule 'safe'/);
  });

  it("raises NumPy's errors for missing loops and bad casting names", () => {
    expect(() => np.divide([1], [2], { dtype: "int8" })).toThrow(/No loop matching.*divide/);
    expect(() => np.mod([1], [2], { dtype: "complex128" })).toThrow(DTypeError);
    expect(() => np.negative([true], { dtype: "bool" })).toThrow(/boolean negative/);
    expect(() => np.add(1, 2, { casting: "bogus" as never })).toThrow(ValueError);
    expect(() => np.add(1, 2, { dtype: "nope" })).toThrow(DTypeError);
  });
});

