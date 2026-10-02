import { describe, expect, it } from "vitest";
import np from "../src/index.js";
import { find_duplicate, format_parser, fromarrays, fromrecords, rec, recarray } from "../src/p16e.js";

describe("P16E np.rec record arrays (D-230)", () => {
  describe("format_parser", () => {
    it("parses named dtype strings", () => {
      const fp = new format_parser(["float64", "int32"], ["x", "y"]);
      expect(fp.dtype).toHaveLength(2);
      expect(fp.dtype[0]).toMatchObject({ name: "x", dtype: "float64", count: 1 });
      expect(fp.dtype[1]).toMatchObject({ name: "y", dtype: "int32", count: 1 });
    });

    it("parses format chars", () => {
      const fp = new format_parser(["f4", "i4", "d"], ["a", "b", "c"]);
      expect(fp.dtype[0]?.dtype).toBe("float32");
      expect(fp.dtype[1]?.dtype).toBe("int32");
      expect(fp.dtype[2]?.dtype).toBe("float64");
    });

    it("parses comma-separated formats string", () => {
      const fp = new format_parser("float64,int32", ["x", "y"]);
      expect(fp.dtype).toHaveLength(2);
      expect(fp.dtype[0]?.dtype).toBe("float64");
    });

    it("generates default names f0, f1, ... when names omitted", () => {
      const fp = new format_parser(["float64", "int32"]);
      expect(fp.dtype[0]?.name).toBe("f0");
      expect(fp.dtype[1]?.name).toBe("f1");
    });

    it("parses repeat count", () => {
      const fp = new format_parser(["3f4"], ["coords"]);
      expect(fp.dtype[0]).toMatchObject({ name: "coords", dtype: "float32", count: 3 });
    });

    it("parses byte-order prefix", () => {
      const fp = new format_parser(["<f4", ">i4"], ["a", "b"]);
      expect(fp.dtype[0]?.dtype).toBe("float32");
      expect(fp.dtype[1]?.dtype).toBe("int32");
    });
  });

  describe("recarray constructor", () => {
    it("creates zero-filled columns", () => {
      const r = new recarray(3, { names: ["x", "y"], formats: ["float64", "int32"] });
      expect(r.names).toEqual(["x", "y"]);
      expect(r.shape).toEqual([3]);
      expect(r.field("x").shape).toEqual([3]);
      expect(r.field("x").dtype.name).toBe("float64");
      expect(r.field("y").dtype.name).toBe("int32");
      expect(r.field("x").toArray()).toEqual([0, 0, 0]);
    });

    it("supports attribute access for field names", () => {
      const r = new recarray(2, { names: ["a", "b"], formats: ["float32", "float64"] });
      expect((r as { a: unknown }).a).toBe(r.field("a"));
      expect((r as { b: unknown }).b).toBe(r.field("b"));
    });

    it("throws ValueError for unknown field", () => {
      const r = new recarray(2, { names: ["x"], formats: ["float64"] });
      expect(() => r.field("z")).toThrow("recarray has no field 'z'");
    });

    it("accepts 2-d shape", () => {
      const r = new recarray([2, 3], { names: ["v"], formats: ["float32"] });
      expect(r.shape).toEqual([2, 3]);
      expect(r.field("v").shape).toEqual([2, 3]);
    });
  });

  describe("fromarrays", () => {
    it("builds from NDArray columns", () => {
      const x = np.array([1, 2, 3]);
      const y = np.array([4.0, 5.0, 6.0]);
      const r = fromarrays([x, y], { names: ["x", "y"] });
      expect(r.names).toEqual(["x", "y"]);
      expect(r.field("x").toArray()).toEqual([1, 2, 3]);
      expect(r.field("y").toArray()).toEqual([4, 5, 6]);
    });

    it("generates default names when names omitted", () => {
      const r = fromarrays([np.array([1, 2]), np.array([3, 4])]);
      expect(r.names).toEqual(["f0", "f1"]);
    });

    it("parses names from comma-separated string", () => {
      const r = fromarrays([np.array([1]), np.array([2])], { names: "a,b" });
      expect(r.names).toEqual(["a", "b"]);
    });

    it("throws on empty arrayList", () => {
      expect(() => fromarrays([])).toThrow("fromarrays: arrayList must be non-empty");
    });

    it("throws on names/arrays length mismatch", () => {
      expect(() => fromarrays([np.array([1])], { names: ["a", "b"] })).toThrow("fromarrays: got 1 arrays but 2 names");
    });

    it("accepts nested arrays as columns", () => {
      const r = fromarrays([[1, 2, 3], [4.0, 5.0, 6.0]], { names: ["x", "y"] });
      expect(r.field("x").toArray()).toEqual([1, 2, 3]);
    });
  });

  describe("fromrecords", () => {
    it("builds from row tuples", () => {
      const r = fromrecords([[1, 2], [3, 4]], { names: ["a", "b"] });
      expect(r.field("a").toArray()).toEqual([1, 3]);
      expect(r.field("b").toArray()).toEqual([2, 4]);
    });

    it("generates default names", () => {
      const r = fromrecords([[1, 2], [3, 4]]);
      expect(r.names).toEqual(["f0", "f1"]);
    });

    it("throws on empty recList", () => {
      expect(() => fromrecords([], { names: ["a"] })).toThrow("fromrecords: recList must be non-empty");
    });

    it("respects formats", () => {
      const r = fromrecords([[1, 2]], { names: ["x", "y"], formats: ["float64", "int32"] });
      expect(r.field("x").dtype.name).toBe("float64");
    });
  });

  describe("rec.array (flexible constructor)", () => {
    it("delegates to fromrecords for rows of primitives", () => {
      const r = rec.array([[1, 2.0], [3, 4.0]], { names: "a,b" });
      expect(r.field("a").toArray()).toEqual([1, 3]);
    });

    it("delegates to fromarrays for array of NDArrays", () => {
      const x = np.array([10, 20]);
      const y = np.array([30, 40]);
      const r = rec.array([x, y], { names: ["p", "q"] });
      expect(r.field("p").toArray()).toEqual([10, 20]);
    });

    it("clones an existing recarray", () => {
      const orig = fromarrays([np.array([1, 2])], { names: ["v"] });
      const copy = rec.array(orig);
      expect(copy.field("v").toArray()).toEqual([1, 2]);
    });
  });

  describe("find_duplicate", () => {
    it("returns duplicate values", () => {
      expect(find_duplicate([1, 2, 1, 3, 2])).toEqual([1, 2]);
    });

    it("returns empty for no duplicates", () => {
      expect(find_duplicate([1, 2, 3])).toEqual([]);
    });

    it("handles strings", () => {
      expect(find_duplicate(["a", "b", "a"])).toEqual(["a"]);
    });

    it("handles empty list", () => {
      expect(find_duplicate([])).toEqual([]);
    });
  });

  describe("rec module shape", () => {
    it("exports all 9 required names", () => {
      expect(typeof rec.recarray).toBe("function");
      expect(typeof rec.record).toBe("function");
      expect(typeof rec.format_parser).toBe("function");
      expect(typeof rec.fromarrays).toBe("function");
      expect(typeof rec.fromrecords).toBe("function");
      expect(typeof rec.array).toBe("function");
      expect(typeof rec.find_duplicate).toBe("function");
      // fromfile/fromstring exist but throw
      expect(() => rec.fromfile(null)).toThrow("not implemented");
      expect(() => rec.fromstring(null)).toThrow("not implemented");
    });

    it("np.rec is accessible", () => {
      expect(np.rec).toBe(rec);
    });

    it("np.recarray is the class", () => {
      expect(np.recarray).toBe(recarray);
    });
  });

  describe("toRecord()", () => {
    it("returns all columns", () => {
      const r = fromarrays([np.array([1, 2]), np.array([3, 4])], { names: ["a", "b"] });
      const record = r.toRecord();
      expect(Object.keys(record)).toEqual(["a", "b"]);
      expect(record["a"]!.toArray()).toEqual([1, 2]);
    });
  });
});

describe("P16E shares_memory (D-230)", () => {
  it("returns true for a view", () => {
    const a = np.arange(6);
    const b = a.reshape([2, 3]);
    expect(np.sharesMemory(a, b)).toBe(true);
  });

  it("returns false for independent arrays", () => {
    const a = np.array([1, 2, 3]);
    const b = np.array([1, 2, 3]);
    expect(np.sharesMemory(a, b)).toBe(false);
  });

  it("returns true for an array with itself", () => {
    const a = np.arange(4);
    expect(np.sharesMemory(a, a)).toBe(true);
  });

  it("returns false for a copy", () => {
    const a = np.array([1, 2, 3]);
    const b = a.copy();
    expect(np.sharesMemory(a, b)).toBe(false);
  });

  it("respects maxWork option (no-op in numera, always exact)", () => {
    const a = np.arange(6);
    const b = a.reshape([2, 3]);
    expect(np.sharesMemory(a, b, { maxWork: 1 })).toBe(true);
  });
});
