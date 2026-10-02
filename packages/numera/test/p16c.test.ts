// Tests for P16C: np.strings and np.char (D-210–D-213).
import { describe, expect, it } from "vitest";
import np, { NDArray } from "../src/index.js";
import { StringArray } from "../src/p16c.js";

const L = (a: unknown): unknown => (a as NDArray).toArray();
const SA = (a: unknown): string[] | unknown => (a as StringArray).toArray();

describe("P16C-1: StringArray class (D-210)", () => {
  it("constructs from flat string[] with shape", () => {
    const sa = new StringArray(["a", "b", "c"], [3]);
    expect(sa.shape).toEqual([3]);
    expect(sa.ndim).toBe(1);
    expect(sa.size).toBe(3);
    expect(sa.dtype).toBe("str_");
  });

  it("scalar shape (0-d)", () => {
    const sa = new StringArray(["hello"], []);
    expect(sa.shape).toEqual([]);
    expect(sa.ndim).toBe(0);
    expect(sa.toArray()).toBe("hello");
  });

  it("1-D toArray", () => {
    const sa = new StringArray(["x", "y", "z"], [3]);
    expect(sa.toArray()).toEqual(["x", "y", "z"]);
  });

  it("flat iterator", () => {
    const sa = new StringArray(["a", "b"], [2]);
    const items = [...sa.flat()];
    expect(items).toEqual(["a", "b"]);
  });

  it("toString contains dtype", () => {
    const sa = new StringArray(["hi"], [1]);
    expect(sa.toString()).toContain("str_");
  });

  it("bytes_ dtype", () => {
    const sa = new StringArray(["a"], [1], "bytes_");
    expect(sa.dtype).toBe("bytes_");
  });
});

describe("P16C-2: case operations (np.strings)", () => {
  it("upper / lower", () => {
    expect(SA(np.strings.upper(["hello", "World"]))).toEqual(["HELLO", "WORLD"]);
    expect(SA(np.strings.lower(["HELLO", "World"]))).toEqual(["hello", "world"]);
  });

  it("capitalize", () => {
    expect(SA(np.strings.capitalize(["hello WORLD", ""]))).toEqual(["Hello world", ""]);
  });

  it("swapcase", () => {
    expect(SA(np.strings.swapcase(["Hello World"]))).toEqual(["hELLO wORLD"]);
  });

  it("title", () => {
    expect(SA(np.strings.title(["hello world"]))).toEqual(["Hello World"]);
  });

  it("center, ljust, rjust", () => {
    expect(SA(np.strings.center(["ab"], 6))).toEqual(["  ab  "]);
    expect(SA(np.strings.ljust(["ab"], 5))).toEqual(["ab   "]);
    expect(SA(np.strings.rjust(["ab"], 5))).toEqual(["   ab"]);
    expect(SA(np.strings.center(["a"], 4, "-"))).toEqual(["-a--"]);
  });

  it("zfill", () => {
    expect(SA(np.strings.zfill(["42"], 5))).toEqual(["00042"]);
    expect(SA(np.strings.zfill(["-42"], 5))).toEqual(["-0042"]);
    expect(SA(np.strings.zfill(["+42"], 5))).toEqual(["+0042"]);
  });

  it("strip, lstrip, rstrip", () => {
    expect(SA(np.strings.strip(["  hi  "]))).toEqual(["hi"]);
    expect(SA(np.strings.lstrip(["  hi  "]))).toEqual(["hi  "]);
    expect(SA(np.strings.rstrip(["  hi  "]))).toEqual(["  hi"]);
    expect(SA(np.strings.strip(["xxhixx"], "x"))).toEqual(["hi"]);
  });

  it("expandtabs", () => {
    expect(SA(np.strings.expandtabs(["\tabc"], 4))).toEqual(["    abc"]);
    expect(SA(np.strings.expandtabs(["ab\tc"], 4))).toEqual(["ab  c"]);
  });
});

describe("P16C-3: find / test operations (np.strings)", () => {
  it("find and rfind", () => {
    expect(L(np.strings.find(["hello"], "l"))).toEqual([2]);
    expect(L(np.strings.rfind(["hello"], "l"))).toEqual([3]);
    expect(L(np.strings.find(["hello"], "x"))).toEqual([-1]);
  });

  it("index and rindex throw on missing", () => {
    expect(() => np.strings.index(["hello"], "x")).toThrow();
    expect(() => np.strings.rindex(["hello"], "x")).toThrow();
    expect(L(np.strings.index(["hello"], "e"))).toEqual([1]);
  });

  it("count", () => {
    expect(L(np.strings.count(["hello"], "l"))).toEqual([2]);
    expect(L(np.strings.count(["abcabc"], "abc"))).toEqual([2]);
  });

  it("startswith / endswith", () => {
    expect(L(np.strings.startswith(["hello"], "he"))).toEqual([true]);
    expect(L(np.strings.endswith(["hello"], "lo"))).toEqual([true]);
    expect(L(np.strings.startswith(["hello"], "lo"))).toEqual([false]);
  });

  it("str_len", () => {
    expect(L(np.strings.str_len(["hello", "hi", ""]))).toEqual([5, 2, 0]);
  });

  it("isalnum / isalpha / isdigit / isdecimal", () => {
    expect(L(np.strings.isalnum(["abc123", "abc!", ""]))).toEqual([true, false, false]);
    expect(L(np.strings.isalpha(["abc", "abc1"]))).toEqual([true, false]);
    expect(L(np.strings.isdigit(["123", "12a"]))).toEqual([true, false]);
    expect(L(np.strings.isdecimal(["123", "12a"]))).toEqual([true, false]);
  });

  it("isspace / islower / isupper / istitle", () => {
    expect(L(np.strings.isspace(["   ", "a b"]))).toEqual([true, false]);
    expect(L(np.strings.islower(["hello", "Hello"]))).toEqual([true, false]);
    expect(L(np.strings.isupper(["HELLO", "Hello"]))).toEqual([true, false]);
    expect(L(np.strings.istitle(["Hello World", "hello world"]))).toEqual([true, false]);
  });
});

describe("P16C-4: string transforms (np.strings)", () => {
  it("add", () => {
    expect(SA(np.strings.add(["hello"], [" world"]))).toEqual(["hello world"]);
  });

  it("multiply", () => {
    expect(SA(np.strings.multiply(["ab"], 3))).toEqual(["ababab"]);
    expect(SA(np.strings.multiply(["ab"], 0))).toEqual([""]);
  });

  it("replace", () => {
    expect(SA(np.strings.replace(["hello world"], "o", "0"))).toEqual(["hell0 w0rld"]);
    expect(SA(np.strings.replace(["hello world"], "o", "0", 1))).toEqual(["hell0 world"]);
  });

  it("translate (Map<string, string|null>)", () => {
    const table = new Map<string, string | null>([["a", "A"], ["b", null]]);
    expect(SA(np.strings.translate(["abcabc"], table))).toEqual(["AcAc"]);
  });

  it("encode / decode are identity stubs (D-213)", () => {
    const sa = np.strings.encode(["hello"], "utf-8") as StringArray;
    expect(SA(sa)).toEqual(["hello"]);
    const sb = np.strings.decode(["hello"], "utf-8") as StringArray;
    expect(SA(sb)).toEqual(["hello"]);
  });

  it("mod (%-formatting, D-213)", () => {
    expect(SA(np.strings.mod(["Hello %s! Value: %d"], ["world", 42]))).toEqual(["Hello world! Value: 42"]);
    expect(SA(np.strings.mod(["%s"], ["test"]))).toEqual(["test"]);
    expect(SA(np.strings.mod(["%%"],[]))).toEqual(["%"]);
  });

  it("slice (np.strings.slice)", () => {
    expect(SA(np.strings.slice(["hello"], 1, 4))).toEqual(["ell"]);
    expect(SA(np.strings.slice(["hello"], 0, 5, 2))).toEqual(["hlo"]);
  });
});

describe("P16C-5: comparisons (np.strings)", () => {
  it("equal / not_equal", () => {
    expect(L(np.strings.equal(["abc", "xyz"], ["abc", "abc"]))).toEqual([true, false]);
    expect(L(np.strings.not_equal(["abc", "xyz"], ["abc", "abc"]))).toEqual([false, true]);
  });

  it("greater / greater_equal / less / less_equal", () => {
    expect(L(np.strings.greater(["b"], ["a"]))).toEqual([true]);
    expect(L(np.strings.greater_equal(["a"], ["a"]))).toEqual([true]);
    expect(L(np.strings.less(["a"], ["b"]))).toEqual([true]);
    expect(L(np.strings.less_equal(["a"], ["a"]))).toEqual([true]);
  });
});

describe("P16C-6: split / partition (np.strings)", () => {
  it("partition splits on first occurrence", () => {
    const parts = np.strings.partition(["hello world"], " ");
    expect(parts.length).toBe(1);
    expect(SA(parts[0])).toEqual(["hello", " ", "world"]);
  });

  it("rpartition splits on last occurrence", () => {
    const parts = np.strings.rpartition(["hello world"], " ");
    expect(parts.length).toBe(1);
    expect(SA(parts[0])).toEqual(["hello", " ", "world"]);
  });

  it("rstrip via strings.rstrip", () => {
    expect(SA(np.strings.rstrip(["hi   "]))).toEqual(["hi"]);
  });
});

describe("P16C-7: np.char surface", () => {
  it("char.array constructs a StringArray", () => {
    const sa = np.char.array(["foo", "bar"]);
    expect(sa).toBeInstanceOf(StringArray);
    expect(SA(sa)).toEqual(["foo", "bar"]);
  });

  it("char.asarray returns same object if dtype matches", () => {
    const sa = new StringArray(["x"], [1]);
    expect(np.char.asarray(sa)).toBe(sa);
    const sa2 = np.char.asarray(sa, "bytes_");
    expect(sa2).not.toBe(sa);
    expect(sa2.dtype).toBe("bytes_");
  });

  it("char.join", () => {
    const result = np.char.join("-", "abc") as StringArray;
    expect(result.toArray()).toBe("a-b-c");
    const result2 = np.char.join("-", ["abc", "xy"]) as StringArray;
    expect(SA(result2)).toEqual(["a-b-c", "x-y"]);
  });

  it("char.compare_chararrays (==, !=, <, >, <=, >=)", () => {
    expect(L(np.char.compare_chararrays(["abc"], ["abc"], "==", false))).toEqual([true]);
    expect(L(np.char.compare_chararrays(["abc  "], ["abc"], "==", true))).toEqual([true]);
    expect(L(np.char.compare_chararrays(["a"], ["b"], "<", false))).toEqual([true]);
    expect(L(np.char.compare_chararrays(["b"], ["a"], ">", false))).toEqual([true]);
    expect(() => np.char.compare_chararrays(["a"], ["b"], "???", false)).toThrow();
  });

  it("char.split returns StringArray per element", () => {
    const parts = np.char.split(["a b c"]);
    expect(parts.length).toBe(1);
    expect(SA(parts[0])).toEqual(["a", "b", "c"]);
  });

  it("char.rsplit returns StringArray per element", () => {
    const parts = np.char.rsplit(["a b c"], " ", 1);
    expect(parts.length).toBe(1);
    expect(SA(parts[0])).toEqual(["a b", "c"]);
  });

  it("char.splitlines returns StringArray per element", () => {
    const parts = np.char.splitlines(["a\nb\nc"]);
    expect(parts.length).toBe(1);
    expect(SA(parts[0])).toEqual(["a", "b", "c"]);
  });

  it("char mirrors strings functions", () => {
    expect(SA(np.char.upper(["hello"]))).toEqual(["HELLO"]);
    expect(SA(np.char.lower(["WORLD"]))).toEqual(["world"]);
    expect(L(np.char.equal(["x"], ["x"]))).toEqual([true]);
  });

  it("char.slice", () => {
    expect(SA(np.char.slice(["hello"], 1, 3))).toEqual(["el"]);
  });
});

describe("P16C-8: np.strings/np.char on np object", () => {
  it("np.strings is the strings namespace", () => {
    expect(typeof np.strings.upper).toBe("function");
    expect(typeof np.strings.equal).toBe("function");
    expect(typeof np.strings.str_len).toBe("function");
  });

  it("np.char is the char namespace", () => {
    expect(typeof np.char.array).toBe("function");
    expect(typeof np.char.asarray).toBe("function");
    expect(typeof np.char.join).toBe("function");
    expect(typeof np.char.compare_chararrays).toBe("function");
  });
});
