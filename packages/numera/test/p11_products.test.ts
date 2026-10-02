import { describe, expect, it } from "vitest";
import np from "../src/index.js";

describe("P11-4 products", () => {
  it("vdot", () => {
    expect(np.vdot([1, 2, 3], [4, 5, 6]).item()).toBe(32);
    expect(np.vdot([[1, 2], [3, 4]], [[1, 1], [1, 1]]).item()).toBe(10);
    const c = np.vdot(np.array([np.complex(1, 2)]), np.array([np.complex(3, 4)])).item() as {
      re: number;
      im: number;
    };
    expect([c.re, c.im]).toEqual([11, -2]);
    expect(np.vdot(np.array([100], { dtype: "int8" }), np.array([2], { dtype: "int8" })).item()).toBe(-56);
    expect(() => np.vdot([1, 2], [1, 2, 3])).toThrow(np.ValueError);
  });
  it("kron", () => {
    expect(np.kron([1, 10], [1, 2, 3]).toArray()).toEqual([1, 2, 3, 10, 20, 30]);
    expect(np.kron(np.eye(2, 2, { dtype: "int64" }), [[1, 2], [3, 4]]).toArray()).toEqual([
      [1, 2, 0, 0],
      [3, 4, 0, 0],
      [0, 0, 1, 2],
      [0, 0, 3, 4],
    ]);
    expect(np.kron([1, 2], [[1], [2]]).toArray()).toEqual([[1, 2], [2, 4]]);
    expect(np.kron(np.ones([2, 3, 1]), np.ones([4, 5])).shape).toEqual([2, 12, 5]);
    expect(np.kron(2, [1, 2]).toArray()).toEqual([2, 4]);
  });
  it("cross", () => {
    expect(np.cross([1, 2, 3], [4, 5, 6]).toArray()).toEqual([-3, 6, -3]);
    expect(np.cross([[1, 0, 0], [0, 1, 0]], [0, 0, 1]).toArray()).toEqual([[0, -1, 0], [1, 0, 0]]);
    const a = np.array([[1, 4], [2, 5], [3, 6]]);
    expect(np.cross(a, a, { axis: 0 }).shape).toEqual([3, 2]);
    expect(np.cross([1, 2, 3], [4, 5, 6], { axisa: 0, axisb: 0, axisc: 0 }).toArray()).toEqual([-3, 6, -3]);
    expect(() => np.cross([1, 2], [3, 4])).toThrow(np.ValueError);
    expect(() => np.cross([1, 2, 3], [4, 5, 6], { axisc: 1 })).toThrow(np.IndexError);
    expect(np.linalg.cross(np.ones([3, 2]), np.ones([3, 2]), { axis: 0 }).shape).toEqual([3, 2]);
  });
  it("tensordot", () => {
    const a = np.arange(60).reshape([3, 4, 5]);
    const b = np.arange(24).reshape([4, 3, 2]);
    const c = np.tensordot(a, b, { axes: [[1, 0], [0, 1]] });
    expect(c.shape).toEqual([5, 2]);
    expect(c.toArray()[0]).toEqual([4400, 4730]);
    expect(np.tensordot(np.ones([2, 3]), np.ones([3, 2]), { axes: 0 }).shape).toEqual([2, 3, 3, 2]);
    expect(np.tensordot([[1, 2], [3, 4]], [[1, 2], [3, 4]]).item()).toBe(30);
    expect(np.linalg.tensordot([1, 2], [3, 4], { axes: 1 }).item()).toBe(11);
    expect(() => np.tensordot(np.ones([2, 3]), np.ones([2, 3]), { axes: [[0], [1]] })).toThrow(np.ValueError);
    expect(() => np.tensordot(np.ones([2, 3]), np.ones([2, 3]), { axes: [[0, 0], [0, 1]] })).toThrow(np.ValueError);
  });
  it("multiDot / NDArray.dot", () => {
    const A = np.arange(6).reshape([2, 3]);
    const B = np.arange(12).reshape([3, 4]);
    const C = np.arange(8).reshape([4, 2]);
    const D = np.arange(6).reshape([2, 3]);
    expect(np.linalg.multiDot([A, B, C, D]).toArray()).toEqual(A.dot(B).dot(C).dot(D).toArray());
    expect(np.linalg.multiDot([[1, 2], [[1, 0], [0, 1]], [3, 4]]).shape).toEqual([]);
    expect(np.linalg.multiDot([[1, 2], [[1, 0], [0, 1]], [[1], [1]]]).toArray()).toEqual([3]);
    expect(np.linalg.multiDot([[1, 2], [3, 4]]).item()).toBe(11);
    expect(() => np.linalg.multiDot([[1]])).toThrow(np.ValueError);
    expect(() => np.linalg.multiDot([[1], np.ones([2, 1, 1]), [1]])).toThrow(np.LinAlgError);
    expect(np.array([[1, 2], [3, 4]]).dot([1, 1]).toArray()).toEqual([3, 7]);
  });
});
