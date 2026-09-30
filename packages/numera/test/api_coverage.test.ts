import { describe, expect, it } from "vitest";
// @ts-expect-error -- plain ESM script without type declarations
import { benchCalls, checkAgainstBaseline, computeCoverage, norm } from "../../../scripts/api-coverage-lib.mjs";

const inventory = {
  numpy_version: "test",
  surfaces: {
    np: { floor_divide: "ufunc", moveaxis: "function", show_config: "function", pi: "constant", cumsum: "function" },
    ndarray: { sum: "method", shape: "attribute" },
  },
};
class FakeArray {
  get shape(): number[] { return []; }
  sum(): number { return 0; }
}
const fakeNp = { floorDivide() {}, moveAxis() {}, pi: Math.PI, NDArray: FakeArray, myCumsum() {} };
const exclusions = { np: { show_config: "build introspection" } };

const cov = (over: Record<string, unknown> = {}) =>
  computeCoverage({
    np: fakeNp,
    inventory,
    exclusions,
    aliases: {},
    natSrc: "np.floorDivide(a, b); np.moveAxis(A, 0, 1); v.sum()",
    npySrc: "np.floor_divide(a, b); np.moveaxis(A, 0, 1); v.sum()",
    ...over,
  });

describe("api coverage (D-032)", () => {
  it("matches names ignoring case and underscores", () => {
    expect(norm("floor_divide")).toBe(norm("floorDivide"));
    expect(norm("moveaxis")).toBe(norm("moveAxis"));
  });

  it("counts implemented, missing and excluded names per surface", () => {
    const r = cov();
    expect(r.surfaces.np).toMatchObject({ implemented: 3, total: 4, excluded: 1, missing: ["cumsum"] });
    expect(r.surfaces.ndarray).toMatchObject({ implemented: 2, total: 2 });
    expect(r.totals).toMatchObject({ implemented: 5, total: 6 });
    expect(r.unbenchmarked).toEqual([]);
  });

  it("uses explicit aliases and reports alias targets that do not exist", () => {
    const r = cov({ aliases: { np: { cumsum: "myCumsum", moveaxis: "nope" } } });
    expect(r.surfaces.np.missing).toEqual(["moveaxis"]);
    expect(r.aliasErrors).toEqual(["np.moveaxis -> nope"]);
  });

  it("requires a benchmark in both suites for every implemented callable", () => {
    const r = cov({ npySrc: "np.floor_divide(a, b)" });
    expect(r.unbenchmarked).toEqual(["ndarray.sum [numpy]", "np.moveaxis [numpy]"]);
    expect(checkAgainstBaseline(r, { implemented: 5, benchIgnore: [] })).toHaveLength(2);
    expect(checkAgainstBaseline(r, { implemented: 5, benchIgnore: r.unbenchmarked })).toEqual([]);
    expect(checkAgainstBaseline(r, { implemented: 6, benchIgnore: r.unbenchmarked })[0]).toMatch(/dropped/);
  });

  it("does not treat member access on other objects as a module call", () => {
    expect(benchCalls("foo.np.sort(x); xnp.cumsum(y)", ["np."])).toEqual(new Set());
    expect(benchCalls("np.sort(x)", ["np."])).toEqual(new Set(["sort"]));
  });
});
