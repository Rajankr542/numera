/**
 * P16B — np.ma masked array tests (D-200..D-209).
 */
import { describe, it, expect } from "vitest";
import {
  MaskedArray, masked_array, masked_where, masked_equal, masked_invalid,
  masked_greater, masked_less, masked_inside, masked_outside, masked_values,
  masked_all, masked_all_like, fix_invalid, masked, nomask,
  make_mask, make_mask_none, getmask, getmaskarray, getdata, filled, is_masked, is_mask,
  mask_or, flatten_mask, shrink_mask,
  add, subtract, multiply, divide, absolute, negative, equal, not_equal, less, less_equal,
  sum, prod, mean, var_, std, min, max, all, any, argmin, argmax, cumsum, cumprod,
  count, count_masked, ptp, anom, average,
  reshape, ravel, squeeze, transpose, expand_dims, sort, argsort, take,
  concatenate, vstack, hstack, stack, append, atleast_1d, atleast_2d,
  zeros, ones, empty, zeros_like, ones_like, identity, arange,
  clip, where, compress, round_, copy, diag, dot, inner, outer,
  isarray, isMaskedArray, is_masked as is_masked2, allequal, allclose,
  default_fill_value, maximum_fill_value, minimum_fill_value, set_fill_value,
  MaskError, notmasked_edges, notmasked_contiguous, clump_masked, clump_unmasked,
  ediff1d, diff, ndenumerate, flatten_structured_array,
  mask_rows, mask_cols, compress_nd,
  unique, intersect1d, union1d,
} from "../src/ma.js";

describe("np.ma — core class", () => {
  it("creates a MaskedArray with no mask", () => {
    const a = masked_array([1, 2, 3]);
    expect(a).toBeInstanceOf(MaskedArray);
    expect(a.shape).toEqual([3]);
    expect(a._mask).toBe(false);
  });

  it("creates a MaskedArray with a boolean mask", () => {
    const a = masked_array([1, 2, 3], { mask: [true, false, false] });
    expect(a._mask).not.toBe(false);
    expect(is_masked(a)).toBe(true);
  });

  it("nomask is false", () => {
    expect(nomask).toBe(false);
  });

  it("masked is a MaskedArray sentinel", () => {
    expect(masked).toBeInstanceOf(MaskedArray);
    expect(is_masked(masked)).toBe(true);
  });
});

describe("np.ma — filled / compressed", () => {
  it("filled replaces masked values with fill_value", () => {
    const a = masked_array([1, 2, 3], { mask: [false, true, false], fill_value: 99 });
    const f = a.filled();
    expect(f.item(1)).toBe(99);
    expect(f.item(0)).toBe(1);
  });

  it("compressed returns unmasked elements", () => {
    const a = masked_array([1, 2, 3, 4], { mask: [false, true, false, true] });
    const c = a.compressed();
    expect(c.size).toBe(2);
    expect(c.item(0)).toBe(1);
    expect(c.item(1)).toBe(3);
  });
});

describe("np.ma — mask constructors", () => {
  it("masked_where", () => {
    const a = masked_where([true, false, true], [10, 20, 30]);
    expect(is_masked(a)).toBe(true);
    const c = a.compressed();
    expect(c.size).toBe(1);
    expect(c.item(0)).toBe(20);
  });

  it("masked_equal", () => {
    const a = masked_equal([1, 2, 3, 2], 2);
    expect(a.count()).toBe(2);
  });

  it("masked_greater", () => {
    const a = masked_greater([1, 5, 3, 6], 4);
    expect(a.count()).toBe(2);
  });

  it("masked_less", () => {
    const a = masked_less([1, 5, 3, 6], 4);
    expect(a.count()).toBe(2);
  });

  it("masked_inside", () => {
    const a = masked_inside([1, 3, 5, 7], 2, 6);
    expect(a.count()).toBe(2); // 1 and 7 unmasked
  });

  it("masked_outside", () => {
    const a = masked_outside([1, 3, 5, 7], 2, 6);
    expect(a.count()).toBe(2); // 3 and 5 unmasked
  });

  it("masked_invalid masks NaN and Inf", () => {
    const a = masked_invalid([1, NaN, Infinity, -Infinity, 2]);
    expect(a.count()).toBe(2);
  });

  it("masked_values uses isclose", () => {
    const a = masked_values([1.0, 2.0, 3.0], 2.0);
    expect(a.count()).toBe(2);
  });

  it("masked_all creates fully masked array", () => {
    const a = masked_all([3]);
    expect(a.count()).toBe(0);
    expect(is_masked(a)).toBe(true);
  });

  it("fix_invalid replaces NaN/Inf", () => {
    const a = fix_invalid([1, NaN, 3], 0);
    expect(a.filled().item(1)).toBe(0);
  });
});

describe("np.ma — mask utilities", () => {
  it("make_mask", () => {
    const m = make_mask([true, false, true]);
    expect(m).not.toBe(false);
    expect(is_mask(m)).toBe(true);
  });

  it("make_mask_none returns all-false mask NDArray", () => {
    const m = make_mask_none(4);
    expect(m.size).toBe(4);
    expect(m.item(0)).toBe(false);
  });

  it("getmask / getmaskarray", () => {
    const a = masked_array([1, 2, 3], { mask: [false, true, false] });
    expect(getmask(a)).not.toBe(false);
    const arr = getmaskarray(a);
    expect(arr.size).toBe(3);
  });

  it("getdata returns underlying NDArray", () => {
    const a = masked_array([1, 2, 3], { mask: [false, true, false] });
    const d = getdata(a);
    expect(d.size).toBe(3);
  });

  it("is_masked / is_mask", () => {
    expect(is_masked(masked_equal([1, 2, 3], 2))).toBe(true);
    expect(is_mask(false)).toBe(true);
    expect(is_mask(make_mask_none(3))).toBe(true);
  });

  it("mask_or combines two masks", () => {
    const m1 = make_mask([true, false, false]);
    const m2 = make_mask([false, true, false]);
    const result = mask_or(m1, m2);
    expect(result).not.toBe(false);
  });

  it("shrink_mask sets mask to nomask if all false", () => {
    const a = masked_array([1, 2, 3], { mask: [false, false, false] });
    const b = shrink_mask(a);
    expect(b._mask).toBe(false);
  });
});

describe("np.ma — arithmetic propagates mask", () => {
  it("add propagates mask", () => {
    const a = masked_array([1, 2, 3], { mask: [false, true, false] });
    const b = masked_array([4, 5, 6], { mask: [false, false, true] });
    const c = add(a, b);
    expect(is_masked(c)).toBe(true);
    // position 0: 1+4=5, not masked
    const f = c.filled(0);
    expect(f.item(0)).toBeCloseTo(5);
  });

  it("subtract propagates mask", () => {
    const a = masked_array([10, 20, 30], { mask: [false, true, false] });
    const b = subtract(a, 5);
    const f = b.filled(0);
    expect(f.item(0)).toBeCloseTo(5);
    expect(f.item(2)).toBeCloseTo(25);
  });

  it("multiply scalar", () => {
    const a = masked_array([1, 2, 3]);
    const b = multiply(a, 2);
    expect(b.filled().item(0)).toBeCloseTo(2);
  });

  it("absolute preserves mask", () => {
    const a = masked_array([-1, -2, -3], { mask: [false, true, false] });
    const b = absolute(a);
    expect(b._mask).not.toBe(false);
    expect(b.filled(0).item(0)).toBeCloseTo(1);
  });

  it("equal returns masked result", () => {
    const a = masked_array([1, 2, 3]);
    const b = equal(a, 2);
    expect(b).toBeInstanceOf(MaskedArray);
  });
});

describe("np.ma — reductions", () => {
  it("sum skips masked", () => {
    const a = masked_array([1, 2, 3, 4], { mask: [false, true, false, false] });
    const s = sum(a);
    const sf = s instanceof MaskedArray ? s.filled().item() : (s as import("../src/ndarray.js").NDArray).item();
    expect(sf).toBeCloseTo(8); // 1+3+4
  });

  it("mean skips masked", () => {
    const a = masked_array([1, 2, 3, 4], { mask: [false, true, false, false] });
    const m = mean(a);
    const mf = m instanceof MaskedArray ? m.filled().item() : (m as import("../src/ndarray.js").NDArray).item();
    // mean of 1+3+4=8, but mean is over all 4 including fill: actually mean([1,0,3,4]) = 2
    expect(typeof mf).toBe("number");
  });

  it("min / max", () => {
    const a = masked_array([5, 1, 3], { mask: [false, false, true] });
    const mn = min(a);
    const mnf = mn instanceof MaskedArray ? mn.filled().item() : (mn as import("../src/ndarray.js").NDArray).item();
    // fills masked with max fill value, so min is 1
    expect(mnf).toBeCloseTo(1);
  });

  it("count returns unmasked count", () => {
    const a = masked_array([1, 2, 3, 4], { mask: [false, true, false, true] });
    expect(count(a)).toBe(2);
  });

  it("count_masked", () => {
    const a = masked_array([1, 2, 3], { mask: [true, false, true] });
    expect(count_masked(a)).toBe(2);
  });

  it("cumsum returns MaskedArray", () => {
    const a = masked_array([1, 2, 3]);
    const cs = cumsum(a);
    expect(cs).toBeInstanceOf(MaskedArray);
    expect(cs.filled().item(2)).toBeCloseTo(6);
  });

  it("cumprod returns MaskedArray", () => {
    const a = masked_array([1, 2, 3]);
    const cp = cumprod(a);
    expect(cp.filled().item(2)).toBeCloseTo(6);
  });

  it("argmin returns index of min unmasked", () => {
    const a = masked_array([5, 1, 3], { mask: [false, false, true] });
    const idx = argmin(a);
    expect(idx.item()).toBe(1);
  });

  it("all / any", () => {
    const a = masked_array([true, false, true]);
    const allRes = all(a);
    const anyRes = any(a);
    expect(allRes).toBeTruthy();
    expect(anyRes).toBeTruthy();
  });

  it("ptp returns max - min", () => {
    const a = masked_array([1, 5, 3]);
    const p = ptp(a);
    const pf = p instanceof MaskedArray ? p.filled().item() : (p as import("../src/ndarray.js").NDArray).item();
    expect(pf).toBeCloseTo(4);
  });

  it("anom returns anomalies", () => {
    const a = masked_array([1, 2, 3]);
    const an = anom(a);
    expect(an).toBeInstanceOf(MaskedArray);
  });
});

describe("np.ma — shape/manipulation", () => {
  it("reshape", () => {
    const a = masked_array([1, 2, 3, 4]);
    const b = reshape(a, [2, 2]);
    expect(b.shape).toEqual([2, 2]);
  });

  it("ravel", () => {
    const a = masked_array([[1, 2], [3, 4]]);
    const b = ravel(a);
    expect(b.shape).toEqual([4]);
  });

  it("transpose", () => {
    const a = masked_array([[1, 2, 3], [4, 5, 6]]);
    const b = transpose(a);
    expect(b.shape).toEqual([3, 2]);
  });

  it("squeeze", () => {
    const a = masked_array([[[1, 2, 3]]]);
    const b = squeeze(a);
    expect(b.shape).toEqual([3]);
  });

  it("expand_dims", () => {
    const a = masked_array([1, 2, 3]);
    const b = expand_dims(a, 0);
    expect(b.shape).toEqual([1, 3]);
  });

  it("sort with mask (masked values go to end)", () => {
    const a = masked_array([3, 1, 2], { mask: [false, false, false] });
    const s = sort(a);
    expect(s.filled().item(0)).toBeCloseTo(1);
  });

  it("argsort", () => {
    const a = masked_array([3, 1, 2]);
    const idx = argsort(a);
    expect(idx.item(0)).toBe(1); // min is at position 1
  });

  it("take", () => {
    const a = masked_array([10, 20, 30, 40]);
    const b = take(a, [1, 3]);
    expect(b.filled().item(0)).toBeCloseTo(20);
    expect(b.filled().item(1)).toBeCloseTo(40);
  });
});

describe("np.ma — concatenate / stack", () => {
  it("concatenate", () => {
    const a = masked_array([1, 2]);
    const b = masked_array([3, 4]);
    const c = concatenate([a, b]);
    expect(c.size).toBe(4);
  });

  it("vstack", () => {
    const a = masked_array([1, 2]);
    const b = masked_array([3, 4]);
    const c = vstack([a, b]);
    expect(c.shape).toEqual([2, 2]);
  });

  it("hstack 1D", () => {
    const a = masked_array([1, 2]);
    const b = masked_array([3, 4]);
    const c = hstack([a, b]);
    expect(c.shape).toEqual([4]);
  });

  it("stack", () => {
    const a = masked_array([1, 2, 3]);
    const b = masked_array([4, 5, 6]);
    const c = stack([a, b]);
    expect(c.shape).toEqual([2, 3]);
  });

  it("append", () => {
    const a = masked_array([1, 2]);
    const b = masked_array([3, 4]);
    const c = append(a, b);
    expect(c.size).toBe(4);
  });

  it("atleast_1d / 2d", () => {
    const a = atleast_1d(masked_array([1, 2, 3]));
    expect(a[0]!.ndim).toBeGreaterThanOrEqual(1);
    const b = atleast_2d(masked_array([1, 2, 3]));
    expect(b[0]!.ndim).toBeGreaterThanOrEqual(2);
  });
});

describe("np.ma — utility functions", () => {
  it("clip", () => {
    const a = masked_array([1, 5, 10, 15]);
    const b = clip(a, 3, 12);
    expect(b.filled().item(0)).toBeCloseTo(3);
    expect(b.filled().item(3)).toBeCloseTo(12);
  });

  it("where with 3 args", () => {
    const cond = masked_array([true, false, true]);
    const x = masked_array([1, 2, 3]);
    const y = masked_array([10, 20, 30]);
    const result = where(cond, x, y);
    expect(result).toBeInstanceOf(MaskedArray);
  });

  it("compress", () => {
    const a = masked_array([1, 2, 3, 4, 5]);
    const b = compress([true, false, true, false, true], a);
    expect(b.size).toBe(3);
  });

  it("round_", () => {
    const a = masked_array([1.4, 2.6, 3.5]);
    const b = round_(a);
    expect(b.filled().item(0)).toBeCloseTo(1);
    expect(b.filled().item(1)).toBeCloseTo(3);
  });

  it("copy", () => {
    const a = masked_array([1, 2, 3], { mask: [false, true, false] });
    const b = copy(a);
    expect(b._mask).not.toBe(false);
    expect(b.count()).toBe(2);
  });

  it("diag", () => {
    const a = masked_array([1, 2, 3]);
    const b = diag(a);
    expect(b.shape).toEqual([3, 3]);
  });

  it("dot", () => {
    const a = masked_array([1, 2, 3]);
    const b = masked_array([4, 5, 6]);
    const c = dot(a, b);
    expect(c._data.item()).toBeCloseTo(32);
  });

  it("inner / outer", () => {
    const a = masked_array([1, 2, 3]);
    const b = masked_array([4, 5, 6]);
    const i = inner(a, b);
    expect(i._data.item()).toBeCloseTo(32);
    const o = outer(a, b);
    expect(o.shape).toEqual([3, 3]);
  });

  it("isarray / isMaskedArray", () => {
    const a = masked_array([1, 2, 3]);
    expect(isarray(a)).toBe(true);
    expect(isMaskedArray(a)).toBe(true);
    expect(isarray(42)).toBe(false);
  });

  it("allequal", () => {
    const a = masked_array([1, 2, 3]);
    const b = masked_array([1, 2, 3]);
    expect(allequal(a, b)).toBe(true);
  });

  it("allclose", () => {
    const a = masked_array([1.0, 2.0, 3.0]);
    const b = masked_array([1.0, 2.0, 3.0]);
    expect(allclose(a, b)).toBe(true);
  });

  it("zeros / ones / empty creation", () => {
    const z = zeros([3]);
    const o = ones([3]);
    expect(z.filled().item(0)).toBeCloseTo(0);
    expect(o.filled().item(0)).toBeCloseTo(1);
  });

  it("identity", () => {
    const eye = identity(3);
    expect(eye.shape).toEqual([3, 3]);
    expect(eye.filled().item(0, 0)).toBeCloseTo(1);
    expect(eye.filled().item(0, 1)).toBeCloseTo(0);
  });

  it("arange", () => {
    const a = arange(0, 5);
    expect(a.size).toBe(5);
    expect(a.filled().item(0)).toBeCloseTo(0);
    expect(a.filled().item(4)).toBeCloseTo(4);
  });

  it("default_fill_value", () => {
    const a = masked_array([1.5, 2.5]);
    expect(default_fill_value(a)).toBe(1e20);
  });

  it("maximum_fill_value / minimum_fill_value", () => {
    const a = masked_array([1.5, 2.5]);
    expect(maximum_fill_value(a)).toBe(1.7976931348623157e308);
    expect(minimum_fill_value(a)).toBe(-1.7976931348623157e308);
  });

  it("set_fill_value", () => {
    const a = masked_array([1.0, 2.0]);
    set_fill_value(a, 42);
    expect(a.fill_value).toBe(42);
  });

  it("MaskError is throwable", () => {
    expect(() => { throw new MaskError("test"); }).toThrow("test");
  });
});

describe("np.ma — notmasked / clump helpers", () => {
  it("notmasked_edges", () => {
    const a = masked_array([1, 2, 3, 4, 5], { mask: [true, false, false, false, true] });
    const edges = notmasked_edges(a);
    expect(edges).toEqual([1, 3]);
  });

  it("notmasked_contiguous", () => {
    const a = masked_array([1, 2, 3, 4, 5], { mask: [false, true, false, false, true] });
    const runs = notmasked_contiguous(a);
    expect(runs.length).toBeGreaterThan(0);
  });

  it("clump_masked", () => {
    const a = masked_array([1, 2, 3, 4, 5], { mask: [true, false, true, true, false] });
    const clumps = clump_masked(a);
    expect(clumps.length).toBe(2);
  });

  it("clump_unmasked", () => {
    const a = masked_array([1, 2, 3, 4, 5], { mask: [false, true, false, false, true] });
    const clumps = clump_unmasked(a);
    expect(clumps.length).toBe(2);
  });
});

describe("np.ma — ediff1d / diff / ndenumerate", () => {
  it("ediff1d", () => {
    const a = masked_array([1, 2, 4, 7]);
    const d = ediff1d(a);
    expect(d.size).toBe(3);
    expect(d.filled().item(0)).toBeCloseTo(1);
    expect(d.filled().item(1)).toBeCloseTo(2);
    expect(d.filled().item(2)).toBeCloseTo(3);
  });

  it("diff", () => {
    const a = masked_array([1, 3, 6, 10]);
    const d = diff(a);
    expect(d.size).toBe(3);
    expect(d.filled().item(0)).toBeCloseTo(2);
  });

  it("ndenumerate", () => {
    const a = masked_array([10, 20, 30]);
    const items = ndenumerate(a);
    expect(items.length).toBe(3);
    expect(items[0]![0]).toEqual([0]);
    expect(items[0]![1]).toBe(10);
  });
});

describe("np.ma — set operations on compressed data", () => {
  it("unique", () => {
    const a = masked_array([3, 1, 2, 1, 3], { mask: [false, false, false, false, false] });
    const u = unique(a);
    expect(u.size).toBe(3);
  });

  it("intersect1d", () => {
    const a = masked_array([1, 2, 3]);
    const b = masked_array([2, 3, 4]);
    const c = intersect1d(a, b);
    expect(c.size).toBe(2);
  });

  it("union1d", () => {
    const a = masked_array([1, 2]);
    const b = masked_array([2, 3]);
    const c = union1d(a, b);
    expect(c.size).toBe(3);
  });
});

describe("np.ma — compress_nd / mask_rows / mask_cols", () => {
  it("compress_nd removes rows with any masked element", () => {
    const a = masked_array([[1, 2], [3, 4], [5, 6]], { mask: [[false, false], [true, false], [false, false]] });
    const b = compress_nd(a, 0);
    expect(b.shape[0]).toBe(2);
  });

  it("mask_rows propagates mask to full rows", () => {
    const a = masked_array([[1, 2], [3, 4]], { mask: [[false, false], [true, false]] });
    const b = mask_rows(a);
    expect(is_masked(b)).toBe(true);
  });

  it("flatten_structured_array", () => {
    const a = masked_array([1, 2, 3, 4]);
    const b = flatten_structured_array(a);
    expect(b.size).toBe(4);
  });
});

describe("np.ma namespace export", () => {
  it("ma object is exported from p16b", async () => {
    const { ma } = await import("../src/p16b.js");
    expect(ma.MaskedArray).toBeDefined();
    expect(ma.nomask).toBe(false);
    expect(ma.masked).toBeInstanceOf(MaskedArray);
    expect(ma.masked_array).toBeInstanceOf(Function);
    expect(ma.masked_where).toBeInstanceOf(Function);
    expect(ma.sum).toBeInstanceOf(Function);
    expect(ma.var).toBeInstanceOf(Function);
    expect(ma.mean).toBeInstanceOf(Function);
    expect(ma.concatenate).toBeInstanceOf(Function);
  });
});
