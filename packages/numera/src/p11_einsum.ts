// P11 np.einsum / np.einsumPath (D-141). Parsing and path search are ports of
// NumPy's einsumfunc.py; the numerics run in the native einsum core.
import { ValueError, wrapNative } from "./errors.js";
import { ellipsis, NDArray } from "./ndarray.js";
import { isComplexLike } from "./complex.js";
import type { ArrayLike } from "./ufunc.js";
import { native, toArray } from "./p11_native.js";

const SYMBOLS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const SYMBOL_SET = new Set(SYMBOLS);

/** A contraction path: `["einsum_path", [i, j], ...]` as returned by `einsumPath`. */
export type EinsumPath = ["einsum_path", ...number[][]];
export type EinsumOptimize =
  | boolean
  | "greedy"
  | "optimal"
  | null
  | readonly (string | readonly number[])[]
  | readonly [string, number];
export interface EinsumOptions {
  /** Contraction order: `false` (default), `true`/`"greedy"`, `"optimal"`, an explicit path, or `[name, memoryLimit]`. */
  optimize?: EinsumOptimize;
}
export type Sublist = readonly (number | typeof ellipsis)[];

const isOptions = (v: unknown): v is EinsumOptions =>
  typeof v === "object" &&
  v !== null &&
  !Array.isArray(v) &&
  !(v instanceof NDArray) &&
  !ArrayBuffer.isView(v) &&
  !isComplexLike(v);

interface Parsed {
  inputs: string[];
  output: string;
  operands: NDArray[];
}

const sortedUnique = (s: string): string[] => [...new Set(s)].sort();
const count = (s: string, c: string): number => s.split(c).length - 1;

function subToString(sub: unknown): string {
  if (!Array.isArray(sub)) throw new ValueError("For this input type lists must contain either int or Ellipsis");
  let out = "";
  for (const s of sub as unknown[]) {
    if (s === ellipsis) out += "...";
    else if (typeof s === "number" && Number.isInteger(s) && s >= 0 && s < 52) out += SYMBOLS[s];
    else if (typeof s === "number" && Number.isInteger(s)) {
      throw new ValueError(`subscript is not within the valid range [0, 52)`);
    } else throw new ValueError("For this input type lists must contain either int or Ellipsis");
  }
  return out;
}

// Port of _parse_einsum_input (ellipsis expansion, implicit output, checks).
function parseInput(args: readonly unknown[]): Parsed {
  if (args.length === 0) throw new ValueError("No input operands");
  let subscripts: string;
  let operands: NDArray[];
  if (typeof args[0] === "string") {
    subscripts = args[0].replace(/ /g, "");
    operands = args.slice(1).map((v) => toArray(v as ArrayLike));
    for (const s of subscripts) {
      if (".,->".includes(s)) continue;
      if (!SYMBOL_SET.has(s)) {
        throw new ValueError(
          `invalid subscript '${s}' in einstein sum subscripts string, subscripts must be letters`,
        );
      }
    }
  } else {
    const tmp = [...args];
    const ops: unknown[] = [];
    const subs: unknown[] = [];
    const pairs = Math.floor(args.length / 2);
    for (let p = 0; p < pairs; p++) {
      ops.push(tmp.shift());
      subs.push(tmp.shift());
    }
    operands = ops.map((v) => toArray(v as ArrayLike));
    subscripts = subs.map(subToString).join(",");
    if (tmp.length) subscripts += "->" + subToString(tmp[tmp.length - 1]);
  }
  if (subscripts.includes("-") || subscripts.includes(">")) {
    const invalid = count(subscripts, "-") > 1 || count(subscripts, ">") > 1;
    if (invalid || count(subscripts, "->") !== 1) {
      throw new ValueError("einstein sum subscript string does not contain proper '->' output specified");
    }
  }
  let output: string;
  let inputPart: string;
  if (subscripts.includes(".")) {
    const used = new Set(subscripts.replace(/[.,]|->/g, ""));
    const ellInds = SYMBOLS.split("").filter((c) => !used.has(c)).join("");
    let longest = 0;
    const hasOut = subscripts.includes("->");
    const [inTmp, outSub = ""] = hasOut ? subscripts.split("->") : [subscripts];
    const split = inTmp!.split(",");
    split.forEach((sub, num) => {
      if (!sub.includes(".")) return;
      if (count(sub, ".") !== 3 || count(sub, "...") !== 1) {
        throw new ValueError("einstein sum subscripts string contains a '.' that is not part of an '...'");
      }
      const op = operands[num];
      if (op === undefined) return;
      let ell = op.ndim === 0 ? 0 : Math.max(op.ndim, 1) - (sub.length - 3);
      if (ell > longest) longest = ell;
      if (ell < 0) {
        throw new ValueError(`einstein sum subscripts string contains too many subscripts for operand ${num}`);
      }
      ell = Math.max(ell, 0);
      split[num] = sub.replace("...", ell === 0 ? "" : ellInds.slice(-ell));
    });
    inputPart = split.join(",");
    const outEll = longest === 0 ? "" : ellInds.slice(-longest);
    if (hasOut) {
      if (outSub.includes(".") && (count(outSub, ".") !== 3 || count(outSub, "...") !== 1)) {
        throw new ValueError("einstein sum subscripts string contains a '.' that is not part of an '...'");
      }
      output = outSub.replace("...", outEll);
    } else {
      const flat = inputPart.replace(/,/g, "");
      const once = sortedUnique(flat).filter((s) => count(flat, s) === 1);
      output = outEll + once.filter((s) => !outEll.includes(s)).join("");
    }
  } else if (subscripts.includes("->")) {
    [inputPart, output] = subscripts.split("->") as [string, string];
  } else {
    inputPart = subscripts;
    const flat = subscripts.replace(/,/g, "");
    output = sortedUnique(flat).filter((s) => count(flat, s) === 1).join("");
  }
  for (const c of output) {
    if (count(output, c) !== 1) {
      throw new ValueError(`einstein sum subscripts string includes output subscript '${c}' multiple times`);
    }
    if (!inputPart.includes(c)) {
      throw new ValueError(
        `einstein sum subscripts string included output subscript '${c}' which never appeared in an input`,
      );
    }
  }
  const inputs = inputPart.split(",");
  if (inputs.length > operands.length) {
    throw new ValueError("fewer operands provided to einstein sum function than specified in the subscripts string");
  }
  if (inputs.length < operands.length) {
    throw new ValueError("more operands provided to einstein sum function than specified in the subscripts string");
  }
  inputs.forEach((t, i) => {
    const nd = operands[i]!.ndim;
    if (t.length > nd) {
      throw new ValueError(`einstein sum subscripts string contains too many subscripts for operand ${i}`);
    }
    if (t.length < nd) {
      throw new ValueError(
        "operand has more dimensions than subscripts given in einstein sum, but no '...' ellipsis provided to broadcast the extra dimensions.",
      );
    }
  });
  return { inputs, output, operands };
}

type LSet = Set<string>;
const sizeOf = (inds: Iterable<string>, dims: Map<string, number>): number => {
  let r = 1;
  for (const i of inds) r *= dims.get(i)!;
  return r;
};
const flopCount = (contract: LSet, inner: boolean, nTerms: number, dims: Map<string, number>): number =>
  sizeOf(contract, dims) * (Math.max(1, nTerms - 1) + (inner ? 1 : 0));

function findContraction(positions: readonly number[], inputSets: readonly LSet[], outputSet: LSet) {
  const contract: LSet = new Set();
  const remain: LSet = new Set(outputSet);
  const remaining: LSet[] = [];
  inputSets.forEach((v, i) => {
    if (positions.includes(i)) for (const c of v) contract.add(c);
    else {
      remaining.push(v);
      for (const c of v) remain.add(c);
    }
  });
  const result: LSet = new Set([...remain].filter((c) => contract.has(c)));
  const removed: LSet = new Set([...contract].filter((c) => !result.has(c)));
  remaining.push(result);
  return { result, remaining, removed, contract };
}

function* combinations2(n: number): Generator<[number, number]> {
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) yield [i, j];
}

function optimalPath(inputSets: LSet[], outputSet: LSet, dims: Map<string, number>, memLimit: number): number[][] {
  let full: [number, number[][], LSet[]][] = [[0, [], inputSets]];
  for (let it = 0; it < inputSets.length - 1; it++) {
    const iter: [number, number[][], LSet[]][] = [];
    for (const [cost, positions, remaining] of full) {
      for (const con of combinations2(inputSets.length - it)) {
        const c = findContraction(con, remaining, outputSet);
        if (sizeOf(c.result, dims) > memLimit) continue;
        const total = cost + flopCount(c.contract, c.removed.size > 0, con.length, dims);
        iter.push([total, [...positions, con], c.remaining]);
      }
    }
    if (iter.length) full = iter;
    else {
      const best = full.reduce((a, b) => (b[0] < a[0] ? b : a));
      return [...best[1], Array.from({ length: inputSets.length - it }, (_, i) => i)];
    }
  }
  if (full.length === 0) return [Array.from({ length: inputSets.length }, (_, i) => i)];
  return full.reduce((a, b) => (b[0] < a[0] ? b : a))[1];
}

type Cand = [[number, number], [number, number], LSet[]];
const lessSort = (a: [number, number], b: [number, number]): boolean => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);

function greedyPath(inputSets: LSet[], outputSet: LSet, dims: Map<string, number>, memLimit: number): number[][] {
  if (inputSets.length === 1) return [[0]];
  if (inputSets.length === 2) return [[0, 1]];
  const all = Array.from({ length: inputSets.length }, (_, i) => i);
  const naive = findContraction(all, inputSets, outputSet);
  const naiveCost = flopCount(naive.contract, naive.removed.size > 0, inputSets.length, dims);
  let combIter: Iterable<[number, number]> = combinations2(inputSets.length);
  let known: Cand[] = [];
  let pathCost = 0;
  const path: number[][] = [];
  const parse = (pos: [number, number]): Cand | null => {
    const c = findContraction(pos, inputSets, outputSet);
    const newSize = sizeOf(c.result, dims);
    if (newSize > memLimit) return null;
    const removedSize = pos.reduce((s, p) => s + sizeOf(inputSets[p]!, dims), 0) - newSize;
    const cost = flopCount(c.contract, c.removed.size > 0, pos.length, dims);
    if (pathCost + cost > naiveCost) return null;
    return [[-removedSize, cost], pos, c.remaining];
  };
  const disjoint = (a: LSet, b: LSet): boolean => ![...a].some((c) => b.has(c));
  for (let it = 0; it < all.length - 1; it++) {
    for (const pos of combIter) {
      if (disjoint(inputSets[pos[0]]!, inputSets[pos[1]]!)) continue;
      const r = parse(pos);
      if (r) known.push(r);
    }
    if (known.length === 0) {
      for (const pos of combinations2(inputSets.length)) {
        const r = parse(pos);
        if (r) known.push(r);
      }
      if (known.length === 0) {
        path.push(Array.from({ length: inputSets.length }, (_, i) => i));
        break;
      }
    }
    let best = known[0]!;
    for (const k of known) if (lessSort(k[0], best[0])) best = k;
    const [bx, by] = best[1];
    const mod: Cand[] = [];
    for (const [cost, [x, y], sets] of known) {
      if (x === bx || x === by || y === bx || y === by) continue;
      sets.splice(by - +(by > x) - +(by > y), 1);
      sets.splice(bx - +(bx > x) - +(bx > y), 1);
      sets.splice(sets.length - 1, 0, best[2][best[2].length - 1]!);
      mod.push([cost, [x - +(x > bx) - +(x > by), y - +(y > bx) - +(y > by)], sets]);
    }
    known = mod;
    inputSets = best[2];
    const nt = inputSets.length - 1;
    combIter = Array.from({ length: nt }, (_, i) => [i, nt] as [number, number]);
    path.push(best[1]);
    pathCost += best[0][1];
  }
  return path;
}

// Python "%.3e" (round half to even) for the non-negative integer counts.
const fmtE = (v: number): string => {
  if (!Number.isSafeInteger(v) || v < 10000) {
    const [m, e] = v.toExponential(3).split("e") as [string, string];
    const n = Number(e);
    return `${m}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`;
  }
  const d = String(v);
  let exp = d.length - 1;
  let head = Number(d.slice(0, 4));
  const rest = d.slice(4);
  const tie = rest[0] === "5" && /^0*$/.test(rest.slice(1));
  if (rest[0]! > "5" || (rest[0] === "5" && !tie) || (tie && head % 2 === 1)) head += 1;
  if (head === 10000) {
    head = 1000;
    exp += 1;
  }
  const h = String(head);
  return `${h[0]}.${h.slice(1)}e+${String(exp).padStart(2, "0")}`;
};

interface Plan {
  parsed: Parsed;
  path: number[][];
  steps: [number[], string][];
  report: string;
}

function plan(
  args: readonly unknown[],
  optimize: EinsumOptimize | undefined,
  defaultType: EinsumOptimize,
  forEinsum: boolean,
): Plan {
  let pathType: unknown = optimize === undefined ? defaultType : optimize;
  if (pathType === true) pathType = "greedy";
  if (pathType === null) pathType = false;
  let explicit = false;
  let memLimit: number | null = null;
  if (pathType === false || typeof pathType === "string") {
    // named or none
  } else if (Array.isArray(pathType) && pathType.length && pathType[0] === "einsum_path") {
    explicit = true;
  } else if (
    Array.isArray(pathType) &&
    pathType.length === 2 &&
    typeof pathType[0] === "string" &&
    typeof pathType[1] === "number"
  ) {
    memLimit = Math.trunc(pathType[1]);
    pathType = pathType[0];
  } else {
    throw new ValueError(`Did not understand the path: ${JSON.stringify(pathType)}`);
  }
  const parsed = parseInput(args);
  const inputList = [...parsed.inputs];
  const n = inputList.length;
  let inputSets: LSet[] = inputList.map((x) => new Set(x));
  const outputSet: LSet = new Set(parsed.output);
  const indices: LSet = new Set(inputList.join(""));
  if (forEinsum) {
    // c_einsum reports a mismatched repeated label before label sizes.
    inputList.forEach((term, t) => {
      const sh = parsed.operands[t]!.shape;
      [...term].forEach((c, k) => {
        const first = term.indexOf(c);
        if (first < k && sh[first] !== sh[k]) {
          throw new ValueError(
            `dimensions in operand ${t} for collapsing index '${c}' don't match (${sh[first]} != ${sh[k]})`,
          );
        }
      });
    });
  }
  const dims = new Map<string, number>();
  inputList.forEach((term, t) => {
    const sh = parsed.operands[t]!.shape;
    [...term].forEach((c, k) => {
      const d = sh[k]!;
      const prev = dims.get(c);
      if (prev === undefined || prev === 1) dims.set(c, d);
      else if (d !== 1 && d !== prev) {
        throw new ValueError(`Size of label '${c}' for operand ${t} (${prev}) does not match previous terms (${d}).`);
      }
    });
  });
  const maxSize = Math.max(...[...inputList, parsed.output].map((t) => sizeOf(t, dims)));
  const memArg = memLimit ?? maxSize;
  let path: number[][];
  const sameSet = indices.size === outputSet.size && [...indices].every((c) => outputSet.has(c));
  if (explicit) {
    path = (pathType as unknown[]).slice(1).map((p) => {
      if (!Array.isArray(p) || !p.every((x) => Number.isInteger(x))) {
        throw new ValueError("einsum_path contractions must be arrays of integers");
      }
      return [...(p as number[])];
    });
  } else if (pathType === false || n === 1 || n === 2 || sameSet) {
    path = [Array.from({ length: n }, (_, i) => i)];
  } else if (pathType === "greedy") {
    path = greedyPath(inputSets, outputSet, dims, memArg);
  } else if (pathType === "optimal") {
    path = optimalPath(inputSets, outputSet, dims, memArg);
  } else {
    throw new ValueError(`Path name ${String(pathType)} not found`);
  }
  const costs: number[] = [];
  const scales: number[] = [];
  const sizes: number[] = [];
  const steps: [number[], string][] = [];
  const lines: [string, string[]][] = [];
  path.forEach((ci, cnum) => {
    const inds = [...ci].sort((a, b) => b - a);
    for (const x of inds) {
      if (x < 0 || x >= inputList.length) throw new ValueError("einsum_path contraction index out of range");
    }
    const c = findContraction(inds, inputSets, outputSet);
    inputSets = c.remaining;
    costs.push(flopCount(c.contract, c.removed.size > 0, inds.length, dims));
    scales.push(c.contract.size);
    sizes.push(sizeOf(c.result, dims));
    const tmp = inds.map((x) => inputList.splice(x, 1)[0]!);
    const res =
      cnum === path.length - 1
        ? parsed.output
        : [...c.result]
            .sort((a, b) => dims.get(a)! - dims.get(b)! || (a < b ? -1 : a > b ? 1 : 0))
            .join("");
    inputList.push(res);
    steps.push([inds, res]);
    lines.push([tmp.join(",") + "->" + res, [...inputList]]);
  });
  if (inputList.length !== 1) {
    throw new ValueError(
      `Invalid einsum_path is specified: ${inputList.length - 1} more operands has to be contracted.`,
    );
  }
  const overall = parsed.inputs.join(",") + "->" + parsed.output;
  const inner = parsed.inputs.reduce((s, x) => s + new Set(x).size, 0) - indices.size > 0;
  const naiveCost = flopCount(indices, inner, n, dims);
  const optCost = costs.reduce((a, b) => a + b, 0) + 1;
  let rep = `  Complete contraction:  ${overall}\n`;
  rep += `         Naive scaling:  ${indices.size}\n`;
  rep += `     Optimized scaling:  ${Math.max(...scales)}\n`;
  rep += `      Naive FLOP count:  ${fmtE(naiveCost)}\n`;
  rep += `  Optimized FLOP count:  ${fmtE(optCost)}\n`;
  rep += `   Theoretical speedup:  ${(naiveCost / optCost).toFixed(3)}\n`;
  rep += `  Largest intermediate:  ${fmtE(Math.max(...sizes))} elements\n`;
  rep += "-".repeat(74) + "\n";
  rep += `${"scaling".padStart(6)} ${"current".padStart(24)} ${"remaining".padStart(40)}\n`;
  rep += "-".repeat(74);
  lines.forEach(([s, remaining], i) => {
    rep += `\n${String(scales[i]).padStart(4)}    ${s.padStart(24)} ${(remaining.join(",") + "->" + parsed.output).padStart(40)}`;
  });
  return { parsed, path, steps, report: rep };
}

function splitArgs(args: readonly unknown[]): [unknown[], EinsumOptions] {
  const last = args[args.length - 1];
  if (args.length > 1 && isOptions(last)) return [args.slice(0, -1), last];
  return [[...args], {}];
}

/**
 * Einstein summation. `einsum("ij,jk->ik", a, b, opts?)` or the sublist form
 * `einsum(a, [0, 1], b, [1, 2], [0, 2], opts?)`.
 */
export function einsum(subscripts: string, ...operands: (ArrayLike | EinsumOptions)[]): NDArray;
export function einsum(...args: (ArrayLike | Sublist | EinsumOptions)[]): NDArray;
export function einsum(...all: unknown[]): NDArray {
  const [args, opts] = splitArgs(all);
  const p = plan(args, opts.optimize, false, true);
  return wrapNative(() =>
    NDArray._wrap(
      native().einsum(
        p.parsed.operands.map((o) => o._native),
        p.parsed.inputs,
        p.steps,
      ),
    ),
  );
}

/** Contraction order for an einsum expression: `[["einsum_path", [i, j], ...], report]`. */
export function einsumPath(subscripts: string, ...operands: (ArrayLike | EinsumOptions)[]): [EinsumPath, string];
export function einsumPath(...args: (ArrayLike | Sublist | EinsumOptions)[]): [EinsumPath, string];
export function einsumPath(...all: unknown[]): [EinsumPath, string] {
  const [args, opts] = splitArgs(all);
  const p = plan(args, opts.optimize, "greedy", false);
  return [["einsum_path", ...p.path], p.report];
}
