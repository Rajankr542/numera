// NumPy API coverage + benchmark coverage core (PLAN §37, D-032).
// CLI: scripts/api-coverage.mjs. Kept separate so tests can import it.
import { readFileSync } from "node:fs";
import { join } from "node:path";

// Match ignoring case and underscores: floor_divide ~ floorDivide, moveaxis ~ moveAxis.
export const norm = (s) => s.replace(/_/g, "").toLowerCase();

export function ownNames(obj) {
  const names = new Set();
  for (let o = obj; o && o !== Object.prototype && o !== Function.prototype; o = Object.getPrototypeOf(o)) {
    for (const k of Object.getOwnPropertyNames(o)) if (k !== "constructor") names.add(k);
  }
  return names;
}

const classOf = (...c) => c.find((x) => typeof x === "function");

/** numera object backing each NumPy surface; undefined => nothing implemented. */
export function targetsFor(np) {
  return {
    np, linalg: np.linalg, fft: np.fft, random: np.random,
    Generator: classOf(np.random?.Generator, np.Generator)?.prototype,
    RandomState: classOf(np.random?.RandomState, np.RandomState)?.prototype,
    ndarray: np.NDArray?.prototype,
    ma: np.ma, polynomial: np.polynomial, strings: np.strings, char: np.char,
    rec: np.rec, emath: np.emath, testing: np.testing,
  };
}

export const benchPrefixes = {
  np: ["np."], linalg: ["np.linalg."], fft: ["np.fft."], random: ["np.random."],
  Generator: ["rng."], RandomState: ["rs."],
  // ndarray methods: calls on the suites' array variables (a, b, v, w, A, B, M, S, ...).
  ndarray: ["a.", "b.", "v.", "w.", "A.", "B.", "M.", "S.", "B1.", "A.T.", "a32.", "ai."],
  ma: ["np.ma."], polynomial: ["np.polynomial."], strings: ["np.strings."],
  char: ["np.char."], rec: ["np.rec."], emath: ["np.emath."], testing: ["np.testing."],
};

/** Normalized names referenced as `<prefix><name>` in a benchmark source. */
export function benchCalls(src, prefixes) {
  const found = new Set();
  for (const p of prefixes) {
    const re = new RegExp(`(?<![A-Za-z0-9_.])${p.replace(/\./g, "\\.")}([A-Za-z_][A-Za-z0-9_]*)`, "g");
    for (const m of src.matchAll(re)) found.add(norm(m[1]));
  }
  return found;
}

/** Computes API and benchmark coverage. Pure given its inputs. */
export function computeCoverage({ np, inventory, exclusions, aliases, natSrc, npySrc }) {
  const targets = targetsFor(np);
  const surfaces = {};
  const aliasErrors = [];
  const unbenchmarked = [];
  let impl = 0, tot = 0;
  for (const [surface, names] of Object.entries(inventory.surfaces)) {
    const have = targets[surface] ? ownNames(targets[surface]) : new Set();
    const haveNorm = new Map([...have].map((k) => [norm(k), k]));
    const al = aliases[surface] ?? {};
    for (const [n, t] of Object.entries(al)) if (!have.has(t)) aliasErrors.push(`${surface}.${n} -> ${t}`);
    const ex = exclusions[surface] ?? {};
    const nat = benchCalls(natSrc, benchPrefixes[surface]);
    const npy = benchCalls(npySrc, benchPrefixes[surface]);
    const implemented = [], missing = [], excluded = [];
    for (const [name, kind] of Object.entries(names)) {
      if (name in ex || `*${kind}` in ex) { excluded.push(name); continue; }
      const js = al[name] ?? haveNorm.get(norm(name));
      if (!js || !have.has(js)) { missing.push(name); continue; }
      implemented.push(name);
      if (kind === "constant" || kind === "class" || kind === "attribute") continue; // not timed
      const inNat = nat.has(norm(js)) || nat.has(norm(name));
      const inNpy = npy.has(norm(name));
      if (!inNat || !inNpy) unbenchmarked.push(`${surface}.${name}${inNat ? "" : " [nativpy]"}${inNpy ? "" : " [numpy]"}`);
    }
    const total = implemented.length + missing.length;
    impl += implemented.length; tot += total;
    surfaces[surface] = {
      total, implemented: implemented.length,
      coverage: total ? +(implemented.length / total * 100).toFixed(1) : 100,
      excluded: excluded.length, missing,
    };
  }
  return {
    numpy_version: inventory.numpy_version,
    totals: { implemented: impl, total: tot, coverage: +(impl / tot * 100).toFixed(1) },
    surfaces, aliasErrors, unbenchmarked: unbenchmarked.sort(),
  };
}

export function loadInputs(root) {
  const readJson = (p) => JSON.parse(readFileSync(join(root, p), "utf8"));
  return {
    inventory: readJson("api/numpy-api.json"),
    exclusions: readJson("api/exclusions.json"),
    aliases: readJson("api/aliases.json"),
    natSrc: readFileSync(join(root, "benchmarks/nativpy/suite.bench.mjs"), "utf8"),
    npySrc: readFileSync(join(root, "benchmarks/numpy/suite_bench.py"), "utf8"),
  };
}

/** Regression check against the baseline. Returns a list of error strings. */
export function checkAgainstBaseline(report, base) {
  const errors = report.aliasErrors.map((e) => `alias target not exported: ${e}`);
  if (report.totals.implemented < base.implemented) {
    errors.push(`implemented names dropped: ${report.totals.implemented} < baseline ${base.implemented}`);
  }
  const ignore = new Set(base.benchIgnore);
  for (const u of report.unbenchmarked) if (!ignore.has(u)) errors.push(`no benchmark in both suites: ${u}`);
  return errors;
}
