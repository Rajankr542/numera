// Compares benchmarks/reports/{nativpy,numpy}-suite.json (PLAN §41, §75).
// Writes reports/comparison.json (PLAN §41 records) and prints a markdown table.
// relative_performance = numpy_ms / nativpy_ms  (> 1: nativpy faster).
import { execSync } from "node:child_process";
import { readFileSync, writeFileSync } from "node:fs";
import { cpus, totalmem } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const dir = join(dirname(fileURLToPath(import.meta.url)), "reports");
const nat = JSON.parse(readFileSync(join(dir, "nativpy-suite.json"), "utf8"));
const npy = JSON.parse(readFileSync(join(dir, "numpy-suite.json"), "utf8"));
const key = (r) => `${r.category}/${r.name}@${r.shape.join("x")}`;
const ref = new Map(npy.results.map((r) => [key(r), r]));

let cpu = cpus()[0]?.model ?? "unknown";
try {
  if (process.platform === "darwin") cpu = execSync("sysctl -n machdep.cpu.brand_string").toString().trim();
} catch { /* keep os.cpus() model */ }
const env = {
  cpu,
  logical_cpus: cpus().length,
  memory_gb: Math.round(totalmem() / 2 ** 30),
  platform: nat.platform,
  node: nat.node,
  python: npy.python,
  numpy: npy.numpy,
  numpy_blas: npy.blas,
  nativpy_linalg_backend: nat.linalg_backend,
  nativpy: JSON.parse(readFileSync(join(dir, "..", "..", "packages", "nativpy", "package.json"), "utf8")).version,
  method: `median of ${nat.repeats} samples, each >= ${nat.min_sample_ms} ms of repeated calls`,
};

const records = [];
for (const r of nat.results) {
  const n = ref.get(key(r));
  const rec = { operation: `${r.category}/${r.name}`, shape: r.shape };
  if (r.not_implemented) rec.nativpy_ms = null;
  else rec.nativpy_ms = +r.median_ms.toPrecision(4);
  rec.numpy_ms = n ? +n.median_ms.toPrecision(4) : null;
  rec.relative_performance =
    rec.nativpy_ms && rec.numpy_ms ? +(rec.numpy_ms / rec.nativpy_ms).toFixed(4) : null;
  if (r.not_implemented) rec.note = "not implemented in nativpy";
  records.push(rec);
}
writeFileSync(join(dir, "comparison.json"), JSON.stringify({ environment: env, results: records }, null, 1));

const fmt = (x) => (x === null ? "—" : x < 0.01 ? x.toExponential(2) : x.toFixed(3));
const verdict = (rp) => (rp === null ? "" : rp >= 1 ? `${rp.toFixed(2)}× faster` : `${(1 / rp).toFixed(2)}× slower`);
console.log("```text");
for (const [k, v] of Object.entries(env)) console.log(`${k}: ${v}`);
console.log("```\n");
console.log("| Operation | Shape | nativpy ms | NumPy ms | nativpy vs NumPy |");
console.log("|---|---|---:|---:|---|");
for (const r of records) {
  const v = r.note ? r.note : verdict(r.relative_performance);
  console.log(`| \`${r.operation}\` | ${r.shape.join("×")} | ${fmt(r.nativpy_ms)} | ${fmt(r.numpy_ms)} | ${v} |`);
}
const rps = records.map((r) => r.relative_performance).filter((x) => x !== null);
const geo = Math.exp(rps.reduce((s, x) => s + Math.log(x), 0) / rps.length);
console.log(`\n${rps.length} comparable cases; geometric mean relative_performance ${geo.toFixed(3)}; ` +
  `${rps.filter((x) => x >= 1).length} faster, ${rps.filter((x) => x < 1).length} slower.`);
console.log(`wrote ${join(dir, "comparison.json")}`);
