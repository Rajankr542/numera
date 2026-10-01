// NumPy API coverage + benchmark coverage (PLAN §37, D-032).
//
//   node scripts/api-coverage.mjs                    report + write api/coverage.json
//   node scripts/api-coverage.mjs --check            fail on regressions (CI)
//   node scripts/api-coverage.mjs --update-baseline  accept the current state
//
// Needs `pnpm build`: the implemented surface is read from the real package.
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { checkAgainstBaseline, computeCoverage, loadInputs } from "./api-coverage-lib.mjs";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const args = new Set(process.argv.slice(2));
const { default: np } = await import(join(root, "packages/numera/dist/index.js"));
const report = computeCoverage({ np, ...loadInputs(root) });
writeFileSync(join(root, "api/coverage.json"), JSON.stringify(report, null, 1) + "\n");

console.log(`NumPy ${report.numpy_version} API coverage (D-032)\n`);
console.log("surface".padEnd(14) + "impl/total".padStart(12) + "coverage".padStart(10) + "excluded".padStart(10));
for (const [k, s] of Object.entries(report.surfaces)) {
  console.log(k.padEnd(14) + `${s.implemented}/${s.total}`.padStart(12) + `${s.coverage}%`.padStart(10) + String(s.excluded).padStart(10));
}
const t = report.totals;
console.log(`\nAPI coverage: ${t.coverage}% (${t.implemented}/${t.total})`);
console.log(`Implemented callables without a benchmark in both suites: ${report.unbenchmarked.length}`);

const basePath = join(root, "api/coverage-baseline.json");
if (args.has("--update-baseline")) {
  const base = { coverage: t.coverage, implemented: t.implemented, benchIgnore: report.unbenchmarked };
  writeFileSync(basePath, JSON.stringify(base, null, 1) + "\n");
  console.log("baseline updated");
}
if (args.has("--check")) {
  const base = JSON.parse(readFileSync(basePath, "utf8"));
  // D-056: build-first milestones list not-yet-benchmarked callables in
  // api/bench-exempt/pNN.json until their V slice adds the bench cases.
  const exemptDir = join(root, "api/bench-exempt");
  for (const f of readdirSync(exemptDir).filter((x) => x.endsWith(".json"))) {
    base.benchIgnore.push(...JSON.parse(readFileSync(join(exemptDir, f), "utf8")));
  }
  const errors = checkAgainstBaseline(report, base);
  if (errors.length) {
    console.error(`\napi-coverage --check FAILED:\n  ${errors.join("\n  ")}`);
    process.exit(1);
  }
  console.log("api-coverage --check passed");
}
