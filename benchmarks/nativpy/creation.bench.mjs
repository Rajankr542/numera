// Creation / conversion micro-benchmarks for nativpy (PLAN §65).
// Run: pnpm build && pnpm bench   (compare with: pnpm bench:numpy)
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import np from "../../packages/nativpy/dist/index.js";

const SIZES = [1_000, 100_000, 1_000_000];
const REPEATS = 7;

function bench(fn) {
  fn(); // warm-up
  const times = [];
  for (let i = 0; i < REPEATS; i++) {
    globalThis.gc?.();
    const t0 = process.hrtime.bigint();
    fn();
    times.push(Number(process.hrtime.bigint() - t0) / 1e6);
  }
  times.sort((a, b) => a - b);
  return { median_ms: times[Math.floor(times.length / 2)], min_ms: times[0] };
}

const results = [];
for (const n of SIZES) {
  const list = Array.from({ length: n }, (_, i) => i * 0.5);
  const typed = new Float64Array(list);
  const arr = np.array(list);
  const cases = {
    "zeros(float64)": () => np.zeros([n]),
    "array(list float64)": () => np.array(list),
    "fromTypedArray(float64)": () => np.fromTypedArray(typed),
    "astype(float32)": () => arr.astype("float32"),
    "copy": () => arr.copy(),
    "toTypedArray": () => arr.toTypedArray(),
  };
  for (const [name, fn] of Object.entries(cases)) {
    const r = bench(fn);
    results.push({ name, n, ...r });
    console.log(`${name.padEnd(26)} n=${String(n).padStart(8)}  median ${r.median_ms.toFixed(3)} ms`);
  }
}

const out = join(dirname(fileURLToPath(import.meta.url)), "..", "reports", "nativpy-creation.json");
mkdirSync(dirname(out), { recursive: true });
writeFileSync(
  out,
  JSON.stringify({ node: process.version, platform: `${process.platform}-${process.arch}`, results }, null, 1),
);
console.log(`wrote ${out}`);
