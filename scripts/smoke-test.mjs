// Minimal end-to-end check of an installed/packed nativpy (D-026).
// Usage: node scripts/smoke-test.mjs [module-specifier-or-path]
// Default specifier is "numera" (resolved from the current directory).
import { pathToFileURL } from "node:url";
import { isAbsolute, resolve } from "node:path";

const arg = process.argv[2] ?? "numera";
const spec = arg.startsWith(".") || isAbsolute(arg) ? pathToFileURL(resolve(arg)).href : arg;
const { default: np } = await import(spec);

function check(label, got, want) {
  const g = JSON.stringify(got);
  const w = JSON.stringify(want);
  if (g !== w) throw new Error(`smoke-test ${label}: got ${g}, want ${w}`);
}

const a = np.array([[1, 2, 3], [4, 5, 6]]);
check("shape", a.shape, [2, 3]);
check("add", np.add(a, 10).toArray(), [[11, 12, 13], [14, 15, 16]]);
check("sum", a.sum({ axis: 0 }).toArray(), [5, 7, 9]);
check("matmul", np.matmul(a, a.T).toArray(), [[14, 32], [32, 77]]);
check("det", Math.round(np.linalg.det(np.array([[1, 2], [3, 4]])).item()), -2);
const rng = np.random.defaultRng(42);
check("random", rng.random([2]).shape, [2]);
console.log(`nativpy smoke test OK (${process.platform}-${process.arch}, node ${process.version})`);
