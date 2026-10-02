// Executes every ```js / ```ts block in a version's guides against that
// version's built package and checks each `expr; // => json` line.
//
//   node scripts/check-examples.mjs --version 1.0.2 --package <dir containing dist/index.js>
//
// Blocks whose info string contains "norun" are skipped (install commands,
// type-only snippets). `import ... from "@cyfora/numera"` lines are replaced by
// the package under test; TypeScript-only syntax is not supported in checked
// blocks, so keep them plain JavaScript.
import { readdirSync, readFileSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { codeBlocks } from "./lib/markdown.mjs";

const here = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []),
);
for (const k of ["version", "package"]) if (!args[k]) throw new Error(`missing --${k}`);
const mod = await import(pathToFileURL(join(resolve(args.package), "dist/index.js")).href);
const np = mod.default;
process.removeAllListeners("warning");
process.on("warning", () => {});

const close = (a, e, path = "$") => {
  if (typeof e === "number" && typeof a === "number") return Math.abs(a - e) <= 1e-9 * Math.max(1, Math.abs(e)) || (Number.isNaN(a) && Number.isNaN(e)) ? null : `${path}: ${a} != ${e}`;
  if (Array.isArray(e)) {
    if (!Array.isArray(a) || a.length !== e.length) return `${path}: ${JSON.stringify(a)} != ${JSON.stringify(e)}`;
    for (let i = 0; i < e.length; i++) { const r = close(a[i], e[i], `${path}[${i}]`); if (r) return r; }
    return null;
  }
  if (e && typeof e === "object") {
    if (!a || typeof a !== "object") return `${path}: ${JSON.stringify(a)} != ${JSON.stringify(e)}`;
    for (const k of Object.keys(e)) { const r = close(a[k], e[k], `${path}.${k}`); if (r) return r; }
    return null;
  }
  return Object.is(a, e) ? null : `${path}: ${JSON.stringify(a)} != ${JSON.stringify(e)}`;
};

const reserved = new Set(["delete", "in", "default", "new", "var", "void", "with", "class", "function", "import", "export", "enum", "eval", "arguments"]);
const named = Object.keys(mod).filter((k) => /^[A-Za-z_$][\w$]*$/.test(k) && !reserved.has(k));
const dir = join(here, "versions", args.version, "guides");
let checked = 0;
const failures = [];
for (const f of readdirSync(dir).filter((x) => x.endsWith(".md")).sort()) {
  const md = readFileSync(join(dir, f), "utf8");
  for (const b of codeBlocks(md)) {
    if (!["js", "ts"].includes(b.lang) || /norun/.test(md.split("\n")[b.line - 1])) continue;
    const results = [];
    let want = 0;
    const body = b.code.split("\n").map((line, i) => {
      if (/^\s*import\s.+from\s+["']@cyfora\/numera["'];?\s*$/.test(line)) return "";
      const at = line.indexOf("// =>");
      if (at < 0) return line;
      want++;
      const expr = line.slice(0, at).trim().replace(/;$/, "");
      const exp = line.slice(at + 5).trim();
      try { JSON.parse(exp); } catch { failures.push(`${f}:${b.line + i + 1}: result is not JSON: ${exp}`); return line; }
      return `__check(${b.line + i + 1}, () => (${expr}), ${exp});`;
    }).join("\n");
    const check = (line, fn, e) => {
      let a;
      try { a = fn(); } catch (err) { a = `throws ${err.constructor.name}`; }
      if (a instanceof np.NDArray) a = a.toArray();
      if (typeof a === "bigint") a = String(a);
      results.push({ line, a, e });
    };
    try {
      new Function("np", "__check", ...named, body)(np, check, ...named.map((k) => mod[k]));
    } catch (err) {
      failures.push(`${f}:${b.line}: block threw ${err.constructor.name}: ${err.message}`);
      continue;
    }
    if (results.length !== want) failures.push(`${f}:${b.line}: only ${results.length}/${want} checks ran`);
    for (const r of results) {
      checked++;
      const d = close(r.a, r.e);
      if (d) failures.push(`${f}:${r.line}: ${d}`);
    }
  }
}
console.log(`v${args.version}: ${checked} guide results checked, ${failures.length} failures`);
for (const x of failures) console.log("  " + x);
process.exit(failures.length ? 1 : 0);
