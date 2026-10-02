// Captures the documentation data of one numera release into versions/<version>/.
//
//   node scripts/snapshot.mjs --version 1.0.2 --repo <numera checkout> \
//        --package <dir containing dist/index.js> [--ref <git ref>] [--date YYYY-MM-DD]
//
// --repo     a checkout of the numera main repository (provides the API inventory,
//            the coverage helpers and the COMPATIBILITY.md packaging rules).
// --ref      git ref whose docs/site/*.mjs and COMPATIBILITY.md are captured
//            (default HEAD; e.g. v1.0.1 for an older release).
// --package  the built package of that release: packages/numera after `pnpm build`,
//            or node_modules/@cyfora/numera from `npm i @cyfora/numera@<version>`.
//
// Writes: api.mjs (+ parts/), COMPATIBILITY.md, names.json, meta.json.
// Hand-written guides in versions/<version>/guides/ are left untouched.
import { execFileSync } from "node:child_process";
import { createRequire } from "node:module";
import { mkdirSync, rmSync, writeFileSync, existsSync, readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const here = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const args = Object.fromEntries(
  process.argv.slice(2).reduce((acc, a, i, all) => (a.startsWith("--") ? [...acc, [a.slice(2), all[i + 1]]] : acc), []),
);
for (const k of ["version", "repo", "package"]) if (!args[k]) throw new Error(`missing --${k}`);
const repo = resolve(args.repo);
const pkg = resolve(args.package);
const ref = args.ref ?? "HEAD";
const out = join(here, "versions", args.version);
const git = (...a) => execFileSync("git", ["-C", repo, ...a], { encoding: "utf8", maxBuffer: 64 << 20 });

// 1. API reference content at <ref>.
rmSync(join(out, "parts"), { recursive: true, force: true });
mkdirSync(out, { recursive: true });
writeFileSync(join(out, "api.mjs"), git("show", `${ref}:docs/site/api.mjs`));
const parts = git("ls-tree", "--name-only", `${ref}`, "docs/site/parts/").split("\n").filter((f) => f.endsWith(".mjs"));
if (parts.length) mkdirSync(join(out, "parts"), { recursive: true });
for (const f of parts) writeFileSync(join(out, "parts", f.split("/").pop()), git("show", `${ref}:${f}`));
const { categories } = await import(pathToFileURL(join(out, "api.mjs")).href + `?t=${Date.now()}`);

// 2. COMPATIBILITY.md, cleaned the same way as the npm package copy.
const stage = await import(pathToFileURL(join(repo, "scripts/stage-package.mjs")).href);
const compat = stage.packageCompatibility(git("show", `${ref}:COMPATIBILITY.md`));
stage.assertSelfContained([["COMPATIBILITY.md", compat]]);
writeFileSync(join(out, "COMPATIBILITY.md"), compat);

// 3. Public names of the built package, mapped to the NumPy inventory.
const pkgJson = JSON.parse(readFileSync(join(pkg, "package.json"), "utf8"));
const { default: np } = await import(pathToFileURL(join(pkg, "dist/index.js")).href);
const lib = await import(pathToFileURL(join(repo, "scripts/api-coverage-lib.mjs")).href);
const inputs = lib.loadInputs(repo);
const report = lib.computeCoverage({ np, ...inputs });
const targets = lib.targetsFor(np);
const internal = new Set(["translateNativeError", "wrapNative", "_setBackend"]);
const surfaces = {};
for (const [surface, inv] of Object.entries(inputs.inventory.surfaces)) {
  const have = targets[surface] ? lib.ownNames(targets[surface]) : new Set();
  const byNorm = new Map([...have].map((k) => [lib.norm(k), k]));
  const al = inputs.aliases[surface] ?? {};
  const ex = inputs.exclusions[surface] ?? {};
  const implemented = [];
  const missing = [];
  const used = new Set();
  for (const [name, kind] of Object.entries(inv)) {
    if (name in ex || `*${kind}` in ex) continue;
    const js = al[name] ?? byNorm.get(lib.norm(name));
    if (!js || !have.has(js)) { missing.push({ numpy: name, kind }); continue; }
    used.add(js);
    let v;
    for (let o = targets[surface]; o && v === undefined; o = Object.getPrototypeOf(o)) {
      const d = Object.getOwnPropertyDescriptor(o, js);
      if (d) v = "value" in d ? d.value : undefined;
      if (d) break;
    }
    const ufunc = typeof v === "function" && typeof v.reduce === "function" && typeof v.at === "function"
      ? "binary" : typeof v === "function" && typeof v.at === "function" ? "unary" : undefined;
    implemented.push({ numpy: name, js, kind, ...(ufunc ? { ufunc } : {}) });
  }
  // Extra JS exports with no NumPy counterpart in the inventory (np surface only).
  const extras = surface === "np"
    ? [...have].filter((k) => !used.has(k) && !internal.has(k) && !/^_/.test(k) && !(k in targets)).sort()
    : [];
  surfaces[surface] = { implemented, missing, ...(extras.length ? { extras } : {}) };
}
// 3b. TypeScript call signatures from the package's declaration files.
const sigs = typeSignatures(join(pkg, "dist/index.d.ts"));
for (const [surface, s] of Object.entries(surfaces)) {
  for (const item of s.implemented) {
    const sig = sigs[surface]?.[item.js];
    if (sig) item.sig = sig;
  }
}
writeFileSync(join(out, "names.json"), JSON.stringify({ numpy_version: inputs.inventory.numpy_version, surfaces }, null, 1) + "\n");

function typeSignatures(dts) {
  const ts = createRequire(join(repo, "package.json"))("typescript");
  const prog = ts.createProgram([dts], {
    target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.NodeNext, moduleResolution: ts.ModuleResolutionKind.NodeNext,
    strict: true, skipLibCheck: true, noEmit: true, types: [],
  });
  const c = prog.getTypeChecker();
  const sf = prog.getSourceFile(dts);
  const def = c.getExportsOfModule(c.getSymbolAtLocation(sf)).find((s) => s.name === "default");
  if (!def) return {};
  const npType = c.getTypeOfSymbolAtLocation(def, sf);
  const fmt = ts.TypeFormatFlags.NoTruncation | ts.TypeFormatFlags.UseAliasDefinedOutsideCurrentScope;
  const members = (type, prefix) => {
    const res = {};
    if (!type) return res;
    for (const p of type.getProperties()) {
      const t = c.getTypeOfSymbolAtLocation(p, sf);
      const call = t.getCallSignatures();
      if (call.length) {
        res[p.name] = call.slice(0, 3).map((s) => `${prefix}${p.name}${c.signatureToString(s, undefined, fmt)}`.replace(/\s+/g, " "));
      } else if (t.getConstructSignatures().length) {
        res[p.name] = t.getConstructSignatures().slice(0, 2)
          .map((s) => `new ${prefix}${p.name}${c.signatureToString(s, undefined, fmt).replace(/:\s*[^:]+$/, "")}`.replace(/\s+/g, " "));
      } else {
        res[p.name] = [`${prefix}${p.name}: ${c.typeToString(t, undefined, fmt)}`.replace(/\s+/g, " ")];
      }
      res[p.name] = res[p.name].map((s) => (s.length > 600 ? s.slice(0, 597) + "..." : s));
    }
    return res;
  };
  const prop = (type, name) => {
    const s = type.getProperty(name);
    return s ? c.getTypeOfSymbolAtLocation(s, sf) : undefined;
  };
  const instance = (ctorType) => ctorType?.getConstructSignatures()[0]?.getReturnType();
  const random = prop(npType, "random");
  const out = {
    np: members(npType, "np."),
    ndarray: members(instance(prop(npType, "NDArray")), "a."),
    random: members(random, "np.random."),
    Generator: members(instance(random && prop(random, "Generator")), "rng."),
    RandomState: members(instance(random && prop(random, "RandomState")), "rs."),
  };
  for (const ns of ["linalg", "fft", "ma", "polynomial", "strings", "char", "rec", "emath", "testing"]) {
    out[ns] = members(prop(npType, ns), `np.${ns}.`);
  }
  return out;
}

// 4. Metadata.
const date = args.date ?? git("log", "-1", "--format=%cs", ref).trim();
const meta = {
  version: args.version,
  package: pkgJson.name,
  date,
  numpy: inputs.inventory.numpy_version,
  node: pkgJson.engines?.node ?? ">=18",
  documented: categories.reduce((n, c) => n + c.entries.length, 0),
  coverage: report.totals,
  excluded: Object.values(report.surfaces).reduce((n, s) => n + s.excluded, 0),
};
writeFileSync(join(out, "meta.json"), JSON.stringify(meta, null, 1) + "\n");
if (!existsSync(join(out, "guides"))) mkdirSync(join(out, "guides"));
console.log(`versions/${args.version}: ${meta.documented} documented entries, ${parts.length} parts, ` +
  `NumPy ${meta.numpy} coverage ${meta.coverage.coverage}% (${meta.coverage.implemented}/${meta.coverage.total})`);
