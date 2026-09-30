// Pack packages/numera and smoke-test the tarball in a clean project (D-027).
// Used by .github/workflows/release.yml.
// Usage: node scripts/ci-pack.mjs <out-dir> [--name <package-name>]
// --name packs under another name, e.g. "@rajankr542/numera" for GitHub
// Packages (owner scope required). package.json is restored afterwards.
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages/numera");
const pkgJsonPath = join(pkgDir, "package.json");
const args = process.argv.slice(2);
const nameIdx = args.indexOf("--name");
const override = nameIdx >= 0 ? args[nameIdx + 1] : null;
if (nameIdx >= 0 && !override) throw new Error("--name needs a value");
const outArg = args.find((a, i) => !a.startsWith("--") && (nameIdx < 0 || i !== nameIdx + 1));
if (!outArg) throw new Error("usage: node scripts/ci-pack.mjs <out-dir> [--name <package-name>]");
const outDir = resolve(outArg);

function run(cmd, cmdArgs, opts = {}) {
  console.log(`$ ${cmd} ${cmdArgs.join(" ")}`);
  execFileSync(cmd, cmdArgs, { stdio: "inherit", ...opts });
}

const original = readFileSync(pkgJsonPath, "utf8");
const pkg = JSON.parse(original);
const name = override ?? pkg.name;
rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });
let packed;
try {
  if (override) writeFileSync(pkgJsonPath, `${JSON.stringify({ ...pkg, name }, null, 2)}\n`);
  packed = JSON.parse(
    execFileSync("npm", ["pack", "--json", "--pack-destination", outDir], { cwd: pkgDir, encoding: "utf8" }),
  )[0];
} finally {
  writeFileSync(pkgJsonPath, original);
}
if (packed.name !== name) throw new Error(`packed name ${packed.name}, expected ${name}`);
console.log(`Tarball: ${packed.filename} (${(packed.size / 1024).toFixed(0)} kB, ${packed.files.length} files)`);
for (const t of ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64"]) {
  if (!packed.files.some((f) => f.path === `prebuilds/${t}/nativpy.node`)) {
    throw new Error(`tarball is missing prebuilds/${t}/nativpy.node`);
  }
}

// Install the tarball in a clean project (no repo build visible) and import it by name.
const smoke = `${outDir}-smoke`;
rmSync(smoke, { recursive: true, force: true });
mkdirSync(smoke, { recursive: true });
writeFileSync(join(smoke, "package.json"), '{"name":"smoke","private":true,"type":"module"}\n');
run("npm", ["install", "--no-audit", "--no-fund", join(outDir, packed.filename)], { cwd: smoke });
copyFileSync(join(root, "scripts/smoke-test.mjs"), join(smoke, "smoke-test.mjs"));
const env = { ...process.env };
delete env.NATIVPY_ADDON_PATH;
run("node", ["smoke-test.mjs", name], { cwd: smoke, env });
rmSync(smoke, { recursive: true, force: true });
