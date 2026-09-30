// Pack packages/nativpy and smoke-test the tarball in a clean project (D-027).
// Used by .github/workflows/release.yml.
// Usage: node scripts/ci-pack.mjs <out-dir>
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages/nativpy");
const { name } = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8"));
const outArg = process.argv[2];
if (!outArg) throw new Error("usage: node scripts/ci-pack.mjs <out-dir>");
const outDir = resolve(outArg);

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { stdio: "inherit", ...opts });
}

rmSync(outDir, { recursive: true, force: true });
mkdirSync(outDir, { recursive: true });
const packed = JSON.parse(
  execFileSync("npm", ["pack", "--json", "--pack-destination", outDir], { cwd: pkgDir, encoding: "utf8" }),
)[0];
console.log(`Tarball: ${packed.filename} (${(packed.size / 1024).toFixed(0)} kB, ${packed.files.length} files)`);
for (const t of ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64"]) {
  if (!packed.files.some((f) => f.path === `prebuilds/${t}/nativpy.node`)) {
    throw new Error(`tarball is missing prebuilds/${t}/nativpy.node`);
  }
}

// Install the tarball in a clean project (no repo build visible) and import it by name.
const smoke = join(outDir, "..", "smoke");
rmSync(smoke, { recursive: true, force: true });
mkdirSync(smoke, { recursive: true });
writeFileSync(join(smoke, "package.json"), '{"name":"smoke","private":true,"type":"module"}\n');
run("npm", ["install", "--no-audit", "--no-fund", join(outDir, packed.filename)], { cwd: smoke });
copyFileSync(join(root, "scripts/smoke-test.mjs"), join(smoke, "smoke-test.mjs"));
const env = { ...process.env };
delete env.NATIVPY_ADDON_PATH;
run("node", ["smoke-test.mjs", name], { cwd: smoke, env });
rmSync(smoke, { recursive: true, force: true });
