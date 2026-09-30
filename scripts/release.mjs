// Local release: `pnpm release [patch|minor|major] [--dry-run] [--targets a,b] [--allow-dirty] [--push]`
// (DECISIONS D-026). Steps: npm auth (browser login) -> pick version -> test ->
// prebuilds -> pack + smoke test the tarball -> npm publish -> commit + tag.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages/nativpy");
const pkgJsonPath = join(pkgDir, "package.json");
// Dev tools (cmake, ninja) come from the repo venv, if present (AGENTS.md setup).
const venvBin = join(root, ".venv/bin");
if (existsSync(venvBin)) process.env.PATH = `${venvBin}${delimiter}${process.env.PATH}`;

const args = process.argv.slice(2);
const opt = (name) => args.includes(name);
const bumpType = args.find((a) => ["patch", "minor", "major"].includes(a)) ?? "patch";
const dryRun = opt("--dry-run");
const targetsIdx = args.indexOf("--targets");
const targets = targetsIdx >= 0 ? args[targetsIdx + 1] : null;

function run(cmd, cmdArgs, opts = {}) {
  console.log(`\n$ ${cmd} ${cmdArgs.join(" ")}`);
  execFileSync(cmd, cmdArgs, { stdio: "inherit", cwd: root, ...opts });
}
function capture(cmd, cmdArgs, opts = {}) {
  return execFileSync(cmd, cmdArgs, { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], ...opts }).trim();
}
function tryCapture(cmd, cmdArgs) {
  try {
    return capture(cmd, cmdArgs);
  } catch {
    return null;
  }
}
function bump(version, type) {
  const [maj, min, pat] = version.split(".").map(Number);
  if (type === "major") return `${maj + 1}.0.0`;
  if (type === "minor") return `${maj}.${min + 1}.0`;
  return `${maj}.${min}.${pat + 1}`;
}
function step(title) {
  console.log(`\n━━ ${title} ━━`);
}
function cleanArtifacts() {
  for (const p of ["prebuilds", "README.md", "LICENSE"]) {
    rmSync(join(pkgDir, p), { recursive: true, force: true });
  }
}

// 1. Clean git tree, so the release commit contains only the version bump.
step("Checking git working tree");
const dirty = capture("git", ["status", "--porcelain"]);
if (dirty && !opt("--allow-dirty")) {
  throw new Error(`Working tree is not clean. Commit or stash first (or pass --allow-dirty):\n${dirty}`);
}

// 2. npm authentication. `npm login --auth-type=web` opens the browser.
step("Checking npm authentication");
let user = tryCapture("npm", ["whoami"]);
if (!user) {
  console.log("Not logged in to npm. Opening the browser for npm login...");
  run("npm", ["login", "--auth-type=web"]);
  user = tryCapture("npm", ["whoami"]);
  if (!user) throw new Error("npm login did not succeed.");
}
console.log(`Logged in to npm as: ${user}`);

// 3. Version: publish the current version if it is not on npm yet, otherwise bump.
step("Choosing version");
const pkg = JSON.parse(readFileSync(pkgJsonPath, "utf8"));
const published = JSON.parse(tryCapture("npm", ["view", pkg.name, "versions", "--json"]) ?? "[]");
const publishedList = Array.isArray(published) ? published : [published];
let version = pkg.version;
while (publishedList.includes(version)) version = bump(version, bumpType);
const originalJson = readFileSync(pkgJsonPath, "utf8");
if (version !== pkg.version) {
  pkg.version = version;
  writeFileSync(pkgJsonPath, JSON.stringify(pkg, null, 2) + "\n");
}
console.log(`${pkg.name}@${version} (npm has: ${publishedList.join(", ") || "none"})`);

try {
  // 4. Verify, then build the stable-ABI prebuilds. Stale prebuilds are removed
  // first so the tests exercise the fresh source build (loader order, D-026).
  cleanArtifacts();
  step("Building and testing");
  run("pnpm", ["build"]);
  run("pnpm", ["typecheck"]);
  run("pnpm", ["test"]);
  step("Building prebuilds");
  run("node", ["scripts/build-prebuilds.mjs", ...(targets ? ["--targets", targets] : [])]);
  copyFileSync(join(root, "README.md"), join(pkgDir, "README.md"));
  copyFileSync(join(root, "LICENSE"), join(pkgDir, "LICENSE"));

  // 5. Pack and smoke-test the real tarball in a clean project (no repo build visible).
  step("Packing and smoke-testing the tarball");
  const tmp = mkdtempSync(join(tmpdir(), "nativpy-release-"));
  const packed = JSON.parse(capture("npm", ["pack", "--json", "--pack-destination", tmp], { cwd: pkgDir }));
  console.log(`Tarball: ${packed[0].filename} (${(packed[0].size / 1024).toFixed(0)} kB, ${packed[0].files.length} files)`);
  writeFileSync(join(tmp, "package.json"), '{"name":"smoke","private":true,"type":"module"}\n');
  run("npm", ["install", "--no-audit", "--no-fund", join(tmp, packed[0].filename)], { cwd: tmp });
  const env = { ...process.env };
  delete env.NATIVPY_ADDON_PATH;
  // Run from inside the temp project so the bare "nativpy" import resolves there.
  copyFileSync(join(root, "scripts/smoke-test.mjs"), join(tmp, "smoke-test.mjs"));
  run("node", ["smoke-test.mjs"], { cwd: tmp, env });
  rmSync(tmp, { recursive: true, force: true });

  // 6. Publish. npm may open the browser again for 2FA confirmation.
  step(dryRun ? "Publishing (dry run)" : "Publishing to npm");
  run("npm", ["publish", ...(dryRun ? ["--dry-run"] : [])], { cwd: pkgDir });
} catch (err) {
  if (version !== JSON.parse(originalJson).version) writeFileSync(pkgJsonPath, originalJson);
  console.error(`\nRelease aborted; ${pkgJsonPath} restored.`);
  throw err;
} finally {
  // Generated, gitignored files; removing them keeps dev runs on build/Release.
  cleanArtifacts();
}

// 7. Record the release in git.
if (dryRun) {
  if (version !== JSON.parse(originalJson).version) writeFileSync(pkgJsonPath, originalJson);
  console.log(`\nDry run complete: ${pkg.name}@${version} was NOT published.`);
} else {
  step("Committing and tagging");
  if (capture("git", ["status", "--porcelain", pkgJsonPath])) {
    run("git", ["add", pkgJsonPath]);
    run("git", ["commit", "-m", `chore(release): ${pkg.name}@${version}`]);
  }
  run("git", ["tag", "-a", `v${version}`, "-m", `${pkg.name}@${version}`]);
  if (opt("--push")) run("git", ["push", "--follow-tags"]);
  console.log(`\n✔ Published ${pkg.name}@${version}: https://www.npmjs.com/package/${pkg.name}`);
  if (!opt("--push")) console.log("Push the release commit and tag with: git push --follow-tags");
}
