// Local release: `pnpm release [patch|minor|major] [--dry-run] [--targets a,b] [--otp code] [--allow-dirty] [--push]`
// (DECISIONS D-026). Steps: npm auth (browser login) -> pick version -> test ->
// prebuilds -> pack + smoke test the tarball -> npm publish -> commit + tag.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline/promises";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages/numera");
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
const otpIdx = args.indexOf("--otp");
const otpArg = otpIdx >= 0 ? args[otpIdx + 1] : null;

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

// 2b. Scoped name (D-028): publishing @scope/name needs the scope to be your
// user or an npm org you belong to. Check now, not after the ~10 min build.
const pkgName = JSON.parse(readFileSync(pkgJsonPath, "utf8")).name;
const scope = pkgName.startsWith("@") ? pkgName.slice(1, pkgName.indexOf("/")) : null;
if (scope && scope !== user) {
  let membership = null;
  let reason = "";
  try {
    membership = capture("npm", ["org", "ls", scope, user, "--json"]);
  } catch (err) {
    reason = String(err.stderr ?? err.message).split("\n").find((l) => /E\d{3}|not found|forbidden/i.test(l)) ?? "";
  }
  const role = membership ? JSON.parse(membership)[user] : undefined;
  if (!role) {
    throw new Error(
      `Cannot publish ${pkgName}: npm user "${user}" is not a member of the "@${scope}" npm organization.` +
        (reason ? `\n(${reason.trim()})` : "") +
        `\nCreate the org (free for public packages) at https://www.npmjs.com/org/create with the name "${scope}",` +
        `\nor ask an owner of @${scope} to add you (npm org set ${scope} ${user} developer), then re-run pnpm release.`,
    );
  }
  console.log(`Member of @${scope} (role: ${role})`);
}

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
  // Run from inside the temp project so the bare package-name import resolves there.
  copyFileSync(join(root, "scripts/smoke-test.mjs"), join(tmp, "smoke-test.mjs"));
  run("node", ["smoke-test.mjs", pkg.name], { cwd: tmp, env });
  rmSync(tmp, { recursive: true, force: true });

  // 6. Publish. With 2FA ("auth-and-writes") npm needs a one-time password:
  // pass --otp <code> / NPM_OTP, or type it when prompted (interactive only).
  // An EOTP failure re-prompts instead of discarding the build.
  step(dryRun ? "Publishing (dry run)" : "Publishing to npm");
  let otp = otpArg ?? process.env.NPM_OTP ?? null;
  for (let attempt = 1; ; attempt++) {
    // Scoped packages default to restricted (paid); --access public publishes them
    // free and public (also set in publishConfig; explicit here as npm advises).
    const publishArgs = [
      "publish",
      "--access",
      "public",
      ...(dryRun ? ["--dry-run"] : []),
      ...(otp ? [`--otp=${otp}`] : []),
    ];
    try {
      console.log(`\n$ npm ${publishArgs.map((a) => (a.startsWith("--otp=") ? "--otp=******" : a)).join(" ")}`);
      execFileSync("npm", publishArgs, { stdio: ["inherit", "inherit", "pipe"], cwd: pkgDir, encoding: "utf8" });
      break;
    } catch (err) {
      const stderr = String(err.stderr ?? "");
      process.stderr.write(stderr);
      if (/too similar to existing package/i.test(stderr)) {
        console.error(
          `\nnpm rejected the name "${pkg.name}" as too similar to an existing package.` +
            `\nUse a scoped name instead (npm skips this check for scopes): set "name" in` +
            `\n${pkgJsonPath} to "@${user}/${pkg.name.replace(/^@[^/]+\//, "")}" (or an npm org scope you belong to)` +
            `\nand re-run pnpm release; it publishes with --access public.`,
        );
        throw err;
      }
      const needsOtp = /EOTP|one-time password/i.test(stderr);
      if (!needsOtp || attempt >= 3 || !process.stdin.isTTY) {
        if (needsOtp) console.error("\nnpm needs a 2FA one-time password. Re-run with: pnpm release -- --otp <code>");
        throw err;
      }
      const rl = createInterface({ input: process.stdin, output: process.stdout });
      otp = (await rl.question("npm one-time password (authenticator app): ")).trim();
      rl.close();
    }
  }
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
