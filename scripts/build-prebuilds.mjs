// Build stable-ABI prebuilt addons into packages/nativpy/prebuilds/ (D-026).
// Usage: node scripts/build-prebuilds.mjs [--targets darwin-arm64,darwin-x64,linux-x64,linux-arm64]
// macOS targets build locally (cross-arch via cmake-js --arch). Linux targets
// build in Docker (node:22-bookworm) with a statically linked C++ runtime.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outRoot = join(root, "packages/nativpy/prebuilds");
const ALL = ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64"];
const DOCKER_IMAGE = "node:22-bookworm";
const STABLE = ["--CDNATIVPY_BUILD_TESTS=OFF", "--CDNATIVPY_NAPI_EXPERIMENTAL=OFF"];

function run(cmd, args, opts = {}) {
  console.log(`$ ${cmd} ${args.join(" ")}`);
  execFileSync(cmd, args, { stdio: "inherit", cwd: root, ...opts });
}

function has(cmd, args) {
  try {
    execFileSync(cmd, args, { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function place(target, builtFile) {
  const dir = join(outRoot, target);
  mkdirSync(dir, { recursive: true });
  copyFileSync(builtFile, join(dir, "nativpy.node"));
  console.log(`✔ prebuilds/${target}/nativpy.node`);
}

function buildDarwin(arch) {
  const out = `build-prebuild-darwin-${arch}`;
  rmSync(join(root, out), { recursive: true, force: true });
  run("npx", ["cmake-js", "compile", "--arch", arch, "--out", out, ...STABLE]);
  place(`darwin-${arch}`, join(root, out, "Release/nativpy.node"));
}

function buildLinux(arch) {
  const platform = arch === "x64" ? "linux/amd64" : "linux/arm64";
  const out = `build-prebuild-linux-${arch}`;
  // Copy the sources (not node_modules/build dirs) into the container, build, and
  // run the unit tests + smoke test against the fresh stable-ABI addon.
  const script = [
    "set -e",
    "apt-get update -qq && apt-get install -y -qq cmake ninja-build >/dev/null",
    "mkdir /w && cd /src && tar --exclude=./node_modules --exclude='./build*' --exclude=./.venv --exclude=./.git -cf - . | tar -xf - -C /w",
    "cd /w && corepack enable && pnpm install --frozen-lockfile --silent",
    `npx cmake-js compile --out ${out} ${STABLE.join(" ")} --CDNATIVPY_STATIC_RUNTIME=ON`,
    "pnpm build:ts >/dev/null",
    `export NATIVPY_ADDON_PATH=/w/${out}/Release/nativpy.node`,
    "npx vitest run",
    "node scripts/smoke-test.mjs ./packages/nativpy/dist/index.js",
    `mkdir -p /src/${out}/Release && cp /w/${out}/Release/nativpy.node /src/${out}/Release/`,
  ].join(" && ");
  run("docker", ["run", "--rm", "--platform", platform, "-v", `${root}:/src`, DOCKER_IMAGE, "bash", "-c", script]);
  place(`linux-${arch}`, join(root, out, "Release/nativpy.node"));
}

const flag = process.argv.indexOf("--targets");
const targets = flag >= 0 ? process.argv[flag + 1].split(",") : ALL;
for (const t of targets) {
  if (!ALL.includes(t)) throw new Error(`unknown target ${t}; expected one of ${ALL.join(", ")}`);
}
if (targets.some((t) => t.startsWith("linux-")) && !has("docker", ["info"])) {
  throw new Error("Linux prebuilds need Docker running (docker info failed). Start Docker Desktop or pass --targets.");
}
if (targets.some((t) => t.startsWith("darwin-")) && process.platform !== "darwin") {
  throw new Error("macOS prebuilds must be built on macOS.");
}

rmSync(outRoot, { recursive: true, force: true });
for (const t of targets) {
  const [os, arch] = t.split("-");
  if (os === "darwin") buildDarwin(arch);
  else buildLinux(arch);
}
for (const t of targets) {
  if (!existsSync(join(outRoot, t, "nativpy.node"))) throw new Error(`missing prebuild for ${t}`);
}
console.log(`Built prebuilds: ${targets.join(", ")}`);
