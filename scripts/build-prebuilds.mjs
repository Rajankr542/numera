// Build stable-ABI prebuilt addons into packages/nativpy/prebuilds/ (D-026).
// Usage: node scripts/build-prebuilds.mjs [--targets darwin-arm64,darwin-x64,linux-x64,linux-arm64]
// macOS targets build locally (cross-arch via cmake-js --arch). Linux targets
// build in Docker (manylinux_2_28, glibc 2.28) with a statically linked C++ runtime.
import { execFileSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outRoot = join(root, "packages/nativpy/prebuilds");
// Dev tools (cmake, ninja) come from the repo venv, if present (AGENTS.md setup).
const venvBin = join(root, ".venv/bin");
if (existsSync(venvBin)) process.env.PATH = `${venvBin}${delimiter}${process.env.PATH}`;
const ALL = ["darwin-arm64", "darwin-x64", "linux-x64", "linux-arm64"];
const pnpmVersion = JSON.parse(readFileSync(join(root, "package.json"), "utf8")).packageManager.split("@")[1];
// ACCELERATE_NEW_LAPACK (cblas_new.h / lapack.h) is introduced in macOS 13.3.
const MACOS_MIN = "13.3";
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

function sleep(ms) {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

// Start Docker Desktop on macOS if it is installed but not running.
function ensureDocker() {
  if (has("docker", ["info"])) return;
  if (process.platform === "darwin" && existsSync("/Applications/Docker.app")) {
    console.log("Docker is not running; starting Docker Desktop...");
    execFileSync("open", ["-a", "Docker"]);
    for (let i = 0; i < 60; i++) {
      sleep(3000);
      if (has("docker", ["info"])) return;
    }
  }
  throw new Error("Linux prebuilds need Docker running (docker info failed). Start Docker Desktop or pass --targets.");
}

// Some macOS setups have a Command Line Tools SDK the linker cannot use
// (seen: MacOSX27.0.sdk -> "tapi error: unknown architecture"). Probe a tiny
// link; if it fails, fall back to the Xcode SDK.
function ensureWorkingMacSdk() {
  if (process.env.SDKROOT) return;
  const dir = mkdtempSync(join(tmpdir(), "nativpy-sdk-probe-"));
  const src = join(dir, "p.cpp");
  writeFileSync(src, "int main() { return 0; }\n");
  const ok = has("c++", [src, "-o", join(dir, "p")]);
  rmSync(dir, { recursive: true, force: true });
  if (ok) return;
  const xcodeDev = "/Applications/Xcode.app/Contents/Developer";
  if (existsSync(xcodeDev)) {
    const sdk = execFileSync("xcrun", ["--sdk", "macosx", "--show-sdk-path"], {
      encoding: "utf8",
      env: { ...process.env, DEVELOPER_DIR: xcodeDev },
    }).trim();
    console.log(`Default macOS SDK cannot link; using ${sdk}`);
    process.env.SDKROOT = sdk;
    return;
  }
  throw new Error("The C++ toolchain cannot link a test program. Fix Xcode/Command Line Tools or set SDKROOT.");
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
  // Without a deployment target the binary requires the build host's macOS
  // (seen: minos 26.5). Pin the minimum so it loads on older macOS too.
  const env = { ...process.env, MACOSX_DEPLOYMENT_TARGET: process.env.MACOSX_DEPLOYMENT_TARGET ?? MACOS_MIN };
  run("npx", ["cmake-js", "compile", "--arch", arch, "--out", out, ...STABLE], { env });
  place(`darwin-${arch}`, join(root, out, "Release/nativpy.node"));
}

function buildLinux(arch) {
  // manylinux_2_28 (AlmaLinux 8, glibc 2.28, GCC 14): the prebuild then loads
  // on any glibc >= 2.28 distro (Ubuntu 20.04+, Debian 10+, RHEL 8+). A
  // Debian 12 image produced a glibc 2.36 requirement (arc4random, _dl_find_object).
  const platform = arch === "x64" ? "linux/amd64" : "linux/arm64";
  const image = `quay.io/pypa/manylinux_2_28_${arch === "x64" ? "x86_64" : "aarch64"}`;
  const nodeUrl = `https://nodejs.org/dist/${process.version}/node-${process.version}-linux-${arch}.tar.xz`;
  const out = `build-prebuild-linux-${arch}`;
  // vitest's esbuild crashes ("fatal error: fault") under QEMU emulation, e.g.
  // linux/amd64 on Apple Silicon. Run the full suite only when native; the
  // smoke test (plain node, no esbuild) runs everywhere.
  const native = arch === process.arch;
  // Copy the sources (not node_modules/build dirs) into the container, build, and
  // test the fresh stable-ABI addon.
  const script = [
    "set -e",
    `mkdir /opt/node && curl -fsSL ${nodeUrl} | tar -xJ -C /opt/node --strip-components=1`,
    "export PATH=/opt/node/bin:$PATH",
    "/opt/python/cp312-cp312/bin/pip install -q ninja && ln -s /opt/python/cp312-cp312/bin/ninja /usr/local/bin/ninja",
    "mkdir /w && cd /src && tar --exclude=./node_modules --exclude='./build*' --exclude=./.venv --exclude=./.git -cf - . | tar -xf - -C /w",
    // corepack in older Node 22.x has stale npm signing keys; install the pinned pnpm with npm.
    `npm install -g --silent pnpm@${pnpmVersion}`,
    "cd /w && pnpm install --frozen-lockfile --silent",
    `npx cmake-js compile --out ${out} ${STABLE.join(" ")} --CDNATIVPY_STATIC_RUNTIME=ON`,
    "pnpm build:ts >/dev/null",
    `export NATIVPY_ADDON_PATH=/w/${out}/Release/nativpy.node`,
    ...(native ? ["npx vitest run"] : ["echo 'emulated container: skipping vitest, running smoke test only'"]),
    "node scripts/smoke-test.mjs ./packages/nativpy/dist/index.js",
    `echo \"max glibc: $(objdump -T ${out}/Release/nativpy.node | grep -oE 'GLIBC_[0-9.]+' | sort -uV | tail -1)\"`,
    `mkdir -p /src/${out}/Release && cp /w/${out}/Release/nativpy.node /src/${out}/Release/`,
  ].join(" && ");
  run("docker", ["run", "--rm", "--platform", platform, "-v", `${root}:/src`, image, "bash", "-c", script]);
  place(`linux-${arch}`, join(root, out, "Release/nativpy.node"));
}

const flag = process.argv.indexOf("--targets");
const targets = flag >= 0 ? process.argv[flag + 1].split(",") : ALL;
for (const t of targets) {
  if (!ALL.includes(t)) throw new Error(`unknown target ${t}; expected one of ${ALL.join(", ")}`);
}
if (targets.some((t) => t.startsWith("linux-"))) ensureDocker();
if (targets.some((t) => t.startsWith("darwin-"))) {
  if (process.platform !== "darwin") throw new Error("macOS prebuilds must be built on macOS.");
  ensureWorkingMacSdk();
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
