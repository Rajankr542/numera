// Pins package.json "homepage" to the API reference shipped in this version's
// tarball, served by unpkg (DECISIONS D-030). Run at deploy time, after the
// version is final and before `npm publish`, by scripts/release.mjs. The
// committed package.json has no homepage.
// Usage: node scripts/set-homepage.mjs
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { docsHomepage } from "./stage-package.mjs";

const pkgDir = join(resolve(dirname(fileURLToPath(import.meta.url)), ".."), "packages/numera");

export function setHomepage() {
  const { name, version } = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8"));
  const url = docsHomepage(name, version);
  execFileSync("npm", ["pkg", "set", `homepage=${url}`], { cwd: pkgDir, stdio: "inherit" });
  return url;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  console.log(`homepage=${setHomepage()}`);
}
