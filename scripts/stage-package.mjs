// Stage the published package contents in packages/numera (DECISIONS D-029).
// The npm package is public but the repository is private, so everything the
// package refers to must ship inside it: this writes a user-facing README.md
// (install, samples, API notes; no development/release sections or repo
// links), COMPATIBILITY.md (without the internal decision column), LICENSE
// and the API reference docs/index.html (D-030, served by unpkg as the npm
// homepage), and removes references to repo-only documents (PLAN §N, D-0NN, DECISIONS,
// ROADMAP) from the compiled dist/ files. Source maps are excluded via
// package.json "files". Run after `pnpm build:ts`; used by scripts/release.mjs.
import { copyFileSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildDocs } from "./build-docs.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages/numera");

/** npm homepage: the API reference shipped in this exact version's tarball (D-030). */
export function docsHomepage(name, version) {
  return `https://unpkg.com/${name}@${version}/docs/index.html`;
}

/** Removes references to internal design documents (not shipped, repo is private). */
export function stripInternalRefs(text) {
  const ref = String.raw`(?:(?:DECISIONS|PLAN)\s+)?(?:D-\d{3}(?:\/D-\d{3})*|§\d+(?:\/§\d+)*|M\d+)(?:\s+inference)?`;
  return text
    .replace(/until D-008\b/g, "until complex element access is implemented")
    .replace(/\(D-005 applies to/g, "(the `toArray` 2^53 limit applies to")
    .replace(new RegExp(String.raw`\s*\((?:see\s+)?${ref}(?:[,;]\s*${ref})*\)`, "g"), "")
    .replace(new RegExp(String.raw`,\s*${ref}(?=\))`, "g"), "")
    .replace(new RegExp(String.raw`\s*[—-]\s*PLAN\s+§\d+`, "g"), "")
    .replace(/,\s*see D-\d{3} and divergences below/g, "; see divergences below")
    .replace(/;\s*see D-\d{3}\./g, ".")
    .replace(/\s*(?:[Ss]emantics(?: and divergences)?|[Ss]ee):?\s+DECISIONS\s+D-\d{3}\.?/g, "")
    .replace(/\s*DECISIONS\s+D-\d{3}\.?/g, "")
    .replace(/\bD-\d{3}(?:\/D-\d{3})*[ \t]*(\n[ \t]*\*)?[ \t]*(?=[a-z])/g, (_m, nl) => (nl ? `${nl} ` : ""));
}

/**
 * Drops the internal last column of every Markdown table whose header's last
 * cell is "Decision" (D-029; milestone branches append such tables, D-056).
 */
function dropDecisionColumn(md) {
  const lines = md.split("\n");
  let inTable = false;
  return lines
    .map((line) => {
      if (!line.startsWith("|")) {
        inTable = false;
        return line;
      }
      const cells = line.split(/(?<!\\)\|/);
      // ["", c1, ..., cN, ""]
      if (!inTable) inTable = cells.length >= 3 && cells[cells.length - 2].trim() === "Decision";
      return inTable ? cells.slice(0, -2).concat("").join("|") : line;
    })
    .join("\n");
}

export function packageReadme(readme) {
  let md = readme.slice(0, readme.indexOf("\n## Development"));
  md = md.replace(
    /> \*\*Status: ([^*]+)\*\* Not all of NumPy is implemented yet\. See\n> \[COMPATIBILITY\.md\]\(\.\/COMPATIBILITY\.md\) for what is verified against NumPy,\n> and \[ROADMAP\.md\]\(\.\/ROADMAP\.md\) for what is planned\./,
    "> **Status: $1** Not all of NumPy is implemented yet. See\n> [COMPATIBILITY.md](./COMPATIBILITY.md) (included in this package) for what is\n> verified against NumPy and what is not implemented.",
  );
  md = md.replace(
    /\nWindows and Alpine\/musl Linux have no prebuilt binaries yet\. On those, use a\n\[source build\]\(#development\)\./,
    "\nWindows and Alpine/musl Linux have no prebuilt binaries yet.",
  );
  md = md.replace(
    /The addon is found automatically, in this order:\n`NATIVPY_ADDON_PATH`, then the prebuild bundled in the package\n\(`prebuilds\/<platform>-<arch>\/nativpy\.node`\), then a local source build in\n`build\/Release\/` \([^)]*\)\. Published prebuilds use the stable Node-API,\nso one binary works on every Node ≥ 18\. Source builds use experimental\nNode-API, which frees memory sooner \([^)]*\)\./,
    "The addon for your platform ships in the package\n(`prebuilds/<platform>-<arch>/nativpy.node`) and is loaded automatically; set\n`NATIVPY_ADDON_PATH` to load a different build. The prebuilds use the stable\nNode-API, so one binary works on every Node ≥ 18.",
  );
  md = md.replace(/\n## Performance\n[\s\S]*?(?=\n## |$)/, "\n");
  md = stripInternalRefs(md);
  return `${md.trimEnd()}\n\n## License\n\nMIT (see \`LICENSE\` in this package).\n`;
}

export function packageCompatibility(compat) {
  let md = dropDecisionColumn(compat);
  md = md.replace(/`pnpm test:diff`/g, "the project's NumPy differential test suite");
  md = md.replace(/\s*Everything from PLAN M11 onward \(see ROADMAP\.md\)\.\s*/g, " ");
  md = md.replace(/\bnativpy\b/g, "numera");
  md = stripInternalRefs(md).replace(/[ \t]+$/gm, "");
  return md;
}

function stageDist(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) stageDist(p);
    else if (/\.(js|d\.ts)$/.test(entry.name)) {
      const src = readFileSync(p, "utf8");
      const out = stripInternalRefs(src)
        .replace(/\/\/ src\/ or dist\/ -> packages\/\w+ -> repo root/g, "// development builds only (repository checkout)")
        .replace(/\bnativpy (data type|error)/g, "numera $1")
        .replace(/raised by nativpy\b/g, "raised by numera");
      if (out !== src) writeFileSync(p, out);
    }
  }
}

/** Throws if a staged file still references repo-only content (private repo). */
export function assertSelfContained(files) {
  const forbidden = /D-\d{3}|\bPLAN\b|DECISIONS|ROADMAP|PERFORMANCE\.md|ARCHITECTURE|AGENTS\.md|github\.com|[Rr]ajankr542|#development|pnpm (?:test|build|release)|[Ss]ource build/;
  const bad = [];
  for (const [name, text] of files) {
    text.split("\n").forEach((line, i) => {
      if (forbidden.test(line)) bad.push(`${name}:${i + 1}: ${line.trim()}`);
    });
  }
  if (bad.length) throw new Error(`Package still references repo-only content:\n${bad.join("\n")}`);
}

function listDist(dir, out = []) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) listDist(p, out);
    else if (/\.(js|d\.ts)$/.test(entry.name)) out.push(p);
  }
  return out;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  writeFileSync(join(pkgDir, "README.md"), packageReadme(readFileSync(join(root, "README.md"), "utf8")));
  writeFileSync(join(pkgDir, "COMPATIBILITY.md"), packageCompatibility(readFileSync(join(root, "COMPATIBILITY.md"), "utf8")));
  copyFileSync(join(root, "LICENSE"), join(pkgDir, "LICENSE"));
  stageDist(join(pkgDir, "dist"));
  buildDocs(join(pkgDir, "docs"));
  const staged = [
    "README.md",
    "COMPATIBILITY.md",
    "docs/index.html",
    ...listDist(join(pkgDir, "dist")).map((p) => p.slice(pkgDir.length + 1)),
  ];
  assertSelfContained(staged.map((f) => [f, readFileSync(join(pkgDir, f), "utf8")]));
  console.log(`Staged and checked ${staged.length} files in packages/numera (README.md, COMPATIBILITY.md, LICENSE, docs/, dist/)`);
}
