// Stage the published package contents in packages/numera (DECISIONS D-029, D-240).
// Writes a user-facing README.md (install, samples, API notes; no
// development/release sections; relative links rewritten to the public GitHub
// repo), COMPATIBILITY.md (without the internal decision column) and LICENSE,
// and removes internal design-document references (PLAN §N, D-0NN) from the
// compiled dist/ files. The API reference is hosted at DOCS_URL, not shipped.
// Source maps are excluded via package.json "files". Run after
// `pnpm build:ts`; used by scripts/release.mjs.
import { copyFileSync, readFileSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages/numera");

/** Public source repository (D-240). */
export const REPO_URL = "https://github.com/Rajankr542/numera";
/** Raw file host for images in the npm README (D-241). */
export const RAW_URL = "https://raw.githubusercontent.com/Rajankr542/numera/main";
/** Hosted API reference and npm homepage (D-240). */
export const DOCS_URL = "https://numera.cyfora.in";

/** Removes references to internal design documents (noise for package users). */
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
      const lastCell = (cells[cells.length - 2] ?? "").trim();
      // Activate on a header row whose last column is "Decision".
      if (!inTable) inTable = cells.length >= 3 && lastCell === "Decision";
      if (inTable) return cells.slice(0, -2).concat("").join("|");
      // Even outside a recognised table, strip a trailing "D-NNN" cell so
      // milestone branches that append rows without a fresh header still pass
      // the self-contained check (D-029, D-056).
      if (/^D-\d{3}$/.test(lastCell)) return cells.slice(0, -2).concat("").join("|");
      return line;
    })
    .join("\n");
}

/**
 * The npm README is the root README up to "## Contributing" (D-241), with
 * relative links made absolute and short Contributing/Links/License sections.
 */
export function packageReadme(readme) {
  const cut = readme.indexOf("\n## Contributing");
  if (cut < 0) throw new Error('README.md has no "## Contributing" section to cut the package README at');
  let md = readme.slice(0, cut);
  // npm does not resolve relative links reliably; point them at the public repo.
  md = md.replace(/(\]\(|srcset="|src=")\.\/(docs\/images\/[^)"]+)/g, `$1${RAW_URL}/$2`);
  md = md.replace(/\]\(\.\/([^)]+)\)/g, `](${REPO_URL}/blob/main/$1)`);
  md = stripInternalRefs(md);
  return `${md.trimEnd()}

## Contributing

Issues and pull requests are welcome. See the
[contributing guide](${REPO_URL}/blob/main/CONTRIBUTING.md) for the development
setup and test suites.

## Links

- API reference: ${DOCS_URL}
- Source code: ${REPO_URL}
- Issues: ${REPO_URL}/issues

## License

MIT (see \`LICENSE\` in this package).
`;
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

/**
 * Throws if a staged file still has internal design references, contributor
 * commands, or relative links that would break outside the repository.
 */
export function assertSelfContained(files) {
  const forbidden = /D-\d{3}|\bPLAN\b|DECISIONS|pnpm (?:test|build|release)/;
  const relativeLink = /\]\((?!https:\/\/)/;
  const bad = [];
  for (const [name, text] of files) {
    const md = name.endsWith(".md");
    text.split("\n").forEach((line, i) => {
      if (forbidden.test(line) || (md && relativeLink.test(line))) bad.push(`${name}:${i + 1}: ${line.trim()}`);
    });
  }
  if (bad.length) throw new Error(`Package still references repo-only content:\n${bad.join("\n")}`);
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  writeFileSync(join(pkgDir, "README.md"), packageReadme(readFileSync(join(root, "README.md"), "utf8")));
  writeFileSync(join(pkgDir, "COMPATIBILITY.md"), packageCompatibility(readFileSync(join(root, "COMPATIBILITY.md"), "utf8")));
  copyFileSync(join(root, "LICENSE"), join(pkgDir, "LICENSE"));
  stageDist(join(pkgDir, "dist"));
  // The user-facing docs must be clean. dist/ comments are stripped best-effort
  // only: any reference left there resolves in the public repo (D-240).
  const checked = ["README.md", "COMPATIBILITY.md"];
  assertSelfContained(checked.map((f) => [f, readFileSync(join(pkgDir, f), "utf8")]));
  console.log("Staged packages/numera README.md, COMPATIBILITY.md, LICENSE and dist/; checked README.md and COMPATIBILITY.md");
}
