import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
// @ts-expect-error -- plain ESM release script without type declarations
import { DOCS_URL, RAW_URL, REPO_URL, assertSelfContained, packageCompatibility, packageReadme, stripInternalRefs } from "../../../scripts/stage-package.mjs";
// @ts-expect-error -- plain ESM docs script without type declarations
import { renderPage } from "../../../scripts/build-docs.mjs";

const root = new URL("../../../", import.meta.url);
const read = (f: string) => readFileSync(new URL(f, root), "utf8");

describe("release: self-contained package docs (D-029)", () => {
  it("strips internal decision/plan references without eating code", () => {
    expect(stripInternalRefs("one JS pass (D-004) done")).toBe("one JS pass done");
    expect(stripInternalRefs("Typed errors (PLAN §28, DECISIONS D-006).")).toBe("Typed errors.");
    expect(stripInternalRefs("Legacy (MT19937, D-019).")).toBe("Legacy (MT19937).");
    expect(stripInternalRefs("Methods live on NDArray; see D-015.")).toBe("Methods live on NDArray.");
    expect(stripInternalRefs("read with `toTypedArray()` (interleaved, D-008).")).toBe(
      "read with `toTypedArray()` (interleaved).",
    );
    expect(stripInternalRefs("the D-004 inference flags")).toBe("the inference flags");
  });

  it("produces a README and COMPATIBILITY.md whose links work outside the repo (D-240)", () => {
    const readme = packageReadme(read("README.md"));
    const compat = packageCompatibility(read("COMPATIBILITY.md"));
    expect(() => assertSelfContained([["README.md", readme], ["COMPATIBILITY.md", compat]])).not.toThrow();
    expect(readme).toContain("npm install @cyfora/numera");
    expect(readme).not.toContain("## Development");
    expect(readme).not.toContain("Publishing a release");
    expect(readme).toContain(`${REPO_URL}/blob/main/CONTRIBUTING.md`);
    // Badges and the hero image are the only links that leave the repo/docs (D-241).
    const allowed = [REPO_URL, DOCS_URL, RAW_URL, "https://img.shields.io/", "https://www.npmjs.com/package/@cyfora/numera"];
    const links = [...readme.matchAll(/\]\(([^)]+)\)/g)].map((m) => m[1]);
    for (const l of links) expect(allowed.some((a) => l.startsWith(a)), l).toBe(true);
    for (const m of readme.matchAll(/(?:srcset|src)="([^"]+)"/g)) expect(m[1].startsWith(`${RAW_URL}/docs/images/`)).toBe(true);
    expect(links).toContain(DOCS_URL);
    expect(links).toContain(`${REPO_URL}/blob/main/COMPATIBILITY.md`);
    expect(readme).toContain(`- Source code: ${REPO_URL}`);
    expect(readme).toContain(`- Issues: ${REPO_URL}/issues`);
    // npm-only publishing: no GitHub Packages / .npmrc instructions (D-030).
    expect(readme).not.toMatch(/GitHub Packages|\.npmrc|npm\.pkg\.github\.com/i);
    expect(readme).not.toContain("unpkg.com");
    expect(compat).toContain("## Documented divergences");
    expect(compat).not.toMatch(/\| Decision \|/);
  });

  it("links npm users to the public repo and the hosted docs; docs are not shipped (D-240)", () => {
    const pkg = JSON.parse(read("packages/numera/package.json"));
    expect(DOCS_URL).toBe("https://numera.cyfora.in");
    expect(REPO_URL).toBe("https://github.com/Rajankr542/numera");
    expect(pkg.homepage).toBe(DOCS_URL);
    expect(pkg.repository).toEqual({ type: "git", url: `git+${REPO_URL}.git`, directory: "packages/numera" });
    expect(pkg.bugs).toEqual({ url: `${REPO_URL}/issues` });
    expect(pkg.files).not.toContain("docs");
    const html: string = renderPage({ name: pkg.name, version: pkg.version, description: pkg.description });
    expect(html).toContain(`<link rel="canonical" href="${DOCS_URL}/">`);
    expect(html).toContain(`href="${REPO_URL}"`);
    expect(existsSync(new URL("scripts/set-homepage.mjs", root))).toBe(false);
    const release = read("scripts/release.mjs");
    expect(release).toContain("checkTarball(");
    expect(release).toContain('"gh", ["release", "create"');
    expect(release).not.toMatch(/npm\.pkg\.github\.com|unpkg/);
  });

  it("releases manually only: no GitHub Actions release workflow (D-031)", () => {
    expect(existsSync(new URL(".github/workflows/release.yml", root))).toBe(false);
    expect(read("README.md")).not.toMatch(/NPM_TOKEN|release\.yml|Run workflow/);
  });

  it("rejects repo-only references", () => {
    expect(() => assertSelfContained([["x.md", "see DECISIONS D-001"]])).toThrow(/x\.md:1/);
    expect(() => assertSelfContained([["x.md", "[c](./COMPATIBILITY.md)"]])).toThrow();
    expect(() => assertSelfContained([["x.d.ts", "[Symbol.iterator](): this;"]])).not.toThrow();
    expect(() => assertSelfContained([["x.md", "[repo](https://github.com/Rajankr542/numera)"]])).not.toThrow();
    expect(() => assertSelfContained([["x.md", "| Platform |"]])).not.toThrow();
  });
});
