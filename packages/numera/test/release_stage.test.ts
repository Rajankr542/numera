import { existsSync, readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
// @ts-expect-error -- plain ESM release script without type declarations
import { assertSelfContained, docsHomepage, packageCompatibility, packageReadme, stripInternalRefs } from "../../../scripts/stage-package.mjs";
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

  it("produces a README and COMPATIBILITY.md that reference only shipped files", () => {
    const readme = packageReadme(read("README.md"));
    const compat = packageCompatibility(read("COMPATIBILITY.md"));
    expect(() => assertSelfContained([["README.md", readme], ["COMPATIBILITY.md", compat]])).not.toThrow();
    expect(readme).toContain("npm install @cyfora/numera");
    expect(readme).not.toContain("## Development");
    const links = [...readme.matchAll(/\]\(([^)]+)\)/g)].map((m) => m[1]);
    for (const l of links) expect(["./COMPATIBILITY.md", "https://unpkg.com/@cyfora/numera/docs/index.html"]).toContain(l);
    // D-030: the package README points at the API reference shipped in the tarball.
    expect(links).toContain("https://unpkg.com/@cyfora/numera/docs/index.html");
    // npm-only publishing: no GitHub Packages / .npmrc instructions (D-030).
    expect(readme).not.toMatch(/GitHub Packages|\.npmrc|npm\.pkg\.github\.com|rajankr542/i);
    expect(compat).toContain("## Documented divergences");
    expect(compat).not.toMatch(/\| Decision \|/);
  });

  it("ships the API reference and pins homepage to this version's copy on unpkg (D-030)", () => {
    const pkg = JSON.parse(read("packages/numera/package.json"));
    expect(pkg.files).toContain("docs");
    expect(pkg).not.toHaveProperty("homepage"); // set at deploy time only
    expect(docsHomepage("@cyfora/numera", "1.2.3-beta.4")).toBe(
      "https://unpkg.com/@cyfora/numera@1.2.3-beta.4/docs/index.html",
    );
    const html = renderPage({ name: pkg.name, version: pkg.version, description: pkg.description });
    expect(() => assertSelfContained([["docs/index.html", html]])).not.toThrow();
    const release = read("scripts/release.mjs");
    expect(release).toContain('"scripts/set-homepage.mjs"');
    expect(release).toContain("checkTarball(");
    expect(release).not.toMatch(/npm\.pkg\.github\.com/);
  });

  it("releases manually only: no GitHub Actions release workflow (D-031)", () => {
    expect(existsSync(new URL(".github/workflows/release.yml", root))).toBe(false);
    expect(read("README.md")).not.toMatch(/NPM_TOKEN|release\.yml|Run workflow/);
  });

  it("rejects repo-only references", () => {
    expect(() => assertSelfContained([["x.md", "see DECISIONS D-001"]])).toThrow(/x\.md:1/);
    expect(() => assertSelfContained([["x.md", "https://github.com/Rajankr542/numera"]])).toThrow();
    expect(() => assertSelfContained([["x.md", "| Platform |"]])).not.toThrow();
  });
});
