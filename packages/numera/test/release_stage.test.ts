import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
// @ts-expect-error -- plain ESM release script without type declarations
import { assertSelfContained, packageCompatibility, packageReadme, stripInternalRefs } from "../../../scripts/stage-package.mjs";

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
    for (const l of links) expect(["./COMPATIBILITY.md"]).toContain(l);
    expect(compat).toContain("## Documented divergences");
    expect(compat).not.toMatch(/\| Decision \|/);
  });

  it("rejects repo-only references", () => {
    expect(() => assertSelfContained([["x.md", "see DECISIONS D-001"]])).toThrow(/x\.md:1/);
    expect(() => assertSelfContained([["x.md", "https://github.com/Rajankr542/numera"]])).toThrow();
    expect(() => assertSelfContained([["x.md", "| Platform |"]])).not.toThrow();
  });
});
