import { describe, expect, it } from "vitest";
import np, { NDArray } from "../src/index.js";
// @ts-expect-error -- plain ESM docs data without type declarations
import { categories } from "../../../docs/site/api.mjs";
// @ts-expect-error -- plain ESM docs script without type declarations
import { highlight, renderPage, slug } from "../../../scripts/build-docs.mjs";

interface Entry { name: string; sig: string; desc: string; returns: string; example: string; args?: { name: string; desc: string }[] }
interface Category { id: string; title: string; intro?: string; entries: Entry[] }
const cats = categories as Category[];
const entries = cats.flatMap((c) => c.entries);

/** Numbers match to 1e-9 relative (docs show rounded floats); everything else exactly. */
function close(actual: unknown, expected: unknown, path = "$"): string | null {
  if (typeof expected === "number" && typeof actual === "number") {
    const tol = 1e-9 * Math.max(1, Math.abs(expected));
    return Math.abs(actual - expected) <= tol ? null : `${path}: ${actual} != ${expected}`;
  }
  if (Array.isArray(expected)) {
    if (!Array.isArray(actual) || actual.length !== expected.length) {
      return `${path}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`;
    }
    for (let i = 0; i < expected.length; i++) {
      const r = close(actual[i], expected[i], `${path}[${i}]`);
      if (r) return r;
    }
    return null;
  }
  return Object.is(actual, expected) ? null : `${path}: ${JSON.stringify(actual)} != ${JSON.stringify(expected)}`;
}

/**
 * Rewrites every `expr; // => json` line into a checked call, runs the example
 * with `np` in scope and returns the failures. Every expected line must execute.
 */
function runExample(example: string): { expected: number; failures: string[] } {
  const results: { line: number; actual: unknown; expected: unknown }[] = [];
  let expected = 0;
  const body = example
    .split("\n")
    .map((line, i) => {
      const at = line.indexOf("// =>");
      if (at < 0) return line;
      expected++;
      const expr = line.slice(0, at).trim().replace(/;$/, "");
      const want = line.slice(at + 5).trim();
      JSON.parse(want); // the documented result must be valid JSON
      return `__check(${i + 1}, (${expr}), ${want});`;
    })
    .join("\n");
  const check = (line: number, actual: unknown, want: unknown) => {
    results.push({ line, actual: actual instanceof NDArray ? actual.toArray() : actual, expected: want });
  };
  new Function("np", "__check", body)(np, check);
  const failures = results.flatMap((r) => {
    const d = close(r.actual, r.expected);
    return d ? [`line ${r.line}: ${d}`] : [];
  });
  if (results.length !== expected) failures.push(`only ${results.length}/${expected} checked lines ran`);
  return { expected, failures };
}

describe("API reference site (D-030)", () => {
  it.each(entries.map((e) => [e.name, e] as const))("example for %s runs and matches its // => results", (_, e) => {
    const { expected, failures } = runExample(e.example);
    expect(expected, "every example documents at least one result").toBeGreaterThan(0);
    expect(failures).toEqual([]);
  });

  it("entries are complete and anchors unique", () => {
    const ids = entries.map((e) => slug(e.name));
    expect(new Set(ids).size).toBe(ids.length);
    for (const e of entries) {
      expect(e.sig && e.desc && e.returns && e.example, e.name).toBeTruthy();
    }
    expect(new Set(cats.map((c) => c.id)).size).toBe(cats.length);
  });

  it("documents every public function and namespace member", () => {
    const text = cats
      .flatMap((c) => [c.title, c.intro ?? "", ...c.entries.flatMap((e) => [e.name, e.sig, e.desc, e.example])])
      .join("\n");
    // Internal plumbing exposed for tests/advanced embedding, intentionally undocumented.
    const internal = new Set(["translateNativeError", "wrapNative", "_setBackend"]);
    const names = new Set<string>();
    const add = (keys: Iterable<string>) => { for (const k of keys) if (!internal.has(k)) names.add(k); };
    add(Object.keys(np));
    add(Object.keys(np.linalg));
    add(Object.keys(np.fft));
    add(Object.keys(np.random));
    add(Object.getOwnPropertyNames(np.random.Generator.prototype).filter((k) => k !== "constructor"));
    add(Object.getOwnPropertyNames(np.random.RandomState.prototype).filter((k) => k !== "constructor"));
    const missing = [...names].filter((n) => !new RegExp(`(^|[^A-Za-z0-9_])${n}([^A-Za-z0-9_]|$)`).test(text));
    expect(missing).toEqual([]);
  });

  it("renders a self-contained page with every entry and no external requests", () => {
    const html: string = renderPage({ name: "@cyfora/numera", version: "9.9.9", description: "d <x>" });
    expect(html.startsWith("<!doctype html>")).toBe(true);
    for (const e of entries) expect(html).toContain(`id="${slug(e.name)}"`);
    expect(html).not.toMatch(/(src|href)\s*=\s*["']?(https?:)?\/\//i);
    expect(html).not.toMatch(/@import|url\(/i);
    expect(html).toContain("d &lt;x&gt;");
    expect(html).toContain("npm install @cyfora/numera");
  });

  it("highlighter escapes HTML and marks results", () => {
    const out: string = highlight('const s = "<b>"; // => 1');
    expect(out).toContain("&lt;b&gt;");
    expect(out).toContain('<span class="t-res">// =&gt; 1</span>');
    expect(out).not.toContain("<b>");
  });
});
