// Builds the static documentation site into html/.
//
//   node scripts/build.mjs            all versions in versions/
//
// Layout of the output (relative links only, works from any static host or
// sub-path, and from file://):
//   html/index.html              redirects to the latest version
//   html/versions.json           list of versions (read by the version picker)
//   html/assets/                 shared CSS/JS/icons
//   html/v<version>/...          one self-contained folder per version
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { esc, highlight } from "./lib/highlight.mjs";
import { inline, renderMarkdown, slugify } from "./lib/markdown.mjs";
import { page, sidebar, SITE, logo } from "./lib/layout.mjs";

const here = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outRoot = join(here, "html");
const numpyDocs = JSON.parse(readFileSync(join(here, "data/numpy-2.5.3.json"), "utf8"));

const cmpVer = (a, b) => {
  const pa = a.split(/[.-]/).map((x) => (isNaN(+x) ? x : +x));
  const pb = b.split(/[.-]/).map((x) => (isNaN(+x) ? x : +x));
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    if (pa[i] === pb[i]) continue;
    if (pa[i] === undefined) return -1;
    if (pb[i] === undefined) return 1;
    return pa[i] < pb[i] ? -1 : 1;
  }
  return 0;
};
const versions = readdirSync(join(here, "versions"))
  .filter((v) => existsSync(join(here, "versions", v, "meta.json")))
  .sort(cmpVer)
  .reverse();
const latest = versions[0];

/** Entry anchor, identical to the npm-bundled reference (scripts/build-docs.mjs in the main repo). */
const anchor = (name) => name.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-|-$/g, "");

// Reference sections are grouped like NumPy's own reference.
const GROUPS = [
  ["Array objects", ["ndarray", "dtype"]],
  ["Universal functions", ["@ufuncs", "math", "logic"]],
  ["Routines", ["creation", "shape", "indexing", "grids", "tri-indices", "reduce", "reduce-where", "sorting", "statistics", "polynomials", "windows", "io", "utilities", "errors"]],
  ["Modules", ["linalg", "fft", "random", "random-bitgen", "random-discrete", "random-continuous", "random-multivariate", "random-state", "emath", "testing", "polynomial", "ma", "strings", "rec"]],
];
const SURFACE_PREFIX = {
  np: "np.", linalg: "np.linalg.", fft: "np.fft.", random: "np.random.", Generator: "rng.", RandomState: "np.random.RandomState#",
  ndarray: "a.", ma: "np.ma.", polynomial: "np.polynomial.", strings: "np.strings.", char: "np.char.", rec: "np.rec.",
  emath: "np.emath.", testing: "np.testing.",
};
const SURFACE_TITLE = {
  np: "numpy", linalg: "numpy.linalg", fft: "numpy.fft", random: "numpy.random", Generator: "numpy.random.Generator",
  RandomState: "numpy.random.RandomState", ndarray: "numpy.ndarray", ma: "numpy.ma", polynomial: "numpy.polynomial",
  strings: "numpy.strings", char: "numpy.char", rec: "numpy.rec", emath: "numpy.emath", testing: "numpy.testing",
};

// Ufunc families for the ufunc page (NumPy's "Available ufuncs" grouping).
const UFUNC_FAMILIES = [
  ["Math operations", /^(add|subtract|multiply|matmul|divide|logaddexp2?|true_divide|floor_divide|negative|positive|power|pow|float_power|remainder|mod|fmod|divmod|absolute|abs|fabs|rint|sign|heaviside|conj|conjugate|exp|exp2|log|log2|log10|expm1|log1p|sqrt|square|cbrt|reciprocal|gcd|lcm|vecdot|matvec|vecmat)$/],
  ["Trigonometric functions", /^(a?sin|a?cos|a?tan|arc\w+|a?sinh|a?cosh|a?tanh|atan2|hypot|degrees|radians|deg2rad|rad2deg)$/],
  ["Bit-twiddling functions", /^(bitwise_\w+|invert|left_shift|right_shift)$/],
  ["Comparison functions", /^(greater|greater_equal|less|less_equal|not_equal|equal|logical_\w+|maximum|minimum|fmax|fmin)$/],
  ["Floating functions", /^(isfinite|isinf|isnan|isnat|fabs|signbit|copysign|nextafter|spacing|modf|ldexp|frexp|floor|ceil|trunc)$/],
];

rmSync(outRoot, { recursive: true, force: true });
mkdirSync(join(outRoot, "assets"), { recursive: true });
cpSync(join(here, "site/assets"), join(outRoot, "assets"), { recursive: true });

const allPages = {};
const built = [];
for (const version of versions) built.push(await loadVersion(version));
for (const v of built) allPages[v.version] = v.pages.map((p) => p.path);
for (const v of built) writeVersion(v);

writeFileSync(join(outRoot, "versions.json"), JSON.stringify({ latest, versions }, null, 1) + "\n");
writeFileSync(join(outRoot, "index.html"), redirectPage(`v${latest}/index.html`));
writeFileSync(join(outRoot, "latest.html"), redirectPage(`v${latest}/index.html`));
writeFileSync(join(outRoot, "404.html"), notFound());
writeFileSync(join(outRoot, "robots.txt"), "User-agent: *\nAllow: /\n");
writeFileSync(join(outRoot, ".nojekyll"), "");
console.log(`html/: ${built.map((v) => `v${v.version} (${v.pages.length} pages)`).join(", ")}; latest v${latest}`);

// ---------------------------------------------------------------------------

async function loadVersion(version) {
  const dir = join(here, "versions", version);
  const meta = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8"));
  const names = JSON.parse(readFileSync(join(dir, "names.json"), "utf8"));
  const { categories } = await import(pathToFileURL(join(dir, "api.mjs")).href);
  const guideOrder = JSON.parse(readFileSync(join(dir, "guides/nav.json"), "utf8"));
  const compat = readFileSync(join(dir, "COMPATIBILITY.md"), "utf8");
  const changelog = readFileSync(join(here, "site/changelog.md"), "utf8");

  // Entries → page + anchor; JS names → entries.
  const entryLoc = new Map();
  for (const c of categories) for (const e of c.entries) entryLoc.set(e.name, { cat: c, entry: e, href: `reference/${c.id}.html#${anchor(e.name)}` });
  const surfaceOfToken = (tok) => {
    for (const [s, pre] of Object.entries(SURFACE_PREFIX).sort((a, b) => b[1].length - a[1].length)) if (tok.startsWith(pre)) return [s, tok.slice(pre.length)];
    return null;
  };
  const byJs = {};
  for (const [s, info] of Object.entries(names.surfaces)) {
    byJs[s] = new Map(info.implemented.map((i) => [i.js, i]));
    for (const i of info.implemented) i.surface = s;
  }
  const docFor = (s, numpyName) => numpyDocs.surfaces[s]?.[numpyName] ?? { summary: "", url: "" };

  // Which implemented names each entry documents (from its name and signature).
  for (const loc of entryLoc.values()) {
    const e = loc.entry;
    const found = [];
    const add = (s, js) => {
      const hit = byJs[s]?.get(js);
      if (hit && !found.includes(hit)) found.push(hit);
    };
    const text = `${e.name.includes(".") || e.name.startsWith("a.") ? "" : "np."}${e.name} ${e.sig}`;
    for (const m of text.matchAll(/(?:new\s+)?((?:np\.(?:linalg|fft|random|ma|polynomial|strings|char|rec|emath|testing)\.|np\.|a\.|rng\.)[A-Za-z_]\w*)/g)) {
      const r = surfaceOfToken(m[1]);
      if (r) add(r[0], r[1]);
    }
    const bare = e.name.split(" ")[0];
    const dotted = bare.match(/^(linalg|fft|random|ma|polynomial|strings|char|rec|emath|testing|Generator)\.(\w+)$/);
    if (dotted) add(dotted[1], dotted[2]);
    if (loc.cat.id.startsWith("random-")) add("Generator", bare);
    if (/^\w+$/.test(bare)) add("np", bare);
    loc.covers = found;
    for (const f of found) (f.docs ??= []).push(loc.href);
  }

  // Ufunc data (only names that are NumPy ufuncs in numpy.* and implemented).
  const ufuncs = names.surfaces.np.implemented.filter((i) => i.kind === "ufunc");

  // Guides.
  const guides = guideOrder.flatMap((g) => g.pages.map((p) => ({ ...p, group: g.title })));
  const pages = [];
  for (const g of guides) {
    const md = readFileSync(join(dir, "guides", `${g.file}.md`), "utf8");
    pages.push({ kind: "guide", path: g.file === "index" ? "index.html" : `guides/${g.file}.html`, guide: g, md });
  }
  pages.push({ kind: "ref-index", path: "reference/index.html" });
  pages.push({ kind: "ufuncs", path: "reference/ufuncs.html" });
  for (const c of categories) pages.push({ kind: "category", path: `reference/${c.id}.html`, cat: c });
  pages.push({ kind: "numpy-index", path: "reference/numpy-index.html" });
  pages.push({ kind: "compat", path: "compatibility.html" });
  pages.push({ kind: "changelog", path: "changelog.html" });
  return { version, meta, names, categories, guides, guideOrder, pages, entryLoc, byJs, docFor, ufuncs, compat, changelog };
}

function writeVersion(V) {
  const verRoot = join(outRoot, `v${V.version}`);
  const ver = { version: V.version, latest, versions, pages: allPages };
  const search = [];
  const rel = (from, to) => "../".repeat(from.split("/").length - 1) + to;

  // Link rewriting for markdown: "/x.html" → version root, "ref:name" → entry.
  const linker = (from) => (h) => {
    if (h.startsWith("ref:")) {
      const key = h.slice(4);
      const [name, frag] = key.split("#");
      const loc = V.entryLoc.get(name) ?? [...V.entryLoc.values()].find((l) => l.covers.some((c) => c.js === name && c.surface === "np"));
      if (!loc) throw new Error(`v${V.version}: unresolved ref:${key} in ${from}`);
      return rel(from, loc.href) + (frag ? "" : "");
    }
    if (h.startsWith("/")) {
      const target = h.slice(1).split("#")[0];
      if (!V.pages.some((p) => p.path === target)) throw new Error(`v${V.version}: broken link ${h} in ${from}`);
      return rel(from, h.slice(1));
    }
    return h;
  };

  // Ordered list of all "reading" pages for prev/next.
  const refOrder = [];
  for (const [, ids] of GROUPS) for (const id of ids) {
    if (id === "@ufuncs") refOrder.push({ path: "reference/ufuncs.html", title: "Universal functions (ufunc)" });
    else {
      const c = V.categories.find((x) => x.id === id);
      if (c) refOrder.push({ path: `reference/${c.id}.html`, title: c.title });
    }
  }
  for (const c of V.categories) if (!refOrder.some((r) => r.path === `reference/${c.id}.html`)) refOrder.push({ path: `reference/${c.id}.html`, title: c.title });
  const guideSeq = V.guides.map((g) => ({ path: g.file === "index" ? "index.html" : `guides/${g.file}.html`, title: g.title }));

  const guideSidebar = (path) =>
    sidebar(V.guideOrder.map((g) => ({
      title: g.title,
      items: g.pages.map((p) => {
        const to = p.file === "index" ? "index.html" : `guides/${p.file}.html`;
        return { title: p.title, href: rel(path, to), active: to === path };
      }),
    })));
  const refSidebar = (path) => {
    const groups = [{ title: "Reference", items: [
      { title: "Overview", href: rel(path, "reference/index.html"), active: path === "reference/index.html" },
      { title: "NumPy name index", href: rel(path, "reference/numpy-index.html"), active: path === "reference/numpy-index.html" },
    ] }];
    const placed = new Set();
    for (const [title, ids] of GROUPS) {
      const items = [];
      for (const id of ids) {
        if (id === "@ufuncs") { items.push({ title: "ufunc overview", href: rel(path, "reference/ufuncs.html"), active: path === "reference/ufuncs.html", badge: V.ufuncs.length }); continue; }
        const c = V.categories.find((x) => x.id === id);
        if (!c) continue;
        placed.add(c.id);
        items.push({ title: c.title, href: rel(path, `reference/${c.id}.html`), active: path === `reference/${c.id}.html`, badge: c.entries.length });
      }
      if (items.length) groups.push({ title, items });
    }
    const rest = V.categories.filter((c) => !placed.has(c.id));
    if (rest.length) groups.push({ title: "More", items: rest.map((c) => ({ title: c.title, href: rel(path, `reference/${c.id}.html`), active: path === `reference/${c.id}.html`, badge: c.entries.length })) });
    return sidebar(groups);
  };
  const neighbours = (seq, path) => {
    const i = seq.findIndex((x) => x.path === path);
    if (i < 0) return {};
    const mk = (x) => x && { href: rel(path, x.path), title: x.title };
    return { prev: mk(seq[i - 1]), next: mk(seq[i + 1]) };
  };

  for (const P of V.pages) {
    let html;
    const base = { ver, toRoot: rel(P.path, "../"), toVer: rel(P.path, ""), path: P.path };
    if (P.kind === "guide") {
      const r = renderMarkdown(fillTokens(P.md, V), { link: linker(P.path) });
      const isHome = P.path === "index.html";
      const body = isHome ? homeBody(V, r, P.path) : `<article class="prose"><p class="eyebrow">${esc(P.guide.group)}</p><h1>${inline(r.title)}</h1>${r.html}</article>`;
      html = page({
        ...base, title: isHome ? "NumPy for JavaScript and TypeScript" : r.title, description: P.guide.description ?? r.lead,
        body, tab: "guides", sidebar: guideSidebar(P.path), toc: isHome ? [] : r.headings.filter((h) => h.level <= 3),
        bodyClass: isHome ? "home" : "", ...neighbours(guideSeq, P.path),
      });
      search.push({ t: r.title || "Introduction", u: P.path, k: "Guide", s: P.guide.description ?? "" });
      for (const h of r.headings.filter((h) => h.level === 2)) search.push({ t: h.text, u: `${P.path}#${h.id}`, k: r.title, s: "" });
    } else if (P.kind === "category") {
      const { body, toc } = categoryBody(V, P.cat, P.path);
      html = page({ ...base, title: P.cat.title, description: stripMd(P.cat.intro ?? `${P.cat.title}: numera API reference.`), body, tab: "reference", sidebar: refSidebar(P.path), toc, ...neighbours(refOrder, P.path) });
      search.push({ t: P.cat.title, u: P.path, k: "Reference section", s: stripMd(P.cat.intro ?? "").slice(0, 140) });
      for (const e of P.cat.entries) {
        const loc = V.entryLoc.get(e.name);
        search.push({ t: displayName(e), u: `${P.path}#${anchor(e.name)}`, k: P.cat.title, s: stripMd(e.desc).slice(0, 140), n: loc.covers.map((c) => SURFACE_TITLE[c.surface] + "." + c.numpy).join(" ") });
      }
    } else if (P.kind === "ref-index") {
      html = page({ ...base, title: "API reference", description: "Every numera function, grouped like the NumPy reference.", body: refIndexBody(V, P.path), tab: "reference", sidebar: refSidebar(P.path), toc: [] });
    } else if (P.kind === "ufuncs") {
      const { body, toc } = ufuncBody(V, P.path, linker(P.path));
      html = page({ ...base, title: "Universal functions (ufunc)", description: "Element-wise functions with broadcasting, type promotion and ufunc methods.", body, tab: "reference", sidebar: refSidebar(P.path), toc, ...neighbours(refOrder, P.path) });
      search.push({ t: "Universal functions (ufunc)", u: P.path, k: "Reference", s: "Broadcasting, type promotion, out=, where=, reduce/accumulate/outer/at" });
    } else if (P.kind === "numpy-index") {
      const { body, toc } = numpyIndexBody(V, P.path);
      html = page({ ...base, title: "NumPy name index", description: `Every NumPy ${V.meta.numpy} name and its numera equivalent.`, body, tab: "reference", sidebar: refSidebar(P.path), toc });
      for (const [s, info] of Object.entries(V.names.surfaces)) for (const i of info.implemented) {
        if (i.docs?.length) continue;
        search.push({ t: (SURFACE_PREFIX[s] ?? "") + i.js, u: `${P.path}#${s}-${i.numpy}`, k: SURFACE_TITLE[s], s: V.docFor(s, i.numpy).summary.slice(0, 140), n: `${SURFACE_TITLE[s]}.${i.numpy}` });
      }
    } else if (P.kind === "compat") {
      const r = renderMarkdown(V.compat.replace(/^# COMPATIBILITY/, "# NumPy compatibility"), { link: linker(P.path) });
      html = page({ ...base, title: "NumPy compatibility", description: "What is verified against NumPy and where numera differs.", tab: "compat", sidebar: guideSidebar(P.path),
        body: `<article class="prose compat"><p class="eyebrow">Reference</p><h1>NumPy compatibility</h1>${r.html}</article>`, toc: r.headings.filter((h) => h.level <= 2) });
      search.push({ t: "NumPy compatibility", u: P.path, k: "Reference", s: "Verified behaviour and documented divergences" });
    } else if (P.kind === "changelog") {
      const r = renderMarkdown(V.changelog, { link: linker(P.path) });
      html = page({ ...base, title: "Changelog", description: "Release notes for every numera version.", tab: "changelog", sidebar: guideSidebar(P.path),
        body: `<article class="prose"><p class="eyebrow">Releases</p><h1>${inline(r.title)}</h1>${r.html}</article>`, toc: r.headings.filter((h) => h.level <= 2) });
      search.push({ t: "Changelog", u: P.path, k: "Releases", s: "Release notes" });
    }
    const file = join(verRoot, P.path);
    mkdirSync(dirname(file), { recursive: true });
    writeFileSync(file, html);
  }
  writeFileSync(join(verRoot, "search-index.js"), `window.NUMERA_INDEX=${JSON.stringify(search)};\n`);
}

// --- page bodies -------------------------------------------------------------

function stripMd(s) { return String(s).replace(/`([^`]+)`/g, "$1").replace(/\*\*([^*]+)\*\*/g, "$1"); }
function displayName(e) {
  if (/^(a|rng)\./.test(e.name) || e.name.startsWith("np.")) return e.name;
  if (/^[a-z]\w*(\.\w+)*$/i.test(e.name) && !/\s/.test(e.name)) return e.name.startsWith("Generator.") ? e.name.replace(/^Generator\./, "rng.") : `np.${e.name}`;
  return e.name;
}

/** {{version}}, {{coverage}} ... tokens in guides. */
function fillTokens(md, V) {
  const m = V.meta;
  const t = {
    version: m.version, package: m.package, numpy: m.numpy, node: m.node, documented: String(m.documented),
    implemented: String(m.coverage.implemented), total: String(m.coverage.total), coverage: String(m.coverage.coverage),
    ufuncs: String(V.ufuncs.length), date: m.date,
  };
  return md.replace(/\{\{(\w+)\}\}/g, (all, k) => (k in t ? t[k] : all));
}

function homeBody(V, r, path) {
  const m = V.meta;
  const toVer = "../".repeat(path.split("/").length - 1);
  const stats = [
    [`${m.coverage.coverage}%`, `of the NumPy ${m.numpy} API (${m.coverage.implemented} of ${m.coverage.total} names)`],
    [String(m.documented), "documented entries with tested examples"],
    [String(V.ufuncs.length), "universal functions"],
    ["C++20", "native core via Node-API"],
  ];
  return `<section class="hero">
  <div class="hero-text">
    <p class="eyebrow">v${esc(m.version)} · released ${esc(m.date)}</p>
    <h1>NumPy for JavaScript, <span>running natively.</span></h1>
    <p class="lede">numera follows NumPy's API and behaviour: n-dimensional arrays, broadcasting, ufuncs, linear algebra, FFT and random numbers. The numerical work runs in a C++ core, and the API is fully typed for TypeScript.</p>
    <div class="cta">
      <a class="btn primary" href="${toVer}guides/quickstart.html">Get started</a>
      <a class="btn" href="${toVer}reference/index.html">Browse the API</a>
    </div>
    <div class="install-tabs" data-tabs>
      <div class="tabbar" role="tablist">${["npm", "pnpm", "yarn"].map((x, i) => `<button role="tab" type="button" aria-selected="${i === 0}">${x}</button>`).join("")}</div>
      ${[`npm install ${m.package}`, `pnpm add ${m.package}`, `yarn add ${m.package}`].map((c, i) => `<div class="codeblock tabpanel"${i ? " hidden" : ""}><div class="codehead"><span>Terminal</span><button class="copy" type="button" aria-label="Copy code">Copy</button></div><pre><code>${esc(c)}</code></pre></div>`).join("")}
    </div>
  </div>
  <div class="hero-code">
    <div class="duo">
      <div class="codeblock"><div class="codehead"><span>Python · NumPy</span></div><pre><code>${highlight(`import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])
a.sum(axis=0)          # -> array([5, 7, 9])
a[:, ::2]              # -> [[1, 3], [4, 6]]
np.linalg.solve([[3, 1], [1, 2]], [9, 8])
                       # -> array([2., 3.])`, "python")}</code></pre></div>
      <div class="codeblock accent"><div class="codehead"><span>TypeScript · numera</span><button class="copy" type="button" aria-label="Copy code">Copy</button></div><pre><code>${highlight(`import np from "${m.package}";

const a = np.array([[1, 2, 3], [4, 5, 6]]);
a.sum({ axis: 0 });    // => [5, 7, 9]
a.slice([null, [null, null, 2]]); // => [[1, 3], [4, 6]]
np.linalg.solve([[3, 1], [1, 2]], [9, 8]);
                       // => [2, 3]`)}</code></pre></div>
    </div>
  </div>
</section>
<section class="stats" aria-label="At a glance">${stats.map(([n, l]) => `<div><strong>${esc(n)}</strong><span>${esc(l)}</span></div>`).join("")}</section>
<article class="prose home-prose">${r.html}</article>`;
}

function refIndexBody(V, path) {
  const toVer = "../";
  const sections = [];
  for (const [title, ids] of GROUPS) {
    const cards = [];
    for (const id of ids) {
      if (id === "@ufuncs") {
        cards.push(`<a class="card" href="ufuncs.html"><h3>Universal functions (ufunc)</h3><p>How element-wise functions broadcast, promote types and take <code>out</code>/<code>where</code>; the full ufunc list.</p><span class="count">${V.ufuncs.length} ufuncs</span></a>`);
        continue;
      }
      const c = V.categories.find((x) => x.id === id);
      if (!c) continue;
      cards.push(`<a class="card" href="${c.id}.html"><h3>${esc(c.title)}</h3><p>${inline(firstSentence(c.intro) || c.entries.slice(0, 6).map((e) => "`" + displayName(e).split(" ")[0] + "`").join(", "))}</p><span class="count">${c.entries.length} entries</span></a>`);
    }
    if (cards.length) sections.push(`<section><h2 id="${slugify(title)}">${esc(title)}</h2><div class="cards">${cards.join("")}</div></section>`);
  }
  const m = V.meta;
  return `<article class="prose wide-prose">
<p class="eyebrow">API reference · v${esc(m.version)}</p>
<h1>API reference</h1>
<p class="lede">This reference describes every function, class and option in numera ${esc(m.version)}, grouped like the <a href="https://numpy.org/doc/stable/reference/" rel="noopener" target="_blank">NumPy reference</a>. Each entry gives the signature, its arguments, the return type and an example. Every example is executed by the test suite, and a <code>// =&gt;</code> comment shows the result: for an <code>NDArray</code>, that is <code>.toArray()</code>.</p>
<aside class="callout tip"><strong>Looking for a NumPy name?</strong><p>The <a href="numpy-index.html">NumPy name index</a> lists all ${m.coverage.total} NumPy ${esc(m.numpy)} names that numera tracks. For each one it gives the JavaScript name, its TypeScript signature and a link to the matching NumPy page.</p></aside>
<div class="conv">
  <div><h3>Naming</h3><p>NumPy names become camelCase: <code>expand_dims</code> → <code>expandDims</code>, <code>linalg.matrix_rank</code> → <code>linalg.matrixRank</code>.</p></div>
  <div><h3>Keyword arguments</h3><p>Python keyword arguments become an options object: <code>np.sum(a, axis=0)</code> → <code>np.sum(a, { axis: 0 })</code>.</p></div>
  <div><h3>Indexing</h3><p>JavaScript has no <code>a[1:3]</code> syntax, so indexing uses <code>a.get(...)</code>, <code>a.slice(...)</code> and <code>a.set(...)</code>.</p></div>
</div>
${sections.join("\n")}
</article>`;
}

function firstSentence(s) { return s ? (s.match(/^.*?[.!?](\s|$)/)?.[0] ?? s).trim() : ""; }

function categoryBody(V, cat, path) {
  const toc = [];
  const items = cat.entries.map((e) => {
    const id = anchor(e.name);
    const loc = V.entryLoc.get(e.name);
    toc.push({ id, text: displayName(e).split(" ")[0], level: 3 });
    const np = loc.covers
      .map((c) => ({ c, d: V.docFor(c.surface, c.numpy) }))
      .filter((x) => x.d.url);
    const npLinks = np.length
      ? `<div class="np-links"><span>NumPy</span>${np.slice(0, 6).map((x) => `<a href="${esc(x.d.url)}" rel="noopener" target="_blank" title="${esc(x.d.summary)}">${esc(SURFACE_TITLE[x.c.surface])}.${esc(x.c.numpy)}</a>`).join("")}</div>`
      : "";
    const tsSigs = loc.covers.flatMap((c) => (c.sig ?? []).map((s) => s)).filter((s, i, a) => a.indexOf(s) === i);
    const args = e.args?.length
      ? `<h4>Parameters</h4><dl class="params">${e.args.map((a) => `<div><dt><code>${esc(a.name)}</code>${a.type ? `<span class="ty">${esc(a.type)}</span>` : ""}</dt><dd>${inline(a.desc)}</dd></div>`).join("")}</dl>`
      : "";
    return `<article class="api" id="${id}">
<header class="api-head"><h3><code>${esc(displayName(e))}</code></h3><a class="anchor" href="#${id}" aria-label="Link to ${esc(e.name)}">#</a></header>
<pre class="sig"><code>${esc(e.sig)}</code></pre>
<p>${inline(e.desc)}</p>
${npLinks}
${args}
<h4>Returns</h4><p class="ret"><code>${esc(e.returns)}</code></p>
<h4>Example</h4><div class="codeblock"><div class="codehead"><span>TypeScript</span><button class="copy" type="button" aria-label="Copy code">Copy</button></div><pre><code>${highlight(e.example)}</code></pre></div>
${tsSigs.length ? `<details class="ts"><summary>TypeScript declaration</summary><pre><code>${tsSigs.map((s) => highlight(s)).join("\n")}</code></pre></details>` : ""}
</article>`;
  });
  return {
    toc,
    body: `<article class="prose ref">
<p class="eyebrow">API reference</p>
<h1>${esc(cat.title)}</h1>
${cat.intro ? `<p class="lede">${inline(cat.intro)}</p>` : ""}
<div class="summary-table table"><table><thead><tr><th>Name</th><th>Summary</th></tr></thead><tbody>${cat.entries
      .map((e) => `<tr><td><a href="#${anchor(e.name)}"><code>${esc(displayName(e).split(" ")[0])}</code></a></td><td>${inline(firstSentence(e.desc))}</td></tr>`)
      .join("")}</tbody></table></div>
${items.join("\n")}
</article>`,
  };
}

function ufuncBody(V, path, link) {
  const md = readFileSync(join(here, "versions", V.version, "guides/ufuncs.md"), "utf8");
  const r = renderMarkdown(fillTokens(md, V), { link });
  const byFamily = UFUNC_FAMILIES.map(([title]) => ({ title, items: [] }));
  const other = { title: "Other", items: [] };
  for (const u of V.ufuncs) {
    const i = UFUNC_FAMILIES.findIndex(([, re]) => re.test(u.numpy));
    (i >= 0 ? byFamily[i] : other).items.push(u);
  }
  const fam = [...byFamily, other].filter((f) => f.items.length);
  const toc = [...r.headings.filter((h) => h.level <= 2), { id: "available-ufuncs", text: "Available ufuncs", level: 2 }, ...fam.map((f) => ({ id: slugify(f.title), text: f.title, level: 3 }))];
  const tables = fam.map((f) => `<h3 id="${slugify(f.title)}">${esc(f.title)}<a class="anchor" href="#${slugify(f.title)}" aria-label="Link to this section">#</a></h3>
<div class="table"><table class="ufunc-table"><thead><tr><th>numera</th><th>NumPy</th><th>Kind</th><th>Description</th></tr></thead><tbody>${f.items
    .map((u) => {
      const d = V.docFor("np", u.numpy);
      const docHref = u.docs?.[0] ? "../" + u.docs[0] : `numpy-index.html#np-${u.numpy}`;
      const kind = u.ufunc === "binary" ? "binary · reduce, accumulate, reduceat, outer, at" : u.ufunc === "unary" ? "unary · at" : (u.sig?.[0]?.match(/\(([^)]*)\)/)?.[1].split(",").length ?? 1) > 1 ? "binary" : "unary";
      return `<tr><td><a href="${esc(docHref)}"><code>np.${esc(u.js)}</code></a></td><td>${d.url ? `<a href="${esc(d.url)}" rel="noopener" target="_blank"><code>${esc(u.numpy)}</code></a>` : `<code>${esc(u.numpy)}</code>`}</td><td class="kind">${esc(kind)}</td><td>${inline(d.summary.replace(/``/g, "`"))}</td></tr>`;
    })
    .join("")}</tbody></table></div>`);
  return {
    toc,
    body: `<article class="prose ref">
<p class="eyebrow">API reference</p>
<h1>${inline(r.title)}</h1>
${r.html}
<h2 id="available-ufuncs">Available ufuncs<a class="anchor" href="#available-ufuncs" aria-label="Link to this section">#</a></h2>
<p>numera ${esc(V.version)} implements ${V.ufuncs.length} of NumPy's ufuncs. The descriptions are NumPy's one-line summaries, and each NumPy name links to its page on numpy.org.</p>
${tables.join("\n")}
</article>`,
  };
}

function numpyIndexBody(V, path) {
  const toc = [];
  const sections = [];
  let implementedTotal = 0;
  let missingTotal = 0;
  for (const [s, info] of Object.entries(V.names.surfaces)) {
    const rows = [
      ...info.implemented.map((i) => ({ ...i, ok: true })),
      ...info.missing.map((m) => ({ numpy: m.numpy, kind: m.kind, ok: false })),
    ].sort((a, b) => a.numpy.localeCompare(b.numpy));
    if (!rows.length) continue;
    implementedTotal += info.implemented.length;
    missingTotal += info.missing.length;
    const id = `s-${s}`;
    toc.push({ id, text: SURFACE_TITLE[s], level: 2 });
    sections.push(`<section class="np-surface"><h2 id="${id}">${esc(SURFACE_TITLE[s])}<a class="anchor" href="#${id}" aria-label="Link to this section">#</a> <small>${info.implemented.length} / ${rows.length}</small></h2>
<div class="table"><table class="np-table"><thead><tr><th>NumPy</th><th>numera</th><th>Summary</th></tr></thead><tbody>${rows
      .map((r) => {
        const d = V.docFor(s, r.numpy);
        const npCell = d.url ? `<a href="${esc(d.url)}" rel="noopener" target="_blank"><code>${esc(r.numpy)}</code></a>` : `<code>${esc(r.numpy)}</code>`;
        if (!r.ok) return `<tr id="${esc(s)}-${esc(r.numpy)}" class="missing"><td>${npCell}<span class="kind">${esc(r.kind)}</span></td><td><span class="pill no">not implemented</span></td><td>${inline(d.summary.replace(/``/g, "`"))}</td></tr>`;
        const jsName = (SURFACE_PREFIX[s] ?? "") + r.js;
        const docLink = r.docs?.[0] ? `<a class="pill yes" href="../${esc(r.docs[0])}">documented</a>` : "";
        const sig = r.sig?.length ? `<details><summary>signature</summary><pre><code>${r.sig.map((x) => highlight(x)).join("\n")}</code></pre></details>` : "";
        return `<tr id="${esc(s)}-${esc(r.numpy)}"><td>${npCell}<span class="kind">${esc(r.kind)}</span></td><td><code>${esc(jsName)}</code> ${docLink}${sig}</td><td>${inline(d.summary.replace(/``/g, "`"))}</td></tr>`;
      })
      .join("")}</tbody></table></div></section>`);
  }
  const m = V.meta;
  return {
    toc,
    body: `<article class="prose ref wide-prose">
<p class="eyebrow">API reference</p>
<h1>NumPy name index</h1>
<p class="lede">Every public NumPy ${esc(m.numpy)} name that numera tracks, with its JavaScript equivalent in v${esc(m.version)}. ${implementedTotal} are implemented and ${missingTotal} are not yet. Names marked <span class="pill yes">documented</span> have a full reference entry with an example. For the others, open <em>signature</em> to see the TypeScript declaration, and follow the NumPy link for the semantics, which numera mirrors.</p>
<p>${m.excluded ? `${m.excluded} NumPy names are excluded on purpose because they have no meaning in JavaScript, such as <code>test</code>, Python-only helpers and build flags, so they are not listed here. ` : ""}The JavaScript name is the NumPy name in camelCase, and Python keyword arguments become an options object. The <a href="../compatibility.html">compatibility page</a> lists every documented difference from NumPy.</p>
<div class="filter"><input type="search" id="np-filter" placeholder="Filter names…" aria-label="Filter NumPy names" autocomplete="off"><label><input type="checkbox" id="np-missing"> only not implemented</label></div>
${sections.join("\n")}
</article>`,
  };
}

function redirectPage(to) {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>numera documentation</title>
<meta http-equiv="refresh" content="0; url=${to}">
<link rel="canonical" href="${to}">
<script>location.replace(${JSON.stringify(to)} + location.hash)</script>
</head><body><p>Redirecting to <a href="${to}">the latest numera documentation</a>…</p></body></html>
`;
}

function notFound() {
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Page not found · numera</title>
<style>body{font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;display:grid;place-items:center;min-height:100vh;margin:0;background:#fbfaf7;color:#1d1f23}a{color:#0b6e4f}svg{width:48px;height:48px;fill:#c9c3b3}svg .hot{fill:#0b6e4f}main{text-align:center;padding:24px}</style></head>
<body><main>${logo}<h1>Page not found</h1><p>This page does not exist in this version of the docs.</p><p><a id="home" href="/latest.html">Go to the latest documentation</a></p></main>
<script>{const m=location.pathname.match(/^(.*?\/)v\d+\.\d+\.\d+[^/]*\//);if(m)document.getElementById("home").href=m[1]+"latest.html";}</script></body></html>
`;
}
