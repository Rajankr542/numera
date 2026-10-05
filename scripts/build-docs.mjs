// Renders the API reference (docs/site/api.mjs) into one self-contained HTML
// page at docs-dist/index.html (DECISIONS D-030, D-240). Deploy that directory
// to https://numera.cyfora.in; it is not shipped in the npm package.
// Usage: node scripts/build-docs.mjs [outDir]
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { categories } from "../docs/site/api.mjs";
import { css } from "./docs-style.mjs";
import { DOCS_URL, REPO_URL } from "./stage-package.mjs";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const pkgDir = join(root, "packages", "numera");

export const esc = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** Escapes text and turns `code` spans and **bold** into HTML. */
export const inline = (s) =>
  esc(s)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

/** Stable anchor id, e.g. "linalg.svd" -> "linalg-svd". */
export const slug = (name) => name.replace(/[^A-Za-z0-9]+/g, "-").replace(/^-|-$/g, "");

const KW = new Set(["const", "let", "new", "try", "catch", "return", "typeof", "true", "false", "null", "import", "from"]);

/** Tiny tokenizer-based JS highlighter; `// =>` comments render as results. */
export function highlight(code) {
  const re = /(\/\/[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(\d+(?:\.\d+)?(?:e-?\d+)?)\b|([A-Za-z_$][\w$]*)/g;
  let out = "";
  let last = 0;
  for (let m; (m = re.exec(code)); ) {
    out += esc(code.slice(last, m.index));
    last = re.lastIndex;
    const [tok, com, str, num, word] = m;
    if (com) out += `<span class="${com.startsWith("// =>") ? "t-res" : "t-com"}">${esc(com)}</span>`;
    else if (str) out += `<span class="t-str">${esc(str)}</span>`;
    else if (num) out += `<span class="t-num">${num}</span>`;
    else if (word && KW.has(word)) out += `<span class="t-kw">${word}</span>`;
    else out += esc(tok);
  }
  return out + esc(code.slice(last));
}

function renderEntry(e) {
  const id = slug(e.name);
  const args = e.args?.length
    ? `<h4>Arguments</h4><ol class="args">${e.args
        .map((a) => `<li><code>${esc(a.name)}</code>${a.type ? ` <span class="ty">(${esc(a.type)})</span>` : ""}: ${inline(a.desc)}</li>`)
        .join("")}</ol>`
    : "";
  return `<article class="fn" id="${id}">
<h3>${esc(e.sig)}<a class="anchor" href="#${id}" aria-label="Link to ${esc(e.name)}">#</a></h3>
<p>${inline(e.desc)}</p>
${args}
<h4>Returns</h4><p class="ret">(${esc(e.returns)})</p>
<h4>Example</h4><pre><code>${highlight(e.example)}</code></pre>
</article>`;
}

function renderToc(cats) {
  return cats
    .map(
      (c) => `<div class="toc-cat" data-cat="${c.id}"><h3><a href="#${c.id}">${esc(c.title)}</a></h3><ul>${c.entries
        .map((e) => `<li><a href="#${slug(e.name)}" data-name="${esc(e.name.toLowerCase())} ${esc(e.sig.toLowerCase())}">${esc(e.name)}</a></li>`)
        .join("")}</ul></div>`,
    )
    .join("\n");
}


// Client-side: filter the sidebar, highlight the section in view, mobile menu.
const script = String.raw`
const q = document.getElementById("q"), toc = document.querySelector(".toc");
const links = [...toc.querySelectorAll("li a")];
q.addEventListener("input", () => {
  const t = q.value.trim().toLowerCase();
  let hits = 0;
  for (const a of links) { const ok = !t || a.dataset.name.includes(t); a.parentElement.classList.toggle("hidden", !ok); hits += ok; }
  for (const c of toc.querySelectorAll(".toc-cat")) c.classList.toggle("hidden", !c.querySelector("li:not(.hidden)"));
  toc.classList.toggle("no-hits", hits === 0);
});
q.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { const a = links.find((l) => !l.parentElement.classList.contains("hidden")); if (a) location.hash = a.getAttribute("href"); }
  if (e.key === "Escape") { q.value = ""; q.dispatchEvent(new Event("input")); }
});
document.addEventListener("keydown", (e) => { if (e.key === "/" && document.activeElement !== q) { e.preventDefault(); q.focus(); } });
const byId = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
const io = new IntersectionObserver((es) => {
  for (const en of es) if (en.isIntersecting) {
    for (const a of links) a.classList.remove("active");
    const a = byId.get(en.target.id); if (a) { a.classList.add("active"); a.scrollIntoView({ block: "nearest" }); }
  }
}, { rootMargin: "0px 0px -75% 0px" });
document.querySelectorAll("article.fn").forEach((el) => io.observe(el));
document.querySelector(".menu").addEventListener("click", () => document.body.classList.toggle("open"));
toc.addEventListener("click", (e) => { if (e.target.closest("a")) document.body.classList.remove("open"); });
`;

/** Full HTML page for the given package metadata. */
export function renderPage({ name, version, description }, cats = categories) {
  const count = cats.reduce((n, c) => n + c.entries.length, 0);
  const body = cats
    .map(
      (c) => `<section class="cat" id="${c.id}"><h2>${esc(c.title)}</h2>${c.intro ? `<p class="intro">${inline(c.intro)}</p>` : ""}
${c.entries.map(renderEntry).join("\n")}</section>`,
    )
    .join("\n");
  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(name)} ${esc(version)} — API reference</title>
<meta name="description" content="${esc(description)}">
<link rel="canonical" href="${DOCS_URL}/">
<style>${css}</style>
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-H8X56EQGVV"></script>
<script>window.dataLayer=window.dataLayer||[];function gtag(){dataLayer.push(arguments)}gtag('js',new Date());gtag('config','G-H8X56EQGVV');</script>
</head>
<body>
<button class="menu" type="button">☰ Menu</button>
<nav class="side" aria-label="API">
<div class="brand"><a href="#top">numera</a><span class="ver">v${esc(version)}</span><p>NumPy for JavaScript</p><p class="links"><a href="${REPO_URL}">GitHub</a> · <a href="https://www.npmjs.com/package/${esc(name)}">npm</a></p></div>
<div class="search"><input id="q" type="search" placeholder="Search ${count} APIs  ( / )" autocomplete="off" aria-label="Search the API"></div>
<div class="toc">${renderToc(cats)}<p class="empty">No matches.</p></div>
</nav>
<main id="top">
<header class="hero">
<h1>${esc(name)} <small class="ver">v${esc(version)}</small></h1>
<p>${inline(description)}</p>
<pre class="install"><code>npm install ${esc(name)}</code></pre>
<pre><code>${highlight(`import np from "${name}";\n\nconst a = np.array([[1, 2, 3], [4, 5, 6]]);\na.sum({ axis: 0 }).toArray(); // => [5, 7, 9]`)}</code></pre>
<p>The API follows NumPy's names and semantics, with camelCase names in JS (<code>expand_dims</code> → <code>expandDims</code>). Keyword arguments become an options object. Examples assume <code>import np from "${esc(name)}"</code>. A <code>// =&gt;</code> comment shows the result; for an <code>NDArray</code>, it shows <code>.toArray()</code>.</p>
</header>
${body}
<footer>${esc(name)} v${esc(version)} · MIT License · <a href="${REPO_URL}">Source on GitHub</a> · <a href="${REPO_URL}/issues">Report an issue</a> · Generated from the package's tested examples.</footer>
</main>
<script>${script}</script>
</body>
</html>
`;
}

/** Writes <outDir>/index.html (default docs-dist/) for the current package.json. */
export function buildDocs(outDir = join(root, "docs-dist")) {
  const pkg = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8"));
  mkdirSync(outDir, { recursive: true });
  const file = join(outDir, "index.html");
  writeFileSync(file, renderPage(pkg));
  return file;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  console.log(`Wrote ${buildDocs(process.argv[2] ? resolve(process.argv[2]) : undefined)}`);
}
