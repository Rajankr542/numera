// Shared page chrome: header, sidebar, right-hand TOC, footer.
import { esc } from "./highlight.mjs";

export const SITE = {
  name: "numera",
  pkg: "@cyfora/numera",
  author: { name: "Rajan Kumar", url: "https://github.com/Rajankr542" },
  company: { name: "Cyfora", url: "https://cyfora.in" },
  npm: "https://www.npmjs.com/package/@cyfora/numera",
};

export const logo = (src) => `<img class="logo-mark" src="${src}" width="24" height="24" alt="Cyfora">`;

const icon = {
  search: `<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>`,
  sun: `<svg viewBox="0 0 24 24" aria-hidden="true" class="i-sun"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>`,
  moon: `<svg viewBox="0 0 24 24" aria-hidden="true" class="i-moon"><path d="M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5Z"/></svg>`,
  menu: `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg>`,
};

/**
 * @param {object} p
 * @param {string} p.title          page title
 * @param {string} p.description    meta description
 * @param {string} p.body           main HTML
 * @param {string} p.toRoot         relative path from this page to the site root ("../../")
 * @param {string} p.toVer          relative path from this page to the version root ("../")
 * @param {string} p.path           page path inside the version ("guides/quickstart.html")
 * @param {string} p.tab            active top tab: guides | reference | compat | changelog
 * @param {string} p.sidebar        sidebar HTML
 * @param {Array<{id,text,level}>} [p.toc]
 * @param {object} p.ver            { version, latest, versions: [...] }
 * @param {string} [p.bodyClass]
 */
export function page(p) {
  const { ver } = p;
  const v = (path) => p.toVer + path;
  const tabs = [
    ["guides", "Guides", "index.html"],
    ["reference", "API Reference", "reference/index.html"],
    ["compat", "NumPy compatibility", "compatibility.html"],
    ["changelog", "Changelog", "changelog.html"],
  ];
  const versionOptions = ver.versions
    .map((x) => `<option value="${esc(x)}"${x === ver.version ? " selected" : ""}>v${esc(x)}${x === ver.latest ? " (latest)" : ""}</option>`)
    .join("");
  const toc = p.toc?.length
    ? `<aside class="otp" aria-label="On this page"><p class="otp-title">On this page</p><ul>${p.toc
        .map((h) => `<li class="l${h.level}"><a href="#${esc(h.id)}">${esc(h.text)}</a></li>`)
        .join("")}</ul></aside>`
    : "";
  const outdated = ver.version !== ver.latest
    ? `<div class="banner">You are reading the docs for <strong>v${esc(ver.version)}</strong>. The latest release is <a href="${p.toRoot}v${esc(ver.latest)}/index.html" data-switch="${esc(ver.latest)}">v${esc(ver.latest)}</a>.</div>`
    : "";
  return `<!doctype html>
<html lang="en" data-root="${esc(p.toRoot)}" data-ver-root="${esc(p.toVer)}" data-version="${esc(ver.version)}" data-path="${esc(p.path)}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(p.title)} · numera v${esc(ver.version)}</title>
<meta name="description" content="${esc(p.description || "")}">
<meta property="og:title" content="${esc(p.title)} · numera">
<meta property="og:description" content="${esc(p.description || "")}">
<meta name="theme-color" content="#0b6e4f">
<link rel="icon" href="${p.toRoot}assets/favicon.ico" sizes="any">
<link rel="icon" href="${p.toRoot}assets/favicon.png" type="image/png">
<link rel="apple-touch-icon" href="${p.toRoot}assets/apple-touch-icon.png">
<link rel="stylesheet" href="${p.toRoot}assets/style.css">
<script>try{var t=localStorage.getItem("numera-theme");if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
</head>
<body class="${esc(p.bodyClass || "")}">
<a class="skip" href="#content">Skip to content</a>
<header class="top">
  <div class="top-in">
    <button class="icon-btn menu" type="button" aria-label="Open navigation" aria-expanded="false">${icon.menu}</button>
    <a class="brand" href="${v("index.html")}">${logo(p.toRoot + "assets/logo.png")}<span>numera</span></a>
    <label class="ver-pick"><span class="sr">Documentation version</span><select id="ver" aria-label="Documentation version">${versionOptions}</select></label>
    <nav class="tabs" aria-label="Sections">${tabs
      .map(([id, label, href]) => `<a href="${v(href)}"${p.tab === id ? ' aria-current="page"' : ""}>${label}</a>`)
      .join("")}</nav>
    <div class="top-actions">
      <button class="search-btn" type="button" data-open-search>${icon.search}<span>Search</span><kbd>/</kbd></button>
      <button class="icon-btn theme" type="button" aria-label="Toggle dark mode">${icon.sun}${icon.moon}</button>
      <a class="npm" href="${SITE.npm}" rel="noopener" target="_blank">npm</a>
    </div>
  </div>
</header>
${outdated}
<div class="shell">
  <nav class="side" aria-label="Documentation">${p.sidebar}</nav>
  <main id="content" class="content${toc ? "" : " wide"}">
${p.body}
${pager(p)}
  </main>
  ${toc}
</div>
${footer(p)}
<div class="palette" hidden role="dialog" aria-modal="true" aria-label="Search the documentation">
  <div class="palette-box">
    <div class="palette-q">${icon.search}<input type="search" placeholder="Search functions, guides and NumPy names…" autocomplete="off" spellcheck="false" aria-label="Search"><kbd>Esc</kbd></div>
    <ul class="palette-res" role="listbox"></ul>
    <p class="palette-foot"><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>↵</kbd> open</span><span>v${esc(ver.version)}</span></p>
  </div>
</div>
<script>window.NUMERA_VERSIONS=${JSON.stringify({ latest: ver.latest, versions: ver.versions, pages: ver.pages })};</script>
<script src="${v("search-index.js")}" defer></script>
<script src="${p.toRoot}assets/app.js" defer></script>
</body>
</html>
`;
}

function pager(p) {
  if (!p.prev && !p.next) return "";
  const link = (x, dir) =>
    x ? `<a class="pg ${dir}" href="${esc(x.href)}"><small>${dir === "prev" ? "Previous" : "Next"}</small><span>${esc(x.title)}</span></a>` : "<span></span>";
  return `<nav class="pager" aria-label="Pagination">${link(p.prev, "prev")}${link(p.next, "next")}</nav>`;
}

function footer(p) {
  const year = new Date().getFullYear();
  return `<footer class="foot">
  <div class="foot-in">
    <div class="foot-brand">
      <a class="brand" href="${p.toVer}index.html">${logo(p.toRoot + "assets/logo.png")}<span>numera</span></a>
      <p>NumPy for JavaScript and TypeScript, backed by a native C++ core.</p>
    </div>
    <div class="foot-cols">
      <div><h2>Docs</h2><ul>
        <li><a href="${p.toVer}guides/quickstart.html">Quickstart</a></li>
        <li><a href="${p.toVer}reference/index.html">API reference</a></li>
        <li><a href="${p.toVer}reference/numpy-index.html">NumPy name index</a></li>
        <li><a href="${p.toVer}changelog.html">Changelog</a></li>
      </ul></div>
      <div><h2>Package</h2><ul>
        <li><a href="${SITE.npm}" rel="noopener" target="_blank">npm: ${SITE.pkg}</a></li>
        <li><a href="${p.toVer}compatibility.html">NumPy compatibility</a></li>
        <li><span>MIT License</span></li>
      </ul></div>
      <div><h2>Made by</h2><ul>
        <li><a href="${SITE.author.url}" rel="noopener author" target="_blank">${SITE.author.name} · @Rajankr542</a></li>
        <li><a href="${SITE.company.url}" rel="noopener" target="_blank">${SITE.company.name} · cyfora.in</a></li>
      </ul></div>
    </div>
  </div>
  <div class="foot-legal">
    <p>© ${year} <a href="${SITE.author.url}" rel="noopener" target="_blank">${SITE.author.name}</a> and <a href="${SITE.company.url}" rel="noopener" target="_blank">Cyfora</a>. numera is released under the MIT License.
    NumPy names, one-line summaries and links come from the <a href="https://numpy.org/doc/stable/" rel="noopener" target="_blank">NumPy documentation</a> (BSD-3-Clause). numera is not affiliated with the NumPy project.</p>
  </div>
</footer>`;
}

/** Sidebar from groups: [{ title, items: [{ title, href, active, badge }] }] */
export function sidebar(groups) {
  return groups
    .map(
      (g) => `<div class="side-group"><p class="side-title">${esc(g.title)}</p><ul>${g.items
        .map(
          (i) =>
            `<li><a href="${esc(i.href)}"${i.active ? ' aria-current="page"' : ""}>${esc(i.title)}${i.badge != null ? `<span class="badge">${esc(i.badge)}</span>` : ""}</a></li>`,
        )
        .join("")}</ul></div>`,
    )
    .join("");
}
