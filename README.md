# numera documentation site (`api-doc` branch)

This branch contains only the documentation website for [`@cyfora/numera`](https://www.npmjs.com/package/@cyfora/numera) and the scripts that generate it. The library itself lives on `main`.

The built site is in **`html/`**. It is plain static HTML, CSS and JS with relative links and no external requests, so you can host it on any static host (GitHub Pages, Netlify, Cloudflare Pages, S3, nginx, …) at the domain root or in a sub-folder.

```
html/
  index.html, latest.html   → redirect to the latest version
  versions.json             → version list read by the version picker
  assets/                   → style.css, app.js, logo.png, favicons
  404.html, robots.txt, .nojekyll
  v1.0.3/  v1.0.2/  v1.0.1/  → one folder per release
    index.html              → introduction
    guides/*.html           → guides
    reference/*.html        → API reference (one page per section), ufuncs, NumPy name index
    compatibility.html, changelog.html, search-index.js
```

## Preview locally

```bash
npm run build
npm run serve          # http://localhost:8000
```

Opening `html/index.html` straight from disk also works. Over `file://` the version picker uses the versions that were built into each page.

## Layout of the sources

| Path | What it is |
| --- | --- |
| `versions/<ver>/` | data captured from one release: `api.mjs` + `parts/` (reference entries), `COMPATIBILITY.md`, `names.json` (NumPy name → JS name, kind, TypeScript signatures), `meta.json` |
| `versions/<ver>/guides/` | hand-written guides (`nav.json` sets the order; `ufuncs.md` is the text of the ufunc page) |
| `site/changelog.md` | the changelog (shared by all versions) |
| `site/assets/` | front-end CSS/JS |
| `data/numpy-2.5.3.json` | NumPy one-line summaries and numpy.org links (from `scripts/numpy_data.py`) |
| `scripts/build.mjs` | builds `html/` from all of the above |

Guides are Markdown with a few extras: `{{version}}`, `{{coverage}}`, `{{implemented}}`, `{{total}}`, `{{ufuncs}}`, `{{numpy}}`, `{{date}}`, `{{package}}` and `{{node}}` tokens, `> **Note**:` callouts, links like `/guides/x.html` (resolved inside the version), and `ref:<name>` links to API entries. The build fails on a broken internal link.

## Adding a new release (e.g. 1.0.4)

1. Build or install that release of the package:
   - from a checkout of `main` at the release commit: `pnpm build` (gives `packages/numera/dist`), or
   - from npm: `npm i @cyfora/numera@1.0.4` in a temp folder.
2. Capture its data (the main checkout needs its `node_modules`, for the TypeScript compiler):
   ```bash
   node scripts/snapshot.mjs --version 1.0.4 --repo ../numera --ref v1.0.4 \
        --package ../numera/packages/numera --date 2026-10-20
   ```
3. Add guides: `cp -r versions/1.0.3/guides versions/1.0.4/` and update them for what changed.
4. Add a `## 1.0.4` section at the top of `site/changelog.md`.
5. Check that every guide example still holds against the real package. Every `expr; // => json` line is executed:
   ```bash
   node scripts/check-examples.mjs --version 1.0.4 --package ../numera/packages/numera
   ```
6. `npm run build && npm run check:links`, then commit `versions/1.0.4`, `site/` and `html/`.

The newest version (by semver) becomes "latest" automatically. The picker on already-deployed pages reads `versions.json` from the site root (over http), so older pages list the new release too.

## Verification

- `scripts/check-examples.mjs` runs every guide example against the package of its own version. Currently v1.0.1 checks 203 results, and v1.0.2 and v1.0.3 check 302 each, with 0 failures.
- The API reference examples come from `docs/site/` on `main` at the release ref, where the test suite (`docs_site.test.ts`) executes them.
- `scripts/check-links.mjs` checks every internal link and `#anchor`.

NumPy summaries in the reference and name index are taken from the NumPy documentation (BSD-3-Clause).
