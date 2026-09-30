// Inline stylesheet for the generated API reference (DECISIONS D-030).
// Kept separate from build-docs.mjs so both files stay short. Self-contained:
// system font stacks only, no web fonts or external requests.
export const css = String.raw`
:root {
  --bg: #fbfaf7; --panel: #f3f1ea; --ink: #1d1f23; --muted: #626770; --line: #e2ded2;
  --accent: #0b6e4f; --accent-soft: #e2f1ea; --code-bg: #1e2127; --code-ink: #e6e6e6;
  --c-com: #7f8896; --c-res: #8fd694; --c-str: #f2c57c; --c-num: #d9a0ff; --c-kw: #7ec4ff;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
  --side: 290px;
}
@media (prefers-color-scheme: dark) {
  :root { --bg: #15171b; --panel: #1b1e23; --ink: #e4e4e0; --muted: #9aa0a8; --line: #2b2f36;
    --accent: #5fd3a4; --accent-soft: #1f3a30; --code-bg: #0f1114; }
}
* { box-sizing: border-box; }
html { scroll-padding-top: 16px; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 15px/1.6 var(--sans); }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code { font: 0.9em var(--mono); background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 0.05em 0.35em; }

.side { position: fixed; inset: 0 auto 0 0; width: var(--side); display: flex; flex-direction: column;
  background: var(--panel); border-right: 1px solid var(--line); }
.brand { padding: 20px 20px 12px; }
.brand a { color: var(--ink); font: 700 22px/1 var(--mono); letter-spacing: -0.02em; }
.brand .ver { margin-left: 6px; font: 12px var(--mono); color: var(--muted); }
.brand p { margin: 6px 0 0; color: var(--muted); font-size: 13px; }
.search { margin: 0 16px 10px; }
.search input { width: 100%; padding: 8px 10px; font: 14px var(--sans); color: var(--ink);
  background: var(--bg); border: 1px solid var(--line); border-radius: 6px; outline: none; }
.search input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
.toc { flex: 1; overflow-y: auto; padding: 0 8px 24px; }
.toc h3 { margin: 16px 12px 4px; font: 600 11px/1 var(--sans); letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.toc ul { list-style: none; margin: 0; padding: 0; }
.toc li a { display: block; padding: 2px 12px; border-radius: 4px; font: 13px/1.7 var(--mono); color: var(--ink); }
.toc li a:hover { background: var(--accent-soft); text-decoration: none; }
.toc li a.active { background: var(--accent-soft); color: var(--accent); }
.toc .empty { display: none; padding: 8px 12px; color: var(--muted); font-size: 13px; }
.toc.no-hits .empty { display: block; }
.hidden { display: none !important; }

main { margin-left: var(--side); max-width: 980px; padding: 32px 48px 120px; }
.hero h1 { margin: 0; font: 700 34px/1.2 var(--sans); letter-spacing: -0.02em; }
.hero p { color: var(--muted); max-width: 70ch; }
.install { display: inline-block; margin: 4px 0 8px; }
section.cat { margin-top: 56px; }
section.cat > h2 { margin: 0 0 6px; padding-bottom: 8px; font: 700 24px/1.3 var(--sans); border-bottom: 2px solid var(--ink); }
section.cat > .intro { color: var(--muted); max-width: 75ch; }
article.fn { padding: 22px 0 8px; border-bottom: 1px solid var(--line); }
article.fn h3 { margin: 0; font: 600 17px/1.4 var(--mono); word-break: break-word; }
article.fn h3 a.anchor { color: var(--muted); margin-left: 6px; font-size: 14px; visibility: hidden; }
article.fn:hover h3 a.anchor { visibility: visible; }
article.fn h4 { margin: 16px 0 6px; font: 600 12px/1 var(--sans); letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
article.fn p { margin: 8px 0; max-width: 75ch; }
ol.args { margin: 0; padding-left: 22px; }
ol.args li { margin: 3px 0; }
ol.args .ty { color: var(--accent); font: 13px var(--mono); }
.ret { font: 13px var(--mono); color: var(--accent); }
pre { margin: 6px 0 12px; padding: 14px 16px; overflow-x: auto; background: var(--code-bg); color: var(--code-ink);
  border-radius: 8px; font: 13px/1.6 var(--mono); }
pre code { background: none; border: 0; padding: 0; font: inherit; color: inherit; }
.t-com { color: var(--c-com); } .t-res { color: var(--c-res); } .t-str { color: var(--c-str); }
.t-num { color: var(--c-num); } .t-kw { color: var(--c-kw); }
footer { margin-top: 64px; color: var(--muted); font-size: 13px; }

.menu { display: none; }
@media (max-width: 860px) {
  .side { transform: translateX(-100%); transition: transform 0.2s; z-index: 10; }
  body.open .side { transform: none; box-shadow: 0 0 40px rgba(0,0,0,0.25); }
  main { margin-left: 0; padding: 64px 20px 80px; }
  .menu { display: block; position: fixed; top: 12px; left: 12px; z-index: 11; padding: 6px 12px;
    font: 600 13px var(--sans); color: var(--ink); background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }
}
`;
