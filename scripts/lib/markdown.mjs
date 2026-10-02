// Minimal Markdown renderer for the guides (no dependencies).
// Supports: # headings (with ids), paragraphs, fenced code, ul/ol, pipe tables,
// > blockquotes (callouts when they start with **Note**/**Tip**/**Warning**),
// raw HTML blocks (lines starting with "<"), and inline `code`, **bold**,
// *italic* and [links](url).
import { highlight, esc } from "./highlight.mjs";

export const slugify = (s) =>
  s.toLowerCase().replace(/<[^>]+>/g, "").replace(/[`*]/g, "").replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");

/**
 * @param {string} s
 * @param {(href: string) => string} link  rewrites hrefs (version-root and ref: links)
 */
export function inline(s, link = (h) => h) {
  const codes = [];
  let out = s.replace(/`([^`]+)`/g, (_m, c) => {
    codes.push(`<code>${esc(c)}</code>`);
    return `\u0000${codes.length - 1}\u0000`;
  });
  out = esc(out)
    .replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_m, t, h) => {
      const href = link(h.replace(/&amp;/g, "&"));
      const ext = /^https?:/.test(href);
      return `<a href="${esc(href)}"${ext ? ' rel="noopener" target="_blank"' : ""}>${t}</a>`;
    })
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\s][^*]*)\*(?=[\s).,;:]|$)/g, "$1<em>$2</em>");
  return out.replace(/\u0000(\d+)\u0000/g, (_m, i) => codes[+i]);
}

/**
 * Renders markdown. Returns { html, headings: [{level, id, text}], title }.
 * The first `# ` heading becomes the page title (not rendered in html).
 */
export function renderMarkdown(md, { link } = {}) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  const headings = [];
  let title = "";
  let lead = "";
  const used = new Set();
  const uniq = (id) => {
    let x = id || "section";
    for (let i = 2; used.has(x); i++) x = `${id}-${i}`;
    used.add(x);
    return x;
  };
  let i = 0;
  const il = (s) => inline(s, link);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    const fence = line.match(/^```(\w*)\s*(.*)$/);
    if (fence) {
      const lang = fence[1] || "text";
      const label = fence[2].replace(/\bnorun\b/, "").trim();
      const body = [];
      for (i++; i < lines.length && !lines[i].startsWith("```"); i++) body.push(lines[i]);
      i++;
      const code = body.join("\n");
      const hl = lang === "js" || lang === "ts" ? highlight(code) : lang === "python" ? highlight(code, "python") : esc(code);
      out.push(
        `<div class="codeblock" data-lang="${esc(lang)}"><div class="codehead"><span>${esc(label || ({ js: "JavaScript", ts: "TypeScript", python: "Python (NumPy)", bash: "Terminal", text: "Output" }[lang] ?? lang))}</span><button class="copy" type="button" aria-label="Copy code">Copy</button></div><pre><code>${hl}</code></pre></div>`,
      );
      continue;
    }
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      const text = h[2].trim();
      if (level === 1 && !title) {
        title = text;
        i++;
        continue;
      }
      const id = uniq(slugify(text));
      headings.push({ level, id, text: inline(text).replace(/<[^>]+>/g, "") });
      out.push(`<h${level} id="${id}">${il(text)}<a class="anchor" href="#${id}" aria-label="Link to this section">#</a></h${level}>`);
      i++;
      continue;
    }
    if (line.startsWith("<")) {
      const body = [];
      while (i < lines.length && lines[i].trim()) body.push(lines[i++]);
      out.push(body.join("\n").replace(/href="\/([^"]*)"/g, (_m, p) => `href="${esc(link("/" + p))}"`));
      continue;
    }
    if (line.startsWith(">")) {
      const body = [];
      while (i < lines.length && lines[i].startsWith(">")) body.push(lines[i++].replace(/^>\s?/, ""));
      const text = body.join(" ");
      const kind = text.match(/^\*\*(Note|Tip|Warning|Important)\*\*:?\s*/);
      if (kind) {
        out.push(`<aside class="callout ${kind[1].toLowerCase()}"><strong>${kind[1]}</strong><p>${il(text.slice(kind[0].length))}</p></aside>`);
      } else out.push(`<blockquote><p>${il(text)}</p></blockquote>`);
      continue;
    }
    if (line.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++]);
      const cells = (r) => r.replace(/^\||\|\s*$/g, "").split(/(?<!\\)\|/).map((c) => c.trim().replace(/\\\|/g, "|"));
      const head = cells(rows[0]);
      const body = rows.slice(2).map(cells);
      out.push(
        `<div class="table"><table><thead><tr>${head.map((c) => `<th>${il(c)}</th>`).join("")}</tr></thead><tbody>${body
          .map((r) => `<tr>${r.map((c) => `<td>${il(c)}</td>`).join("")}</tr>`)
          .join("")}</tbody></table></div>`,
      );
      continue;
    }
    const li = line.match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
    if (li) {
      const ordered = /\d/.test(li[2]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*]|\d+\.)\s+(.*)$/);
        if (m) { items.push(m[3]); i++; continue; }
        if (lines[i].match(/^\s{2,}\S/) && items.length) { items[items.length - 1] += " " + lines[i].trim(); i++; continue; }
        break;
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>${items.map((t) => `<li>${il(t)}</li>`).join("")}</${tag}>`);
      continue;
    }
    const para = [];
    while (i < lines.length && lines[i].trim() && !/^(```|#{1,4}\s|>|\||<|\s*([-*]|\d+\.)\s)/.test(lines[i])) para.push(lines[i++]);
    const text = para.join(" ");
    if (!lead && !out.length) lead = text;
    out.push(`<p>${il(text)}</p>`);
  }
  return { html: out.join("\n"), headings, title, lead };
}

/** Extracts fenced code blocks: [{ lang, code, line }]. */
export function codeBlocks(md) {
  const blocks = [];
  const lines = md.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^```(\w*)/);
    if (!m) continue;
    const start = i + 1;
    const body = [];
    for (i++; i < lines.length && !lines[i].startsWith("```"); i++) body.push(lines[i]);
    blocks.push({ lang: m[1] || "text", code: body.join("\n"), line: start });
  }
  return blocks;
}
