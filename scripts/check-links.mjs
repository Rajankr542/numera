// Checks every relative href/src in html/ points at an existing file and, for
// "#fragment" links, at an existing id in that file.
import { readdirSync, readFileSync, statSync, existsSync } from "node:fs";
import { join, dirname, resolve, relative } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "../html");
const files = [];
const walk = (d) => { for (const f of readdirSync(d)) { const p = join(d, f); statSync(p).isDirectory() ? walk(p) : files.push(p); } };
walk(root);
const ids = new Map();
const idsOf = (f) => {
  if (!ids.has(f)) ids.set(f, new Set([...readFileSync(f, "utf8").matchAll(/\sid="([^"]+)"/g)].map((m) => m[1].replace(/&amp;/g, "&"))));
  return ids.get(f);
};
let links = 0;
const bad = [];
for (const f of files.filter((x) => x.endsWith(".html"))) {
  const html = readFileSync(f, "utf8");
  for (const m of html.matchAll(/\s(?:href|src)="([^"]*)"/g)) {
    const h = m[1].replace(/&amp;/g, "&");
    if (/^(https?:|mailto:|data:|javascript:)/.test(h) || (f.endsWith("404.html") && h.startsWith("/"))) continue;
    links++;
    const [path, frag] = h.split("#");
    const target = path ? resolve(dirname(f), path) : f;
    if (!target.startsWith(root) || !existsSync(target)) { bad.push(`${relative(root, f)}: missing ${h}`); continue; }
    if (frag && target.endsWith(".html") && !idsOf(target).has(decodeURIComponent(frag))) bad.push(`${relative(root, f)}: no #${frag} in ${relative(root, target)}`);
  }
}
console.log(`${files.length} files, ${links} internal links, ${bad.length} broken`);
for (const b of bad.slice(0, 40)) console.log("  " + b);
process.exit(bad.length ? 1 : 0);
