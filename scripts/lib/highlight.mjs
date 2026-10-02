export const esc = (s) =>
  String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

const JS_KW = new Set([
  "const", "let", "var", "new", "try", "catch", "finally", "return", "typeof", "instanceof", "true", "false", "null",
  "undefined", "import", "from", "export", "default", "function", "for", "of", "in", "if", "else", "throw", "await", "async", "type", "interface",
]);
const PY_KW = new Set(["import", "as", "from", "None", "True", "False", "for", "in", "if", "else", "def", "return", "lambda", "with", "print"]);

/**
 * Tokenizer-based highlighter for JS/TS (default) or Python snippets.
 * `// =>` comments (JS) and `# ->` comments (Python) render as results.
 */
export function highlight(code, lang = "js") {
  const py = lang === "python";
  const re = py
    ? /(#[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|\b(\d+(?:\.\d+)?(?:e-?\d+)?j?)\b|([A-Za-z_][\w]*)/g
    : /(\/\/[^\n]*)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|`(?:[^`\\]|\\.)*`)|\b(\d+(?:\.\d+)?(?:e-?\d+)?n?)\b|([A-Za-z_$][\w$]*)/g;
  const kw = py ? PY_KW : JS_KW;
  let out = "";
  let last = 0;
  for (let m; (m = re.exec(code)); ) {
    out += esc(code.slice(last, m.index));
    last = re.lastIndex;
    const [tok, com, str, num, word] = m;
    if (com) {
      const res = py ? com.startsWith("# ->") : com.startsWith("// =>");
      out += `<span class="${res ? "t-res" : "t-com"}">${esc(com)}</span>`;
    } else if (str) out += `<span class="t-str">${esc(str)}</span>`;
    else if (num) out += `<span class="t-num">${num}</span>`;
    else if (word && kw.has(word)) out += `<span class="t-kw">${word}</span>`;
    else if (word && code[re.lastIndex] === "(") out += `<span class="t-fn">${esc(word)}</span>`;
    else out += esc(tok);
  }
  return out + esc(code.slice(last));
}
