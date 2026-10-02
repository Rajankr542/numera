(() => {
  const root = document.documentElement;
  const toRoot = root.dataset.root || "./";
  const toVer = root.dataset.verRoot || "./";
  const current = root.dataset.version;
  const path = root.dataset.path || "index.html";
  const meta = window.NUMERA_VERSIONS || { versions: [current], pages: {} };

  // Theme
  const themeBtn = document.querySelector(".theme");
  const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  themeBtn?.addEventListener("click", () => {
    root.dataset.theme = isDark() ? "light" : "dark";
    try { localStorage.setItem("numera-theme", root.dataset.theme); } catch (e) {}
  });

  // Version switcher: same page in the other version when it exists.
  const target = (v) => {
    const pages = meta.pages?.[v] || [];
    const keep = pages.includes(path) ? path : "index.html";
    return `${toRoot}v${v}/${keep}${pages.includes(path) ? location.hash : ""}`;
  };
  const sel = document.getElementById("ver");
  sel?.addEventListener("change", () => { location.href = target(sel.value); });
  document.querySelectorAll("[data-switch]").forEach((a) => a.setAttribute("href", target(a.dataset.switch)));
  // Pick up versions published after this page was built (served over http only).
  if (sel && location.protocol.startsWith("http")) {
    fetch(`${toRoot}versions.json`, { cache: "no-cache" }).then((r) => r.ok && r.json()).then((j) => {
      if (!j || !Array.isArray(j.versions)) return;
      const have = new Set([...sel.options].map((o) => o.value));
      for (const v of j.versions) if (!have.has(v)) {
        const o = document.createElement("option");
        o.value = v; o.textContent = `v${v}${v === j.latest ? " (latest)" : ""}`;
        sel.insertBefore(o, sel.firstChild);
        meta.pages[v] = meta.pages[v] || [];
      }
      for (const o of sel.options) o.textContent = `v${o.value}${o.value === j.latest ? " (latest)" : ""}`;
    }).catch(() => {});
  }

  // Mobile navigation (tabs are cloned into the drawer).
  const menu = document.querySelector(".menu");
  const side = document.querySelector(".side");
  const tabs = document.querySelector(".tabs");
  if (side && tabs) {
    const m = document.createElement("div");
    m.className = "mobile-tabs side-group";
    m.innerHTML = `<p class="side-title">Sections</p><ul>${[...tabs.querySelectorAll("a")].map((a) => `<li><a href="${a.getAttribute("href")}"${a.hasAttribute("aria-current") ? ' aria-current="page"' : ""}>${a.textContent}</a></li>`).join("")}</ul>`;
    side.prepend(m);
  }
  menu?.addEventListener("click", () => {
    const open = document.body.classList.toggle("nav-open");
    menu.setAttribute("aria-expanded", String(open));
  });
  side?.addEventListener("click", (e) => { if (e.target.closest("a")) document.body.classList.remove("nav-open"); });
  side?.querySelector('[aria-current="page"]')?.scrollIntoView({ block: "center" });

  // Copy buttons
  document.addEventListener("click", async (e) => {
    const b = e.target.closest(".copy");
    if (!b) return;
    const code = b.closest(".codeblock").querySelector("pre").innerText;
    try { await navigator.clipboard.writeText(code); } catch (err) {
      const t = document.createElement("textarea"); t.value = code; document.body.append(t); t.select(); document.execCommand("copy"); t.remove();
    }
    b.textContent = "Copied"; b.classList.add("done");
    setTimeout(() => { b.textContent = "Copy"; b.classList.remove("done"); }, 1400);
  });

  // Tabs (install commands)
  document.querySelectorAll("[data-tabs]").forEach((box) => {
    const btns = [...box.querySelectorAll('[role="tab"]')];
    const panels = [...box.querySelectorAll(".tabpanel")];
    btns.forEach((b, i) => b.addEventListener("click", () => {
      btns.forEach((x, j) => x.setAttribute("aria-selected", String(i === j)));
      panels.forEach((p, j) => (p.hidden = i !== j));
    }));
  });

  // On-this-page scrollspy
  const otpLinks = [...document.querySelectorAll(".otp a")];
  if (otpLinks.length && "IntersectionObserver" in window) {
    const map = new Map(otpLinks.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
    const io = new IntersectionObserver((es) => {
      for (const en of es) if (en.isIntersecting) {
        otpLinks.forEach((a) => a.classList.remove("on"));
        const a = map.get(en.target.id);
        if (a) { a.classList.add("on"); a.scrollIntoView({ block: "nearest" }); }
      }
    }, { rootMargin: "-70px 0px -70% 0px" });
    map.forEach((_, id) => { const el = document.getElementById(id); if (el) io.observe(el); });
  }

  // NumPy index filter
  const f = document.getElementById("np-filter");
  const onlyMissing = document.getElementById("np-missing");
  if (f) {
    const rows = [...document.querySelectorAll(".np-table tbody tr")];
    const apply = () => {
      const q = f.value.trim().toLowerCase().replace(/_/g, "");
      for (const r of rows) {
        const hit = (!q || r.textContent.toLowerCase().replace(/_/g, "").includes(q)) && (!onlyMissing.checked || r.classList.contains("missing"));
        r.hidden = !hit;
      }
      document.querySelectorAll(".np-surface").forEach((s) => (s.hidden = !s.querySelector("tbody tr:not([hidden])")));
    };
    f.addEventListener("input", apply);
    onlyMissing.addEventListener("change", apply);
  }

  // Search palette
  const pal = document.querySelector(".palette");
  const input = pal.querySelector("input");
  const list = pal.querySelector(".palette-res");
  let results = [];
  let sel_i = 0;
  const norm = (s) => s.toLowerCase().replace(/[_\s]/g, "");
  const escH = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const score = (item, q, nq) => {
    const t = item.t.toLowerCase();
    const nt = norm(item.t);
    const last = nt.split(".").pop();
    if (last === nq || nt === nq) return 100;
    if (last.startsWith(nq)) return 80 - Math.min(last.length - nq.length, 20);
    if (nt.includes(nq)) return 60 - Math.min(nt.indexOf(nq), 20);
    if (item.n && norm(item.n).includes(nq)) return 45;
    if (t.includes(q)) return 40;
    if ((item.s || "").toLowerCase().includes(q)) return 20;
    if ((item.k || "").toLowerCase().includes(q)) return 10;
    return 0;
  };
  const render = () => {
    const q = input.value.trim().toLowerCase();
    const idx = window.NUMERA_INDEX || [];
    if (!q) results = idx.filter((x) => x.k === "Guide").slice(0, 12);
    else {
      const nq = norm(q).replace(/^(np|numpy)\./, "");
      results = idx.map((x) => [x, score(x, q, nq)]).filter((x) => x[1] > 0).sort((a, b) => b[1] - a[1] || a[0].t.length - b[0].t.length).slice(0, 40).map((x) => x[0]);
    }
    sel_i = 0;
    list.innerHTML = results.length
      ? results.map((r, i) => `<li role="option" aria-selected="${i === 0}"><a href="${toVer}${r.u}"><span class="r-t">${escH(r.t)}</span><span class="r-k">${escH(r.k)}</span>${r.s ? `<span class="r-s">${escH(r.s)}</span>` : ""}</a></li>`).join("")
      : `<li class="empty">No results for “${escH(input.value)}”.</li>`;
  };
  const move = (d) => {
    const items = [...list.querySelectorAll("li[role=option]")];
    if (!items.length) return;
    items[sel_i]?.setAttribute("aria-selected", "false");
    sel_i = (sel_i + d + items.length) % items.length;
    items[sel_i].setAttribute("aria-selected", "true");
    items[sel_i].scrollIntoView({ block: "nearest" });
  };
  let lastFocus = null;
  const open = () => { lastFocus = document.activeElement; pal.hidden = false; input.value = ""; render(); input.focus(); };
  const close = () => { pal.hidden = true; lastFocus?.focus?.(); };
  document.querySelectorAll("[data-open-search]").forEach((b) => b.addEventListener("click", open));
  pal.addEventListener("click", (e) => { if (e.target === pal) close(); else if (e.target.closest("a")) close(); });
  input.addEventListener("input", render);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") { const a = list.querySelectorAll("li[role=option] a")[sel_i]; if (a) { location.href = a.href; close(); } }
    else if (e.key === "Escape") close();
  });
  document.addEventListener("keydown", (e) => {
    const typing = /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
    if ((e.key === "/" && !typing) || (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey))) { e.preventDefault(); pal.hidden ? open() : close(); }
    else if (e.key === "Escape" && !pal.hidden) close();
  });
})();
